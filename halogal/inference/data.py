"""Observational data containers for inference.

A :class:`Dataset` bundles one observable measured at one redshift together
with its covariance, and knows how to obtain the corresponding model
prediction from a :class:`~halogal.model.Observables` instance. Three kinds are
supported:

``uvlf``
    UV luminosity function Phi(M_UV) at a grid of magnitudes.
``clustering``
    Angular (or projected/real) correlation function w(theta) of a
    magnitude-limited sample.
``number_density``
    Cumulative number density n(<M_UV thresh) (a scalar constraint).

The likelihood is Gaussian, so each dataset stores a covariance matrix. If only
a 1-D ``sigma`` is given it is promoted to a diagonal covariance -- the
covariance path is the same either way, so a full covariance (e.g. a jackknife
estimate for clustering) can be supplied later without code changes. The
Cholesky factor and Gaussian normalisation constant are cached at construction.

Asymmetric (upper/lower) errors are supported through a *split-normal*
likelihood: pass ``sigma_low`` and ``sigma_upp`` instead of ``sigma``/``cov``.
Each point then uses ``sigma_low`` when the model lies below the datum and
``sigma_upp`` when it lies above (see :meth:`Dataset.chi2`).
"""

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .parametrization import MODEL_PARAM_NAMES

__all__ = ["Dataset"]

_KINDS = ("uvlf", "clustering", "number_density")


def _check_sigma(name, sigma, n):
    """Validate a 1-D error array of length ``n``; return it as floats."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    if sigma.shape != (n,):
        raise ValueError(f"{name} has length {sigma.shape}, expected ({n},)")
    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError(f"{name} must be finite and strictly positive")
    return sigma


def _build_covariance(n, cov, sigma, sigma_low=None, sigma_upp=None):
    """Return ``(cov, sigma_low, sigma_upp)`` from exactly one error spec.

    ``cov`` is an (n, n) covariance, ``sigma`` 1-D symmetric errors, and
    ``sigma_low``/``sigma_upp`` (given together) 1-D asymmetric errors. For the
    asymmetric case the returned covariance is ``diag(((lo + hi) / 2)**2)``,
    whose Gaussian normalisation equals that of the split-normal density; the
    returned ``sigma_low``/``sigma_upp`` are None otherwise.
    """
    asym = sigma_low is not None or sigma_upp is not None
    if asym and (sigma_low is None or sigma_upp is None):
        raise ValueError("'sigma_low' and 'sigma_upp' must be given together")
    n_given = sum([cov is not None, sigma is not None, asym])
    if n_given > 1:
        raise ValueError(
            "Provide exactly one of 'cov', 'sigma', or "
            "'sigma_low'/'sigma_upp'"
        )
    if n_given == 0:
        raise ValueError(
            "One of 'cov', 'sigma', or 'sigma_low'/'sigma_upp' is required"
        )
    if asym:
        lo = _check_sigma("sigma_low", sigma_low, n)
        hi = _check_sigma("sigma_upp", sigma_upp, n)
        return np.diag((0.5 * (lo + hi)) ** 2), lo, hi
    if cov is not None:
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if cov.shape != (n, n):
            raise ValueError(
                f"cov has shape {cov.shape}, expected ({n}, {n})"
            )
        if not np.all(np.isfinite(cov)):
            raise ValueError("cov contains non-finite values")
        if not np.allclose(cov, cov.T, rtol=1e-8, atol=0.0):
            raise ValueError("cov must be symmetric")
        return cov, None, None
    sigma = _check_sigma("sigma", sigma, n)
    return np.diag(sigma ** 2), None, None


class Dataset:
    """A single observable at a single redshift with its covariance.

    Use the classmethod constructors :meth:`uvlf`, :meth:`clustering`, and
    :meth:`number_density` rather than instantiating directly.
    """

    def __init__(self, kind, z, data, cov=None, sigma=None, label=None,
                 sigma_low=None, sigma_upp=None):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self.kind = kind
        self.z = float(z)
        self.data = np.atleast_1d(np.asarray(data, dtype=float))
        if self.data.ndim != 1:
            raise ValueError("data must be one-dimensional")
        if not np.all(np.isfinite(self.data)):
            raise ValueError(
                "data contains non-finite values; mask them out (together "
                "with their errors) before building the Dataset"
            )
        self.label = label or f"{kind}_z{z:g}"

        self.cov, self.sigma_low, self.sigma_upp = _build_covariance(
            self.data.size, cov, sigma, sigma_low, sigma_upp
        )
        # Cache Cholesky factor and the Gaussian normalisation constant.
        try:
            self._cho = cho_factor(self.cov)
        except LinAlgError as exc:
            raise ValueError(
                f"covariance of dataset '{self.label}' is not positive "
                f"definite ({exc})"
            ) from None
        logdet = 2.0 * np.sum(np.log(np.diag(self._cho[0])))
        self._lognorm = -0.5 * (self.data.size * np.log(2.0 * np.pi) + logdet)

        # kind-specific attributes (set by constructors)
        self.x = None
        self.MUV_thresh1 = None
        self.MUV_thresh2 = 0.0
        self.corr_kwargs = {}
        self.dMUV = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def uvlf(cls, z, MUV, phi, cov=None, sigma=None, label=None,
             sigma_low=None, sigma_upp=None, dMUV=None):
        """UV luminosity function data.

        Parameters
        ----------
        z : float
            Redshift.
        MUV : array_like
            Absolute UV magnitudes (data abscissa).
        phi : array_like
            Measured Phi(M_UV) in Mpc^-3 mag^-1.
        cov, sigma : array_like, optional
            Covariance (n, n) or 1-D Gaussian errors.
        label : str, optional
            Human-readable label.
        sigma_low, sigma_upp : array_like, optional
            1-D lower and upper errors on ``phi`` for a split-normal
            likelihood (given together). Exactly one of ``cov``, ``sigma``, or
            the ``sigma_low``/``sigma_upp`` pair is required.
        dMUV : float or array_like, optional
            Width [mag] of the magnitude bin centred on each ``MUV``: a scalar
            for uniform bins or one value per point. The model is then the
            bin-averaged Phi over ``[MUV - dMUV/2, MUV + dMUV/2]``, which is
            what a binned (e.g. 1/Vmax) UVLF measures. Pass the data's true
            bin width; do not infer it from adjacent points, since empty bins
            dropped from the data make that difference too large. Default
            None uses ``Observables.luminosity_function``'s narrow default
            (0.05 mag), i.e. effectively Phi at the bin centre.
        """
        obj = cls("uvlf", z, phi, cov=cov, sigma=sigma, label=label,
                  sigma_low=sigma_low, sigma_upp=sigma_upp)
        obj.x = np.atleast_1d(np.asarray(MUV, dtype=float))
        if obj.x.shape != obj.data.shape:
            raise ValueError("MUV and phi must have the same length")
        if dMUV is not None:
            dMUV = np.asarray(dMUV, dtype=float)
            if dMUV.ndim == 1 and dMUV.shape != obj.x.shape:
                raise ValueError(
                    f"dMUV has length {dMUV.shape}, expected a scalar or "
                    f"{obj.x.shape}"
                )
            if dMUV.ndim > 1 or not (np.all(np.isfinite(dMUV))
                                     and np.all(dMUV > 0)):
                raise ValueError(
                    "dMUV must be a finite, strictly positive scalar or 1-D "
                    "array"
                )
            dMUV = float(dMUV) if dMUV.ndim == 0 else dMUV
        obj.dMUV = dMUV
        return obj

    @classmethod
    def number_density(cls, z, MUV_thresh, n, cov=None, sigma=None, label=None):
        """Cumulative number density n(<M_UV thresh).

        Parameters
        ----------
        z : float
            Redshift.
        MUV_thresh : float
            UV magnitude threshold (brighter than).
        n : float
            Measured number density in Mpc^-3.
        cov, sigma : optional
            1x1 covariance or scalar/1-element error.
        label : str, optional
            Label.
        """
        obj = cls("number_density", z, np.atleast_1d(n),
                  cov=cov, sigma=sigma, label=label)
        obj.MUV_thresh1 = float(MUV_thresh)
        return obj

    @classmethod
    def clustering(cls, z, theta, w, cov=None, sigma=None, MUV_thresh1=None,
                   MUV_thresh2=0.0, corr_kwargs=None, label=None):
        """Angular correlation function w(theta) of a magnitude-limited sample.

        Parameters
        ----------
        z : float
            Redshift.
        theta : array_like
            Angular separations in arcsec (data abscissa).
        w : array_like
            Measured w(theta).
        cov, sigma : array_like, optional
            Covariance (n, n) or 1-D errors. Exactly one required. For
            clustering a full covariance is strongly preferred because theta
            bins are correlated; a diagonal sigma is accepted but will
            underestimate parameter uncertainties.
        MUV_thresh1 : float
            Faint magnitude limit of the clustered sample (required). On its
            own it defines a threshold sample of everything brighter.
        MUV_thresh2 : float, optional
            Bright magnitude limit, for a magnitude-*binned* sample; must be
            more negative (brighter) than ``MUV_thresh1``. The default 0.0
            means a pure threshold sample.
        corr_kwargs : dict, optional
            Extra keyword arguments forwarded to
            :meth:`~halogal.model.Observables.initialize_correlation_model`
            (e.g. ``p1`` redshift distribution, ``correlation_type``,
            ``theta_min``/``theta_max``, ``znum``).
        label : str, optional
            Label.
        """
        if MUV_thresh1 is None:
            raise ValueError("clustering dataset requires MUV_thresh1")
        obj = cls("clustering", z, w, cov=cov, sigma=sigma, label=label)
        obj.x = np.atleast_1d(np.asarray(theta, dtype=float))
        if obj.x.shape != obj.data.shape:
            raise ValueError("theta and w must have the same length")
        if not (np.all(np.isfinite(obj.x)) and np.all(obj.x > 0)):
            raise ValueError("theta must be finite and strictly positive")
        obj.MUV_thresh1 = float(MUV_thresh1)
        obj.MUV_thresh2 = float(MUV_thresh2)
        if obj.MUV_thresh2 < 0 and obj.MUV_thresh2 >= obj.MUV_thresh1:
            raise ValueError(
                f"MUV_thresh2 ({obj.MUV_thresh2}) must be brighter (more "
                f"negative) than MUV_thresh1 ({obj.MUV_thresh1})"
            )
        obj.corr_kwargs = dict(corr_kwargs or {})
        return obj

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def check_separation_range(self, separation_model):
        """Require the data abscissa to lie inside the model separation grid.

        The clustering model is evaluated on its own grid (set by
        ``theta_min``/``theta_max`` etc. in ``corr_kwargs``) and interpolated
        onto the data. Outside that grid ``np.interp`` would silently return
        the edge value, so out-of-range data are rejected up front.

        Parameters
        ----------
        separation_model : array_like
            Model separations, in the same units as the data abscissa.

        Raises
        ------
        ValueError
            If any data separation falls outside the model grid.
        """
        sep = np.asarray(separation_model, dtype=float)
        lo, hi = sep.min(), sep.max()
        tol = 1e-9 * hi
        if self.x.min() < lo - tol or self.x.max() > hi + tol:
            raise ValueError(
                f"dataset '{self.label}': data separations span "
                f"[{self.x.min():g}, {self.x.max():g}] but the model grid "
                f"covers only [{lo:g}, {hi:g}]. Widen the model grid via "
                f"corr_kwargs (e.g. theta_min/theta_max, rp_min/rp_max) or "
                f"drop the out-of-range points."
            )

    def predict(self, observables, model_params):
        """Model prediction aligned with this dataset's abscissa.

        Parameters
        ----------
        observables : halogal.model.Observables
            Observables calculator for this dataset's redshift bin. For a
            clustering dataset its correlation model must already be initialised
            (the likelihood handles this).
        model_params : dict
            Model HOD parameters (linear masses), e.g. from
            :meth:`ModelParametrization.params_for_redshift`.

        Returns
        -------
        ndarray
            Model vector, same length as ``self.data``.
        """
        if self.kind == "uvlf":
            if self.dMUV is None:
                return np.atleast_1d(
                    observables.luminosity_function(self.x, **model_params)
                )
            if np.ndim(self.dMUV) == 0:
                return np.atleast_1d(observables.luminosity_function(
                    self.x, self.dMUV, **model_params))
            # per-point bin widths: one bin-averaged evaluation per point
            return np.array([
                observables.luminosity_function(x, dm, **model_params)[0]
                for x, dm in zip(self.x, self.dMUV)
            ])
        if self.kind == "number_density":
            n = observables.number_density(self.MUV_thresh1, **model_params)
            return np.atleast_1d(n)
        if self.kind == "clustering":
            corr_params = {
                k: v for k, v in model_params.items()
                if k in MODEL_PARAM_NAMES
            }
            result = observables.update_correlation_model(**corr_params)
            theta_model = np.asarray(result["separation"], dtype=float)
            w_model = np.asarray(result["correlation"], dtype=float)
            # Interpolate in log-separation (theta spans decades); keep w
            # linear so any sign is handled.
            return np.interp(
                np.log10(self.x), np.log10(theta_model), w_model
            )
        raise RuntimeError(f"Unknown kind {self.kind!r}")  # pragma: no cover

    @property
    def is_asymmetric(self):
        """Whether this dataset uses split-normal (asymmetric) errors."""
        return self.sigma_low is not None

    def effective_sigma(self, model):
        """Per-point error actually used against ``model``.

        For split-normal data this is ``sigma_low`` where the model lies below
        the datum (``data - model > 0``) and ``sigma_upp`` otherwise; for
        symmetric data it is ``sqrt(diag(cov))``.
        """
        if not self.is_asymmetric:
            return np.sqrt(np.diag(self.cov))
        r = self.data - np.asarray(model, dtype=float)
        return np.where(r > 0, self.sigma_low, self.sigma_upp)

    def chi2(self, model):
        """Generalised chi-square for a model vector.

        ``r^T C^-1 r`` with ``r = data - model``. For split-normal data the
        points are independent and ``chi2 = sum((r / s)**2)`` with ``s`` from
        :meth:`effective_sigma` -- the same side convention as galfind's
        ``MCMC_Fitter`` (lower error when the model is below the datum).
        """
        r = self.data - np.asarray(model, dtype=float)
        if self.is_asymmetric:
            return float(np.sum((r / self.effective_sigma(model)) ** 2))
        return float(r @ cho_solve(self._cho, r))

    def log_likelihood(self, model):
        """Gaussian log-likelihood (with normalisation) for a model vector.

        For split-normal data the normalisation is the proper split-normal
        constant ``sum(log(sqrt(2/pi) / (sigma_low + sigma_upp)))``, which is
        independent of the model, so the likelihood is continuous where the
        model crosses a datum. (galfind instead adds a per-point
        ``log(s**2)`` that jumps at that crossing.)
        """
        return self._lognorm - 0.5 * self.chi2(model)

    def __repr__(self):
        return f"Dataset(kind='{self.kind}', z={self.z:g}, n={self.data.size})"
