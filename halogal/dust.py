"""General UV dust attenuation built on an observed beta-M_UV distribution.

The attenuation of a galaxy with UV slope beta is

    A_UV(beta) = max(0, dA_dbeta * (beta - beta_intr)),

so galaxies bluer than the intrinsic slope ``beta_intr`` are dust free rather
than given a negative attenuation. At fixed *observed* M_UV, beta follows a
Gaussian distribution with mean ``mu_beta(z, M_UV)`` and width
``sigma_beta(z, M_UV)`` (a :class:`BetaMUVRelation`). The attenuation applied
to the population is an average of A_UV(beta) over that distribution
(:class:`BetaDust`, see ``scatter_average``).

The default is the Meurer et al. (1999) relation A_1600 = 4.43 + 1.99 beta
(dA_dbeta = 1.99, beta_intr = -4.43/1.99) with the Bouwens et al. (2014)
beta-M_UV relation and sigma_beta = 0.34.

Every redshift-dependent quantity (the beta-M_UV parameters, dA_dbeta and
beta_intr) may be given as a number, a ``(z_nodes, values)`` table, one of the
``*Z`` classes below, or any callable ``f(z)``; see :func:`as_z_function`.

All objects here are immutable after construction and compare (and hash) by
value, so models that hold them cache correctly and pickle to multiprocessing
workers. A user-supplied callable is compared by identity and must itself be
picklable (a module-level function, not a lambda) for parallel sampling.
"""

import numbers

import numpy as np
from scipy.special import log_ndtr, ndtr

from .config import (
    BETA_C, BETA_MUV0, BETA_MUV_AT_M0, BETA_Z_DATA, DBETA_DMUV,
    DUST_BETA_INTR, DUST_C1, DUST_SCATTER_AVERAGE, DUST_SIGMA_BETA,
    EPOCHS_PAPER1_BETA_MUV,
)

__all__ = [
    "ZFunction", "ConstantZ", "LinearZ", "PowerLawZ", "TabulatedZ",
    "as_z_function",
    "BetaMUVRelation", "LinearBetaMUV", "Bouwens14BetaMUV", "FunctionBetaMUV",
    "DustModel", "NoDust", "BetaDust", "DEFAULT_DUST", "resolve_dust",
]

_K = 0.4 * np.log(10.0)          # 10**(0.4 A) = exp(_K A)
_SCATTER_AVERAGES = ("luminosity", "mean", "none")


# --------------------------------------------------------------------------
# Immutable, value-comparable base
# --------------------------------------------------------------------------
class _Frozen:
    """Immutable after ``__init__``; equality, hash and repr from _state()."""

    _frozen = False

    def _freeze(self):
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"{type(self).__name__} is immutable; use with_params() to "
                f"make a modified copy")
        object.__setattr__(self, name, value)

    def _state(self):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._state() == other._state()

    def __hash__(self):
        return hash((type(self).__name__, self._state()))

    def __repr__(self):
        return f"{type(self).__name__}{self._state()}"

    def __reduce__(self):
        # Rebuild through __init__ so the frozen flag never blocks unpickling.
        return (type(self), self._init_args())

    def _init_args(self):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Redshift evolution of a scalar
# --------------------------------------------------------------------------
class ZFunction(_Frozen):
    """A scalar quantity as a function of redshift."""

    def __call__(self, z):
        raise NotImplementedError


class ConstantZ(ZFunction):
    """Redshift-independent value."""

    def __init__(self, value):
        self.value = float(value)
        self._freeze()

    def __call__(self, z):
        return np.full_like(np.asarray(z, dtype=float), self.value)

    def _state(self):
        return (self.value,)

    _init_args = _state


def _clip(z, z_clip):
    z = np.asarray(z, dtype=float)
    return z if z_clip is None else np.clip(z, *z_clip)


class LinearZ(ZFunction):
    """``A0 + A1 * (z - z_pivot)``.

    ``z_clip=(z_min, z_max)`` holds the value constant outside that range
    (e.g. the redshift range a fit was constrained over) instead of
    extrapolating.
    """

    def __init__(self, A0, A1, z_pivot=0.0, z_clip=None):
        self.A0, self.A1, self.z_pivot = float(A0), float(A1), float(z_pivot)
        self.z_clip = None if z_clip is None else tuple(map(float, z_clip))
        self._freeze()

    def __call__(self, z):
        return self.A0 + self.A1 * (_clip(z, self.z_clip) - self.z_pivot)

    def _state(self):
        return (self.A0, self.A1, self.z_pivot, self.z_clip)

    _init_args = _state


class PowerLawZ(ZFunction):
    """``A0 * ((1 + z) / (1 + z_pivot)) ** A1``; ``z_clip`` as in LinearZ."""

    def __init__(self, A0, A1, z_pivot=0.0, z_clip=None):
        self.A0, self.A1, self.z_pivot = float(A0), float(A1), float(z_pivot)
        self.z_clip = None if z_clip is None else tuple(map(float, z_clip))
        self._freeze()

    def __call__(self, z):
        z = _clip(z, self.z_clip)
        return self.A0 * ((1.0 + z) / (1.0 + self.z_pivot)) ** self.A1

    def _state(self):
        return (self.A0, self.A1, self.z_pivot, self.z_clip)

    _init_args = _state


class TabulatedZ(ZFunction):
    """Linear interpolation in a table, held constant beyond its ends."""

    def __init__(self, z_nodes, values):
        z_nodes = np.asarray(z_nodes, dtype=float)
        values = np.asarray(values, dtype=float)
        if z_nodes.ndim != 1 or z_nodes.shape != values.shape:
            raise ValueError("z_nodes and values must be 1D and equal length")
        order = np.argsort(z_nodes)
        self.z_nodes = tuple(z_nodes[order])
        self.values = tuple(values[order])
        self._freeze()

    def __call__(self, z):
        return np.interp(np.asarray(z, dtype=float), self.z_nodes, self.values)

    def _state(self):
        return (self.z_nodes, self.values)

    _init_args = _state


def as_z_function(spec):
    """Turn ``spec`` into a callable ``f(z)``.

    Accepts a number (constant), a ``(z_nodes, values)`` pair (interpolated
    table), a :class:`ZFunction`, or any other callable ``f(z)``.
    """
    if isinstance(spec, ZFunction):
        return spec
    if isinstance(spec, numbers.Real):
        return ConstantZ(spec)
    if callable(spec):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return TabulatedZ(*spec)
    raise TypeError(
        f"cannot interpret {spec!r} as a function of redshift: give a number, "
        f"a (z_nodes, values) pair, a ZFunction or a callable f(z)")


# --------------------------------------------------------------------------
# beta - M_UV distributions
# --------------------------------------------------------------------------
class BetaMUVRelation(_Frozen):
    """Gaussian distribution of beta at fixed redshift and observed M_UV."""

    def mean(self, z, MUV):
        """Mean UV slope at redshift ``z`` and observed magnitude ``MUV``."""
        raise NotImplementedError

    def sigma(self, z, MUV):
        """Gaussian scatter in beta (>= 0) at ``z`` and ``MUV``."""
        raise NotImplementedError


class LinearBetaMUV(BetaMUVRelation):
    """``beta = beta_M0(z) + dbeta_dMUV(z) * (M_UV - M0)``, scatter sigma_beta(z).

    Each of ``beta_M0``, ``dbeta_dMUV`` and ``sigma_beta`` may be anything
    :func:`as_z_function` accepts.

    Examples
    --------
    >>> # EPOCHS Paper-1 fits, A0 + A1 (z - 7)
    >>> rel = LinearBetaMUV.epochs_paper1()
    >>> # Per-redshift-bin values, interpolated
    >>> rel = LinearBetaMUV.from_table([7, 8, 9], [-2.3, -2.4, -2.3],
    ...                                [-0.08, 0.03, -0.03], [0.36, 0.35, 0.38])
    """

    def __init__(self, beta_M0, dbeta_dMUV, sigma_beta=0.0, M0=-19.0):
        self.beta_M0 = as_z_function(beta_M0)
        self.dbeta_dMUV = as_z_function(dbeta_dMUV)
        self.sigma_beta = as_z_function(sigma_beta)
        self.M0 = float(M0)
        self._freeze()

    @classmethod
    def from_linear_z(cls, beta_M0, dbeta_dMUV, sigma_beta=(0.0, 0.0),
                      z_pivot=7.0, M0=-19.0, z_clip=None):
        """From ``(A0, A1)`` pairs of ``A0 + A1 * (z - z_pivot)`` fits."""
        def lin(pair):
            return LinearZ(*pair, z_pivot=z_pivot, z_clip=z_clip)
        return cls(lin(beta_M0), lin(dbeta_dMUV), lin(sigma_beta), M0=M0)

    @classmethod
    def from_table(cls, z, beta_M0, dbeta_dMUV, sigma_beta=None, M0=-19.0):
        """From per-redshift values, linearly interpolated in z."""
        sig = 0.0 if sigma_beta is None else TabulatedZ(z, sigma_beta)
        return cls(TabulatedZ(z, beta_M0), TabulatedZ(z, dbeta_dMUV), sig,
                   M0=M0)

    @classmethod
    def epochs_paper1(cls):
        """EPOCHS Paper-1 beta(M_UV=-19), dbeta/dM_UV and sigma_beta fits
        (``config.EPOCHS_PAPER1_BETA_MUV``), held constant outside the fitted
        redshift range."""
        p = EPOCHS_PAPER1_BETA_MUV
        return cls.from_linear_z(p["beta_M0"], p["dbeta_dMUV"],
                                 p["sigma_beta"], z_pivot=p["z_pivot"],
                                 M0=p["M0"], z_clip=p["z_range"])

    def mean(self, z, MUV):
        z, MUV = np.broadcast_arrays(np.asarray(z, float), np.asarray(MUV, float))
        return self.beta_M0(z) + self.dbeta_dMUV(z) * (MUV - self.M0)

    def sigma(self, z, MUV):
        z, _ = np.broadcast_arrays(np.asarray(z, float), np.asarray(MUV, float))
        return np.maximum(self.sigma_beta(z), 0.0)

    def _state(self):
        return (self.beta_M0, self.dbeta_dMUV, self.sigma_beta, self.M0)

    _init_args = _state


class Bouwens14BetaMUV(BetaMUVRelation):
    """Bouwens et al. (2013, 2014) beta-M_UV relation (the package default).

    Linear in M_UV brighter than M_UV = -19.5 and exponentially approaching
    beta = -2.33 fainter than it, with the normalisation and slope
    interpolated in redshift (held constant outside z = 2.5-8).
    """

    def __init__(self, sigma_beta=DUST_SIGMA_BETA):
        self.sigma_beta = as_z_function(sigma_beta)
        self._freeze()

    def mean(self, z, MUV):
        z, MUV = np.broadcast_arrays(np.asarray(z, float), np.asarray(MUV, float))
        betaM0 = np.interp(z, BETA_Z_DATA, BETA_MUV_AT_M0)
        dbeta = (MUV - BETA_MUV0) * np.interp(z, BETA_Z_DATA, DBETA_DMUV)
        faint = (betaM0 - BETA_C) * np.exp(dbeta / (betaM0 - BETA_C)) + BETA_C
        bright = dbeta + betaM0
        return (faint * np.heaviside(MUV - BETA_MUV0, 0.5)
                + bright * np.heaviside(BETA_MUV0 - MUV, 0.5))

    def sigma(self, z, MUV):
        z, _ = np.broadcast_arrays(np.asarray(z, float), np.asarray(MUV, float))
        return np.maximum(self.sigma_beta(z), 0.0)

    def _state(self):
        return (self.sigma_beta,)

    _init_args = _state


class FunctionBetaMUV(BetaMUVRelation):
    """Fully general: ``mean_fn(z, MUV)`` and ``sigma_fn(z, MUV)`` callables.

    ``sigma_fn`` may also be a number. Callables are compared by identity and
    must be picklable (module-level functions) for parallel sampling.
    """

    def __init__(self, mean_fn, sigma_fn=0.0):
        if not callable(mean_fn):
            raise TypeError("mean_fn must be callable as mean_fn(z, MUV)")
        if not (callable(sigma_fn) or isinstance(sigma_fn, numbers.Real)):
            raise TypeError("sigma_fn must be a number or callable(z, MUV)")
        self.mean_fn = mean_fn
        self.sigma_fn = sigma_fn if callable(sigma_fn) else float(sigma_fn)
        self._freeze()

    def mean(self, z, MUV):
        z, MUV = np.broadcast_arrays(np.asarray(z, float), np.asarray(MUV, float))
        return np.asarray(self.mean_fn(z, MUV), dtype=float)

    def sigma(self, z, MUV):
        z, MUV = np.broadcast_arrays(np.asarray(z, float), np.asarray(MUV, float))
        if callable(self.sigma_fn):
            s = np.asarray(self.sigma_fn(z, MUV), dtype=float)
        else:
            s = np.full_like(MUV, self.sigma_fn)
        return np.maximum(np.broadcast_to(s, MUV.shape), 0.0)

    def _state(self):
        return (self.mean_fn, self.sigma_fn)

    _init_args = _state


# --------------------------------------------------------------------------
# Dust models
# --------------------------------------------------------------------------
class DustModel(_Frozen):
    """UV attenuation (mag) as a function of redshift and observed M_UV."""

    def A_UV(self, z, MUV_obs):
        raise NotImplementedError

    def __call__(self, z, MUV_obs):
        return self.A_UV(z, MUV_obs)

    def with_params(self, **kwargs):
        """Copy of this model with some constructor arguments replaced."""
        names = self._init_names()
        unknown = set(kwargs) - set(names)
        if unknown:
            raise TypeError(f"unknown parameter(s) {sorted(unknown)}; "
                            f"expected any of {list(names)}")
        args = dict(zip(names, self._init_args()))
        args.update(kwargs)
        return type(self)(**args)

    def _init_names(self):
        return ()


class NoDust(DustModel):
    """No attenuation."""

    def __init__(self):
        self._freeze()

    def A_UV(self, z, MUV_obs):
        z, MUV = np.broadcast_arrays(np.asarray(z, float),
                                     np.asarray(MUV_obs, float))
        return np.zeros_like(MUV)

    def _state(self):
        return ()

    _init_args = _state


class BetaDust(DustModel):
    """Attenuation from a beta-M_UV distribution and an A_UV-beta relation.

    Per galaxy, ``A_UV = max(0, dA_dbeta * (beta - beta_intr))``; beta is
    Gaussian at fixed z and observed M_UV, as given by ``beta_relation``.

    Parameters
    ----------
    beta_relation : BetaMUVRelation, optional
        Distribution of beta. Default :class:`Bouwens14BetaMUV` with
        sigma_beta = 0.34.
    dA_dbeta : float, table or callable, optional
        Slope of A_UV against beta. Default 1.99 (Meurer et al. 1999).
    beta_intr : float, table or callable, optional
        Intrinsic (dust-free) UV slope. Default -4.43/1.99 (Meurer+99).
    scatter_average : {"luminosity", "mean", "none"}, optional
        How the scatter in beta enters the population attenuation at fixed
        observed M_UV:

        ``"luminosity"`` (default)
            ``2.5 log10 <10**(0.4 A_UV)>``, the attenuation that converts the
            observed to the mean intrinsic luminosity. Reduces to the
            ``A(<beta>) + 0.2 ln10 (dA_dbeta sigma_beta)**2`` of Smit et al.
            (2012) when the whole distribution is redder than beta_intr.
        ``"mean"``
            ``<A_UV>``, the mean (clipped) attenuation.
        ``"none"``
            ``A_UV(<beta>)``; the scatter is ignored.

        Both averages are evaluated in closed form and are never negative.
    z_max : float, optional
        No dust above this redshift. Default None (dust at all redshifts).
    """

    def __init__(self, beta_relation=None, dA_dbeta=DUST_C1, beta_intr=None,
                 scatter_average=DUST_SCATTER_AVERAGE, z_max=None):
        if beta_relation is None:
            beta_relation = Bouwens14BetaMUV()
        if not isinstance(beta_relation, BetaMUVRelation):
            raise TypeError("beta_relation must be a BetaMUVRelation")
        if scatter_average not in _SCATTER_AVERAGES:
            raise ValueError(f"scatter_average must be one of "
                             f"{_SCATTER_AVERAGES}, not {scatter_average!r}")
        self.beta_relation = beta_relation
        self.dA_dbeta = as_z_function(dA_dbeta)
        self.beta_intr = as_z_function(
            DUST_BETA_INTR if beta_intr is None else beta_intr)
        self.scatter_average = scatter_average
        self.z_max = None if z_max is None else float(z_max)
        self._freeze()

    def A_UV(self, z, MUV_obs):
        z, MUV = np.broadcast_arrays(np.asarray(z, float),
                                     np.asarray(MUV_obs, float))
        slope = self.dA_dbeta(z)
        m = slope * (self.beta_relation.mean(z, MUV) - self.beta_intr(z))
        if self.scatter_average == "none":
            A = np.maximum(m, 0.0)
        else:
            s = np.abs(slope) * self.beta_relation.sigma(z, MUV)
            A = _average_clipped(m, s, self.scatter_average)
        if self.z_max is not None:
            A = np.where(z > self.z_max, 0.0, A)
        return A

    def _state(self):
        return (self.beta_relation, self.dA_dbeta, self.beta_intr,
                self.scatter_average, self.z_max)

    _init_args = _state

    def _init_names(self):
        return ("beta_relation", "dA_dbeta", "beta_intr", "scatter_average",
                "z_max")


def _average_clipped(m, s, mode):
    """Average of max(0, x) for x ~ N(m, s**2), per ``mode``; s = 0 -> max(0, m)."""
    m, s = np.broadcast_arrays(np.asarray(m, float), np.asarray(s, float))
    shape = m.shape
    m, s = m.ravel(), s.ravel()
    out = np.maximum(m, 0.0)
    pos = s > 0
    if not np.any(pos):
        return out.reshape(shape)
    mp, sp = m[pos], s[pos]
    t = mp / sp
    if mode == "mean":
        val = sp * np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi) + mp * ndtr(t)
    else:  # "luminosity": (1/K) log <exp(K max(0, x))>
        log_mean = np.logaddexp(
            log_ndtr(-t),
            _K * mp + 0.5 * (_K * sp) ** 2 + log_ndtr(t + _K * sp))
        val = log_mean / _K
    out[pos] = np.maximum(val, 0.0)
    return out.reshape(shape)


#: Package default: Bouwens+14 beta-M_UV (sigma_beta 0.34) with Meurer+99.
DEFAULT_DUST = BetaDust()


def resolve_dust(dust):
    """Normalise a ``dust`` argument to a :class:`DustModel`.

    A DustModel is returned unchanged; ``None`` or ``False`` mean no dust and
    ``True`` the package default (:data:`DEFAULT_DUST`).
    """
    if isinstance(dust, DustModel):
        return dust
    if dust is None or dust is False:
        return NoDust()
    if dust is True:
        return DEFAULT_DUST
    raise TypeError(f"dust must be a DustModel, None or a bool, not {dust!r}")
