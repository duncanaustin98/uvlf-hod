"""Mapping from the sampling vector to per-redshift model parameters.

This is the layer that unifies the two redshift regimes behind a single
likelihood. Given the named parameter dictionary produced by a
:class:`~halogal.inference.parameters.ParameterSet`, a
:class:`ModelParametrization` returns, for any redshift, the dictionary of
**model** HOD parameters that :class:`~halogal.model.Observables` consumes
(``eps0, Mc, a, b, sigma_UV, Mcut, Msat, asat`` with *linear* masses in
M_sun).

Three modes:

``"fixed_z"``
    The sampled quantities *are* the HOD parameters at a single redshift.
    Masses are sampled in ``log10`` (``logMc``, ``logMcut``, ``logMsat``).

``"evolution"``
    The sampled quantities are the linear redshift-evolution coefficients
    (``d_<x>_dz``, ``C_<x>``) and the per-bin HOD parameters are evaluated with
    :class:`~halogal.models.parametrization.RedshiftParametrization` -- the same
    engine underlying the package's ``*_fz`` helpers. Coefficient names match
    :data:`halogal.config.DEFAULT_REDSHIFT_EVOLUTION` so fits can be compared to
    the published values directly. The mass coefficients act in ``log10`` space.

``"powerlaw"``
    Power-law redshift evolution normalised at a pivot redshift ``z_pivot``,
    ``A(z) = A_0 * ((1 + z) / (1 + z_pivot))**A_1``. The coefficients are
    named ``<x>_0`` (the value at ``z_pivot``) and ``<x>_1`` (the exponent),
    with ``<x>`` the fixed-z sampling name: ``eps0_0``, ``eps0_1``, ...,
    ``logMc_0``, ``logMc_1``. For masses ``logM_0`` is ``log10 M`` at the pivot
    and ``logM_1`` the exponent of ``M``, i.e.
    ``log10 M(z) = logM_0 + logM_1 * log10((1 + z) / (1 + z_pivot))``.
    A parameter with neither coefficient follows the linear law of
    ``"evolution"`` mode (``d_<x>_dz``, ``C_<x>``; by default the published
    values), so some parameters can evolve as power laws while the rest track
    the published evolution or, with a zero slope, stay constant.

The ``log10`` -> linear mass conversion lives **only** here, at the boundary
before the model is called.
"""

import math

from ..config import DEFAULT_HOD_PARAMS, DEFAULT_REDSHIFT_EVOLUTION
from ..models.parametrization import RedshiftParametrization

__all__ = [
    "HOD_PARAM_SPEC",
    "MODEL_PARAM_NAMES",
    "fixed_z_param_names",
    "evolution_param_names",
    "powerlaw_param_names",
    "default_fixed_z_values",
    "default_evolution_values",
    "ModelParametrization",
]


# Canonical specification of the eight HOD parameters.
# Each entry: (model_name, sample_name, is_log, slope_name, offset_name)
#   model_name  : key passed to Observables (linear mass for Mc/Mcut/Msat)
#   sample_name : fixed-z sampling-space name (log10 for masses)
#   is_log      : whether the sampled / evolved quantity is log10(mass)
#   slope_name  : evolution slope coefficient name (matches config)
#   offset_name : evolution offset coefficient name (matches config)
HOD_PARAM_SPEC = [
    ("eps0", "eps0", False, "d_eps0_dz", "C_eps0"),
    ("Mc", "logMc", True, "d_logMc_dz", "C_logMc"),
    ("a", "a", False, "d_a_dz", "C_a"),
    ("b", "b", False, "d_b_dz", "C_b"),
    ("sigma_UV", "sigma_UV", False, "d_sigmaUV_dz", "C_sigmaUV"),
    ("Mcut", "logMcut", True, "d_Mcut_dz", "C_Mcut"),
    ("Msat", "logMsat", True, "d_Msat_dz", "C_Msat"),
    ("asat", "asat", False, "d_asat_dz", "C_asat"),
]

# HOD parameter names understood by Observables / HODModel.update_parameters.
MODEL_PARAM_NAMES = [spec[0] for spec in HOD_PARAM_SPEC]


def fixed_z_param_names():
    """Sampling-space parameter names for ``fixed_z`` mode."""
    return [spec[1] for spec in HOD_PARAM_SPEC]


def evolution_param_names():
    """Sampling-space parameter names for ``evolution`` mode (16 coefficients)."""
    names = []
    for _, _, _, slope_name, offset_name in HOD_PARAM_SPEC:
        names.extend([slope_name, offset_name])
    return names


def powerlaw_param_names():
    """Sampling-space names for ``powerlaw`` mode: ``<x>_0`` (value at the
    pivot redshift) and ``<x>_1`` (exponent of ``1 + z``) per HOD parameter,
    with ``<x>`` the fixed-z sampling name (16 coefficients)."""
    names = []
    for _, sample_name, _, _, _ in HOD_PARAM_SPEC:
        names.extend(_powerlaw_names(sample_name))
    return names


def _powerlaw_names(sample_name):
    return f"{sample_name}_0", f"{sample_name}_1"


def default_fixed_z_values():
    """Published fixed-z parameter defaults (from ``DEFAULT_HOD_PARAMS``).

    Masses are returned in ``log10`` space under the sampling-space names
    (``logMc``, ``logMcut``, ``logMsat``); ``config`` stores these log values
    under the keys ``logMc``, ``Mcut``, ``Msat`` respectively.
    """
    hp = DEFAULT_HOD_PARAMS
    return {
        "eps0": hp["eps0"],
        "logMc": hp["logMc"],
        "a": hp["a"],
        "b": hp["b"],
        "sigma_UV": hp["sigma_UV"],
        "logMcut": hp["Mcut"],
        "logMsat": hp["Msat"],
        "asat": hp["asat"],
    }


def default_evolution_values():
    """Published evolution-coefficient defaults (``DEFAULT_REDSHIFT_EVOLUTION``)."""
    return dict(DEFAULT_REDSHIFT_EVOLUTION)


class ModelParametrization:
    """Translate a named parameter dict into per-redshift model parameters.

    Parameters
    ----------
    mode : {"fixed_z", "evolution", "powerlaw"}
        Redshift handling. ``"fixed_z"`` samples HOD parameters directly at one
        redshift; ``"evolution"`` samples the linear evolution coefficients and
        evaluates per-bin parameters; ``"powerlaw"`` evolves each parameter
        with power-law coefficients ``<x>_0``, ``<x>_1`` as
        ``A_0 * ((1 + z) / (1 + z_pivot))**A_1``, falling back to the linear
        law for parameters without them.
    parameter_set : ParameterSet, optional
        If supplied, its parameter names are validated against the names
        recognised by ``mode`` (typos are rejected).
    use_linear : bool, optional
        Evolution scheme passed to :class:`RedshiftParametrization`:
        ``param(z) = slope * z + offset`` (default) or the ``z / (1 + z)``
        normalised form. Used in ``"evolution"`` mode, and in ``"powerlaw"``
        mode for the parameters that follow the linear law.
    defaults : dict, optional
        Sampling-space values used to fill any parameter the ``ParameterSet``
        does not specify. Defaults to the published values
        (:func:`default_fixed_z_values` / :func:`default_evolution_values`), so
        a user may vary only a subset of parameters and leave the rest at their
        fiducial values. Pass an explicit dict to override. In ``"powerlaw"``
        mode the default is the published linear coefficients, so a parameter
        without power-law coefficients follows the published evolution.
    z_pivot : float, optional
        Pivot redshift of ``"powerlaw"`` mode, where ``A = A_0``. Default 0,
        i.e. ``A_0 * (1 + z)**A_1``.
    """

    _VALID_MODES = ("fixed_z", "evolution", "powerlaw")

    def __init__(self, mode, parameter_set=None, use_linear=True, defaults=None,
                 z_pivot=0.0):
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got {mode!r}"
            )
        self.mode = mode
        self.use_linear = use_linear
        self.z_pivot = float(z_pivot)
        self._evolution = RedshiftParametrization(use_linear=use_linear)
        if defaults is None:
            defaults = (
                default_fixed_z_values() if mode == "fixed_z"
                else default_evolution_values()
            )
        self.defaults = dict(defaults)
        if parameter_set is not None:
            self.validate(parameter_set)

    def required_names(self):
        """All sampling-space names recognised for the chosen mode."""
        if self.mode == "fixed_z":
            return fixed_z_param_names()
        if self.mode == "powerlaw":
            return powerlaw_param_names() + evolution_param_names()
        return evolution_param_names()

    def _uses_powerlaw(self, sample_name, available):
        """Whether a parameter follows the power law (``powerlaw`` mode):
        both of its coefficients are in ``available`` (names or a dict) or
        ``defaults``. Exactly one is an error: the law would be undefined."""
        has = [n in available or n in self.defaults
               for n in _powerlaw_names(sample_name)]
        if all(has):
            return True
        if any(has):
            raise ValueError(
                f"Power-law evolution of '{sample_name}' needs both "
                f"{' and '.join(_powerlaw_names(sample_name))}")
        return False

    def validate(self, parameter_set):
        """Reject unknown parameter names and unresolvable required names.

        Every name in ``parameter_set`` must be a recognised name for the mode
        (guards against typos), and every required name must be resolvable from
        either the parameter set or ``defaults``.
        """
        recognised = set(self.required_names())
        unknown = [
            p.name for p in parameter_set.parameters if p.name not in recognised
        ]
        if unknown:
            raise ValueError(
                f"Unrecognised parameter name(s) for mode '{self.mode}': "
                f"{unknown}. Valid names: {sorted(recognised)}"
            )
        provided = {p.name for p in parameter_set.parameters}
        if self.mode == "powerlaw":
            needed = []
            for _, sample_name, _, slope_name, offset_name in HOD_PARAM_SPEC:
                if not self._uses_powerlaw(sample_name, provided):
                    needed += [slope_name, offset_name]
        else:
            needed = recognised
        unresolvable = [
            n for n in needed
            if n not in provided and n not in self.defaults
        ]
        if unresolvable:
            raise ValueError(
                f"No value (parameter or default) for: {unresolvable}"
            )
        return True

    def params_for_redshift(self, named_params, z):
        """Return the model HOD parameter dict at redshift ``z``.

        Parameters
        ----------
        named_params : dict
            Named parameter values (e.g. from ``ParameterSet.to_dict``).
        z : float
            Redshift at which to evaluate the parameters.

        Returns
        -------
        dict
            ``{eps0, Mc, a, b, sigma_UV, Mcut, Msat, asat}`` with masses in
            linear M_sun, ready to pass to :class:`~halogal.model.Observables`.
        """
        def lookup(name):
            if name in named_params:
                return named_params[name]
            return self.defaults[name]

        out = {}
        for model_name, sample_name, is_log, slope_name, offset_name in (
            HOD_PARAM_SPEC
        ):
            if self.mode == "fixed_z":
                val = lookup(sample_name)
            elif (self.mode == "powerlaw"
                  and self._uses_powerlaw(sample_name, named_params)):
                amp_name, exp_name = _powerlaw_names(sample_name)
                amp, exponent = lookup(amp_name), lookup(exp_name)
                ratio = (1.0 + z) / (1.0 + self.z_pivot)
                if is_log:   # log10 M = logM_0 + logM_1 * log10(ratio)
                    val = amp + exponent * math.log10(ratio)
                else:
                    try:
                        val = amp * ratio ** exponent
                    except OverflowError:  # absurd exponent; rejected
                        val = math.inf
            else:
                slope = lookup(slope_name)
                offset = lookup(offset_name)
                val = float(self._evolution.evaluate(z, slope, offset)[0])
            if is_log:
                try:
                    val = 10.0 ** val
                except OverflowError:  # absurd log-mass; rejected downstream
                    val = math.inf
            out[model_name] = val
        return out

    @staticmethod
    def is_physical(model_params):
        """Whether a per-redshift model parameter dict is physically valid.

        Requires every value to be finite and ``eps0``, ``sigma_UV`` and the
        masses to be strictly positive. In ``fixed_z`` mode the priors normally
        guarantee this; in ``evolution`` mode they do not -- priors act on the
        slopes/offsets, and a linear law can cross zero inside the fitted
        redshift range. A non-positive ``sigma_UV`` does not make the model
        fail: it flips the sign of the error-function argument in ``N_cen``
        and yields a finite but meaningless prediction, so the likelihood
        rejects such points explicitly.

        Parameters
        ----------
        model_params : dict
            Output of :meth:`params_for_redshift`.

        Returns
        -------
        bool
        """
        for value in model_params.values():
            if not math.isfinite(value):
                return False
        for name in ("eps0", "sigma_UV", "Mc", "Mcut", "Msat"):
            if name in model_params and not model_params[name] > 0:
                return False
        return True

    def __repr__(self):
        if self.mode == "powerlaw":
            return (f"ModelParametrization(mode='powerlaw', "
                    f"z_pivot={self.z_pivot:g})")
        return f"ModelParametrization(mode='{self.mode}')"
