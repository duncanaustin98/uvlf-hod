"""UV luminosity and dust attenuation functions.

The dust prescription itself lives in :mod:`halogal.dust`; the functions here
take a ``dust`` argument (a :class:`~halogal.dust.DustModel`, or ``None`` for
no dust).
"""

import warnings

import numpy as np

from .config import C_UV_DEFAULT, DUST_C0, DUST_C1, DUST_Z_MAX
from .dust import (DEFAULT_DUST, BetaDust, Bouwens14BetaMUV, NoDust,
                   resolve_dust)


def _dust_from_alias(dust, flag, flag_name):
    """``dust`` model, or the deprecated boolean ``flag`` if it was given."""
    if flag is None:
        return resolve_dust(dust)
    warnings.warn(
        f"{flag_name}= is deprecated; pass dust=<DustModel> (or dust=None "
        f"for no dust) instead", DeprecationWarning, stacklevel=3)
    return resolve_dust(bool(flag))


def beta_color(z, MUV):
    """UV spectral slope (beta) as a function of redshift and magnitude.

    The Bouwens et al. (2013, 2014) relation; equivalent to
    ``Bouwens14BetaMUV().mean(z, MUV)``.

    Parameters
    ----------
    z : float or array_like
        Redshift (broadcast against ``MUV``)
    MUV : float or array_like
        UV absolute magnitude

    Returns
    -------
    beta : ndarray
        UV spectral slope
    """
    return np.atleast_1d(Bouwens14BetaMUV().mean(z, MUV))


def dust_attenuation(z, MUV, high_z_dust=True, C0=None, C1=None, dust=None):
    """UV dust attenuation as a function of redshift and observed magnitude.

    Parameters
    ----------
    z : float or array_like
        Redshift (broadcast against ``MUV``)
    MUV : float or array_like
        UV absolute magnitude (observed)
    high_z_dust : bool, optional
        Whether to include dust at z > 8. Default is True. Ignored if
        ``dust`` is given.
    C0, C1 : float, optional
        Meurer-type ``A_UV = C0 + C1 beta`` coefficients, i.e. dA_dbeta = C1
        and beta_intr = -C0/C1. Default Meurer+99. Ignored if ``dust`` is
        given.
    dust : DustModel, optional
        Dust model to evaluate. Default: :class:`~halogal.dust.BetaDust`
        built from ``C0``, ``C1`` and ``high_z_dust``.

    Returns
    -------
    A_UV : ndarray
        UV dust attenuation in magnitudes (>= 0)

    Notes
    -----
    The beta-M_UV relation is calibrated on observed magnitudes, so
    :func:`MUV_from_SFR` applies this iteratively.
    """
    if dust is None:
        C0 = DUST_C0 if C0 is None else C0
        C1 = DUST_C1 if C1 is None else C1
        dust = BetaDust(dA_dbeta=C1, beta_intr=-C0 / C1,
                        z_max=None if high_z_dust else DUST_Z_MAX)
    return np.atleast_1d(resolve_dust(dust).A_UV(z, MUV))


def MUV_from_SFR(sfr, z, dust=DEFAULT_DUST, c_uv=None,
                 convergence_threshold=0.02, add_dust=None):
    """Convert star formation rate to UV absolute magnitude.

    Parameters
    ----------
    sfr : float or array_like
        Star formation rate in M_sun/yr
    z : float
        Redshift
    dust : DustModel or None, optional
        Dust model (see :mod:`halogal.dust`). Default
        :data:`~halogal.dust.DEFAULT_DUST`; ``None`` for no dust.
    c_uv : float, optional
        SFR to UV luminosity conversion factor.
        Default is Salpeter IMF value from config.
    convergence_threshold : float, optional
        Convergence criterion for dust iteration. Default is 0.02 (2%).
    add_dust : bool, optional
        Deprecated; use ``dust``.

    Returns
    -------
    MUV_obs : ndarray
        Observed (dust-attenuated) UV absolute magnitude

    Notes
    -----
    Dust attenuation is added iteratively until convergence because
    the M_UV-beta relation is calibrated on observed magnitudes.
    """
    dust = _dust_from_alias(dust, add_dust, "add_dust")
    if c_uv is None:
        c_uv = C_UV_DEFAULT

    sfr = np.atleast_1d(sfr)

    # Convert SFR to UV luminosity, then intrinsic AB magnitude
    LUV = sfr / c_uv  # erg/s/Hz
    M_UV = 51.63 - 2.5 * np.log10(LUV)

    # Add dust attenuation iteratively
    M_UV_obs = M_UV.copy()
    if not isinstance(dust, NoDust):
        M_UV_obs_prev = np.ones_like(M_UV_obs)
        max_iterations = 20
        iteration = 0
        while (np.sum(np.abs((M_UV_obs_prev - M_UV_obs) / M_UV_obs)) >
               convergence_threshold and iteration < max_iterations):
            M_UV_obs_prev = M_UV_obs.copy()
            M_UV_obs = M_UV + dust.A_UV(z, M_UV_obs)
            iteration += 1

    return M_UV_obs


def SFR_from_MUV(MUV, z, dust=DEFAULT_DUST, c_uv=None, remove_dust=None):
    """Convert UV absolute magnitude to star formation rate.

    Parameters
    ----------
    MUV : float or array_like
        Observed UV absolute magnitude
    z : float
        Redshift
    dust : DustModel or None, optional
        Dust model to remove. Default :data:`~halogal.dust.DEFAULT_DUST`;
        ``None`` for no dust correction.
    c_uv : float, optional
        SFR to UV luminosity conversion factor.
    remove_dust : bool, optional
        Deprecated; use ``dust``.

    Returns
    -------
    sfr : ndarray
        Star formation rate in M_sun/yr
    """
    dust = _dust_from_alias(dust, remove_dust, "remove_dust")
    if c_uv is None:
        c_uv = C_UV_DEFAULT

    MUV = np.atleast_1d(MUV)
    MUV_intrinsic = MUV - dust.A_UV(z, MUV)

    LUV = 10**((51.63 - MUV_intrinsic) / 2.5)
    return LUV * c_uv
