"""Unified model classes for UVHMR and HOD calculations.

This module provides a class-based interface for all galaxy population
modeling calculations, combining UV-halo mass relations with halo
occupation distributions.
"""

import numpy as np
from scipy import interpolate, integrate
from scipy.special import erf
from scipy.stats import norm
from abc import ABC, abstractmethod

from .config import BARYON_FRACTION
from .luminosity import MUV_from_SFR
from .cosmology import get_halo_mass_function, get_halo_bias

try:
    import halomod
    import halomod.hod
    HALOMOD_AVAILABLE = True
except ImportError:
    HALOMOD_AVAILABLE = False


class HMRModel(ABC):
    """Base class for Halo Mass Relation models.
        
    This class handles the relationship between halo mass and UV luminosity
    through star formation. It serves as the foundation for more complex
    models that include occupation distributions.
    
    Parameters
    ----------
    z : float
        Redshift
    eps0 : float, optional
        Star formation efficiency normalization. Default from config.
    Mc : float, optional
        Characteristic halo mass in M_sun. Default from config.
    a : float, optional
        Low-mass power-law slope. Default from config.
    b : float, optional
        High-mass power-law slope. Default from config.
        
    Attributes
    ----------
    z, eps0, Mc, a, b : float
        Model parameters
        
    Examples
    --------
    >>> # Use all defaults
    >>> model = HMRModel(z=6.0)
    >>> 
    >>> # Override specific parameters
    >>> model = HMRModel(z=6.0, eps0=0.2, Mc=10**12)
    """

    #: Names accepted by ``update_parameters`` (extended by subclasses).
    _PARAM_NAMES = ('z', 'eps0', 'Mc', 'a', 'b')

    @classmethod
    def param_names(cls):
        """Names of all parameters accepted by :meth:`update_parameters`."""
        return cls._PARAM_NAMES

    def __init__(self, z, eps0=None, Mc=None, a=None, b=None):
        """Initialize UV/Mstar HMR model."""
        from .config import DEFAULT_HOD_PARAMS
        
        self.z = z
        self.eps0 = eps0 if eps0 is not None else DEFAULT_HOD_PARAMS['eps0']
        self.Mc = Mc if Mc is not None else 10**DEFAULT_HOD_PARAMS['logMc']
        self.a = a if a is not None else DEFAULT_HOD_PARAMS['a']
        self.b = b if b is not None else DEFAULT_HOD_PARAMS['b']

    @abstractmethod
    def __call__(self, Mh):
        pass
 
    def __repr__(self):
        return (f"{type(self).__name__}(z={self.z}, eps0={self.eps0:.3f}, "
                f"Mc={self.Mc:.2e}, a={self.a:.2f}, b={self.b:.2f})")

    def _settings(self):
        """Non-parameter options to report in ``__str__``."""
        return {}

    @abstractmethod
    def property(self, Mh):
        pass

    def star_formation_efficiency(self, Mh):
        """Compute star formation efficiency.
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        epsilon : float or ndarray
            Star formation efficiency
        """
        Mh = np.atleast_1d(Mh)
        return 2 * self.eps0 / ((Mh / self.Mc)**(-self.a) + (Mh / self.Mc)**self.b)

    def Mhalo(self, property, M_min=1e6, M_max=1e16, num_points=1000):
        """Inverse: halo mass from relevant property (i.e. MUV or Mstar).
        
        Parameters
        ----------
        property : float or array_like
            UV absolute magnitude or stellar mass
        M_min, M_max : float, optional
            Halo mass range for interpolation
        num_points : int, optional
            Number of points for interpolation
            
        Returns
        -------
        Mh : float or ndarray
            Halo mass in M_sun
        """
        # Create forward relation
        Mhalo_grid = np.logspace(np.log10(M_min), np.log10(M_max), num_points)
        property_grid = self.property(Mhalo_grid)
        
        # Invert using interpolation
        interp_func = interpolate.interp1d(property_grid, Mhalo_grid, 
                                           fill_value='extrapolate')
        return interp_func(property)
    
    def update_parameters(self, **kwargs):
        """Update model parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters to update: z, eps0, Mc, a, b, add_dust
        """
        if 'z' in kwargs:
            self.z = kwargs['z']
        if 'eps0' in kwargs:
            self.eps0 = kwargs['eps0']
        if 'Mc' in kwargs:
            self.Mc = kwargs['Mc']
        if 'a' in kwargs:
            self.a = kwargs['a']
        if 'b' in kwargs:
            self.b = kwargs['b']


class UVHMRModel(HMRModel):
    """Base class for UV-Halo Mass Relation models.
    
    This class handles the relationship between halo mass and UV luminosity
    through star formation. It serves as the foundation for more complex
    models that include occupation distributions.
    
    Parameters
    ----------
    z : float
        Redshift
    eps0 : float, optional
        Star formation efficiency normalization. Default from config.
    Mc : float, optional
        Characteristic halo mass in M_sun. Default from config.
    a : float, optional
        Low-mass power-law slope. Default from config.
    b : float, optional
        High-mass power-law slope. Default from config.
    add_dust : bool, optional
        Whether to include dust attenuation. Default is True.
        
    Attributes
    ----------
    z, eps0, Mc, a, b : float
        Model parameters
    add_dust : bool
        Whether dust is included
        
    Examples
    --------
    >>> # Use all defaults (fitted to Bouwens+2021)
    >>> model = UVHMRModel(z=6.0)
    >>> 
    >>> # Override specific parameters
    >>> model = UVHMRModel(z=6.0, eps0=0.2, Mc=10**12)
    """
    
    _PARAM_NAMES = HMRModel._PARAM_NAMES + ('add_dust',)

    def __init__(self, z, eps0=None, Mc=None, a=None, b=None, add_dust=True):
        """Initialize UVHMR model."""
        self.add_dust = add_dust
        super().__init__(z, eps0=eps0, Mc=Mc, a=a, b=b)
    
    def halo_accretion_rate(self, Mh):
        """Compute halo mass accretion rate.
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        dMh_dt : float or ndarray
            Halo mass accretion rate in M_sun/yr
        """
        Mh = np.atleast_1d(Mh)
        return 0.03e-9 * Mh * (Mh / 1e12)**0.14 * (1 + self.z)**2.5
    
    def __call__(self, Mh):
        """Abstract method override equivalent to the SFR
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        sfr : float or ndarray
            Star formation rate in M_sun/yr
        """
        f_star = self.star_formation_efficiency(Mh)
        return BARYON_FRACTION * f_star * self.halo_accretion_rate(Mh)

    def sfr(self, Mh):
        """Star formation rate from halo mass.
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        sfr : float or ndarray
            Star formation rate in M_sun/yr
        """
        return self(Mh)

    def property(self, Mh):
        """Abstract method override equivalent to MUV
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        MUV : float or ndarray
            UV absolute magnitude
        """
        sfr = self.sfr(Mh)
        return MUV_from_SFR(sfr, self.z, add_dust=self.add_dust)

    def MUV(self, Mh):
        """UV absolute magnitude from halo mass.
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        MUV : float or ndarray
            UV absolute magnitude
        """
        return self.property(Mh)

    def _settings(self):
        return {'add_dust': self.add_dust}

    def update_parameters(self, **kwargs):
        """Update model parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters to update: z, eps0, Mc, a, b, add_dust
        """
        if 'add_dust' in kwargs:
            self.add_dust = kwargs['add_dust']
            #kwargs.pop('add_dust')
        super().update_parameters(**kwargs)


class SHMRModel(HMRModel):
    """Base class for Stellar-Halo Mass Relation models.
    
    This class handles the relationship between stellar and halo mass
    through star formation. It serves as the foundation for more complex
    models that include occupation distributions.
    
    Parameters
    ----------
    z : float
        Redshift
    eps0 : float, optional
        Star formation efficiency normalization. Default from config.
    Mc : float, optional
        Characteristic halo mass in M_sun. Default from config.
    a : float, optional
        Low-mass power-law slope. Default from config.
    b : float, optional
        High-mass power-law slope. Default from config.
        
    Attributes
    ----------
    z, eps0, Mc, a, b : float
        Model parameters
        
    Examples
    --------
    >>> # Use all defaults
    >>> model = SHMRModel(z=6.0)
    >>> 
    >>> # Override specific parameters
    >>> model = SHMRModel(z=6.0, eps0=0.2, Mc=10**12)
    """
    
    def __call__(self, Mh):
        """Abstract method override equivalent to the stellar mass
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        Mstar : float or ndarray
            Stellar mass in M_sun
        """
        f_star = self.star_formation_efficiency(Mh)
        return BARYON_FRACTION * f_star * Mh

    def Mstar(self, Mh):
        """Stellar mass from halo mass.
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        Mstar : float or ndarray
            Stellar mass in M_sun
        """
        return self(Mh)

    def property(self, Mh):
        """Abstract method override equivalent to Mstar
        
        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
            
        Returns
        -------
        Mstar : float or ndarray
            Stellar mass
        """
        return self(Mh)


class BaseHODModel(HMRModel):
    """Shared Halo Occupation Distribution machinery.

    Adds central and satellite occupation functions on top of an
    :class:`HMRModel`. Concrete models combine this with a specific HMR
    (e.g. ``class UVHODModel(BaseHODModel, UVHMRModel)``) and supply:

    * :meth:`_selection_variable` -- mean M_UV, log10 Mstar, ... at fixed Mh
    * ``_selection_sign`` -- +1 to select above the threshold, -1 below
    * ``_sigma_name`` -- name of the scatter parameter (e.g. ``'sigma_UV'``),
      which is also its ``DEFAULT_HOD_PARAMS`` key
    * ``_sigma_unit`` -- unit label of the scatter used in ``__str__``

    The central and satellite occupations, parameter updates and printing
    are all shared.

    Parameters
    ----------
    z : float
        Redshift
    eps0, Mc, a, b : float, optional
        HMR parameters. Default from config.
    sigma : float, optional
        Scatter of the central occupation, stored as the attribute named by
        ``_sigma_name``. Subclasses expose it under that name. Default from
        config.
    Mcut : float, optional
        Satellite cutoff mass in M_sun. Default from config.
    Msat : float, optional
        Satellite normalization mass in M_sun. Default from config.
    asat : float, optional
        Satellite power-law slope. Default from config.
    **kwargs
        Extra HMR arguments (e.g. ``add_dust`` for :class:`UVHMRModel`).

    Attributes
    ----------
    Mcut, Msat, asat : float
        Satellite HOD parameters
    """

    _sigma_name = None
    _sigma_unit = ''
    _SAT_PARAMS = ('Mcut', 'Msat', 'asat')

    #: +1 if the sample is selected *above* ``thresh`` (e.g. stellar mass),
    #: -1 if *below* it (e.g. magnitudes, where brighter is more negative).
    _selection_sign = None

    def __init__(self, z, eps0=None, Mc=None, a=None, b=None,
                 sigma=None, Mcut=None, Msat=None, asat=None, **kwargs):
        from .config import DEFAULT_HOD_PARAMS

        super().__init__(z, eps0=eps0, Mc=Mc, a=a, b=b, **kwargs)

        setattr(self, self._sigma_name,
                sigma if sigma is not None
                else DEFAULT_HOD_PARAMS[self._sigma_name])
        self.Mcut = Mcut if Mcut is not None else 10**DEFAULT_HOD_PARAMS['Mcut']
        self.Msat = Msat if Msat is not None else 10**DEFAULT_HOD_PARAMS['Msat']
        self.asat = asat if asat is not None else DEFAULT_HOD_PARAMS['asat']

    @classmethod
    def param_names(cls):
        """Names of all parameters accepted by :meth:`update_parameters`."""
        return super().param_names() + (cls._sigma_name, *cls._SAT_PARAMS)

    @property
    def _sigma(self):
        """Value of the scatter parameter, whatever the subclass calls it."""
        return getattr(self, self._sigma_name)

    @abstractmethod
    def _selection_variable(self, Mh):
        """Mean selection variable at fixed halo mass.

        The quantity whose scatter at fixed Mh is Gaussian with width
        given by the scatter parameter, and which ``thresh`` is compared
        against (e.g. M_UV or log10 Mstar).
        """

    def Ncen(self, Mh, thresh):
        """Central galaxy occupation function.

        Fraction of haloes of mass Mh whose central passes ``thresh``,
        assuming Gaussian scatter in the selection variable::

            N_cen = 0.5 * [1 + erf(s * (x(Mh) - thresh) / (sqrt(2) sigma))]

        with ``x`` = :meth:`_selection_variable`, ``s`` =
        ``_selection_sign`` and ``sigma`` the model's scatter parameter.

        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
        thresh : float
            Sample threshold in the units of the selection variable

        Returns
        -------
        N_cen : float or ndarray
            Average number of central galaxies
        """
        x_mean = self._selection_variable(Mh)
        return 0.5 * (1 + erf(self._selection_sign * (x_mean - thresh) /
                              (np.sqrt(2) * self._sigma)))

    def Nsat(self, Mh, thresh):
        """Satellite galaxy occupation function.

        Average number of satellites passing ``thresh``, modulated by the
        central occupation.

        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
        thresh : float
            Sample threshold, in the units expected by :meth:`Ncen`

        Returns
        -------
        N_sat : float or ndarray
            Average number of satellite galaxies
        """
        # astype(float): an integer mass array would truncate N_sat to
        # whole numbers (and zeros_like would make it an integer array).
        Mh = np.atleast_1d(Mh).astype(float)
        N_cen = self.Ncen(Mh, thresh)

        # Satellites only exist above cutoff mass
        N_sat = np.zeros_like(Mh)
        mask = Mh > self.Mcut
        N_sat[mask] = ((Mh[mask] - self.Mcut) / self.Msat)**self.asat

        return N_sat * N_cen

    def Ngal(self, Mh, thresh):
        """Total galaxy occupation (central + satellite).

        Parameters
        ----------
        Mh : float or array_like
            Halo mass in M_sun
        thresh : float
            Sample threshold, in the units expected by :meth:`Ncen`

        Returns
        -------
        N_gal : float or ndarray
            Average total number of galaxies
        """
        return self.Ncen(Mh, thresh) + self.Nsat(Mh, thresh)

    def update_parameters(self, **kwargs):
        """Update model parameters.

        Parameters
        ----------
        **kwargs : dict
            HMR parameters plus the scatter parameter, Mcut, Msat and asat.
        """
        super().update_parameters(**kwargs)
        for name in (self._sigma_name, *self._SAT_PARAMS):
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def __repr__(self):
        return (f"{type(self).__name__}(z={self.z}, eps0={self.eps0:.3f}, "
                f"Mc={self.Mc:.2e}, a={self.a:.2f}, b={self.b:.2f}, "
                f"{self._sigma_name}={self._sigma:.2f})")

    def __str__(self):
        s = f"{type(self).__name__} at z={self.z}\n"
        s += "="*50 + "\n"
        s += "HMR Parameters:\n"
        s += f"  eps0 = {self.eps0:.3f}\n"
        s += f"  Mc = {self.Mc:.2e} M_sun\n"
        s += f"  a = {self.a:.2f}\n"
        s += f"  b = {self.b:.2f}\n"
        s += "\nHOD Parameters:\n"
        s += f"  {self._sigma_name} = {self._sigma:.2f} {self._sigma_unit}\n"
        s += f"  Mcut = {self.Mcut:.2e} M_sun\n"
        s += f"  Msat = {self.Msat:.2e} M_sun\n"
        s += f"  asat = {self.asat:.2f}\n"
        settings = self._settings()
        if settings:
            s += "\nSettings:\n"
            for k, v in settings.items():
                s += f"  {k} = {v}\n"
        s += "="*50
        return s


class UVHODModel(BaseHODModel, UVHMRModel):
    """UV-selected Halo Occupation Distribution model.

    Combines :class:`UVHMRModel` with central and satellite occupation
    functions for samples brighter than a UV magnitude threshold.

    Parameters
    ----------
    z : float
        Redshift
    eps0, Mc, a, b : float, optional
        UVHMR parameters. Default from config.
    sigma_UV : float, optional
        Scatter in M_UV at fixed halo mass, in mag. Default from config.
    Mcut, Msat, asat : float, optional
        Satellite parameters. Default from config.
    add_dust : bool, optional
        Whether to include dust attenuation. Default is True.

    Attributes
    ----------
    sigma_UV, Mcut, Msat, asat : float
        HOD parameters

    Examples
    --------
    >>> model = UVHODModel(z=6.0, sigma_UV=0.5)
    >>> Mh = np.logspace(10, 13, 100)
    >>> N_cen = model.Ncen(Mh, -19.0)   # M_UV < -19
    >>> N_sat = model.Nsat(Mh, -19.0)
    """

    _sigma_name = 'sigma_UV'
    _sigma_unit = 'mag'
    _selection_sign = -1    # M_UV < thresh (brighter than)

    def __init__(self, z, eps0=None, Mc=None, a=None, b=None,
                 sigma_UV=None, Mcut=None, Msat=None, asat=None,
                 add_dust=True):
        super().__init__(z, eps0, Mc, a, b, sigma_UV, Mcut, Msat, asat,
                         add_dust=add_dust)

    def _selection_variable(self, Mh):
        return self.MUV(Mh)


class MstarHODModel(BaseHODModel, SHMRModel):
    """Stellar-mass-selected Halo Occupation Distribution model.

    Combines :class:`SHMRModel` with central and satellite occupation
    functions for samples above a stellar mass threshold.

    Parameters
    ----------
    z : float
        Redshift
    eps0, Mc, a, b : float, optional
        SHMR parameters. Default from config.
    sigma_Mstar : float, optional
        Scatter in log10(Mstar) at fixed halo mass, in dex.
        Default from config.
    Mcut, Msat, asat : float, optional
        Satellite parameters. Default from config.

    Attributes
    ----------
    sigma_Mstar, Mcut, Msat, asat : float
        HOD parameters

    Examples
    --------
    >>> model = MstarHODModel(z=6.0, sigma_Mstar=0.3)
    >>> Mh = np.logspace(10, 13, 100)
    >>> N_cen = model.Ncen(Mh, 9.0)   # log10(Mstar/M_sun) > 9
    """

    _sigma_name = 'sigma_Mstar'
    _sigma_unit = 'dex'
    _selection_sign = +1    # log10(Mstar) > thresh (more massive than)

    def __init__(self, z, eps0=None, Mc=None, a=None, b=None,
                 sigma_Mstar=None, Mcut=None, Msat=None, asat=None):
        super().__init__(z, eps0, Mc, a, b, sigma_Mstar, Mcut, Msat, asat)

    def _selection_variable(self, Mh):
        return np.log10(self.Mstar(Mh))


# Backward-compatible alias: HODModel has always meant the UV-selected model.
HODModel = UVHODModel


#: Sentinel for "leave unchanged" in update_correlation_model, so that
#: ``None`` can mean "no second threshold".
_UNSET = object()


class Observables:
    """Compute observables from an HOD model.

    Base class for :class:`UVObservables` and :class:`MstarObservables`.
    Calling ``Observables(model)`` directly returns the subclass matching
    the model type, so ``Observables(UVHODModel(...))`` is a
    :class:`UVObservables` and ``Observables(MstarHODModel(...))`` a
    :class:`MstarObservables`.

    The generic methods work in terms of the model's *selection variable*
    ``x`` (M_UV for UV models, log10 Mstar for stellar-mass models) and a
    threshold ``thresh`` on it; see :meth:`BaseHODModel.Ncen`. Subclasses add
    named wrappers (``luminosity_function``, ``stellar_mass_function``).

    All observable methods accept optional ``**params`` keyword arguments
    (any of ``hod_model.param_names()``) to update the underlying HOD model
    before computing. This enables efficient MCMC loops without recreating
    any objects.

    For correlation functions specifically, use the initialize/update pattern:
    1. Call initialize_correlation_model() once
    2. Call update_correlation_model() in your MCMC loop

    Parameters
    ----------
    hod_model : BaseHODModel
        The HOD model to use for computations

    Attributes
    ----------
    hod_model : BaseHODModel
        The underlying HOD model

    Examples
    --------
    >>> obs = Observables(UVHODModel(z=6.0))        # -> UVObservables
    >>> MUV = np.linspace(-22, -16, 20)
    >>> phi = obs.luminosity_function(MUV)

    >>> # MCMC fitting - all observables support inline parameter updates
    >>> for eps0, sigma_UV in mcmc_samples:
    ...     phi = obs.luminosity_function(MUV, eps0=eps0, sigma_UV=sigma_UV)
    ...     bg = obs.galaxy_bias(MUV, eps0=eps0, sigma_UV=sigma_UV)

    >>> # Correlation function MCMC (uses halomod caching)
    >>> obs.initialize_correlation_model(MUV_thresh1=-19.1)
    >>> for eps0, sigma_UV in mcmc_samples:
    ...     result = obs.update_correlation_model(eps0=eps0, sigma_UV=sigma_UV)
    ...     w_theta = result['correlation']
    """

    #: HOD model class this observables class works with (set by subclasses).
    _model_class = None
    #: Halomod HOD wrapper used for correlation functions (set by subclasses).
    _halomod_hod_class = None

    def __new__(cls, hod_model=None, *args, **kwargs):
        # Observables(model) dispatches on the model type. hod_model defaults
        # to None so that unpickling (which calls cls.__new__(cls)) works.
        if cls is Observables:
            for sub in cls.__subclasses__():
                if sub._model_class is not None and isinstance(hod_model, sub._model_class):
                    cls = sub
                    break
            else:
                raise TypeError(
                    f"No Observables subclass for {type(hod_model).__name__}; "
                    "expected a UVHODModel or MstarHODModel."
                )
        return super().__new__(cls)

    def __init__(self, hod_model):
        """Initialize observables calculator.

        Parameters
        ----------
        hod_model : BaseHODModel
            The HOD model to use
        """
        if not isinstance(hod_model, self._model_class):
            raise TypeError(
                f"{type(self).__name__} requires a {self._model_class.__name__}, "
                f"got {type(hod_model).__name__}"
            )
        self.hod_model = hod_model
        self._hmf_cache = {}

        # For efficient correlation function updates
        self._halomod_model = None
        self._halomod_config = {}
        self._hod_wrapper = None

    def _apply_params(self, params):
        """Apply parameter updates to the underlying HOD model.

        Parameters
        ----------
        params : dict
            Keyword arguments, possibly containing HOD parameter overrides.

        Returns
        -------
        remaining : dict
            Any kwargs that are not HOD parameters (always empty: unknown
            keys raise).

        Raises
        ------
        TypeError
            If any keyword is not a recognised HOD parameter. Every caller
            used to discard the leftovers, so a misspelling such as
            ``sigmaUV`` for ``sigma_UV`` was silently ignored and the whole
            calculation quietly ran at the default value.
        """
        names = self.hod_model.param_names()
        hod_updates = {k: v for k, v in params.items() if k in names}
        remaining = {k: v for k, v in params.items() if k not in names}
        if remaining:
            raise TypeError(
                f"unrecognised parameter(s) {sorted(remaining)}; expected any "
                f"of {sorted(names)}"
            )
        if hod_updates:
            self.hod_model.update_parameters(**hod_updates)
        return remaining

    def _get_hmf(self, recalculate=False):
        """Get HMF - from halomod if available, otherwise from Colossus.
        Use halomod HMF when available for
        consistency with correlation functions, otherwise use Colossus.
        Parameters
        ----------
        recalculate : bool, optional
            Force recalculation. Default is False.
        Returns
        -------
        log10_mass : ndarray
            Array of log10(M/M_sun) in physical units
        hmf : ndarray
            Halo mass function dn/dlog10(M) in physical units of
            Mpc^-3 dex^-1, to be integrated against ``log10_mass``.

        Notes
        -----
        Both branches return dn/dlog10M, matching the measure every caller
        integrates with (``x=log10_Mh``). halomod natively supplies dn/dlnM,
        so the halomod branch applies the ln(10) change of measure; omitting
        it made every absolute quantity on that path low by exactly 1/ln(10)
        = 0.4343, while ratio quantities (bias, mean halo mass) were immune
        and so looked correct.
        """
        # If halomod model exists, use its HMF for consistency
        if self._halomod_model is not None:
            # Extract from halomod
            M_h = self._halomod_model.m  # M_sun/h
            dndlnm_h = self._halomod_model.dndlnm  # (h/Mpc)^3

            # Convert to physical units, then dlnM -> dlog10M
            h = self._halomod_config['h']
            M_physical = M_h / h  # M_sun
            hmf_physical = dndlnm_h * h**3 * np.log(10)  # Mpc^-3 dex^-1

            return np.log10(M_physical), hmf_physical

        # Otherwise use Colossus (backward compatible)
        cache_key = self.hod_model.z
        if recalculate or cache_key not in self._hmf_cache:
            log10_Mh, hmf = get_halo_mass_function(
                self.hod_model.z, M_min=6, M_max=15, num_points=2048
            )
            self._hmf_cache[cache_key] = (log10_Mh, hmf)
        return self._hmf_cache[cache_key]

    def _bin_occupation(self, Mh, x, dx):
        """Mean galaxies per halo with selection variable in [x - dx/2, x + dx/2].

        ``Ngal(Mh, thresh)`` counts galaxies on the selected side of
        ``thresh``, so a bin is the difference of two thresholds, signed so
        that it is positive whichever side the model selects on.
        """
        s = self.hod_model._selection_sign
        return s * (self.hod_model.Ngal(Mh, x - dx/2)
                    - self.hod_model.Ngal(Mh, x + dx/2))

    def abundance_function(self, x_arr, dx=0.05, **params):
        """Abundance per unit selection variable, e.g. the UVLF or SMF.

        Parameters
        ----------
        x_arr : array_like
            Values of the selection variable (M_UV or log10 Mstar)
        dx : float, optional
            Bin width. Default is 0.05.
        **params : dict, optional
            HOD parameter overrides. Applied to the model before computing.

        Returns
        -------
        phi : ndarray
            Abundance in Mpc^-3 per unit x (mag^-1 or dex^-1)
        """
        self._apply_params(params)

        # astype(float): np.arange(-22, -16) is integer, and zeros_like
        # would then silently return an all-zero integer abundance.
        x_arr = np.atleast_1d(x_arr).astype(float)
        log10_Mh, hmf = self._get_hmf()
        Mh = 10**log10_Mh

        phi = np.zeros_like(x_arr)
        for i, x in enumerate(x_arr):
            integrand = self._bin_occupation(Mh, x, dx) * hmf
            phi[i] = integrate.simpson(integrand, x=log10_Mh) / dx

        return phi

    def galaxy_bias(self, x_arr, dx=0.05, **params):
        """Galaxy bias in bins of the selection variable.

        Parameters
        ----------
        x_arr : array_like
            Values of the selection variable (M_UV or log10 Mstar)
        dx : float, optional
            Bin width. Default is 0.05.
        **params : dict, optional
            HOD parameter overrides. Applied to the model before computing.

        Returns
        -------
        bias : ndarray
            Galaxy bias
        """
        self._apply_params(params)

        x_arr = np.atleast_1d(x_arr).astype(float)  # see abundance_function
        log10_Mh, hmf = self._get_hmf()
        Mh = 10**log10_Mh
        bias_halo = get_halo_bias(Mh, self.hod_model.z, mdef='vir', model='tinker10')

        bg = np.zeros_like(x_arr)
        for i, x in enumerate(x_arr):
            N_bin = self._bin_occupation(Mh, x, dx)
            ngal = integrate.simpson(N_bin * hmf, x=log10_Mh)
            if ngal > 0:
                bg[i] = integrate.simpson(N_bin * hmf * bias_halo, x=log10_Mh) / ngal
            else:
                bg[i] = 0.0

        return bg

    def mean_halo_mass(self, thresh, **params):
        """Mean halo mass of galaxies passing ``thresh``.

        Parameters
        ----------
        thresh : float
            Sample threshold on the selection variable
        **params : dict, optional
            HOD parameter overrides.

        Returns
        -------
        log10_Mh_mean : float
            log10 of the mean halo mass
        """
        self._apply_params(params)

        log10_Mh, hmf = self._get_hmf()
        Mh = 10**log10_Mh
        N_gal = self.hod_model.Ngal(Mh, thresh)

        ngal = integrate.simpson(N_gal * hmf, x=log10_Mh)
        if ngal > 0:
            mh_mean = integrate.simpson(N_gal * hmf * Mh, x=log10_Mh) / ngal
            return np.log10(mh_mean)
        return np.nan

    def mean_bias(self, thresh, **params):
        """Mean galaxy bias of galaxies passing ``thresh``.

        Parameters
        ----------
        thresh : float
            Sample threshold on the selection variable
        **params : dict, optional
            HOD parameter overrides.

        Returns
        -------
        bias_mean : float
            Mean galaxy bias
        """
        self._apply_params(params)

        log10_Mh, hmf = self._get_hmf()
        Mh = 10**log10_Mh
        bias_halo = get_halo_bias(Mh, self.hod_model.z, mdef='vir', model='tinker10')
        N_gal = self.hod_model.Ngal(Mh, thresh)

        ngal = integrate.simpson(N_gal * hmf, x=log10_Mh)
        if ngal > 0:
            return integrate.simpson(N_gal * hmf * bias_halo, x=log10_Mh) / ngal
        return np.nan

    def number_density(self, thresh, **params):
        """Cumulative number density of galaxies passing ``thresh``.

        Parameters
        ----------
        thresh : float
            Sample threshold on the selection variable
        **params : dict, optional
            HOD parameter overrides.

        Returns
        -------
        n_gal : float
            Galaxy number density in Mpc^-3
        """
        self._apply_params(params)

        log10_Mh, hmf = self._get_hmf()
        Mh = 10**log10_Mh
        N_gal = self.hod_model.Ngal(Mh, thresh)

        return integrate.simpson(N_gal * hmf, x=log10_Mh)

    # ------------------------------------------------------------------
    # Correlation functions (halomod)
    # ------------------------------------------------------------------
    def _halomod_hod_params(self):
        """Current HOD parameters in the form the halomod wrapper expects."""
        cfg = self._halomod_config
        params = {n: getattr(self.hod_model, n)
                  for n in self.hod_model.param_names()}
        params.update(z=cfg['z_center'], thresh1=cfg['thresh1'],
                      thresh2=cfg['thresh2'], h=cfg['h'], M_min=cfg['M_min'])
        return params

    def _correlation_result(self):
        """Separation and correlation arrays from the cached halomod model."""
        correlation_type = self._halomod_config['correlation_type']
        h = self._halomod_config['h']
        kind = correlation_type.lower()
        if kind == 'angular':
            separation = self._halomod_model.theta * (180.0 / np.pi) * 3600.0
            correlation = self._halomod_model.angular_corr_gal
        elif kind == 'real':
            separation = self._halomod_model.r / h
            correlation = self._halomod_model.corr_auto_tracer
        elif kind == 'projected':
            separation = self._halomod_model.rp / h
            correlation = self._halomod_model.projected_corr_gal
        else:
            raise ValueError(f"Unknown correlation_type: {correlation_type}")
        return {
            'model': self._halomod_model,
            'separation': separation,
            'correlation': correlation,
            'type': correlation_type,
        }

    def initialize_correlation_model(self, thresh1, thresh2=None,
                                    cosmo_model=None, correlation_type='angular',
                                    zmin=None, zmax=None, znum=150,
                                    p1=None,
                                    theta_min=1.0, theta_max=7200.0, theta_num=100,
                                    rmin=1e-3, rmax=200.0, rnum=200,
                                    rp_min=0.1, rp_max=100.0, rp_num=100,
                                    pi_max=100.0,
                                    hmf_model='Watson', bias_model='Tinker10',
                                    concentration_model='Duffy08',
                                    halo_profile='NFW',
                                    exclusion_model='NgMatched_',
                                    sd_bias_model=None,
                                    **kwargs):
        """Initialize halomod model for efficient parameter updates.

        Call this once before MCMC fitting. Then use update_correlation_model()
        to efficiently update HOD parameters.

        This creates and caches the halomod model, allowing fast parameter
        updates via halomod's update() method.

        Parameters
        ----------
        thresh1 : float
            Sample threshold on the selection variable. On its own it defines
            a threshold sample.
        thresh2 : float, optional
            Second bound for a binned sample, on the *selected* side of
            ``thresh1`` (brighter for UV, more massive for Mstar). ``None``
            (default) gives a threshold sample.
        correlation_type : {'angular', 'real', 'projected'}
            Which correlation function to compute.
        **kwargs
            Remaining halomod settings (see source for defaults).

        Returns
        -------
        result : dict
            Dictionary with 'model', 'separation', 'correlation', 'type'
        """
        if not HALOMOD_AVAILABLE:
            raise ImportError("halomod package required")

        # Set up redshift range
        z_center = self.hod_model.z
        if zmin is None:
            zmin = max(0.1, z_center - 1.0)
        if zmax is None:
            zmax = z_center + 1.0
        
        # Set up cosmology and extract h. sigma_8 and n are passed separately
        # from cosmo_params (see config.halomod_cosmo_kwargs); omitting them
        # silently leaves halomod on its own Planck18 defaults while Colossus
        # uses halogal's, so the UVLF and the clustering describe different
        # universes.
        from .config import DEFAULT_COSMO_PARAMS, halomod_cosmo_kwargs
        power_params = {}
        if cosmo_model is None:
            _kw = halomod_cosmo_kwargs()
            cosmo_params = _kw.pop('cosmo_params')
            power_params = _kw          # sigma_8, n, transfer_model
            h = DEFAULT_COSMO_PARAMS['H0'] / 100.0
        elif isinstance(cosmo_model, dict):
            cosmo_params = cosmo_model
            if 'H0' not in cosmo_params:
                raise ValueError("cosmo_model dict must include 'H0'")
            h = cosmo_params['H0'] / 100.0
        else:
            cosmo_params = cosmo_model
            h = cosmo_model.H0.value / 100.0
        
        # Store configuration for updates
        self._halomod_config = {
            'thresh1': thresh1,
            'thresh2': thresh2,
            'correlation_type': correlation_type,
            'h': h,
            'z_center': z_center,
            'zmin': zmin,
            'zmax': zmax,
            'znum': znum,
            'cosmo_params': cosmo_params,
            'M_min': kwargs.pop('M_min', 9.0),
        }

        # Halomod parameters
        halomod_params = {
            'cosmo_params' : cosmo_params,
            **power_params,
            'z': z_center,
            'rmin': rmin * h,
            'rmax': rmax * h,
            'rnum': rnum,
            'Mmin': kwargs.pop('Mmin', 5),
            'Mmax': kwargs.pop('Mmax', 16),
            'dlog10m': kwargs.pop('dlog10m', 0.1),
            'dlnk': kwargs.pop('dlnk', 0.2),
            'lnk_min': kwargs.pop('lnk_min', -6),
            'lnk_max': kwargs.pop('lnk_max', 7),
            'hmf_model': hmf_model,
            'tracer_concentration_model': concentration_model,
            'halo_profile_model': halo_profile,
            'hc_spectrum': kwargs.pop('hc_spectrum', 'nonlinear'),
            'exclusion_model': exclusion_model,
            'bias_model': bias_model,
            'sd_bias_model': sd_bias_model,
            'hod_model' : self._halomod_hod_class,
            'hod_params' : self._halomod_hod_params(),
        }

        halomod_params.update(kwargs)

        if correlation_type.lower() == 'angular':
            theta_min_rad = theta_min / 3600.0 * (np.pi / 180.0)
            theta_max_rad = theta_max / 3600.0 * (np.pi / 180.0)
            
            if p1 is None:
                # Create a normalized callable directly
                def p1(z):
                    return norm.pdf(z, loc=z_center, scale=0.5)
            
            self._halomod_model = halomod.AngularCF(
                zmin=zmin,
                zmax=zmax,
                znum=znum,
                p1=p1,
                theta_min=theta_min_rad,
                theta_max=theta_max_rad,
                theta_num=theta_num,
                logu_min=kwargs.pop('logu_min', -3),
                logu_max=kwargs.pop('logu_max', 2.0),
                unum=kwargs.pop('unum', 100),
                dr_table=kwargs.pop('dr_table', 0.03),
                **halomod_params)
            
        elif correlation_type.lower() == 'real':
            self._halomod_model = halomod.TracerHaloModel(
                **halomod_params
            )
            
        elif correlation_type.lower() == 'projected':
            self._halomod_model = halomod.ProjectedCF(
                rp_min=rp_min * h,
                rp_max=rp_max * h,
                rp_num=rp_num,
                proj_limit=pi_max * h,
                **halomod_params
            )
            
        else:
            raise ValueError(f"Unknown correlation_type: {correlation_type}")

        return self._correlation_result()

    def update_correlation_model(self, z_center=None, p_of_z=None,
                                 thresh1=_UNSET, thresh2=_UNSET, **params):
        """Update HOD parameters and recompute correlation function efficiently.

        This uses halomod's update() method to efficiently recompute the
        correlation function with new parameters, without recreating the
        entire model.

        Must call initialize_correlation_model() first.

        Parameters
        ----------
        z_center : float, optional
            Update the redshift center
        p_of_z : callable, optional
            Update the redshift distribution function
        thresh1, thresh2 : float or None, optional
            Update the sample thresholds. Omit to leave unchanged;
            ``thresh2=None`` switches to a threshold sample.
        **params
            HOD parameter updates (any of ``hod_model.param_names()``).
            ``None`` values are ignored (left unchanged).

        Returns
        -------
        result : dict
            Dictionary with 'model', 'separation', 'correlation', 'type'

        Examples
        --------
        >>> result = obs.initialize_correlation_model(-19.1)
        >>> for eps0, sigma_UV in zip(eps0_chain, sigma_UV_chain):
        ...     result = obs.update_correlation_model(
        ...         eps0=eps0, sigma_UV=sigma_UV
        ...     )
        ...     likelihood = compute_likelihood(result['correlation'])
        """
        if self._halomod_model is None:
            raise ValueError(
                "Must call initialize_correlation_model() before update_correlation_model(). "
                "Or use compute_correlation_function() for one-off calculations."
            )

        self._apply_params({k: v for k, v in params.items() if v is not None})

        if thresh1 is not _UNSET:
            self._halomod_config['thresh1'] = thresh1
        if thresh2 is not _UNSET:
            self._halomod_config['thresh2'] = thresh2
        if z_center is not None:
            self._halomod_config['z_center'] = z_center

        # Always pass the full current HOD parameter set
        halomod_update = {'hod_params': self._halomod_hod_params()}
        if z_center is not None:
            halomod_update['z'] = z_center
        if p_of_z is not None:
            halomod_update['p1'] = p_of_z

        self._halomod_model.update(**halomod_update)

        return self._correlation_result()

    def compute_correlation_function(self, thresh1, thresh2=None, **kwargs):
        """Compute correlation function (convenience method).

        This creates a new halomod model each time. For MCMC fitting,
        use initialize_correlation_model() + update_correlation_model() instead.

        Parameters are the same as initialize_correlation_model().

        Returns
        -------
        result : dict
            Dictionary with 'model', 'separation', 'correlation', 'type'
        """
        return self.initialize_correlation_model(thresh1, thresh2, **kwargs)


class UVObservables(Observables):
    """Observables for a :class:`UVHODModel`.

    Thresholds are UV absolute magnitudes; a sample "passes" when it is
    brighter (more negative) than the threshold. Keeps the ``MUV``-named
    API, including the ``MUV_thresh2=0`` convention for "no bright limit".

    Examples
    --------
    >>> obs = UVObservables(UVHODModel(z=6.0))
    >>> phi = obs.luminosity_function(np.linspace(-22, -16, 20))
    >>> n = obs.number_density(-19.0)
    """

    _model_class = UVHODModel

    @staticmethod
    def _bright_limit(MUV_thresh2):
        """Map the UV ``MUV_thresh2 = 0`` "no bright limit" convention to None."""
        if MUV_thresh2 is None or MUV_thresh2 >= 0:
            return None
        return MUV_thresh2

    def luminosity_function(self, MUV_arr, dMUV=0.05, **params):
        """UV luminosity function Φ(M_UV) in Mpc^-3 mag^-1.

        Parameters
        ----------
        MUV_arr : array_like
            UV absolute magnitudes
        dMUV : float, optional
            Magnitude bin width. Default is 0.05.
        **params : dict, optional
            HOD parameter overrides (eps0, Mc, a, b, sigma_UV, Mcut,
            Msat, asat, add_dust). Applied to the model before computing.

        Examples
        --------
        >>> obs.luminosity_function(MUV)                          # current params
        >>> obs.luminosity_function(MUV, eps0=0.3, sigma_UV=0.5)  # override
        """
        return self.abundance_function(MUV_arr, dMUV, **params)

    def galaxy_bias(self, MUV_arr, dMUV=0.05, **params):
        """Galaxy bias in UV magnitude bins of width ``dMUV``."""
        return super().galaxy_bias(MUV_arr, dMUV, **params)

    def mean_halo_mass(self, MUV_thresh, **params):
        """log10 mean halo mass of galaxies brighter than ``MUV_thresh``."""
        return super().mean_halo_mass(MUV_thresh, **params)

    def mean_bias(self, MUV_thresh, **params):
        """Mean galaxy bias of galaxies brighter than ``MUV_thresh``."""
        return super().mean_bias(MUV_thresh, **params)

    def number_density(self, MUV_thresh, **params):
        """Number density (Mpc^-3) of galaxies brighter than ``MUV_thresh``."""
        return super().number_density(MUV_thresh, **params)

    def initialize_correlation_model(self, MUV_thresh1, MUV_thresh2=0, **kwargs):
        """Initialize the halomod correlation model for a UV sample.

        Parameters
        ----------
        MUV_thresh1 : float
            Faint magnitude limit of the sample.
        MUV_thresh2 : float, optional
            Bright magnitude limit for a magnitude bin (more negative than
            ``MUV_thresh1``). 0 (default) gives a threshold sample.
        **kwargs
            See :meth:`Observables.initialize_correlation_model`.
        """
        return super().initialize_correlation_model(
            MUV_thresh1, self._bright_limit(MUV_thresh2), **kwargs)

    def update_correlation_model(self, z_center=None, p_of_z=None,
                                 MUV_thresh1=None, MUV_thresh2=None, **params):
        """Update the halomod correlation model; see the base class.

        ``MUV_thresh1``/``MUV_thresh2`` left as None are unchanged;
        ``MUV_thresh2=0`` switches to a threshold sample.
        """
        return super().update_correlation_model(
            z_center=z_center, p_of_z=p_of_z,
            thresh1=_UNSET if MUV_thresh1 is None else MUV_thresh1,
            thresh2=_UNSET if MUV_thresh2 is None else self._bright_limit(MUV_thresh2),
            **params)

    def compute_correlation_function(self, MUV_thresh1, MUV_thresh2=0, **kwargs):
        """One-off correlation function; see :meth:`initialize_correlation_model`."""
        return self.initialize_correlation_model(MUV_thresh1, MUV_thresh2, **kwargs)


class MstarObservables(Observables):
    """Observables for a :class:`MstarHODModel`.

    Thresholds are log10(Mstar / M_sun); a sample "passes" when it is more
    massive than the threshold. For correlation functions, ``thresh2`` is
    the upper mass limit of a bin (``None`` for a threshold sample).

    Examples
    --------
    >>> obs = MstarObservables(MstarHODModel(z=6.0))
    >>> phi = obs.stellar_mass_function(np.linspace(8, 11, 13))
    >>> n = obs.number_density(9.0)        # log10(Mstar/M_sun) > 9
    """

    _model_class = MstarHODModel

    def stellar_mass_function(self, log10_Mstar_arr, dlog10_Mstar=0.05, **params):
        """Stellar mass function Φ(log10 Mstar) in Mpc^-3 dex^-1.

        Parameters
        ----------
        log10_Mstar_arr : array_like
            log10(Mstar / M_sun)
        dlog10_Mstar : float, optional
            Bin width in dex. Default is 0.05.
        **params : dict, optional
            HOD parameter overrides (eps0, Mc, a, b, sigma_Mstar, Mcut,
            Msat, asat). Applied to the model before computing.
        """
        return self.abundance_function(log10_Mstar_arr, dlog10_Mstar, **params)


class HalomodHOD(halomod.hod.HODPoisson if HALOMOD_AVAILABLE else object):
    """Halomod-compatible HOD wrapping a :class:`BaseHODModel`.

    Base class for :class:`HalomodUVHOD` and :class:`HalomodMstarHOD`, which
    set ``_model_class``. Bridges our HOD models with halomod's expectations.

    IMPORTANT: Handles unit conversions between halomod (M_sun/h) and the
    HOD models (M_sun).

    Parameters
    ----------
    z : float
        Redshift
    thresh1 : float, optional
        Sample threshold on the model's selection variable. On its own it
        defines a threshold sample.
    thresh2 : float, optional
        Second bound for a binned sample, on the *selected* side of
        ``thresh1`` (brighter for UV, more massive for Mstar). Occupations
        are formed as ``N(thresh1) - N(thresh2)``, so the reverse order
        would give negative occupations. ``None`` for a threshold sample.
    h : float, optional
        Hubble parameter H0/100
    M_min : float, optional
        Stored for halomod; not used by the occupations.
    **kwargs
        HOD model parameters (any of ``_model_class.param_names()``; None
        means the model default). Anything else is passed to halomod.
    """

    _model_class = None

    def __init__(self, z=None, thresh1=None, thresh2=None, h=None, M_min=None,
                 **kwargs):
        """Initialize halomod HOD wrapper."""
        if not HALOMOD_AVAILABLE:
            raise ImportError("halomod package required")

        model_names = self._model_class.param_names()
        model_params = {k: kwargs.pop(k) for k in list(kwargs) if k in model_names}
        model_params = {k: v for k, v in model_params.items() if v is not None}

        super().__init__(**kwargs)

        from .config import DEFAULT_COSMO_PARAMS

        self.z = z if z is not None else self._defaults['z']
        self.thresh1 = thresh1 if thresh1 is not None else self._defaults['thresh1']
        self.thresh2 = thresh2

        # A bin is formed as N(thresh1) - N(thresh2), so thresh2 must lie on
        # the selected side of thresh1. The reverse order yields negative
        # occupations, and hence a negative mean_tracer_den and NaN in
        # sigma_central, which the n_bar^2 normalisation then hides in the
        # auto-correlation. Fail loudly instead.
        s = self._model_class._selection_sign
        if self.thresh2 is not None and not s * (self.thresh2 - self.thresh1) > 0:
            side = "more negative (brighter)" if s < 0 else "larger (more massive)"
            raise ValueError(
                f"thresh2 ({self.thresh2}) must be {side} than "
                f"thresh1 ({self.thresh1}): thresh1 is the outer limit of the "
                "sample and thresh2 the inner limit of the bin."
            )

        self.M_min = M_min if M_min is not None else self._defaults['M_min']
        self.h = h if h is not None else DEFAULT_COSMO_PARAMS['H0'] / 100.0

        model_params.pop('z', None)
        self.hod_model = self._model_class(z=self.z, **model_params)

    def _occupation(self, N, M):
        """Occupation ``N`` for the sample, with halomod's M_sun/h masses.

        UNIT CONVERSION: M [M_sun] = M [M_sun/h] / h
        """
        M_physical = M / self.h
        if self.thresh2 is not None:
            return N(M_physical, self.thresh1) - N(M_physical, self.thresh2)
        return N(M_physical, self.thresh1)

    def _central_occupation(self, M):
        """Central occupation function for halomod."""
        return self._occupation(self.hod_model.Ncen, M)

    def _satellite_occupation(self, M):
        """Satellite occupation function for halomod."""
        return self._occupation(self.hod_model.Nsat, M)


def _halomod_defaults(model_class, thresh1):
    return {"z": 5.0, "thresh1": thresh1, "thresh2": None, "M_min": 8.0,
            **{n: None for n in model_class.param_names() if n != 'z'}}


class HalomodUVHOD(HalomodHOD):
    """Halomod wrapper for :class:`UVHODModel` (thresholds in M_UV).

    Examples
    --------
    >>> hod = HalomodUVHOD(z=5.0, thresh1=-19.1, h=0.7)
    """

    _model_class = UVHODModel
    _defaults = _halomod_defaults(UVHODModel, thresh1=-16.0)


class HalomodMstarHOD(HalomodHOD):
    """Halomod wrapper for :class:`MstarHODModel` (thresholds in log10 Mstar).

    Examples
    --------
    >>> hod = HalomodMstarHOD(z=5.0, thresh1=9.0, h=0.7)
    """

    _model_class = MstarHODModel
    _defaults = _halomod_defaults(MstarHODModel, thresh1=9.0)


UVObservables._halomod_hod_class = HalomodUVHOD
MstarObservables._halomod_hod_class = HalomodMstarHOD
