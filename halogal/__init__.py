"""
uvlf-hod: UV Luminosity Functions with Halo Occupation Distribution models.

This package provides tools for modeling high-redshift galaxy populations
using UV-halo mass relations and HOD models.

Main components:
    - UV-Halo Mass Relation (UVHMR)
    - Halo Occupation Distribution (HOD) models
    - UV Luminosity Function calculations
    - Galaxy bias calculations
    - Dust attenuation models
"""

from .__version__ import __version__

# Main model classes (new unified interface)
from .model import (
    HMRModel,
    UVHMRModel,
    SHMRModel,
    HMRModel_FFB,
    UVHMRModel_FFB,
    SHMRModel_FFB,
    BaseHODModel,
    UVHODModel,
    MstarHODModel,
    UVHODModel_FFB,
    MstarHODModel_FFB,
    HODModel,
    Observables,
    UVObservables,
    MstarObservables,
    HalomodHOD,
    HalomodUVHOD,
    HalomodMstarHOD,
    HalomodUVHOD_FFB,
    HalomodMstarHOD_FFB,
)

# Cosmology
from .cosmology import (
    HaloMassFunction,
    get_halo_mass_function,
    get_halo_bias,
)

# Luminosity
from .luminosity import (
    beta_color,
    dust_attenuation,
    MUV_from_SFR,
    SFR_from_MUV,
)

# Dust
from .dust import (
    DustModel,
    NoDust,
    BetaDust,
    DEFAULT_DUST,
    BetaMUVRelation,
    LinearBetaMUV,
    Bouwens14BetaMUV,
    FunctionBetaMUV,
    ConstantZ,
    LinearZ,
    PowerLawZ,
    TabulatedZ,
    as_z_function,
)

# Configuration
from .config import (
    CosmologyConfig,
    default_cosmology,
    DEFAULT_HOD_PARAMS,
    DEFAULT_REDSHIFT_EVOLUTION,
)

# Cross-correlations between two magnitude-selected samples
from .crosscorr import (
    Selection,
    CrossCorrelation,
)

# Models
from .models.parametrization import (
    RedshiftParametrization,
    eps0_fz,
    Mc_fz,
    a_fz,
    b_fz,
    sigma_UV_fz,
    Mcut_fz,
    Msat_fz,
    asat_fz,
)

# Inference (MCMC fitting). Sampler/plotting backends are imported lazily
# inside the relevant methods, so this import only needs numpy/scipy.
from .inference import (
    Parameter,
    ParameterSet,
    Uniform,
    Normal,
    LogUniform,
    ModelParametrization,
    Dataset,
    GaussianLikelihood,
    EmceeSampler,
    DynestySampler,
    InferenceResult,
)




__all__ = [
    # Version
    '__version__',
    # Cross-correlations
    'Selection',
    'CrossCorrelation',
    
    # Main model classes
    'HMRModel',
    'UVHMRModel',
    'SHMRModel',
    'HMRModel_FFB',
    'UVHMRModel_FFB',
    'SHMRModel_FFB',
    'BaseHODModel',
    'UVHODModel',
    'MstarHODModel',
    'UVHODModel_FFB',
    'MstarHODModel_FFB',
    'HODModel',
    'Observables',
    'UVObservables',
    'MstarObservables',
    'HalomodHOD',
    'HalomodUVHOD',
    'HalomodMstarHOD',
    'HalomodUVHOD_FFB',
    'HalomodMstarHOD_FFB',

    # Cosmology
    'HaloMassFunction',
    'get_halo_mass_function',
    'get_halo_bias',
    
    # Luminosity
    'beta_color',
    'dust_attenuation',
    'MUV_from_SFR',
    'SFR_from_MUV',

    # Dust
    'DustModel',
    'NoDust',
    'BetaDust',
    'DEFAULT_DUST',
    'BetaMUVRelation',
    'LinearBetaMUV',
    'Bouwens14BetaMUV',
    'FunctionBetaMUV',
    'ConstantZ',
    'LinearZ',
    'PowerLawZ',
    'TabulatedZ',
    'as_z_function',
    
    # Config
    'CosmologyConfig',
    'default_cosmology',
    'DEFAULT_HOD_PARAMS',
    'DEFAULT_REDSHIFT_EVOLUTION',
    
    # Redshift parametrization
    'RedshiftParametrization',
    'eps0_fz',
    'Mc_fz',
    'a_fz',
    'b_fz',
    'sigma_UV_fz',
    'Mcut_fz',
    'Msat_fz',
    'asat_fz',

    # Inference
    'Parameter',
    'ParameterSet',
    'Uniform',
    'Normal',
    'LogUniform',
    'ModelParametrization',
    'Dataset',
    'GaussianLikelihood',
    'EmceeSampler',
    'DynestySampler',
    'InferenceResult',
]


# Package metadata
__author__ = "Marko Shuntov"
__email__ = "marko.shuntov@nbi.ku.dk"
__url__ = "https://github.com/mshuntov/uvlf-hod"
__description__ = "UV Luminosity Functions with HOD models for high-redshift galaxies"