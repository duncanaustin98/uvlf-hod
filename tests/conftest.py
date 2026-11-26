"""Shared pytest fixtures."""

import numpy as np
import pytest

from halogal.dust import LinearBetaMUV, NoDust


@pytest.fixture
def muv_grid():
    """Observed UV magnitudes spanning the bright to the faint end."""
    return np.linspace(-24.0, -14.0, 41)


@pytest.fixture
def Mh_grid():
    """Halo masses in M_sun."""
    return np.logspace(9.5, 12.5, 25)


@pytest.fixture
def epochs_beta():
    """EPOCHS Paper-1 beta-M_UV relation (linear in z about z = 7)."""
    return LinearBetaMUV.epochs_paper1()


@pytest.fixture
def simple_linear_beta():
    """Redshift-independent linear beta-M_UV relation with scatter."""
    return LinearBetaMUV(beta_M0=-2.0, dbeta_dMUV=-0.1, sigma_beta=0.3,
                         M0=-19.0)


@pytest.fixture
def no_dust():
    return NoDust()
