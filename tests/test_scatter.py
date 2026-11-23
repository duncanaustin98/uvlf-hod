"""Tests for halo mass- and redshift-dependent UV scatter sigma_UV(Mh, z)."""

import numpy as np
import pytest

from halogal import HODModel, sigma_UV_linear
from halogal.config import SIGMA_UV_MIN
from halogal.model import Observables

MH = np.logspace(9, 13, 50)
MUV = np.linspace(-22, -17, 6)


def lin_model(z=6.0, **params):
    p = dict(sigma_UV0=0.69, d_sigma_UV_dlogMh=0.0, d_sigma_UV_dz=0.0)
    p.update(params)
    return HODModel(z=z, sigma_UV=sigma_UV_linear, sigma_UV_params=p)


class TestBackwardCompatibility:
    def test_float_scatter_unchanged(self):
        model = HODModel(z=6.0, sigma_UV=0.69)
        assert model.scatter(MH) == 0.69
        assert not model.sigma_UV_is_callable

    def test_constant_callable_matches_float(self):
        ref = HODModel(z=6.0, sigma_UV=0.69)
        const = HODModel(z=6.0, sigma_UV=lambda Mh, z: 0.69)
        np.testing.assert_allclose(const.Ncen(MH, -19.0), ref.Ncen(MH, -19.0))
        np.testing.assert_allclose(
            Observables(const).luminosity_function(MUV),
            Observables(ref).luminosity_function(MUV),
        )

    def test_linear_zero_slopes_matches_float(self):
        ref = HODModel(z=6.0, sigma_UV=0.69)
        np.testing.assert_allclose(lin_model().Ngal(MH, -19.0), ref.Ngal(MH, -19.0))


class TestDependence:
    def test_mass_dependence(self):
        model = lin_model(d_sigma_UV_dlogMh=-0.1)
        sig = model.scatter(MH)
        assert sig.shape == MH.shape
        assert sig[0] > sig[-1]
        np.testing.assert_allclose(model.scatter(1e11), 0.69)

    def test_mass_dependence_changes_lf(self):
        phi0 = Observables(lin_model()).luminosity_function(MUV)
        phi1 = Observables(lin_model(d_sigma_UV_dlogMh=-0.2)).luminosity_function(MUV)
        assert not np.allclose(phi0, phi1)

    def test_redshift_dependence(self):
        model = lin_model(z=5.0, d_sigma_UV_dz=-0.03)
        s5 = model.scatter(1e11)
        model.update_parameters(z=8.0)
        np.testing.assert_allclose(model.scatter(1e11) - s5, -0.09)

    def test_floor(self):
        model = lin_model(sigma_UV0=0.1, d_sigma_UV_dlogMh=-1.0)
        sig = model.scatter(MH)
        assert np.all(sig >= SIGMA_UV_MIN)
        assert np.isclose(sig.min(), SIGMA_UV_MIN)
        assert np.all(np.isfinite(model.Ncen(MH, -19.0)))


class TestParameterUpdates:
    def test_update_individual_param(self):
        model = lin_model()
        model.update_parameters(d_sigma_UV_dlogMh=-0.3, eps0=0.2)
        assert model.sigma_UV_params['d_sigma_UV_dlogMh'] == -0.3
        assert model.eps0 == 0.2

    def test_update_params_dict_merges(self):
        model = lin_model()
        model.update_parameters(sigma_UV_params={'sigma_UV0': 0.5})
        assert model.sigma_UV_params['sigma_UV0'] == 0.5
        assert 'd_sigma_UV_dz' in model.sigma_UV_params

    def test_replace_with_float_clears_params(self):
        model = lin_model()
        model.update_parameters(sigma_UV=0.5)
        assert model.sigma_UV_params == {}
        assert model.scatter(MH) == 0.5

    def test_observables_routing(self):
        obs = Observables(lin_model())
        obs.luminosity_function(MUV, d_sigma_UV_dlogMh=-0.15)
        assert obs.hod_model.sigma_UV_params['d_sigma_UV_dlogMh'] == -0.15
        obs.number_density(-19.0, sigma_UV0=0.4)
        assert obs.hod_model.sigma_UV_params['sigma_UV0'] == 0.4

    def test_param_names(self):
        assert 'd_sigma_UV_dlogMh' in lin_model().param_names
        assert 'd_sigma_UV_dlogMh' not in HODModel(z=6.0).param_names


class TestValidationAndRepr:
    def test_params_with_float_raises(self):
        with pytest.raises(ValueError):
            HODModel(z=6.0, sigma_UV=0.7, sigma_UV_params={'x': 1.0})

    def test_reserved_name_clash_raises(self):
        with pytest.raises(ValueError):
            HODModel(z=6.0, sigma_UV=lambda Mh, z, eps0: 0.7,
                     sigma_UV_params={'eps0': 1.0})

    def test_repr_str(self):
        model = lin_model(d_sigma_UV_dlogMh=-0.1)
        assert 'sigma_UV_linear' in repr(model)
        assert 'd_sigma_UV_dlogMh=-0.1' in str(model)
        assert 'sigma_UV=0.69' in repr(HODModel(z=6.0, sigma_UV=0.69))
