"""Tests for the forward model: UVHMR, SHMR, HOD occupations and observables.

Fast unit tests only (no halomod model is built here).
"""

import numpy as np
import pytest

from halogal import (HODModel, MstarHODModel, MstarObservables, SHMRModel,
                     UVHMRModel, UVHODModel, UVObservables)
from halogal.config import BARYON_FRACTION, DEFAULT_HOD_PARAMS
from halogal.cosmology import get_halo_bias, get_halo_mass_function
from halogal.luminosity import MUV_from_SFR, SFR_from_MUV
from halogal.model import Observables


# --------------------------------------------------------------------------
# UV-halo mass relation
# --------------------------------------------------------------------------
class TestUVHMR:
    def test_defaults_come_from_config(self):
        m = UVHMRModel(z=6.0)
        assert m.eps0 == DEFAULT_HOD_PARAMS["eps0"]
        assert m.Mc == pytest.approx(10 ** DEFAULT_HOD_PARAMS["logMc"])

    def test_sfe_double_power_law(self):
        m = UVHMRModel(z=6.0, eps0=0.1, Mc=1e12, a=0.6, b=0.35)
        # 2 eps0 / (x^-a + x^b) equals eps0 at x = 1
        assert m.star_formation_efficiency(1e12)[0] == pytest.approx(0.1)
        low = m.star_formation_efficiency(np.array([1e9, 1e10]))
        assert low[1] / low[0] == pytest.approx(10 ** 0.6, rel=2e-2)
        high = m.star_formation_efficiency(np.array([1e14, 1e15]))
        assert high[1] / high[0] == pytest.approx(10 ** -0.35, rel=2e-2)

    def test_accretion_rate_redshift_scaling(self):
        lo, hi = UVHMRModel(z=4.0), UVHMRModel(z=8.0)
        ratio = hi.halo_accretion_rate(1e11) / lo.halo_accretion_rate(1e11)
        assert ratio[0] == pytest.approx((9.0 / 5.0) ** 2.5)

    def test_sfr_positive_and_monotonic(self):
        m = UVHMRModel(z=6.0)
        sfr = m.sfr(np.logspace(9, 13, 50))
        assert np.all(sfr > 0) and np.all(np.diff(sfr) > 0)

    def test_more_massive_haloes_are_brighter(self):
        m = UVHMRModel(z=6.0)
        muv = m.MUV(np.logspace(9.5, 12.5, 30))
        assert np.all(np.diff(muv) < 0)

    def test_inverse_relation_roundtrip(self):
        m = UVHMRModel(z=6.0)
        Mh = np.array([1e10, 1e11, 1e12])
        # MUV(Mh) is only converged to ~0.02 mag by the dust iteration, which
        # limits the inversion to a few per cent in mass.
        assert m.Mhalo(m.MUV(Mh)) == pytest.approx(Mh, rel=0.1)

    def test_update_parameters(self):
        m = UVHMRModel(z=6.0)
        before = m.MUV(1e11)[0]
        m.update_parameters(eps0=2 * m.eps0)
        assert m.MUV(1e11)[0] < before          # brighter

    def test_sfr_muv_conversion_roundtrip_without_dust(self):
        sfr = np.array([0.1, 1.0, 10.0])
        muv = MUV_from_SFR(sfr, 6.0, add_dust=False)
        assert SFR_from_MUV(muv, 6.0, remove_dust=False) == pytest.approx(sfr)

    def test_dust_only_dims(self):
        sfr = np.array([1.0, 30.0, 300.0])
        dusty = MUV_from_SFR(sfr, 5.0, add_dust=True)
        clean = MUV_from_SFR(sfr, 5.0, add_dust=False)
        assert np.all(dusty >= clean - 1e-9)


# --------------------------------------------------------------------------
# Stellar-halo mass relation
# --------------------------------------------------------------------------
class TestSHMR:
    def test_defaults_come_from_config(self):
        m = SHMRModel(z=6.0)
        assert m.eps0 == DEFAULT_HOD_PARAMS["eps0"]
        assert m.Mc == pytest.approx(10 ** DEFAULT_HOD_PARAMS["logMc"])

    def test_sfe_double_power_law(self):
        m = SHMRModel(z=6.0, eps0=0.1, Mc=1e12, a=0.6, b=0.35)
        # 2 eps0 / (x^-a + x^b) equals eps0 at x = 1
        assert m.star_formation_efficiency(1e12)[0] == pytest.approx(0.1)
        low = m.star_formation_efficiency(np.array([1e9, 1e10]))
        assert low[1] / low[0] == pytest.approx(10 ** 0.6, rel=2e-2)
        high = m.star_formation_efficiency(np.array([1e14, 1e15]))
        assert high[1] / high[0] == pytest.approx(10 ** -0.35, rel=2e-2)

    def test_mstar_is_baryon_fraction_times_sfe_times_mh(self):
        m = SHMRModel(z=6.0)
        Mh = np.logspace(9, 13, 5)
        expected = BARYON_FRACTION * m.star_formation_efficiency(Mh) * Mh
        assert m.Mstar(Mh) == pytest.approx(expected)

    def test_mstar_aliases_agree(self):
        m = SHMRModel(z=6.0)
        Mh = np.logspace(9, 13, 5)
        assert m(Mh) == pytest.approx(m.Mstar(Mh))
        assert m.property(Mh) == pytest.approx(m.Mstar(Mh))

    def test_mstar_positive_and_monotonic(self):
        m = SHMRModel(z=6.0)
        mstar = m.Mstar(np.logspace(9, 13, 50))
        assert np.all(mstar > 0) and np.all(np.diff(mstar) > 0)

    def test_mstar_below_baryon_budget_at_low_mass(self):
        # Far below Mc the efficiency falls as (Mh/Mc)^a, so Mstar << fb Mh.
        m = SHMRModel(z=6.0)
        Mh = np.logspace(8, 10, 5)
        assert np.all(m.Mstar(Mh) < 0.1 * BARYON_FRACTION * Mh)

    def test_inverse_relation_roundtrip(self):
        m = SHMRModel(z=6.0)
        Mh = np.array([1e10, 1e11, 1e12])
        # Analytic forward relation, so only the interpolation grid limits it.
        assert m.Mhalo(m.Mstar(Mh)) == pytest.approx(Mh, rel=1e-3)

    def test_update_parameters(self):
        m = SHMRModel(z=6.0)
        before = m.Mstar(1e11)[0]
        m.update_parameters(eps0=2 * m.eps0)
        assert m.Mstar(1e11)[0] == pytest.approx(2 * before)   # linear in eps0

    def test_redshift_does_not_enter(self):
        # Unlike the UVHMR (accretion rate), the SHMR has no explicit z term.
        Mh = np.logspace(9, 13, 5)
        assert SHMRModel(z=4.0).Mstar(Mh) == pytest.approx(
            SHMRModel(z=8.0).Mstar(Mh))


# --------------------------------------------------------------------------
# Occupation functions
# --------------------------------------------------------------------------
class TestUVHOD:
    def test_hodmodel_alias(self):
        assert HODModel is UVHODModel

    def test_sigma_UV_default_and_update(self):
        m = UVHODModel(z=6.0)
        assert m.sigma_UV == DEFAULT_HOD_PARAMS["sigma_UV"]
        m.update_parameters(sigma_UV=0.3)
        assert m.sigma_UV == 0.3

    def test_ncen_is_a_probability_and_monotonic(self):
        m = UVHODModel(z=6.0)
        Mh = np.logspace(9, 14, 80)
        ncen = m.Ncen(Mh, -19.0)
        assert np.all((ncen >= 0) & (ncen <= 1))
        assert np.all(np.diff(ncen) >= -1e-12)
        assert ncen[0] < 1e-3 and ncen[-1] > 0.99

    def test_ncen_is_half_at_threshold(self):
        m = UVHODModel(z=6.0)
        assert m.Ncen(1e11, m.MUV(1e11)[0])[0] == pytest.approx(0.5)

    def test_brighter_threshold_selects_fewer(self):
        m = UVHODModel(z=6.0)
        Mh = np.logspace(10, 13, 40)
        assert np.all(m.Ncen(Mh, -21.0) <= m.Ncen(Mh, -19.0))

    def test_nsat_zero_below_mcut(self):
        m = UVHODModel(z=6.0, Mcut=1e11)
        Mh = np.array([1e9, 5e10, 1e11, 1e12, 1e13])
        nsat = m.Nsat(Mh, -18.0)
        assert np.all(nsat[:3] == 0) and np.all(nsat[3:] > 0)

    def test_integer_masses_are_not_truncated(self):
        m = UVHODModel(z=6.0)
        as_int = m.Nsat(np.array([10 ** 12, 10 ** 13]), -18.0)
        as_float = m.Nsat(np.array([1e12, 1e13]), -18.0)
        assert as_int == pytest.approx(as_float)

    def test_ngal_is_sum(self):
        m = UVHODModel(z=6.0)
        Mh = np.logspace(10, 13, 20)
        assert m.Ngal(Mh, -19.0) == pytest.approx(
            m.Ncen(Mh, -19.0) + m.Nsat(Mh, -19.0))


class TestMstarHOD:
    # Thresholds are log10(Mstar / M_sun).

    def test_sigma_Mstar_default_and_update(self):
        m = MstarHODModel(z=6.0)
        assert m.sigma_Mstar == DEFAULT_HOD_PARAMS["sigma_Mstar"]
        m.update_parameters(sigma_Mstar=0.2)
        assert m.sigma_Mstar == 0.2

    def test_ncen_is_a_probability_and_monotonic(self):
        m = MstarHODModel(z=6.0)
        Mh = np.logspace(9, 14, 80)
        ncen = m.Ncen(Mh, 9.0)
        assert np.all((ncen >= 0) & (ncen <= 1))
        assert np.all(np.diff(ncen) >= -1e-12)
        assert ncen[0] < 1e-3 and ncen[-1] > 0.99

    def test_ncen_is_half_at_threshold(self):
        m = MstarHODModel(z=6.0)
        thresh = np.log10(m.Mstar(1e11)[0])
        assert m.Ncen(1e11, thresh)[0] == pytest.approx(0.5)

    def test_more_massive_threshold_selects_fewer(self):
        m = MstarHODModel(z=6.0)
        Mh = np.logspace(10, 13, 40)
        assert np.all(m.Ncen(Mh, 10.0) <= m.Ncen(Mh, 9.0))

    def test_nsat_zero_below_mcut(self):
        m = MstarHODModel(z=6.0, Mcut=1e11)
        Mh = np.array([1e9, 5e10, 1e11, 1e12, 1e13])
        nsat = m.Nsat(Mh, 8.0)
        assert np.all(nsat[:3] == 0) and np.all(nsat[3:] > 0)

    def test_integer_masses_are_not_truncated(self):
        m = MstarHODModel(z=6.0)
        as_int = m.Nsat(np.array([10 ** 12, 10 ** 13]), 8.0)
        as_float = m.Nsat(np.array([1e12, 1e13]), 8.0)
        assert as_int == pytest.approx(as_float)

    def test_ngal_is_sum(self):
        m = MstarHODModel(z=6.0)
        Mh = np.logspace(10, 13, 20)
        assert m.Ngal(Mh, 9.0) == pytest.approx(
            m.Ncen(Mh, 9.0) + m.Nsat(Mh, 9.0))


# --------------------------------------------------------------------------
# Halo mass function / bias conventions
# --------------------------------------------------------------------------
class TestCosmology:
    def test_hmf_shape_and_monotonic(self):
        log10m, hmf = get_halo_mass_function(6.0, M_min=8, M_max=13,
                                             num_points=64)
        assert log10m.shape == hmf.shape == (64,)
        assert np.all(hmf > 0) and np.all(np.diff(hmf) < 0)

    def test_hmf_drops_with_redshift_at_high_mass(self):
        _, lo = get_halo_mass_function(5.0, M_min=11, M_max=12, num_points=8)
        _, hi = get_halo_mass_function(8.0, M_min=11, M_max=12, num_points=8)
        assert np.all(hi < lo)

    def test_bias_increases_with_mass(self):
        b = get_halo_bias(np.logspace(10, 13, 10), 6.0)
        assert np.all(np.diff(b) > 0) and b[-1] > 1


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def obs():
    o = Observables(HODModel(z=6.0))
    # coarse grid keeps these tests fast; accuracy is checked below
    o._hmf_cache[6.0] = get_halo_mass_function(
        6.0, M_min=6, M_max=15, num_points=256)
    return o


class TestObservables:

    def test_uvlf_positive_and_falls_to_bright_end(self, obs):
        MUV = np.linspace(-22, -17, 11)
        phi = obs.luminosity_function(MUV)
        assert np.all(phi > 0) and np.all(np.diff(phi) > 0)

    def test_integer_magnitudes_give_nonzero_uvlf(self, obs):
        phi_int = obs.luminosity_function(np.arange(-21, -17))
        phi_flt = obs.luminosity_function(np.arange(-21.0, -17.0))
        assert np.all(phi_int > 0) and phi_int == pytest.approx(phi_flt)

    def test_uvlf_integrates_to_number_density(self, obs):
        # n(<M) must equal the integral of Phi over magnitudes brighter than M
        MUV = np.linspace(-26.0, -19.0, 281)
        phi = obs.luminosity_function(MUV)
        n_int = np.trapz(phi, MUV)
        assert n_int == pytest.approx(obs.number_density(-19.0), rel=2e-2)

    def test_inline_parameter_override_persists(self, obs):
        MUV = np.array([-20.0])
        base = obs.luminosity_function(MUV, eps0=0.19)[0]
        boosted = obs.luminosity_function(MUV, eps0=0.30)[0]
        assert boosted > base
        assert obs.hod_model.eps0 == 0.30
        obs.luminosity_function(MUV, eps0=0.19)      # restore for other tests

    def test_unknown_parameter_raises(self, obs):
        with pytest.raises(TypeError, match="unrecognised"):
            obs.luminosity_function(np.array([-20.0]), sigmaUV=0.5)

    def test_bias_grows_with_luminosity(self, obs):
        bg = obs.galaxy_bias(np.array([-21.5, -20.0, -18.5]))
        assert bg[0] > bg[1] > bg[2] > 1

    def test_mean_halo_mass_and_bias_grow_with_luminosity(self, obs):
        assert obs.mean_halo_mass(-21.0) > obs.mean_halo_mass(-19.0)
        assert obs.mean_bias(-21.0) > obs.mean_bias(-19.0)

    def test_coarse_hmf_grid_is_accurate(self):
        MUV = np.linspace(-22, -17, 6)
        fine = Observables(HODModel(z=6.0)).luminosity_function(MUV)
        coarse = Observables(HODModel(z=6.0))
        coarse._hmf_cache[6.0] = get_halo_mass_function(
            6.0, M_min=6, M_max=15, num_points=256)
        assert coarse.luminosity_function(MUV) == pytest.approx(fine, rel=5e-3)


class TestObservablesDispatch:
    def test_base_dispatches_on_model_type(self):
        assert type(Observables(UVHODModel(z=6.0))) is UVObservables
        assert type(Observables(MstarHODModel(z=6.0))) is MstarObservables

    def test_subclass_rejects_wrong_model(self):
        with pytest.raises(TypeError):
            UVObservables(MstarHODModel(z=6.0))
        with pytest.raises(TypeError):
            MstarObservables(UVHODModel(z=6.0))

    def test_base_rejects_unknown_model(self):
        with pytest.raises(TypeError):
            Observables(UVHMRModel(z=6.0))

    def test_parameters_are_model_specific(self):
        uv = Observables(UVHODModel(z=6.0))
        with pytest.raises(TypeError, match="unrecognised"):
            uv.number_density(-19.0, sigma_Mstar=0.3)
        ms = Observables(MstarHODModel(z=6.0))
        with pytest.raises(TypeError, match="unrecognised"):
            ms.number_density(9.0, sigma_UV=0.5)

    def test_uv_lf_equals_generic_abundance_function(self, obs):
        MUV = np.array([-21.0, -19.0])
        assert obs.luminosity_function(MUV) == pytest.approx(
            obs.abundance_function(MUV))


@pytest.fixture(scope="module")
def mobs():
    o = Observables(MstarHODModel(z=6.0))
    # coarse grid keeps these tests fast
    o._hmf_cache[6.0] = get_halo_mass_function(
        6.0, M_min=6, M_max=15, num_points=256)
    return o


class TestMstarObservables:
    # Thresholds are log10(Mstar / M_sun).

    def test_smf_positive_and_falls_to_high_mass(self, mobs):
        logM = np.linspace(8.0, 10.5, 11)
        phi = mobs.stellar_mass_function(logM)
        assert np.all(phi > 0) and np.all(np.diff(phi) < 0)

    def test_integer_masses_give_nonzero_smf(self, mobs):
        phi_int = mobs.stellar_mass_function(np.arange(8, 11))
        phi_flt = mobs.stellar_mass_function(np.arange(8.0, 11.0))
        assert np.all(phi_int > 0) and phi_int == pytest.approx(phi_flt)

    def test_smf_integrates_to_number_density(self, mobs):
        # n(>M) must equal the integral of Phi over masses above M
        logM = np.linspace(9.0, 12.5, 141)
        phi = mobs.stellar_mass_function(logM)
        n_int = np.trapz(phi, logM)
        assert n_int == pytest.approx(mobs.number_density(9.0), rel=2e-2)

    def test_inline_parameter_override_persists(self, mobs):
        logM = np.array([9.5])
        base = mobs.stellar_mass_function(logM, eps0=0.19)[0]
        boosted = mobs.stellar_mass_function(logM, eps0=0.30)[0]
        assert boosted > base
        assert mobs.hod_model.eps0 == 0.30
        mobs.stellar_mass_function(logM, eps0=0.19)    # restore for other tests

    def test_unknown_parameter_raises(self, mobs):
        with pytest.raises(TypeError, match="unrecognised"):
            mobs.stellar_mass_function(np.array([9.5]), sigmaMstar=0.3)

    def test_bias_grows_with_stellar_mass(self, mobs):
        bg = mobs.galaxy_bias(np.array([10.5, 9.5, 8.5]))
        assert bg[0] > bg[1] > bg[2] > 1

    def test_mean_halo_mass_and_bias_grow_with_stellar_mass(self, mobs):
        assert mobs.mean_halo_mass(10.0) > mobs.mean_halo_mass(9.0)
        assert mobs.mean_bias(10.0) > mobs.mean_bias(9.0)


# --------------------------------------------------------------------------
# Selection-variable cache (M_UV(Mh) / log Mstar(Mh) computed once per
# parameter set and mass grid)
# --------------------------------------------------------------------------
class TestSelectionVariableCache:
    @staticmethod
    def _uncached(monkeypatch):
        from halogal.model import BaseHODModel
        monkeypatch.setattr(BaseHODModel, "_cached_selection_variable",
                            lambda self, Mh: self._selection_variable(Mh))

    @pytest.mark.parametrize("cls, x", [
        (UVHODModel, np.linspace(-23, -16, 8)),
        (MstarHODModel, np.linspace(7.5, 11, 6)),
    ])
    def test_abundance_identical_to_uncached(self, monkeypatch, cls, x):
        kw = dict(eps0=0.12, Mc=3e11, a=0.8)
        cached = Observables(cls(z=8.0)).abundance_function(x, 0.5, **kw)
        self._uncached(monkeypatch)
        ref = Observables(cls(z=8.0)).abundance_function(x, 0.5, **kw)
        assert np.array_equal(cached, ref)

    def test_ffb_model_identical_to_uncached(self, monkeypatch):
        from halogal import UVHODModel_FFB
        x = np.linspace(-23, -16, 8)
        kw = dict(eps_max_ffb=0.6, ffb_delta=0.2)
        cached = Observables(UVHODModel_FFB(z=10.0)).abundance_function(
            x, 0.5, **kw)
        self._uncached(monkeypatch)
        ref = Observables(UVHODModel_FFB(z=10.0)).abundance_function(
            x, 0.5, **kw)
        assert np.array_equal(cached, ref)

    def test_computed_once_per_parameter_set(self, monkeypatch):
        m = UVHODModel(z=7.0)
        calls = []
        orig = type(m)._selection_variable
        monkeypatch.setattr(type(m), "_selection_variable",
                            lambda self, Mh: calls.append(1) or orig(self, Mh))
        Mh = np.logspace(9, 13, 50)
        for thresh in (-21, -20, -19):
            m.Ngal(Mh, thresh)
        assert len(calls) == 1

    def test_invalidated_by_any_change(self):
        m = UVHODModel(z=7.0)
        Mh = np.logspace(9, 13, 50)
        base = m.Ncen(Mh, -19).copy()
        m.update_parameters(eps0=0.5)
        assert not np.array_equal(m.Ncen(Mh, -19), base)
        m.eps0 = DEFAULT_HOD_PARAMS["eps0"]          # direct attribute
        assert np.array_equal(m.Ncen(Mh, -19), base)
        m.add_dust = False                            # setting
        assert not np.array_equal(m.Ncen(Mh, -19), base)
        m.add_dust = True
        assert not np.array_equal(m.Ncen(np.logspace(9.5, 13, 50), -19),
                                  base)               # new mass grid

    def test_cached_value_is_read_only(self):
        m = UVHODModel(z=7.0)
        x = m._cached_selection_variable(np.logspace(9, 13, 10))
        with pytest.raises(ValueError):
            x[0] = 0.0
