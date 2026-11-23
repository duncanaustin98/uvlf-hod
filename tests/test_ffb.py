"""Tests for the feedback-free starburst (FFB) models.

Covers HMRModel_FFB (FFB threshold, fraction and SFE), the UV/stellar-mass HMR
and HOD FFB variants built from it by multiple inheritance, their observables
and the halomod wrapper selection.

Fast unit tests only (no halomod model is built here).
"""

import pickle

import numpy as np
import pytest

from halogal import (HalomodMstarHOD, HalomodMstarHOD_FFB, HalomodUVHOD,
                     HalomodUVHOD_FFB, HMRModel, HMRModel_FFB, MstarHODModel,
                     MstarHODModel_FFB, MstarObservables, SHMRModel,
                     SHMRModel_FFB, UVHMRModel, UVHMRModel_FFB, UVHODModel,
                     UVHODModel_FFB, UVObservables)
from halogal.config import BARYON_FRACTION, DEFAULT_HOD_PARAMS
from halogal.cosmology import get_halo_mass_function
from halogal.model import Observables

FFB_PARAMS = ("ffb_delta", "eps_max_ffb")
MH = np.logspace(9, 14, 81)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# FFB threshold, fraction and star formation efficiency
# --------------------------------------------------------------------------
class TestFFBThreshold:
    def test_defaults_come_from_config(self):
        m = UVHMRModel_FFB(z=10.0)
        assert m.ffb_delta == DEFAULT_HOD_PARAMS["ffb_delta"]
        assert m.eps_max_ffb == DEFAULT_HOD_PARAMS["eps_max_ffb"]

    def test_M_ffb_pivot_at_z9(self):
        # M_ffb = 10^10.8 ((1+z)/10)^-6.2 is exactly 10^10.8 at z = 9
        assert UVHMRModel_FFB(z=9.0).M_ffb == pytest.approx(10 ** 10.8)

    def test_M_ffb_redshift_scaling(self):
        lo, hi = UVHMRModel_FFB(z=6.0), UVHMRModel_FFB(z=12.0)
        assert hi.M_ffb / lo.M_ffb == pytest.approx((13.0 / 7.0) ** -6.2)

    def test_M_ffb_follows_redshift_update(self):
        m = UVHMRModel_FFB(z=10.0)
        m.update_parameters(z=9.0)
        assert m.M_ffb == pytest.approx(10 ** 10.8)


class TestFFBFraction:
    def test_fraction_is_a_probability(self):
        f = UVHMRModel_FFB(z=10.0).ffb_fraction(MH)
        assert np.all((f >= 0) & (f <= 1))

    def test_fraction_matches_product_of_logistics(self):
        m = UVHMRModel_FFB(z=10.0, ffb_delta=0.2)
        logMh = np.log10(MH)
        expected = (logistic((logMh - np.log10(m.M_ffb)) / 0.2)
                    * logistic((np.log10(m.Mc) - logMh) / 0.2))
        assert m.ffb_fraction(MH) == pytest.approx(expected, rel=1e-10, abs=1e-300)

    def test_fraction_is_half_at_M_ffb_when_window_is_wide(self):
        # with Mc far above M_ffb the turn-off sigmoid is ~1 at M_ffb
        m = UVHMRModel_FFB(z=10.0, Mc=1e15)
        assert m.ffb_fraction(m.M_ffb)[0] == pytest.approx(0.5, rel=1e-6)

    def test_fraction_switches_on_above_M_ffb_and_off_above_Mc(self):
        m = UVHMRModel_FFB(z=12.0, Mc=1e13)
        f = m.ffb_fraction(np.array([m.M_ffb / 100, 1e11, m.Mc * 100]))
        assert f[0] < 1e-5 and f[1] > 0.99 and f[2] < 1e-5

    def test_narrower_transition_is_sharper(self):
        Mh = np.array([1e10])          # below M_ffb ~ 10^10.54 at z = 10
        wide = UVHMRModel_FFB(z=10.0, ffb_delta=0.3).ffb_fraction(Mh)
        narrow = UVHMRModel_FFB(z=10.0, ffb_delta=0.1).ffb_fraction(Mh)
        assert narrow[0] < wide[0]

    def test_window_closes_when_M_ffb_exceeds_Mc(self):
        # at z = 4, M_ffb ~ 10^12.7 lies above the default Mc ~ 10^11.6
        m = UVHMRModel_FFB(z=4.0)
        assert m.M_ffb > m.Mc
        assert m.ffb_fraction(MH).max() < 1e-2

    def test_no_overflow_for_extreme_masses(self):
        m = UVHMRModel_FFB(z=10.0, ffb_delta=1e-3)
        with np.errstate(over="raise"):
            f = m.ffb_fraction(np.array([1e4, 1e18]))
        assert np.all(np.isfinite(f)) and np.all(f == pytest.approx(0.0))


class TestFFBEfficiency:
    def test_sfe_is_blend_of_standard_and_ffb(self):
        ffb = UVHMRModel_FFB(z=10.0, eps_max_ffb=0.6)
        std = UVHMRModel(z=10.0)
        f = ffb.ffb_fraction(MH)
        expected = (1 - f) * std.star_formation_efficiency(MH) + f * 0.6
        assert ffb.star_formation_efficiency(MH) == pytest.approx(expected)

    def test_sfe_lies_between_standard_and_eps_max(self):
        ffb = UVHMRModel_FFB(z=10.0)
        std = UVHMRModel(z=10.0).star_formation_efficiency(MH)
        eps = ffb.star_formation_efficiency(MH)
        assert np.all(eps >= std - 1e-15) and np.all(eps <= ffb.eps_max_ffb)

    def test_sfe_tends_to_eps_max_inside_window(self):
        m = UVHMRModel_FFB(z=12.0, Mc=1e14, eps_max_ffb=0.7)
        assert m.star_formation_efficiency(1e12)[0] == pytest.approx(0.7, rel=1e-3)

    def test_sfe_unchanged_when_ffb_inactive(self):
        ffb = UVHMRModel_FFB(z=4.0)
        std = UVHMRModel(z=4.0)
        assert ffb.star_formation_efficiency(MH) == pytest.approx(
            std.star_formation_efficiency(MH), rel=1e-2)


# --------------------------------------------------------------------------
# HMR FFB variants (multiple inheritance)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ffb_cls, base_cls", [
    (UVHMRModel_FFB, UVHMRModel),
    (SHMRModel_FFB, SHMRModel),
])
class TestHMRFFBInheritance:
    def test_ffb_parent_slots_in_before_HMRModel(self, ffb_cls, base_cls):
        mro = ffb_cls.__mro__
        assert mro[1] is base_cls
        assert mro.index(base_cls) < mro.index(HMRModel_FFB) < mro.index(HMRModel)
        assert issubclass(ffb_cls, base_cls)

    def test_param_names_extend_base_without_duplicates(self, ffb_cls, base_cls):
        names = ffb_cls.param_names()
        assert set(names) == set(base_cls.param_names()) | set(FFB_PARAMS)
        assert len(names) == len(set(names))

    def test_base_param_names_unchanged(self, ffb_cls, base_cls):
        assert not set(FFB_PARAMS) & set(base_cls.param_names())

    def test_ffb_kwargs_reach_HMRModel_FFB(self, ffb_cls, base_cls):
        m = ffb_cls(z=10.0, eps0=0.1, ffb_delta=0.3, eps_max_ffb=0.5)
        assert (m.eps0, m.ffb_delta, m.eps_max_ffb) == (0.1, 0.3, 0.5)

    def test_base_rejects_ffb_kwargs(self, ffb_cls, base_cls):
        with pytest.raises(TypeError):
            base_cls(z=10.0, ffb_delta=0.3)

    def test_update_parameters(self, ffb_cls, base_cls):
        m = ffb_cls(z=10.0)
        m.update_parameters(ffb_delta=0.25, eps_max_ffb=0.4, eps0=0.1)
        assert (m.ffb_delta, m.eps_max_ffb, m.eps0) == (0.25, 0.4, 0.1)

    def test_settings_report_ffb_parameters(self, ffb_cls, base_cls):
        settings = ffb_cls(z=10.0, ffb_delta=0.3)._settings()
        assert settings["ffb_delta"] == 0.3
        assert settings["eps_max_ffb"] == DEFAULT_HOD_PARAMS["eps_max_ffb"]

    def test_pickle_roundtrip(self, ffb_cls, base_cls):
        m = ffb_cls(z=10.0, ffb_delta=0.3, eps_max_ffb=0.5)
        m2 = pickle.loads(pickle.dumps(m))
        assert type(m2) is ffb_cls
        assert m2.ffb_fraction(MH) == pytest.approx(m.ffb_fraction(MH))


class TestUVHMRFFB:
    # add_dust=False: the dust iteration only converges MUV to ~0.02 mag,
    # which would swamp the small differences compared here.

    def test_add_dust_still_passed_through(self):
        m = UVHMRModel_FFB(z=10.0, add_dust=False, ffb_delta=0.3)
        assert m.add_dust is False and m.ffb_delta == 0.3
        assert m._settings()["add_dust"] is False

    def test_ffb_haloes_are_brighter(self):
        ffb = UVHMRModel_FFB(z=10.0, add_dust=False)
        std = UVHMRModel(z=10.0, add_dust=False)
        dMUV = ffb.MUV(MH) - std.MUV(MH)
        assert np.all(dMUV <= 1e-9)
        assert dMUV.min() < -1.0           # strong boost inside the FFB window

    def test_boost_is_confined_to_ffb_window(self):
        ffb = UVHMRModel_FFB(z=10.0, add_dust=False)
        std = UVHMRModel(z=10.0, add_dust=False)
        outside = np.array([1e8, 1e15])
        assert ffb.MUV(outside) == pytest.approx(std.MUV(outside), abs=1e-3)

    def test_matches_standard_when_ffb_inactive(self):
        ffb = UVHMRModel_FFB(z=4.0, add_dust=False)
        std = UVHMRModel(z=4.0, add_dust=False)
        assert ffb.MUV(MH) == pytest.approx(std.MUV(MH), abs=1e-2)

    def test_eps_max_update_changes_luminosity(self):
        m = UVHMRModel_FFB(z=10.0, add_dust=False)
        before = m.MUV(1e11)[0]
        m.update_parameters(eps_max_ffb=0.3)
        assert m.MUV(1e11)[0] > before        # fainter

    def test_inverse_relation_roundtrip(self):
        m = UVHMRModel_FFB(z=10.0, add_dust=False)
        Mh = np.array([1e10, 1e11, 1e12])
        assert m.Mhalo(m.MUV(Mh)) == pytest.approx(Mh, rel=0.05)


class TestSHMRFFB:
    def test_mstar_is_baryon_fraction_times_ffb_sfe_times_mh(self):
        m = SHMRModel_FFB(z=10.0)
        assert m.Mstar(MH) == pytest.approx(
            BARYON_FRACTION * m.star_formation_efficiency(MH) * MH)

    def test_ffb_haloes_are_more_massive(self):
        ffb = SHMRModel_FFB(z=10.0).Mstar(MH)
        std = SHMRModel(z=10.0).Mstar(MH)
        assert np.all(ffb >= std) and np.max(ffb / std) > 3

    def test_matches_standard_when_ffb_inactive(self):
        assert SHMRModel_FFB(z=4.0).Mstar(MH) == pytest.approx(
            SHMRModel(z=4.0).Mstar(MH), rel=1e-2)

    def test_mstar_below_baryon_budget(self):
        m = SHMRModel_FFB(z=10.0, eps_max_ffb=1.0)
        assert np.all(m.Mstar(MH) <= BARYON_FRACTION * MH * (1 + 1e-12))


# --------------------------------------------------------------------------
# HOD FFB variants
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ffb_cls, base_cls, sigma_name", [
    (UVHODModel_FFB, UVHODModel, "sigma_UV"),
    (MstarHODModel_FFB, MstarHODModel, "sigma_Mstar"),
])
class TestHODFFBInheritance:
    def test_ffb_parent_slots_in_before_HMRModel(self, ffb_cls, base_cls,
                                                 sigma_name):
        mro = ffb_cls.__mro__
        assert mro[1] is base_cls
        assert mro.index(base_cls) < mro.index(HMRModel_FFB) < mro.index(HMRModel)

    def test_param_names_extend_base_without_duplicates(self, ffb_cls,
                                                        base_cls, sigma_name):
        names = ffb_cls.param_names()
        assert set(names) == set(base_cls.param_names()) | set(FFB_PARAMS)
        assert sigma_name in names and len(names) == len(set(names))

    def test_all_kwargs_reach_their_class(self, ffb_cls, base_cls, sigma_name):
        m = ffb_cls(z=10.0, Mcut=1e10, ffb_delta=0.3, eps_max_ffb=0.5,
                    **{sigma_name: 0.4})
        assert getattr(m, sigma_name) == 0.4
        assert (m.Mcut, m.ffb_delta, m.eps_max_ffb) == (1e10, 0.3, 0.5)

    def test_update_parameters(self, ffb_cls, base_cls, sigma_name):
        m = ffb_cls(z=10.0)
        m.update_parameters(eps_max_ffb=0.4, asat=1.1, **{sigma_name: 0.2})
        assert (m.eps_max_ffb, m.asat, getattr(m, sigma_name)) == (0.4, 1.1, 0.2)

    def test_str_reports_ffb_settings(self, ffb_cls, base_cls, sigma_name):
        s = str(ffb_cls(z=10.0, ffb_delta=0.3))
        assert ffb_cls.__name__ in s and "ffb_delta = 0.3" in s
        assert "eps_max_ffb" in s

    def test_pickle_roundtrip(self, ffb_cls, base_cls, sigma_name):
        m2 = pickle.loads(pickle.dumps(ffb_cls(z=10.0, eps_max_ffb=0.5)))
        assert type(m2) is ffb_cls and m2.eps_max_ffb == 0.5


class TestUVHODFFB:
    def test_add_dust_still_passed_through(self):
        assert UVHODModel_FFB(z=10.0, add_dust=False).add_dust is False

    def test_ncen_is_a_probability(self):
        ncen = UVHODModel_FFB(z=10.0).Ncen(MH, -19.0)
        assert np.all((ncen >= 0) & (ncen <= 1))

    def test_ncen_is_half_at_threshold(self):
        m = UVHODModel_FFB(z=10.0)
        assert m.Ncen(1e11, m.MUV(1e11)[0])[0] == pytest.approx(0.5)

    def test_ffb_raises_central_occupation(self):
        ffb = UVHODModel_FFB(z=10.0, add_dust=False).Ncen(MH, -20.0)
        std = UVHODModel(z=10.0, add_dust=False).Ncen(MH, -20.0)
        assert np.all(ffb >= std - 1e-12) and np.max(ffb - std) > 0.1

    def test_ngal_is_sum(self):
        m = UVHODModel_FFB(z=10.0)
        assert m.Ngal(MH, -19.0) == pytest.approx(
            m.Ncen(MH, -19.0) + m.Nsat(MH, -19.0))


class TestMstarHODFFB:
    # Thresholds are log10(Mstar / M_sun).

    def test_ncen_is_half_at_threshold(self):
        m = MstarHODModel_FFB(z=10.0)
        thresh = np.log10(m.Mstar(1e11)[0])
        assert m.Ncen(1e11, thresh)[0] == pytest.approx(0.5)

    def test_ffb_raises_central_occupation(self):
        ffb = MstarHODModel_FFB(z=10.0).Ncen(MH, 9.0)
        std = MstarHODModel(z=10.0).Ncen(MH, 9.0)
        assert np.all(ffb >= std - 1e-12) and np.max(ffb - std) > 0.1

    def test_ngal_is_sum(self):
        m = MstarHODModel_FFB(z=10.0)
        assert m.Ngal(MH, 9.0) == pytest.approx(
            m.Ncen(MH, 9.0) + m.Nsat(MH, 9.0))


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------
Z_FFB = 10.0


def _coarse_obs(model):
    o = Observables(model)
    # coarse grid keeps these tests fast (accuracy checked in test_model.py)
    o._hmf_cache[model.z] = get_halo_mass_function(
        model.z, M_min=6, M_max=15, num_points=256)
    return o


@pytest.fixture(scope="module")
def uv_pair():
    """(FFB, standard) UV observables at Z_FFB."""
    return (_coarse_obs(UVHODModel_FFB(z=Z_FFB)),
            _coarse_obs(UVHODModel(z=Z_FFB)))


@pytest.fixture(scope="module")
def mstar_pair():
    """(FFB, standard) stellar-mass observables at Z_FFB."""
    return (_coarse_obs(MstarHODModel_FFB(z=Z_FFB)),
            _coarse_obs(MstarHODModel(z=Z_FFB)))


class TestFFBObservablesDispatch:
    def test_base_dispatches_ffb_models(self):
        assert type(Observables(UVHODModel_FFB(z=Z_FFB))) is UVObservables
        assert type(Observables(MstarHODModel_FFB(z=Z_FFB))) is MstarObservables

    def test_subclass_rejects_wrong_ffb_model(self):
        with pytest.raises(TypeError):
            UVObservables(MstarHODModel_FFB(z=Z_FFB))
        with pytest.raises(TypeError):
            MstarObservables(UVHODModel_FFB(z=Z_FFB))

    def test_base_rejects_ffb_hmr_model(self):
        with pytest.raises(TypeError):
            Observables(UVHMRModel_FFB(z=Z_FFB))

    def test_standard_observables_reject_ffb_parameters(self, uv_pair):
        _, std = uv_pair
        with pytest.raises(TypeError, match="unrecognised"):
            std.luminosity_function(np.array([-20.0]), eps_max_ffb=0.5)


class TestUVFFBObservables:
    def test_uvlf_positive_and_falls_to_bright_end(self, uv_pair):
        ffb, _ = uv_pair
        phi = ffb.luminosity_function(np.linspace(-23, -17, 13))
        assert np.all(phi > 0) and np.all(np.diff(phi) > 0)

    def test_ffb_boosts_bright_end(self, uv_pair):
        ffb, std = uv_pair
        MUV = np.linspace(-23, -17, 7)
        ratio = ffb.luminosity_function(MUV) / std.luminosity_function(MUV)
        assert np.all(ratio >= 1)
        assert np.all(np.diff(ratio) < 0)      # boost grows to the bright end
        assert ratio[0] > 10

    def test_uvlf_integrates_to_number_density(self, uv_pair):
        ffb, _ = uv_pair
        MUV = np.linspace(-27.0, -19.0, 321)
        n_int = np.trapz(ffb.luminosity_function(MUV), MUV)
        assert n_int == pytest.approx(ffb.number_density(-19.0), rel=2e-2)

    def test_inline_eps_max_override_persists(self, uv_pair):
        ffb, _ = uv_pair
        MUV = np.array([-22.0])
        default = DEFAULT_HOD_PARAMS["eps_max_ffb"]
        full = ffb.luminosity_function(MUV, eps_max_ffb=default)[0]
        weak = ffb.luminosity_function(MUV, eps_max_ffb=0.3)[0]
        assert weak < full
        assert ffb.hod_model.eps_max_ffb == 0.3
        ffb.luminosity_function(MUV, eps_max_ffb=default)  # restore for other tests

    def test_bias_grows_with_luminosity(self, uv_pair):
        ffb, _ = uv_pair
        bg = ffb.galaxy_bias(np.array([-22.0, -20.5, -19.0]))
        assert bg[0] > bg[1] > bg[2] > 1

    def test_ffb_hosts_bright_galaxies_in_lighter_haloes(self, uv_pair):
        ffb, std = uv_pair
        assert ffb.mean_halo_mass(-21.0) < std.mean_halo_mass(-21.0)

    def test_matches_standard_when_ffb_inactive(self):
        MUV = np.linspace(-22, -17, 6)
        ffb = _coarse_obs(UVHODModel_FFB(z=4.0)).luminosity_function(MUV)
        std = _coarse_obs(UVHODModel(z=4.0)).luminosity_function(MUV)
        assert ffb == pytest.approx(std, rel=2e-2)


class TestMstarFFBObservables:
    # Thresholds are log10(Mstar / M_sun).

    def test_smf_positive_and_falls_to_high_mass(self, mstar_pair):
        ffb, _ = mstar_pair
        phi = ffb.stellar_mass_function(np.linspace(8.0, 10.5, 11))
        assert np.all(phi > 0) and np.all(np.diff(phi) < 0)

    def test_ffb_boosts_high_mass_end(self, mstar_pair):
        ffb, std = mstar_pair
        logM = np.linspace(8.0, 10.5, 6)
        ratio = ffb.stellar_mass_function(logM) / std.stellar_mass_function(logM)
        assert np.all(ratio >= 1)
        assert np.all(np.diff(ratio) > 0)      # boost grows to high mass
        assert ratio[-1] > 10

    def test_smf_integrates_to_number_density(self, mstar_pair):
        ffb, _ = mstar_pair
        logM = np.linspace(9.0, 13.0, 161)
        n_int = np.trapz(ffb.stellar_mass_function(logM), logM)
        assert n_int == pytest.approx(ffb.number_density(9.0), rel=2e-2)

    def test_inline_eps_max_override_persists(self, mstar_pair):
        ffb, _ = mstar_pair
        logM = np.array([10.0])
        default = DEFAULT_HOD_PARAMS["eps_max_ffb"]
        full = ffb.stellar_mass_function(logM, eps_max_ffb=default)[0]
        weak = ffb.stellar_mass_function(logM, eps_max_ffb=0.3)[0]
        assert weak < full
        assert ffb.hod_model.eps_max_ffb == 0.3
        ffb.stellar_mass_function(logM, eps_max_ffb=default)  # restore


# --------------------------------------------------------------------------
# Halomod wrapper selection (no halomod model is built)
# --------------------------------------------------------------------------
class TestFFBHalomodWrapper:
    def test_wrapper_model_classes(self):
        assert HalomodUVHOD_FFB._model_class is UVHODModel_FFB
        assert HalomodMstarHOD_FFB._model_class is MstarHODModel_FFB
        assert issubclass(HalomodUVHOD_FFB, HalomodUVHOD)
        assert issubclass(HalomodMstarHOD_FFB, HalomodMstarHOD)

    def test_wrapper_defaults_include_ffb_parameters(self):
        for wrapper in (HalomodUVHOD_FFB, HalomodMstarHOD_FFB):
            assert set(FFB_PARAMS) <= set(wrapper._defaults)
        for wrapper in (HalomodUVHOD, HalomodMstarHOD):
            assert not set(FFB_PARAMS) & set(wrapper._defaults)

    @pytest.mark.parametrize("model, wrapper", [
        (UVHODModel_FFB, HalomodUVHOD_FFB),
        (UVHODModel, HalomodUVHOD),
        (MstarHODModel_FFB, HalomodMstarHOD_FFB),
        (MstarHODModel, HalomodMstarHOD),
    ])
    def test_observables_pick_matching_wrapper(self, model, wrapper):
        assert Observables(model(z=Z_FFB))._halomod_wrapper_class() is wrapper

    def test_ffb_wrapper_builds_ffb_model_with_parameters(self):
        pytest.importorskip("halomod")
        hod = HalomodUVHOD_FFB(z=Z_FFB, thresh1=-19.0, h=0.7,
                               ffb_delta=0.25, eps_max_ffb=0.4)
        assert type(hod.hod_model) is UVHODModel_FFB
        assert (hod.hod_model.ffb_delta, hod.hod_model.eps_max_ffb) == (0.25, 0.4)
