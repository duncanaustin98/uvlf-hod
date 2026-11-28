"""Tests for the general dust prescription (halogal.dust) and its use in the
UV luminosity functions and models.

Fast, deterministic unit tests only (no halomod model is built here).
"""

import pickle

import numpy as np
import pytest

from halogal import UVHMRModel, UVHMRModel_FFB, UVHODModel, UVHODModel_FFB
from halogal.config import (DUST_BETA_INTR, DUST_C0, DUST_C1, DUST_SIGMA_BETA,
                            EPOCHS_PAPER1_BETA_MUV)
from halogal.dust import (DEFAULT_DUST, BetaDust, Bouwens14BetaMUV, ConstantZ,
                          FunctionBetaMUV, LinearBetaMUV, LinearZ, NoDust,
                          PowerLawZ, TabulatedZ, as_z_function, resolve_dust)
from halogal.luminosity import (MUV_from_SFR, SFR_from_MUV, beta_color,
                                dust_attenuation)
from halogal.model import Observables

K = 0.4 * np.log(10.0)
MODES = ("luminosity", "mean", "none")

# Regression values from the pre-refactor code (HEAD f5eaf40):
# beta_color(z, M) and the old dust_attenuation(z, M) for
REF_M = np.array([-23.0, -21.0, -19.5, -18.0, -16.0])
REF_BETA = {
    4.0: [-1.4575, -1.6875, -1.86, -2.0043863949, -2.1303931432],
    6.0: [-1.3045454545, -1.7045454545, -2.0045454545, -2.2005318251,
          -2.2921214565],
    7.5: [-1.4775, -1.8275, -2.09, -2.2496100697, -2.3112994004],
    10.0: [-1.605, -1.905, -2.13, -2.2650695065, -2.3155120486],
}
REF_OLD_AUV_Z6 = [2.0447735077, 1.2487735077, 0.6517735077, 0.2617606303,
                  0.0794972639]
REF_MUV_NODUST_Z6 = [-15.7182553991, -18.2182553991, -20.7182553991,
                     -23.2182553991]


def mc_average(m, s, mode, n=1_000_000, seed=1):
    """Monte Carlo average of max(0, x), x ~ N(m, s^2)."""
    x = np.random.default_rng(seed).normal(m, s, n)
    A = np.maximum(x, 0.0)
    if mode == "mean":
        return A.mean()
    return np.log(np.mean(np.exp(K * A))) / K


# --------------------------------------------------------------------------
# Redshift functions
# --------------------------------------------------------------------------
class TestZFunctions:
    def test_constant(self):
        f = ConstantZ(1.5)
        assert np.all(f(np.array([3.0, 9.0])) == 1.5)

    def test_linear_pivot_and_slope(self):
        f = LinearZ(-2.3, -0.03, z_pivot=7.0)
        assert f(7.0) == pytest.approx(-2.3)
        assert f(10.0) == pytest.approx(-2.3 - 0.09)

    def test_linear_clip_holds_value_outside_range(self):
        f = LinearZ(1.0, 0.5, z_pivot=7.0, z_clip=(6.5, 13.5))
        assert f(3.0) == pytest.approx(f(6.5))
        assert f(20.0) == pytest.approx(f(13.5))
        assert f(10.0) == pytest.approx(2.5)

    def test_powerlaw(self):
        f = PowerLawZ(0.36, 0.5, z_pivot=7.0)
        assert f(7.0) == pytest.approx(0.36)
        assert f(15.0) == pytest.approx(0.36 * 2 ** 0.5)

    def test_tabulated_interpolates_and_clamps(self):
        f = TabulatedZ([9.0, 7.0, 8.0], [3.0, 1.0, 2.0])   # unsorted input
        assert f(7.5) == pytest.approx(1.5)
        assert f(5.0) == pytest.approx(1.0)
        assert f(12.0) == pytest.approx(3.0)

    def test_tabulated_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            TabulatedZ([1.0, 2.0], [1.0])

    def test_as_z_function_accepts_all_forms(self):
        assert as_z_function(2) == ConstantZ(2.0)
        assert isinstance(as_z_function(([1, 2], [3, 4])), TabulatedZ)
        lin = LinearZ(1.0, 2.0)
        assert as_z_function(lin) is lin
        assert as_z_function(np.sqrt) is np.sqrt
        with pytest.raises(TypeError):
            as_z_function("not a function")


# --------------------------------------------------------------------------
# beta - M_UV relations
# --------------------------------------------------------------------------
class TestBetaRelations:
    @pytest.mark.parametrize("z", sorted(REF_BETA))
    def test_bouwens14_matches_legacy_beta_color(self, z):
        ref = REF_BETA[z]
        assert Bouwens14BetaMUV().mean(z, REF_M) == pytest.approx(ref, abs=1e-9)
        assert beta_color(z, REF_M) == pytest.approx(ref, abs=1e-9)

    def test_bouwens14_default_scatter(self):
        assert np.all(Bouwens14BetaMUV().sigma(6.0, REF_M) == DUST_SIGMA_BETA)

    def test_epochs_preset_values(self, epochs_beta):
        p = EPOCHS_PAPER1_BETA_MUV
        assert epochs_beta.mean(7.0, -19.0) == pytest.approx(p["beta_M0"][0])
        assert epochs_beta.sigma(7.0, -19.0) == pytest.approx(p["sigma_beta"][0])
        # slope in M_UV at z = 10
        dbeta = p["dbeta_dMUV"][0] + 3 * p["dbeta_dMUV"][1]
        b10 = epochs_beta.mean(10.0, np.array([-19.0, -20.0]))
        assert b10[1] - b10[0] == pytest.approx(-dbeta)
        assert b10[0] == pytest.approx(p["beta_M0"][0] + 3 * p["beta_M0"][1])

    def test_epochs_preset_clamped_outside_fit_range(self, epochs_beta):
        assert epochs_beta.mean(5.0, -19.0) == pytest.approx(
            epochs_beta.mean(6.5, -19.0))
        assert epochs_beta.mean(16.0, -19.0) == pytest.approx(
            epochs_beta.mean(13.5, -19.0))

    def test_from_table_interpolates_between_bins(self):
        rel = LinearBetaMUV.from_table([7, 8], [-2.2, -2.4], [-0.1, 0.0],
                                       [0.3, 0.5])
        assert rel.mean(7.5, -19.0) == pytest.approx(-2.3)
        assert rel.mean(7.5, -20.0) == pytest.approx(-2.3 + 0.05)
        assert rel.sigma(7.5, -19.0) == pytest.approx(0.4)

    def test_from_table_without_scatter(self):
        rel = LinearBetaMUV.from_table([7, 8], [-2.2, -2.4], [-0.1, 0.0])
        assert np.all(rel.sigma(7.5, REF_M) == 0.0)

    def test_negative_sigma_is_clipped(self):
        rel = LinearBetaMUV(-2.0, -0.1, sigma_beta=LinearZ(0.1, -1.0))
        assert rel.sigma(5.0, -19.0) == 0.0

    def test_function_relation(self):
        rel = FunctionBetaMUV(lambda z, M: -2.0 - 0.01 * z + 0.0 * M,
                              lambda z, M: 0.1 * np.ones_like(M))
        assert rel.mean(10.0, REF_M) == pytest.approx(np.full(5, -2.1))
        assert rel.sigma(10.0, REF_M) == pytest.approx(np.full(5, 0.1))
        assert FunctionBetaMUV(np.negative, 0.2).sigma(5.0, REF_M) == \
            pytest.approx(np.full(5, 0.2))
        with pytest.raises(TypeError):
            FunctionBetaMUV(1.0)

    def test_array_redshift_broadcasts(self, simple_linear_beta):
        z = np.array([6.0, 8.0, 10.0])
        assert simple_linear_beta.mean(z, -20.0).shape == (3,)


# --------------------------------------------------------------------------
# Dust attenuation
# --------------------------------------------------------------------------
class TestBetaDust:
    def test_meurer_defaults(self):
        d = BetaDust()
        assert d.dA_dbeta(6.0) == pytest.approx(DUST_C1)
        assert d.beta_intr(6.0) == pytest.approx(-DUST_C0 / DUST_C1)
        assert DUST_BETA_INTR == pytest.approx(-2.2261306533)
        assert d.scatter_average == "luminosity"
        assert d.beta_relation == Bouwens14BetaMUV()
        assert DEFAULT_DUST == d

    @pytest.mark.parametrize("mode", MODES)
    def test_never_negative(self, mode, muv_grid):
        for sigma in (0.0, 0.2, 0.6):
            rel = LinearBetaMUV(-2.4, -0.2, sigma_beta=sigma)
            d = BetaDust(rel, scatter_average=mode)
            for z in (5.0, 8.0, 12.0):
                assert np.all(d.A_UV(z, muv_grid) >= 0.0)

    @pytest.mark.parametrize("mode", MODES)
    def test_bluer_than_intrinsic_is_dust_free(self, mode):
        rel = LinearBetaMUV(-3.0, 0.0, sigma_beta=0.0)
        assert np.all(BetaDust(rel, scatter_average=mode).A_UV(7.0, REF_M) == 0)

    @pytest.mark.parametrize("mode", MODES)
    def test_no_scatter_is_clipped_linear(self, mode):
        rel = LinearBetaMUV(-2.0, -0.2, sigma_beta=0.0)
        d = BetaDust(rel, scatter_average=mode)
        m = DUST_C1 * (rel.mean(7.0, REF_M) - DUST_BETA_INTR)
        assert d.A_UV(7.0, REF_M) == pytest.approx(np.maximum(m, 0.0))

    @pytest.mark.parametrize("mode", ["luminosity", "mean"])
    @pytest.mark.parametrize("m", [-1.0, -0.2, 0.0, 0.3, 1.5])
    def test_closed_form_matches_monte_carlo(self, mode, m):
        s = 0.6
        rel = LinearBetaMUV(DUST_BETA_INTR + m / DUST_C1, 0.0,
                            sigma_beta=s / DUST_C1)
        A = BetaDust(rel, scatter_average=mode).A_UV(7.0, -19.0)
        assert A == pytest.approx(mc_average(m, s, mode), abs=3e-3)

    def test_luminosity_average_exceeds_mean(self, simple_linear_beta,
                                             muv_grid):
        lum = BetaDust(simple_linear_beta).A_UV(7.0, muv_grid)
        mean = BetaDust(simple_linear_beta,
                        scatter_average="mean").A_UV(7.0, muv_grid)
        assert np.all(lum >= mean - 1e-12)

    def test_luminosity_reduces_to_smit12_when_red(self):
        # whole distribution redder than beta_intr: A + 0.2 ln10 s^2
        sigma = 0.2
        rel = LinearBetaMUV(-1.0, 0.0, sigma_beta=sigma)
        m = DUST_C1 * (-1.0 - DUST_BETA_INTR)
        expected = m + 0.2 * np.log(10) * (DUST_C1 * sigma) ** 2
        assert BetaDust(rel).A_UV(6.0, -21.0) == pytest.approx(expected,
                                                               rel=1e-6)

    def test_default_matches_legacy_at_bright_end(self):
        # The legacy formula only differs where part of the beta
        # distribution is bluer than beta_intr (the faint end).
        old = np.array(REF_OLD_AUV_Z6)
        new = DEFAULT_DUST.A_UV(6.0, REF_M)
        assert new[0] == pytest.approx(old[0], abs=2e-3)
        assert np.all(new >= old - 1e-9)

    def test_general_slope_and_intrinsic_beta(self):
        rel = LinearBetaMUV(-1.5, 0.0)
        d = BetaDust(rel, dA_dbeta=1.0, beta_intr=-2.5)
        assert d.A_UV(7.0, -20.0) == pytest.approx(1.0)
        # redshift-evolving intrinsic slope
        d = BetaDust(rel, dA_dbeta=1.0, beta_intr=LinearZ(-2.5, -0.1, 7.0))
        assert d.A_UV(9.0, -20.0) == pytest.approx(1.2)

    def test_z_max_switches_dust_off(self):
        d = BetaDust(z_max=8.0)
        assert np.all(d.A_UV(9.0, REF_M) == 0.0)
        assert np.all(d.A_UV(np.array([7.0, 9.0]), -22.0) * [0, 1] == 0.0)
        assert d.A_UV(7.0, -22.0) > 0

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            BetaDust(scatter_average="median")
        with pytest.raises(TypeError):
            BetaDust(beta_relation="Bouwens")

    def test_no_dust(self, no_dust):
        assert np.all(no_dust.A_UV(7.0, REF_M) == 0.0)
        assert no_dust(7.0, REF_M).shape == REF_M.shape


class TestDustObjects:
    def test_immutable_and_with_params(self):
        d = BetaDust()
        with pytest.raises(AttributeError):
            d.dA_dbeta = 1.0
        d2 = d.with_params(scatter_average="mean")
        assert d2.scatter_average == "mean" and d.scatter_average == "luminosity"
        assert d2.beta_relation == d.beta_relation
        with pytest.raises(TypeError):
            d.with_params(nonsense=1)

    def test_value_equality_and_hash(self, epochs_beta):
        a = BetaDust(epochs_beta, dA_dbeta=1.5)
        b = BetaDust(LinearBetaMUV.epochs_paper1(), dA_dbeta=1.5)
        assert a == b and hash(a) == hash(b)
        assert a != a.with_params(dA_dbeta=1.6)
        assert NoDust() == NoDust() and NoDust() != DEFAULT_DUST

    @pytest.mark.parametrize("obj", [
        DEFAULT_DUST, NoDust(), BetaDust(LinearBetaMUV.epochs_paper1()),
        BetaDust(LinearBetaMUV.from_table([7, 9], [-2.2, -2.4], [-0.1, 0.0],
                                          [0.3, 0.4])),
        LinearZ(1.0, 2.0, 7.0, (6.5, 13.5)), PowerLawZ(1.0, 2.0),
        FunctionBetaMUV(np.negative, 0.2),
    ])
    def test_pickle_roundtrip(self, obj):
        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj2 == obj and hash(obj2) == hash(obj)

    def test_resolve_dust(self):
        d = BetaDust(dA_dbeta=1.0)
        assert resolve_dust(d) is d
        assert resolve_dust(None) == NoDust()
        assert resolve_dust(False) == NoDust()
        assert resolve_dust(True) is DEFAULT_DUST
        with pytest.raises(TypeError):
            resolve_dust("yes")


# --------------------------------------------------------------------------
# Luminosity conversions
# --------------------------------------------------------------------------
class TestLuminosityFunctions:
    SFR = np.array([0.1, 1.0, 10.0, 100.0])

    def test_no_dust_unchanged(self):
        assert MUV_from_SFR(self.SFR, 6.0, dust=None) == pytest.approx(
            REF_MUV_NODUST_Z6, abs=1e-9)

    def test_dust_only_dims(self, epochs_beta):
        clean = MUV_from_SFR(self.SFR, 8.0, dust=None)
        for d in (DEFAULT_DUST, BetaDust(epochs_beta)):
            assert np.all(MUV_from_SFR(self.SFR, 8.0, dust=d) >= clean)

    def test_iteration_is_self_consistent(self, epochs_beta):
        d = BetaDust(epochs_beta)
        obs = MUV_from_SFR(self.SFR, 8.0, dust=d, convergence_threshold=1e-8)
        intr = MUV_from_SFR(self.SFR, 8.0, dust=None)
        assert obs - intr == pytest.approx(d.A_UV(8.0, obs), abs=1e-6)

    def test_sfr_roundtrip_with_dust_model(self, simple_linear_beta):
        d = BetaDust(simple_linear_beta)
        obs = MUV_from_SFR(self.SFR, 7.0, dust=d, convergence_threshold=1e-10)
        assert SFR_from_MUV(obs, 7.0, dust=d) == pytest.approx(self.SFR,
                                                               rel=1e-4)

    def test_dust_attenuation_wrapper(self):
        assert dust_attenuation(6.0, REF_M) == pytest.approx(
            DEFAULT_DUST.A_UV(6.0, REF_M))
        assert np.all(dust_attenuation(9.0, REF_M, high_z_dust=False) == 0)
        d = BetaDust(dA_dbeta=1.0)
        assert dust_attenuation(6.0, REF_M, dust=d) == pytest.approx(
            d.A_UV(6.0, REF_M))
        custom = dust_attenuation(6.0, REF_M, C0=2.0, C1=1.0)
        assert custom == pytest.approx(
            BetaDust(dA_dbeta=1.0, beta_intr=-2.0).A_UV(6.0, REF_M))

    def test_deprecated_flags(self):
        with pytest.warns(DeprecationWarning):
            nodust = MUV_from_SFR(self.SFR, 6.0, add_dust=False)
        assert nodust == pytest.approx(REF_MUV_NODUST_Z6, abs=1e-9)
        with pytest.warns(DeprecationWarning):
            dusty = MUV_from_SFR(self.SFR, 6.0, add_dust=True)
        assert dusty == pytest.approx(MUV_from_SFR(self.SFR, 6.0))
        with pytest.warns(DeprecationWarning):
            sfr = SFR_from_MUV(np.array([-20.0]), 6.0, remove_dust=False)
        assert sfr == pytest.approx(SFR_from_MUV(np.array([-20.0]), 6.0,
                                                 dust=None))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class TestModelDust:
    def test_default_and_none(self, Mh_grid):
        assert UVHMRModel(z=7.0).dust == DEFAULT_DUST
        m = UVHMRModel(z=7.0, dust=None)
        assert m.dust == NoDust()
        assert m.MUV(Mh_grid) == pytest.approx(
            MUV_from_SFR(m.sfr(Mh_grid), 7.0, dust=None))

    def test_custom_dust_is_used(self, Mh_grid, epochs_beta):
        d = BetaDust(epochs_beta)
        m = UVHMRModel(z=8.0, dust=d)
        assert m.MUV(Mh_grid) == pytest.approx(
            MUV_from_SFR(m.sfr(Mh_grid), 8.0, dust=d))
        assert np.all(m.MUV(Mh_grid) >= UVHMRModel(z=8.0, dust=None).MUV(Mh_grid))

    def test_param_names_and_settings(self):
        assert "dust" in UVHMRModel.param_names()
        assert "add_dust" not in UVHMRModel.param_names()
        assert "add_dust" in UVHMRModel.accepted_param_names()
        assert UVHODModel(z=7.0)._settings()["dust"] == DEFAULT_DUST

    def test_cache_invalidated_by_dust_change(self, Mh_grid, epochs_beta):
        m = UVHODModel(z=8.0)
        base = m.Ncen(Mh_grid, -19.0).copy()
        m.update_parameters(dust=BetaDust(epochs_beta))
        changed = m.Ncen(Mh_grid, -19.0)
        assert not np.array_equal(changed, base)
        m.update_parameters(dust=BetaDust(LinearBetaMUV.epochs_paper1()))
        assert np.array_equal(m.Ncen(Mh_grid, -19.0), changed)   # equal value
        m.dust = DEFAULT_DUST
        assert np.array_equal(m.Ncen(Mh_grid, -19.0), base)

    @pytest.mark.parametrize("cls", [UVHMRModel_FFB, UVHODModel,
                                     UVHODModel_FFB])
    def test_subclasses_accept_dust(self, cls, epochs_beta):
        d = BetaDust(epochs_beta, scatter_average="mean")
        assert cls(z=10.0, dust=d).dust == d
        assert cls(z=10.0, dust=None).dust == NoDust()

    def test_model_with_custom_dust_pickles(self, Mh_grid, epochs_beta):
        m = UVHODModel(z=9.0, dust=BetaDust(epochs_beta))
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.dust == m.dust
        assert m2.MUV(Mh_grid) == pytest.approx(m.MUV(Mh_grid))

    def test_observables_accept_dust_override(self, epochs_beta):
        obs = Observables(UVHODModel(z=8.0))
        obs._apply_params({"dust": BetaDust(epochs_beta)})
        assert obs.hod_model.dust == BetaDust(epochs_beta)
        with pytest.warns(DeprecationWarning):
            obs._apply_params({"add_dust": False})
        assert obs.hod_model.dust == NoDust()

    def test_deprecated_add_dust(self):
        with pytest.warns(DeprecationWarning):
            m = UVHODModel(z=7.0, add_dust=False)
        assert m.dust == NoDust() and m.add_dust is False
        with pytest.warns(DeprecationWarning):
            m.add_dust = True
        assert m.dust == DEFAULT_DUST and m.add_dust is True
        with pytest.warns(DeprecationWarning):
            m.update_parameters(add_dust=False)
        assert m.dust == NoDust()


# --------------------------------------------------------------------------
# Inference: fitting with a dust model
# --------------------------------------------------------------------------
class TestLikelihoodDust:
    @staticmethod
    def _uvlf(z=7.0):
        from halogal.inference import Dataset
        MUV = np.linspace(-22.0, -18.0, 5)
        obs = Observables(UVHODModel(z=z))
        phi = obs.luminosity_function(MUV)
        return Dataset.uvlf(z=z, MUV=MUV, phi=phi, sigma=0.1 * phi)

    def test_model_kwargs_reach_models_and_survive_pickle(self, epochs_beta):
        from halogal.inference import GaussianLikelihood, ModelParametrization
        d = BetaDust(epochs_beta)
        like = GaussianLikelihood([self._uvlf()],
                                  ModelParametrization("fixed_z"),
                                  hmf_points=256, model_kwargs={"dust": d})
        like._ensure_setup()
        assert like._bins[7.0]["obs"].hod_model.dust == d
        like2 = pickle.loads(pickle.dumps(like))
        like2._ensure_setup()
        assert like2._bins[7.0]["obs"].hod_model.dust == d

    def test_dust_changes_likelihood(self, epochs_beta):
        from halogal.inference import GaussianLikelihood, ModelParametrization
        ds = self._uvlf()   # generated with the default dust
        lls = {name: GaussianLikelihood(
                   [ds], ModelParametrization("fixed_z"), hmf_points=256,
                   model_kwargs=kw).log_likelihood({})
               for name, kw in [("default", {}),
                                ("none", {"dust": None}),
                                ("epochs", {"dust": BetaDust(epochs_beta)})]}
        assert lls["default"] > lls["none"]
        assert lls["default"] > lls["epochs"]
