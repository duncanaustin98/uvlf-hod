# Changelog

## 0.2.0 (unreleased)

### Added

- **`halogal.dust`**: a general dust prescription replacing the on/off
  `add_dust` switch. Models take `dust=<DustModel>` (or `dust=None` for no
  dust).
  - `BetaDust(beta_relation, dA_dbeta, beta_intr, scatter_average, z_max)`:
    per galaxy `A_UV = max(0, dA_dbeta * (beta - beta_intr))`, so galaxies
    bluer than the intrinsic slope are dust free instead of getting a
    negative attenuation. Defaults: Meurer+99 (`dA_dbeta = 1.99`,
    `beta_intr = -4.43/1.99`) with the Bouwens+14 beta-M_UV relation and
    `sigma_beta = 0.34`.
  - The scatter in beta enters through a closed-form average over the
    Gaussian beta distribution at fixed observed M_UV:
    `scatter_average="luminosity"` (default; `2.5 log10 <10**(0.4 A_UV)>`,
    which reduces to the old `+0.2 ln10 sigma^2 C1^2` term for red
    populations), `"mean"` (`<A_UV>`) or `"none"`.
  - beta-M_UV distributions: `LinearBetaMUV` (beta at a pivot M_UV, its slope
    and scatter, each any function of z; `from_linear_z`, `from_table`,
    `epochs_paper1()`), `Bouwens14BetaMUV` and the fully general
    `FunctionBetaMUV(mean_fn, sigma_fn)`.
  - Redshift-evolution helpers `ConstantZ`, `LinearZ` (with optional
    `z_clip`), `PowerLawZ`, `TabulatedZ` and `as_z_function`; `dA_dbeta` and
    `beta_intr` accept them too.
  - `config.EPOCHS_PAPER1_BETA_MUV`: EPOCHS Paper-1 fits of beta(M_UV=-19),
    dbeta/dM_UV and sigma_beta, each `A0 + A1 (z - 7)` over 6.5 < z < 13.5.
- `tests/conftest.py` (shared fixtures) and `tests/test_dust.py`.
- `satellites=False` on the HOD models (`UVHODModel`, `MstarHODModel` and
  the FFB variants): `N_sat = 0`, so the abundance, bias and halomod
  clustering are those of centrals only. It is a model parameter (in
  `param_names()`, `update_parameters`, the caches and the halomod wrapper).
- `examples/fit_uvlf.py --satellites / --no-satellites`: fits are now
  centrals only by default. `logMcut`, `logMsat` and `asat` are then dropped
  from the fit (and cannot be given to `--fix` / `--free`); such runs go in
  `..._nosat` folders and record `"satellites": false` in the run settings.
  `--satellites` restores the previous behaviour (satellite HOD fixed at
  Shuntov+25 unless freed), with the same folder names and settings as
  before.
- `GaussianLikelihood(model_kwargs=...)`: fixed settings passed to every
  bin's `HODModel`, e.g. `model_kwargs={"dust": BetaDust(...)}` to fit with a
  different dust prescription.
- `examples/fit_uvlf.py`:
  - `--dust {bouwens14,epochs,none}` chooses the dust prescription
    (`bouwens14`, the halogal default, is also accepted as `default`). The choice
    is in the output folder name (`..._dust=<choice>`), and the full dust
    model is stored in the chain's run settings, so a chain is never
    resumed or re-plotted with different dust. Folders without `_dust=`
    predate the option and used the earlier, unclipped dust formula.
  - `auv_mhalo.pdf`: A_UV(Mh) of the mean central (posterior median and
    16-84%, the reference curve and the fitted halo-mass range), made for
    every fit with dust, single-z and per bin of a joint fit.

- **`halogal.inference`**: Bayesian fitting of the UVHMR/HOD model to UV
  luminosity functions, angular/projected clustering and number densities.
  - `Parameter`, `ParameterSet` and the `Uniform`, `Normal`, `LogUniform`
    priors; any parameter can be free or fixed.
  - `ModelParametrization` with two regimes: `fixed_z` (HOD parameters at one
    redshift) and `evolution` (linear redshift-evolution coefficients fitted
    jointly across bins).
  - `Dataset.uvlf`, `Dataset.clustering`, `Dataset.number_density`, each with a
    full covariance or diagonal errors.
  - `GaussianLikelihood`, which keeps one `halomod` model per redshift bin and
    only updates it while sampling; `check()` validates a setup before a run.
  - `EmceeSampler` and `DynestySampler` behind one interface. Both take
    `processes=N` and build the model once per worker.
  - `EmceeSampler.run`: HDF5 checkpoint/resume (`backend=`) and
    autocorrelation-based stopping (`convergence=True`).
  - `EmceeSampler.run(callback=..., callback_every=100)`: calls
    `callback(sampler)` every N steps in both modes, e.g. for progress plots
    during a long run.
  - `InferenceResult`: summaries, weighted quantiles, corner plot, evidence,
    `save()` / `load()`.
  - `Dataset.uvlf(sigma_low=..., sigma_upp=...)`: asymmetric errors through a
    split-normal likelihood. Each point uses its lower error when the model
    lies below it and its upper error otherwise. The normalisation is model
    independent, so the likelihood is continuous where the model crosses a
    point. `Dataset.effective_sigma(model)` returns the errors in use.
  - `Dataset.uvlf(dMUV=...)`: the bin width (scalar or per point). The model
    is then averaged over each data bin, as a binned UVLF measures. The default
    is unchanged: the 0.05 mag width, effectively Phi at the bin centre.
- HOD occupation functions cache the selection variable (M_UV(Mh) with its
  iterative dust correction, or log Mstar(Mh)). It is recomputed only when a
  parameter, setting or the halo-mass grid changes, not for every threshold:
  a UVLF likelihood call is ~5x faster, with identical results.
- The EPOCHS-v2 (Austin et al. 2026, EPOCHS-DR2) UVLF at z = 7-12.5, in
  `examples/austin26_data/` with the loader `examples/austin26_data.py`.
- `examples/fit_uvlf.py`: a single-redshift fit of the EPOCHS-v2 UVLF with
  `eps0`, `a` and `sigma_UV` free. `logMc`, the high-mass slope `b` and the
  satellite HOD are fixed to the Shuntov+2025 evolution at that redshift. It
  writes a corner plot, a data + model + residuals plot and a convergence
  diagnostic to `examples/outputs/` (git-ignored). Convergence is logged to a
  text file at every check. The fit and plots are written when the run
  finishes, or from the saved chain when it is stopped with Ctrl-C or SIGTERM.
  The checkpoint stores the settings defining the posterior, and a resume
  with different settings is refused (with `--outdir`). `--nsteps` caps the total number of
  steps including resumed ones. `--fix NAME` fixes any parameter at its
  Shuntov+25 value, `--fix NAME=VALUE` at an explicit value, and `--free NAME`
  fits one that is fixed by default. `--shuntov z7.3` (or `z4.3`, `z5.4`)
  takes the Shuntov+25 values from that bin's Table 2 posterior medians
  instead of their redshift evolution (`--shuntov evo`, the default). Outputs go to
  `EPOCHS_DR2_UVLF/fixed=<p1>+<p2>+..._shuntov25_<shuntov>[_n]/z=<z>/` (e.g. `..._shuntov25_z7.3`). Each run saves its fixed values
  and priors to `params.h5`, and a re-run uses the first `_n`
  folder whose records match, or starts a new one. `--init prior` (default) starts
  new chains from draws from the priors; `--init fiducial` starts them in a
  Gaussian ball about the Shuntov+25 values (`--init-scatter`). The
  starting positions are saved in `params.h5` (group `walker_start`). The fitted data
  points are saved in `params.h5` (group `data`). A run with different
  fitted data or settings (M_UV cut, HMF grid, walkers) for the same z goes
  to a new `z=<z>_2`, `_3`, ... folder. Every fit discards burn-in and thins as
  galfind does: tau is the largest autocorrelation time (N/50 if emcee
  cannot yet estimate it), and the first 2 tau steps are dropped and the rest
  thinned by tau/2. The corner plot reports N_walkers, N_steps and
  N_steps/(50 tau_max). eps0 is shown as log10 in the corner
  plot and the walker traces (it is still sampled linearly). `--z` takes
  several bins (or `all`) and fits them in turn; an interrupt writes the
  current bin's fit and plots, skips the rest and exits with status 130. Walkers
  move with emcee's differential-evolution proposals by default (`--moves de`;
  `--moves stretch` for the stretch move). The plots of an already converged chain
  are not remade if they exist (`--replot` forces them). The UVLF plot lists the fixed values in the
  main panel and shows log10(Phi_obs/Phi_model) with the reduced chi^2 and
  p-value below it. Each fit also plots the star-formation efficiency
  eps(Mh) (`sfe_mhalo.pdf`), the mean M_UV(Mh) with its sigma_UV scatter
  (`muv_mhalo.pdf`) and the galaxy bias in M_UV bins and of cumulative
  M_UV < M_UV,thresh samples, against Shuntov+25's published values
  (`galaxy_bias.pdf`), all with posterior 16-84% bands from the same draws as
  the UVLF band. Re-running on a converged chain adds any of them that are
  missing.
- `ModelParametrization("powerlaw", z_pivot=...)`: power-law redshift
  evolution `A(z) = A_0 ((1 + z)/(1 + z_pivot))**A_1`, with coefficients
  `<x>_0` (value at the pivot) and `<x>_1` (exponent) per HOD parameter
  (`powerlaw_param_names()`); for masses `log10 M` is linear in
  `log10(1 + z)`. A parameter without power-law coefficients follows the
  linear law (`d_<x>_dz`, `C_<x>`; by default the published Shuntov+25
  evolution).
- `examples/fit_uvlf.py --joint`: fits several EPOCHS-v2 bins at once with
  each free parameter evolving as a power law normalised at `--z-pivot`
  (default 7). `--no-evolve NAME` keeps a free parameter constant in z.
  Fixed parameters follow Shuntov+25. Outputs go to
  `EPOCHS_DR2_UVLF/joint_powerlaw/`, with a parameters-vs-z plot, a plot
  overlaying every bin (`combined_z.pdf`: UVLF with data, SFE, M_UV-Mhalo
  and bias, coloured by z) and the single-z plots of every bin. `--joint` with a single bin runs the ordinary single-z fit.
- `examples/fit_uvlf.py --lit [NAME ...]`: also fits literature UVLFs, by
  default the bright-end Bowler+17, Varadaraj+23 and Varadaraj+26 points at
  z=7 (copied from flags_data into `examples/literature_data/`, loader
  `examples/literature_data.py`). Each set is fitted with its own bin widths,
  shown in the UVLF plot with its residuals, saved in `params.h5` and gets
  outputs go to a folder named after the fitted data, e.g.
  `EPOCHS_DR2+Bowler17+Varadaraj23+Varadaraj26_UVLF/` instead of
  `EPOCHS_DR2_UVLF/`.
- Parallel emcee pool workers ignore SIGINT, so Ctrl-C stops a
  `processes=N` run cleanly from the main process.
- `halogal.crosscorr`: `Selection` and `CrossCorrelation` for cross-correlations
  between two magnitude-selected samples.
- `environment.yml` with a known-good conda-forge stack, and the
  `halogal[inference]` extra.

### Fixed

- Cosmology is defined once (`config.DEFAULT_COSMO_PARAMS`) and passed to
  `halomod` as well as Colossus; previously the clustering ran with `halomod`'s
  own `sigma8`/`ns`/transfer function while the UVLF used halogal's.
- The `halomod` halo mass function is converted to `dn/dlog10M` like the
  Colossus one; absolute quantities on that path were low by `1/ln(10)`.
- `h` in the unit conversions follows the active cosmology instead of a
  hard-coded 0.7.
- Integer magnitude or mass arrays no longer truncate the UVLF / `N_sat`.
- Misspelled parameter names passed to `Observables` methods raise `TypeError`
  instead of being ignored.
- Magnitude-binned clustering samples (`MUV_thresh2`) are handled correctly
  and mis-ordered thresholds are rejected.
- Two typos in the Bouwens+2021 UVLF errors shipped in
  `examples/bouwens21_data.py` (z~5 and z~6, a factor of 10 too small).

### Changed

- Default dust is clipped per galaxy before averaging over the beta scatter,
  so faint galaxies whose mean beta is near or below `beta_intr` get slightly
  more attenuation than before (e.g. ~0.08 -> ~0.28 mag at M_UV = -16,
  z = 6). Bright galaxies are unchanged.
- `UVHMRModel.param_names()` lists `dust` instead of `add_dust`.
- `examples/fit_uvlf.py`: the default `--nsteps` cap is 100000 steps per
  walker (was 50000).

- Requires Python >= 3.9 and `numpy < 2` (a `camb` limitation).
- Only the `halogal` package is installed; the top-level `tests` and `examples`
  directories are no longer shipped as packages.
- `tests/test_core.py`, which targeted a module that no longer exists, is
  replaced by `tests/test_model.py`.

### Deprecated

- `add_dust=` (models, `MUV_from_SFR`) and `remove_dust=` (`SFR_from_MUV`)
  are still accepted with a `DeprecationWarning`: `True` maps to the default
  dust model and `False` to no dust. The `add_dust` model attribute is now a
  view of `dust`; setting it warns.

## 0.1.0

Initial release: UV-halo mass relation, HOD, UV luminosity function, galaxy
bias and `halomod` correlation functions.
