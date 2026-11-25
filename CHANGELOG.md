# Changelog

## 0.2.0 (unreleased)

### Added

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
  N_steps/(50 tau_max). eps0 and sigma_UV are shown as log10 in the corner
  plot and the walker traces (they are still sampled linearly). `--z` takes
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

- Requires Python >= 3.9 and `numpy < 2` (a `camb` limitation).
- Only the `halogal` package is installed; the top-level `tests` and `examples`
  directories are no longer shipped as packages.
- `tests/test_core.py`, which targeted a module that no longer exists, is
  replaced by `tests/test_model.py`.

## 0.1.0

Initial release: UV-halo mass relation, HOD, UV luminosity function, galaxy
bias and `halomod` correlation functions.
