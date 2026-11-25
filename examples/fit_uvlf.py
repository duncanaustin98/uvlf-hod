#!/usr/bin/env python
"""Fit the UVHOD model to the EPOCHS-v2 UVLF at a single redshift.

By default the free parameters are ``eps0, a, sigma_UV``. The characteristic
mass ``logMc``, the high-mass slope ``b`` and the satellite HOD (``logMcut,
logMsat, asat``) are held fixed, at their Shuntov+2025 values: the linear
redshift evolution (``halogal.config.DEFAULT_REDSHIFT_EVOLUTION``) evaluated
at the fit redshift. On the command line (all repeatable):
    --fix NAME          fix NAME at its Shuntov+25 value
    --fix NAME=VALUE    fix NAME at VALUE (masses in log10), e.g. asat=0.8
    --free NAME         fit NAME although it is fixed by default, e.g.
                        ``--free logMc`` for the eps0, logMc, a, sigma_UV fit
    --shuntov z7.3      take the Shuntov+25 values from their z~7.3 bin
                        (Table 2 posterior medians) instead of the redshift
                        evolution; also z4.3, z5.4 (default: evo)

Walkers move with emcee's differential-evolution proposals by default
(``--moves de``: 80% DEMove + 20% DESnookerMove), which explore correlated
posteriors much faster than the default stretch move (``--moves stretch``).

Walkers of a new chain start from draws from the priors (``--init prior``,
the default), or from a Gaussian ball about the Shuntov+25 values
(``--init fiducial``, fractional width ``--init-scatter``, default 1e-3).

The data are the fiducial EPOCHS-v2 (Austin+2026) UVLF from
``examples/austin26_data``. Its asymmetric errors enter through a split-normal
likelihood (``Dataset.uvlf(sigma_low=..., sigma_upp=...)``). By default only
bins with M_UV <= -18 are fitted, as in the EPOCHS-v2 fiducial HOD fits.

The model is averaged over each 0.5 mag data bin (``dMUV=BIN_WIDTH``), which
is what the binned 1/Vmax UVLF measures. Unlike the galfind-based EPOCHS-v2
fitters, the width is not inferred from adjacent points: empty bins are dropped,
so at z=7 and z=10 that difference is 1 mag.

``--lit`` also fits literature UVLFs (``examples/literature_data.py``): with
no names, the defaults of each bin (z=7: the bright-end Bowler+17,
Varadaraj+23 and Varadaraj+26), or ``--lit Bowler+17 Varadaraj+26`` for chosen
sets. Each set is its own dataset, with the model averaged over its own bin
widths; the M_UV cut applies to it too. These ground-based samples share
VISTA imaging, so they are not fully independent of each other. Runs with
literature go under a folder named after the data, e.g.
``examples/outputs/EPOCHS_DR2+Bowler17+Varadaraj23+Varadaraj26_UVLF/``
instead of ``EPOCHS_DR2_UVLF/`` (same layout below it), and params.h5 stores the fitted points (group ``literature``). The UVLF plot
shows them, with their residuals and in the chi^2.

Usage (run as a script so multiprocessing works):

    python examples/fit_uvlf.py --z 7
    python examples/fit_uvlf.py --z 9 --processes 8 --nsteps 40000
    python examples/fit_uvlf.py --z 7 8 9 --processes 8     # one after another
    python examples/fit_uvlf.py --z all --processes 8       # every bin

    python examples/fit_uvlf.py --z all --joint --processes 8   # jointly
    python examples/fit_uvlf.py --z all --joint --no-evolve sigma_UV
    python examples/fit_uvlf.py --z 7 --lit --processes 8   # + bright end

``--joint`` with a single bin runs the ordinary single-z fit (no evolution).

Joint fit (``--joint``, two or more bins): all bins are fitted at once, each
free parameter A evolving as a power law normalised at ``--z-pivot`` z_p
(default 7, the lowest bin):
    A(z) = A(z_p) * ((1 + z) / (1 + z_p))**A_1
and, for the masses, log10 M(z) = log10 M(z_p) + logM_1 log10((1+z)/(1+z_p)).
The sampled coefficients are ``<x>_0`` = A(z_p), with the single-z prior of
A (PRIORS), and ``<x>_1`` = the exponent, with EXPONENT_PRIOR (flat in
[-5, 5]). ``--no-evolve NAME`` keeps a free parameter constant in z (its
exponent fixed at 0). Fixed parameters follow the Shuntov+25 linear evolution
(``--shuntov evo``), or are constant in z (a ``--shuntov`` bin, or
``--fix NAME=VALUE``). Outputs go to ``<data>_UVLF/joint_powerlaw/
fixed=..._shuntov25_<shuntov>[_const=<no-evolve names>]/z=7+8+...[_n]/``:
chain.h5, params.h5 (with ``model``, ``z_pivot`` and the redshift of every
fitted point), convergence.log/.pdf, corner.pdf (red: a power-law
approximation of Shuntov+25 over the fitted z), params_vs_z.pdf (each
evolving parameter against z, with the total chi^2) and, per bin, ``z=<z>/``
with the single-z UVLF, SFE, M_UV-Mhalo and bias plots (chi^2 there is per
bin over all its points, as the free parameters are shared).

With several redshifts and no ``--joint``, the bins are fitted in turn, each
in its own folder as for a single run. Ctrl-C or kill writes the fit and plots of the bin being
sampled, skips the rest and exits with status 130; re-running the same command
resumes that bin, and bins already converged (or at the --nsteps cap) are only
re-plotted, not re-sampled.

By default the chain runs until the emcee autocorrelation criterion is met
(chain > 50 tau, tau stable to 1%), with ``--nsteps`` as a cap on the TOTAL
steps per walker. Pass ``--no-converge`` to run to exactly ``--nsteps`` total
steps instead.

The chain is checkpointed to ``<outdir>/chain.h5``, and re-running the
same command resumes it (delete the file to start afresh). The checkpoint
stores the settings defining the posterior (z, M_UV cut, bin width, HMF grid,
walkers, priors, fixed values and a hash of the fitted data). A resume with
different settings, or different fitted data (stored in ``params.h5``), goes
to a new ``z=<z>_2``, ``_3``, ... folder, so two posteriors are never mixed. Re-running
on a checkpoint that has already converged, or that is at the ``--nsteps``
cap, re-makes the fit without sampling; the plots of a converged chain are
only remade if any is missing, or with ``--replot``.

Outputs go to
``examples/outputs/EPOCHS_DR2_UVLF/fixed=<p1>+<p2>+..._shuntov25_<shuntov>/z=<z>/``
(``EPOCHS_DR2+<sets>_UVLF`` with ``--lit``), which git ignores: one folder per fixed set, named from the fixed parameter
names and the ``--shuntov`` source of their values (default
``fixed=logMc+b+logMcut+logMsat+asat_shuntov25_evo``; with ``--shuntov z7.3``,
``..._shuntov25_z7.3``). Each run records its fixed values (and
whether they are Shuntov+25 or ``--fix`` values) and its priors in
``z=<z>/params.h5``. On every run the folder is chosen by checking
``fixed=...``, then ``fixed=..._2``, ``_3``, ... and taking the first whose
records match this run's fixed values and priors (for another z: the same
priors and fixing method, as the Shuntov+25 values evolve with z). If none
matches, the next unused suffix starts a new run. Each redshift then gets its
own ``z=<z>`` subfolder (e.g. ``z=7``, ``z=12.5``). ``--outdir`` gives the
run folder directly and skips this search. Files in the run folder:
    chain.h5                 emcee checkpoint: the full chain, from which
                             every fit is rebuilt
    corner.pdf               posterior corner plot (Shuntov+25 values marked)
    uvlf_residuals.pdf       data + model (median, 16-84%), and
                             log10(Phi_obs / Phi_model) with reduced chi^2
                             and p-value
    sfe_mhalo.pdf            star-formation efficiency eps(Mh) (median,
                             16-84%), with the halo masses the fitted M_UV
                             bins map onto
    muv_mhalo.pdf            mean M_UV(Mh) of centrals (median, 16-84%,
                             +-sigma_UV scatter), with the fitted M_UV bins
    galaxy_bias.pdf          galaxy bias b_g in M_UV bins and of cumulative
                             M_UV < M_UV,thresh samples (median, 16-84%),
                             with Shuntov+25's published model values
    convergence.pdf         autocorrelation time vs chain length against the
                             N/50 criterion, plus walker traces
    params.h5                fixed values and their method (Shuntov+25 or
                             --fix), the priors of the free parameters, the
                             number of walkers, the starting walker
                             positions (walker_start) and the fitted data
                             points (data)
    convergence.log          one row per convergence check (every
                             ``--check-every`` steps): tau per parameter,
                             N/(50 tau_max), max dtau/tau, converged

The fit (posterior summary, reduced chi^2 and p-value) and
then the plots are written when the run finishes, or when it is stopped with
Ctrl-C or ``kill <pid>`` (SIGTERM, also sent by SLURM at walltime). In that
case they use the chain saved so far.

Every fit discards burn-in and thins as galfind's ``MCMC_Fitter`` does: with
tau the largest autocorrelation time (N/50 if emcee cannot yet estimate it
reliably), the first 2 tau steps of each walker are dropped and the rest
thinned by tau/2. The corner plot reports N_walkers, N_steps and
N_steps / (50 tau_max), with tau_max emcee's estimate (no fallback). ``kill -9`` cannot be caught: re-run the command to resume and finish.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))

from halogal.inference import (  # noqa: E402
    Parameter, ParameterSet, Uniform,
    ModelParametrization, Dataset, GaussianLikelihood, EmceeSampler,
)
from halogal.inference.parametrization import HOD_PARAM_SPEC  # noqa: E402
from halogal.model import HODModel, Observables  # noqa: E402
from literature_data import (  # noqa: E402
    DEFAULT_LITERATURE, LITERATURE, literature_tag, load_literature_uvlf)
from austin26_data import (  # noqa: E402
    BIN_WIDTH, DATA_FILE, REDSHIFTS, ZBIN_EDGES, load_austin26_uvlf)


# Free parameters (sampling space) and their flat priors.
# Flat priors for every parameter, used whenever it is free. Those fixed by
# default (FIXED_SHUNTOV) are only sampled when freed with ``--free``; their
# ranges are wide and cover the Shuntov+25 values at all EPOCHS-v2 redshifts
# (b 0.22-0.55, logMcut 10.9-15.6, logMsat 12.8-13.1, asat 0.84-1.06).
PRIORS = {
    "eps0": Uniform(1e-6, 100.0),
    "logMc": Uniform(10.0, 14.0),
    "a": Uniform(0.0, 2.0),
    "b": Uniform(0.0, 2.0),
    "sigma_UV": Uniform(1e-3, 10.0),
    "logMcut": Uniform(7.0, 17.0),
    "logMsat": Uniform(10.0, 16.0),
    "asat": Uniform(0.0, 2.0),
}
# Fixed by default, at their Shuntov+25 values at z. On the command line,
# ``--fix NAME`` fixes NAME at its Shuntov+25 value, ``--fix NAME=VALUE`` at
# VALUE, and ``--free NAME`` fits a parameter that is fixed by default.
FIXED_SHUNTOV = ("logMc", "b", "logMcut", "logMsat", "asat")

OUTPUT_ROOT = REPO / "examples" / "outputs" / "EPOCHS_DR2_UVLF"


def output_root(lit_names=()):
    """Output folder named after the fitted data: ``EPOCHS_DR2_UVLF`` (the
    default, :data:`OUTPUT_ROOT`), or with literature sets e.g.
    ``EPOCHS_DR2+Bowler17+Varadaraj23+Varadaraj26_UVLF``."""
    if not lit_names:
        return OUTPUT_ROOT
    return OUTPUT_ROOT.parent / ("EPOCHS_DR2+" + "+".join(
        literature_tag(n) for n in lit_names) + "_UVLF")


def fixed_spec(fix_args=(), free_args=()):
    """Fixed parameters as ``{name: value or None}``, None = Shuntov+25 at z.

    Starts from :data:`FIXED_SHUNTOV`. Each item of ``fix_args`` is ``NAME``
    (fix at the Shuntov+25 value) or ``NAME=VALUE`` (fix at VALUE; masses in
    log10); each name in ``free_args`` is fitted instead. Ordered as in
    ``HOD_PARAM_SPEC``.
    """
    valid = [spec[1] for spec in HOD_PARAM_SPEC]
    spec = {n: None for n in FIXED_SHUNTOV}
    seen = set()
    for item in fix_args:
        name, sep, value = item.partition("=")
        name = name.strip()
        if name not in valid:
            raise SystemExit(f"--fix {item!r}: expected NAME or NAME=VALUE "
                             f"with NAME one of {valid}")
        if name in seen:
            raise SystemExit(f"--fix {name} given more than once")
        seen.add(name)
        if not sep or value.strip().lower() in ("", "shuntov", "shuntov25"):
            spec[name] = None
        else:
            try:
                spec[name] = float(value)
            except ValueError:
                raise SystemExit(f"--fix {item!r}: VALUE must be a number "
                                 "(or omit it for the Shuntov+25 value)")
    for name in free_args:
        if name not in valid:
            raise SystemExit(f"--free {name!r}: NAME must be one of {valid}")
        if name in seen:
            raise SystemExit(f"{name} given to both --fix and --free")
        spec.pop(name, None)
    return {n: spec[n] for n in valid if n in spec}


def fixed_label(spec, source="evo"):
    """Folder name: ``fixed=<p1>+<p2>+..._shuntov25_<source>``.

    The fixed parameter names, then where their Shuntov+25 values come from
    (``--shuntov``: ``evo``, ``z4.3``, ``z5.4`` or ``z7.3``), e.g.
    ``fixed=logMc+b+logMcut+logMsat+asat_shuntov25_z7.3``.
    """
    return ("fixed=" + ("+".join(spec) if spec else "none")
            + f"_shuntov25_{source}")


def _prior_record(prior):
    for attrs in (("low", "high"), ("mu", "sigma")):
        if all(hasattr(prior, a) for a in attrs):
            return {"type": type(prior).__name__,
                    **{a: float(getattr(prior, a)) for a in attrs}}
    return {"type": type(prior).__name__, "repr": repr(prior)}


def params_record(params, spec, z, source="evo"):
    """JSON-ready record of the fixed values and the priors of a run.

    Each fixed parameter has its ``value``, its ``method`` (``"Shuntov25"``:
    the Shuntov+25 linear evolution at z; ``"Shuntov25_z7.3"`` etc.: that
    Shuntov+25 bin's posterior medians; ``"explicit"``: ``--fix``) and a
    readable ``source``.
    """
    import json

    record = {
        "z": z,
        "masses": "log10(M / Msun)",
        "fixed": {
            p.name: {
                "value": float(p.value),
                "method": (shuntov_method(source) if spec[p.name] is None
                           else "explicit"),
                "source": ("--fix" if spec[p.name] is not None
                           else f"Shuntov+25 linear evolution at z={z:g}"
                           if source == "evo"
                           else f"Shuntov+25 z~{source[1:]} bin "
                                "(Table 2 posterior median)"),
            } for p in params.fixed_params},
        "priors": {p.name: _prior_record(p.prior) for p in params.free},
    }
    return json.loads(json.dumps(record))  # normalise as stored


def write_params(path, record, nwalkers=None):
    """Save a :func:`params_record` to HDF5.

    Layout: file attributes ``z``, ``masses`` and ``nwalkers``; a group
    ``fixed/<name>``
    per fixed parameter with attributes ``value``, ``method``, ``source``;
    a group ``priors/<name>`` per free parameter with attributes ``type`` and
    its limits (``low``/``high`` or ``mu``/``sigma``). Any ``walker_start``
    group (see :func:`write_walker_start`) is kept, since a resumed run does
    not re-create the start.
    """
    import h5py

    with h5py.File(path, "a") as f:
        f.attrs["z"] = record["z"]
        f.attrs["masses"] = record["masses"]
        for key in ("model", "z_pivot"):   # joint power-law fits
            if key in record:
                f.attrs[key] = record[key]
        if nwalkers is not None:
            f.attrs["nwalkers"] = int(nwalkers)
        for section in ("fixed", "priors"):
            if section in f:
                del f[section]
            grp = f.create_group(section)
            for name, entry in record[section].items():
                sub = grp.create_group(name)
                for key, val in entry.items():
                    sub.attrs[key] = val


def write_walker_start(path, positions, names, method, seed, scatter=None):
    """Save the starting walker positions of a new chain to the params file.

    Group ``walker_start``: dataset ``positions`` (nwalkers, nfree), columns
    in the order of attribute ``names``; attributes ``nwalkers``, ``method``
    (``prior``/``fiducial``), ``seed`` and, for ``fiducial``, ``scatter``.
    Replaces any earlier start (a new chain was begun).
    """
    import h5py

    with h5py.File(path, "a") as f:
        if "walker_start" in f:
            del f["walker_start"]
        grp = f.create_group("walker_start")
        positions = np.asarray(positions, float)
        grp.create_dataset("positions", data=positions)
        grp.attrs["nwalkers"] = positions.shape[0]
        grp.attrs["names"] = list(names)
        grp.attrs["method"] = method
        grp.attrs["seed"] = int(seed)
        if scatter is not None:
            grp.attrs["scatter"] = float(scatter)


def read_params(path):
    """Read a params record written by :func:`write_params`.

    Also reads the ``fixed_params_z<z>.json`` files of earlier versions.
    """
    import json

    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import h5py

    def plain(v):
        v = v.decode() if isinstance(v, bytes) else v
        return v.item() if hasattr(v, "item") else v

    with h5py.File(path, "r") as f:
        record = {k: plain(v) for k, v in f.attrs.items()}
        for section in ("fixed", "priors"):
            record[section] = {
                name: {k: plain(v) for k, v in f[section][name].attrs.items()}
                for name in f[section]}
    return record


def z_folder(z):
    """Redshift subfolder name, e.g. ``z=7`` or ``z=12.5``."""
    return f"z={z:g}"


def params_files(fixed_dir, z=None):
    """Params records under a ``fixed=...`` folder.

    ``fixed=.../z=<z>/params.h5``, plus the per-z files that earlier versions
    wrote directly in the ``fixed=...`` folder (``params_z<z>.h5``,
    ``fixed_params_z<z>.json``). With ``z``, only that redshift's record.
    """
    if z is not None:
        legacy = f"z{z:g}"
        paths = ([fixed_dir / z_folder(z) / "params.h5"]
                 + sorted(fixed_dir.glob(f"{z_folder(z)}_*/params.h5"))
                 + [fixed_dir / f"params_{legacy}.h5",
                    fixed_dir / f"fixed_params_{legacy}.json"])
        return [p for p in paths if p.exists()][:1]
    return (sorted(fixed_dir.glob("z=*/params.h5"))
            + sorted(fixed_dir.glob("params_z*.h5"))
            + sorted(fixed_dir.glob("fixed_params_z*.json")))


def record_matches(stored, current, same_z):
    """Whether a stored params record is consistent with the current one.

    Always required: identical priors, the same fixed names, and the same
    method per fixed parameter (with identical values for explicit ones).
    For the same z, Shuntov+25 values must also agree; across redshifts they
    legitimately differ, since the Shuntov+25 values evolve with z.
    """
    try:
        if stored["priors"] != current["priors"]:
            return False
        if set(stored["fixed"]) != set(current["fixed"]):
            return False
        for name, cur in current["fixed"].items():
            old = stored["fixed"][name]
            if old["method"] != cur["method"]:
                return False
            if (same_z or cur["method"] == "explicit") and not np.isclose(
                    old["value"], cur["value"], rtol=1e-12, atol=0.0):
                return False
    except (KeyError, TypeError):  # record from an older version
        return False
    return True


def choose_outdir(root, spec, record, z, source="evo"):
    """Find the ``fixed=...`` folder for this run.

    Tries ``root/fixed=...``, then ``..._2``, ``..._3``, ... and returns the
    first whose stored records (``z=*/params.h5``) are consistent with
    ``record`` (see :func:`record_matches`): the same-z record if there is one,
    else every record in the folder. A folder with no records yet is also
    used. If none matches, the first unused suffix is returned (a new run).
    The run itself goes in the ``z=<z>`` subfolder of the returned folder.
    """
    base = fixed_label(spec, source)
    k = 1
    while True:
        outdir = root / (base if k == 1 else f"{base}_{k}")
        if not outdir.exists():
            return outdir, "new"
        same = params_files(outdir, z)
        if same:
            ok = record_matches(read_params(same[0]), record, True)
        else:
            ok = all(record_matches(read_params(p), record, False)
                     for p in params_files(outdir))
        if ok:
            return outdir, "existing"
        print(f"  {outdir.name}: fixed values / priors differ from this run")
        k += 1


LABELS = {
    "eps0": r"$\epsilon_0$",
    "logMc": r"$\log_{10}(M_c/M_\odot)$",
    "a": r"$a$",
    "b": r"$b$",
    "sigma_UV": r"$\sigma_{\rm UV}$",
    "logMcut": r"$\log_{10}(M_{\rm cut}/M_\odot)$",
    "logMsat": r"$\log_{10}(M_{\rm sat}/M_\odot)$",
    "asat": r"$\alpha_{\rm sat}$",
}
# Joint power-law fits (--joint): <x>_0 is the value at the pivot redshift
# z_p, <x>_1 the exponent of (1+z)/(1+z_p); for masses of M itself.
_PL_SYMBOLS = {"eps0": r"\epsilon_0", "logMc": r"M_c", "a": "a", "b": "b",
               "sigma_UV": r"\sigma_{\rm UV}", "logMcut": r"M_{\rm cut}",
               "logMsat": r"M_{\rm sat}", "asat": r"\alpha_{\rm sat}"}
for _n, _sym in _PL_SYMBOLS.items():
    LABELS[f"{_n}_0"] = (rf"$\log_{{10}}({_sym}/M_\odot)\,|_{{z_p}}$"
                         if _n.startswith("log") else rf"${_sym}\,|_{{z_p}}$")
    LABELS[f"{_n}_1"] = rf"${_sym}$ exponent"


# Shuntov et al. 2025 (A&A 699, A231), Table 2 (Appendix B): posterior
# medians of the HOD parameters in each redshift bin (Halpha at z~4.3 and
# z~5.4, [OIII] at z~7.3). Sampling-space names: a = beta, b = gamma,
# asat = alpha; masses are log10(M / Msun). The z~5.4 column equals
# halogal.config.DEFAULT_HOD_PARAMS.
SHUNTOV25_BINS = {
    "z4.3": {"eps0": 0.19, "logMc": 11.48, "a": 0.67, "b": 0.71,
             "sigma_UV": 0.68, "logMcut": 8.61, "logMsat": 12.67,
             "asat": 0.75},
    "z5.4": {"eps0": 0.19, "logMc": 11.64, "a": 0.69, "b": 0.65,
             "sigma_UV": 0.69, "logMcut": 9.57, "logMsat": 12.65,
             "asat": 0.85},
    "z7.3": {"eps0": 0.24, "logMc": 11.94, "a": 0.73, "b": 0.52,
             "sigma_UV": 0.56, "logMcut": 11.40, "logMsat": 12.75,
             "asat": 0.92},
}
SHUNTOV_SOURCES = ("evo",) + tuple(SHUNTOV25_BINS)


def shuntov25_values(z, source="evo"):
    """Shuntov+25 HOD parameters in sampling space (log10 masses).

    ``source="evo"``: their linear redshift evolution
    (``DEFAULT_REDSHIFT_EVOLUTION``) evaluated at ``z``. ``"z4.3"``,
    ``"z5.4"``, ``"z7.3"``: that bin's Table 2 posterior medians, used as
    they are whatever ``z`` is fitted.
    """
    if source in SHUNTOV25_BINS:
        return dict(SHUNTOV25_BINS[source])
    if source != "evo":
        raise ValueError(f"unknown Shuntov+25 source {source!r}")
    model = ModelParametrization("evolution").params_for_redshift({}, z)
    return {sample: (np.log10(model[name]) if is_log else model[name])
            for name, sample, is_log, _, _ in HOD_PARAM_SPEC}


def shuntov_label(source):
    """Short label of a Shuntov+25 source, e.g. ``Shuntov+25 z=7.3``."""
    return ("Shuntov+25" if source == "evo"
            else f"Shuntov+25 z={source[1:]}")


def shuntov_method(source):
    """``method`` recorded in params.h5 for parameters fixed to Shuntov+25
    (plain ``Shuntov25`` for the evolution law, as in earlier records)."""
    return "Shuntov25" if source == "evo" else f"Shuntov25_{source}"


def build_parameters(shuntov, spec=None):
    """Free parameters (all not in ``spec``) with their PRIORS; fixed ones at
    their Shuntov+25 value (``None`` in ``spec``) or the explicit value."""
    spec = fixed_spec() if spec is None else spec
    params = []
    for name, prior in PRIORS.items():
        if name in spec:
            continue
        # Start walkers at the Shuntov+25 value when it lies inside the prior.
        start = shuntov[name] if np.isfinite(prior.logpdf(shuntov[name])) else None
        params.append(Parameter(name, prior, value=start))
    if not params:
        raise SystemExit("Every parameter is fixed: nothing to fit.")
    params += [Parameter(n, fixed=True, value=shuntov[n] if v is None else v)
               for n, v in spec.items()]
    return ParameterSet(params)


def model_uvlf(obs, parametrization, named, z, muv, dmuv=BIN_WIDTH):
    """Model Phi averaged over bins of width ``dmuv`` centred on ``muv``."""
    return obs.luminosity_function(
        muv, dmuv, **parametrization.params_for_redshift(named, z))


# Free parameters shown in log10 in the corner plot and the convergence
# traces (display only: they are sampled, and their priors defined, in linear
# space, and tau is estimated on the sampled values).
LOG10_DISPLAY = {"eps0": r"$\log_{10}\,\epsilon_0$",
                 "sigma_UV": r"$\log_{10}\,\sigma_{\rm UV}$",
                 "eps0_0": r"$\log_{10}\,\epsilon_0\,|_{z_p}$",
                 "sigma_UV_0": r"$\log_{10}\,\sigma_{\rm UV}\,|_{z_p}$"}


def fixed_lines(spec, shuntov, ref):
    """One line per fixed parameter: its value and where it comes from."""
    return [f"{LABELS[n]} = {shuntov[n]:.2f} ({ref})" if v is None
            else f"{LABELS[n]} = {v:g}" for n, v in spec.items()]


def plot_corner(result, truths, path, header, fixed_txt=""):
    """Corner plot with ``truths`` (dict by name; missing = none) marked,
    and ``header``, the chain length and ``fixed_txt`` to the right."""
    from halogal.inference import InferenceResult

    samples = np.array(result.samples, copy=True)
    labels, truth_list = [], []
    for j, n in enumerate(result.names):
        truth = truths.get(n)
        if n in LOG10_DISPLAY:
            samples[:, j] = np.log10(samples[:, j])
            labels.append(LOG10_DISPLAY[n])
            truth = (np.log10(truth) if truth is not None and truth > 0
                     else None)
        else:
            labels.append(LABELS.get(n, n))
        truth_list.append(truth)
    shown = InferenceResult(result.names, samples, weights=result.weights)
    fig = shown.corner_plot(
        labels=labels, truths=truth_list,
        truth_color="tab:red", title_fmt=".2f",
        title_kwargs={"fontsize": 11}, label_kwargs={"fontsize": 13},
    )
    ex = result.extras
    n_steps = ex["n_iterations"]
    # The emcee tau estimate itself (no N/50 fallback, which would make this
    # ratio identically 1): convergence needs N_steps / (50 tau_max) > 1.
    tau_est = float(np.max(ex["autocorr_time"]))
    chain_txt = (
        rf"$N_{{\rm walkers}}$ = {ex['nwalkers']}, "
        rf"$N_{{\rm steps}}$ = {n_steps}" "\n"
        rf"$N_{{\rm steps}}/(50\,\tau_{{\max}})$ = "
        f"{n_steps / (50 * tau_est):.2f}")
    # To the right of the grid, so it never overlaps a panel whatever the
    # number of free parameters (savefig's tight bbox widens the figure).
    fig.text(1.0, 0.98,
             header + "\n\n" + chain_txt
             + ("\n\nfixed:\n" + fixed_txt if fixed_txt else ""),
             fontsize=11, va="top", ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def goodness_of_fit(chi2, n_data, n_free):
    """Reduced chi^2 and the chi^2 p-value for ``n_data - n_free`` dof.

    The p-value is P(chi^2 >= observed) for a chi^2 distribution with
    ``dof`` degrees of freedom. It is approximate here: the split-normal
    chi^2 is only chi^2-distributed when the errors are near-symmetric, and
    the parameters are evaluated at the posterior median, not the best fit.

    ``n_sigma`` is the equivalent two-sided Gaussian significance of the
    p-value (p = 0.0027 <-> 3 sigma), computed from log p so that it stays
    finite for p far below double precision.
    """
    from scipy.special import ndtri_exp
    from scipy.stats import chi2 as chi2_dist

    dof = n_data - n_free
    if dof <= 0:
        return dict(chi2=chi2, dof=dof, chi2_red=np.nan, p_value=np.nan,
                    n_sigma=np.nan)
    log_p = chi2_dist.logsf(chi2, dof)
    return dict(chi2=chi2, dof=dof, chi2_red=chi2 / dof,
                p_value=float(np.exp(log_p)),
                n_sigma=float(-ndtri_exp(log_p - np.log(2.0))))


def plot_observables(like, z):
    """``Observables`` at z on the likelihood's HMF grid.

    The same coarse HMF grid as the likelihood (as GaussianLikelihood seeds
    it), so the curves are the fitted model and ~2x faster to compute.
    """
    obs = Observables(HODModel(z=z))
    if like.hmf_points is not None:
        from halogal.cosmology import get_halo_mass_function
        obs._hmf_cache[z] = get_halo_mass_function(
            z, M_min=6, M_max=15, num_points=like.hmf_points)
    return obs


def posterior_draws(result, n_draws=200, seed=0):
    """``n_draws`` posterior samples (named dicts), drawn without
    replacement. The same ``seed`` gives the same draws in every plot."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(result.samples),
                     size=min(n_draws, len(result.samples)), replace=False)
    return [result.parameter_set.to_dict(result.samples[i]) for i in idx]


def fit_title(result, z, what="EPOCHS-v2 UVLF"):
    zlo, zhi = ZBIN_EDGES[z]
    free_txt = ", ".join(LABELS.get(n, n) for n in result.names)
    return f"{what}, ${zlo:g} < z < {zhi:g}$\nfree: {free_txt}"


def fixed_box(ax, spec, shuntov, ref, x=0.03, y=0.97, ha="left", va="top"):
    """Text box listing the fixed parameters and their values."""
    lines = fixed_lines(spec, shuntov, ref)
    if lines:
        ax.text(x, y, "fixed:\n" + "\n".join(lines), transform=ax.transAxes,
                ha=ha, va=va, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


def plot_uvlf_residuals(result, like, dataset, data_all, shuntov, z, muv_max,
                        path, n_draws=200, seed=0, grid_step=0.1, spec=None,
                        ref="Shuntov+25", n_free=None, lit=()):
    """Data, model band and log residuals of one redshift bin.

    ``n_free`` (default: the free parameters of ``result``) is subtracted
    from the bin's points for the dof; a joint fit passes 0, as its free
    parameters are shared by all bins. ``lit`` is a list of fitted
    literature sets ``(Dataset, data dict)`` at this z: shown with their own
    markers, and included in the residuals and chi^2. The model curves are
    averaged over the EPOCHS-v2 bin width; each set's residuals use the model
    averaged over its own bin widths, as fitted.
    """
    parametrization = like.parametrization
    obs = plot_observables(like, z)

    bright = min([-23.5] + [float(np.min(d["MUV"] - d["dMUV"] / 2)) - 0.2
                            for _, d in lit])
    muv_grid = np.arange(np.floor(bright * 2) / 2, -16.5 + grid_step / 2,
                         grid_step)
    draws = np.array([model_uvlf(obs, parametrization, named, z, muv_grid)
                      for named in posterior_draws(result, n_draws, seed)])
    lo, hi = np.percentile(draws, [16, 84], axis=0)
    median = result.median_params()
    phi_med = model_uvlf(obs, parametrization, median, z, muv_grid)
    phi_shuntov = model_uvlf(obs, parametrization, shuntov, z, muv_grid)

    # Goodness of fit at the posterior median, using the likelihood's own
    # model and split-normal chi^2 (side-selected errors).
    predicted = like.predict_all(median)
    model_fit = predicted[dataset.label]["model"]
    fitted = [dataset] + [ds for ds, _ in lit]
    stats = goodness_of_fit(
        sum(ds.chi2(predicted[ds.label]["model"]) for ds in fitted),
        sum(ds.data.size for ds in fitted),
        len(result.names) if n_free is None else n_free)

    # Points excluded by the magnitude cut (shown, not fitted).
    excl = data_all["MUV"] > muv_max if muv_max is not None else \
        np.zeros(len(data_all["MUV"]), bool)
    model_excl = model_uvlf(obs, parametrization, median, z,
                            data_all["MUV"][excl]) if excl.any() else []

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 7.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    # Model curves are bin-averaged over BIN_WIDTH, like the fitted points.
    ax.fill_between(muv_grid, lo, hi, color="tab:blue", alpha=0.3, lw=0,
                    label="posterior 16-84%")
    ax.plot(muv_grid, phi_med, color="tab:blue", lw=2, label="posterior median")
    ax.plot(muv_grid, phi_shuntov, color="tab:red", ls="--", lw=1.5,
            label=f"{ref} at $z={z:g}$" if ref == "Shuntov+25" else ref)

    fit = ~excl
    ax.errorbar(data_all["MUV"][fit], data_all["phi"][fit],
                yerr=[data_all["phi_err_low"][fit], data_all["phi_err_upp"][fit]],
                fmt="o", color="k", ms=6, capsize=2,
                label="EPOCHS-v2 (fitted)", zorder=5)
    if excl.any():
        ax.errorbar(data_all["MUV"][excl], data_all["phi"][excl],
                    yerr=[data_all["phi_err_low"][excl],
                          data_all["phi_err_upp"][excl]],
                    fmt="o", mfc="none", color="0.5", ms=6, capsize=2,
                    label="EPOCHS-v2 (not fitted)", zorder=5)
    for ds, d in lit:
        ax.errorbar(d["MUV"], d["phi"], yerr=[d["phi_err_low"],
                                               d["phi_err_upp"]],
                    capsize=2, ms=6, mec="k", zorder=6,
                    label=f"{d['name']} ({d['survey']})",
                    **LIT_STYLES.get(d["name"], dict(fmt="s")))
    # Lower error bars reaching zero would vanish on a log axis; clip the view.
    phi_all = np.concatenate([data_all["phi"]] + [d["phi"] for _, d in lit])
    low_all = np.concatenate([data_all["phi_err_low"]]
                             + [d["phi_err_low"] for _, d in lit])
    upp_all = np.concatenate([data_all["phi_err_upp"]]
                             + [d["phi_err_upp"] for _, d in lit])
    ymin = 0.3 * np.min(phi_all - low_all, where=phi_all - low_all > 0,
                        initial=np.min(phi_all))
    ax.set_yscale("log")
    ax.set_ylim(max(ymin, 1e-9), 3 * np.max(phi_all + upp_all))
    ax.set_ylabel(r"$\Phi\ [{\rm Mpc^{-3}\,mag^{-1}}]$")
    ax.set_title(fit_title(result, z), fontsize=11)
    spec = fixed_spec() if spec is None else spec
    fixed_box(ax, spec, shuntov, ref)
    ax.legend(loc="lower right", fontsize=9,
              title=rf"model averaged over $\Delta M_{{\rm UV}}={BIN_WIDTH:g}$",
              title_fontsize=9)

    # log10(Phi_obs / Phi_model) at the posterior median. The error bars are
    # the data errors in log space; a lower bar reaching Phi_obs <= 0 is cut
    # at the bottom of the panel.
    def log_ratio(phi, err_lo, err_hi, model):
        y = np.log10(phi / model)
        lo_edge = phi - err_lo
        y_lo = np.where(lo_edge > 0,
                        np.log10(np.where(lo_edge > 0, lo_edge, 1.0) / model),
                        -np.inf)
        y_hi = np.log10((phi + err_hi) / model)
        return y, y_lo, y_hi

    y_fit, lo_fit, hi_fit = log_ratio(dataset.data, dataset.sigma_low,
                                      dataset.sigma_upp, model_fit)
    parts = [(dataset.x, y_fit, lo_fit, hi_fit,
              dict(fmt="o", color="k", ms=6, capsize=2))]
    if excl.any():
        y_ex, lo_ex, hi_ex = log_ratio(
            data_all["phi"][excl], data_all["phi_err_low"][excl],
            data_all["phi_err_upp"][excl], model_excl)
        parts.append((data_all["MUV"][excl], y_ex, lo_ex, hi_ex,
                      dict(fmt="o", mfc="none", color="0.5", ms=6, capsize=2)))
    for ds, d in lit:
        y_l, lo_l, hi_l = log_ratio(ds.data, ds.sigma_low, ds.sigma_upp,
                                    predicted[ds.label]["model"])
        parts.append((ds.x, y_l, lo_l, hi_l,
                      dict(ms=6, capsize=2, mec="k",
                           **LIT_STYLES.get(d["name"], dict(fmt="s")))))
    finite = np.concatenate([np.concatenate([lo[np.isfinite(lo)], hi])
                             for _, _, lo, hi, _ in parts])
    lim = max(0.5, 1.15 * np.max(np.abs(finite)))
    axr.axhline(0, color="k", lw=0.8)
    for x, y, lo, hi, style in parts:
        lo_clipped = np.where(np.isfinite(lo), lo, -lim)
        axr.errorbar(x, y, yerr=[y - lo_clipped, hi - y], **style)
    axr.set_ylim(-lim, lim)
    axr.grid(axis="y", color="0.9", lw=0.6)
    axr.set_ylabel(r"$\log_{10}(\Phi_{\rm obs}/\Phi_{\rm model})$")
    axr.set_xlabel(r"$M_{\rm UV}$")
    axr.text(0.02, 0.95,
             rf"$\chi^2_\nu = {stats['chi2']:.1f}/{stats['dof']} = "
             rf"{stats['chi2_red']:.2f}$,  $p = {stats['p_value']:.2g}$ "
             rf"(${stats['n_sigma']:.1f}\sigma$)",
             transform=axr.transAxes, ha="left", va="top", fontsize=10,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    for a in (ax, axr):
        if muv_max is not None:
            a.axvline(muv_max + BIN_WIDTH / 2, color="0.4", ls=":", lw=1)
    axr.set_xlim(muv_grid[0], muv_grid[-1])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return stats


# Markers of the literature sets (as in EPOCHS-v2/scripts/hod_param_impact.py).
LIT_STYLES = {
    "Bowler+17": dict(fmt="p", color="tab:green"),
    "Varadaraj+23": dict(fmt="v", color="tab:cyan"),
    "Varadaraj+26": dict(fmt="h", color="tab:purple"),
}


def literature_sets(lit_arg, z, muv_max):
    """Literature UVLF sets to fit at z: ``[(Dataset, data dict), ...]``.

    ``lit_arg`` is ``--lit``: None (no literature), [] (the defaults at z,
    :data:`DEFAULT_LITERATURE`) or a list of names. Sets with no data within
    0.5 of z (or none passing the M_UV cut) are skipped with a note.
    """
    if lit_arg is None:
        return []
    names = (DEFAULT_LITERATURE.get(float(z), ()) if len(lit_arg) == 0
             else lit_arg)
    out = []
    for name in names:
        d = load_literature_uvlf(name, z, muv_max=muv_max)
        if d is None:
            print(f"  {name}: no data at z = {z:g} (within 0.5); not fitted")
            continue
        ds = Dataset.uvlf(z=z, MUV=d["MUV"], phi=d["phi"],
                          sigma_low=d["phi_err_low"],
                          sigma_upp=d["phi_err_upp"], dMUV=d["dMUV"],
                          label=f"{name} z={z:g}")
        print(f"  {name} ({d['survey']}, z = {d['z']:g}): fitting "
              f"{ds.data.size} bins, M_UV = {d['MUV'].min():g} to "
              f"{d['MUV'].max():g}")
        out.append((ds, d))
    if not out:
        print(f"  no literature data at z = {z:g}: fitting EPOCHS-v2 only")
    return out


def literature_line(lit):
    """``\n+ Bowler+17, ...`` for plot headers; empty without literature."""
    names = list(dict.fromkeys(d["name"] for _, d in lit))
    return "\n+ " + ", ".join(names) if names else ""


def literature_settings(lit):
    """Run settings of the literature sets (added only when some are fitted,
    so the settings of EPOCHS-only runs, and their resumes, are unchanged)."""
    import hashlib

    if not lit:
        return {}
    h = hashlib.sha256()
    for ds, _ in lit:
        for arr in (np.atleast_1d(ds.z), ds.x, ds.data, ds.sigma_low,
                    ds.sigma_upp, np.broadcast_to(np.asarray(ds.dMUV, float),
                                                  ds.x.shape)):
            h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    return {"literature": [ds.label for ds, _ in lit],
            "literature_sha256": h.hexdigest()}


def write_literature(path, lit):
    """Save the fitted literature points to the params file: group
    ``literature/<set label>`` with datasets ``MUV``, ``phi``, ``phi_err_low``,
    ``phi_err_upp``, ``dMUV`` and attributes ``name``, ``survey``, ``z``
    (fit) and ``z_table``. The group is removed when none are fitted."""
    import h5py

    with h5py.File(path, "a") as f:
        if "literature" in f:
            del f["literature"]
        if not lit:
            return
        grp = f.create_group("literature")
        for ds, d in lit:
            sub = grp.create_group(ds.label)
            for k in DATA_ARRAYS + ("dMUV",):
                sub.create_dataset(k, data=np.asarray(d[k], float))
            sub.attrs.update(name=d["name"], survey=d["survey"],
                             z=float(ds.z), z_table=d["z"])


# Halo-mass grid of the SFE-Mhalo and M_UV-Mhalo plots, log10(Mh / Msun).
LOG10_MH_GRID = np.linspace(8.0, 14.0, 241)
SHUNTOV25_BIAS_FILE = REPO / "examples" / "shuntov25_data" / \
    "Galaxy-bias_vs_z.ecsv"


def ref_label(ref, z):
    return f"{ref} at $z={z:g}$" if ref == "Shuntov+25" else ref


def set_model(obs, parametrization, named, z):
    """Set the HOD model of ``obs`` to the named (sampling-space) parameters
    at z, and return it."""
    obs.hod_model.update_parameters(
        **parametrization.params_for_redshift(named, z))
    return obs.hod_model


def halo_relation(result, like, shuntov, z, func, n_draws=200, seed=0):
    """``func(model, Mh)`` on :data:`LOG10_MH_GRID` for the posterior.

    Returns the 16th/84th percentiles over ``n_draws`` posterior draws (the
    same draws as the UVLF band), and the curves at the posterior median and
    at the Shuntov+25 reference values.
    """
    obs = plot_observables(like, z)
    par = like.parametrization
    Mh = 10**LOG10_MH_GRID
    draws = np.array([func(set_model(obs, par, named, z), Mh)
                      for named in posterior_draws(result, n_draws, seed)])
    lo, hi = np.percentile(draws, [16, 84], axis=0)
    return dict(lo=lo, hi=hi,
                median=func(set_model(obs, par, result.median_params(), z), Mh),
                ref=func(set_model(obs, par, shuntov, z), Mh))


def fitted_muv_range(dataset):
    """Brightest and faintest edge of the fitted M_UV bins of a Dataset, or
    of a list of them (EPOCHS-v2 and any literature sets)."""
    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    lo, hi = [], []
    for d in datasets:
        half = np.broadcast_to(np.asarray(d.dMUV, float), d.x.shape) / 2
        lo.append(np.min(d.x - half))
        hi.append(np.max(d.x + half))
    return min(lo), max(hi)


def fitted_mh_range(muv_of_mh, dataset):
    """log10 Mh range over which the (median) mean M_UV(Mh) lies within the
    fitted M_UV bins, or None. Approximate: scatter (sigma_UV) spreads each
    magnitude over a range of halo masses."""
    bright, faint = fitted_muv_range(dataset)
    inside = (muv_of_mh >= bright) & (muv_of_mh <= faint)
    if not inside.any():
        return None
    return LOG10_MH_GRID[inside].min(), LOG10_MH_GRID[inside].max()


def plot_sfe_mhalo(result, like, dataset, shuntov, z, path, n_draws=200,
                   seed=0, spec=None, ref="Shuntov+25"):
    """Star-formation efficiency eps(Mh) = SFR / (f_b dMh/dt): posterior
    median and 16-84% band, the reference curve, and the halo-mass range the
    fitted UVLF bins map onto."""
    sfe = halo_relation(result, like, shuntov, z,
                        lambda m, Mh: m.star_formation_efficiency(Mh),
                        n_draws, seed)
    muv = halo_relation(result, like, shuntov, z,
                        lambda m, Mh: m.MUV(Mh), n_draws=1, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    mh_fit = fitted_mh_range(muv["median"], dataset)
    if mh_fit is not None:
        ax.axvspan(*mh_fit, color="0.92", lw=0, zorder=0,
                   label="fitted $M_{\\rm UV}$ bins (median relation)")
    ax.fill_between(LOG10_MH_GRID, sfe["lo"], sfe["hi"], color="tab:blue",
                    alpha=0.3, lw=0, label="posterior 16-84%")
    ax.plot(LOG10_MH_GRID, sfe["median"], color="tab:blue", lw=2,
            label="posterior median")
    ax.plot(LOG10_MH_GRID, sfe["ref"], color="tab:red", ls="--", lw=1.5,
            label=ref_label(ref, z))
    ax.set_yscale("log")
    ax.set_ylim(max(1e-4, 0.5 * np.min(sfe["median"])),
                2 * max(np.max(sfe["hi"]), np.max(sfe["ref"])))
    ax.set_xlim(LOG10_MH_GRID[0], LOG10_MH_GRID[-1])
    ax.set_xlabel(r"$\log_{10}(M_{\rm h}/M_\odot)$")
    ax.set_ylabel(r"$\epsilon(M_{\rm h}) = {\rm SFR}\,/\,"
                  r"(f_{\rm b}\,\dot M_{\rm h})$")
    ax.set_title(fit_title(result, z, "SFE from the EPOCHS-v2 UVLF fit"),
                 fontsize=11)
    fixed_box(ax, fixed_spec() if spec is None else spec, shuntov, ref)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_muv_mhalo(result, like, dataset, shuntov, z, path, n_draws=200,
                   seed=0, spec=None, ref="Shuntov+25"):
    """Mean M_UV(Mh) of centrals (dust attenuated, as fitted): posterior
    median and 16-84% band, +-sigma_UV about the median (the intrinsic
    log-normal scatter of the HOD), the reference curve, and the fitted M_UV
    bins."""
    muv = halo_relation(result, like, shuntov, z,
                        lambda m, Mh: m.MUV(Mh), n_draws, seed)
    sigma = float(like.parametrization.params_for_redshift(
        result.median_params(), z)["sigma_UV"])   # free or fixed

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bright, faint = fitted_muv_range(dataset)
    ax.axhspan(bright, faint, color="0.92", lw=0, zorder=0,
               label=r"fitted $M_{\rm UV}$ bins")
    for sign in (-1, 1):
        ax.plot(LOG10_MH_GRID, muv["median"] + sign * sigma,
                color="tab:blue", ls=":", lw=1.2,
                label=(rf"median $\pm\,\sigma_{{\rm UV}}$ "
                       rf"($\sigma_{{\rm UV}}={sigma:.2f}$, scatter)"
                       if sign > 0 else None))
    ax.fill_between(LOG10_MH_GRID, muv["lo"], muv["hi"], color="tab:blue",
                    alpha=0.3, lw=0, label="posterior 16-84%")
    ax.plot(LOG10_MH_GRID, muv["median"], color="tab:blue", lw=2,
            label="posterior median")
    ax.plot(LOG10_MH_GRID, muv["ref"], color="tab:red", ls="--", lw=1.5,
            label=ref_label(ref, z))
    ax.set_xlim(LOG10_MH_GRID[0], LOG10_MH_GRID[-1])
    curves = np.concatenate([muv["lo"], muv["hi"], muv["ref"]])
    ax.set_ylim(min(np.max(curves), faint + 4) + 0.5,
                min(np.min(curves), bright) - 0.5)   # brighter up
    ax.set_xlabel(r"$\log_{10}(M_{\rm h}/M_\odot)$")
    ax.set_ylabel(r"$\langle M_{\rm UV}\rangle(M_{\rm h})$ (dust attenuated)")
    ax.set_title(fit_title(result, z, r"$M_{\rm UV}$-$M_{\rm h}$ from the "
                           "EPOCHS-v2 UVLF fit"), fontsize=11)
    fixed_box(ax, fixed_spec() if spec is None else spec, shuntov, ref)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def shuntov25_threshold_bias(z):
    """Shuntov+25's published model bias of M_UV < thresh samples at z
    (interpolated in their bias-vs-z table): ``(thresholds, bias)``, or None
    if z is outside the table."""
    from astropy.table import Table

    if not SHUNTOV25_BIAS_FILE.exists():
        return None
    tab = Table.read(SHUNTOV25_BIAS_FILE, format="ascii.ecsv")
    zs = np.asarray(tab["Redshift"], float)
    if not zs.min() <= z <= zs.max():
        return None
    cols = [c for c in tab.colnames if c.startswith("Bias_MUV")]
    thresh = np.array([float(c[len("Bias_MUV"):]) for c in cols])
    bias = np.array([np.interp(z, zs, np.asarray(tab[c], float))
                     for c in cols])
    return thresh, bias


def plot_galaxy_bias(result, like, dataset, shuntov, z, muv_max, path,
                     n_draws=200, seed=0, grid_step=0.1, spec=None,
                     ref="Shuntov+25"):
    """Large-scale galaxy bias predicted by the fit (Tinker+10 halo bias).

    Left: b_g in M_UV bins of the data bin width (as the UVLF is averaged).
    Right: b_g of the cumulative M_UV < M_UV,thresh samples that clustering
    measurements use, with Shuntov+25's published model values at z.
    """
    par = like.parametrization
    obs = plot_observables(like, z)
    draws = posterior_draws(result, n_draws, seed)
    median = result.median_params()
    muv_grid = np.arange(-23.5, -16.5 + grid_step / 2, grid_step)
    thresh_grid = np.arange(-23.0, -16.0 + grid_step / 2, grid_step)

    def binned(named):
        b = obs.galaxy_bias(muv_grid, BIN_WIDTH,
                            **par.params_for_redshift(named, z))
        return np.where(b > 0, b, np.nan)   # 0 = no galaxies in the bin

    def cumulative(named):
        p = par.params_for_redshift(named, z)
        return np.array([obs.mean_bias(t, **p) for t in thresh_grid])

    panels = []
    for grid, func in ((muv_grid, binned), (thresh_grid, cumulative)):
        curves = np.array([func(d) for d in draws])
        lo, hi = np.nanpercentile(curves, [16, 84], axis=0)
        panels.append((grid, lo, hi, func(median), func(shuntov)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    for ax, (grid, lo, hi, med, refc) in zip(axes, panels):
        ax.fill_between(grid, lo, hi, color="tab:blue", alpha=0.3, lw=0,
                        label="posterior 16-84%")
        ax.plot(grid, med, color="tab:blue", lw=2, label="posterior median")
        ax.plot(grid, refc, color="tab:red", ls="--", lw=1.5,
                label=ref_label(ref, z))
        if muv_max is not None:
            ax.axvline(muv_max + BIN_WIDTH / 2, color="0.4", ls=":", lw=1,
                       label="fit cut")
        ax.set_xlim(grid[0], grid[-1])
        ax.grid(color="0.93", lw=0.6)
    published = shuntov25_threshold_bias(z)
    if published is not None:
        axes[1].plot(*published, "D", color="tab:red", mfc="white", ms=7,
                     zorder=5, label="Shuntov+25 published\n"
                                     f"(z-evolution model) at $z={z:g}$")
    axes[0].set_xlabel(r"$M_{\rm UV}$ (bin of width "
                       rf"$\Delta M_{{\rm UV}}={BIN_WIDTH:g}$)")
    axes[1].set_xlabel(r"$M_{\rm UV,thresh}$  (sample $M_{\rm UV} < "
                       r"M_{\rm UV,thresh}$)")
    axes[0].set_ylabel(r"galaxy bias $b_{\rm g}$")
    axes[0].set_title(r"$b_{\rm g}(M_{\rm UV})$, binned", fontsize=11)
    axes[1].set_title(r"$b_{\rm g}(<M_{\rm UV,thresh})$, cumulative",
                      fontsize=11)
    finite = np.concatenate([np.ravel(p[1:]) for p in panels])
    finite = finite[np.isfinite(finite)]
    axes[0].set_ylim(0.9 * finite.min(), 1.1 * finite.max())
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    # Right of the panels, clear of the curves and published points.
    fixed_box(axes[1], fixed_spec() if spec is None else spec, shuntov, ref,
              x=1.03, y=1.0)
    fig.suptitle(fit_title(result, z, "Galaxy bias from the EPOCHS-v2 UVLF "
                           "fit"), fontsize=11, y=1.03)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(source, names, burnin, path, n_tau=50,
                     tau_change=0.01, check_every=50, n_points=30):
    """How close the chain is to the sampler's convergence criterion.

    Left: integrated autocorrelation time tau(N) per parameter, estimated
    from the first N steps of every walker, against the N / n_tau line.
    The criterion (``EmceeSampler``) is met once every tau lies below that
    line AND tau changed by < ``tau_change`` since the previous check
    ``check_every`` steps earlier. Right: walker traces per parameter and of
    log-probability, with the burn-in discarded from the posterior marked.

    ``source`` is an emcee HDF checkpoint path, or anything with emcee's
    ``get_chain()`` / ``get_log_prob()`` (a live sampler or backend).

    Returns
    -------
    dict
        ``n_steps``, ``tau`` (at the full length), ``length_ratio`` (=
        N / (n_tau * max tau); >= 1 needed), ``dtau`` (max fractional change
        since the previous check, NaN if the chain is shorter than
        ``check_every``) and ``converged``.
    """
    import emcee

    if isinstance(source, (str, Path)):
        source = emcee.backends.HDFBackend(str(source), read_only=True)
    chain = source.get_chain()             # (N, nwalkers, ndim)
    log_prob = source.get_log_prob()       # (N, nwalkers)
    n_steps = chain.shape[0]

    def tau_at(n):
        return emcee.autocorr.integrated_time(chain[:n], tol=0)

    n_min = min(n_steps, 50)
    ns = np.unique(np.geomspace(n_min, n_steps, n_points).astype(int))
    taus = np.array([tau_at(n) for n in ns])
    tau = taus[-1]
    length_ratio = n_steps / (n_tau * np.max(tau))
    if n_steps > check_every:
        tau_prev = tau_at(n_steps - check_every)
        dtau = float(np.max(np.abs(tau_prev - tau) / tau))
    else:
        dtau = np.nan
    converged = bool(length_ratio > 1 and dtau < tau_change)

    ndim = len(names)
    fig = plt.figure(figsize=(14, 2.0 * (ndim + 1) + 0.5))
    gs = fig.add_gridspec(ndim + 1, 2, width_ratios=[1.1, 1.4],
                          hspace=0.08, wspace=0.25)

    ax = fig.add_subplot(gs[:, 0])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, name in enumerate(names):
        ax.plot(ns, taus[:, j], "o-", ms=3, color=colors[j % len(colors)],
                label=LABELS.get(name, name))
    n_line = np.geomspace(ns[0], max(ns[-1], n_tau * np.max(tau)) * 1.5, 50)
    ax.plot(n_line, n_line / n_tau, "k--", lw=1.2,
            label=rf"$N/{n_tau}$ (need $\tau$ below)")
    ax.fill_between(n_line, 0, n_line / n_tau, color="tab:green", alpha=0.08,
                    lw=0)
    ax.axvline(n_steps, color="0.4", ls=":", lw=1)
    if length_ratio < 1:
        need = n_tau * np.max(tau)
        ax.axvline(need, color="tab:red", ls=":", lw=1)
        ax.text(need, np.min(taus) * 0.9,
                rf" $\gtrsim{need:.0f}$ steps needed", color="tab:red",
                rotation=90, va="bottom", ha="right", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("chain length per walker, $N$ [steps]")
    ax.set_ylabel(r"integrated autocorrelation time $\hat\tau$ [steps]")
    ax.legend(loc="upper left", fontsize=9)
    status = "CONVERGED" if converged else "NOT CONVERGED"
    dtau_txt = (f"{100 * dtau:.1f}% (need < {100 * tau_change:g}%)"
                if np.isfinite(dtau) else f"n/a (N < {check_every})")
    ax.text(0.97, 0.03,
            f"{status}\n"
            f"N = {n_steps} steps x {chain.shape[1]} walkers\n"
            rf"$N\,/\,({n_tau}\,\tau_{{\max}})$ = {length_ratio:.2f} (need > 1)"
            f"\n" rf"$\Delta\tau/\tau$ over last {check_every} steps = "
            + dtau_txt,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="white",
                      ec="tab:green" if converged else "tab:red"))
    ax.set_title(r"$\hat\tau$ is unreliable while $\hat\tau > N/50$",
                 fontsize=10)

    steps = np.arange(1, n_steps + 1)
    traces = [np.log10(chain[:, :, j]) if n in LOG10_DISPLAY
              else chain[:, :, j] for j, n in enumerate(names)] + [log_prob]
    ylabels = [LOG10_DISPLAY.get(n, LABELS.get(n, n)) for n in names] \
        + [r"$\ln p$"]
    for k, (trace, ylab) in enumerate(zip(traces, ylabels)):
        axt = fig.add_subplot(gs[k, 1])
        axt.plot(steps, trace, color="k", alpha=max(0.05, 3 / trace.shape[1]),
                 lw=0.5)
        if k == ndim:  # log-prob: zoom past the initial transient
            lo = np.percentile(trace[n_steps // 10:], 1)
            if np.max(trace) > lo:
                axt.set_ylim(lo, np.max(trace) + 0.1 * (np.max(trace) - lo))
        axt.axvspan(0, burnin, color="0.8", alpha=0.5, lw=0)
        axt.set_ylabel(ylab)
        axt.set_xlim(0, n_steps)
        if k < ndim:
            axt.tick_params(labelbottom=False)
        else:
            axt.set_xlabel("step")
        if k == 0:
            axt.set_title(f"walker traces (grey: burn-in, {burnin} steps, "
                          "discarded)", fontsize=10)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return dict(n_steps=n_steps, tau=tau, length_ratio=length_ratio,
                dtau=dtau, converged=converged)


PLOT_KINDS = ("corner", "uvlf_residuals", "sfe_mhalo", "muv_mhalo",
              "galaxy_bias", "convergence")


def make_plots(result, burnin, chain_source, ctx, n_draws, grid_step=0.1):
    """Write the corner, UVLF + residuals, SFE-Mhalo, M_UV-Mhalo, galaxy
    bias and convergence plots (:data:`PLOT_KINDS`).

    Returns the goodness-of-fit and convergence summaries.
    """
    a = ctx["args"]
    ref = ctx["ref"]
    lit = ctx.get("lit", [])
    plot_corner(result, ctx["shuntov"], a.outdir / "corner.pdf",
                header=(f"EPOCHS-v2 UVLF, $z = {a.z:g}$"
                        + literature_line(lit) + f"\nred: {ref}"
                        + (f" at $z = {a.z:g}$" if ref == "Shuntov+25"
                           else "")),
                fixed_txt="\n".join(fixed_lines(ctx["spec"], ctx["shuntov"],
                                                ref)))
    gof = plot_uvlf_residuals(
        result, ctx["like"], ctx["dataset"], ctx["data_all"], ctx["shuntov"],
        a.z, a.muv_max, a.outdir / "uvlf_residuals.pdf",
        n_draws=n_draws, seed=a.seed, grid_step=grid_step, spec=ctx["spec"],
        ref=ctx["ref"], lit=lit)
    common = dict(n_draws=n_draws, seed=a.seed, spec=ctx["spec"],
                  ref=ctx["ref"])
    fitted = [ctx["dataset"]] + [ds for ds, _ in lit]
    plot_sfe_mhalo(result, ctx["like"], fitted, ctx["shuntov"], a.z,
                   a.outdir / "sfe_mhalo.pdf", **common)
    plot_muv_mhalo(result, ctx["like"], fitted, ctx["shuntov"], a.z,
                   a.outdir / "muv_mhalo.pdf", **common)
    plot_galaxy_bias(result, ctx["like"], ctx["dataset"], ctx["shuntov"],
                     a.z, a.muv_max, a.outdir / "galaxy_bias.pdf",
                     grid_step=grid_step, **common)
    conv = plot_convergence(chain_source, result.names, burnin,
                            a.outdir / "convergence.pdf",
                            check_every=a.check_every)
    return gof, conv


class ConvergenceLog:
    """``EmceeSampler.run`` callback: append a convergence row to a text log.

    Called every ``--check-every`` steps, on the same steps as the sampler's
    own convergence check, and applies the same test: chain > n_tau * tau for
    every parameter AND max |delta tau / tau| < tau_change since the previous
    check. A row is written (and flushed) at every check, so the file can be
    followed with ``tail -f`` during a run. On a resumed run the first check
    has no previous tau, so delta tau is NaN there, as in the sampler.
    """

    def __init__(self, path, names, check_every, start_step, n_tau=50,
                 tau_change=0.01, start_note=""):
        import time
        self.path = Path(path)
        self.n_tau, self.tau_change = n_tau, tau_change
        self.prev_tau = None
        new = not self.path.exists() or self.path.stat().st_size == 0
        cols = (["time", "step"] + [f"tau_{n}" for n in names]
                + [f"N/({n_tau}tau_max)", "max_dtau/tau", "converged"])
        with open(self.path, "a") as fh:
            if new:
                fh.write(f"# fit_uvlf.py convergence log. Converged when "
                         f"N > {n_tau} tau for all parameters AND "
                         f"max|dtau/tau| < {tau_change:g} between checks "
                         f"{check_every} steps apart.\n")
                fh.write("# " + " ".join(cols) + "\n")
            fh.write(f"# run started {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                     f"at step {start_step}, checks every {check_every} "
                     f"steps{start_note}\n")

    def __call__(self, sampler):
        import time
        n = sampler.iteration
        tau = sampler.get_autocorr_time(tol=0)
        if np.all(np.isfinite(tau)):
            ratio = n / (self.n_tau * np.max(tau))
            dtau = (np.max(np.abs(self.prev_tau - tau) / tau)
                    if self.prev_tau is not None else np.nan)
            converged = bool(np.all(self.n_tau * tau < n)
                             and dtau < self.tau_change)
            self.prev_tau = tau
        else:  # as in the sampler, a non-finite tau is skipped
            ratio, dtau, converged = np.nan, np.nan, False
        row = ([time.strftime("%Y-%m-%dT%H:%M:%S"), str(n)]
               + [f"{t:.2f}" for t in tau]
               + [f"{ratio:.3f}", f"{dtau:.4f}", str(converged)])
        with open(self.path, "a") as fh:
            fh.write(" ".join(row) + "\n")


def galfind_burn_thin(backend):
    """Burn-in and thinning as in galfind's ``MCMC_Fitter``.

    ``tau`` is the maximum over parameters of emcee's autocorrelation time,
    using emcee's strict estimate (which refuses a chain shorter than 50 tau).
    If that fails, galfind falls back to ``tau = N / 50``. The first
    ``int(2 * tau)`` steps of every walker are discarded and the rest thinned
    by ``int(tau / 2)`` (at least 1).

    Returns ``(tau_max, reliable, discard, thin)``.
    """
    n = backend.iteration
    try:
        tau_max = float(np.nanmax(backend.get_autocorr_time()))
        reliable = True
    except Exception:  # emcee.autocorr.AutocorrError: chain < 50 tau
        tau_max = n / 50
        reliable = False
    discard = min(int(2 * tau_max), max(n - 1, 0))
    thin = max(int(tau_max / 2), 1)
    return tau_max, reliable, discard, thin


def result_from_checkpoint(path, params, converged=False):
    """Posterior from the emcee HDF checkpoint, burnt-in and thinned as in
    galfind (:func:`galfind_burn_thin`). Used for every fit: after a finished
    run, an interrupt, or when the checkpoint is already complete. Returns
    None if no step was saved.
    """
    import emcee
    from halogal.inference import InferenceResult

    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    n = backend.iteration
    if n == 0:
        return None
    tau_max, reliable, discard, thin = galfind_burn_thin(backend)
    extras = {
        "autocorr_time": backend.get_autocorr_time(tol=0),  # per parameter
        "tau_max": tau_max,
        "tau_reliable": reliable,
        "acceptance_fraction": float(np.mean(backend.accepted / n)),
        "log_prob": backend.get_log_prob(discard=discard, thin=thin,
                                         flat=True),
        "burnin": discard,
        "thin": thin,
        "converged": bool(converged),
        "n_iterations": int(n),
        "nwalkers": int(backend.shape[0]),
    }
    return InferenceResult(
        names=params.names,
        samples=backend.get_chain(discard=discard, thin=thin, flat=True),
        parameter_set=params, extras=extras, backend="emcee")


SETTINGS_ATTR = "fit_uvlf_settings"      # JSON of run_settings()
CONVERGED_ATTR = "fit_uvlf_converged_at"  # step at which the run converged


def run_settings(args, params, dataset):
    """Everything that defines the posterior being sampled.

    A checkpoint is only resumed if these match exactly. Changing any of them
    for the same z must start a new chain (another ``--outdir``), or the old
    and new posteriors would be mixed. ``--seed``, ``--nsteps``,
    ``--check-every`` and ``--processes`` are free to change on a resume.
    """
    import hashlib
    import json

    h = hashlib.sha256()
    for arr in (dataset.x, dataset.data, dataset.sigma_low, dataset.sigma_upp):
        h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    settings = {
        "z": args.z,
        "muv_max": args.muv_max,
        "dMUV": dataset.dMUV,
        "hmf_points": args.hmf_points,
        "nwalkers": args.nwalkers,
        "free_priors": {p.name: repr(p.prior) for p in params.free},
        "fixed": {p.name: float(p.value) for p in params.fixed_params},
        "data_sha256": h.hexdigest(),
    }
    return json.loads(json.dumps(settings))  # normalise as stored


def prepare_checkpoint(path, settings, ndim):
    """Create or validate the checkpoint; return (steps done, converged).

    A new checkpoint is initialised and stamped with ``settings`` (as a
    file-level HDF5 attribute, which emcee's ``reset`` leaves alone). An
    existing one is resumed only if its stamped settings match, otherwise
    the script exits listing the differences. A checkpoint from before
    settings were stamped is adopted with a warning.
    """
    import json
    import h5py
    import emcee

    def stamp():
        with h5py.File(path, "a") as f:
            f.attrs[SETTINGS_ATTR] = json.dumps(settings)
            if CONVERGED_ATTR in f.attrs:
                del f.attrs[CONVERGED_ATTR]

    def fresh():
        emcee.backends.HDFBackend(str(path)).reset(settings["nwalkers"], ndim)
        stamp()
        return 0, False

    if not path.exists():
        return fresh()
    try:
        n = emcee.backends.HDFBackend(str(path), read_only=True).iteration
    except (KeyError, OSError):
        n = 0
    if n == 0:
        return fresh()

    with h5py.File(path, "r") as f:
        stored = f.attrs.get(SETTINGS_ATTR)
        converged_at = f.attrs.get(CONVERGED_ATTR)
    if stored is None:
        warnings.warn(
            f"{path} has {n} steps but no stored run settings (it predates "
            "this check). Adopting the current settings: make sure they are "
            "the ones it was run with, or delete it to start afresh.",
            RuntimeWarning)
        stamp()
        return n, False

    stored = json.loads(stored)
    diffs = [k for k in sorted(set(stored) | set(settings))
             if stored.get(k) != settings.get(k)]
    if diffs:
        lines = "\n".join(f"  {k}: checkpoint {stored.get(k)!r}  vs  now "
                          f"{settings.get(k)!r}" for k in diffs)
        raise SystemExit(
            f"Refusing to resume {path}: it was run with different "
            f"settings:\n{lines}\nUse another --outdir for these settings, "
            "or delete the checkpoint to start afresh.")
    converged = converged_at is not None and int(converged_at) == n
    return n, converged


INIT_ATTR = "fit_uvlf_init"  # how the walkers of a new chain were started


def initial_positions(mode, params, like, nwalkers, seed, scatter=1e-3,
                      max_tries=100):
    """Starting positions ``(nwalkers, ndim)`` for a NEW chain.

    ``"prior"``: independent draws from each free parameter's prior (uniform
    over the prior range for the default priors). Draws where the model gives
    a non-finite log-probability are redrawn, so no walker starts stuck.
    ``"fiducial"``: a Gaussian ball around the Shuntov+25 values (prior
    midpoint if outside the prior), of fractional width ``scatter``, clipped
    into the prior (``ParameterSet.initial``).
    """
    rng = np.random.default_rng(seed)
    if mode == "fiducial":
        return params.initial(nwalkers, rng=rng, scatter=scatter)
    if mode != "prior":
        raise ValueError(f"unknown init mode {mode!r}")

    def log_post(theta):
        lp = params.log_prior(theta)
        return lp + like.log_likelihood(params.to_dict(theta)) \
            if np.isfinite(lp) else -np.inf

    pos = np.empty((nwalkers, params.ndim))
    for i in range(nwalkers):
        for _ in range(max_tries):
            theta = np.array([p.prior.sample(rng) for p in params.free])
            if np.isfinite(log_post(theta)):
                pos[i] = theta
                break
        else:
            raise RuntimeError(
                f"no finite log-probability in {max_tries} prior draws for "
                f"walker {i}; check the priors")
    return pos


DATA_ARRAYS = ("MUV", "phi", "phi_err_low", "phi_err_upp")


def data_record(data_fit, muv_max):
    """The fitted UVLF points of a run (as passed to ``Dataset.uvlf``)."""
    rec = {k: np.asarray(data_fit[k], dtype=float) for k in DATA_ARRAYS}
    rec["dMUV"] = float(data_fit["dMUV"])
    rec["muv_max"] = np.nan if muv_max is None else float(muv_max)
    rec["source"] = DATA_FILE.name
    return rec


def write_data(path, rec):
    """Save the fitted points to the params file (group ``data``).

    Datasets ``MUV``, ``phi``, ``phi_err_low``, ``phi_err_upp`` (one entry
    per fitted bin); attributes ``dMUV`` (bin width), ``muv_max`` (the cut;
    NaN = none), ``source`` (data file) and ``n_points``.
    """
    import h5py

    with h5py.File(path, "a") as f:
        if "data" in f:
            del f["data"]
        grp = f.create_group("data")
        for k in DATA_ARRAYS + ("z",):
            if k in rec:   # z: redshift of each point, joint fits only
                grp.create_dataset(k, data=rec[k])
        grp.attrs["dMUV"] = rec["dMUV"]
        grp.attrs["muv_max"] = rec["muv_max"]
        grp.attrs["source"] = rec["source"]
        grp.attrs["n_points"] = len(rec["MUV"])


def read_data(path):
    """The ``data`` group of a params file, or None if absent."""
    import h5py

    if not Path(path).exists():
        return None
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return None
        grp = f["data"]
        rec = {k: grp[k][()] for k in DATA_ARRAYS + ("z",) if k in grp}
        rec["dMUV"] = float(grp.attrs["dMUV"])
        rec["muv_max"] = float(grp.attrs["muv_max"])
    return rec


def data_matches(stored, current):
    """Same fitted points, bin width and cut (NaN cuts compare equal)."""
    keys = DATA_ARRAYS + ("z",)
    if any((k in stored) != (k in current)
           or (k in stored and not np.array_equal(stored[k], current[k]))
           for k in keys):
        return False
    return (stored["dMUV"] == current["dMUV"]
            and (stored["muv_max"] == current["muv_max"]
                 or (np.isnan(stored["muv_max"])
                     and np.isnan(current["muv_max"]))))


def checkpoint_state(path):
    """``(steps, stored settings or None)`` of an emcee checkpoint."""
    import json
    import h5py
    import emcee

    try:
        n = emcee.backends.HDFBackend(str(path), read_only=True).iteration
    except (KeyError, OSError):
        n = 0
    with h5py.File(path, "r") as f:
        stored = f.attrs.get(SETTINGS_ATTR)
    return n, (json.loads(stored) if stored is not None else None)


def choose_run_dir(fixed_dir, base, settings, data):
    """Run folder within ``fixed_dir`` for these settings and data.

    Tries ``base`` (e.g. ``z=7``, or ``z=7+8+9`` for a joint fit), then
    ``<base>_2``, ``<base>_3``, ... and skips a folder
    whose ``params.h5`` holds different fitted data (:func:`data_matches`),
    or whose chain was run with different settings (:func:`run_settings`).
    It returns the first remaining folder, which either has no chain yet or
    a matching one (or one predating stored settings). So different data, a
    different M_UV cut, HMF grid or number of walkers each get their own
    folder instead of being refused.
    """
    k = 1
    while True:
        run_dir = fixed_dir / (base if k == 1 else f"{base}_{k}")
        stored_data = read_data(run_dir / "params.h5")
        if stored_data is not None and not data_matches(stored_data, data):
            print(f"  {run_dir.name}: fitted data differ")
            k += 1
            continue
        ck = run_dir / "chain.h5"
        if not ck.exists():
            return run_dir, "new" if not run_dir.exists() else "existing"
        n, stored = checkpoint_state(ck)
        if n == 0 or stored is None or stored == settings:
            return run_dir, "existing"
        diff = sorted(k_ for k_ in set(stored) | set(settings)
                      if stored.get(k_) != settings.get(k_))
        print(f"  {run_dir.name}: run settings differ ({', '.join(diff)})")
        k += 1


# ---------------------------------------------------------------------------
# Joint fit of several redshift bins with power-law evolution (--joint)
# ---------------------------------------------------------------------------
# Each free parameter A evolves as A(z) = A_0 ((1+z)/(1+z_p))^A_1, with the
# pivot z_p = --z-pivot (default 7, the lowest EPOCHS-v2 bin). A_0, the value
# at z_p, has the single-z prior of A (PRIORS); the exponent A_1 has
# EXPONENT_PRIOR. For masses, log10 M(z) = logM_0 + logM_1 log10((1+z)/(1+z_p)).
# Fixed parameters follow the Shuntov+25 linear evolution (--shuntov evo), or
# are constant in z (a --shuntov bin, or --fix NAME=VALUE). --no-evolve NAME
# keeps a free parameter constant in z (exponent fixed at 0).
EXPONENT_PRIOR = Uniform(-5.0, 5.0)
JOINT_DIR = "joint_powerlaw"   # subfolder of output_root() for joint fits
JOINT_PLOT_KINDS = ("uvlf_residuals", "sfe_mhalo", "muv_mhalo", "galaxy_bias")


def shuntov_linear_coeffs(source):
    """Shuntov+25 as linear-law coefficients ``{d_<x>_dz: .., C_<x>: ..}``:
    their evolution (``evo``), or a bin's values held constant in z."""
    from halogal.config import DEFAULT_REDSHIFT_EVOLUTION

    if source == "evo":
        return dict(DEFAULT_REDSHIFT_EVOLUTION)
    vals = shuntov25_values(0.0, source)
    out = {}
    for _, sample, _, slope, offset in HOD_PARAM_SPEC:
        out[slope], out[offset] = 0.0, vals[sample]
    return out


def joint_reference(z, source):
    """Shuntov+25 reference for a joint fit, at z: their sampling-space
    values (used by the plot labels and fixed boxes) together with their
    linear-law coefficients. In ``powerlaw`` mode a parameter without
    power-law coefficients follows the linear law, so this dict makes the
    joint model reproduce Shuntov+25 exactly."""
    return {**shuntov25_values(z, source), **shuntov_linear_coeffs(source)}


def shuntov_powerlaw(source, zs, z_pivot):
    """Power-law approximation of Shuntov+25 over the fitted redshifts:
    ``{<x>_0, <x>_1}`` per parameter. ``<x>_0`` is their value at the pivot;
    ``<x>_1`` the least-squares slope of log A (log M for masses) against
    log10((1+z)/(1+z_p)) through the pivot (0 if A <= 0 somewhere). Used as
    the corner-plot reference and the 'fiducial' walker start."""
    x = np.log10((1 + np.asarray(zs, float)) / (1 + z_pivot))
    at_pivot = shuntov25_values(z_pivot, source)
    out = {}
    for _, sample, is_log, _, _ in HOD_PARAM_SPEC:
        vals = np.array([shuntov25_values(z, source)[sample] for z in zs])
        amp = at_pivot[sample]
        if is_log:
            y = vals - amp
        elif np.all(vals > 0) and amp > 0:
            y = np.log10(vals / amp)
        else:
            y = np.zeros_like(x)
        out[f"{sample}_0"] = amp
        out[f"{sample}_1"] = (float(np.sum(x * y) / np.sum(x * x))
                              if np.any(x != 0) else 0.0)
    return out


def build_joint_parameters(spec, source, zs, z_pivot, no_evolve=()):
    """ParameterSet of a joint power-law fit, and the role of each name.

    Free parameters (not in ``spec``): ``<x>_0`` with prior PRIORS[x] and
    ``<x>_1`` with EXPONENT_PRIOR (fixed at 0 if in ``no_evolve``). Fixed
    ones: their linear-law coefficients, the Shuntov+25 ones (``None`` in
    ``spec``) or slope 0 and the explicit value. Walkers start at
    :func:`shuntov_powerlaw` where inside the prior.

    Returns ``(ParameterSet, roles)``, with ``roles[name] = (hod_name,
    method)`` for every fixed name (method as recorded in params.h5).
    """
    start = shuntov_powerlaw(source, zs, z_pivot)
    linear = shuntov_linear_coeffs(source)
    params, roles = [], {}
    for _, sample, _, slope, offset in HOD_PARAM_SPEC:
        if sample in spec:
            value = spec[sample]
            if value is None:
                method = shuntov_method(source)
                coeffs = (linear[slope], linear[offset])
            else:
                method, coeffs = "explicit", (0.0, value)
            for name, v in zip((slope, offset), coeffs):
                params.append(Parameter(name, fixed=True, value=v))
                roles[name] = (sample, method)
            continue
        amp, exp = f"{sample}_0", f"{sample}_1"
        for name, prior in ((amp, PRIORS[sample]), (exp, EXPONENT_PRIOR)):
            if name == exp and sample in no_evolve:
                params.append(Parameter(name, fixed=True, value=0.0))
                roles[name] = (sample, "no_evolve")
                continue
            ok = np.isfinite(prior.logpdf(start[name]))
            params.append(Parameter(name, prior,
                                    value=start[name] if ok else None))
    if not params or all(p.fixed for p in params):
        raise SystemExit("Every parameter is fixed: nothing to fit.")
    return ParameterSet(params), roles


def joint_params_record(params, roles, zs, z_pivot, source):
    """JSON-ready record of a joint fit's fixed values and priors (layout of
    :func:`params_record`, plus ``model`` and ``z_pivot``)."""
    import json

    def where(name):
        hod, method = roles[name]
        if method == "explicit":
            return f"--fix {hod} (constant in z)"
        if method == "no_evolve":
            return f"--no-evolve {hod} (exponent 0)"
        return ("Shuntov+25 linear evolution" if source == "evo"
                else f"Shuntov+25 z~{source[1:]} bin, constant in z")

    record = {
        "z": [float(z) for z in zs],
        "model": "powerlaw",
        "z_pivot": float(z_pivot),
        "masses": "log10(M / Msun)",
        "fixed": {p.name: {"value": float(p.value),
                           "method": roles[p.name][1],
                           "source": where(p.name)}
                  for p in params.fixed_params},
        "priors": {p.name: _prior_record(p.prior) for p in params.free},
    }
    return json.loads(json.dumps(record))


def joint_run_settings(args, params, datasets):
    """:func:`run_settings` of a joint fit: everything defining its
    posterior (redshifts, pivot, cut, grid, walkers, priors, fixed values,
    and a hash of the fitted data of every bin)."""
    import hashlib
    import json

    h = hashlib.sha256()
    for d in datasets:
        for arr in (np.atleast_1d(d.z), d.x, d.data, d.sigma_low,
                    d.sigma_upp):
            h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    settings = {
        "model": "powerlaw",
        "z": [float(z) for z in args.redshifts],
        "z_pivot": args.z_pivot,
        "muv_max": args.muv_max,
        "dMUV": datasets[0].dMUV,
        "hmf_points": args.hmf_points,
        "nwalkers": args.nwalkers,
        "free_priors": {p.name: repr(p.prior) for p in params.free},
        "fixed": {p.name: float(p.value) for p in params.fixed_params},
        "data_sha256": h.hexdigest(),
    }
    return json.loads(json.dumps(settings))


def joint_fixed_lines(spec, source, ref, no_evolve):
    """Fixed parameters of a joint fit, one line each, for the plots."""
    lines = [(f"{LABELS[n]}: {ref} z-evolution" if source == "evo"
              else f"{LABELS[n]} = {shuntov25_values(0, source)[n]:.2f} "
                   f"({ref})") if v is None
             else f"{LABELS[n]} = {v:g}" for n, v in spec.items()]
    lines += [f"{LABELS[n]}: free, constant in z" for n in no_evolve]
    return lines


def plot_params_vs_z(result, like, zs, source, ref, path, n_draws=200,
                     seed=0, gof=None):
    """Each power-law parameter against z: posterior median and 16-84% band
    over the fitted range (extended by 0.5 either side), the Shuntov+25
    evolution, the fitted bin centres and the pivot, with the exponent's
    posterior quoted per panel."""
    par = like.parametrization
    model_of = {sample: (model, is_log)
                for model, sample, is_log, _, _ in HOD_PARAM_SPEC}
    shown = [sample for _, sample, *_ in HOD_PARAM_SPEC
             if f"{sample}_0" in result.parameter_set.names
             or f"{sample}_1" in result.parameter_set.names]
    z_grid = np.linspace(min(zs) - 0.5, max(zs) + 0.5, 56)

    def curves(named):
        rows = [par.params_for_redshift(named, z) for z in z_grid]
        out = {}
        for sample in shown:
            model, is_log = model_of[sample]
            v = np.array([r[model] for r in rows])
            out[sample] = np.log10(v) if is_log else v
        return out

    draws = [curves(d) for d in posterior_draws(result, n_draws, seed)]
    med = curves(result.median_params())
    refc = curves(joint_reference(zs[0], source))
    quant = result.parameter_set.names

    ncol = min(3, len(shown))
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                             squeeze=False)
    for ax, sample in zip(axes.flat, shown):
        lo, hi = np.percentile([d[sample] for d in draws], [16, 84], axis=0)
        ax.fill_between(z_grid, lo, hi, color="tab:blue", alpha=0.3, lw=0,
                        label="posterior 16-84%")
        ax.plot(z_grid, med[sample], color="tab:blue", lw=2,
                label="posterior median")
        ax.plot(z_grid, refc[sample], color="tab:red", ls="--", lw=1.5,
                label=ref)
        for z in zs:
            ax.axvline(z, color="0.85", lw=0.8, zorder=0)
        ax.axvline(par.z_pivot, color="0.4", ls=":", lw=1,
                   label=f"pivot $z_p={par.z_pivot:g}$")
        exp = f"{sample}_1"
        if exp in result.names:
            j = result.names.index(exp)
            q16, q50, q84 = np.percentile(result.samples[:, j], [16, 50, 84])
            txt = (rf"exponent $= {q50:.2f}^{{+{q84 - q50:.2f}}}"
                   rf"_{{-{q50 - q16:.2f}}}$")
        else:
            txt = "constant in z"
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right",
                va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_ylabel(LABELS[sample])
        if sample in LOG10_DISPLAY:   # eps0, sigma_UV: log axis, as corner
            ax.set_yscale("log")
        ax.set_xlabel("$z$")
        ax.set_xlim(z_grid[0], z_grid[-1])
    for ax in list(axes.flat)[len(shown):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower left", fontsize=8)
    title = ("EPOCHS-v2 UVLF joint fit, $z = "
             + ", ".join(f"{z:g}" for z in zs) + "$: "
             r"$A(z) = A(z_p)\,[(1+z)/(1+z_p)]^{\rm exponent}$")
    if gof is not None:
        title += ("\n" + rf"$\chi^2/{{\rm dof}} = {gof['chi2']:.1f}/{gof['dof']}"
                  rf" = {gof['chi2_red']:.2f}$, $p = {gof['p_value']:.2g}$"
                  " (all bins, posterior median)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def joint_plot_paths(outdir, zs):
    return ([outdir / f"{k}.pdf" for k in ("corner", "params_vs_z",
                                           "convergence")]
            + [outdir / z_folder(z) / f"{k}.pdf"
               for z in zs for k in JOINT_PLOT_KINDS])


def make_joint_plots(result, burnin, chain_source, ctx):
    """Corner, parameters-vs-z and convergence plots in the run folder, and
    the single-z plots (UVLF + residuals, SFE, M_UV-Mh, bias) of each bin
    in its ``z=<z>`` subfolder. Returns the convergence summary."""
    a, like, zs = ctx["args"], ctx["like"], ctx["zs"]
    ref, spec, source = ctx["ref"], ctx["spec"], a.shuntov
    truths = shuntov_powerlaw(source, zs, a.z_pivot)
    plot_corner(
        result, truths, a.outdir / "corner.pdf",
        header=("EPOCHS-v2 UVLF, joint fit\n$z = "
                + ", ".join(f"{z:g}" for z in zs) + "$"
                + literature_line([x for z in zs for x in ctx["lit"][z]])
                + "\n" + rf"power law in $(1+z)/(1+z_p)$, $z_p = {a.z_pivot:g}$"
                f"\nred: {ref} (power-law\napproximation over these z)"),
        fixed_txt="\n".join(joint_fixed_lines(spec, source, ref,
                                              a.no_evolve)))
    gof = goodness_of_fit(like.chi2(result.median_params()), like.ndata,
                          len(result.names))
    plot_params_vs_z(result, like, zs, source, ref,
                     a.outdir / "params_vs_z.pdf", n_draws=a.plot_draws,
                     seed=a.seed, gof=gof)
    common = dict(n_draws=a.plot_draws, seed=a.seed, spec=spec, ref=ref)
    for z, dataset in zip(zs, ctx["datasets"]):
        print(f"  plots of z = {z:g}", flush=True)
        zdir = a.outdir / z_folder(z)
        zdir.mkdir(exist_ok=True)
        refz = joint_reference(z, source)
        lit = ctx["lit"][z]
        fitted = [dataset] + [ds for ds, _ in lit]
        plot_uvlf_residuals(result, like, dataset, ctx["data_all"][z], refz,
                            z, a.muv_max, zdir / "uvlf_residuals.pdf",
                            n_free=0, lit=lit, **common)
        plot_sfe_mhalo(result, like, fitted, refz, z,
                       zdir / "sfe_mhalo.pdf", **common)
        plot_muv_mhalo(result, like, fitted, refz, z,
                       zdir / "muv_mhalo.pdf", **common)
        plot_galaxy_bias(result, like, dataset, refz, z, a.muv_max,
                         zdir / "galaxy_bias.pdf", **common)
    return plot_convergence(chain_source, result.names, burnin,
                            a.outdir / "convergence.pdf",
                            check_every=a.check_every)


def run_joint(args):
    """Fit all ``args.redshifts`` bins at once with power-law evolution.

    Same sampling, checkpointing, resuming and outputs as
    :func:`run_redshift`, in ``output_root()/joint_powerlaw/
    fixed=..._shuntov25_<src>
    [_const=..]/z=<z1>+<z2>+...[_n]/``.
    """
    zs = args.redshifts
    spec = fixed_spec(args.fix, args.free)
    bad = [n for n in args.no_evolve if n in spec or n not in PRIORS]
    if bad:
        raise SystemExit(f"--no-evolve {', '.join(bad)}: only free "
                         "parameters can be held constant in z")
    ref = shuntov_label(args.shuntov)
    print(f"Joint fit of z = {', '.join(f'{z:g}' for z in zs)}: free "
          "parameters evolve as A(z) = A(z_p) [(1+z)/(1+z_p)]^A_1, "
          f"z_p = {args.z_pivot:g}")

    datasets, data_all, fitted = [], {}, []
    for z in zs:
        data_all[z] = load_austin26_uvlf(z)
        d = load_austin26_uvlf(z, muv_max=args.muv_max)
        fitted.append(d)
        datasets.append(Dataset.uvlf(
            z=z, MUV=d["MUV"], phi=d["phi"], sigma_low=d["phi_err_low"],
            sigma_upp=d["phi_err_upp"], dMUV=d["dMUV"],
            label=f"EPOCHS-v2 UVLF z={z:g}"))
        print(f"  z = {z:g}: fitting {datasets[-1].data.size} of "
              f"{len(data_all[z]['MUV'])} bins (M_UV <= {args.muv_max})")
    lit = {z: literature_sets(args.lit, z, args.muv_max) for z in zs}
    lit_all = [item for z in zs for item in lit[z]]

    params, roles = build_joint_parameters(spec, args.shuntov, zs,
                                           args.z_pivot, args.no_evolve)
    print("Free: " + ", ".join(params.names))
    record = joint_params_record(params, roles, zs, args.z_pivot,
                                 args.shuntov)
    settings = {**joint_run_settings(args, params, datasets),
                **literature_settings(lit_all)}
    data_rec = {k: np.concatenate([np.asarray(d[k], float) for d in fitted])
                for k in DATA_ARRAYS}
    data_rec["z"] = np.concatenate([np.full(len(d["MUV"]), z)
                                    for z, d in zip(zs, fitted)])
    data_rec.update(dMUV=float(BIN_WIDTH), source=DATA_FILE.name,
                    muv_max=(np.nan if args.muv_max is None
                             else float(args.muv_max)))
    if args.outdir is None:
        label = fixed_label(spec, args.shuntov) + (
            f"_const={'+'.join(args.no_evolve)}" if args.no_evolve else "")
        root = output_root(list(dict.fromkeys(d["name"]
                                              for _, d in lit_all)))
        fixed_dir = root / JOINT_DIR / label
        base = "z=" + "+".join(f"{z:g}" for z in zs)
        print(f"Choosing the output folder in {fixed_dir}:")
        args.outdir, how = choose_run_dir(fixed_dir, base, settings,
                                          data_rec)
        print(f"  using {args.outdir.name} ({how})")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {args.outdir}")
    params_path = args.outdir / "params.h5"
    stored = read_data(params_path)
    if stored is not None and not data_matches(stored, data_rec):
        raise SystemExit(f"{params_path} holds different fitted data from "
                         "this run; use another --outdir.")
    write_params(params_path, record, nwalkers=args.nwalkers)
    write_data(params_path, data_rec)
    write_literature(params_path, lit_all)

    like = GaussianLikelihood(
        datasets + [ds for ds, _ in lit_all],
        ModelParametrization("powerlaw", params, z_pivot=args.z_pivot),
        hmf_points=args.hmf_points)
    ctx = dict(args=args, like=like, zs=zs, ref=ref, spec=spec,
               datasets=datasets, data_all=data_all, lit=lit)
    return run_chain(
        args, params, like, settings, ref, joint_reference(zs[0],
                                                           args.shuntov),
        plot_paths=joint_plot_paths(args.outdir, zs),
        plot_fn=lambda result, burnin, ck: make_joint_plots(
            result, burnin, ck, ctx))


MOVES_HELP = {
    "de": "differential evolution: 80% DEMove + 20% DESnookerMove",
    "stretch": "emcee's default stretch move",
}


def make_moves(name):
    """emcee proposal for ``--moves``.

    ``"de"``: emcee's recommended mix of differential-evolution moves, which
    adapt to correlated, elongated posteriors (such as the eps0 - Mc - a
    degeneracy of a UVLF fit) much better than the stretch move.
    ``"stretch"``: emcee's default affine-invariant stretch move.
    The proposal changes only how efficiently the posterior is explored, not
    the posterior itself, so it may differ between sessions of one chain.
    """
    import emcee

    if name == "de":
        return [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    if name == "stretch":
        return emcee.moves.StretchMove()
    raise ValueError(f"unknown moves {name!r}")


def mark_converged(path, n):
    import h5py
    with h5py.File(path, "a") as f:
        f.attrs[CONVERGED_ATTR] = int(n)


def _sigterm_to_keyboardinterrupt():
    """Make ``kill <pid>`` (SIGTERM, also sent by SLURM at walltime) behave
    like Ctrl-C in this process, so the fit and plots are still written.

    Pool workers forked later inherit the handler; there it restores the
    default and re-raises, so ``Pool.terminate()`` still kills them.
    """
    import os
    import signal

    main_pid = os.getpid()

    def handler(signum, frame):
        if os.getpid() != main_pid:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, handler)


def parse_muv_max(text):
    return None if text.lower() == "none" else float(text)


def parse_redshifts(values):
    """``--z`` values -> sorted unique EPOCHS-v2 bin centres ("all" = every
    bin)."""
    zs = []
    for v in values:
        if v.lower() == "all":
            zs.extend(REDSHIFTS)
            continue
        try:
            z = float(v)
        except ValueError:
            raise SystemExit(f"--z {v!r}: expected a redshift or 'all'")
        if not any(np.isclose(z, zz) for zz in REDSHIFTS):
            raise SystemExit(f"--z {v}: not an EPOCHS-v2 bin; choose from "
                             f"{REDSHIFTS} or 'all'")
        zs.append(z)
    return sorted(set(zs))


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", nargs="+", default=["7"], metavar="Z",
                    help="EPOCHS-v2 redshift-bin centre(s), fitted one after "
                         f"another: any of {', '.join(f'{z:g}' for z in REDSHIFTS)}"
                         ", or 'all' (default 7)")
    ap.add_argument("--muv-max", type=parse_muv_max, default=-18.0,
                    help="fit only bins with M_UV <= this; 'none' fits all "
                         "(default -18, the EPOCHS-v2 fiducial cut)")
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=50000,
                    help="cap on the TOTAL steps per walker, including any "
                         "already in a resumed checkpoint (with "
                         "--no-converge: the total length)")
    ap.add_argument("--no-converge", action="store_true",
                    help="run exactly --nsteps instead of stopping on the "
                         "autocorrelation criterion")
    ap.add_argument("--check-every", type=int, default=50,
                    help="steps between convergence checks")
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--init", choices=("prior", "fiducial"), default="prior",
                    help="walker start for a NEW chain: 'prior' (default) "
                         "draws each walker from the priors; 'fiducial' is a "
                         "Gaussian ball around the Shuntov+25 values (width "
                         "--init-scatter). Ignored when resuming.")
    ap.add_argument("--init-scatter", type=float, default=1e-3,
                    help="fractional Gaussian width of the 'fiducial' start "
                         "(default 1e-3)")
    ap.add_argument("--moves", choices=tuple(MOVES_HELP), default="de",
                    help="emcee proposal: 'de' (default; "
                         f"{MOVES_HELP['de'].replace('%', '%%')}) or 'stretch' "
                         f"({MOVES_HELP['stretch'].replace('%', '%%')}). May be changed on a "
                         "resume.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the walker start (new chains only)")
    ap.add_argument("--hmf-points", type=int, default=256,
                    help="HMF mass-grid points for the UVLF integral")
    ap.add_argument("--replot", action="store_true",
                    help="remake the plots of an already converged chain "
                         "(by default they are only made if missing)")
    ap.add_argument("--plot-draws", type=int, default=200,
                    help="posterior draws for the UVLF model band")
    ap.add_argument("--fix", action="append", default=[],
                    metavar="NAME[=VALUE]",
                    help="fix NAME (eps0, logMc, a, b, sigma_UV, logMcut, "
                         "logMsat, asat; masses in log10): at its Shuntov+25 "
                         "value at z if no VALUE is given, else at VALUE. "
                         f"Default fixed set: {', '.join(FIXED_SHUNTOV)} "
                         "(Shuntov+25). Repeatable")
    ap.add_argument("--shuntov", choices=SHUNTOV_SOURCES, default="evo",
                    help="where the Shuntov+25 values (fixed parameters, "
                         "'fiducial' walker start, plot reference) come "
                         "from: 'evo' (default) = their linear "
                         "redshift evolution at the fitted z; 'z4.3', "
                         "'z5.4', 'z7.3' = that bin's Table 2 posterior "
                         "medians, whatever z is fitted")
    ap.add_argument("--free", action="append", default=[], metavar="NAME",
                    help="fit NAME although it is fixed by default (flat "
                         "prior from PRIORS). Repeatable")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="run folder (with several --z: parent of one "
                         "z=<z> folder each); default: the matching "
                         "examples/outputs/EPOCHS_DR2[+<--lit sets>]_UVLF/"
                         "fixed=<p1>+<p2>..._shuntov25_<shuntov>[_n]/"
                         "z=<z>[_n] (see "
                         "above); "
                         "giving --outdir skips that search")
    ap.add_argument("--joint", action="store_true",
                    help="fit all --z bins at once, each free parameter A "
                         "evolving as A(z) = A(z_p) [(1+z)/(1+z_p)]^A_1 "
                         "(with one bin: the single-z fit, no evolution)")
    ap.add_argument("--z-pivot", type=float, default=7.0,
                    help="--joint: pivot redshift z_p where each power law "
                         "is normalised (default 7)")
    ap.add_argument("--no-evolve", action="append", default=[],
                    metavar="NAME",
                    help="--joint: keep free parameter NAME constant in z "
                         "(exponent fixed at 0). Repeatable")
    ap.add_argument("--lit", nargs="*", default=None, metavar="NAME",
                    help="also fit literature UVLFs: with no NAME, the "
                         "defaults of each bin ("
                         + "; ".join(f"z={z:g}: {', '.join(n)}"
                                     for z, n in DEFAULT_LITERATURE.items())
                         + "); or the named sets, any of "
                         + ", ".join(LITERATURE)
                         + ". Default: EPOCHS-v2 only")
    args = ap.parse_args()
    args.redshifts = parse_redshifts(args.z)
    if args.lit:
        unknown = [n for n in args.lit if n not in LITERATURE]
        if unknown:
            ap.error(f"--lit {', '.join(unknown)}: unknown; choose from "
                     f"{', '.join(LITERATURE)}")
        args.lit = list(dict.fromkeys(args.lit))
    if args.joint and len(args.redshifts) < 2:
        # One bin: nothing to evolve. Run the ordinary single-z fit.
        print(f"--joint with one bin (z = {args.redshifts[0]:g}): fitting it "
              "without redshift evolution (the single-z fit)"
              + ("; --no-evolve ignored" if args.no_evolve else ""))
        args.joint, args.no_evolve = False, []
    if args.no_evolve and not args.joint:
        ap.error("--no-evolve only applies to --joint fits")
    return args


def run_redshift(args):
    """Fit one redshift bin, ``args.z``, writing the fit and plots.

    Returns how the posterior was obtained: "run" (sampled now),
    "interrupted" (Ctrl-C / SIGTERM: the fit and plots are still written),
    "converged" or "cap" (complete checkpoint, nothing sampled), or "empty"
    (interrupted before any step was saved).
    """
    z = args.z
    spec = fixed_spec(args.fix, args.free)

    shuntov = shuntov25_values(z, args.shuntov)
    ref = shuntov_label(args.shuntov)
    print(f"{ref} parameters" + (f" at z={z:g}:" if args.shuntov ==
                                 "evo" else " (Table 2 medians):"))
    for name, val in shuntov.items():
        if name not in spec:
            state = "free (start)"
        elif spec[name] is None:
            state = "fixed"
        else:
            state = f"NOT USED: fixed to {spec[name]:g}"
        print(f"  {name:>9s} = {val:.4f}  [{state}]")

    data_all = load_austin26_uvlf(z)
    data_fit = load_austin26_uvlf(z, muv_max=args.muv_max)
    dataset = Dataset.uvlf(
        z=z, MUV=data_fit["MUV"], phi=data_fit["phi"],
        sigma_low=data_fit["phi_err_low"], sigma_upp=data_fit["phi_err_upp"],
        dMUV=data_fit["dMUV"], label=f"EPOCHS-v2 UVLF z={z:g}")
    print(f"Fitting {dataset.data.size} of {len(data_all['MUV'])} UVLF bins "
          f"(M_UV <= {args.muv_max}), model bin-averaged over "
          f"dMUV = {dataset.dMUV:g} mag")
    lit = literature_sets(args.lit, z, args.muv_max)

    params = build_parameters(shuntov, spec)
    record = params_record(params, spec, z, args.shuntov)
    settings = {**run_settings(args, params, dataset),
                **literature_settings(lit)}
    data_rec = data_record(data_fit, args.muv_max)
    if args.outdir is None:
        root = output_root([d["name"] for _, d in lit])
        print(f"Choosing the output folder in {root}:")
        fixed_dir, how = choose_outdir(root, spec, record, z,
                                       args.shuntov)
        print(f"  using {fixed_dir.name} ({how})")
        args.outdir, how = choose_run_dir(fixed_dir, z_folder(z), settings,
                                          data_rec)
        print(f"  using {args.outdir.name} ({how})")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {args.outdir}")
    params_path = args.outdir / "params.h5"
    stored_data = read_data(params_path)
    if stored_data is not None and not data_matches(stored_data, data_rec):
        raise SystemExit(
            f"{params_path} holds different fitted data from this run; use "
            "another --outdir (or omit it to choose a folder automatically).")
    write_params(params_path, record, nwalkers=args.nwalkers)
    write_data(params_path, data_rec)
    write_literature(params_path, lit)
    print(f"Fixed parameter values, priors, nwalkers and the "
          f"{len(data_rec['MUV'])} fitted data points saved to {params_path}"
          + (" (literature points: group literature)" if lit else ""))
    like = GaussianLikelihood(
        [dataset] + [ds for ds, _ in lit],
        ModelParametrization("fixed_z", params), hmf_points=args.hmf_points)
    ctx = dict(args=args, shuntov=shuntov, ref=ref, like=like, spec=spec,
               dataset=dataset, data_all=data_all, params=params, lit=lit)
    return run_chain(
        args, params, like, settings, ref, shuntov,
        plot_paths=[args.outdir / f"{kind}.pdf" for kind in PLOT_KINDS],
        plot_fn=lambda result, burnin, ck: make_plots(
            result, burnin, ck, ctx, n_draws=args.plot_draws)[1])


def run_chain(args, params, like, settings, ref, ref_named, plot_paths,
              plot_fn):
    """Sample (or resume) the chain in ``args.outdir``, then write the fit
    and plots. Shared by the single-redshift and joint fits.

    ``ref_named`` is the reference (Shuntov+25) parameter dict whose
    log-likelihood is printed; ``plot_fn(result, burnin, checkpoint)`` makes
    the plots (``plot_paths``) and returns :func:`plot_convergence`'s summary.
    Returns how the posterior was obtained (see :func:`run_redshift`).
    """
    print(f"Log-likelihood at {ref} values:", like.check(ref_named))
    params_path = args.outdir / "params.h5"
    converge = not args.no_converge
    checkpoint = args.outdir / "chain.h5"
    start_step, already_converged = prepare_checkpoint(
        checkpoint, settings, params.ndim)
    remaining = args.nsteps - start_step
    if start_step:
        print(f"Resuming {checkpoint} at step {start_step} (settings match).")

    # How the posterior is obtained: "run" (sampled now), "interrupted",
    # "converged" (checkpoint already converged) or "cap" (already at
    # --nsteps); the last two re-make the fit and plots without sampling.
    if already_converged:
        source = "converged"
        print("Checkpoint already converged: making the fit from it without "
              "sampling.")
    elif remaining <= 0:
        source = "cap"
        print(f"Checkpoint already has {start_step} >= --nsteps "
              f"{args.nsteps} steps: making the fit and plots from it "
              "without sampling (raise --nsteps to continue).")
    else:
        source = "run"

    if source == "run":
        import h5py
        if start_step == 0:
            init_desc = (f"fiducial (Gaussian, fractional scatter "
                         f"{args.init_scatter:g}, about {ref})"
                         if args.init == "fiducial" else "prior (draws from "
                         "the priors)")
            print(f"Starting {args.nwalkers} walkers: {init_desc}")
            initial = initial_positions(args.init, params, like, args.nwalkers,
                                        args.seed, scatter=args.init_scatter)
            write_walker_start(
                params_path, initial, params.names, args.init, args.seed,
                scatter=args.init_scatter if args.init == "fiducial" else None)
            print(f"Starting walker positions saved to {params_path} "
                  "(group walker_start)")
            with h5py.File(checkpoint, "a") as f:
                f.attrs[INIT_ATTR] = init_desc
            start_note = f"; walkers started from {init_desc}"
        else:
            initial = None
            with h5py.File(checkpoint, "r") as f:
                init_desc = f.attrs.get(INIT_ATTR, "unknown")
            print(f"Resuming: walkers continue from step {start_step} (they "
                  f"were started from {init_desc}; --init is ignored)")
            start_note = ""
        start_note += f"; moves: {MOVES_HELP[args.moves]}"
        conv_log_path = args.outdir / "convergence.log"
        conv_log = ConvergenceLog(conv_log_path, params.names,
                                  args.check_every, start_step,
                                  start_note=start_note)
        print(f"Running emcee: nwalkers={args.nwalkers}, up to {remaining} "
              f"more steps (total cap {args.nsteps}"
              f"{', stopping at convergence' if converge else ''}), "
              f"processes={args.processes}, moves: {MOVES_HELP[args.moves]}, "
              f"checkpoint {checkpoint}")
        print(f"Convergence is logged every {args.check_every} steps to "
              f"{conv_log_path}. The fit and plots are made at the end, or "
              "after Ctrl-C / kill (SIGTERM).")
        try:
            result = EmceeSampler(params, like).run(
                nwalkers=args.nwalkers, nsteps=remaining, max_steps=remaining,
                convergence=converge, check_every=args.check_every,
                processes=args.processes if args.processes > 1 else None,
                seed=args.seed, progress=True, backend=str(checkpoint),
                initial=initial, moves=make_moves(args.moves),
                callback=conv_log, callback_every=args.check_every)
            run_converged = result.extras["converged"]
            if run_converged:
                mark_converged(checkpoint, result.extras["n_iterations"])
        except KeyboardInterrupt:
            source = "interrupted"
            print(f"\nInterrupted: building the fit from the chain saved so "
                  f"far in {checkpoint}")
    # Every fit is built from the checkpoint the same way: burn-in 2 tau and
    # thinning tau/2, as in galfind.
    converged = (source == "converged"
                 or (source == "run" and run_converged))
    result = result_from_checkpoint(checkpoint, params, converged=converged)
    if result is None:
        print("No steps were saved; nothing to fit or plot.")
        return "empty"

    # ---- the fit ---------------------------------------------------------
    ex = result.extras
    print(result)
    print(f"walkers: {ex['nwalkers']} | steps: {ex['n_iterations']} | "
          f"acceptance: {np.mean(ex['acceptance_fraction']):.3f} | "
          f"tau: {np.round(ex['autocorr_time'], 1)}")
    print(f"galfind burn-in/thinning: tau_max = {ex['tau_max']:.1f}"
          f"{'' if ex['tau_reliable'] else ' (fallback N/50: chain < 50 tau)'}"
          f", discard 2 tau = {ex['burnin']} steps, thin tau/2 = {ex['thin']}"
          f" -> {len(result.samples)} posterior samples")
    if source == "interrupted":
        warnings.warn(
            f"Run interrupted after {ex['n_iterations']} steps, before "
            "convergence; the fit uses the chain so far (burn-in and "
            "thinning as in galfind). Re-run the same command to resume.",
            RuntimeWarning)
    elif source == "cap" or (converge and not ex["converged"]):
        warnings.warn(
            f"Chain is at the {args.nsteps}-step cap WITHOUT having met the "
            "autocorrelation convergence criterion; raise --nsteps (the run "
            "resumes from the checkpoint).", RuntimeWarning)
    elif ex["converged"]:
        print("Converged (autocorrelation criterion met).")

    gof = goodness_of_fit(like.chi2(result.median_params()),
                          like.ndata, len(result.names))
    print(f"At the posterior median: chi2 = {gof['chi2']:.2f} for "
          f"{like.ndata} points - {len(result.names)} free = "
          f"{gof['dof']} dof; reduced chi2 = {gof['chi2_red']:.2f}; "
          f"p-value = {gof['p_value']:.3g} ({gof['n_sigma']:.1f} sigma)")

    # ---- the plots -------------------------------------------------------
    if (source == "converged" and not args.replot
            and all(p.exists() for p in plot_paths)):
        print("Checkpoint already converged and its plots exist: not "
              "remaking them (use --replot to force).")
        return source
    print("Making plots...", flush=True)
    conv = plot_fn(result, ex["burnin"], checkpoint)
    print(f"Convergence: N/(50 tau_max) = {conv['length_ratio']:.2f} "
          f"(need > 1), max dtau/tau = {conv['dtau']:.3f} (need < 0.01)")
    print("Saved " + ", ".join(str(p) for p in plot_paths))
    return source


def main():
    import copy

    args = parse_args()
    zs = args.redshifts
    if args.joint:
        _sigterm_to_keyboardinterrupt()
        try:
            status = run_joint(args)
        except KeyboardInterrupt:
            status = "interrupted"
            print("\nInterrupted outside sampling: no fit written.")
        if status in ("interrupted", "empty"):
            sys.exit(130)
        return
    # Ctrl-C and kill (SIGTERM) both arrive as KeyboardInterrupt. During
    # sampling run_redshift catches it and still writes that bin's fit and
    # plots; either way the remaining bins are skipped and we exit with 130.
    _sigterm_to_keyboardinterrupt()
    if len(zs) > 1:
        print(f"Fitting {len(zs)} redshift bins in turn: "
              + ", ".join(f"{z:g}" for z in zs))
    done = []
    for i, z in enumerate(zs):
        a = copy.copy(args)
        a.z = z
        if args.outdir is not None and len(zs) > 1:
            a.outdir = args.outdir / z_folder(z)
        if len(zs) > 1:
            print(f"\n{'=' * 72}\n[{i + 1}/{len(zs)}] z = {z:g}\n{'=' * 72}")
        try:
            status = run_redshift(a)
        except KeyboardInterrupt:
            status = "interrupted"
            print(f"\nInterrupted at z = {z:g} (outside sampling: no fit "
                  "written for this bin).")
        done.append((z, status))
        if status in ("interrupted", "empty"):
            skipped = [f"{zz:g}" for zz in zs[i + 1:]]
            if skipped:
                print(f"Stopping: skipped z = {', '.join(skipped)}. Re-run "
                      "the same command to resume.")
            sys.exit(130)
    if len(zs) > 1:
        print("\nAll bins done: " + ", ".join(f"z={z:g} ({st})"
                                              for z, st in done))


if __name__ == "__main__":
    main()
