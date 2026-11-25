"""
Literature UV luminosity functions to fit alongside EPOCHS-v2.

Ground-based bright-end UVLFs at z ~ 7. The tables in ``literature_data/``
are verbatim copies of the flags_data exports
(``LUV/obs/binned/<file>.ecsv``). Errors are symmetric; each point has its own
bin width ``deltaM``, which the fit uses to bin-average the model.

    Bowler+17     UltraVISTA DR3 + UDS (2017MNRAS.466.3612B)
    Varadaraj+23  VISTA: COSMOS + XMM-LSS (2023MNRAS.524.4586V)
    Varadaraj+26  VISTA + Euclid (A&A 707, A239)

These samples share VISTA imaging (Varadaraj+23 includes the UltraVISTA area
of Bowler+17; Varadaraj+26 adds Euclid to it), so their points are not fully
independent: fitting them all together counts the brightest galaxies more
than once.
"""

from pathlib import Path

import numpy as np
from astropy.table import Table

DATA_DIR = Path(__file__).resolve().parent / "literature_data"

LITERATURE = {
    "Bowler+17": {"file": "bowler17.ecsv", "survey": "UltraVISTA"},
    "Varadaraj+23": {"file": "varadaraj23.ecsv", "survey": "VISTA"},
    "Varadaraj+26": {"file": "varadaraj26.ecsv", "survey": "VISTA+Euclid"},
}

# Sets fitted by ``--lit`` (no names) in each EPOCHS-v2 redshift bin.
DEFAULT_LITERATURE = {
    7.0: ("Bowler+17", "Varadaraj+23", "Varadaraj+26"),
}


def literature_tag(name):
    """Folder-safe form of a set name, e.g. ``Bowler17``."""
    return name.replace("+", "")


def load_literature_uvlf(name, z, muv_max=None, dz_max=0.5):
    """UVLF points of literature set ``name`` at the redshift nearest ``z``.

    Parameters
    ----------
    name : str
        A key of :data:`LITERATURE`.
    z : float
        Redshift of the fit.
    muv_max : float, optional
        Keep only points with ``MUV <= muv_max`` (as for EPOCHS-v2).
    dz_max : float, optional
        Largest allowed difference between ``z`` and the tabulated redshift.

    Returns
    -------
    dict or None
        ``MUV``, ``phi``, ``phi_err_low``, ``phi_err_upp``, ``dMUV`` (per
        point), sorted bright to faint, plus ``z`` (tabulated redshift),
        ``name`` and ``survey``. None if the set has no redshift within
        ``dz_max`` of ``z``, or no point passes the cut.
    """
    if name not in LITERATURE:
        raise ValueError(f"unknown literature set {name!r}; choose from "
                         f"{list(LITERATURE)}")
    info = LITERATURE[name]
    t = Table.read(DATA_DIR / info["file"], format="ascii.ecsv")
    zs = np.unique(np.asarray(t["z"], dtype=float))
    z_near = zs[np.argmin(np.abs(zs - z))]
    if abs(z_near - z) > dz_max:
        return None
    t = t[np.isclose(np.asarray(t["z"], dtype=float), z_near)]
    out = {
        "MUV": np.asarray(t["M"], dtype=float),
        "phi": np.asarray(t["phi"], dtype=float),
        "phi_err_low": np.asarray(t["phi_err_low"], dtype=float),
        "phi_err_upp": np.asarray(t["phi_err_upp"], dtype=float),
        "dMUV": np.asarray(t["deltaM"], dtype=float),
    }
    order = np.argsort(out["MUV"])
    keep = order if muv_max is None else order[out["MUV"][order] <= muv_max]
    if keep.size == 0:
        return None
    out = {k: v[keep] for k, v in out.items()}
    out.update(z=float(z_near), name=name, survey=info["survey"])
    return out
