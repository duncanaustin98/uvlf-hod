"""
EPOCHS-v2 (Austin et al. 2026, EPOCHS-DR2) fiducial UV luminosity function.

Binned 1/Vmax UVLF at 6.5 < z < 13.5 from JWST/NIRCam + HST/ACS imaging
(JADES, PRIMER, NEP, CEERS, NGDEEP). Photometric redshifts come from EAZY
(sfhz_blue_agn templates). Errors include Poisson and cosmic variance, and are
asymmetric for low-N bins. Bins without galaxies are dropped.

The table in ``austin26_data/UVLF_EPOCHS-DR2.ecsv`` is a verbatim copy of the
flags_data export (``LUV/obs/binned/austin26.ecsv``). ADS reference:
2026arXiv260924987A.
"""

from pathlib import Path

import numpy as np
from astropy.table import Table

DATA_FILE = Path(__file__).resolve().parent / "austin26_data" / "UVLF_EPOCHS-DR2.ecsv"

# Redshift-bin centres: 6.5-7.5, 7.5-8.5, 8.5-9.5, 9.5-10.5, 10.5-11.5, 11.5-13.5
REDSHIFTS = (7.0, 8.0, 9.0, 10.0, 11.0, 12.5)
# Every M_UV bin is 0.5 mag wide. The edges are np.arange(-22.75, -15.75, 0.5) in
# EPOCHS-v2/Paper-2/uvlf_to_flags_data.py, and empty bins are dropped from the
# table. Use this as Dataset.uvlf(dMUV=...); do NOT infer it from adjacent
# points: at z=7 (-22.0 empty) and z=10 (-21.5 empty) that difference is 1 mag.
BIN_WIDTH = 0.5

ZBIN_EDGES = {7.0: (6.5, 7.5), 8.0: (7.5, 8.5), 9.0: (8.5, 9.5),
              10.0: (9.5, 10.5), 11.0: (10.5, 11.5), 12.5: (11.5, 13.5)}


def load_austin26_uvlf(z, muv_max=None):
    """EPOCHS-v2 UVLF points in one redshift bin.

    Parameters
    ----------
    z : float
        Redshift-bin centre; one of :data:`REDSHIFTS`.
    muv_max : float, optional
        If given, keep only bins with centre ``MUV <= muv_max`` (inclusive,
        as galfind's ``crop_to_xbin`` applies the EPOCHS-v2 fit cuts).

    Returns
    -------
    dict
        ``MUV`` [mag], ``phi``, ``phi_err_low``, ``phi_err_upp``
        [Mpc^-3 mag^-1], sorted bright to faint, and ``dMUV`` (the bin
        width, :data:`BIN_WIDTH`).
    """
    z = float(z)
    if not any(np.isclose(z, zz) for zz in REDSHIFTS):
        raise ValueError(f"z={z:g} is not an EPOCHS-v2 bin; choose from {REDSHIFTS}")
    t = Table.read(DATA_FILE)
    t = t[np.isclose(np.asarray(t["z"]), z)]
    out = {
        "MUV": np.asarray(t["M"], dtype=float),
        "phi": np.asarray(t["phi"], dtype=float),
        "phi_err_low": np.asarray(t["phi_err_low"], dtype=float),
        "phi_err_upp": np.asarray(t["phi_err_upp"], dtype=float),
    }
    order = np.argsort(out["MUV"])
    keep = order if muv_max is None else order[out["MUV"][order] <= muv_max]
    out = {k: v[keep] for k, v in out.items()}
    out["dMUV"] = BIN_WIDTH
    return out
