import csv
import os
from typing import Tuple

import numpy as np

from shared.vision.tactile.pointnet_encoder import TactilePointNet

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Map taxel count -> fingertip coordinate CSV. Matches the Paxini fingertip
# models: 26mm/127pts for the thumb, 15mm/52pts for index/middle/ring. The
# association is by point count, exactly as midas_hand_api's Qt visualizer does.
_COORD_CSV = {
    127: os.path.join(_ASSET_DIR, "paxini_fingertip_26mm_127pts.csv"),
    52: os.path.join(_ASSET_DIR, "paxini_fingertip_15mm_52pts.csv"),
}


def _load_coords(path: str) -> np.ndarray:
    """Read an (X, Y, Z) fingertip coordinate CSV into an [N, 3] array.

    Taxel row i in the CSV corresponds to force row i in the raw tactile arrays.
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["X"]), float(row["Y"]), float(row["Z"])))
    if not rows:
        raise ValueError(f"Empty coordinate file: {path}")
    return np.asarray(rows, dtype=np.float32)


def get_tactile_pointnet(
    n_points: int,
    out_dim: int = 64,
    hidden: Tuple[int, ...] = (64, 128, 256),
    pool: str = "max",
) -> TactilePointNet:
    """
    Build a TactilePointNet for a finger with ``n_points`` taxels, loading the
    matching taxel coordinate map from the bundled assets.

    args:
        n_points : Taxel count (127 for thumb, 52 for index/middle/ring).
        out_dim : Output embedding size.
        hidden : Shared per-point MLP widths.
        pool : Symmetric pooling ("max", "mean", "maxmean").
    """
    csv_path = _COORD_CSV.get(int(n_points))
    if csv_path is None:
        raise ValueError(
            f"No tactile coordinate asset for n_points={n_points}. "
            f"Known counts: {sorted(_COORD_CSV)}"
        )
    coords = _load_coords(csv_path)
    if coords.shape[0] != int(n_points):
        raise ValueError(
            f"Coordinate file {csv_path} has {coords.shape[0]} points, "
            f"expected {n_points}"
        )
    return TactilePointNet(coords=coords, out_dim=out_dim, hidden=tuple(hidden), pool=pool)
