import numpy as np
import torch
import torch.nn as nn

from typing import Tuple


class TactilePointNet(nn.Module):
    """
    PointNet-style set encoder for a single tactile fingertip.

    The Paxini fingertip is an irregular set of taxels on a curved 3D surface,
    not a 2D pixel grid, so we treat each taxel as a point. Per-taxel input is
    the force triple [Fx, Fy, Fz] (shear-x, shear-y, normal-z); the fixed taxel
    coordinates [x, y, z] (in the fingertip-local mm frame) are held as a static
    buffer and concatenated on, giving 6 features per point. A shared per-point
    MLP (1x1 Conv1d over the point axis) lifts each point, and a symmetric pool
    over points produces a permutation-invariant, taxel-count-invariant embedding.

    Dimension keys:
        B: batch (typically the flattened batch*obs_steps of the policy)
        N: number of taxels (127 thumb, 52 index/middle/ring)
    """

    def __init__(
        self,
        coords: np.ndarray,
        out_dim: int = 64,
        hidden: Tuple[int, ...] = (64, 128, 256),
        pool: str = "max",
    ):
        """
        args:
            coords : [N, 3] static taxel positions (x, y, z) in the fingertip
                     -local mm frame. Row i must correspond to force row i.
            out_dim : Size of the output embedding.
            hidden : Widths of the shared per-point MLP layers.
            pool : Symmetric pooling over points: "max", "mean", or "maxmean"
                   (concatenation of both).
        """
        super().__init__()

        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be [N, 3], got {coords.shape}")
        if pool not in ("max", "mean", "maxmean"):
            raise ValueError(f"Unsupported pool '{pool}'")

        self.num_points = coords.shape[0]
        self.out_dim = out_dim
        self.pool = pool

        # Center + scale coordinates per finger so the geometry lands in a
        # roughly unit range regardless of fingertip size (thumb dome ~13mm vs
        # index ~5mm). These are fixed geometry, not learned.
        centered = coords - coords.mean(axis=0, keepdims=True)
        scale = float(np.abs(centered).max())
        if scale < 1e-6:
            scale = 1.0
        norm_coords = centered / scale
        # Buffer shape [1, 3, N] so it broadcasts over the batch and aligns with
        # the [B, 3, N] channels-first force layout used inside forward().
        coord_buf = torch.from_numpy(norm_coords.T[None].copy())
        self.register_buffer("coords", coord_buf)

        layers = []
        in_ch = 6  # [x, y, z, Fx, Fy, Fz]
        for width in hidden:
            layers.append(nn.Conv1d(in_ch, width, kernel_size=1))
            layers.append(nn.GroupNorm(num_groups=min(8, width), num_channels=width))
            layers.append(nn.ReLU(inplace=True))
            in_ch = width
        self.point_mlp = nn.Sequential(*layers)

        pooled_dim = in_ch * (2 if pool == "maxmean" else 1)
        self.head = nn.Linear(pooled_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x : [B, N, 3] per-taxel force vectors [Fx, Fy, Fz] (already
                normalized by the policy's LinearNormalizer). Positions are NOT
                part of x -- they come from the static coords buffer.
        returns:
            [B, out_dim] embedding.
        """
        if x.dim() != 3 or x.shape[1] != self.num_points or x.shape[2] != 3:
            raise ValueError(
                f"Expected forces [B, {self.num_points}, 3], got {tuple(x.shape)}"
            )

        B = x.shape[0]
        # [B, N, 3] -> [B, 3, N] channels-first for the shared 1x1 conv.
        forces = x.transpose(1, 2)
        coords = self.coords.expand(B, -1, -1)
        # [B, 6, N] = [x, y, z, Fx, Fy, Fz] per point.
        points = torch.cat([coords, forces], dim=1)

        feats = self.point_mlp(points)  # [B, C, N]

        if self.pool == "max":
            pooled = feats.max(dim=-1).values
        elif self.pool == "mean":
            pooled = feats.mean(dim=-1)
        else:  # maxmean
            pooled = torch.cat([feats.max(dim=-1).values, feats.mean(dim=-1)], dim=-1)

        return self.head(pooled)
