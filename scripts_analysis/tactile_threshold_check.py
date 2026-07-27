"""Evaluate whether the old 0.15 N contact-threshold gate was well placed.

Reconstructs the OLD training tactile pipeline on data_midas12:
  raw per-taxel Fz  ->  subtract per-taxel baseline (median of first 60 frames)
  ->  residual = clip(Fz - baseline, 0, None)  ->  gate at threshold  ->
  mean over active taxels.

Produces, per finger (thumb, index):
  1. Histogram of raw taxel Fz vs baseline-subtracted residual, zoomed to the
     low-force band, with the 0.15 N gate drawn in -- shows where the noise
     floor sits relative to the threshold.
  2. Fraction of "active" taxels vs threshold sweep -- shows how sensitive the
     gate is; the knee is roughly where noise ends and contact begins.
  3. A time series for one contact-rich episode: per-finger average Fz BEFORE
     residual subtraction (raw mean over all taxels) vs AFTER (mean over
     active taxels only, gated at 0.15 N).
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = "/data/jun/Jonathan/data_midas12"
OUT_DIR = "/data/jun/Jonathan/plots"
FINGERS = ("thumb", "index")
THRESHOLD = 0.15
BASELINE_FRAMES = 60

os.makedirs(OUT_DIR, exist_ok=True)


def load_fz(path, finger):
    """Raw normal force [T, N] for one finger from a processed tactile npz."""
    return np.load(path)[finger][:, :, 2].astype(np.float32)


def baseline_of(fz):
    """Per-taxel at-rest offset: median over the first BASELINE_FRAMES frames."""
    n = min(BASELINE_FRAMES, fz.shape[0])
    return np.median(fz[:n], axis=0, keepdims=True)  # [1, N]


tac_files = sorted(glob.glob(os.path.join(DATA_ROOT, "processed_tactile_t", "*.npz")))
print(f"{len(tac_files)} tactile episodes")

# Accumulate raw + residual taxel values across all episodes, per finger.
raw_all = {f: [] for f in FINGERS}
res_all = {f: [] for f in FINGERS}
for path in tac_files:
    for f in FINGERS:
        fz = load_fz(path, f)
        base = baseline_of(fz)
        residual = np.clip(fz - base, 0.0, None)
        raw_all[f].append(fz.reshape(-1))
        res_all[f].append(residual.reshape(-1))
raw_all = {f: np.concatenate(v) for f, v in raw_all.items()}
res_all = {f: np.concatenate(v) for f, v in res_all.items()}

# ---- Plot 1: raw vs residual histogram, per finger --------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, f in zip(axes, FINGERS):
    bins = np.linspace(-0.5, 2.0, 200)
    ax.hist(raw_all[f], bins=bins, alpha=0.5, label="raw Fz", color="tab:blue",
            log=True)
    ax.hist(res_all[f], bins=bins, alpha=0.5, label="residual (Fz - baseline)",
            color="tab:orange", log=True)
    ax.axvline(THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"0.15 N gate")
    ax.set_title(f"{f}  (per-taxel force distribution)")
    ax.set_xlabel("Fz (N)")
    ax.set_ylabel("count (log)")
    ax.legend()
fig.suptitle("data_midas12 tactile: raw vs baseline-subtracted per-taxel Fz")
fig.tight_layout()
p1 = os.path.join(OUT_DIR, "tactile_raw_vs_residual_hist.png")
fig.savefig(p1, dpi=130)
print("saved", p1)

# ---- Plot 2: active-taxel fraction vs threshold sweep -----------------------
sweep = np.linspace(0.0, 1.0, 101)
fig, ax = plt.subplots(figsize=(8, 5))
for f in FINGERS:
    r = res_all[f]
    frac = [(r >= t).mean() * 100 for t in sweep]
    ax.plot(sweep, frac, label=f)
ax.axvline(THRESHOLD, color="red", ls="--", lw=1.5, label="0.15 N gate")
ax.set_xlabel("threshold (N)")
ax.set_ylabel("% of taxel-frames counted active")
ax.set_title("Active-taxel fraction vs threshold (residual)")
ax.set_yscale("log")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
p2 = os.path.join(OUT_DIR, "tactile_active_fraction_vs_threshold.png")
fig.savefig(p2, dpi=130)
print("saved", p2)

# ---- Plot 3: before vs after time series for a contact-rich episode ---------
# Pick the episode with the highest residual energy so contact is visible.
best_path, best_energy = tac_files[0], -1.0
for path in tac_files:
    e = 0.0
    for f in FINGERS:
        fz = load_fz(path, f)
        e += float(np.clip(fz - baseline_of(fz), 0.0, None).sum())
    if e > best_energy:
        best_energy, best_path = e, path
print("time-series episode:", os.path.basename(best_path))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, f in zip(axes, FINGERS):
    fz = load_fz(best_path, f)
    base = baseline_of(fz)
    residual = np.clip(fz - base, 0.0, None)
    raw_mean = fz.mean(axis=1)                      # avg over ALL taxels, no gate
    mask = residual >= THRESHOLD
    counts = mask.sum(axis=1)
    gated_mean = np.where(counts > 0,
                          (residual * mask).sum(axis=1) / np.maximum(counts, 1),
                          0.0)
    t = np.arange(fz.shape[0])
    ax.plot(t, raw_mean, label="before: raw mean over all taxels",
            color="tab:blue", lw=1)
    ax.plot(t, gated_mean, label="after: mean over active taxels (>0.15 N residual)",
            color="tab:orange", lw=1)
    ax.set_title(f"{f}")
    ax.set_ylabel("Fz (N)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("tactile frame")
fig.suptitle(f"Per-finger avg Fz before vs after residual+gate "
             f"({os.path.basename(best_path)})")
fig.tight_layout()
p3 = os.path.join(OUT_DIR, "tactile_before_after_timeseries.png")
fig.savefig(p3, dpi=130)
print("saved", p3)

# ---- Console summary --------------------------------------------------------
print("\n=== residual percentiles per finger (N) ===")
for f in FINGERS:
    r = res_all[f]
    pct = np.percentile(r, [50, 90, 95, 99, 99.9])
    print(f"{f:6s} p50={pct[0]:.3f} p90={pct[1]:.3f} p95={pct[2]:.3f} "
          f"p99={pct[3]:.3f} p99.9={pct[4]:.3f}  "
          f"frac>=0.15: {(r >= THRESHOLD).mean()*100:.2f}%")
