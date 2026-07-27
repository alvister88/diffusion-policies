# Tactile encoding architecture (MIDAS diffusion policy)

This document describes how per-taxel tactile sensing is encoded and fed to the
MIDAS diffusion policy. It replaces the previous hand-reduced tactile feature
(`tactile_z = [thumb_mean_Fz, thumb_max_Fz, index_mean_Fz, index_max_Fz]`) with a
learned **point-cloud (PointNet-style) encoder** over the full per-taxel force
field.

## Why this change

The old `tactile_z` kept only the mean and max of the **normal** force per finger,
discarding (a) both shear axes and (b) the spatial distribution of contact across
the fingertip. That is almost all of the signal. We instead feed the full force
field through a small learned encoder so the policy can use *where* and *how* the
fingertip is being loaded, not just how hard on average.

## Sensor data format

Each fingertip (Paxini) reports, per frame, a set of taxels; each taxel carries a
3-axis force vector:

```
raw tactile.npz  (per raw episode)
  thumb : (T, 127, 3)   # 127 taxels
  index : (T,  52, 3)   #  52 taxels
  columns = [Fx, Fy, Fz] = [shear-x, shear-y, normal-z]  (Newtons)
```

Scope: **thumb + index** only (middle/ring exist in the raw data but are not used).

### The taxels are NOT a 2D image

The taxels are an **irregular set of points on a curved 3D fingertip surface**, not a
rectangular pixel grid, and the counts (127, 52) are not factorable into clean
grids. A stock 2D ResNet therefore does not apply. Instead the SDK ships a **3D
coordinate map** per fingertip:

```
shared/vision/tactile/assets/
  paxini_fingertip_26mm_127pts.csv   # thumb   (X, Y, Z) mm, 127 rows
  paxini_fingertip_15mm_52pts.csv    # index   (X, Y, Z) mm,  52 rows
```

**Index contract:** taxel row `i` in the force array corresponds to coordinate row
`i` in the CSV. (Copied from `midas_hand_api`, which relies on the same contract in
its Qt visualizer.)

### Coordinate frame

The CSV coordinates are **per-fingertip local coordinates, in millimeters**, in each
sensor's own frame — fingers are **not** co-registered in a common hand frame:

- **X** = lateral width across the pad (symmetric about 0).
- **Y** = along the finger length.
- **Z** = outward surface height (the curvature of the gel dome).

Approx ranges: thumb X∈[-12.8,12.8], Y∈[-4.0,23.8], Z∈[-0.3,13.0]; index
X∈[-7.5,7.5], Y∈[-1.6,19.3], Z∈[1.9,6.7]. Because scale and curvature differ per
finger, each finger gets its **own** encoder and its coordinates are centered +
scaled **inside** that encoder.

## Forces vs. coordinates — what flows where

This is the key distinction:

- The **data pipeline** (dataset → normalizer → policy) only ever carries the
  time-varying **forces**, shape `[B, N, 3] = [Fx, Fy, Fz]`. The "3" here is the
  force triple, the same "3" as the raw `(T, N, 3)`.
- The **static coordinates** `[N, 3] = [x, y, z]` never appear in the data. They are
  fixed geometry, loaded once from the CSV and stored as a buffer **inside** the
  encoder module.
- Inside `TactilePointNet.forward`, the coordinates are concatenated onto the forces
  to form the actual per-point input `[B, N, 6] = [x, y, z, Fx, Fy, Fz]`.

```
forces  x[B, N, 3] = [Fx, Fy, Fz]   ← from dataset, normalized
coords  buf[N, 3]  = [x,  y,  z]    ← static, baked into the module (mm)
                   │  concat per point
                   ▼
        [B, N, 6]  = [x, y, z, Fx, Fy, Fz]   ← PointNet input
```

## Encoder architecture

`shared/vision/tactile/pointnet_encoder.py` — `TactilePointNet` (one per finger):

```
input forces        [B, N, 3]        (normalized [Fx,Fy,Fz])
  + static coords    [1, 3, N]  ->  concat -> points [B, 6, N]
shared per-point MLP  Conv1d(1x1): 6 -> 64 -> 128 -> 256   (+GroupNorm, ReLU)
symmetric pool        max over the N points               -> [B, 256]
linear head           256 -> out_dim (64)                  -> [B, 64]
```

- **Shared MLP via `Conv1d` kernel-size-1** = the same MLP applied to every point.
- **Global max-pool** makes the embedding permutation- and count-invariant, so the
  127-taxel thumb and 52-taxel index reuse the identical architecture.
- ~59k params per finger.

Factory: `shared/vision/tactile/model_getter.py :: get_tactile_pointnet(n_points,
out_dim=64, ...)` loads the matching coordinate CSV by point count.

## End-to-end data flow

```
midas-piper-control/data_collection/postprocess_dataset.py
    raw/episode_XXXX/tactile.npz  ── keep full per-finger arrays ──►
    processed_tactile_t/{stem}.npz   { thumb:(T,127,3), index:(T,52,3) }
                                     (synced timeline; index via syncs' tactile_idx)
        │
        ▼  diffusion-policies
shared/env/midas/midas_image_dataset.py
    load .npz per finger, index by tactile_idx, store per-finger arrays in the
    replay buffer as obs keys tactile_thumb [T,127,3], tactile_index [T,52,3]
        │  get_normalizer(): SingleFieldLinearNormalizer.create_fit per key
        │  (last_n_dims=1 → per-axis Fx/Fy/Fz stats)
        ▼
shared/vision/common/multi_image_obs_encoder.py :: ObsEncoder
    obs type "tactile" → per-key TactilePointNet → [B, 64] each
    concat with camera features + hand_pose_abs → Fo
        │
        ▼
diffusion_policy/unet/unet_image_policy.py :: DiffusionUnetImagePolicy
    global conditioning cond_BG = Fo * num_obs_steps → ConditionalUnet1D (FiLM)
```

`ObsEncoder.output_shape()` recomputes `Fo` automatically, so no UNet dims are
hand-edited. Example: `Fo = dinov2_cls(384) + hand(13) + 64 + 64 = 525`
(or `512 + 13 + 64 + 64 = 653` with a ResNet-18 camera backbone).

## Config knobs

`diffusion_policy/config/tasks/midas_image_hand_cond.yaml` — obs keys:

```yaml
tactile_thumb: { shape: [127, 3], type: tactile }
tactile_index: { shape: [52, 3],  type: tactile }
```

`diffusion_policy/config/train_unet_image_midas_hand_cond_dinov2_policy.yaml` —
`obs_encoder.tactile_encoders`:

```yaml
tactile_encoders:
  tactile_thumb: { _target_: shared.vision.tactile.model_getter.get_tactile_pointnet, n_points: 127, out_dim: 64 }
  tactile_index: { _target_: shared.vision.tactile.model_getter.get_tactile_pointnet, n_points: 52,  out_dim: 64 }
```

- `n_points` selects the coordinate CSV (127 thumb / 52 index).
- `out_dim` sets the per-finger embedding width (added global-cond cost ≈
  `n_fingers × out_dim × num_obs_steps`).
- Finger scope is set by which `tactile_*` keys exist in `shape_meta` **and** which
  fingers `postprocess_dataset.py` emits (`TACTILE_FINGER_ORDER`).

## Data regeneration

Re-run `postprocess_dataset.py` so `processed_tactile_t/` holds `.npz` files (was
`.npy` of shape `[T,4]`). The dataset cache key hashes `shape_meta`, so any stale
cache invalidates automatically.

## Inference (deployment follow-up)

`shared/env/midas/midas_inference.py` now accepts tactile: pass
`predict(image, robot_state, tactile={"tactile_thumb": [127,3], "tactile_index":
[52,3]})`. It keeps a per-finger history like the image history and validates
shapes. **Remaining deployment work (in `midas-piper-control`):** the live executor
loop must read the Paxini sensor (`PaxiniHandSensor.read_latest()`), assemble the
per-finger `[N,3]` arrays, and pass them into `predict(...)`. Until that is wired,
real-robot rollouts would run without tactile; training and offline use are
unaffected.
