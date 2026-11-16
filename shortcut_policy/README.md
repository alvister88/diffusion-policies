# Flow matching and shortcut policies

`UnetImageShortcutPolicy` predicts action-trajectory velocities conditioned on
observations, flow time, and integration step size. It trains with linear
conditional flow matching plus binary self-consistency. This is an adaptation of
[One Step Diffusion via Shortcut Models](https://arxiv.org/abs/2410.12557),
not a DDPM/DDIM noise-prediction model.

## Objective and sampling

Let `x0 ~ Normal(0,I)` and let `x1` be a normalized demonstration action chunk.
The flow-matching examples use:

```
t ~ Uniform(0,1)
xt = (1-t)*x0 + t*x1
velocity_target = x1-x0
loss = MSE(s(xt,t,0,obs), velocity_target)
```

Time runs from noise at 0 to actions at 1. For shortcut examples, two teacher
steps of length `h` supervise one student step of length `2h`:

```
v1 = stop_gradient(s(xt, t, h, obs))
xmid = xt + h*v1
v2 = stop_gradient(s(xmid, t+h, h, obs))
target = (v1+v2)/2
loss = MSE(s(xt,t,2h,obs), target)
```

At the finest teacher step (`h=1/128` by default), query `s(...,d=0)`
instead, connecting the shortcut hierarchy to the supervised flow field.
Teacher calls run without gradients. They use the current network weights;
the workspace's EMA copy is used for evaluation, not as the bootstrap teacher.
The paper also describes EMA teachers as a stability improvement. This code
does not reproduce every paper training choice (including guidance or its
image-generation optimizer settings).

The batch is split into 75% flow-matching and 25% shortcut examples by default,
then combined for one student forward/backward pass. Fractions round to whole
examples; a batch of one uses only the flow objective. Self-consistency starts
are grid aligned with `t+2h <= 1`; student lengths range from `2/128` to `1`.

Inference starts with fresh Gaussian noise and repeats
`x = x + d*s(x,t,d,obs)` for `N` steps, where `d=1/N`. At `N=128`, the
network is queried at `d=0` and integrated with `dt=1/128`.
`predict_action` honors `policy.num_inference_steps`.
Supported shortcut budgets are powers of two from 1 through
`policy.max_train_steps` (default 128). Smaller budgets require evaluation;
the training objective alone does not guarantee good one-step robot actions.

Set `policy.shortcut_fraction=0` to train plain flow matching. It queries the
network at `d=0` for both training and inference and allows any positive integer
integration budget. This setting requires a separate training run for a fair
comparison with shortcut training.

## Architecture for Trimi

With the supplied Trimi config:

1. Observations are `camera_0: [B,1,3,240,320]` and
   `hand_pose_abs: [B,1,13]`. The dataset normalizer transforms both.
2. Images resize to 240x240, crop to 224x224, receive color jitter, and use
   ImageNet normalization, matching the diffusion reference config.
   DINOv2 ViT-S/14 produces a 384-dimensional CLS vector. The observation
   encoder concatenates the normalized 13-joint hand state, giving 397 features.
   Observation histories are flattened into global conditioning; here `To=1`.
3. An independent sinusoidal embedding + MLP encodes step size in 128 dimensions.
   Flow time has its own 128-dimensional embedding inside the U-Net. Both
   continuous inputs are scaled by 128 before sinusoidal encoding. Together
   with the observation features, these form 653 conditioning features used
   by the residual blocks' FiLM scale and bias layers.
4. A temporal 1D U-Net takes `[B,16,19]` interpolated action chunks. Its channel
   widths are `[256,512,1024]`, kernel size 5, with GroupNorm, Mish activations,
   residual blocks, and skip connections. The temporal resolutions are
   16 -> 8 -> 4 -> 8 -> 16. Its output is `[B,16,19]` velocities, not image pixels.
5. Sampling unnormalizes the resulting 16-action chunk and returns 8 actions,
   beginning at `num_obs_steps-1` (index zero here). The 19 action coordinates
   are the reused MIDAS loader's window-relative 6D EEF pose and 13 hand-joint
   deltas. These retain the original Trimi motor order and units.

Only global observation conditioning is supported. Invalid local conditioning
and incompatible horizon lengths fail at construction.

## Train with the existing Trimi settings

From the repository root, in your training environment:

```bash
python shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py \
  --config-name=train_unet_image_trimi_hand_cond_dinov2_shortcut_policy \
  tasks.dataset.dataset_path=/absolute/path/to/data_trimorph
```

The training YAML explicitly defines the encoder, policy, optimizer, EMA,
loaders, schedule, logging, checkpoints, and output paths. It loads the local
`shortcut_policy/config/tasks/trimi_image_hand_cond.yaml` for observation shapes,
dataset settings, and the runner. Neither YAML inherits diffusion-policy configs.
The settings match the diffusion Trimi reference:

| Setting | Value |
| --- | --- |
| Prediction / observation / executed action horizons | 16 / 1 / 8 |
| Inference steps | 16 |
| Latency steps | 0 |
| Encoder | Pretrained DINOv2 ViT-S/14, CLS, trainable |
| Train / validation batch size | 256 / 256 |
| AdamW | lr 1e-4, betas [0.95,0.999], weight decay 1e-6 |
| Schedule | 500 epochs, cosine, 500 warmup optimizer updates |
| EMA | power 0.75, maximum decay 0.9999 |
| Validation / samples / checkpoints | Every 5 / 10 / 50 epochs |
| Dataset split / sync downsampling stride | 15% validation / 2 |

It uses the shortcut policy and training workspace, with
`shortcut_fraction=0.25`, `max_train_steps=128`. The DDIM beta schedule,
50 diffusion training timesteps, epsilon target, and sample clipping are not
part of the flow objective. `max_train_steps` is a time-resolution parameter,
not an optimizer-update limit. `training.max_train_steps` separately limits
batches per epoch for debugging.

For a plain flow-matching baseline with the same settings:

```bash
python shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py \
  --config-name=train_unet_image_trimi_hand_cond_dinov2_shortcut_policy \
  tasks.dataset.dataset_path=/absolute/path/to/data_trimorph \
  policy.shortcut_fraction=0 exp_name=trimi_hand_cond_dinov2_flow
```

Useful overrides include `num_inference_steps=4`,
`dataloader.batch_size=64 val_dataloader.batch_size=64`,
`training.freeze_encoder=true`, `logging.mode=online`, and
`training.debug=true`. Changing the batch size or freezing the encoder changes
the comparison with your diffusion run. `num_action_steps` controls action
execution length; `num_inference_steps` controls network evaluations per chunk.

Inspect the fully composed config without training:

```bash
python shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py \
  --config-name=train_unet_image_trimi_hand_cond_dinov2_shortcut_policy \
  --cfg job --resolve
```

The Trimi data format and frame conventions still apply. The MIDAS runner
is a stub and performs no robot rollouts. Action MSE is a diagnostic; compare
held-out losses within each objective and evaluate executed behavior separately.
The optional multi-budget timing/MSE benchmark is disabled by default; enable
it with `training.do_shortcut_speed_test=true`. Its MSE samples fresh noise at
each budget, and its timings include observation encoding and action conversion.

## Correctness fixes and verification

The previous implementation dropped `distance` before the network, generated
student steps as large as 2 and starts with `t+2h>1`, lacked a flow-supervised
bootstrap base case, cached the construction-time CPU device, squeezed singleton
batch times to scalars, assumed an observation key named `image`, and hardcoded
two-step inference. These are corrected. The trainer also evaluates validation
in eval mode, steps gradient accumulation at group boundaries, updates EMA once
per optimizer step, retains its update count in checkpoints, and honors the
benchmark toggle with synchronized CUDA timing.

The added step-size encoder and larger FiLM conditioning change checkpoint
shapes. Retrain corrected policies; old shortcut and DDIM checkpoints cannot
be loaded directly as corrected shortcut policies.

Regression checks use synthetic observations and a small U-Net without
fetching DINOv2 weights or requiring robot data:

```bash
python -m unittest discover -s tests -v
```
