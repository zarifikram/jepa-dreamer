<p align="center">
  <img src="https://github.com/zarifikram/lp-dreamer/blob/master/assets/banner.png?raw=true" alt="Latent Linear Prior — method overview" width="100%">
</p>

<p align="center"><em>The latent linear prior (left) and value-loss rejection of anomalous imagined transitions (right).</em></p>

# LP-DreamerV3 — World Model Anomaly Detection with a Latent Linear Prior

Official code for the TMLR paper
**["World Model Anomaly Detection with a Latent Linear Prior"](https://openreview.net/forum?id=VLIzLK3CfR)**.

LP augments a [DreamerV3](https://github.com/NM512/dreamerv3-torch) world model with a **latent
linear prior**: a single learned linear map that the model's latent transitions are trained to
follow. Imagined transitions that *violate* this prior are flagged as **anomalies** (likely
world-model hallucinations), and their value updates are switched off — so the actor–critic never
learns from imagined states the model predicts poorly.

---

## The idea in three steps

1. **Linear prior.** A learned matrix `W` predicts the next deterministic latent state,
   `h_t ≈ W h_{t-1}`. A decay-weighted *consistency loss* over the next `K = 32` states trains the
   world model so its latents evolve approximately linearly ("perceptual straightening").

2. **Anomaly detection.** For each imagined transition we standardize the consistency error,
   `z = (‖ĥ_t − W ĥ_{t-1}‖² − μ) / σ`, using the train-time mean `μ` and std `σ`, and compare it
   to a threshold `k_σ` (default **1.645**, the one-sided 95% quantile). `z > k_σ` ⇒ anomalous.

3. **Rejection.** The **critic loss is zeroed** on anomalous imagined transitions, so hallucinated
   states do not corrupt value learning.

```
   real latent ──►  W·h_{t-1}  ──►  predicted next latent
                        │
   imagined ĥ_t ────────┴──► consistency error z ──► z > k_σ ? ──► zero critic loss
```

---

## Repository layout

```
dreamer.py            # training entry point
models.py             # world model + actor–critic; LP wiring (use_tcl_loss + value masking)
networks.py           # RSSM, CNN encoder/decoder, MLP heads
delusion/losses.py    # THE method: TemporalConsistancyLoss (consistency loss + rejection rule)
configs.yaml          # config blocks: defaults · atari100k · updates · tcl · debug
envs/                 # atari.py, wrappers.py  (Atari 100k)
tools.py utils.py exploration.py parallel.py
run_lp.sh             # convenience runner (smoke / full)
analysis/             # paper analysis scripts (see below)
assets/               # README banner (banner.png + banner.pdf source)
```

> This release is scoped to the **Atari 100k** benchmark used in the paper.

---

## Installation (tested, Python 3.9)

The code uses the legacy `atari_py` backend, so the environment stack is pinned.

```bash
conda create -n lpdreamer python=3.9 -y && conda activate lpdreamer

# PyTorch (CUDA 11.7)
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117

# Core deps
pip install numpy==1.21.0 "ruamel.yaml" einops==0.3.0 x-transformers==1.6.4 \
            opencv-python wandb==0.25.0 tensorboard scikit-learn==1.1.3

# Atari (old gym + atari_py; build with matching build tools)
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip install --no-build-isolation gym==0.19.0 atari-py==0.2.9

# Atari ROMs
pip install autorom && AutoROM --accept-license --install-dir /tmp/roms
python -m atari_py.import_roms /tmp/roms
```

For the analysis scripts also install `matplotlib scipy pandas`.

---

## Usage

LP is selected by the config stack **`atari100k updates tcl`**:
`atari100k` (benchmark settings) · `updates` (declares all method flags, default off) ·
`tcl` (turns on the linear-prior consistency loss + rejection).

**Quick smoke test** (wandb off, ~3 min):

```bash
./run_lp.sh smoke pong
```

**Full Atari 100k run** (400k frames = 100k agent steps):

```bash
./run_lp.sh full kung_fu_master 0        # game, seed

# ...or directly:
python dreamer.py --task atari_kung_fu_master --seed 0 \
    --configs atari100k updates tcl --compile False \
    --logdir ./logdir/kfm_lp_s0
```

Enable Weights & Biases with `wandb login`, then
`--wandb_enabled True --wandb_proj lp-dreamer`.

### Key LP flags (from the `tcl` config)

| flag | default | meaning |
|---|---|---|
| `--k_sigma` | `1.645` | rejection threshold (one-sided 95% quantile) |
| `--value_adjust` | `True` | zero the critic loss on anomalous transitions (the core mechanism) |
| `--consistency_detach` | `False` | train `W` but stop its gradient into the world model (rejection-only ablation) |
| `tcl_config.k` | `32` | number of look-ahead states in the consistency loss |
| `tcl_config.delta` | `0.5` | decay weight over the look-ahead states |

---

## Where the method lives

- **`delusion/losses.py` → `TemporalConsistancyLoss`** — the consistency loss (`calculate_loss`)
  and the rejection rule (`calculate_rejection_mask_and_distance_from_generated_outputs`, which
  standardizes the error and thresholds at `k_σ`).
- **`models.py` → `ImagBehavior._train`** — applies the rejection mask by zeroing `value_loss`
  on flagged imagined transitions (gated by `use_tcl_loss` + `value_adjust`).

---

## Analysis scripts (`analysis/`)

- **`e2_keystone.py`** — loads a trained checkpoint and, on held-out trajectories, measures
  (i) consistency-residual normality, (ii) the correlation between the linearity error and
  one-step prediction error, and (iii) whether the most-anomalous transitions are the
  worst-predicted. Run: `python analysis/e2_keystone.py --logdir <run> --task atari_<game>`.
- **`rejection_dynamics.py`** — plots the value-loss rejection rate over training for different
  `k_σ`.
- **`table1_stats.py`** — per-seed mean/std for the results tables.

---

## Citation

```bibtex
@article{
ikram2026world,
title={World Model Anomaly Detection with a Latent Linear Prior},
author={Zarif Ikram and Harry Zhao and Ling Pan and Alex Lamb and Dianbo Liu},
journal={Transactions on Machine Learning Research},
year={2026},
url={https://openreview.net/forum?id=VLIzLK3CfR},
note={}
}
```

## Acknowledgements

Built on the DreamerV3 PyTorch implementation,
[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch).
