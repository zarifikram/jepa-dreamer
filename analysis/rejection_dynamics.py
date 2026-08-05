"""Anomaly-rejection rate over training (Kung Fu Master k-ablation).

The deployed safeguard is conservative: on a *converged* checkpoint it rejects ~0%
(see e2_keystone E2e), so a post-hoc k_sigma sweep on the final model is flat. The
informative threshold signal is the *dynamics*: how often the value-loss mask fires
over training, as a function of the rejection threshold k_sigma.

Metric provenance (models.py ImagBehavior._train):
  metrics["rejection_rate/timestep_i"] = COUNT of rejected transitions at imagined
  transition step i, out of batch_size*batch_length = 16*64 = 1024 trajectories.
  There are 14 transitions (imag_horizon=15 states). So
      rejection rate = mean_i(timestep_i) / 1024.

Reads ./logdir/kungfumaster_k{ks}_s0/metrics.jsonl for ks in the sweep and writes
analysis/figs/rejection_dynamics.png. CPU-only; safe to run alongside training.

NOTE: each k_sigma is an INDEPENDENT run, so curves are only approximately ordered
by k_sigma (their consistency-error distributions diverge); the figure shows the
within-run *decay*, not a calibrated cross-run rejection-vs-k_sigma curve.
"""
import json
import pathlib
import sys

import numpy as np

DENOM = 16 * 64  # batch_size * batch_length


def load_series(metrics_path):
    steps, rates = [], []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts = [v for k, v in d.items() if k.startswith("rejection_rate/timestep_")]
            if "step" not in d or not counts:
                continue
            steps.append(d["step"])
            rates.append(float(np.mean(counts)) / DENOM * 100.0)  # percent
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(rates)[order]


def rolling(y, w=15):
    if len(y) < 3:
        return y
    w = min(w, len(y))
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = pathlib.Path(__file__).resolve().parent.parent
    sweep = sys.argv[1:] or ["1.0", "1.645", "2.0", "3.0"]
    colors = {"1.0": "#d62728", "1.645": "#1f77b4", "2.0": "#2ca02c", "3.0": "#9467bd"}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for ks in sweep:
        mp = root / "logdir" / f"kungfumaster_k{ks}_s0" / "metrics.jsonl"
        if not mp.exists():
            print(f"[skip] {mp} missing")
            continue
        steps, rates = load_series(mp)
        if len(steps) == 0:
            print(f"[skip] {ks}: no rejection metrics yet")
            continue
        lw = 2.6 if ks == "1.645" else 1.4
        label = f"$k_\\sigma$={ks}" + (" (deployed)" if ks == "1.645" else "")
        ax.plot(steps, rolling(rates), color=colors.get(ks, None), lw=lw, label=label)
        ax.plot(steps, rates, color=colors.get(ks, None), lw=0.5, alpha=0.18)
        print(f"k={ks}: n={len(steps)} steps, last-step={steps[-1]}, "
              f"rate first/mid/last = {rates[0]:.3f}% / "
              f"{rates[len(rates) // 2]:.3f}% / {rates[-1]:.3f}%")

    ax.axvline(400000, ls=":", c="grey", lw=1)
    ax.text(400000, ax.get_ylim()[1] * 0.95, " Atari100k", fontsize=8, color="grey", va="top")
    ax.set_xlabel("training step (environment frames)")
    ax.set_ylabel("value-loss rejection rate (%)")
    ax.set_title("Anomaly-rejection rate decays as the world model matures (Kung Fu Master)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    outdir = root / "analysis" / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "rejection_dynamics.png"
    fig.savefig(out, dpi=140)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
