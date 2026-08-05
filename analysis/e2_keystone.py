"""E2 keystone analysis for the LP rebuttal.

Loads a trained LP-DreamerV3 checkpoint, runs OPEN-LOOP imagination on real held-out
trajectories (ground-truth future observations available), and quantifies the link
between latent linearity error and world-model prediction quality:

  E2a  residual r_t = s_t - W s_{t-1} : histogram + QQ-plot  -> verifies the Gaussian
       residual assumption behind Theorem 1 (reviewers 3kMR, uGMF, mKCF).
  E2b  corr(linearity error, one-step reconstruction error / reward error)
       -> quantifies Fig 3 (cpod) and "non-linear regions -> false predictions" (uGMF).
  E2c  rejected vs retained imagined transitions: reconstruction / reward error
       -> "rejected states are worse critic targets" causal link (uGMF).
  E2d  (hook) DreamerV3 vs STORM linearity-error distribution -- needs a STORM ckpt.

Run (after a checkpoint exists, e.g. once a run passes ~100k env steps):
  python analysis/e2_keystone.py --logdir ./logdir/kungfumaster_k1.645_s0 \
      --task atari_kung_fu_master

Outputs printed stats + figures under analysis/figs/<run-name>/.

NOTE: tcl.mu/tcl.std are NOT stored in the checkpoint (set during training), so this
script recomputes them by running tcl.calculate_loss on held-out batches first.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import yaml  # noqa: E402
import tools  # noqa: E402
import dreamer as dreamer_mod  # noqa: E402
from dreamer import Dreamer, make_env, make_dataset  # noqa: E402


def build_config(task, logdir, configs=("atari100k", "updates", "tcl"), overrides=None):
    """Replicate dreamer.py's config merge + main() adjustments, headless."""
    root = pathlib.Path(__file__).resolve().parent.parent
    raw = yaml.safe_load((root / "configs.yaml").read_text())

    def recursive_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                recursive_update(base[k], v)
            else:
                base[k] = v

    merged = {}
    for name in ["defaults", *configs]:
        recursive_update(merged, raw[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(merged.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    argv = ["--task", task, "--logdir", str(logdir), "--wandb_enabled", "False",
            "--compile", "False"]  # checkpoints are trained with --compile False (no _orig_mod prefix)
    for k, v in (overrides or {}).items():
        argv += [f"--{k}", str(v)]
    config = parser.parse_args(argv)
    for _k in list(vars(config)):
        setattr(config, _k, tools.coerce_numbers(getattr(config, _k)))

    config.traindir = config.traindir or pathlib.Path(config.logdir) / "train_eps"
    config.evaldir = config.evaldir or pathlib.Path(config.logdir) / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    return config


def load_agent(config):
    """Build the Dreamer agent and load latest.pt, mirroring dreamer.py main()."""
    logdir = pathlib.Path(config.logdir)
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    env = make_env(config, "train", 0)
    env = dreamer_mod.Damy(env)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    logger = tools.Logger(logdir, 0)
    dataset = make_dataset(train_eps, config)
    agent = Dreamer(env.observation_space, acts, config, logger, dataset).to(config.device)
    agent.requires_grad_(False)
    ckpt = torch.load(logdir / "latest.pt", map_location=config.device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent, dataset, env


@torch.no_grad()
def calibrate_tcl(wm, dataset, n_batches=20):
    """Recompute tcl.mu/std (train-time consistency-error stats) on held-out batches."""
    mus, stds = [], []
    for _ in range(n_batches):
        data = wm.preprocess(next(dataset))
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        wm.tcl.calculate_loss(post["deter"])  # sets wm.tcl.mu / wm.tcl.std
        mus.append(float(wm.tcl.mu))
        stds.append(float(wm.tcl.std))
    wm.tcl.mu = torch.tensor(np.mean(mus), device=next(wm.parameters()).device)
    wm.tcl.std = torch.tensor(np.mean(stds), device=next(wm.parameters()).device)
    print(f"[calibrate] tcl.mu={float(wm.tcl.mu):.5f} tcl.std={float(wm.tcl.std):.5f}")


@torch.no_grad()
def collect(wm, dataset, n_batches, context):
    """Open-loop imagination on real trajectories.
    Returns flat arrays over all imagined transitions: linearity z-distance,
    rejection mask, reconstruction error, reward error; plus residuals for E2a.
    """
    dev = next(wm.parameters()).device
    lin, rej, recon_err, rew_err, residuals = [], [], [], [], []
    for _ in range(n_batches):
        data = wm.preprocess(next(dataset))
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])

        # ---- E2a: residuals on the filtered (real) latent trajectory ----
        deter = F.normalize(post["deter"], dim=-1)                # (B,L,D) as trained
        mu = wm.tcl.predictor(deter[:, :-1]).chunk(2, dim=-1)[0]  # predict s_t from s_{t-1}
        residuals.append((deter[:, 1:] - mu).reshape(-1).cpu().numpy())

        # ---- open-loop rollout from a short real context ----
        init = {k: v[:, context - 1] for k, v in post.items()}
        prior = wm.dynamics.imagine_with_action(data["action"][:, context:], init)
        feat = wm.dynamics.get_feat(prior)                        # (B,H,feat)
        img = wm.heads["decoder"](feat)["image"].mode()          # (B,H,64,64,3)
        rew = wm.heads["reward"](feat).mode()                    # (B,H)
        truth_img = data["image"][:, context:]
        truth_rew = data["reward"][:, context:]

        re = ((img - truth_img) ** 2).mean(dim=[2, 3, 4])         # (B,H) recon error
        we = (rew.squeeze(-1) - truth_rew).abs() if rew.dim() > 2 else (rew - truth_rew).abs()

        # ---- linearity z-distance + rejection on consecutive imagined states ----
        B, H, _ = feat.shape
        f0 = feat[:, :-1].reshape(B * (H - 1), -1)
        f1 = feat[:, 1:].reshape(B * (H - 1), -1)
        mask, dist = wm.tcl.calculate_rejection_mask_and_distance_from_generated_outputs(f0, f1)
        mask = mask.reshape(B, H - 1)
        dist = dist.reshape(B, H - 1)

        # align: transition (t-1 -> t) describes generated state t (index 1..H-1)
        lin.append(dist.reshape(-1).cpu().numpy())
        rej.append(mask.reshape(-1).cpu().numpy())
        recon_err.append(re[:, 1:].reshape(-1).cpu().numpy())
        rew_err.append(we[:, 1:].reshape(-1).cpu().numpy())

    return (np.concatenate(lin), np.concatenate(rej).astype(bool),
            np.concatenate(recon_err), np.concatenate(rew_err),
            np.concatenate(residuals))


def report(lin, rej, recon_err, rew_err, residuals, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    outdir.mkdir(parents=True, exist_ok=True)
    # cache the collected arrays so figures can be re-plotted without a GPU pass
    np.savez(outdir / "e2_arrays.npz", lin=lin, rej=rej, recon_err=recon_err,
             rew_err=rew_err, residuals=residuals)

    # E2a: residual normality
    r = residuals[np.isfinite(residuals)]
    r = r[np.abs(r) < np.quantile(np.abs(r), 0.999)]  # clip extreme tail for plotting
    sk, ku = stats.skew(r), stats.kurtosis(r)  # excess kurtosis (0 = normal)
    print(f"[E2a] residual skew={sk:.3f} excess_kurtosis={ku:.3f} (0,0 = Gaussian)")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].hist(r, bins=120, density=True, alpha=0.8)
    ax[0].set_title(f"residual hist (skew {sk:.2f}, kurt {ku:.2f})")
    stats.probplot(r[:: max(1, len(r) // 50000)], dist="norm", plot=ax[1])
    ax[1].set_title("residual QQ vs normal")
    fig.tight_layout(); fig.savefig(outdir / "E2a_residual.png", dpi=130); plt.close(fig)

    # E2b: linearity error vs prediction error
    def corr(a, b, name):
        m = np.isfinite(a) & np.isfinite(b)
        pr = stats.pearsonr(a[m], b[m])[0]
        sp = stats.spearmanr(a[m], b[m])[0]
        print(f"[E2b] corr(linearity_z, {name}): pearson={pr:.3f} spearman={sp:.3f}")
        return pr, sp

    corr(lin, recon_err, "recon_error")
    corr(lin, rew_err, "reward_error")

    def binned(ax, x, y, nbins=12):
        """Overlay mean(y) per quantile-bin of x to expose the trend through the cloud."""
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        cx = 0.5 * (edges[:-1] + edges[1:])
        my = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan
                       for b in range(len(cx))])
        ax.plot(cx, my, "o-", color="crimson", lw=2, ms=4, label="binned mean")
        ax.legend(loc="upper left", fontsize=8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    s = max(1, len(lin) // 20000)
    ax[0].scatter(lin[::s], recon_err[::s], s=3, alpha=0.25)
    binned(ax[0], lin, recon_err)
    ax[0].set_xlabel("linearity z-distance"); ax[0].set_ylabel("recon error")
    ax[1].scatter(lin[::s], rew_err[::s], s=3, alpha=0.25)
    binned(ax[1], lin, rew_err)
    ax[1].set_xlabel("linearity z-distance"); ax[1].set_ylabel("reward error")
    fig.tight_layout(); fig.savefig(outdir / "E2b_corr.png", dpi=130); plt.close(fig)

    # E2c: anomalous vs normal transitions are worse one-step predictions (uGMF).
    # grp() compares err[mask] (flagged) vs err[~mask] (rest); guarded against
    # empty groups (a converged model rejects ~0% at the deployed k_sigma).
    def grp(err, name, mask):
        a, b = err[mask], err[~mask]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            print(f"[E2c] {name}: SKIP -- empty group (flagged n={len(a)}, rest n={len(b)})")
            return
        u = stats.mannwhitneyu(a, b, alternative="greater")
        print(f"[E2c] {name}: flagged mean={np.mean(a):.5f} (n={len(a)}) | "
              f"rest mean={np.mean(b):.5f} (n={len(b)}) | MannWhitney p={u.pvalue:.2e}")

    # (i) deployed-threshold view (may be degenerate on a converged ckpt)
    print(f"[E2c] deployed k_sigma=1.645 rejection rate = {rej.mean():.4f}")
    grp(recon_err, "recon_error @k1.645", rej)
    grp(rew_err, "reward_error @k1.645", rej)

    # (ii) threshold-free causal test: most-non-linear top-q% vs the rest.
    # Always non-empty; ranks transitions by linearity z so it is robust to the
    # absolute z-offset and to checkpoint maturity.
    finite = np.isfinite(lin)
    for q in (0.01, 0.05):
        thr = float(np.quantile(lin[finite], 1 - q))
        topmask = lin > thr
        print(f"[E2c'] top-{int(q * 100)}% most non-linear (z>{thr:.2f}):")
        grp(recon_err, f"  recon_error top{int(q * 100)}", topmask)
        grp(rew_err, f"  reward_error top{int(q * 100)}", topmask)

    # E2e: post-hoc k_sigma sweep on the SAME fixed checkpoint.
    # lin is the z-scored linearity distance (losses.py: (err-mu)/std), and the
    # training rule rejects when z > k_sigma, so (lin > ks) reproduces the rule
    # for any threshold ks. Independently-trained runs cannot give a clean curve
    # (their z-distributions diverge); thresholding ONE fixed ckpt does. (3kMR)
    l = lin[np.isfinite(lin)]
    ks_grid = [0.5, 1.0, 1.5, 1.645, 2.0, 2.5, 3.0]
    rates = [float((l > ks).mean()) for ks in ks_grid]
    print("[E2e] post-hoc rejection rate vs k_sigma (fixed ckpt):")
    for ks, rate in zip(ks_grid, rates):
        tag = "  <-- deployed" if abs(ks - 1.645) < 1e-6 else ""
        print(f"        k_sigma={ks:<6} rejection_rate={rate:.4f}{tag}")
    print(f"[E2e] sanity: live rej.mean()={rej.mean():.4f} vs swept@1.645={(l > 1.645).mean():.4f} "
          "(should match)")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks_grid, rates, "o-")
    ax.axvline(1.645, ls="--", c="grey", lw=1, label="deployed k=1.645")
    ax.set_xlabel(r"$k_\sigma$ (rejection threshold)")
    ax.set_ylabel("rejection rate")
    ax.set_title("Post-hoc rejection rate vs $k_\\sigma$ (fixed checkpoint)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "E2e_ksigma_sweep.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--context", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # deterministic batches + predictor sampling so reported stats are reproducible
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args.task, args.logdir, overrides={"device": args.device})
    agent, dataset, env = load_agent(config)
    wm = agent._wm
    calibrate_tcl(wm, dataset)
    data = collect(wm, dataset, args.n_batches, args.context)
    outdir = pathlib.Path(__file__).resolve().parent / "figs" / pathlib.Path(args.logdir).name
    report(*data, outdir)
    print(f"figures -> {outdir}")
    env.close()


if __name__ == "__main__":
    main()
