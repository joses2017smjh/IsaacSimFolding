"""Generate the component figures in the README.

Everything here is produced by RUNNING the code in src/lehome_fold, not drawn
by hand. Regenerate with:

    python scripts/make_figures.py --out docs/img

What these are: honest demonstrations of the mechanisms, driven by synthetic
outcomes with known ground truth. That is the point -- with known truth you can
show that the estimator recovers it, which a policy rollout cannot do.

What these are NOT: garment-folding results. There are none. Isaac Sim 5.1's
RTX renderer segfaults on this cluster, so no rollout footage exists, and
inventing some would be worse than having none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#21262d"
ACCENT = "#58a6ff"
GOOD = "#3fb950"
BAD = "#f85149"
WARN = "#d29922"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_calibration(out: Path) -> None:
    """G2: the gate that catches an uncalibrated success head."""
    from lehome_fold import calibration as C

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, 40000) < p).astype(float)

    cases = [
        ("calibrated", p, y, GOOD),
        ("miscalibrated", np.clip(p * 0.5, 0, 1), y, BAD),
        ("degenerate\n(demo-only data)", np.full(2000, 0.93), np.ones(2000), WARN),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (name, probs, labels, colour) in zip(axes, cases):
        rel = C.evaluate(probs, labels)
        ok, reasons = C.gate(rel)
        m = rel.counts > 0
        ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, zorder=1)
        ax.plot(rel.mean_pred[m], rel.mean_obs[m], "o-", color=colour, lw=2,
                ms=5, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted P(success)")
        ax.grid(alpha=.4, lw=.6)
        verdict = "G2 PASS" if ok else "G2 FAIL"
        ax.set_title(f"{name}\n{verdict}  ·  ECE {rel.ece:.3f}",
                     color=GOOD if ok else BAD)
        # The degenerate panel is ONE point, and an almost-empty plot reads as
        # "nothing here" rather than "this is the failure". Say what it is.
        if int(m.sum()) < 3:
            ax.annotate(
                "one populated bin:\nevery label is 1,\nevery prediction the\nsame. Accurate and\nuseless.",
                xy=(rel.mean_pred[m][0], rel.mean_obs[m][0]), xytext=(0.13, 0.42),
                color=WARN, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2))
            ax.text(.5, .06, "this is what the released demos give you",
                    transform=ax.transAxes, ha="center", color=MUTED, fontsize=8,
                    style="italic")
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("G2 — the success head must be calibrated, or Stage 3 advantages are noise",
                 color=FG, fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out / "g2_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_awr(out: Path) -> None:
    """AWR: why beta needs a guard, shown as effective sample size."""
    from lehome_fold import awr

    rng = np.random.default_rng(1)
    adv = rng.normal(0, 1, 256)
    betas = np.geomspace(0.05, 5.0, 40)
    ess = [awr.effective_sample_size(awr.weights(adv, beta=b)) for b in betas]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.semilogx(betas, ess, color=ACCENT, lw=2.2)
    a1.axhline(len(adv), ls="--", color=MUTED, lw=1)
    a1.axhline(0.05 * len(adv), ls="--", color=BAD, lw=1)
    a1.text(0.055, 0.05 * len(adv) + 6, "warn threshold", color=BAD, fontsize=8)
    a1.text(0.055, len(adv) - 22, "batch size 256", color=MUTED, fontsize=8)
    a1.set_xlabel("AWR temperature  β"); a1.set_ylabel("effective sample size")
    a1.set_title("Small β collapses the batch onto a few episodes")
    a1.grid(alpha=.4, lw=.6)

    for b, c in ((0.15, BAD), (1.0, ACCENT), (3.0, GOOD)):
        w = awr.weights(adv, beta=b)
        a2.hist(w, bins=40, alpha=.65, color=c,
                label=f"β={b}  ESS={awr.effective_sample_size(w):.0f}")
    a2.set_xlabel("per-sample weight  exp(A/β)"); a2.set_ylabel("count")
    a2.set_title("Weight distribution"); a2.legend(frameon=False, labelcolor=FG, fontsize=8)
    a2.grid(alpha=.4, lw=.6)
    fig.tight_layout()
    fig.savefig(out / "awr_ess.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_g3(out: Path) -> None:
    """G3: stale rollouts rejected rather than silently mislabelled."""
    from lehome_fold import ckpt as K

    ref = K.CheckpointRef(version=12, step=6000, path="/ckpt/v12",
                          digest="9410887cd599e776", written_at=0.0)
    rng = np.random.default_rng(3)
    versions = np.clip(12 - rng.geometric(0.55, 400) + 1, 0, 12)
    accepted, rejected = [], []
    for v in versions:
        rec = K.stamp({"e": 1}, K.CheckpointRef(int(v), 0, "", ref.digest, 0.0))
        try:
            accepted.append(K.verify_stamp(rec, ref, max_lag=2))
        except K.StaleCheckpoint:
            rejected.append(12 - int(v))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bins = np.arange(-0.5, 13.5)
    ax.hist(accepted, bins=bins, color=GOOD, label=f"accepted  ({len(accepted)})")
    ax.hist(rejected, bins=bins, color=BAD, label=f"rejected as stale  ({len(rejected)})")
    ax.axvline(2.5, ls="--", color=WARN, lw=1.4)
    ax.text(2.75, ax.get_ylim()[1] * .78, "max_lag = 2", color=WARN, fontsize=9)
    ax.set_xlabel("checkpoint versions behind the trainer")
    ax.set_ylabel("rollout episodes")
    ax.set_title("G3 — a stale worker's advantage labels are wrong against the right-looking data")
    ax.legend(frameon=False, labelcolor=FG)
    ax.grid(alpha=.4, lw=.6, axis="y")
    fig.tight_layout()
    fig.savefig(out / "g3_staleness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def gif_thompson(out: Path) -> None:
    """Stage 4: Thompson sampling separating arms, animated over the budget."""
    import imageio.v2 as imageio

    from lehome_fold import thompson as T

    arms = T.grid()
    truth = {a.name: min(0.92, 0.12 + 0.030 * a.n_candidates
                         + 0.10 * (a.temperature == 0.5)
                         + 0.02 * (a.flow_steps == 10)) for a in arms}
    ts = T.ThompsonSampler(arms, seed=1, baseline_pulls=40)
    rng = np.random.default_rng(7)

    best_true = max(truth.values())
    frames, hist = [], []
    tmp = out / "_frames"
    tmp.mkdir(exist_ok=True)

    for step in range(1, 1201):
        arm = ts.select()
        ts.update(arm, rng.uniform() < truth[arm.name])
        hist.append(truth[ts.best().name])
        if step % 40 or step < 80:
            if step % 40:
                continue

        order = sorted(arms, key=lambda a: -truth[a.name])
        pulls = np.array([ts.pulls(a) for a in order], dtype=float)
        means = np.array([ts.posterior_mean(a) for a in order])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8),
                                     gridspec_kw={"width_ratios": [2, 1.15]})
        x = np.arange(len(order))
        a1.bar(x, pulls, color=[GOOD if truth[a.name] > best_true - .02 else ACCENT
                                for a in order], width=.82)
        bl = [i for i, a in enumerate(order) if a.name == ts.baseline.name][0]
        a1.bar([bl], [pulls[bl]], color=WARN, width=.82)
        a1.set_xticks([]); a1.set_xlabel("36 inference-hyperparameter arms  →  worst")
        a1.set_ylabel("episodes spent")
        a1.set_title(f"Budget allocation after {step} episodes")
        a1.grid(alpha=.35, lw=.6, axis="y")
        a1.text(.99, .93, "green = truly best   amber = fixed defaults",
                transform=a1.transAxes, ha="right", color=MUTED, fontsize=8)

        a2.plot(np.arange(1, len(hist) + 1), hist, color=ACCENT, lw=2)
        a2.axhline(best_true, ls="--", color=GOOD, lw=1.2)
        a2.set_ylim(0.1, best_true + .06); a2.set_xlim(0, 1200)
        a2.set_xlabel("episodes"); a2.set_ylabel("true rate of chosen arm")
        a2.set_title("Converging on the best arm")
        a2.grid(alpha=.35, lw=.6)

        fig.suptitle("Stage 4 — Thompson sampling over inference hyperparameters "
                     "(synthetic outcomes, known ground truth)",
                     color=FG, fontsize=11, fontweight="bold", y=1.05)
        fig.tight_layout()
        f = tmp / f"{step:05d}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        frames.append(f)

    imgs = [imageio.imread(f) for f in frames]
    h = min(i.shape[0] for i in imgs)
    w = min(i.shape[1] for i in imgs)
    imgs = [i[:h, :w] for i in imgs]
    imageio.mimsave(out / "thompson.gif", imgs + [imgs[-1]] * 8, duration=0.22, loop=0)
    for f in frames:
        f.unlink()
    tmp.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_calibration(out)
    print("  g2_calibration.png")
    fig_awr(out)
    print("  awr_ess.png")
    fig_g3(out)
    print("  g3_staleness.png")
    gif_thompson(out)
    print("  thompson.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
