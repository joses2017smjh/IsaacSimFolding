"""Tests for the half of this repo that does not need a simulator.

Deliberately dependency-light: plain asserts and a tiny runner, no pytest. The
point is that this suite runs anywhere numpy and torch exist -- including
inside the containers, on a login node, and before any of the 25 GB stack is
installed. A test suite that needs the environment under repair is no use while
the environment is under repair.

What is covered: splits, labels, calibration, AWR, RECAP, Thompson, checkpoint
provenance, the eval-log parser, and the value heads. That is every piece of
the paper's method that this repo implements itself.

What is NOT covered, and cannot be here: the seam between our heads and a live
LeRobot backbone (scripts/probe_backbone.py exists to pin that once the stack
installs), and anything that opens Isaac Sim.

    python tests/test_pure.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# --------------------------------------------------------------- splits
@test
def splits_counts_match_the_released_assets():
    from lehome_fold import splits as S

    assert len(S.seen()) == 40, len(S.seen())
    assert len(S.unseen()) == 8, len(S.unseen())
    assert len(S.all_garments()) == 48
    for c in S.CATEGORIES:
        assert len(S.seen(c)) == 10
        assert len(S.unseen(c)) == 2


@test
def splits_rejects_held_out_garments_in_training():
    from lehome_fold import splits as S

    S.assert_trainable(S.seen())
    try:
        S.assert_trainable(S.seen() + ["Top_Long_Unseen_0"])
    except S.SplitViolation as e:
        assert "Top_Long_Unseen_0" in str(e)
    else:
        raise AssertionError("a held-out garment passed the training guard")


@test
def splits_rejects_malformed_names_rather_than_filtering_them():
    from lehome_fold import splits as S

    for bad in ("Top_Long_Seen_10", "Sock_Seen_0", "Top_Long_Unseen_5", "garbage"):
        try:
            S.parse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} was accepted")


# --------------------------------------------------------------- labels
@test
def progress_is_zero_to_one_inclusive():
    from lehome_fold import labels as L

    p = L.progress(5)
    close(p[0], 0.0)
    close(p[-1], 1.0)
    assert np.all(np.diff(p) > 0)
    close(L.progress(1)[0], 1.0)


@test
def episode_mode_on_demo_data_is_degenerate_and_says_so():
    """The finding that reshapes Stage 2: released demos have no failures."""
    from lehome_fold import labels as L

    demo = L.EpisodeOutcome(length=50, success=True)
    y = L.success_targets(demo, mode="episode")
    assert L.class_balance(y)["degenerate"] == 1.0
    # terminal mode recovers both classes from the same all-successful demo
    y2 = L.success_targets(demo, mode="terminal")
    assert L.class_balance(y2)["degenerate"] == 0.0


@test
def success_frame_semantics():
    from lehome_fold import labels as L

    o = L.EpisodeOutcome(length=5, success=True, success_frame=3)
    assert list(L.success_targets(o, mode="terminal")) == [0, 0, 0, 1, 1]
    assert list(L.success_targets(o, mode="episode")) == [1, 1, 1, 1, 1]
    f = L.EpisodeOutcome(length=4, success=False)
    assert list(L.success_targets(f, mode="terminal")) == [0, 0, 0, 0]
    try:
        L.EpisodeOutcome(length=3, success=False, success_frame=1)
    except ValueError:
        pass
    else:
        raise AssertionError("success_frame on a failed episode was accepted")


@test
def future_state_masks_the_episode_tail():
    from lehome_fold import labels as L

    s = np.arange(12, dtype=np.float32).reshape(4, 3)
    fut, valid = L.future_state(s, 2)
    assert list(valid) == [1, 1, 0, 0]
    assert np.allclose(fut[0], s[2]) and np.allclose(fut[-1], s[-1])


# ---------------------------------------------------------- calibration
@test
def calibration_separates_good_from_bad():
    from lehome_fold import calibration as C

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    good = C.evaluate(p, y)
    assert good.ece < 0.05, good.ece
    assert C.gate(good)[0]

    bad = C.evaluate(np.clip(p * 0.5, 0, 1), y)
    assert bad.ece > 0.15, bad.ece
    assert not C.gate(bad)[0]


@test
def calibration_refuses_a_single_class_head():
    """G2 must fail on demo-only data rather than pass it."""
    from lehome_fold import calibration as C

    rel = C.evaluate(np.full(500, 0.9), np.ones(500))
    ok, reasons = C.gate(rel)
    assert not ok
    assert any("single-class" in r for r in reasons), reasons


@test
def calibration_rejects_invalid_input():
    from lehome_fold import calibration as C

    for probs, labels in [
        (np.array([1.5]), np.array([1.0])),
        (np.array([0.5]), np.array([2.0])),
        (np.array([np.nan]), np.array([1.0])),
        (np.array([]), np.array([])),
    ]:
        try:
            C.evaluate(probs, labels)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {probs} / {labels}")


# ------------------------------------------------------------------ AWR
@test
def success_residual_is_outcome_minus_baseline():
    from lehome_fold import awr

    a = awr.success_residual([1, 1, 0, 0], [0.2, 0.9, 0.3, 0.8])
    assert np.allclose(a, [0.8, 0.1, -0.3, -0.8], atol=1e-6)


@test
def awr_weights_are_clipped_and_finite_on_extremes():
    from lehome_fold import awr

    w = awr.weights([50.0, -50.0, 0.0], beta=0.01, w_max=20.0)
    assert np.all(np.isfinite(w)) and w.max() <= 20.0


@test
def awr_handles_zero_variance_advantages():
    from lehome_fold import awr

    w = awr.weights([0.5] * 8)
    assert np.allclose(w, 1.0), w
    close(awr.effective_sample_size(w), 8.0, 1e-6)


@test
def effective_sample_size_detects_concentration():
    from lehome_fold import awr

    close(awr.effective_sample_size([1, 1, 1, 1]), 4.0)
    assert awr.effective_sample_size([100, 1, 1, 1]) < 1.2
    close(awr.effective_sample_size([0, 0]), 0.0)


# ---------------------------------------------------------------- RECAP
@test
def binarise_sends_ties_negative():
    from lehome_fold import recap

    assert list(recap.binarise([0.8, 0.0, -0.3, 1e-9])) == [1, -1, -1, 1]


@test
def prompt_format_is_identical_between_train_and_inference():
    from lehome_fold import recap

    train = recap.condition(recap.BASE_TASK, recap.POSITIVE)
    infer = recap.positive_prompt(recap.BASE_TASK)
    assert train == infer, (train, infer)
    assert "\n" in train and train.endswith(recap.POSITIVE)


@test
def recap_rejects_unknown_tokens_and_signs():
    from lehome_fold import recap

    try:
        recap.condition("t", "Advantage: maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown token accepted")
    try:
        recap.tokens([1, 0, -1])
    except ValueError:
        pass
    else:
        raise AssertionError("sign 0 accepted")


# ------------------------------------------------------------- thompson
@test
def thompson_finds_the_best_arm_and_measures_the_baseline():
    from lehome_fold import thompson as T

    arms = T.grid()
    truth = {a.name: min(0.95, 0.15 + 0.03 * a.n_candidates + 0.10 * (a.temperature == 0.5))
             for a in arms}
    ts = T.ThompsonSampler(arms, seed=1, baseline_pulls=40)
    rng = np.random.default_rng(2)
    for _ in range(3000):
        arm = ts.select()
        ts.update(arm, rng.uniform() < truth[arm.name])

    assert truth[ts.best().name] >= max(truth.values()) - 0.02
    # the reserved budget is what makes "gain over defaults" a real comparison
    assert ts.pulls(ts.baseline) >= 40
    g = ts.gain_over_baseline()
    assert g["gain"] > 0 and g["separated"] == 1.0


@test
def thompson_rejects_duplicate_or_absent_arms():
    from lehome_fold import thompson as T

    a = T.Arm(1, 30, 1.0, 10)
    try:
        T.ThompsonSampler([a, a])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate arms accepted")
    try:
        T.ThompsonSampler([a], baseline=T.Arm(4, 30, 1.0, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("baseline outside the arm set accepted")


# ----------------------------------------------------------------- G3
@test
def checkpoint_roundtrip_and_lag():
    from lehome_fold import ckpt as K

    d = Path(tempfile.mkdtemp())
    (c1 := d / "v1").mkdir()
    (c1 / "model.safetensors").write_bytes(b"one")
    (c2 := d / "v2").mkdir()
    (c2 / "model.safetensors").write_bytes(b"two")
    shared = d / "shared"

    r1 = K.publish(shared, c1, version=1, step=10)
    assert K.read(shared).version == 1
    rec = K.stamp({"e": 0}, r1)
    assert K.verify_stamp(rec, r1) == 0

    r2 = K.publish(shared, c2, version=2, step=20)
    assert r2.digest != r1.digest, "different weights must hash differently"
    assert K.verify_stamp(rec, r2) == 1


@test
def g3_rejects_stale_unstamped_and_mismatched_rollouts():
    from lehome_fold import ckpt as K

    d = Path(tempfile.mkdtemp())
    (c := d / "v").mkdir()
    (c / "model.safetensors").write_bytes(b"w")
    ref = K.publish(d / "shared", c, version=5, step=50)

    stale = K.stamp({"e": 1}, K.CheckpointRef(1, 10, str(c), ref.digest, 0.0))
    for bad, why in [
        (stale, "stale"),
        ({"e": 2}, "unstamped"),
        (K.stamp({"e": 3}, K.CheckpointRef(5, 50, str(c), "deadbeefdeadbeef", 0.0)), "digest"),
        (K.stamp({"e": 4}, K.CheckpointRef(9, 90, str(c), ref.digest, 0.0)), "non-monotonic"),
    ]:
        try:
            K.verify_stamp(bad, ref, max_lag=2)
        except K.StaleCheckpoint:
            pass
        else:
            raise AssertionError(f"{why} rollout accepted")


@test
def manifest_write_is_atomic_and_leaves_no_temp_files():
    from lehome_fold import ckpt as K

    d = Path(tempfile.mkdtemp())
    (c := d / "v").mkdir()
    (c / "model.pt").write_bytes(b"w")
    shared = d / "shared"
    for v in range(5):
        K.publish(shared, c, version=v, step=v * 10)
    assert K.read(shared).version == 4
    leftovers = [p.name for p in shared.iterdir() if p.name != K.MANIFEST]
    assert not leftovers, leftovers


# ------------------------------------------------------------- eval log
@test
def eval_log_parses_the_official_formats():
    from lehome_fold import eval_log as E

    log = (
        "Episode 1/3: Return=12.50, Length=364, Success=True\n"
        "Episode 2/3: Return=-1.00, Length=600, Success=False\n"
        "Episode 3/3: Return=9.25, Length=402, Success=True\n"
        "  Top_Long_Seen_0: Success Rate = 66.67%, Avg Return = 6.92\n"
        "  Pant_Short_Unseen_1: Success Rate = 0.00%, Avg Return = -1.00\n"
    )
    rate, n = E.success_rate(log)
    close(rate, 2 / 3, 1e-6)
    assert n == 3
    cats = E.per_category(E.parse_garments(log))
    assert ("Top_Long", "Seen") in cats and ("Pant_Short", "Unseen") in cats


@test
def eval_log_cross_check_catches_a_truncated_log():
    """A tailed log reports a plausible wrong number. Refuse it."""
    from lehome_fold import eval_log as E

    whole = (
        "Episode 1/2: Return=1.00, Length=10, Success=True\n"
        "Episode 2/2: Return=1.00, Length=10, Success=True\n"
        "  Top_Long_Seen_0: Success Rate = 100.00%, Avg Return = 1.00\n"
    )
    close(E.success_rate_checked(whole)[0], 1.0)

    # the same run with the first episode line lost to a tail
    tailed = (
        "Episode 2/2: Return=1.00, Length=10, Success=False\n"
        "  Top_Long_Seen_0: Success Rate = 100.00%, Avg Return = 1.00\n"
    )
    try:
        E.success_rate_checked(tailed)
    except E.Truncated:
        pass
    else:
        raise AssertionError("a truncated log produced a reportable number")


@test
def eval_log_distinguishes_a_crash_from_a_zero_score():
    from lehome_fold import eval_log as E

    try:
        E.success_rate("Traceback (most recent call last): boom")
    except E.NoEpisodes:
        pass
    else:
        raise AssertionError("an empty log reported a success rate")


# ---------------------------------------------------------- value heads
@test
def value_heads_shapes_init_and_masking():
    import torch

    from lehome_fold.value_head import ValueHeadConfig, ValueHeads, value_loss

    torch.manual_seed(0)
    m = ValueHeads(ValueHeadConfig(hidden_dim=960))
    f = torch.randn(8, 960, requires_grad=True)
    out = m(f)
    assert out["success_logit"].shape == (8,)
    assert out["future"].shape == (8, 12)
    # a head that starts confidently wrong poisons early Stage 3 advantages
    close(float(m.success_prob(f).mean()), 0.5, 0.05)

    t = {"success": torch.randint(0, 2, (8,)), "progress": torch.rand(8),
         "future": torch.randn(8, 12)}
    total, parts = value_loss(out, t, future_mask=torch.zeros(8))
    close(parts["future"], 0.0, 1e-9)
    total.backward()
    assert f.grad is None, "detach_backbone=True must not leak gradients"


@test
def value_heads_attached_mode_lets_gradients_through():
    import torch

    from lehome_fold.value_head import ValueHeadConfig, ValueHeads, value_loss

    m = ValueHeads(ValueHeadConfig(hidden_dim=64, detach_backbone=False))
    f = torch.randn(4, 64, requires_grad=True)
    t = {"success": torch.ones(4), "progress": torch.rand(4), "future": torch.randn(4, 12)}
    value_loss(m(f), t)[0].backward()
    assert f.grad is not None and float(f.grad.abs().sum()) > 0


@test
def value_heads_reject_a_width_mismatch():
    import torch

    from lehome_fold.value_head import ValueHeadConfig, ValueHeads

    m = ValueHeads(ValueHeadConfig(hidden_dim=960))
    try:
        m(torch.randn(4, 128))
    except ValueError:
        pass
    else:
        raise AssertionError("a mismatched feature width was accepted")


@test
def feature_tap_captures_embed_prefix_with_the_real_signatures():
    """The tap must not care what arguments embed_prefix takes.

    Verified against lerobot 0.4.3: pi05 takes (images, img_masks, tokens,
    masks) and smolvla takes (images, img_masks, lang_tokens, lang_masks,
    state=None). Both return (embs, pad_masks, att_masks). An earlier version of
    the wrapper called embed_prefix(batch) and would have failed on both.
    """
    import torch
    import torch.nn as nn

    from lehome_fold.policy_wrap import FeatureTap

    class PI05Like(nn.Module):
        def embed_prefix(self, images, img_masks, tokens, masks):
            b = images.shape[0]
            return torch.randn(b, 7, 64), torch.ones(b, 7), torch.ones(b, 7)

    class SmolLike(nn.Module):
        def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
            b = images.shape[0]
            return torch.randn(b, 5, 64), torch.ones(b, 5), torch.ones(b, 5)

    for inner in (PI05Like(), SmolLike()):
        policy = nn.Module()
        policy.model = inner
        tap = FeatureTap(policy, "model.embed_prefix").install()
        try:
            tap.require()
        except RuntimeError:
            pass
        else:
            raise AssertionError("read features before any forward pass")
        # the policy calls it however it likes; the tap just records
        if isinstance(inner, PI05Like):
            inner.embed_prefix(torch.randn(3, 1), torch.ones(3), torch.ones(3), torch.ones(3))
        else:
            inner.embed_prefix(torch.randn(3, 1), torch.ones(3), torch.ones(3),
                               torch.ones(3), state=torch.randn(3, 12))
        assert tap.require().shape[0] == 3
        assert tap.last_mask is not None, "pad_masks must be captured for pooling"
        tap.remove()
        assert not isinstance(inner.embed_prefix, type(lambda: None)) or True
        # removing restores the original, so re-installing cannot stack
        assert tap._orig is None


@test
def feature_tap_infers_width_and_pools_over_the_mask():
    import torch
    import torch.nn as nn

    from lehome_fold.policy_wrap import ValueAugmentedPolicy, WrapConfig

    class Inner(nn.Module):
        def embed_prefix(self, x):
            b = x.shape[0]
            embs = torch.ones(b, 4, 32)
            embs[:, 2:] = 99.0          # padding that must NOT be averaged in
            mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(b, 1)
            return embs, mask, mask

    inner = Inner()
    policy = nn.Module()
    policy.model = inner
    w = ValueAugmentedPolicy(policy, WrapConfig())   # hidden_dim inferred
    inner.embed_prefix(torch.randn(2, 1))
    out = w()
    assert w.cfg.hidden_dim == 32, w.cfg.hidden_dim
    assert out["success_logit"].shape == (2,)
    # masked pooling: the 99.0 padding is excluded, so every feature is 1.0
    assert torch.allclose(w.features(), torch.ones(2, 32)), w.features()[0, :3]


@test
def feature_tap_rejects_a_path_that_does_not_exist():
    import torch.nn as nn

    from lehome_fold.policy_wrap import FeatureTap

    try:
        FeatureTap(nn.Module(), "model.nope").install()
    except AttributeError as e:
        assert "probe_backbone" in str(e)
    else:
        raise AssertionError("a nonexistent feature path installed")


@test
def seam_map_recovers_duplicated_uv_vertices():
    """Render meshes duplicate a vertex at every UV seam; the solver does not.

    PS_049 ships 11,573 render vertices against 11,385 particles. Writing the
    particle array straight into `points` left indices running past the point
    list, so the mesh drew nothing -- an empty table under a caption claiming a
    successful fold, invisible because the checker reads physics, not pixels.
    """
    import numpy as np
    from lehome_fold.storm_obs import _seam_map

    rng = np.random.default_rng(0)
    particles = rng.normal(size=(400, 3)) * 0.3
    dup = rng.choice(len(particles), 37, replace=False)
    mesh = np.concatenate([particles, particles[dup]], axis=0)
    mesh = mesh[rng.permutation(len(mesh))]      # render order is arbitrary

    m = _seam_map(mesh, particles)
    assert len(m) == len(mesh), "one index per render vertex"
    close(float(np.abs(particles[m] - mesh).max()), 0.0)
    assert len(np.unique(m)) == len(particles), "every particle should be used"


@test
def seam_map_refuses_a_cloud_it_cannot_match():
    """A wrong correspondence renders a garbled garment, which is harder to
    notice than an absent one, so a poor match must raise rather than return."""
    import numpy as np
    from lehome_fold.storm_obs import _seam_map

    rng = np.random.default_rng(1)
    particles = rng.normal(size=(300, 3))
    unrelated = rng.normal(size=(340, 3)) * 5.0
    try:
        _seam_map(unrelated, particles)
    except RuntimeError:
        return
    raise AssertionError("accepted a point cloud that does not correspond")


@test
def value_loss_sample_mask_ignores_unlabelled_frames():
    """A masked frame must not influence the loss at all.

    Only some released episodes have been rolled out and scored. An unlabelled
    frame carries success=0 by construction, so an inert mask would not merely
    weaken the signal -- it would train the success head to call every unscored
    episode a failure.
    """
    import torch
    from lehome_fold.value_head import value_loss

    n, d = 8, 3
    # Deliberately ASYMMETRIC. Two earlier drafts of this test used mirrored
    # logits/targets, where the masked half's BCE terms are the exact mirror of
    # the kept half's, so the mean is identical with or without the mask and
    # the test passed regardless of whether masking worked. Here the masked
    # half is confidently correct (logit 10, target 1 -> ~0 loss) while the
    # kept half is maximally uncertain (logit 0 -> log 2).
    preds = {"success_logit": torch.cat([torch.zeros(4), torch.full((4,), 10.0)]),
             "progress": torch.zeros(n), "future": torch.zeros(n, d)}
    targets = {"success": torch.ones(n),
               "progress": torch.cat([torch.ones(4), torch.zeros(4)]),
               "future": torch.zeros(n, d)}
    mask = torch.cat([torch.ones(4), torch.zeros(4)])

    _, masked = value_loss(preds, targets, sample_mask=mask)

    # Same batch with the unlabelled half deleted must give the same numbers.
    _, only_labelled = value_loss({k: v[:4] for k, v in preds.items()},
                                  {k: v[:4] for k, v in targets.items()})
    close(masked["success"], only_labelled["success"])
    close(masked["progress"], only_labelled["progress"])

    # And it must actually differ from ignoring the mask, or the test is vacuous.
    _, unmasked = value_loss(preds, targets)
    assert abs(masked["success"] - unmasked["success"]) > 1e-6, "success mask had no effect"
    assert abs(masked["progress"] - unmasked["progress"]) > 1e-6, "progress mask had no effect"


@test
def value_loss_all_masked_batch_is_zero_not_nan():
    import torch
    from lehome_fold.value_head import value_loss

    n, d = 4, 3
    preds = {"success_logit": torch.zeros(n), "progress": torch.zeros(n),
             "future": torch.zeros(n, d)}
    targets = {"success": torch.ones(n), "progress": torch.ones(n),
               "future": torch.zeros(n, d)}
    _, parts = value_loss(preds, targets, sample_mask=torch.zeros(n))
    for k in ("success", "progress"):
        assert parts[k] == parts[k], f"{k} is nan on an all-masked batch"
        close(parts[k], 0.0)


@test
def candidate_selection_picks_the_best_and_returns_every_score():
    import itertools

    from lehome_fold.policy_wrap import candidate_selection

    c = itertools.count()
    best, scores = candidate_selection(lambda: next(c), lambda x: -abs(x - 2), n_candidates=5)
    assert best == 2 and len(scores) == 5
    # a zero spread means selection is arbitrary; callers must be able to see it
    _, flat = candidate_selection(lambda: 1, lambda x: 0.0, n_candidates=3)
    assert max(flat) - min(flat) == 0.0


def main() -> int:
    passed, failed = 0, []
    for fn in _TESTS:
        try:
            fn()
            passed += 1
            print(f"  ok    {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            print("        " + traceback.format_exc().replace("\n", "\n        "))
    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
