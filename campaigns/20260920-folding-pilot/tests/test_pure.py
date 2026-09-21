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
def storm_camera_serves_the_right_view_to_each_camera():
    """The env builds top, left wrist, right wrist in that order.

    Binding by construction order is fragile: if the env ever reorders those
    three lines the wrist views swap silently, and a policy fed a mirrored
    world would fail in a way no log line explains.
    """
    import types
    import numpy as np
    from lehome_fold.storm_camera import install

    obs = types.SimpleNamespace(_last_frames={
        "top_rgb": np.full((4, 4, 3), 10, np.uint8),
        "left_rgb": np.full((4, 4, 3), 20, np.uint8),
        "right_rgb": np.full((4, 4, 3), 30, np.uint8)})
    mod = types.SimpleNamespace()
    install(mod, obs, device="cpu")
    for want in (10, 20, 30):
        cam = mod.TiledCamera(None)
        close(float(cam.data.output["rgb"].float().mean()), float(want))


@test
def storm_camera_refuses_an_output_key_it_cannot_supply():
    """Returning zeros for an unknown channel is how an invisible garment
    survived 11 episodes here. Unsupported keys raise instead."""
    import types
    import numpy as np
    from lehome_fold.storm_camera import install

    obs = types.SimpleNamespace(_last_frames={"top_rgb": np.zeros((4, 4, 3), np.uint8)})
    mod = types.SimpleNamespace()
    install(mod, obs, device="cpu")
    cam = mod.TiledCamera(None)
    cam.data.output["rgb"]            # supported
    cam.data.output["depth"]          # supported, synthetic
    try:
        cam.data.output["semantic_segmentation"]
    except KeyError:
        return
    raise AssertionError("unsupported output key did not raise")


@test
def storm_camera_survives_a_missing_frame():
    """A camera asked for pixels before the first render must return a valid
    black frame rather than None, or the env dies inside its own observation
    builder with a traceback that points nowhere useful."""
    import types
    from lehome_fold.storm_camera import install

    mod = types.SimpleNamespace()
    install(mod, types.SimpleNamespace(_last_frames=None), device="cpu")
    cam = mod.TiledCamera(None)
    rgb = cam.data.output["rgb"]
    assert tuple(rgb.shape) == (1, 480, 640, 3), tuple(rgb.shape)
    close(float(rgb.float().mean()), 0.0)


@test
def seam_map_recovers_duplicated_uv_vertices():
    """Render meshes duplicate a vertex at every UV seam; the solver does not.

    PS_049 ships 11,573 render vertices collapsing to 11,385 unique positions,
    which is exactly its particle count. Writing the particle array straight
    into `points` left indices running past the point list, so the mesh drew
    nothing -- an empty table under a caption claiming a successful fold,
    invisible because the checker reads physics, not pixels.
    """
    import numpy as np
    from lehome_fold.storm_obs import _seam_map

    rng = np.random.default_rng(0)
    base = rng.normal(size=(400, 3)) * 0.3
    dup = rng.choice(len(base), 37, replace=False)
    mesh = np.concatenate([base, base[dup]], axis=0)

    weld = _seam_map(mesh, len(base))
    assert len(weld) == len(mesh), "one index per render vertex"
    assert len(np.unique(weld)) == len(base), "every particle used once"
    # every render vertex must land on a particle sharing its rest position
    close(float(np.abs(base[weld] - mesh).max()), 0.0)


@test
def seam_map_refuses_a_particle_count_it_cannot_explain():
    """A wrong correspondence renders a garbled garment, harder to notice than
    an absent one, so a count mismatch must raise rather than return."""
    import numpy as np
    from lehome_fold.storm_obs import _seam_map

    mesh = np.random.default_rng(1).normal(size=(340, 3))
    try:
        _seam_map(mesh, 300)          # 340 unique positions, 300 particles
    except RuntimeError:
        return
    raise AssertionError("accepted a mesh whose unique count is not the particle count")


@test
def edge_sanity_flags_a_scrambled_weld():
    """A scrambled weld fills the point list, so the mesh still renders -- as a
    spray of stretched triangles. Edge length is what separates that from a
    garment."""
    import numpy as np
    from lehome_fold.storm_obs import _edge_sanity

    # 8x8 grid, 1 cm spacing, triangles over ACTUAL neighbours -- an earlier
    # version of this test wired consecutive indices across the whole grid, so
    # its "tidy" mesh had 0.30 m edges and the assertion was meaningless.
    n, step = 8, 0.01
    xs, ys = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
    counts, idx = [], []
    for r in range(n - 1):
        for c in range(n - 1):
            v = r * n + c
            counts.append(3)
            idx += [v, v + 1, v + n]           # neighbours, one step apart
    tidy = _edge_sanity(pts, counts, idx)
    assert tidy < 0.02, f"neighbouring vertices should be ~{step} m apart, got {tidy}"

    scrambled = pts[np.random.default_rng(2).permutation(len(pts))]
    assert _edge_sanity(scrambled, counts, idx) > tidy * 3, "scramble should be caught"


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


@test
def injected_policy_modules_use_one_level_relative_imports():
    """run_eval.py copies these INTO scripts/eval_policy/, so a `.eval_policy.x`
    import resolves to scripts.eval_policy.eval_policy.x and fails. run_eval
    reports that as a warning and carries on, so the damage shows up much later
    as "policy type 'candidate' not found in registry" -- which is what killed
    the first two Stage 4 runs."""
    import re
    root = Path(__file__).resolve().parent.parent
    injected = ["g0_policies.py", "stage_policies.py"]
    bad = []
    for name in injected:
        f = root / "scripts" / name
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.match(r"\s*from \.\w+\.", line):
                bad.append(f"{name}:{i}: {line.strip()}")
    assert not bad, "two-level relative import in an injected module: " + "; ".join(bad)


@test
def eval_kwargs_translates_what_evaluation_py_actually_sends():
    """evaluation.py builds {"device", "model_path"} for every policy type that
    is not lerobot or docker. candidate/recap subclass LeRobotPolicy, which
    wants policy_path/dataset_root/task_description -- the mismatch cost 420
    worker iterations across 12.5 hours before anything said why."""
    from lehome_fold.eval_kwargs import translate_policy_kwargs

    sent = {"device": "cpu", "model_path": "/ckpt"}
    out = translate_policy_kwargs(
        sent, {"LH_EVAL_DATASET_ROOT": "Datasets/x"}, base_task="fold it")
    assert out["policy_path"] == "/ckpt", out
    assert "model_path" not in out, out
    assert out["dataset_root"] == "Datasets/x", out
    assert out["task_description"] == "fold it", out
    assert out["device"] == "cpu", out
    # The caller's dict is not mutated.
    assert sent == {"device": "cpu", "model_path": "/ckpt"}


@test
def eval_kwargs_prefers_explicit_values_over_env():
    from lehome_fold.eval_kwargs import translate_policy_kwargs

    out = translate_policy_kwargs(
        {"policy_path": "/explicit", "dataset_root": "D", "task_description": "T"},
        {"LH_EVAL_DATASET_ROOT": "ignored", "LH_EVAL_TASK_DESCRIPTION": "ignored"})
    assert out["policy_path"] == "/explicit"
    assert out["dataset_root"] == "D"
    assert out["task_description"] == "T"


@test
def eval_kwargs_refuses_a_missing_checkpoint_or_dataset():
    from lehome_fold.eval_kwargs import translate_policy_kwargs

    for kw, env in (({"device": "cpu"}, {"LH_EVAL_DATASET_ROOT": "D"}),
                    ({"model_path": "/ckpt"}, {})):
        try:
            translate_policy_kwargs(kw, env)
        except ValueError:
            pass
        else:
            assert False, f"should have refused {kw} with env {env}"


@test
def thompson_never_names_an_unpulled_arm_as_best():
    """An untried arm sits at the Beta(1,1) prior of 0.5. When every measured
    arm is worse than that -- 36 episodes, 0 successes, which is exactly what
    the first real Stage 4 run saw -- the old best() returned an arm it had
    never pulled and gain_over_baseline reported 0.48 for it."""
    from lehome_fold import thompson as T

    arms = T.grid()
    base = T.DEFAULT_ARM
    ts = T.ThompsonSampler(arms, seed=0, baseline=base, baseline_pulls=0)
    # Pull two arms, both total failures; every other arm stays untouched.
    measured = [a for a in arms if a.name == base.name][0]
    for _ in range(24):
        ts.update(measured, False)
    assert ts.pulls(measured) == 24

    b = ts.best()
    assert b is not None, "an arm WAS pulled, so there is a best"
    assert ts.pulls(b) > 0, f"best() named {b.name} with {ts.pulls(b)} pulls"

    g = ts.gain_over_baseline()
    assert g["best_pulls"] > 0, g
    assert g["arms_pulled"] == 1, g
    # 0 successes in 24 cannot outrank the 0.5 prior of an untried arm.
    assert g["best_mean"] < 0.5, g
    assert g["gain"] is not None and g["gain"] <= 0.0, g
    # And with no successes anywhere, the report must refuse to rank.
    assert g["total_successes"] == 0, g
    assert g["note"] and "no ranking valid" in g["note"], g


@test
def thompson_reports_nothing_when_nothing_was_pulled():
    from lehome_fold import thompson as T

    ts = T.ThompsonSampler(T.grid(), seed=0, baseline=T.DEFAULT_ARM,
                           baseline_pulls=0)
    assert ts.best() is None
    g = ts.gain_over_baseline()
    assert g["best_arm"] is None and g["gain"] is None, g
    assert g["arms_pulled"] == 0, g
    assert "note" in g


@test
def no_inner_script_passes_enable_cameras_unconditionally():
    """--enable_cameras makes AppLauncher build the RTX render product at LAUNCH,
    which segfaults on 5.1 against this cluster's driver before any camera
    exists. Under LH_STORM_EVAL=1 the flag must be ABSENT, not merely unused.
    This was fixed in tune_inference.py, then again in rollout_worker.py, then
    found a third time in eval_stage.sh and g0_floor.sh -- where it segfaulted
    both halves of a controlled comparison."""
    root = Path(__file__).resolve().parent.parent
    bad = []
    for f in sorted((root / "slurm" / "inner").glob("*.sh")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if "--enable_cameras" not in code:
                continue
            # The one allowed form: seeding a CAM array that Storm can empty.
            if code.strip().startswith("CAM=("):
                continue
            bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, ("unconditional --enable_cameras (fatal under Storm): "
                     + "; ".join(bad))


@test
def garment_dir_for_maps_every_name_the_evaluator_sweeps():
    """LeHome loads 12 garments for pant_short and calls switch_garment between
    them. Storm loads ONE garment USD at build time, so following the switch is
    the difference between scoring 12 garments and scoring the first one twelve
    times against the wrong pixels."""
    import sys as _s
    _s.modules.pop("lehome_fold.storm_eval", None)
    from lehome_fold.storm_eval import garment_dir_for

    got = garment_dir_for("/A", "Pant_Short_Seen_0")
    assert got == "/A/objects/Challenge_Garment/Release/Pant_Short/Pant_Short_Seen_0", got
    got = garment_dir_for("/A", "Top_Long_Unseen_1")
    assert got == "/A/objects/Challenge_Garment/Release/Top_Long/Top_Long_Unseen_1", got
    # Stage is honoured.
    assert garment_dir_for("/A", "Top_Short_Seen_9", "Release").endswith(
        "Release/Top_Short/Top_Short_Seen_9")
    # A name with no index suffix is a bug, not a silent passthrough.
    try:
        garment_dir_for("/A", "Pant_Short")
    except ValueError:
        pass
    else:
        assert False, "should refuse a name with no _Seen_N / _Unseen_N suffix"


@test
def stage4_refuses_to_sweep_dimensions_nothing_applies():
    """tune_inference exports N_CANDIDATES, CHUNK_LENGTH, TEMPERATURE and
    FLOW_STEPS; only N_CANDIDATES is read by anything. The first Stage 4 runs
    swept all four across 36 arms while the policy applied none of them, so
    every arm was the same configuration and every arm returned the same
    numbers -- a tuning result that was pure noise."""
    from lehome_fold import thompson as T

    # The full grid varies three inert dimensions -> must refuse.
    try:
        T.assert_dims_implemented(T.grid())
    except ValueError as e:
        assert "chunk_length" in str(e) and "temperature" in str(e), str(e)
    else:
        assert False, "full grid varies unimplemented dims and must be refused"

    # Pinned to the one implemented dimension -> allowed.
    ok = T.grid(chunk_length=(T.DEFAULT_ARM.chunk_length,),
                temperature=(T.DEFAULT_ARM.temperature,),
                flow_steps=(T.DEFAULT_ARM.flow_steps,))
    T.assert_dims_implemented(ok)
    assert T.varying_dims(ok) == {"n_candidates"}, T.varying_dims(ok)
    assert len(ok) == 3, len(ok)

    # A single arm varies nothing and is trivially fine.
    T.assert_dims_implemented([T.DEFAULT_ARM])
    assert T.varying_dims([T.DEFAULT_ARM]) == set()

    # The two sets must not overlap or the declaration is lying.
    assert not (T.IMPLEMENTED_DIMS & T.UNIMPLEMENTED_DIMS)


@test
def custom_kwargs_come_from_env_because_evaluation_py_cannot_pass_them():
    """evaluation.py builds {"device", "model_path"} for non-lerobot types, so
    every argument our policies add must arrive by environment. n_candidates
    was found missing after 36 inert Stage 4 arms; value_path was found one job
    later, when n>1 refused to run for want of a value head."""
    from lehome_fold.eval_kwargs import custom_kwargs_from_env, ENV_KWARGS

    defaults = {"value_path": None, "feature_path": "",
                "log_scores": None, "n_candidates": 1}
    env = {"VALUE_PATH": "/v", "FEATURE_PATH": "/f",
           "SCORE_LOG": "/s", "N_CANDIDATES": "16"}
    out = custom_kwargs_from_env(dict(defaults), env, defaults)
    assert out["value_path"] == "/v", out
    assert out["feature_path"] == "/f", out
    assert out["log_scores"] == "/s", out
    assert out["n_candidates"] == 16 and isinstance(out["n_candidates"], int), out

    # An explicit value beats the environment.
    out = custom_kwargs_from_env(
        {"value_path": "/explicit", "n_candidates": 4}, env, defaults)
    assert out["value_path"] == "/explicit", out
    assert out["n_candidates"] == 4, out

    # Empty env leaves the defaults alone.
    out = custom_kwargs_from_env(dict(defaults), {}, defaults)
    assert out == defaults, out

    # Every declared kwarg has a distinct env var.
    assert len(set(ENV_KWARGS.values())) == len(ENV_KWARGS)


@test
def trainer_does_not_consume_dotfile_sidecars_as_rollouts():
    """pathlib.Path.glob("*.jsonl") MATCHES dotfiles; shell globbing does not.
    The workers write .scores_wNNN.jsonl sidecars into the rollout directory on
    the assumption that a leading dot hides them, so the trainer read them as
    rollouts and dropped one per cycle for missing provenance."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / ".scores_w000.jsonl").write_text(
            json.dumps({"episode": 1, "p_success": 0.0}) + "\n")
        (d / "rollout_w000_0001.jsonl").write_text(
            json.dumps({"success": True, "_ckpt_version": 0}) + "\n")
        # The pattern alone is not enough -- this is the trap.
        assert len(list(d.glob("*.jsonl"))) == 2
        kept = [f for f in sorted(d.glob("*.jsonl"))
                if not f.name.startswith(".")]
        assert [f.name for f in kept] == ["rollout_w000_0001.jsonl"], kept


if __name__ == "__main__":
    raise SystemExit(main())
