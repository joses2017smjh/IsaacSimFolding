"""Roll out a trained policy on Isaac Sim 5.1, with Storm supplying the RGB.

This is the assembly of every piece proven separately in this repo:

  cloth_sim51.py       the LeHome env runs on 5.1 with cameras stubbed, and
                       PhysX particle cloth actually simulates (sim.device
                       must be CUDA -- on CPU the particle solver silently
                       no-ops)
  probe_storm51.py     OpenUSD Storm rasterises on 5.1 without the RTX
                       delegate that segfaults on this cluster
  probe_storm_in_kit   ...and it does so INSIDE a live Kit process, so the
                       renderer can share a process with the simulation
  storm_obs.py         a stage that is built once and re-rendered per step

Each env step: read particles and link poses -> update the Storm stage ->
render the three challenge cameras -> hand them to the policy -> step.

The scorer stays LeHome's own `_get_success`, which is geometric over particle
positions, so the verdict on each episode is the challenge's and not mine.

The deviation, stated plainly and repeated wherever a number from this path is
reported: the policy is shown RASTERISED frames while it was trained on
PATH-TRACED demonstrations, and Storm cannot evaluate the MDL materials
Omniverse assets ship. If a policy scores poorly here, domain gap is a live
explanation and has to be ruled out before concluding anything about the
policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--lehome", required=True)
ap.add_argument("--policy_path", required=True)
ap.add_argument("--dataset_root", default=None,
                help="unused: kept so older job scripts still parse")
ap.add_argument("--garment", default="Top_Short_Seen_0")
ap.add_argument("--garment_dir", required=True)
ap.add_argument("--assets", required=True)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--frames_out", default="")
ap.add_argument("--match_pose", default="",
                help="'x:y:z:rx:ry:rz' -- pin the garment to this exact spawn "
                     "pose. Colon-separated, NOT comma: sbatch --export splits "
                     "its own argument on commas, so a comma-separated pose "
                     "arrives truncated to its first number.")
ap.add_argument("--match_scale", type=float, default=0.0,
                help="override garment scale (demos recorded at 0.45)")
ap.add_argument("--replay_parquet", default="",
                help="replay recorded demo actions instead of running the policy")
ap.add_argument("--replay_episode", type=int, default=0)
ap.add_argument("--settle_steps", type=int, default=0,
                help="physics steps holding the initial pose before the policy acts")
ap.add_argument("--shadow_policy", type=int, default=0,
                help="during replay, also record what the policy would do on "
                     "the SAME state, rendered by Storm")
ap.add_argument("--gif_every", type=int, default=3,
                help="keep every Nth rendered frame for the GIF")
ap.add_argument("--result_out", default="results/rollout.json")
ap.add_argument("--task", default="fold the garment on the table")
ap.add_argument("--sim_device", default="cuda:0")
args = ap.parse_args()


def log(*a):
    print("[rollout]", *a, flush=True)


# lehome/devices imports a teleop keyboard listener at package import time.
if "pynput" not in sys.modules:
    _p = types.ModuleType("pynput"); _k = types.ModuleType("pynput.keyboard")
    _k.Listener = type("L", (), {"__init__": lambda s,*a,**k: None,
                                 "start": lambda s: None, "stop": lambda s: None,
                                 "join": lambda s,*a,**k: None})
    _k.Key = type("K", (), {"__getattr__": lambda s,n: f"<{n}>"})()
    _k.KeyCode = type("KC", (), {"from_char": staticmethod(lambda c: c)})
    _p.keyboard = _k
    sys.modules["pynput"] = _p; sys.modules["pynput.keyboard"] = _k

from isaaclab.app import AppLauncher  # noqa: E402

# cameras OFF -- this is what keeps 5.1 away from the RTX delegate.
app = AppLauncher(headless=True, enable_cameras=False).app
log("5.1 app launched, cameras off")

os.chdir(args.lehome)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lehome_fold.storm_obs import StormObsConfig, StormObserver  # noqa: E402

rc, success, n_rendered = 3, False, 0
try:
    import lehome.tasks.bedroom  # noqa: F401
    import lehome.tasks.bedroom.garment_bi_v2 as gbv
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg
    from lehome.tasks.bedroom.garment_bi_v2 import GarmentEnv

    class _ZeroOut(dict):
        def __missing__(self, k):
            v = (torch.zeros((1, 480, 640, 1), dtype=torch.float32) if k == "depth"
                 else torch.zeros((1, 480, 640, 3), dtype=torch.uint8))
            self[k] = v
            return v

    class _NoCamera:
        def __init__(self, cfg=None, *a, **k):
            self.cfg = cfg
            self.data = types.SimpleNamespace(output=_ZeroOut())
        def update(self, *a, **k): pass
        def reset(self, *a, **k): pass
        def __del__(self): pass

    # The env's own cameras stay stubbed: Storm supplies the real images, and
    # asking Isaac Lab for a render product is what crashes 5.1.
    gbv.TiledCamera = _NoCamera

    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.garment_cfg_base_path = os.path.join(args.lehome, "Assets/objects/Challenge_Garment")
    particle_cfg = os.path.join(
        args.lehome, "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml")

    # Pin the garment's spawn pose and scale.
    #
    # The env samples the pose randomly from ranges in this YAML; the recorded
    # `object_initial_pose` in garment_info.json is only ever WRITTEN by
    # record.py, never read back. So replaying a demonstration's actions
    # against a freshly sampled pose drives a recorded trajectory at a garment
    # that is somewhere else -- which is why an open-loop replay folded one
    # axis (dist(p2,p3) 41.65 -> 3.27) and missed the other two. Collapsing
    # each range to a single point reproduces the episode the actions came
    # from.
    #
    # Scale matters for the same reason: the YAML says 0.4 while every
    # recorded episode says 0.45, an 11% size difference the demo motions were
    # never calibrated against.
    # Pin by overriding the sampler itself, not the YAML.
    #
    # Editing `objects.common` in the config only moved z (0.73 -> 0.67) and
    # left x, y and orientation random, because _get_config_value prefers the
    # per-garment `garment_config` handed to GarmentObject's constructor over
    # anything in `common`. Since the garment spawns during construction there
    # is no post-hoc hook either, so the reliable seam is the method that
    # draws the pose.
    if args.match_pose:
        from lehome.assets.object.Garment import GarmentObject
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        _v = [float(x) for x in args.match_pose.replace(",", ":").split(":")]
        if len(_v) != 6:
            raise ValueError(f"--match_pose needs 6 values, got {len(_v)}")
        _sc = args.match_scale

        def _pinned_pose(self):
            pos, ori = list(_v[:3]), list(_v[3:6])
            scale, _src = self._get_config_value("scale", "common")
            if _sc:
                scale = [_sc, _sc, _sc]
            self.reset_pose = np.concatenate(
                [np.array(pos, dtype=np.float32), np.array(ori, dtype=np.float32)])
            return pos, euler_angles_to_quat(ori, degrees=True), scale

        GarmentObject._get_initial_pose = _pinned_pose

        # Overriding _get_initial_pose alone is not enough: env.reset() runs
        # GarmentObject.reset(), which re-samples from soft_reset_pos_range /
        # soft_reset_rot_range and moves the garment again. Both those ranges
        # and the initial ones resolve through _get_config_value, and all of
        # them prefer the per-garment config, so this is the one seam that
        # controls every path. Collapsing each range to a single point also
        # makes reset()'s own min==max check treat it as a zero range and skip
        # repositioning, which leaves the garment exactly where the pinned
        # spawn put it.
        _orig_cfg_value = GarmentObject._get_config_value

        def _pinned_cfg_value(self, field_name, default_source="common"):
            if field_name in ("initial_pos_range", "soft_reset_pos_range"):
                return [_v[0], _v[1], _v[2], _v[0], _v[1], _v[2]], "pinned"
            if field_name in ("initial_rot_range", "soft_reset_rot_range"):
                return [_v[3], _v[4], _v[5], _v[3], _v[4], _v[5]], "pinned"
            if field_name == "scale" and _sc:
                return [_sc, _sc, _sc], "pinned"
            return _orig_cfg_value(self, field_name, default_source)

        GarmentObject._get_config_value = _pinned_cfg_value
        log(f"PINNED sampler+config pos={_v[:3]} ori={_v[3:6]} scale={_sc or 'config'}")

    if args.match_scale and not args.match_pose:
        import yaml as _yaml
        with open(particle_cfg) as fh:
            _cfg = _yaml.safe_load(fh)
        common = _cfg.setdefault("objects", {}).setdefault("common", {})
        if args.match_pose:
            v = [float(x) for x in args.match_pose.replace(",", ":").split(":")]
            if len(v) != 6:
                raise ValueError(
                    f"--match_pose needs 6 colon-separated values, got {len(v)}: "
                    f"{args.match_pose!r}")
            common["initial_pos_range"] = [v[0], v[1], v[2], v[0], v[1], v[2]]
            common["initial_rot_range"] = [v[3], v[4], v[5], v[3], v[4], v[5]]
            common["soft_reset_pos_range"] = list(common["initial_pos_range"])
            common["soft_reset_rot_range"] = list(common["initial_rot_range"])
        if args.match_scale:
            common["scale"] = [args.match_scale] * 3
        particle_cfg = os.path.join(args.frames_out or ".", "particle_pinned.yaml")
        with open(particle_cfg, "w") as fh:
            _yaml.safe_dump(_cfg, fh)
        log(f"PINNED pose={args.match_pose or 'unchanged'} "
            f"scale={args.match_scale or 'unchanged'} -> {particle_cfg}")
    cfg.particle_cfg_path = particle_cfg
    cfg.sim.device = args.sim_device      # particle cloth needs GPU dynamics
    cfg.scene.num_envs = 1
    env = GarmentEnv(cfg=cfg)
    obj = env.object

    # Initialise the cloth prim view against the LIVE physics sim view before
    # anything reads particle positions. On GPU, _get_initial_info() goes
    # straight to _cloth_prim_view.get_world_positions(), and that view has no
    # _max_particles_per_cloth until it is bound to a sim view -- DirectRLEnv
    # never binds it for this object. Skipping this is what made the first
    # rollout die inside env.reset() with
    #   AttributeError: 'GarmentObject' object has no attribute
    #   'initial_points_positions'
    # because _get_initial_info() had thrown and the failure was swallowed.
    view = getattr(obj, "_cloth_prim_view", None)
    if view is not None:
        for sv in (getattr(getattr(env, "sim", None), "physics_sim_view", None), None):
            try:
                view.initialize(sv) if sv is not None else view.initialize()
                log(f"cloth_prim_view.initialize({'sim_view' if sv else 'no-arg'}) ok")
                break
            except Exception as exc:  # noqa: BLE001
                log(f"cloth_prim_view.initialize failed: {type(exc).__name__}: {exc}")

    # _get_initial_info() BEFORE env.reset(): GarmentObject.reset() restores
    # `initial_points_positions`, which only this creates. Failures are LOGGED,
    # never swallowed -- a silent one here surfaces 40 lines later as a
    # confusing AttributeError inside reset().
    for setup in ("post_reset", "_get_initial_info"):
        fn = getattr(obj, setup, None)
        if callable(fn):
            try:
                fn()
                log(f"object.{setup}() ok")
            except Exception as exc:  # noqa: BLE001
                log(f"object.{setup}() FAILED: {type(exc).__name__}: {exc}")
    if not hasattr(obj, "initial_points_positions"):
        raise RuntimeError(
            "initial_points_positions absent after setup -- the cloth view never "
            "initialised, so env.reset() would die downstream. Refusing to continue.")

    env.reset()
    log("env up, cloth spawned")

    def particles():
        fn = getattr(obj, "get_current_mesh_points", None)
        v = fn()
        v = v[0] if isinstance(v, (tuple, list)) else v
        return np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).reshape(-1, 3)

    def link_poses():
        out = {}
        for side, art in (("left", env.left_arm), ("right", env.right_arm)):
            d = art.data
            out[side] = (np.asarray(d.body_pos_w[0].detach().cpu()),
                         np.asarray(d.body_quat_w[0].detach().cpu()))
        return out

    obs = StormObserver(StormObsConfig(
        assets=args.assets, garment_dir=args.garment_dir,
        workdir=args.frames_out or "", extra={}))
    obs.update(particles(), link_poses())
    log("Storm stage built")

    # --- policy ---------------------------------------------------------
    # Register the policy's draccus choice before PreTrainedConfig parses the
    # checkpoint, or it fails with "Couldn't find a choice class for 'smolvla'".
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # Load the policy straight from the checkpoint, NOT via
    # make_policy(ds_meta=LeRobotDatasetMetadata(...)).
    #
    # ds_meta is only used to derive the input/output feature spec, and that
    # spec is already frozen into the checkpoint's config.json -- so pulling it
    # back out of the training dataset adds a dependency that buys nothing.
    # It cost a run: the dataset lives under lehome-data/Datasets, the path
    # here pointed inside the challenge checkout, and LeRobot responded to the
    # missing directory by trying to fetch repo "lehome" from the Hub and
    # dying on a 401. A rollout should not need the training set to be mounted,
    # let alone reach the network.
    #
    # The normalisation statistics travel with the checkpoint too, as
    # policy_preprocessor_step_5_normalizer_processor.safetensors, and
    # make_pre_post_processors reads them from pretrained_path.
    # Replay mode: drive the arms with the RECORDED demonstration actions.
    #
    # This is the experiment that separates the two live explanations for the
    # grippers hovering ~12 cm above the cloth. The demonstrations are, by
    # construction, successful folds. If replaying one in THIS env also hovers
    # and fails, the mismatch is in the scene (the cloth is not where the demo
    # motions expect it) and no amount of policy training fixes it. If the
    # replay reaches the cloth and folds, the scene is right and the failure
    # belongs to the policy.
    replay = None
    if args.replay_parquet:
        import pandas as pd
        df = pd.read_parquet(args.replay_parquet,
                             columns=["action", "episode_index"])
        ep = df[df["episode_index"] == args.replay_episode]
        replay = np.stack(ep["action"].values).astype(np.float32)
        log(f"REPLAY episode {args.replay_episode}: {replay.shape[0]} recorded actions")

    pcfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides={})
    pcfg.pretrained_path = args.policy_path
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(dev)
    pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=args.policy_path)
    log(f"policy loaded from {args.policy_path} "
        f"(params={sum(q.numel() for q in policy.parameters())/1e6:.0f}M)")

    # Keep every Nth frame for the GIF. The observer renders into fixed
    # filenames and overwrites them each step, so without this the run leaves
    # behind exactly one frame per camera and no animation at all.
    # Let the garment settle before the policy sees anything.
    #
    # The cloth spawns high and falls 13.4 cm onto the table (z 0.663 ->
    # 0.529) over roughly the first 25 steps. Without this the policy issues
    # its opening actions against a garment that is still in mid-air, and the
    # grippers were observed to stall 12.5-13.8 cm short of the cloth -- close
    # to that same 13.4 cm drop, as though reaching for where the garment was
    # rather than where it landed. The demonstrations are recorded from a
    # settled garment, so acting before the settle is an initial-state
    # mismatch with the training data.
    if args.settle_steps:
        hold = torch.from_numpy(
            np.concatenate([
                np.asarray(env.left_arm.data.joint_pos[0].detach().cpu()),
                np.asarray(env.right_arm.data.joint_pos[0].detach().cpu()),
            ]).astype(np.float32)).reshape(1, 12)
        for _ in range(args.settle_steps):
            env.step(hold)
        p0 = particles()
        log(f"SETTLED {args.settle_steps} steps, cloth_z={p0.mean(axis=0)[2]:.4f}")
        obs.update(p0, link_poses())

    frames = []
    shadow = []
    for i in range(args.steps):
        imgs = obs.render()
        n_rendered += 1
        if i % args.gif_every == 0:
            frames.append(np.concatenate(
                [imgs["left_rgb"], imgs["top_rgb"], imgs["right_rgb"]], axis=1))
        joint = env.left_arm.data.joint_pos[0].detach().cpu().numpy()
        joint = np.concatenate([joint, env.right_arm.data.joint_pos[0].detach().cpu().numpy()])
        # SmolVLA wants batched (b, c, h, w) float in [0,1]; Storm hands back
        # (h, w, c) uint8. Passing the raw array through died in
        # resize_with_pad with "(b,c,h,w) expected, but [256, 640, 3]".
        def _img(a):
            x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
            return (x.float() / 255.0).unsqueeze(0).to(dev)

        observation = {
            "observation.state": torch.from_numpy(
                joint.astype(np.float32)).unsqueeze(0).to(dev),
            "observation.images.top_rgb": _img(imgs["top_rgb"]),
            "observation.images.left_rgb": _img(imgs["left_rgb"]),
            "observation.images.right_rgb": _img(imgs["right_rgb"]),
            "task": args.task,
        }
        if replay is not None:
            a = replay[min(i, replay.shape[0] - 1)]
            # Teacher-forced fidelity ON STORM FRAMES.
            #
            # On path-traced dataset frames this policy scores skill +0.966
            # against a mean-action baseline -- it has learned the mapping.
            # Yet driving the same policy in sim leaves the cloth untouched.
            # Two explanations survive and need different fixes: the policy
            # cannot parse RASTERISED frames (domain gap), or it parses them
            # fine and drifts once it drives its own state distribution
            # (compounding error).
            #
            # Replay pins the state to the demonstration's, so asking the
            # policy what it WOULD have done here isolates perception from
            # drift: the state distribution is identical to the fidelity test,
            # only the renderer differs. A collapse in skill is the domain gap;
            # skill holding up leaves compounding error.
            if args.shadow_policy:
                batch = pre(observation) if pre else observation
                with torch.inference_mode():
                    sa = policy.select_action(batch)
                if post:
                    sa = post(sa)
                sa = np.asarray(sa.squeeze(0).detach().cpu()).reshape(-1)[:12]
                shadow.append((a.copy(), sa))
        else:
            batch = pre(observation) if pre else observation
            with torch.inference_mode():
                act = policy.select_action(batch)
            if post:
                act = post(act)
            a = np.asarray(act.squeeze(0).detach().cpu() if hasattr(act, "detach") else act
                           ).reshape(-1)[:12]
        env.step(torch.from_numpy(a.astype(np.float32)).reshape(1, 12))
        pts = particles()
        obs.update(pts, link_poses())

        # "success=False" says nothing about WHY. These three numbers separate
        # the candidate explanations: a policy emitting near-zero actions, a
        # policy that moves the arms but never contacts the cloth, and a policy
        # that manipulates the cloth but folds it wrongly.
        if i == 0:
            pts0 = pts.copy()
            joint0 = joint.copy()
        # The decisive number: how close does either arm's nearest link ever
        # get to the nearest cloth particle? The arms sweep ~1.5 rad while the
        # cloth sits frozen after settling, which says they never contact it.
        # If this distance never approaches gripper scale, the failure is
        # spatial -- the robot and the garment are not in the same place -- and
        # no amount of further training addresses it.
        if i % 25 == 0 or i == args.steps - 1:
            lp = link_poses()
            # Report WHICH link is closest, not just the minimum. A min taken
            # over all links sat at a near-constant 10.6 cm while the arms
            # swept 1.5 rad, which means it was being set by a proximal link
            # that barely moves -- so it said nothing about whether the
            # GRIPPER ever approached the cloth. The end-effector distance is
            # the one that decides whether contact was possible.
            cc = pts.mean(axis=0)
            per = {}
            for sd in ("left", "right"):
                dm = np.linalg.norm(pts[None, :, :] - lp[sd][0][:, None, :],
                                    axis=2).min(axis=1)   # per link
                per[sd] = dm
            gap = min(float(per[sd].min()) for sd in per)
            who = {sd: (int(per[sd].argmin()), float(per[sd].min()),
                        len(per[sd]), float(per[sd][-1])) for sd in per}
            log("LINKS " + " ".join(
                f"{sd}: closest_link={w[0]}/{w[2]} at {w[1]:.4f}, "
                f"last_link={w[3]:.4f}" for sd, w in who.items())
                + " ee_z=" + " ".join(
                    f"{sd}:{lp[sd][0][-1][2]:.4f}" for sd in ("left", "right"))
                + f" cloth_z={cc[2]:.4f}")
            log(f"REACH step {i:4d} min_link_to_cloth={gap:.4f} "
                f"cloth_centroid=({cc[0]:.3f},{cc[1]:.3f},{cc[2]:.3f}) "
                f"left_base=({lp['left'][0][0][0]:.3f},{lp['left'][0][0][1]:.3f},{lp['left'][0][0][2]:.3f}) "
                f"right_base=({lp['right'][0][0][0]:.3f},{lp['right'][0][0][1]:.3f},{lp['right'][0][0][2]:.3f})")

        if i % 25 == 0 or i == args.steps - 1:
            log(f"DIAG step {i:4d} "
                f"|a|={np.abs(a).mean():.4f} amax={np.abs(a).max():.4f} "
                f"djoint={np.abs(joint - joint0).max():.4f} "
                f"dcloth_max={np.abs(pts - pts0).max():.4f} "
                f"dcloth_mean={np.linalg.norm(pts - pts0, axis=1).mean():.4f}")

        if not success:
            s = env._get_success()
            success = bool(np.asarray(s.detach().cpu() if hasattr(s, "detach") else s).ravel()[0])
            if success:
                log(f"OFFICIAL CHECKER FIRED at step {i}")
        if i % 50 == 0:
            log(f"step {i:4d}  success={success}")

    if shadow:
        T = np.stack([x[0] for x in shadow])
        P = np.stack([x[1] for x in shadow])
        mse_p = float(((P - T) ** 2).mean())
        mse_m = float(((T.mean(axis=0, keepdims=True) - T) ** 2).mean())
        skill = 1.0 - mse_p / mse_m if mse_m else float("nan")
        log(f"SHADOW n={len(shadow)} policy_mse={mse_p:.5f} "
            f"mean_baseline_mse={mse_m:.5f} skill={skill:+.3f} "
            f"var_ratio={float(P.var(axis=0).mean() / T.var(axis=0).mean()):.3f}")
        json.dump({"n": len(shadow), "mse_policy": mse_p,
                   "mse_mean_baseline": mse_m, "skill_vs_mean": skill,
                   "renderer": "storm", "episode": args.replay_episode,
                   "garment": args.garment,
                   "note": "teacher-forced on replay states, Storm frames; "
                           "compare with results/action_fidelity_*.json which "
                           "is the same measurement on path-traced frames"},
                  open(args.result_out.replace(".json", "_shadow.json"), "w"),
                  indent=2)

    log(f"FINAL success={success} after {args.steps} steps")

    # Name the GIF by the OFFICIAL verdict, so the success/failure split can
    # never drift from what LeHome's own checker said about the episode.
    if frames:
        import imageio.v2 as imageio
        # Encode WHAT PRODUCED the episode, not just the garment and verdict.
        # A replay and a policy rollout of the same garment with the same
        # verdict previously resolved to one filename, and the replay silently
        # overwrote the policy's GIF -- destroying a labelled artefact and
        # making a demonstration replay look like a policy result.
        tag = "success" if success else "failure"
        mode = "replay" if replay is not None else "policy"
        ep = f"_ep{args.replay_episode}" if replay is not None else ""
        # Derive the GIF name from the RESULT PATH, so any suffix that
        # distinguishes a run (a camera fix, an ablation) distinguishes its
        # GIF too. Deriving it from garment+mode+episode alone let a re-run
        # silently overwrite the artefact it was meant to be compared against.
        stem = os.path.splitext(os.path.basename(args.result_out))[0]
        gif = os.path.join(os.path.dirname(args.result_out),
                           f"{stem}_{mode}{ep}_{tag}.gif")
        imageio.mimsave(gif, frames, duration=0.08, loop=0)
        log(f"GIF {gif} ({len(frames)} frames, {tag})")
    json.dump({"garment": args.garment, "success": success,
               "mode": "replay" if replay is not None else "policy",
               "replay_episode": args.replay_episode if replay is not None else None,
               "match_pose": args.match_pose or None,
               "match_scale": args.match_scale or None,
               "steps": args.steps, "policy": args.policy_path,
               "renderer": "storm", "note": "rasterised obs, path-traced training data"},
              open(args.result_out, "w"), indent=2)
    rc = 0
except Exception:
    import traceback
    log("TRACEBACK " + traceback.format_exc().replace("\n", "\n  "))
    rc = 3

# Only an episode that actually ran gets a verdict. A crash printed as
# "success=False" is indistinguishable from the policy genuinely failing to
# fold, and would poison the success/failure labels these GIFs are built from.
if rc == 0:
    log(f"VERDICT rollout success={success} rendered={n_rendered}")
else:
    log(f"NO VERDICT -- episode crashed before completing (rendered={n_rendered})")
# Kit does not reliably come down. Every task in sweep 21100518 wrote its
# verdict and GIF, then sat in shutdown holding a GPU -- task 0 for 39 idle
# minutes -- against the account's GPU-minute cap, which is the same limit
# that previously blocked the user's own training array. All results are
# already written and flushed by this point, so there is nothing left to
# clean up: leave via os._exit, which skips the teardown that hangs.
# app.close() is the call that hangs, and a hang is not an exception, so it
# cannot be guarded with try/except -- it has to be skipped. The OS reclaims
# Kit's resources on process exit.
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)
