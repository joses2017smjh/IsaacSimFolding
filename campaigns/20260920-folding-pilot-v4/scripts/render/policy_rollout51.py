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
import random
import sys
import types
import time
from pathlib import Path
wall_started = time.monotonic()

from rollout_audit_utils import (EpisodeAudit, configure_action_queue,
                                 validate_arguments, validate_camera_images,
                                 write_camera_artifacts)

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
ap.add_argument("--capture_out", default="",
                help="during replay, save (storm RGB, state, recorded action) "
                     "per step as an .npz for rasterised fine-tuning")
ap.add_argument("--capture_every", type=int, default=2)
ap.add_argument("--capture_chunk", type=int, default=50,
                help="future actions per frame; must match the policy chunk_size")
ap.add_argument("--shadow_policy", type=int, default=0,
                help="during replay, also record what the policy would do on "
                     "the SAME state, rendered by Storm")
ap.add_argument("--gif_every", type=int, default=3,
                help="keep every Nth rendered frame for the GIF")
ap.add_argument("--result_out", default="results/rollout.json")
ap.add_argument("--task", default="fold the garment on the table")
ap.add_argument("--sim_device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0,
                help="seed Python, NumPy, Torch, and LeHome's independent garment RNG")
ap.add_argument("--n_action_steps", type=int, default=0,
                help="SmolVLA execution queue length; 0 preserves checkpoint configuration")
ap.add_argument("--policy_variant", default="checkpoint",
                help="explicit experiment label written into result metadata")
ap.add_argument("--trajectory_out", default="",
                help="optional NPZ of sampled policy observations and executed SINGLE actions, including failures")
ap.add_argument("--trajectory_every", type=int, default=3)
from asset_paths import robot_asset_path, configure_robot_assets

ap.add_argument("--terminal_settle_steps", type=int, default=60)
ap.add_argument("--switch_to", default="", help="optional bounded diagnostic after saving the isolated episode")
args = ap.parse_args()
if args.terminal_settle_steps < 1:
    ap.error("terminal_settle_steps must be positive")
try:
    pinned_pose = validate_arguments(args)
except ValueError as exc:
    ap.error(str(exc))

# Resolve output paths before the LeHome cwd change. Each new audit job gets
# its own directory; all camera files belong to that one result.
for _name in ("lehome", "policy_path", "garment_dir", "assets", "frames_out",
              "result_out", "capture_out", "replay_parquet", "trajectory_out"):
    _value = getattr(args, _name)
    if _value:
        setattr(args, _name, os.path.abspath(os.path.expanduser(_value)))
for _path in (args.result_out, args.capture_out, args.trajectory_out):
    if _path:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
# Validate the explicit asset root BEFORE launching Isaac or importing LeHome.
robot_usd = robot_asset_path(args.assets)
# LeHome registers tasks during its package import; set this BEFORE AppLauncher.
os.environ["LEHOME_ASSETS_ROOT"] = args.assets
random.seed(args.seed)


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

from lehome_fold.storm_obs import StormObsConfig
from lehome_fold.strict_observer import StrictStormObserver as StormObserver
from lehome_fold.folding_geometry import bind_landmarks, mapped_positions_cm, fresh_geometry  # noqa: E402

def seed_rngs():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_rngs()
audit = EpisodeAudit(args.steps)
rc, success, n_rendered = 3, False, 0
first_success_details = None
terminal_details = None
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: refusing a frozen-cloth CPU rollout")
    import lehome.tasks.bedroom  # noqa: F401
    import lehome.tasks.bedroom.garment_bi_v2 as gbv
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg
    from lehome.tasks.bedroom.garment_bi_v2 import GarmentEnv
    import lehome.utils.success_checker_chanllege as official_scorer
    # JSON landmarks address authored vertices; preserve the upstream predicates.
    official_scorer.get_object_particle_position = mapped_positions_cm
    native_events = []
    scoring_phase = "initialization"
    current_action = 0
    original_official_checker = gbv.success_checker_garment_fold
    def observed_official_checker(obj, garment_type):
        result = original_official_checker(obj, garment_type)
        if isinstance(result, dict) and scoring_phase == "policy":
            native_events.append({"step": current_action, "success": bool(result["success"])})
        return result
    gbv.success_checker_garment_fold = observed_official_checker

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
    configure_robot_assets(cfg, robot_usd)
    log(f"ROBOT_ASSET {robot_usd}")
    # --garment drives PHYSICS, --garment_dir drives what Storm RENDERS.
    # Nothing tied them together, so a mismatched pair would simulate one
    # garment and show the policy another -- silently, and scored as a real
    # result. That is exactly the bug that made eleven of twelve garments
    # meaningless under the official evaluator (see storm_eval.garment_dir_for).
    _dir_name = os.path.basename(os.path.normpath(args.garment_dir))
    if _dir_name != args.garment:
        raise SystemExit(
            f"garment mismatch: physics gets --garment {args.garment!r} but "
            f"Storm renders --garment_dir .../{_dir_name!r}. These must name "
            f"the same garment or the episode is scored against wrong pixels.")
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
        _v = pinned_pose
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
    cfg.seed = args.seed
    cfg.use_random_seed = False
    cfg.random_seed = args.seed
    cfg.scene.num_envs = 1
    step_dt = float(cfg.sim.dt) * int(cfg.decimation)
    # The fixed budget describes ONE episode. Fail before running if the
    # upstream environment would auto-reset at its timeout during this budget.
    if args.steps + args.settle_steps + args.terminal_settle_steps >= int(np.ceil(cfg.episode_length_s / step_dt)) - 1:
        raise ValueError("requested rollout would reach LeHome's auto-reset timeout; reduce the step budget")
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

    mesh_path = next(Path(args.garment_dir).glob("*.usd"))
    landmark_proof = bind_landmarks(obj, mesh_path)
    log("LANDMARK_PROOF " + json.dumps(landmark_proof))
    env.reset()
    log("env up, cloth spawned")

    def particles():
        fn = getattr(obj, "get_current_mesh_points", None)
        v = fn()
        v = v[0] if isinstance(v, (tuple, list)) else v
        points = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError("cloth particle positions are empty or nonfinite")
        return points

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
    dev = args.sim_device
    policy = policy.to(dev)
    queue_metadata = configure_action_queue(policy, args.n_action_steps)
    inference_times = []
    prediction_count = [0]
    original_predict = policy._get_action_chunk
    def measured_predict(*a, **kw):
        started = time.monotonic()
        chunk = original_predict(*a, **kw)
        torch.cuda.synchronize()
        inference_times.append(time.monotonic() - started)
        if chunk.shape[1] != 50:
            raise ValueError(f"Prediction chunk changed: {tuple(chunk.shape)}")
        prediction_count[0] += 1
        return chunk
    policy._get_action_chunk = measured_predict
    pcfg.n_action_steps = queue_metadata["effective_n_action_steps"]
    pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=args.policy_path)
    log(f"policy loaded from {args.policy_path} "
        f"(params={sum(q.numel() for q in policy.parameters())/1e6:.0f}M)")
    log(f"AUDIT variant={args.policy_variant} seed={args.seed} queue={queue_metadata}")

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

    # Frame keys are completed action counts: frame 0 precedes the first
    # action; frame N is the state AFTER action N. Store actual camera views,
    # never crops of the composite. Off-stride success/final frames are kept.
    frames = {}
    camera_keys = ("top_rgb", "left_rgb", "right_rgb")
    render_integrity = {"validated_render_calls": 0, "first_frame": None, "final_frame": None, "acquisitions": []}

    def render_images():
        images = obs.render()
        stats = validate_camera_images(images, obs.cfg.width, obs.cfg.height)
        if obs.last_acquisition["update_serial"] != obs.update_serial:
            raise ValueError("Camera acquisition is stale relative to particle update")
        render_integrity["acquisitions"].append({"completed_actions": audit.completed_steps, **obs.last_acquisition})
        render_integrity["validated_render_calls"] += 1
        if render_integrity["first_frame"] is None:
            render_integrity["first_frame"] = stats
        render_integrity["final_frame"] = stats
        return images

    def keep_frame(step, images):
        if step not in frames:
            frames[step] = {key: np.asarray(images[key], dtype=np.uint8).copy()
                            for key in camera_keys}

    def checker_details():
        garment_type = env.garment_loader.get_garment_type(cfg.garment_name)
        return fresh_geometry(obj, garment_type, official_scorer)

    # Reset policy randomness after construction/settling so paired seeds
    # start from the same sampling stream. GPU cloth is not bitwise deterministic.
    seed_rngs()
    policy.reset()
    pts0 = particles().copy()
    pts_previous = pts0.copy()
    initial_centroid = pts0.mean(axis=0)
    joint0 = np.concatenate([
        np.asarray(env.left_arm.data.joint_pos[0].detach().cpu()),
        np.asarray(env.right_arm.data.joint_pos[0].detach().cpu())]).copy()
    max_particle_displacement = 0.0
    peak_mean_particle_displacement = 0.0
    peak_step_particle_displacement = 0.0
    cumulative_mean_particle_motion = 0.0
    shadow = []
    cap = []
    trajectory = []
    geometry_trajectory = []
    manipulation_trajectory = []
    previous_action = None
    action_jumps = []
    scoring_phase = "policy"
    for i in range(args.steps):
        current_action = i + 1
        imgs = render_images()
        n_rendered += 1
        if i % args.gif_every == 0:
            keep_frame(i, imgs)
        joint = env.left_arm.data.joint_pos[0].detach().cpu().numpy()
        joint = np.concatenate([joint, env.right_arm.data.joint_pos[0].detach().cpu().numpy()])
        if joint.shape != (12,) or not np.isfinite(joint).all():
            raise ValueError("policy state must contain twelve finite joint positions")
        for arm in (env.left_arm, env.right_arm):
            if not bool(arm.data.joint_pos.isfinite().all()) or not bool(arm.data.joint_vel.isfinite().all()):
                raise ValueError("Nonfinite robot joint state")
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
            # Capture the pairing the fine-tune needs: what the renderer SHOWS
            # against what the demonstration DID. Replay drives the recorded
            # actions, so the action label is ground truth and the image is the
            # distribution the policy is actually deployed on -- which is the
            # entire mismatch, measured at a 1.014 skill gap.
            if args.capture_out and i % args.capture_every == 0:
                # Capture the ACTION CHUNK, not a single action. SmolVLA
                # predicts chunk_size future actions; a single action tiled to
                # fill that shape trains the model to emit a constant
                # trajectory, which is trivially easy to fit (training loss
                # collapses) and destroys action production (skill fell from
                # -0.038 to -0.201 on the first attempt). Tail frames repeat
                # the last recorded action, which is what the demonstration
                # does anyway once the arms stop.
                j0 = min(i, replay.shape[0] - 1)
                idx = np.minimum(np.arange(j0, j0 + args.capture_chunk),
                                 replay.shape[0] - 1)
                cap.append((
                    np.stack([imgs["top_rgb"], imgs["left_rgb"], imgs["right_rgb"]]),
                    joint.astype(np.float32), replay[idx].astype(np.float32)))
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
        if np.asarray(a).shape != (12,) or not np.isfinite(a).all():
            raise ValueError("executed action must contain twelve finite joint targets")
        if args.trajectory_out and i % args.trajectory_every == 0:
            trajectory.append((i, np.stack([imgs[k] for k in camera_keys]).copy(),
                               joint.astype(np.float32).copy(),
                               np.asarray(a, dtype=np.float32).copy()))
        if previous_action is not None:
            action_jumps.append(float(np.linalg.norm(a - previous_action)))
        previous_action = a.copy()
        env.step(torch.from_numpy(a.astype(np.float32)).reshape(1, 12).to(args.sim_device))
        audit.record_step(None)
        pts = particles()
        obs.update(pts, link_poses())
        displacement = np.linalg.norm(pts - pts0, axis=1)
        step_motion = np.linalg.norm(pts - pts_previous, axis=1)
        max_particle_displacement = max(max_particle_displacement, float(displacement.max()))
        peak_mean_particle_displacement = max(peak_mean_particle_displacement, float(displacement.mean()))
        peak_step_particle_displacement = max(peak_step_particle_displacement, float(step_motion.max()))
        cumulative_mean_particle_motion += float(step_motion.mean())
        pts_previous = pts.copy()
        geometric = checker_details()
        geometry_trajectory.append({"phase": "policy", "step": i + 1, **geometric})
        links = link_poses()
        near = {side: float(np.linalg.norm(pts - links[side][0][-1], axis=1).min())
                for side in ("left", "right")}
        manipulation_trajectory.append({"step": i + 1, "last_link_distance_m": near,
            "gripper_commands": [float(a[5]), float(a[11])],
            "max_cloth_height_m": float(pts[:, 2].max()),
            "mean_displacement_m": float(displacement.mean())})

        # "success=False" says nothing about WHY. These three numbers separate
        # the candidate explanations: a policy emitting near-zero actions, a
        # policy that moves the arms but never contacts the cloth, and a policy
        # that manipulates the cloth but folds it wrongly.
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

        # Keep native checker sampling cadence independent of earlier success.
        env._get_success()
        checked_success = any(e["step"] == i + 1 and e["success"] for e in native_events[-2:])
        first_success = audit.record_check(checked_success)
        success = audit.first_success_step is not None
        if first_success:
            first_success_details = checker_details()
            success_images = render_images()
            n_rendered += 1
            keep_frame(audit.completed_steps, success_images)
            log(f"OFFICIAL CHECKER FIRED after {audit.completed_steps} actions (step index {i})")
        if i % 50 == 0:
            log(f"step {i:4d}  success={success}")

    # Continue the full requested budget even after a successful fold. The
    # final checker call distinguishes first attainment from maintaining it.
    scoring_phase = "terminal_settle"
    for settle_i in range(args.terminal_settle_steps):
        env.step(torch.from_numpy(a.astype(np.float32)).reshape(1, 12).to(args.sim_device))
        pts = particles()
        for arm in (env.left_arm, env.right_arm):
            if not bool(arm.data.joint_pos.isfinite().all()) or not bool(arm.data.joint_vel.isfinite().all()):
                raise ValueError("Nonfinite terminal robot state")
        obs.update(pts, link_poses())
        geometry_trajectory.append({"phase": "terminal_settle", "step": args.steps + settle_i + 1,
                                    **checker_details()})
    terminal_details = checker_details()
    terminal_success = terminal_details["success"]
    verdict = audit.finish(False)
    verdict["terminal_success"] = terminal_success
    verdict["success_criterion"] = "native official checker fired during policy actions, with verified authored landmark mapping"
    success = verdict["success"]
    final_images = render_images()
    n_rendered += 1
    keep_frame(audit.completed_steps + args.terminal_settle_steps, final_images)

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

    if cap:
        os.makedirs(os.path.dirname(args.capture_out) or ".", exist_ok=True)
        np.savez_compressed(
            args.capture_out,
            images=np.stack([c[0] for c in cap]).astype(np.uint8),   # (N,3,H,W,3)
            state=np.stack([c[1] for c in cap]),
            action=np.stack([c[2] for c in cap]),
            garment=args.garment, episode=args.replay_episode,
            success=bool(success))
        log(f"CAPTURED {len(cap)} frames -> {args.capture_out} "
            f"({os.path.getsize(args.capture_out)/1e6:.1f} MB)")

    log(f"FINAL success_seen={success} terminal_success={terminal_success} "
        f"after {audit.completed_steps} completed actions")

    tag = "success" if success else "failure"
    mode = "replay" if replay is not None else "policy"
    ep = f"_ep{args.replay_episode}" if replay is not None else ""
    prefix = os.path.splitext(args.result_out)[0] + f"_{mode}{ep}_{tag}"
    media = write_camera_artifacts(frames, prefix, step_dt, args.gif_every,
                                   audit.first_success_step)
    for view_name, gif_path in media["gifs"].items():
        log(f"GIF {view_name} {gif_path} ({len(frames)} sampled frames, {tag})")

    trajectory_metadata = None
    if trajectory:
        # Explicitly named executed_action, NOT action chunks. Repeating one
        # executed target into a training chunk caused an earlier bad fine-tune.
        with open(args.trajectory_out, "wb") as trajectory_file:
            np.savez_compressed(
                trajectory_file,
                step_index=np.asarray([row[0] for row in trajectory], dtype=np.int32),
                images=np.stack([row[1] for row in trajectory]),
                state=np.stack([row[2] for row in trajectory]),
                executed_action=np.stack([row[3] for row in trajectory]),
                camera_keys=np.asarray(camera_keys),
                success=np.asarray(success), terminal_success=np.asarray(terminal_success),
                first_success_step=np.asarray(audit.first_success_step if success else -1),
                seed=np.asarray(args.seed), policy_variant=np.asarray(args.policy_variant))
        trajectory_metadata = {"path": args.trajectory_out, "samples": len(trajectory),
                               "sample_every": args.trajectory_every,
                               "action_semantics": "one executed target per sampled observation; not future action chunks",
                               "post_action_final_rgb": media["snapshots"]["final"]}

    final_displacement = np.linalg.norm(pts - pts0, axis=1)
    cloth_motion = {
        "units": "meters", "reference": "immediately before first policy action, after optional settling",
        "particle_count": len(pts0),
        "initial_centroid": initial_centroid.tolist(),
        "terminal_centroid": pts.mean(axis=0).tolist(),
        "terminal_centroid_displacement": float(np.linalg.norm(pts.mean(axis=0) - initial_centroid)),
        "max_particle_displacement": max_particle_displacement,
        "peak_mean_particle_displacement": peak_mean_particle_displacement,
        "terminal_mean_particle_displacement": float(final_displacement.mean()),
        "terminal_max_particle_displacement": float(final_displacement.max()),
        "peak_step_particle_displacement": peak_step_particle_displacement,
        "cumulative_mean_particle_motion": cumulative_mean_particle_motion,
    }
    render_integrity["first_to_final_changed_pixel_fraction"] = {
        key: float(np.any(frames[0][key] != final_images[key], axis=2).mean())
        for key in camera_keys}
    geometric_success_steps = [r["step"] for r in geometry_trajectory if r["success"] and r["phase"] == "policy"]
    earliest_near = next((r["step"] for r in manipulation_trajectory if min(r["last_link_distance_m"].values()) < 0.03), None)
    min_distance = min(min(r["last_link_distance_m"].values()) for r in manipulation_trajectory)
    failure = ("success_then_unfolded" if geometric_success_steps and not terminal_success else
               "never_approached_cloth" if min_distance > 0.05 else
               f"partial_fold_{terminal_details['conditions_passed']}_of_{terminal_details['conditions_total']}")
    if terminal_success:
        failure = None
    from lehome_fold.policy_media import write_mp4
    media_caption = (f"POLICY — {args.policy_variant} | checkpoint={Path(args.policy_path).name} | {args.garment} | seed={args.seed}\n"
                     f"Execution horizon={queue_metadata['effective_n_action_steps']} | prediction chunk=50 | official ever={success} | terminal settled={terminal_success}\n"
                     f"Terminal conditions={terminal_details['conditions_passed']}/{terminal_details['conditions_total']} | failure={failure or 'none'} | left wrist / top / right wrist")
    media["mp4"] = write_mp4(frames, prefix + ".mp4", step_dt, media_caption)
    result = {
        **verdict, "schema_version": 3, "status": "completed",
        "landmark_mapping": landmark_proof,
        "official_ever_success": success, "native_checker_samples": native_events,
        "geometric_ever_success": bool(geometric_success_steps),
        "geometric_first_success_action": geometric_success_steps[0] if geometric_success_steps else None,
        "geometric_success_persisted": (all(r["success"] for r in geometry_trajectory if r["step"] >= geometric_success_steps[0]) if geometric_success_steps else None),
        "terminal_settle_steps": args.terminal_settle_steps,
        "terminal_settle_control": "hold last executed joint target",
        "geometry_trajectory": geometry_trajectory,
        "manipulation_trajectory": manipulation_trajectory,
        "first_contact": None, "first_grasp": None, "failed_grasps": None,
        "contact_measurement_status": "force/contact ground truth unavailable; nearest gripper-link proximity is recorded separately",
        "first_gripper_proximity_under_3cm_action": earliest_near,
        "failure_category": failure, "failure_classification_basis": "geometry and gripper-link proximity; no unsupported grasp-region claims",
        "replanning_count": prediction_count[0],
        "inference_seconds": inference_times,
        "max_action_discontinuity_l2": max(action_jumps, default=0.0),
        "wall_seconds": time.monotonic() - wall_started,
        "physics_finite": True, "robot_finite": True,
        "garment": args.garment, "mode": mode,
        "replay_episode": args.replay_episode if replay is not None else None,
        "match_pose": args.match_pose or None, "match_scale": args.match_scale or None,
        "pinned_pose": ({"position_xyz": pinned_pose[:3], "euler_xyz_degrees": pinned_pose[3:]}
                        if pinned_pose is not None else None),
        "actual_reset_pose": getattr(obj, "reset_pose", None),
        "steps": audit.completed_steps, "settle_steps": args.settle_steps,
        "simulation_seconds": audit.completed_steps * step_dt,
        "policy": args.policy_path, "policy_realpath": os.path.realpath(args.policy_path),
        "policy_variant": args.policy_variant, "task": args.task,
        "robot_asset": robot_usd,
        "seed": args.seed, "seeded_rngs": ["python", "numpy", "torch", "torch_cuda", "lehome_garment_rng"],
        "bitwise_simulation_determinism_guaranteed": False,
        "sim_device": args.sim_device, "physics_dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation), **queue_metadata,
        "first_success_checker": first_success_details, "terminal_checker": terminal_details,
        "cloth_motion": cloth_motion, "renderer": "storm",
        "render_integrity": render_integrity, "n_rendered": n_rendered,
        "gif_views": media["gifs"], "media": media,
        "trajectory": trajectory_metadata,
        "note": "Fixed-budget single-garment rollout with official geometric scoring and Storm rasterized observations; success records attainment, terminal_success records the final state.",
    }

    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        raise TypeError(f"cannot serialize {type(value).__name__}")

    temporary_result = args.result_out + ".partial"
    with open(temporary_result, "w") as result_file:
        json.dump(result, result_file, indent=2, default=_json_default, allow_nan=False)
        result_file.flush()
        os.fsync(result_file.fileno())
    os.replace(temporary_result, args.result_out)
    rc = 0
    if args.switch_to:
        from lehome_fold.switch_trace import trace_switch
        trace_switch(env, args.switch_to, args.result_out + ".switch_trace.jsonl")
except Exception as exc:
    import traceback
    error_traceback = traceback.format_exc()
    log("TRACEBACK " + error_traceback.replace("\n", "\n  "))
    # A simulator/camera/artifact error is not a failed fold. Preserve useful
    # accounting in a separately named file, leaving success undecided.
    try:
        with open(args.result_out + ".error.json", "w") as error_file:
            json.dump({"status": "error", "success": None, "terminal_success": None,
                       "completed_steps": audit.completed_steps, "requested_steps": args.steps,
                       "first_success_step": audit.first_success_step,
                       "policy": args.policy_path, "policy_variant": args.policy_variant,
                       "garment": args.garment, "seed": args.seed, "n_rendered": n_rendered,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": error_traceback}, error_file, indent=2)
    except Exception as write_exc:
        log(f"could not write error metadata: {write_exc}")
    rc = 3

# Only an episode that actually ran gets a verdict. A crash printed as
# "success=False" is indistinguishable from the policy genuinely failing to
# fold, and would poison the success/failure labels these GIFs are built from.
if rc == 0:
    log(f"VERDICT rollout success={success} rendered={n_rendered}")
else:
    log(f"NO COMPLETED RESULT -- setup, episode, or artifact writing failed "
        f"(completed_steps={audit.completed_steps}, rendered={n_rendered})")
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
