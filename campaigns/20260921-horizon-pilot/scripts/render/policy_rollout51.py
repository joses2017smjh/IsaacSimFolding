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
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
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
ap.add_argument("--causal_out", default="",
                help="optional JSON output for in-process replan causality branches")
ap.add_argument("--causal_branch_steps", type=int, default=120,
                help="steps per fresh-replan branch after a saved H50 state")
ap.add_argument("--causal_action_jsonl", default="",
                help="optional completed-pilot behavior JSONL used as the cached H50 control")
ap.add_argument("--boundary_capture_out", default="",
                help="replay-only NPZ of exact post-action-5/10 H50 observations and suffix targets")
ap.add_argument("--boundary_eval_trained_path", default="",
                help="trained checkpoint for the two-boundary snapshot-only comparison")
ap.add_argument("--boundary_eval_out", default="",
                help="JSON output for the snapshot-only baseline/trained comparison")
ap.add_argument("--rtc_guidance", action="store_true",
                help="enable the installed LeRobot SmolVLA RTC branch in causal diagnostics")
ap.add_argument("--queue_diagnostic", action="store_true",
                help="run the deterministic hard retained-prefix queue diagnostic")
ap.add_argument("--observation_diagnostic", action="store_true",
                help="run the pose-3 observation-component causal diagnostic")
ap.add_argument("--observation_execute_best", action="store_true",
                help="execute at most the preregistered best two observation probes")
ap.add_argument("--queue_delays", default="2,5,10",
                help="comma-separated retained-prefix delays for queue diagnostic")
ap.add_argument("--hard_queue_delay", type=int, default=0,
                help="online H10 hard retained-prefix delay for conditional validation")
ap.add_argument("--fixed_rng_seed", type=int, default=-1,
                help="optional fixed Torch/CUDA seed for the causal seed panel")
ap.add_argument("--seed_panel", default="101,102,103,104",
                help="comma-separated predetermined Torch/CUDA seeds for causal plan sensitivity")
args = ap.parse_args()
if args.terminal_settle_steps < 1:
    ap.error("terminal_settle_steps must be positive")
try:
    args.queue_delays = tuple(sorted({int(x.strip()) for x in args.queue_delays.split(",") if x.strip()}))
except ValueError as exc:
    ap.error(f"invalid --queue_delays={args.queue_delays!r}")
if args.queue_diagnostic and (not args.causal_out):
    ap.error("--queue_diagnostic requires --causal_out")
if args.queue_diagnostic and args.rtc_guidance:
    ap.error("--queue_diagnostic cannot be combined with --rtc_guidance")
if args.observation_diagnostic and not args.causal_out:
    ap.error("--observation_diagnostic requires --causal_out")
if args.observation_diagnostic and (args.rtc_guidance or args.queue_diagnostic):
    ap.error("--observation_diagnostic cannot be combined with RTC or queue diagnostics")
if args.observation_execute_best and not args.observation_diagnostic:
    ap.error("--observation_execute_best requires --observation_diagnostic")
if args.queue_diagnostic and any(x < 1 or x > 50 for x in args.queue_delays):
    ap.error("queue delays must be in [1, 50]")
if args.hard_queue_delay < 0 or args.hard_queue_delay > 10:
    ap.error("--hard_queue_delay must be 0..10")
if args.hard_queue_delay and (args.causal_out or args.rtc_guidance):
    ap.error("--hard_queue_delay cannot be combined with causal or RTC diagnostics")
try:
    pinned_pose = validate_arguments(args)
except ValueError as exc:
    ap.error(str(exc))

# Resolve output paths before the LeHome cwd change. Each new audit job gets
# its own directory; all camera files belong to that one result.
for _name in ("lehome", "policy_path", "garment_dir", "assets", "frames_out",
              "result_out", "capture_out", "replay_parquet", "trajectory_out",
              "causal_action_jsonl", "boundary_capture_out"):
    _value = getattr(args, _name)
    if _value:
        setattr(args, _name, os.path.abspath(os.path.expanduser(_value)))
if args.causal_out:
    args.causal_out = os.path.abspath(os.path.expanduser(args.causal_out))
    os.makedirs(os.path.dirname(args.causal_out), exist_ok=True)
if args.boundary_capture_out:
    if not args.causal_action_jsonl:
        ap.error("--boundary_capture_out requires the successful cached H50 --causal_action_jsonl")
    if args.causal_out or args.rtc_guidance or args.queue_diagnostic or args.observation_diagnostic:
        ap.error("--boundary_capture_out is replay-only and cannot run with causal/inference diagnostics")
    os.makedirs(os.path.dirname(args.boundary_capture_out), exist_ok=True)
if bool(args.boundary_eval_trained_path) != bool(args.boundary_eval_out):
    ap.error("--boundary_eval_trained_path and --boundary_eval_out must be supplied together")
if args.boundary_eval_trained_path:
    if not args.causal_out:
        ap.error("boundary evaluation requires --causal_out for the source H50 snapshot audit")
    if args.rtc_guidance or args.queue_diagnostic or args.observation_diagnostic or args.observation_execute_best:
        ap.error("boundary evaluation cannot use RTC, queue, or observation diagnostics")
    args.boundary_eval_trained_path = os.path.abspath(os.path.expanduser(args.boundary_eval_trained_path))
    args.boundary_eval_out = os.path.abspath(os.path.expanduser(args.boundary_eval_out))
    os.makedirs(os.path.dirname(args.boundary_eval_out), exist_ok=True)
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

    causal_replay_actions = None
    if args.causal_action_jsonl:
        causal_replay_actions = [json.loads(line)["executed_action_rad"]
                                 for line in Path(args.causal_action_jsonl).read_text().splitlines()
                                 if line.strip()]
        if len(causal_replay_actions) < args.steps:
            raise ValueError(f"causal cached control has {len(causal_replay_actions)} actions, need {args.steps}")
        log(f"CAUSAL_CACHED_CONTROL {args.causal_action_jsonl}: {len(causal_replay_actions)} actions")

    pcfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides={})
    pcfg.pretrained_path = args.policy_path
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).eval()
    dev = args.sim_device
    policy = policy.to(dev)
    queue_metadata = configure_action_queue(policy, args.n_action_steps)
    hard_queue_enabled = bool(args.hard_queue_delay)
    if hard_queue_enabled:
        queue_metadata = {**queue_metadata, "effective_n_action_steps": 10,
                          "hard_retained_prefix_delay": args.hard_queue_delay,
                          "internal_prediction_queue_length": 50}
    inference_times = []
    prediction_count = [0]
    prediction_chunks = []
    prediction_timing_audit = []

    def _sim_counter():
        value = getattr(env, "_sim_step_counter", 0)
        try:
            return int(value)
        except Exception:
            return int(value.item()) if hasattr(value, "item") else 0

    def _episode_counter():
        value = getattr(env, "episode_length_buf", None)
        if value is None:
            return None
        try:
            return int(value.reshape(-1)[0].item())
        except Exception:
            return None

    def _rng_digest():
        digest = hashlib.sha256()
        digest.update(torch.get_rng_state().cpu().numpy().tobytes())
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
        return digest.hexdigest()

    original_predict = policy._get_action_chunk
    def measured_predict(*a, **kw):
        started = time.monotonic()
        sim_before = _sim_counter()
        episode_before = _episode_counter()
        rng_before = _rng_digest()
        chunk = original_predict(*a, **kw)
        torch.cuda.synchronize()
        sim_after = _sim_counter()
        episode_after = _episode_counter()
        prediction_timing_audit.append({
            "source": "active_rollout_controller",
            "sim_step_before": sim_before,
            "sim_step_after": sim_after,
            "sim_step_delta": sim_after - sim_before,
            "episode_length_before": episode_before,
            "episode_length_after": episode_after,
            "episode_length_delta": (None if episode_before is None or episode_after is None
                                       else episode_after - episode_before),
            "rng_before_digest": rng_before,
            "rng_after_digest": _rng_digest(),
            "inference_blocks_before_env_step": True,
        })
        if sim_after != sim_before or (episode_before is not None and episode_after != episode_before):
            raise RuntimeError("policy inference advanced simulator state before env.step()")
        inference_times.append(time.monotonic() - started)
        if chunk.shape[1] != 50:
            raise ValueError(f"Prediction chunk changed: {tuple(chunk.shape)}")
        prediction_chunks.append(chunk.detach().cpu().numpy()[0].copy())
        prediction_count[0] += 1
        return chunk
    policy._get_action_chunk = measured_predict
    pcfg.n_action_steps = queue_metadata["effective_n_action_steps"]
    pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=args.policy_path)
    observation_feature_tap = None
    observation_feature_status = {
        "requested": bool(args.observation_diagnostic),
        "available": False,
        "path": "model.embed_prefix",
        "output_unchanged": None,
        "error": None,
    }
    if args.observation_diagnostic:
        # This is the repository's existing passive prefix-feature hook. It
        # wraps the method and returns its original value; it is not a second
        # policy, a feature substitution, or a change to action production.
        try:
            from lehome_fold.policy_wrap import (DEFAULT_FEATURE_PATH,
                                                 FeatureTap,
                                                 probe_feature_source)
            observation_feature_status["path"] = DEFAULT_FEATURE_PATH
            observation_feature_status["source"] = probe_feature_source(policy)
            observation_feature_tap = FeatureTap(policy, DEFAULT_FEATURE_PATH).install()
            observation_feature_status["available"] = True
            observation_feature_status["output_unchanged"] = True
        except Exception as exc:  # noqa: BLE001 - record unavailable, do not alter policy
            observation_feature_status["error"] = f"{type(exc).__name__}: {exc}"
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

    def make_observation(images, joint):
        """Build the exact policy input used by the normal rollout loop."""
        def _img(a):
            x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
            return (x.float() / 255.0).unsqueeze(0).to(dev)
        return {
            "observation.state": torch.from_numpy(
                np.asarray(joint, dtype=np.float32)).reshape(1, -1).to(dev),
            "observation.images.top_rgb": _img(images["top_rgb"]),
            "observation.images.left_rgb": _img(images["left_rgb"]),
            "observation.images.right_rgb": _img(images["right_rgb"]),
            "task": args.task,
        }

    def policy_batch(images, joint):
        batch = make_observation(images, joint)
        return pre(batch) if pre else batch

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
    hard_queue_previous_plan = None
    hard_queue_period = None
    hard_queue_sources = None
    hard_queue_handoffs = []
    causal_snapshots = {}
    causal_trace = {}
    executed_actions = {}
    causal_initial_observation = None
    causal_enabled = bool(args.causal_out)
    boundary_capture_enabled = bool(args.boundary_capture_out)
    snapshot_enabled = causal_enabled or boundary_capture_enabled
    boundary_capture_rows = {}
    causal_initial_torch_rng = torch.get_rng_state().cpu().numpy().copy()
    causal_initial_cuda_rng = [x.cpu().numpy().copy() for x in torch.cuda.get_rng_state_all()]
    try:
        causal_seed_panel = [int(x.strip()) for x in args.seed_panel.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --seed_panel={args.seed_panel!r}") from exc
    if not causal_enabled:
        causal_seed_panel = []
    from lehome_fold.behavior_telemetry import BehaviorTelemetry
    telemetry_horizon = 10 if hard_queue_enabled else args.n_action_steps
    behavior = BehaviorTelemetry(args.result_out + ".behavior.jsonl", pts0, telemetry_horizon, step_dt,
        {"left": env.left_arm.body_names, "right": env.right_arm.body_names})

    def _as_cpu_array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).copy()

    def _cloth_velocity_array():
        view = getattr(obj, "_cloth_prim_view", None)
        getter = getattr(view, "get_velocities", None)
        if not callable(getter):
            return None
        try:
            return _as_cpu_array(getter())
        except Exception:
            return None

    def _snapshot_state(boundary, images):
        """Capture all state setters available in this Isaac/PhysX path."""
        view = getattr(obj, "_cloth_prim_view", None)
        positions = _as_cpu_array(view.get_world_positions()) if view is not None else None
        velocities = _cloth_velocity_array()
        if positions is None or velocities is None:
            raise RuntimeError("causal branch requires cloth positions and velocities")
        arms = {}
        for side, arm in (("left", env.left_arm), ("right", env.right_arm)):
            arms[side] = {
                "joint_pos": _as_cpu_array(arm.data.joint_pos),
                "joint_vel": _as_cpu_array(arm.data.joint_vel),
                "joint_pos_target": _as_cpu_array(arm.data.joint_pos_target),
                "joint_vel_target": _as_cpu_array(arm.data.joint_vel_target),
            }
        return {
            "boundary": int(boundary),
            "cloth_positions": positions,
            "cloth_velocities": velocities,
            "arms": arms,
            "images": {k: np.asarray(images[k]).copy() for k in camera_keys},
            "joint": np.concatenate([arms["left"]["joint_pos"].reshape(-1),
                                     arms["right"]["joint_pos"].reshape(-1)]),
            "torch_rng": torch.get_rng_state().cpu().numpy().copy(),
            "cuda_rng": [x.cpu().numpy().copy() for x in torch.cuda.get_rng_state_all()],
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "episode_length": _as_cpu_array(env.episode_length_buf),
            "common_step_counter": int(getattr(env, "common_step_counter", 0)),
            "sim_step_counter": int(getattr(env, "_sim_step_counter", 0)),
        }

    def _restore_state(snapshot):
        view = getattr(obj, "_cloth_prim_view", None)
        if view is None or not callable(getattr(view, "set_world_positions", None)):
            raise RuntimeError("cloth position restore API unavailable")
        if not callable(getattr(view, "set_velocities", None)):
            raise RuntimeError("cloth velocity restore API unavailable")
        cloth_pos = torch.as_tensor(snapshot["cloth_positions"], dtype=torch.float32, device=dev)
        cloth_vel = torch.as_tensor(snapshot["cloth_velocities"], dtype=torch.float32, device=dev)
        view.set_world_positions(cloth_pos)
        view.set_velocities(cloth_vel)
        for side, arm in (("left", env.left_arm), ("right", env.right_arm)):
            state = snapshot["arms"][side]
            pos = torch.as_tensor(state["joint_pos"], dtype=torch.float32, device=dev)
            vel = torch.as_tensor(state["joint_vel"], dtype=torch.float32, device=dev)
            arm.write_joint_state_to_sim(pos, vel)
            if hasattr(arm, "set_joint_position_target"):
                arm.set_joint_position_target(torch.as_tensor(state["joint_pos_target"], dtype=torch.float32, device=dev))
            if hasattr(arm, "set_joint_velocity_target"):
                arm.set_joint_velocity_target(torch.as_tensor(state["joint_vel_target"], dtype=torch.float32, device=dev))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        env.episode_length_buf[...] = torch.as_tensor(snapshot["episode_length"], device=env.episode_length_buf.device)
        env.common_step_counter = snapshot["common_step_counter"]
        env._sim_step_counter = snapshot["sim_step_counter"]
        torch.set_rng_state(torch.as_tensor(snapshot["torch_rng"], dtype=torch.uint8))
        torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8, device="cpu") for x in snapshot["cuda_rng"]])
        np.random.set_state(snapshot["numpy_rng"])
        random.setstate(snapshot["python_rng"])
        torch.cuda.synchronize()
        restored = _as_cpu_array(view.get_world_positions())
        fidelity = float(np.sqrt(np.mean((restored - snapshot["cloth_positions"]) ** 2)))
        if not np.isfinite(fidelity) or fidelity > 1e-5:
            raise RuntimeError(f"causal restore cloth RMS error {fidelity:.3e} m exceeds 1e-5 m")
        obs.update(particles(), link_poses())

    def _trace_state(step):
        links = link_poses()
        points = particles()
        joint = np.concatenate([_as_cpu_array(env.left_arm.data.joint_pos[0]),
                                _as_cpu_array(env.right_arm.data.joint_pos[0])])
        return {"step": int(step), "joint": joint, "cloth_centroid": points.mean(axis=0),
                "left_ee": _as_cpu_array(links["left"][0][-1]),
                "right_ee": _as_cpu_array(links["right"][0][-1])}

    scoring_phase = "policy"
    reconstructed_original_chunk = None
    reconstructed_original_chunk_raw = None
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
        observation = make_observation(imgs, joint)
        if causal_enabled and i == 0:
            causal_initial_observation = {
                "images": {k: np.asarray(imgs[k], dtype=np.uint8).copy() for k in camera_keys},
                "joint": np.asarray(joint, dtype=np.float32).copy(),
                "task": args.task,
                "source": "original_action_0_H50_observation",
            }
            # Reconstruct the first H50 sampler call without altering the
            # cached-control episode. The historical pilot did not persist
            # its pre-call RNG bytes, so this uses the state captured after
            # the frozen episode seed and records that limitation explicitly.
            _torch_before = torch.get_rng_state().cpu().numpy().copy()
            _cuda_before = [x.cpu().numpy().copy() for x in torch.cuda.get_rng_state_all()]
            with torch.inference_mode():
                _chunk = original_predict(policy_batch(imgs, joint))
            reconstructed_original_chunk_raw = np.asarray(
                _chunk.detach().cpu().numpy()[0] if hasattr(_chunk, "detach") else _chunk
            ).copy()
            if post:
                _chunk = post(_chunk)
            reconstructed_original_chunk = np.asarray(_chunk.detach().cpu().numpy()[0] if hasattr(_chunk, "detach") else _chunk)[0:].copy()
            torch.set_rng_state(torch.as_tensor(_torch_before, dtype=torch.uint8))
            torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8, device="cpu") for x in _cuda_before])
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
        elif causal_replay_actions is not None:
            a = np.asarray(causal_replay_actions[i], dtype=np.float32)
        elif hard_queue_enabled:
            # Conditional validation path: predict a full 50-row chunk at each
            # H10 observation, then apply the exact hard handoff rule used by
            # the pose-3 inference-only diagnostic. The first chunk has no
            # previous plan; subsequent chunks retain d rows from the current
            # remainder and discard fresh rows [0:d].
            if i % 10 == 0:
                batch = pre(observation) if pre else observation
                with torch.inference_mode():
                    chunk = policy._get_action_chunk(batch)
                if post:
                    chunk = post(chunk)
                if hasattr(chunk, "detach"):
                    chunk = chunk.detach().cpu().numpy()
                fresh = np.asarray(chunk)[0].astype(np.float32, copy=True)
                if fresh.shape != (50, 12):
                    raise RuntimeError(f"hard queue fresh chunk has shape {fresh.shape}, expected (50, 12)")
                if i == 0:
                    hard_queue_period = fresh[:10].copy()
                    hard_queue_sources = [{"source": "fresh_chunk", "source_index": j} for j in range(10)]
                else:
                    if hard_queue_previous_plan is None or len(hard_queue_previous_plan) < args.hard_queue_delay:
                        raise RuntimeError("hard queue previous-plan remainder is shorter than retained delay")
                    retained = hard_queue_previous_plan[:args.hard_queue_delay].copy()
                    hard_queue_period = np.concatenate([retained, fresh[args.hard_queue_delay:10]], axis=0)
                    hard_queue_sources = ([{"source": "previous_plan_remainder", "source_index": j}
                                           for j in range(args.hard_queue_delay)]
                                          + [{"source": "fresh_chunk_suffix", "source_index": j}
                                             for j in range(args.hard_queue_delay, 10)])
                    hard_queue_handoffs.append({
                        "boundary_action": i,
                        "delay": args.hard_queue_delay,
                        "retained_actions_exact": bool(np.array_equal(retained, hard_queue_period[:args.hard_queue_delay])),
                        "discarded_fresh_indices": list(range(args.hard_queue_delay)),
                        "executed_fresh_indices": list(range(args.hard_queue_delay, 10)),
                        "next_previous_indices": list(range(10, 50)),
                        "fresh_chunk_sha256": hashlib.sha256(fresh.tobytes()).hexdigest(),
                        "fresh_chunk": fresh.copy(),
                    })
                hard_queue_previous_plan = fresh[10:].copy()
            a = hard_queue_period[i % 10].copy()
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
        if snapshot_enabled:
            executed_actions[i + 1] = np.asarray(a, dtype=np.float32).copy()
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

        behavior.record(i + 1, a, joint, imgs, pts, links, geometric)

        if causal_enabled:
            causal_trace[i + 1] = _trace_state(i + 1)
        if snapshot_enabled and i + 1 in (5, 10):
            if causal_enabled and not prediction_chunks and causal_replay_actions is None:
                raise RuntimeError("causal snapshot reached before first policy chunk")
            boundary_images = render_images()
            snapshot = _snapshot_state(i + 1, boundary_images)
            if causal_enabled:
                causal_snapshots[i + 1] = snapshot
                log(f"CAUSAL_SNAPSHOT boundary={i + 1} chunks={len(prediction_chunks)}")
            if boundary_capture_enabled:
                if len(causal_replay_actions) < 50:
                    raise RuntimeError("boundary capture requires at least one complete cached H50 chunk")
                boundary = i + 1
                valid_target = np.asarray(causal_replay_actions[boundary:50], dtype=np.float32)
                if len(valid_target) != 50 - boundary:
                    raise RuntimeError(
                        f"boundary {boundary} target length {len(valid_target)} does not match H50 suffix")
                padded = np.concatenate([
                    valid_target,
                    np.repeat(valid_target[-1:, :], boundary, axis=0),
                ], axis=0)
                target_mask = np.zeros(50, dtype=np.bool_)
                target_mask[:len(valid_target)] = True
                target_indices = np.full(50, -1, dtype=np.int32)
                target_indices[:len(valid_target)] = np.arange(boundary, 50, dtype=np.int32)
                boundary_capture_rows[boundary] = {
                    "images": np.stack([boundary_images[k] for k in camera_keys]).astype(np.uint8),
                    "state": np.asarray(snapshot["joint"], dtype=np.float32),
                    "target_actions_rad": padded.astype(np.float32),
                    "target_valid_mask": target_mask,
                    "target_chunk_indices": target_indices,
                    "boundary": int(boundary),
                    "torch_rng": np.asarray(snapshot["torch_rng"], dtype=np.uint8),
                    "cuda_rng": [np.asarray(x, dtype=np.uint8) for x in snapshot["cuda_rng"]],
                }

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
    try:
        media["mp4"] = write_mp4(frames, prefix + ".mp4", step_dt, media_caption)
    except Exception as exc:
        # Causal branch data are independent of optional video encoding. A
        # broken ffmpeg pipe must not discard a completed simulator branch.
        log(f"MEDIA_WARNING {type(exc).__name__}: {exc}")
        media["mp4"] = {"error": repr(exc), "path": None}

    if boundary_capture_enabled:
        if set(boundary_capture_rows) != {5, 10}:
            raise RuntimeError(
                f"boundary capture expected action-5 and action-10 rows, got {sorted(boundary_capture_rows)}")
        rows = [boundary_capture_rows[b] for b in (5, 10)]
        checkpoint_files = {}
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                     "policy_preprocessor_step_5_normalizer_processor.safetensors",
                     "policy_postprocessor.json",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
            path = Path(args.policy_path) / name
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checkpoint_files[name] = {"path": str(path), "sha256": digest}
        source_path = Path(args.causal_action_jsonl)
        metadata = {
            "schema_version": 1,
            "boundaries": [5, 10],
            "garment": args.garment,
            "match_pose": args.match_pose,
            "match_scale": args.match_scale,
            "seed": args.seed,
            "task": args.task,
            "active_observation_keys": [
                "observation.images.top_rgb", "observation.images.left_rgb",
                "observation.images.right_rgb", "observation.state", "task",
            ],
            "observation_alignment": "images and state are captured after exactly boundary actions from the cached H50 episode",
            "target_alignment": "target row j is cached H50 chunk row boundary+j, zero-based; valid rows are boundary:50",
            "target_source": "successful cached H50 action stream, not a fresh replan, stale/hybrid input, RTC output, demonstration state, or interpolation",
            "target_storage": "raw executed_action_rad; training applies the unchanged checkpoint preprocessor normalizer",
            "boundary_rng_storage": "torch_rng and cuda_rng are captured immediately after the boundary action and are restored before each model prediction",
            "action_dim": 12,
            "chunk_size": 50,
            "valid_target_lengths": {"5": 45, "10": 40},
            "source_successful_h50_episode": str(source_path),
            "source_successful_h50_episode_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "checkpoint": str(args.policy_path),
            "checkpoint_files": checkpoint_files,
            "normalization_statistics": {
                name: checkpoint_files[name]["sha256"] for name in (
                    "policy_preprocessor.json",
                    "policy_preprocessor_step_5_normalizer_processor.safetensors",
                ) if name in checkpoint_files
            },
            "source_policy_variant": args.policy_variant,
            "original_checkpoint_untouched": True,
        }
        np.savez_compressed(
            args.boundary_capture_out,
            images=np.stack([row["images"] for row in rows]),
            state=np.stack([row["state"] for row in rows]),
            target_actions_rad=np.stack([row["target_actions_rad"] for row in rows]),
            target_valid_mask=np.stack([row["target_valid_mask"] for row in rows]),
            target_chunk_indices=np.stack([row["target_chunk_indices"] for row in rows]),
            boundaries=np.asarray([5, 10], dtype=np.int32),
            task=np.asarray([args.task, args.task]),
            torch_rng=np.stack([row["torch_rng"] for row in rows]),
            cuda_rng=np.stack([np.stack(row["cuda_rng"]) for row in rows]),
        )
        Path(args.boundary_capture_out).with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        log(f"BOUNDARY_CAPTURE_WRITTEN {args.boundary_capture_out}")

    causal_result = None
    if causal_enabled:
        # Branching is deliberately performed only after the ordinary H50
        # episode has finished. The saved snapshots are restored in-process;
        # no second simulator is allowed to silently change the scene.
        original_chunks = [np.asarray(x).copy() for x in prediction_chunks]
        original_trace = {int(k): v for k, v in causal_trace.items()}

        def _jsonable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.generic,)):
                return value.item()
            if isinstance(value, tuple):
                return [_jsonable(x) for x in value]
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_jsonable(x) for x in value]
            return value

        def _original_action(global_step):
            if int(global_step) not in executed_actions:
                raise RuntimeError(f"missing cached H50 executed action {global_step}")
            return executed_actions[int(global_step)].copy()

        def _branch_observation():
            images = render_images()
            joint = np.concatenate([_as_cpu_array(env.left_arm.data.joint_pos[0]),
                                    _as_cpu_array(env.right_arm.data.joint_pos[0])])
            return images, joint

        def _branch_step(action):
            env.step(torch.as_tensor(action, dtype=torch.float32, device=dev).reshape(1, 12))
            points = particles()
            links = link_poses()
            obs.update(points, links)
            geometric = checker_details()
            joint = np.concatenate([_as_cpu_array(env.left_arm.data.joint_pos[0]),
                                    _as_cpu_array(env.right_arm.data.joint_pos[0])])
            return {
                "joint": joint,
                "cloth_centroid": points.mean(axis=0),
                "left_ee": _as_cpu_array(links["left"][0][-1]),
                "right_ee": _as_cpu_array(links["right"][0][-1]),
                "conditions_passed": int(geometric["conditions_passed"]),
                "conditions_total": int(geometric["conditions_total"]),
                "geometric_success": bool(geometric["success"]),
                "condition_details": geometric.get("details", {}),
            }

        def _settle_branch(last_action):
            """Apply the same fixed hold-settle protocol used by the episode."""
            settled_trace = []
            for settle_step in range(1, args.terminal_settle_steps + 1):
                row = _branch_step(last_action)
                row["settle_step"] = settle_step
                settled_trace.append(row)
            terminal = settled_trace[-1] if settled_trace else None
            return {
                "steps": args.terminal_settle_steps,
                "trace": settled_trace,
                "terminal": terminal,
                "settled_4_of_4": bool(terminal and terminal["conditions_passed"] == 4),
                "settled_success": bool(terminal and terminal["geometric_success"]),
            }

        def _restore_and_validate(snapshot):
            _restore_state(snapshot)
            points = particles()
            joints = np.concatenate([_as_cpu_array(env.left_arm.data.joint_pos[0]),
                                     _as_cpu_array(env.right_arm.data.joint_pos[0])])
            return {
                "cloth_rms_m": float(np.sqrt(np.mean((points.reshape(-1, 3) - snapshot["cloth_positions"].reshape(-1, 3)) ** 2))),
                "joint_rms_rad": float(np.sqrt(np.mean((joints - snapshot["joint"]) ** 2))),
            }

        def _set_policy_rng(torch_state, cuda_state=None):
            """Restore the exact sampler state used for a reconstructed call."""
            torch.set_rng_state(torch.as_tensor(np.asarray(torch_state, dtype=np.uint8), device="cpu"))
            if cuda_state is not None:
                torch.cuda.set_rng_state_all([
                    torch.as_tensor(np.asarray(x, dtype=np.uint8), device="cpu")
                    for x in cuda_state
                ])

        def _predict_observed(batch, audit_label="causal_diagnostic", **kwargs):
            """Call inference and prove that no simulator action elapsed."""
            sim_before = _sim_counter()
            episode_before = _episode_counter()
            rng_before = _rng_digest()
            started = time.monotonic()
            chunk = original_predict(batch, **kwargs)
            torch.cuda.synchronize()
            sim_after = _sim_counter()
            episode_after = _episode_counter()
            timing = {
                "source": audit_label,
                "prediction_index": len(prediction_timing_audit),
                "sim_step_before": sim_before,
                "sim_step_after": sim_after,
                "sim_step_delta": sim_after - sim_before,
                "episode_length_before": episode_before,
                "episode_length_after": episode_after,
                "episode_length_delta": (None if episode_before is None or episode_after is None
                                           else episode_after - episode_before),
                "elapsed_seconds": time.monotonic() - started,
                "rng_before_digest": rng_before,
                "rng_after_digest": _rng_digest(),
                "inference_blocks_before_env_step": True,
            }
            prediction_timing_audit.append(timing)
            if sim_after != sim_before or (episode_before is not None and episode_after != episode_before):
                raise RuntimeError("policy inference advanced simulator state before env.step()")
            return chunk

        def _same_rng_chunk(snapshot):
            fidelity = _restore_and_validate(snapshot)
            policy.reset()
            _set_policy_rng(causal_initial_torch_rng, causal_initial_cuda_rng)
            images, joint = snapshot["images"], snapshot["joint"]
            return fidelity, _fresh_chunk(policy_batch(images, joint))

        def _rtc_configure(enabled):
            """Attach the installed LeRobot RTCProcessor without changing weights."""
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            if enabled:
                policy.config.rtc_config = RTCConfig(
                    enabled=True,
                    prefix_attention_schedule=RTCAttentionSchedule.EXP,
                    max_guidance_weight=10.0,
                    execution_horizon=10,
                    debug=False,
                )
            else:
                policy.config.rtc_config = None
            policy.init_rtc_processor()

        def _rtc_chunk(batch, previous_raw):
            kwargs = {
                "prev_chunk_left_over": None if previous_raw is None else torch.as_tensor(
                    previous_raw, dtype=torch.float32, device=dev).unsqueeze(0),
                "inference_delay": 0,
                "execution_horizon": 10,
            }
            # RTC's upstream processor temporarily enables autograd for its
            # prefix correction, so use no_grad here rather than inference_mode.
            with torch.no_grad():
                chunk = _predict_observed(batch, **kwargs)
            raw = np.asarray(chunk.detach().cpu().numpy()[0] if hasattr(chunk, "detach") else chunk).copy()
            processed = post(chunk) if post else chunk
            processed = np.asarray(processed.detach().cpu().numpy()[0] if hasattr(processed, "detach") else processed).copy()
            return raw, processed

        def _rtc_no_previous_check(snapshot):
            """RTC with no previous chunk must equal ordinary inference and not step sim."""
            fidelity = _restore_and_validate(snapshot)
            policy.reset()
            _set_policy_rng(causal_initial_torch_rng, causal_initial_cuda_rng)
            _rtc_configure(False)
            ordinary_raw, ordinary = _rtc_chunk(policy_batch(snapshot["images"], snapshot["joint"]), None)
            ordinary_rng_after = _rng_digest()
            _restore_and_validate(snapshot)
            policy.reset()
            _set_policy_rng(causal_initial_torch_rng, causal_initial_cuda_rng)
            _rtc_configure(True)
            rtc_raw, rtc = _rtc_chunk(policy_batch(snapshot["images"], snapshot["joint"]), None)
            rtc_rng_after = _rng_digest()
            return {
                "restore": fidelity,
                "max_abs_postprocessed_action": float(np.max(np.abs(ordinary - rtc))),
                "max_abs_raw_action": float(np.max(np.abs(ordinary_raw - rtc_raw))),
                "ordinary_rng_after": ordinary_rng_after,
                "rtc_rng_after": rtc_rng_after,
                "rng_after_equal": ordinary_rng_after == rtc_rng_after,
                "simulator_stepped": False,
                "config": {"execution_horizon": 10, "prefix_attention_schedule": "EXP", "max_guidance_weight": 10.0},
            }

        def _run_rtc(snapshot, horizon):
            fidelity = _restore_and_validate(snapshot)
            policy.reset()
            _set_policy_rng(causal_initial_torch_rng, causal_initial_cuda_rng)
            _rtc_configure(True)
            boundary = int(snapshot["boundary"])
            previous_raw = None
            if reconstructed_original_chunk_raw is not None:
                previous_raw = np.asarray(reconstructed_original_chunk_raw)[boundary:].copy()
            trace, actions, chunks = [], [], []
            best = 0
            last = None
            current_processed = None
            for local in range(args.causal_branch_steps):
                global_step = boundary + local + 1
                if local % horizon == 0:
                    if local == 0:
                        images, joint = snapshot["images"], snapshot["joint"]
                    else:
                        images, joint = _branch_observation()
                    raw, current_processed = _rtc_chunk(policy_batch(images, joint), previous_raw)
                    chunks.append({"raw": raw.copy(), "processed": current_processed.copy(), "previous_raw_length": 0 if previous_raw is None else len(previous_raw)})
                    previous_raw = raw[horizon:].copy()
                action = current_processed[local % horizon].copy()
                actions.append(action)
                row = _branch_step(action)
                row["global_step"] = global_step
                trace.append(row)
                best = max(best, row["conditions_passed"])
                last = row
            return {"restore": fidelity, "actions": actions, "trace": trace, "chunks": chunks,
                    "best_conditions": best, "terminal": last,
                    "geometric_ever_success": any(x["geometric_success"] for x in trace),
                    "settled": _settle_branch(actions[-1]),
                    "config": {"execution_horizon": 10, "prefix_attention_schedule": "EXP", "max_guidance_weight": 10.0, "inference_delay": 0}}

        def _run_same_rng(snapshot, horizon):
            """Execute a branch whose first replan uses the reconstructed original RNG state."""
            fidelity, first_chunk = _same_rng_chunk(snapshot)
            first_prediction_timing = dict(prediction_timing_audit[-1])
            boundary = int(snapshot["boundary"])
            trace, actions, chunks = [], [], [first_chunk.copy()]
            best = 0
            last = None
            current_chunk = first_chunk
            for local in range(args.causal_branch_steps):
                global_step = boundary + local + 1
                if local and local % horizon == 0:
                    images, joint = _branch_observation()
                    current_chunk = _fresh_chunk(policy_batch(images, joint))
                    chunks.append(current_chunk.copy())
                action = current_chunk[local % horizon].copy()
                actions.append(action)
                row = _branch_step(action)
                row["global_step"] = global_step
                trace.append(row)
                best = max(best, row["conditions_passed"])
                last = row
            return {"restore": fidelity, "actions": actions, "trace": trace, "chunks": chunks,
                    "best_conditions": best, "terminal": last,
                    "geometric_ever_success": any(x["geometric_success"] for x in trace),
                    "first_prediction_timing": first_prediction_timing,
                    "settled": _settle_branch(actions[-1])}

        def _seed_panel(snapshot):
            plans = []
            batch = policy_batch(snapshot["images"], snapshot["joint"])
            for seed in causal_seed_panel:
                _restore_and_validate(snapshot)
                policy.reset()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                plans.append({"seed": seed, "chunk": _fresh_chunk(batch)})
            return plans

        def _compare_branch_trace(boundary, trace):
            errors = []
            for local, row in enumerate(trace, start=1):
                reference = original_trace.get(boundary + local)
                if reference is None:
                    continue
                errors.append({
                    "local_step": local,
                    "global_step": boundary + local,
                    "joint_rms_rad": float(np.sqrt(np.mean((row["joint"] - reference["joint"]) ** 2))),
                    "cloth_centroid_error_m": float(np.linalg.norm(row["cloth_centroid"] - reference["cloth_centroid"])),
                    "left_ee_error_m": float(np.linalg.norm(row["left_ee"] - reference["left_ee"])),
                    "right_ee_error_m": float(np.linalg.norm(row["right_ee"] - reference["right_ee"])),
                })
            return errors

        def _run_cached(snapshot):
            fidelity = _restore_and_validate(snapshot)
            boundary = int(snapshot["boundary"])
            trace, actions = [], []
            best = 0
            last = None
            for global_step in range(boundary + 1, min(600, boundary + args.causal_branch_steps) + 1):
                action = _original_action(global_step)
                actions.append(action)
                row = _branch_step(action)
                row["global_step"] = global_step
                trace.append(row)
                best = max(best, row["conditions_passed"])
                last = row
            return {"restore": fidelity, "actions": actions, "trace": trace,
                    "restore_fidelity_errors": _compare_branch_trace(boundary, trace),
                    "best_conditions": best, "terminal": last,
                    "geometric_ever_success": any(x["geometric_success"] for x in trace),
                    "settled": _settle_branch(actions[-1])}

        def _fresh_chunk(batch, audit_label="causal_fresh_replan"):
            with torch.inference_mode():
                chunk = _predict_observed(batch, audit_label=audit_label)
            # _get_action_chunk returns the model-space normalized chunk. The
            # rollout applies the postprocessor to each selected action, so
            # apply the same postprocessor to the whole saved chunk here.
            if post:
                chunk = post(chunk)
            if hasattr(chunk, "detach"):
                chunk = chunk.detach().cpu().numpy()
            return np.asarray(chunk)[0].copy()

        observation_probe_specs = (
            ("current-all", "control", "current images and current state"),
            ("old-all-images-current-state", "attribution_probe",
             "action-0 H50 images with current state"),
            ("current-images-old-state", "attribution_probe",
             "current images with action-0 H50 state"),
            ("old-top-only", "attribution_probe",
             "action-0 top image with current wrists and state"),
            ("old-left-only", "attribution_probe",
             "action-0 left-wrist image with current top/right and state"),
            ("old-right-only", "attribution_probe",
             "action-0 right-wrist image with current top/left and state"),
            ("old-both-wrists", "attribution_probe",
             "action-0 left/right-wrist images with current top and state"),
        )

        def _prefix_feature_vector():
            """Return the existing passive prefix feature tap, if available."""
            if observation_feature_tap is None or observation_feature_tap.last is None:
                return None
            value = observation_feature_tap.last.detach().float()
            mask = observation_feature_tap.last_mask
            if value.dim() == 3:
                if mask is None:
                    value = value.mean(dim=1)
                else:
                    mask = mask.to(value.dtype)
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    value = (value * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            if value.dim() != 2:
                return None
            return value[0].detach().cpu().numpy().astype(np.float32, copy=True)

        def _feature_distance(reference, candidate):
            if reference is None or candidate is None:
                return {"available": False, "rms": None, "l2": None}
            reference = np.asarray(reference, dtype=np.float64).reshape(-1)
            candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if reference.shape != candidate.shape:
                return {"available": False, "rms": None, "l2": None,
                        "shape_reference": list(reference.shape),
                        "shape_candidate": list(candidate.shape)}
            delta = candidate - reference
            return {"available": True,
                    "rms": float(np.sqrt(np.mean(delta * delta))),
                    "l2": float(np.linalg.norm(delta)),
                    "dimension": int(delta.size)}

        def _counterfactual_observation(name, current_images, current_joint):
            if causal_initial_observation is None:
                raise RuntimeError("action-0 H50 observation was not captured")
            old_images = causal_initial_observation["images"]
            old_joint = causal_initial_observation["joint"]
            images = {key: np.asarray(current_images[key], dtype=np.uint8).copy()
                      for key in camera_keys}
            joint = np.asarray(current_joint, dtype=np.float32).copy()
            if name == "old-all-images-current-state":
                images = {key: old_images[key].copy() for key in camera_keys}
            elif name == "current-images-old-state":
                joint = old_joint.copy()
            elif name == "old-top-only":
                images["top_rgb"] = old_images["top_rgb"].copy()
            elif name == "old-left-only":
                images["left_rgb"] = old_images["left_rgb"].copy()
            elif name == "old-right-only":
                images["right_rgb"] = old_images["right_rgb"].copy()
            elif name == "old-both-wrists":
                images["left_rgb"] = old_images["left_rgb"].copy()
                images["right_rgb"] = old_images["right_rgb"].copy()
            elif name != "current-all":
                raise KeyError(f"unknown observation component probe {name!r}")
            return images, joint

        def _observation_action_metrics(old_plan, chunk):
            """Preregis­tered suffix-distance metrics, including per-joint deltas."""
            old_plan = np.asarray(old_plan, dtype=np.float64)
            chunk = np.asarray(chunk, dtype=np.float64)
            metrics = {}
            for width in (1, 5, 10):
                width = min(width, len(old_plan), len(chunk))
                delta = chunk[:width] - old_plan[:width]
                metrics[str(width)] = {
                    "width": width,
                    "rms_rad": float(np.sqrt(np.mean(delta * delta))),
                    "mean_abs_rad": float(np.abs(delta).mean()),
                    "max_abs_rad": float(np.abs(delta).max(initial=0.0)),
                    "per_joint_rms_rad": np.sqrt(np.mean(delta * delta, axis=0)).tolist(),
                    "per_joint_mean_abs_rad": np.abs(delta).mean(axis=0).tolist(),
                    "action_l2_rad": np.linalg.norm(delta, axis=1).tolist(),
                }
            return metrics

        def _rng_state_digest(torch_state, cuda_state):
            digest = hashlib.sha256()
            digest.update(np.asarray(torch_state, dtype=np.uint8).tobytes())
            for state in cuda_state:
                digest.update(np.asarray(state, dtype=np.uint8).tobytes())
            return digest.hexdigest()

        def _predict_exact_observation(snapshot, images, joint, audit_label,
                                       use_initial_rng):
            """Restore simulator/RNG, then predict; prediction itself cannot step sim."""
            fidelity = _restore_and_validate(snapshot)
            policy.reset()
            if use_initial_rng:
                torch_state, cuda_state = causal_initial_torch_rng, causal_initial_cuda_rng
                rng_source = "reconstructed_initial_rng"
            else:
                torch_state, cuda_state = snapshot["torch_rng"], snapshot["cuda_rng"]
                rng_source = "reconstructed_continuation_rng"
            _set_policy_rng(torch_state, cuda_state)
            rng_restore_exact = _rng_digest() == _rng_state_digest(torch_state, cuda_state)
            if observation_feature_tap is not None:
                # Prevent a missing forward-hook call from reusing the prior
                # candidate's feature vector.
                observation_feature_tap.last = None
                observation_feature_tap.last_mask = None
            chunk = _fresh_chunk(policy_batch(images, joint), audit_label=audit_label)
            timing = dict(prediction_timing_audit[-1])
            feature = _prefix_feature_vector()
            return {
                "chunk": chunk,
                "feature": feature,
                "restore": fidelity,
                "prediction_timing": timing,
                "rng_source": rng_source,
                "rng_restore_exact": bool(rng_restore_exact),
                "simulator_stepped_during_prediction": bool(
                    timing["sim_step_delta"] or timing["episode_length_delta"] not in (0, None)),
            }

        def _probe_observation_components(snapshot, old_plan, same_rng_chunk):
            """Evaluate all stale/hybrid inputs without executing any probe action."""
            boundary = int(snapshot["boundary"])
            records = {}
            reference_feature = None
            for name, kind, description in observation_probe_specs:
                images, joint = _counterfactual_observation(
                    name, snapshot["images"], snapshot["joint"])
                prediction = _predict_exact_observation(
                    snapshot, images, joint,
                    audit_label=f"observation_probe_{name}_boundary{boundary}",
                    use_initial_rng=True)
                if name == "current-all":
                    reference_feature = prediction["feature"]
                action_identity = _action_identity(prediction["chunk"][:10],
                                                   same_rng_chunk[:10])
                records[name] = {
                    "name": name,
                    "classification": kind,
                    "description": description,
                    "deployable_controller": False if kind == "attribution_probe" else True,
                    "input_components": {
                        "state": "old_action_0_H50" if name == "current-images-old-state" else "current",
                        "top_rgb": "old_action_0_H50" if name in ("old-all-images-current-state", "old-top-only") else "current",
                        "left_rgb": "old_action_0_H50" if name in ("old-all-images-current-state", "old-left-only", "old-both-wrists") else "current",
                        "right_rgb": "old_action_0_H50" if name in ("old-all-images-current-state", "old-right-only", "old-both-wrists") else "current",
                        "task": "unchanged",
                    },
                    "restore": prediction["restore"],
                    "prediction_timing": prediction["prediction_timing"],
                    "rng_source": prediction["rng_source"],
                    "rng_restore_exact": prediction["rng_restore_exact"],
                    "simulator_stepped_during_prediction": prediction["simulator_stepped_during_prediction"],
                    "same_rng_current_all_first10_exact": action_identity["exact"],
                    "first10_chunk_vs_same_rng": action_identity,
                    "action_metrics_vs_cached_h50_suffix": _observation_action_metrics(
                        old_plan, prediction["chunk"]),
                    "prefix_feature_distance_to_current_all": None,
                    "chunk_first_10": prediction["chunk"][:10].copy(),
                }
                records[name]["_feature"] = prediction["feature"]
            for name in records:
                records[name]["prefix_feature_distance_to_current_all"] = _feature_distance(
                    reference_feature, records[name].pop("_feature"))
            timing_records = [records[name]["prediction_timing"] for name, _, _ in observation_probe_specs]
            return {
                "boundary": boundary,
                "probes": records,
                "all_snapshot_restores_within_tolerance": all(
                    row["restore"]["cloth_rms_m"] <= 1e-5 and row["restore"]["joint_rms_rad"] <= 1e-5
                    for row in records.values()),
                "all_predictions_zero_sim_steps": all(
                    not row["simulator_stepped_during_prediction"] for row in records.values()),
                "all_rng_restores_exact": all(row["rng_restore_exact"] for row in records.values()),
                "current_all_matches_same_rng_control": records["current-all"]["same_rng_current_all_first10_exact"],
                "feature_status": observation_feature_status,
                "prediction_timing_records": len(timing_records),
            }

        def _run_observation_candidate(snapshot, horizon, name):
            """Execute one selected attribution probe, auditing each H10 prediction."""
            boundary = int(snapshot["boundary"])
            branch_steps = min(args.causal_branch_steps, args.steps - boundary)
            trace, actions, chunks, prediction_events = [], [], [], []
            current_chunk = None
            best = 0
            last = None
            for local in range(branch_steps):
                global_step = boundary + local + 1
                if local % horizon == 0:
                    if local == 0:
                        images, joint = snapshot["images"], snapshot["joint"]
                        prediction_snapshot = snapshot
                        use_initial_rng = True
                    else:
                        images, joint = _branch_observation()
                        prediction_snapshot = _snapshot_state(global_step - 1, images)
                        use_initial_rng = False
                    probe_images, probe_joint = _counterfactual_observation(name, images, joint)
                    prediction = _predict_exact_observation(
                        prediction_snapshot, probe_images, probe_joint,
                        audit_label=f"observation_execute_{name}_boundary{boundary}_replan{local // horizon}",
                        use_initial_rng=use_initial_rng)
                    current_chunk = prediction["chunk"]
                    chunks.append(current_chunk.copy())
                    prediction_events.append({
                        "replan_index": local // horizon,
                        "global_action_start": global_step,
                        "observation_name": name,
                        "restore": prediction["restore"],
                        "prediction_timing": prediction["prediction_timing"],
                        "rng_source": prediction["rng_source"],
                        "rng_restore_exact": prediction["rng_restore_exact"],
                        "simulator_stepped_during_prediction": prediction["simulator_stepped_during_prediction"],
                        "chunk_first_10": current_chunk[:10].copy(),
                    })
                action = current_chunk[local % horizon].copy()
                actions.append(action)
                row = _branch_step(action)
                row["global_step"] = global_step
                trace.append(row)
                best = max(best, row["conditions_passed"])
                last = row
            if not actions:
                raise RuntimeError("observation candidate branch produced no action")
            settled = _settle_branch(actions[-1])
            return {
                "observation_name": name,
                "classification": "attribution_probe",
                "deployable_controller": False,
                "restore": prediction_events[0]["restore"],
                "branch_steps": branch_steps,
                "horizon": horizon,
                "actions": actions,
                "chunks": chunks,
                "trace": trace,
                "condition_trajectory": [
                    {"global_step": row["global_step"],
                     "conditions_passed": row["conditions_passed"],
                     "conditions_total": row["conditions_total"],
                     "geometric_success": row["geometric_success"],
                     "condition_details": row["condition_details"]}
                    for row in trace
                ],
                "prediction_events": prediction_events,
                "divergence_from_h50": _compare_branch_trace(boundary, trace),
                "best_conditions": best,
                "terminal": last,
                "geometric_ever_success": any(x["geometric_success"] for x in trace),
                "settled": settled,
                "all_snapshot_restores_within_tolerance": all(
                    event["restore"]["cloth_rms_m"] <= 1e-5
                    and event["restore"]["joint_rms_rad"] <= 1e-5
                    for event in prediction_events),
                "all_predictions_zero_sim_steps": all(
                    not event["simulator_stepped_during_prediction"]
                    for event in prediction_events),
                "all_rng_restores_exact": all(
                    event["rng_restore_exact"] for event in prediction_events),
            }

        def _run_fresh(snapshot, horizon):
            fidelity = _restore_and_validate(snapshot)
            boundary = int(snapshot["boundary"])
            trace, actions, chunks = [], [], []
            best = 0
            last = None
            current_chunk = None
            for local in range(args.causal_branch_steps):
                global_step = boundary + local + 1
                if local % horizon == 0:
                    if local == 0:
                        images, joint = snapshot["images"], snapshot["joint"]
                    else:
                        images, joint = _branch_observation()
                    current_chunk = _fresh_chunk(policy_batch(images, joint))
                    chunks.append(current_chunk.copy())
                action = current_chunk[local % horizon].copy()
                actions.append(action)
                row = _branch_step(action)
                row["global_step"] = global_step
                trace.append(row)
                best = max(best, row["conditions_passed"])
                last = row
            return {"restore": fidelity, "actions": actions, "trace": trace, "chunks": chunks,
                    "best_conditions": best, "terminal": last,
                    "geometric_ever_success": any(x["geometric_success"] for x in trace),
                    "settled": _settle_branch(actions[-1])}

        def _action_identity(actual, expected):
            actual = np.asarray(actual)
            expected = np.asarray(expected)
            if actual.shape != expected.shape:
                return {"exact": False, "shape_actual": list(actual.shape),
                        "shape_expected": list(expected.shape), "max_abs": None, "l2": None}
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            return {"exact": bool(np.array_equal(actual, expected)),
                    "shape_actual": list(actual.shape), "shape_expected": list(expected.shape),
                    "max_abs": float(np.max(np.abs(delta), initial=0.0)),
                    "l2": float(np.linalg.norm(delta))}

        def _run_queue(snapshot, horizon, delay, old_plan):
            """Run a hard retained-prefix handoff with a deterministic fresh plan.

            At every H10 boundary the previous plan contributes exactly ``delay``
            actions. The fresh 50-row plan is generated from the current
            observation/RNG stream, its rows [0:delay] are discarded, and rows
            [delay:10] fill the remainder of that ten-action execution period.
            The unconsumed fresh suffix [10:50] becomes the previous-plan
            remainder at the next boundary.
            """
            fidelity = _restore_and_validate(snapshot)
            policy.reset()
            _set_policy_rng(causal_initial_torch_rng, causal_initial_cuda_rng)
            boundary = int(snapshot["boundary"])
            branch_steps = min(args.causal_branch_steps, args.steps - boundary)
            previous_plan = np.asarray(old_plan, dtype=np.float32).copy()
            previous_indices = np.arange(boundary, boundary + len(previous_plan), dtype=np.int32)
            previous_source = {"kind": "cached_h50", "replan_index": None,
                               "indices": previous_indices.tolist()}
            trace, actions, handoffs = [], [], []
            best = 0
            last = None
            period_previous = None
            period_previous_indices = None
            period_fresh = None
            period_event = None
            for local in range(branch_steps):
                global_step = boundary + local + 1
                if local % horizon == 0:
                    replan_index = local // horizon
                    if local == 0:
                        images, joint = snapshot["images"], snapshot["joint"]
                    else:
                        images, joint = _branch_observation()
                    period_previous = previous_plan.copy()
                    period_previous_indices = previous_indices.copy()
                    if len(period_previous) < delay:
                        raise RuntimeError(
                            f"queue delay {delay} exceeds previous-plan remainder "
                            f"length {len(period_previous)} at boundary {boundary}+{local}"
                        )
                    fresh = _fresh_chunk(
                        policy_batch(images, joint),
                        audit_label=f"queue_d{delay}_pose_boundary{boundary}_replan{replan_index}",
                    )
                    if fresh.shape != (50, 12):
                        raise RuntimeError(f"queue fresh chunk has shape {fresh.shape}, expected (50, 12)")
                    timing = dict(prediction_timing_audit[-1])
                    period_fresh = fresh.copy()
                    period_event = {
                        "replan_index": replan_index,
                        "boundary_action": boundary,
                        "global_action_start": global_step,
                        "delay": delay,
                        "horizon": horizon,
                        "previous_plan_source": previous_source,
                        "previous_plan_length": int(len(period_previous)),
                        "retained_previous_indices": period_previous_indices[:delay].tolist(),
                        "discarded_previous_suffix_count": int(len(period_previous) - delay),
                        "fresh_chunk_length": int(len(fresh)),
                        "fresh_chunk_sha256": hashlib.sha256(fresh.tobytes()).hexdigest(),
                        "fresh_chunk": fresh.copy(),
                        "discarded_fresh_indices": list(range(delay)),
                        "executed_fresh_indices": list(range(delay, horizon)),
                        "next_previous_indices": list(range(horizon, len(fresh))),
                        "prediction_timing": timing,
                        "rng_before_generation": timing["rng_before_digest"],
                        "rng_after_generation": timing["rng_after_digest"],
                        "simulator_stepped_during_prediction": bool(timing["sim_step_delta"]),
                    }
                    handoffs.append(period_event)
                    previous_plan = fresh[horizon:].copy()
                    previous_indices = np.arange(horizon, len(fresh), dtype=np.int32)
                    previous_source = {"kind": "fresh_chunk_remainder",
                                       "replan_index": replan_index,
                                       "indices": previous_indices.tolist()}
                offset = local % horizon
                if offset < delay:
                    action = period_previous[offset].copy()
                    expected = period_previous[offset]
                    source = "previous_plan_remainder"
                    source_index = int(period_previous_indices[offset])
                else:
                    action = period_fresh[offset].copy()
                    expected = period_fresh[offset]
                    source = "fresh_chunk_suffix"
                    source_index = offset
                identity = _action_identity(action, expected)
                if not identity["exact"]:
                    raise RuntimeError(f"queue action identity failed at global action {global_step}")
                actions.append(action)
                row = _branch_step(action)
                row.update({"global_step": global_step, "queue_delay": delay,
                            "replan_index": local // horizon, "period_offset": offset,
                            "action_source": source, "source_index": source_index,
                            "action_identity": identity})
                trace.append(row)
                best = max(best, row["conditions_passed"])
                last = row
            if period_event is None or last is None:
                raise RuntimeError("queue branch produced no action period")
            for event in handoffs:
                event["fresh_prefix_executed"] = any(
                    row["replan_index"] == event["replan_index"]
                    and row["period_offset"] < delay
                    and row["action_source"] == "fresh_chunk_suffix"
                    for row in trace
                )
                event["retained_actions_exact"] = all(
                    row["replan_index"] == event["replan_index"]
                    and row["period_offset"] < delay
                    and row["action_source"] == "previous_plan_remainder"
                    and row["action_identity"]["exact"]
                    for row in trace
                    if row["replan_index"] == event["replan_index"]
                    and row["period_offset"] < delay
                )
                event["fresh_suffix_indexing_exact"] = all(
                    row["replan_index"] == event["replan_index"]
                    and row["period_offset"] >= delay
                    and row["action_source"] == "fresh_chunk_suffix"
                    and row["source_index"] == row["period_offset"]
                    for row in trace
                    if row["replan_index"] == event["replan_index"]
                    and row["period_offset"] >= delay
                )
            settled = _settle_branch(actions[-1])
            return {
                "restore": fidelity,
                "delay": delay,
                "horizon": horizon,
                "branch_steps": branch_steps,
                "actions": actions,
                "trace": trace,
                "condition_trajectory": [
                    {"global_step": row["global_step"],
                     "conditions_passed": row["conditions_passed"],
                     "conditions_total": row["conditions_total"],
                     "geometric_success": row["geometric_success"],
                     "condition_details": row["condition_details"]}
                    for row in trace
                ],
                "handoff_events": handoffs,
                "first_fresh_chunk": handoffs[0]["fresh_chunk"],
                "divergence_from_h50": _compare_branch_trace(boundary, trace),
                "best_conditions": best,
                "terminal": last,
                "geometric_ever_success": any(x["geometric_success"] for x in trace),
                "settled": settled,
                "all_retained_actions_exact": all(x["retained_actions_exact"] for x in handoffs),
                "all_fresh_prefixes_discarded": all(not x["fresh_prefix_executed"] for x in handoffs),
                "all_fresh_suffix_indices_exact": all(x["fresh_suffix_indexing_exact"] for x in handoffs),
                "all_predictions_zero_sim_steps": all(
                    int(x["prediction_timing"]["sim_step_delta"]) == 0
                    and (x["prediction_timing"]["episode_length_delta"] in (0, None))
                    for x in handoffs
                ),
            }

        def _repeat_predictions(snapshot, count=4):
            deterministic, varied = [], []
            batch = policy_batch(snapshot["images"], snapshot["joint"])
            for _ in range(count):
                _restore_and_validate(snapshot)
                policy.reset()
                with torch.inference_mode():
                    chunk = _predict_observed(batch)
                deterministic.append(np.asarray(chunk.detach().cpu().numpy()[0]).copy())
            for seed in range(count):
                _restore_and_validate(snapshot)
                policy.reset()
                torch.manual_seed(seed + 101)
                torch.cuda.manual_seed_all(seed + 101)
                with torch.inference_mode():
                    chunk = _predict_observed(batch)
                varied.append(np.asarray(chunk.detach().cpu().numpy()[0]).copy())
            return {"deterministic_repeat": deterministic, "varied_seed": varied}

        def _plan_metrics(old_plan, new_plan):
            output = {}
            for width in (1, 3, 5, 10):
                old = np.asarray(old_plan[:width])
                new = np.asarray(new_plan[:width])
                delta = new - old
                output[str(width)] = {
                    "mean_abs_rad": float(np.abs(delta).mean()),
                    "l2_per_action_rad": np.linalg.norm(delta, axis=1).tolist(),
                    "left_arm_l2_rad": np.linalg.norm(delta[:, :6], axis=1).tolist(),
                    "right_arm_l2_rad": np.linalg.norm(delta[:, 6:], axis=1).tolist(),
                    "gripper_delta_rad": {"left": delta[:, 5].tolist(), "right": delta[:, 11].tolist()},
                    "direction_cosine": [float(np.dot(old[i], new[i]) / (np.linalg.norm(old[i]) * np.linalg.norm(new[i]) + 1e-12)) for i in range(len(delta))],
                }
            return output

        branches = {}
        observation_component_records = {}

        def _checkpoint_hashes(checkpoint):
            names = ("config.json", "model.safetensors", "policy_preprocessor.json",
                     "policy_preprocessor_step_5_normalizer_processor.safetensors",
                     "policy_postprocessor.json",
                     "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
            return {
                name: {"path": str(Path(checkpoint) / name),
                       "sha256": hashlib.sha256((Path(checkpoint) / name).read_bytes()).hexdigest()}
                for name in names if (Path(checkpoint) / name).exists()
            }

        def _deviation_metrics(target, candidate):
            target = np.asarray(target, dtype=np.float64)
            candidate = np.asarray(candidate, dtype=np.float64)
            output = {}
            for width in (1, 10):
                delta = candidate[:width] - target[:width]
                output[str(width)] = {
                    "width": width,
                    "rms_rad": float(np.sqrt(np.mean(delta * delta))),
                    "max_abs_rad": float(np.abs(delta).max(initial=0.0)),
                    "mean_abs_rad": float(np.abs(delta).mean()),
                    "per_joint_rms_rad": np.sqrt(np.mean(delta * delta, axis=0)).tolist(),
                    "per_joint_max_abs_rad": np.abs(delta).max(axis=0).tolist(),
                    "first_action_delta_rad": delta[0].tolist(),
                }
            return output

        def _run_boundary_checkpoint_eval():
            """Compare base/trained plans on restored sim snapshots only."""
            trained_path = Path(args.boundary_eval_trained_path)
            if not trained_path.is_dir():
                raise RuntimeError(f"trained boundary checkpoint is missing: {trained_path}")
            # save_pretrained() preserves the model config but this older
            # LeRobot checkout writes a standalone config.json without the
            # top-level policy-choice `type`. Reuse the already parsed
            # baseline policy config; processors still read their files from
            # the explicit trained checkpoint path below.
            trained_cfg = pcfg
            trained_cfg.pretrained_path = str(trained_path)
            # The fine-tuner's save_pretrained config is model-shaped but this
            # checkout's loader also requires the baseline policy-choice
            # `type`. Use a temporary compatibility directory so the trained
            # artifact itself stays unchanged: all trained files are symlinked
            # and only config.json is replaced with the untouched baseline
            # policy config for parsing.
            compat_dir = Path(tempfile.mkdtemp(prefix="boundary-eval-"))
            for file in trained_path.iterdir():
                os.symlink(file, compat_dir / file.name)
            shutil.copy2(Path(args.policy_path) / "config.json", compat_dir / "config.json")
            trained_model = SmolVLAPolicy.from_pretrained(str(compat_dir)).eval().to(dev)
            trained_pre, trained_post = make_pre_post_processors(
                policy_cfg=trained_cfg, pretrained_path=str(trained_path))
            checkpoints = {
                "baseline": {"path": args.policy_path, "model": policy,
                              "pre": pre, "post": post},
                "trained": {"path": str(trained_path), "model": trained_model,
                             "pre": trained_pre, "post": trained_post},
            }
            records = {}
            for boundary, snapshot in sorted(causal_snapshots.items()):
                if boundary not in (5, 10):
                    continue
                target = np.asarray(causal_replay_actions[boundary:50], dtype=np.float32)
                model_records = {}
                rng_source = "reconstructed_initial_rng_after_seed_rngs"
                for label, item in checkpoints.items():
                    fidelity = _restore_and_validate(snapshot)
                    model = item["model"]
                    model.reset()
                    _set_policy_rng(causal_initial_torch_rng, causal_initial_cuda_rng)
                    rng_before = _rng_digest()
                    sim_before = _sim_counter()
                    episode_before = _episode_counter()
                    started = time.monotonic()
                    batch = make_observation(snapshot["images"], snapshot["joint"])
                    batch = item["pre"](batch) if item["pre"] else batch
                    predictor = original_predict if label == "baseline" else model._get_action_chunk
                    with torch.inference_mode():
                        raw = predictor(batch)
                    torch.cuda.synchronize()
                    if item["post"]:
                        raw = item["post"](raw)
                    chunk = np.asarray(raw.detach().cpu().numpy()[0] if hasattr(raw, "detach") else raw).copy()
                    sim_after = _sim_counter()
                    episode_after = _episode_counter()
                    rng_after = _rng_digest()
                    timing = {
                        "source": f"boundary_eval_{label}",
                        "boundary": boundary,
                        "elapsed_seconds": time.monotonic() - started,
                        "sim_step_before": sim_before,
                        "sim_step_after": sim_after,
                        "sim_step_delta": sim_after - sim_before,
                        "episode_length_before": episode_before,
                        "episode_length_after": episode_after,
                        "episode_length_delta": (None if episode_before is None or episode_after is None
                                                   else episode_after - episode_before),
                        "rng_before_digest": rng_before,
                        "rng_after_digest": rng_after,
                        "inference_blocks_before_env_step": True,
                    }
                    prediction_timing_audit.append(timing)
                    if chunk.shape != (50, 12):
                        raise RuntimeError(f"{label} boundary chunk has shape {chunk.shape}, expected (50, 12)")
                    if sim_after != sim_before or (episode_before is not None and episode_after != episode_before):
                        raise RuntimeError(f"{label} boundary inference advanced simulator state")
                    model_records[label] = {
                        "checkpoint": item["path"],
                        "rng_source": rng_source,
                        "rng_restore_exact": rng_before == _rng_state_digest(
                            causal_initial_torch_rng, causal_initial_cuda_rng),
                        "simulator_stepped_during_prediction": False,
                        "snapshot_restore": fidelity,
                        "prediction_timing": timing,
                        "chunk_first_10": chunk[:10],
                        "deviation_from_cached_h50_suffix": _deviation_metrics(target, chunk),
                    }
                if model_records["baseline"]["prediction_timing"]["rng_before_digest"] != model_records["trained"]["prediction_timing"]["rng_before_digest"]:
                    raise RuntimeError(f"boundary {boundary} baseline/trained RNG streams differ")
                records[str(boundary)] = {
                    "target_source": "cached successful H50 executed actions after this boundary",
                    "target_length": len(target),
                    "baseline_and_trained_same_rng": True,
                    "models": model_records,
                }
            del trained_model
            torch.cuda.empty_cache()
            shutil.rmtree(compat_dir, ignore_errors=True)
            return {
                "schema_version": 1,
                "mode": "snapshot_only_boundary_replay",
                "closed_loop_execution": False,
                "rtc_used": False,
                "queue_or_retained_prefix_used": False,
                "observation_substitution_used": False,
                "checkpoints": {label: {"path": item["path"],
                                         "files": _checkpoint_hashes(item["path"])}
                                 for label, item in checkpoints.items()},
                "boundaries": records,
                "all_snapshot_restores_within_tolerance": all(
                    row["models"][label]["snapshot_restore"]["cloth_rms_m"] <= 1e-5 and
                    row["models"][label]["snapshot_restore"]["joint_rms_rad"] <= 1e-5
                    for row in records.values() for label in ("baseline", "trained")),
                "all_predictions_zero_sim_steps": all(
                    not row["models"][label]["simulator_stepped_during_prediction"]
                    for row in records.values() for label in ("baseline", "trained")),
                "all_rng_restores_exact": all(
                    row["models"][label]["rng_restore_exact"]
                    for row in records.values() for label in ("baseline", "trained")),
            }

        if args.boundary_eval_trained_path:
            boundary_eval_result = _run_boundary_checkpoint_eval()
            branches["boundary_eval"] = boundary_eval_result
            with open(args.boundary_eval_out, "w") as boundary_eval_file:
                json.dump(_jsonable(boundary_eval_result), boundary_eval_file, indent=2, allow_nan=False)
            log(f"BOUNDARY_EVAL_WRITTEN {args.boundary_eval_out}")
        else:
            boundary_eval_result = None
            for boundary, snapshot in sorted(causal_snapshots.items()):
                h = 10 if boundary == 10 else 5
                _restore_and_validate(snapshot)
                old_plan = np.stack([executed_actions[step]
                                     for step in range(boundary + 1, args.steps + 1)])
                fresh = _run_fresh(snapshot, h)
                same_rng = _run_same_rng(snapshot, h)
                cached = _run_cached(snapshot)
                repeats = _repeat_predictions(snapshot)
                seed_panel = _seed_panel(snapshot)
                branch_record = {
                    "horizon": h,
                    "cached_plan_control": cached,
                    "fresh_replan": fresh,
                    "same_rng_replan": same_rng,
                    "repeated_prediction": repeats,
                    "seed_panel": seed_panel,
                    "old_vs_fresh_plan_metrics": _plan_metrics(old_plan, fresh["chunks"][0]),
                    "old_plan_suffix_first_50": old_plan[:50],
                    "fresh_plan_first_50": fresh["chunks"][0],
                    "deterministic_repeat_first_50": repeats["deterministic_repeat"],
                    "varied_seed_first_50": repeats["varied_seed"],
                }
                if args.observation_diagnostic:
                    observation_component_records[str(boundary)] = _probe_observation_components(
                        snapshot, old_plan, same_rng["chunks"][0])
                    branch_record["observation_component_probes"] = observation_component_records[str(boundary)]
                if args.queue_diagnostic:
                    queue_branches = {}
                    for delay in args.queue_delays:
                        queued = _run_queue(snapshot, h, delay, old_plan)
                        queued["first_fresh_vs_same_rng"] = _action_identity(
                            queued["first_fresh_chunk"], same_rng["chunks"][0]
                        )
                        queued["same_rng_first_chunk_exact"] = queued["first_fresh_vs_same_rng"]["exact"]
                        queued["rng_reconstruction_invariant"] = (
                            queued["handoff_events"][0]["rng_before_generation"]
                            == same_rng["first_prediction_timing"]["rng_before_digest"]
                        )
                        queue_branches[str(delay)] = queued
                    branch_record["queue_diagnostic"] = queue_branches
                if args.rtc_guidance:
                    branch_record["rtc_no_previous_check"] = _rtc_no_previous_check(snapshot)
                    rtc = _run_rtc(snapshot, 10)
                    branch_record["rtc_guided_replan"] = rtc
                    branch_record["rtc_plan_metrics_vs_cached_suffix"] = _plan_metrics(
                        old_plan, rtc["chunks"][0]["processed"]
                    )
                branches[str(boundary)] = branch_record

        observation_selection = None
        observation_executions = {}
        if args.observation_diagnostic:
            # The gate was written before any probe prediction. It is a fixed
            # 0.05-rad first-10 RMS improvement at BOTH validated boundaries,
            # measured against the same-RNG current-all control. No seed or
            # candidate-specific tuning is allowed.
            meaningful_improvement_rad = 0.05
            candidate_names = [name for name, kind, _ in observation_probe_specs
                               if kind == "attribution_probe"]
            ranking = []
            for name in candidate_names:
                boundary_rows = []
                for boundary in (5, 10):
                    record = observation_component_records[str(boundary)]["probes"][name]
                    control = observation_component_records[str(boundary)]["probes"]["current-all"]
                    control_rms = control["action_metrics_vs_cached_h50_suffix"]["10"]["rms_rad"]
                    candidate_rms = record["action_metrics_vs_cached_h50_suffix"]["10"]["rms_rad"]
                    improvement = control_rms - candidate_rms
                    boundary_rows.append({
                        "boundary": boundary,
                        "control_first10_rms_rad": control_rms,
                        "candidate_first10_rms_rad": candidate_rms,
                        "improvement_rad": improvement,
                        "meaningful_threshold_rad": meaningful_improvement_rad,
                        "passes_meaningful_threshold": bool(improvement >= meaningful_improvement_rad),
                    })
                ranking.append({
                    "name": name,
                    "classification": "attribution_probe",
                    "deployable_controller": False,
                    "boundary_scores": boundary_rows,
                    "min_improvement_rad": min(x["improvement_rad"] for x in boundary_rows),
                    "mean_improvement_rad": float(np.mean([x["improvement_rad"] for x in boundary_rows])),
                    "passes_both_boundary_improvement_gate": all(
                        x["passes_meaningful_threshold"] for x in boundary_rows),
                })
            ranking.sort(key=lambda row: (-row["min_improvement_rad"],
                                          -row["mean_improvement_rad"], row["name"]))
            selected = [row["name"] for row in ranking
                        if row["passes_both_boundary_improvement_gate"]][:2]
            observation_selection = {
                "preregistered_meaningful_first10_rms_improvement_rad": meaningful_improvement_rad,
                "baseline": "same-RNG current-all fresh H10 at each boundary",
                "ranking_metric": "descending minimum first-10 RMS improvement across action-5 and action-10; mean improvement and name are tie-breakers",
                "seed_tuning": False,
                "ranking": ranking,
                "selected_at_most_two": selected,
                "execution_requested": bool(args.observation_execute_best),
            }
            if args.observation_execute_best:
                for name in selected:
                    observation_executions[name] = {}
                    for boundary, snapshot in sorted(causal_snapshots.items()):
                        h = 10 if boundary == 10 else 5
                        observation_executions[name][str(boundary)] = _run_observation_candidate(
                            snapshot, h, name)
                for name, by_boundary in observation_executions.items():
                    by_boundary["both_boundaries_settled_4_of_4"] = all(
                        by_boundary[str(boundary)]["settled"]["settled_4_of_4"]
                        for boundary in (5, 10))
            observation_selection["executions"] = observation_executions
            observation_selection["candidate_settled_4_of_4_both_boundaries"] = {
                name: bool(by_boundary.get("both_boundaries_settled_4_of_4", False))
                for name, by_boundary in observation_executions.items()
            }
            observation_selection["larger_validation_submitted"] = False
            observation_selection["larger_validation_reason"] = (
                "broad pose validation is prohibited in this diagnostic; candidate settling is recorded only"
            )

        causal_result = {
            "schema_version": 1,
            "garment": args.garment,
            "seed": args.seed,
            "match_pose": args.match_pose,
            "checkpoint": args.policy_path,
            "baseline_horizon": 50,
            "baseline_source": ("completed pilot executed H50 action stream"
                                 if causal_replay_actions is not None else "fresh H50 policy capture"),
            "branch_steps": args.causal_branch_steps,
            "original_prediction_chunks": original_chunks,
            "reconstructed_original_h50_chunk": reconstructed_original_chunk,
            "reconstructed_original_h50_chunk_raw": reconstructed_original_chunk_raw,
            "branches": branches,
            "boundary_evaluation": boundary_eval_result,
            "interpretation_guard": "Branch conclusions are valid only when cached-plan restoration errors remain below the stated tolerances.",
            "simulator_bitwise_determinism": False,
            "rng_audit": {
                "initial_torch_state": causal_initial_torch_rng,
                "initial_cuda_state": causal_initial_cuda_rng,
                "state_provenance": "captured immediately after seed_rngs() in this run; exact original pilot pre-call state was not persisted",
                "reconstructed_original_call": True,
                "seed_panel": causal_seed_panel,
            },
            "rtc_reference": {
                "enabled": bool(args.rtc_guidance),
                "implementation": "installed LeRobot RTCProcessor via SmolVLAPolicy.init_rtc_processor",
                "source_root": "/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome51-site/lerobot/policies/rtc",
                "configuration": {"execution_horizon": 10, "prefix_attention_schedule": "EXP", "max_guidance_weight": 10.0, "inference_delay": 0},
            },
            "queue_reference": {
                "enabled": bool(args.queue_diagnostic),
                "rtc_guidance_used": False if args.queue_diagnostic else None,
                "execution_horizon": 10,
                "delays": list(args.queue_delays) if args.queue_diagnostic else [],
                "handoff_rule": "execute exactly d actions from the unconsumed previous-plan remainder; discard fresh_chunk[:d]; execute fresh_chunk[d:10]; retain fresh_chunk[10:50] for the next H10 replan",
            },
            "observation_component_reference": {
                "enabled": bool(args.observation_diagnostic),
                "initial_observation": causal_initial_observation,
                "active_observation_keys": [
                    "observation.state", "observation.images.top_rgb",
                    "observation.images.left_rgb", "observation.images.right_rgb", "task",
                ],
                "task_unchanged": True,
                "rtc_guidance_used": False if args.observation_diagnostic else None,
                "probes_not_initially_executed": True,
                "attribution_probe_warning": "stale/hybrid observations are causal attribution probes, not deployable controllers",
                "feature_hook": observation_feature_status,
                "records": observation_component_records,
                "selection": observation_selection,
            },
            "prediction_timing_audit": prediction_timing_audit,
        }
        with open(args.causal_out, "w") as causal_file:
            json.dump(_jsonable(causal_result), causal_file, indent=2, allow_nan=False)
        log(f"CAUSAL_WRITTEN {args.causal_out}")
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
        "behavior_telemetry": behavior.finish(),
        "first_contact": None, "first_grasp": None, "failed_grasps": None,
        "contact_measurement_status": "force/contact ground truth unavailable; nearest gripper-link proximity is recorded separately",
        "first_gripper_proximity_under_3cm_action": earliest_near,
        "failure_category": failure, "failure_classification_basis": "geometry and gripper-link proximity; no unsupported grasp-region claims",
        "replanning_count": prediction_count[0],
        "inference_seconds": inference_times,
        "prediction_timing_audit": prediction_timing_audit,
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
        "hard_queue": ({"enabled": True, "delay": args.hard_queue_delay,
                        "execution_horizon": 10, "prediction_chunk_size": 50,
                        "handoff_rule": "retain previous remainder[:d], discard fresh[:d], execute fresh[d:10], retain fresh[10:50]",
                        "handoffs": hard_queue_handoffs}
                       if hard_queue_enabled else None),
        "first_success_checker": first_success_details, "terminal_checker": terminal_details,
        "cloth_motion": cloth_motion, "renderer": "storm",
        "render_integrity": render_integrity, "n_rendered": n_rendered,
        "gif_views": media["gifs"], "media": media,
        "trajectory": trajectory_metadata,
        "causal_replan": {"path": args.causal_out, "enabled": causal_enabled,
                          "branch_steps": args.causal_branch_steps,
                          "queue_diagnostic": bool(args.queue_diagnostic),
                          "prediction_timing_audit": prediction_timing_audit} if causal_enabled else None,
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
