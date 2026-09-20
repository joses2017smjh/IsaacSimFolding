"""Small, simulator-independent checks for the isolated September rollout audit."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


def parse_pose(value: str) -> list[float] | None:
    if not value:
        return None
    values = [float(v) for v in value.replace(",", ":").split(":")]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("--match_pose needs six finite x:y:z:rx:ry:rz values")
    return values


def validate_arguments(args) -> list[float] | None:
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.sim_device):
        raise ValueError("PhysX particle cloth requires --sim_device cuda[:N]; CPU would freeze it")
    for name in ("steps", "gif_every", "capture_every", "capture_chunk", "trajectory_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.settle_steps < 0 or args.n_action_steps < 0:
        raise ValueError("--settle_steps and --n_action_steps must be nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if not math.isfinite(args.match_scale) or args.match_scale < 0:
        raise ValueError("--match_scale must be finite and nonnegative")
    if not args.policy_variant.strip():
        raise ValueError("--policy_variant must be a nonempty label")
    return parse_pose(args.match_pose)


def configure_action_queue(policy, requested: int) -> dict:
    """Change execution length, preserving the trained prediction chunk size.

    Installed LeRobot SmolVLA reset() allocates a deque using n_action_steps;
    select_action() fills it from the first n_action_steps of each prediction.
    Resetting after the override is essential to change the deque capacity too.
    """
    original = int(policy.config.n_action_steps)
    chunk_size = int(policy.config.chunk_size)
    effective = requested or original
    if requested < 0 or not 1 <= effective <= chunk_size:
        raise ValueError(f"n_action_steps={effective} must be between 1 and chunk_size={chunk_size}")
    policy.config.n_action_steps = effective
    policy.reset()
    return {"checkpoint_n_action_steps": original,
            "requested_n_action_steps": requested,
            "effective_n_action_steps": effective,
            "prediction_chunk_size": chunk_size}


def validate_camera_images(images: dict, width: int, height: int) -> dict:
    import numpy as np

    stats = {}
    for key in ("top_rgb", "left_rgb", "right_rgb"):
        array = np.asarray(images[key])
        if array.shape != (height, width, 3) or array.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 {(height, width, 3)}, got {array.dtype} {array.shape}")
        std = float(array.std())
        spatial_std = float(array.reshape(-1, 3).std(axis=0).max())
        if spatial_std == 0.0:
            raise ValueError(f"{key}: constant RGB frame; refusing a blank-camera policy rollout")
        stats[key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                      "std": std, "max_channel_spatial_std": spatial_std,
                      "min": int(array.min()), "max": int(array.max())}
    return stats


@dataclass
class EpisodeAudit:
    requested_steps: int
    completed_steps: int = 0
    first_success_step: int | None = None
    terminal_success: bool | None = None

    def record_step(self, checker_success: bool | None) -> bool:
        self.completed_steps += 1
        if self.completed_steps > self.requested_steps:
            raise ValueError("completed more policy steps than requested")
        return self.record_check(checker_success)

    def record_check(self, checker_success: bool | None) -> bool:
        if self.completed_steps < 1:
            raise ValueError("post-action checks require at least one completed action")
        first = bool(checker_success) and self.first_success_step is None
        if first:
            self.first_success_step = self.completed_steps
        return first

    def finish(self, terminal_success: bool) -> dict:
        if self.completed_steps != self.requested_steps:
            raise ValueError("an incomplete rollout cannot receive a completed episode verdict")
        self.terminal_success = bool(terminal_success)
        if self.terminal_success and self.first_success_step is None:
            self.first_success_step = self.completed_steps
        return {"success": self.first_success_step is not None,
                "success_criterion": "official checker true at least once after an action within the fixed budget",
                "first_success_step": self.first_success_step,
                "first_success_step_index": (self.first_success_step - 1
                                             if self.first_success_step is not None else None),
                "step_numbering": "first_success_step counts completed actions (1-based); frame step 0 is pre-action",
                "terminal_success": self.terminal_success,
                "requested_steps": self.requested_steps,
                "completed_steps": self.completed_steps,
                "episode_complete": True}


def gif_durations_ms(frame_steps: list[int], step_dt: float, gif_every: int) -> list[int]:
    """GIF stores centiseconds; round cumulative simulation timestamps.

    Keeping actual sampled step gaps includes an off-stride first-success
    frame without stretching the whole animation. The final frame is held for
    one normal capture interval. Quantization is at most 10 ms per interval
    except when a simulator runs faster than the format's 100 fps limit.
    """
    if not frame_steps or step_dt <= 0 or not math.isfinite(step_dt) or gif_every <= 0:
        raise ValueError("GIF timing requires frames, positive finite dt, and a positive stride")
    if frame_steps[0] < 0 or any(b <= a for a, b in zip(frame_steps, frame_steps[1:])):
        raise ValueError("GIF frame steps must be nonnegative and strictly increasing")
    edges = [*frame_steps, frame_steps[-1] + gif_every]
    ticks = [round(step * step_dt * 100) for step in edges]
    return [max(1, b - a) * 10 for a, b in zip(ticks, ticks[1:])]


def write_camera_artifacts(frames: dict, prefix: str, step_dt: float,
                           gif_every: int, first_success_step: int | None) -> dict:
    """Write the three uncropped observation streams and a left/top/right view."""
    from PIL import Image
    import numpy as np

    steps = sorted(frames)
    durations = gif_durations_ms(steps, step_dt, gif_every)
    views = {"top": "top_rgb", "left_wrist": "left_rgb", "right_wrist": "right_rgb"}
    gifs, snapshots = {}, {}

    def view_image(step, view):
        data = frames[step]
        array = (np.concatenate([data["left_rgb"], data["top_rgb"], data["right_rgb"]], axis=1)
                 if view == "triptych" else data[views[view]])
        return Image.fromarray(np.asarray(array, dtype=np.uint8))

    for name in (*views, "triptych"):
        gif_path = f"{prefix}_{name}.gif"
        images = [view_image(step, name) for step in steps]
        try:
            images[0].save(gif_path, save_all=True, append_images=images[1:],
                           duration=durations, loop=0, optimize=False, disposal=2)
        finally:
            for image in images:
                image.close()
        gifs[name] = gif_path

    events = {"final": steps[-1]}
    if first_success_step is not None:
        if first_success_step not in frames:
            raise ValueError("the exact first-success frame was not captured")
        events["first_success"] = first_success_step
    for event, step in events.items():
        snapshots[event] = {}
        for name in (*views, "triptych"):
            path = f"{prefix}_{event}_{name}.png"
            with view_image(step, name) as image:
                image.save(path)
            snapshots[event][name] = path
    return {"gifs": gifs, "snapshots": snapshots,
            "frame_steps": steps, "sampled_frames_per_view": len(steps),
            "frame_durations_ms": durations,
            "simulation_step_seconds": step_dt,
            "nominal_capture_interval_seconds": step_dt * gif_every,
            "final_frame_hold_seconds": step_dt * gif_every,
            "gif_duration_seconds": sum(durations) / 1000,
            "timing_note": "simulation timing quantized to GIF centiseconds; final frame held for one normal capture interval",
            "triptych_order": ["left_wrist", "top", "right_wrist"],
            "camera_keys": views}
