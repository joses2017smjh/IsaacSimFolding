"""Passive measurements of already computed actions, images and simulated states.

No RNG, physics step, camera render, contact sensor, or action transformation.
Link-origin proximity is never represented as contact or verified acquisition.
"""
import json
import time
import numpy as np


class BehaviorTelemetry:
    def __init__(self, path, initial_points, horizon, dt, body_names):
        self.stream = open(path, "w", buffering=1)
        self.path, self.horizon, self.dt = str(path), horizon, dt
        self.initial_z = np.asarray(initial_points)[:, 2].copy()
        self.names = {side: list(names) for side, names in body_names.items()}
        self.indices = {side: [i for i, name in enumerate(names)
                              if name.lower() in ("gripper", "jaw")]
                        for side, names in self.names.items()}
        if any(not ids for ids in self.indices.values()):
            raise ValueError(f"Cannot identify gripper/jaw link origins: {self.names}")
        self.previous_action = None
        self.previous_images = None
        self.rows, self.boundaries, self.camera_changes = [], [], []
        self.wall_seconds = 0.0

    def record(self, step, action, joint, images, points, links, geometry):
        started = time.monotonic()
        action = np.asarray(action)
        boundary = (step - 1) % self.horizon == 0
        jump = None if self.previous_action is None else action - self.previous_action
        boundary_record = None
        if boundary and jump is not None:
            boundary_record = {"action": step, "l2_rad": float(np.linalg.norm(jump)),
                               "max_abs_rad": float(np.abs(jump).max()),
                               "delta_rad": jump.tolist()}
            self.boundaries.append(boundary_record)
        camera_change = None
        if boundary:
            small = {key: np.asarray(value)[::8, ::8].astype(np.int16)
                     for key, value in images.items() if key in ("top_rgb", "left_rgb", "right_rgb")}
            if self.previous_images is not None:
                camera_change = {key: {"mean_abs_rgb_0_255": float(np.abs(value - self.previous_images[key]).mean()),
                    "changed_pixel_fraction_over_8": float(np.any(np.abs(value - self.previous_images[key]) > 8, axis=2).mean())}
                    for key, value in small.items()}
                self.camera_changes.append({"action": step, "views": camera_change})
            self.previous_images = {key: value.copy() for key, value in small.items()}
        distances, closest = {}, {}
        for side, ids in self.indices.items():
            origins = np.asarray(links[side][0])[ids]
            matrix = np.linalg.norm(np.asarray(points)[None, :, :] - origins[:, None, :], axis=2)
            which, particle = np.unravel_index(matrix.argmin(), matrix.shape)
            distances[side] = float(matrix[which, particle])
            closest[side] = {"link": self.names[side][ids[which]], "particle_index": int(particle),
                             "link_origin_xyz_m": origins[which].tolist(),
                             "particle_xyz_m": np.asarray(points)[particle].tolist()}
        z_displacement = np.asarray(points)[:, 2] - self.initial_z
        row = {"action": step, "simulation_seconds": step * self.dt,
               "executed_action_rad": action.tolist(), "pre_action_joint_rad": np.asarray(joint).tolist(),
               "replan_boundary": boundary, "boundary_jump": boundary_record,
               "all_step_jump_l2_rad": None if jump is None else float(np.linalg.norm(jump)),
               "camera_change_since_previous_replan": camera_change,
               "gripper_link_origin_distance_m": distances, "nearest_particle": closest,
               "gripper_target_rad": {"left": float(action[5]), "right": float(action[11])},
               "gripper_target_delta_rad": None if jump is None else {"left": float(jump[5]), "right": float(jump[11])},
               "maximum_particle_lift_m": float(z_displacement.max()),
               "p95_particle_lift_m": float(np.percentile(z_displacement, 95)),
               "conditions_passed": geometry["conditions_passed"],
               "conditions_total": geometry["conditions_total"],
               "geometric_success": geometry["success"], "condition_details": geometry["details"]}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        self.previous_action = action.copy()
        self.wall_seconds += time.monotonic() - started

    def finish(self):
        self.stream.flush()
        self.stream.close()
        first_near = {}
        decreasing_near, decreasing_away = {}, {}
        for side in ("left", "right"):
            first_near[side] = next((r["action"] for r in self.rows if r["gripper_link_origin_distance_m"][side] < .03), None)
            decreasing_near[side] = next((r["action"] for r in self.rows if r["gripper_target_delta_rad"] is not None
                and r["gripper_target_delta_rad"][side] < -.02 and r["gripper_link_origin_distance_m"][side] < .03), None)
            decreasing_away[side] = next((r["action"] for r in self.rows if r["gripper_target_delta_rad"] is not None
                and r["gripper_target_delta_rad"][side] < -.02 and r["gripper_link_origin_distance_m"][side] > .05), None)
        return {"path": self.path, "steps": len(self.rows), "passive": True,
            "measurement_status": "link-origin proximity and command direction proxies only; no actual cloth contact sensor",
            "first_cloth_contact": None, "contacting_gripper": None,
            "first_close_command_near_cloth": None, "successful_cloth_acquisition": None,
            "failed_close_above_or_away": None, "cloth_retained": None, "cloth_release_or_drop": None,
            "unmeasured_reason": "No validated particle-cloth contact pairs or finger-surface aperture calibration; no control/physics changes introduced",
            "first_link_origin_proximity_under_3cm_action": first_near,
            "first_decreasing_command_near_cloth_proxy": decreasing_near,
            "first_decreasing_command_away_from_cloth_proxy": decreasing_away,
            "command_proxy_note": "Angle decrease is recorded, not asserted to be a calibrated close event or failed grasp",
            "minimum_gripper_link_origin_distance_m": {side: min(r["gripper_link_origin_distance_m"][side] for r in self.rows) for side in ("left", "right")},
            "maximum_particle_lift_m": max(r["maximum_particle_lift_m"] for r in self.rows),
            "best_policy_conditions": max(r["conditions_passed"] for r in self.rows),
            "replan_boundary_jumps": self.boundaries, "camera_changes_between_replans": self.camera_changes,
            "camera_change_note": "Stride-8 samples of actual input RGB; visual change alone does not prove useful feedback",
            "measurement_wall_seconds": self.wall_seconds}
