"""Restartable, duplicate-safe driver for the closed-loop campaign.

    driver.py --campaign ROOT tick     advance every stage that can advance, then
                                       schedule the next tick; safe to run any time
    driver.py --campaign ROOT status   print the resumable state

Why a driver. Queued stages must not wait on a chat prompt. Each tick inspects
Slurm and the filesystem, submits whatever is now unblocked, and submits one
CPU "chain tick" that depends on every job it is waiting for -- so the
campaign advances itself. A time-based watchdog tick recovers a chain whose
tick died.

Idempotence is the design. Every submission is keyed by a stage name in
ledger/driver_state.json and is never repeated while that key holds a live or
finished job. A tick run by hand, by the chain and by the watchdog at the same
moment serialises on a lock and cannot double-submit.

The site sbatch (/apps/slurm/bin/sbatch) is a wrapper that expands "$@"
UNQUOTED, word-splitting and glob-expanding every argument. The driver calls
the real binary and adds the one flag that wrapper exists to add.

Decisions this file does NOT make on evidence it has not seen: the choice of
the next iteration's single changed factor comes from an explicit committed
plan (plans/iterationK.json). Only if none exists after a grace period does
the preregistered default ladder below apply, so progression never stalls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

REAL_SBATCH = "/apps/slurm/current/bin/sbatch"
PYTHON = "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python"
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING",
          "RESIZING", "SUSPENDED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
          "STAGE_OUT"}
GRACE_MINUTES = 90
# Jobs submitted before the driver existed, mapped onto the stage keys the
# driver would have used. Without this a first tick would resubmit them.
ADOPT = {"collect:iteration1": "iter1.collect", "compile:iteration1": "iter1.compile",
         "train:iteration1": "iter1.train"}
# A tick's Slurm allocation is 20 minutes, so any lock older than this is
# certainly abandoned even when its holder cannot be checked directly.
LOCK_STALE_SECONDS = 25 * 60
DEV_LABEL = "baseline"


# ------------------------------------------------------------------ time
def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp(t: dt.datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


# ------------------------------------------------------------ pure logic
def summarize(states: list[str]) -> str:
    """Collapse the per-task Slurm states of one stage into one verdict."""
    if not states:
        return "unknown"
    norm = [s.split()[0].rstrip("+") for s in states]
    if any(s in ACTIVE for s in norm):
        return "active"
    if all(s == "COMPLETED" for s in norm):
        return "completed"
    return "failed"


def select_checkpoint(training: dict) -> tuple[dict | None, str]:
    """The preregistered within-iteration rule, plus amendment A1's double breach."""
    by_step = {int(c["step"]): c for c in training.get("checkpoints", [])}
    final = by_step.get(max(by_step)) if by_step else None
    fallback = by_step.get(200)
    if final is not None and final.get("heldout_gate"):
        return final, f"step_{final['step']:06d} passes the raster retention guard"
    if fallback is not None and fallback.get("heldout_gate"):
        return fallback, "final step breached the guard; preregistered fallback to step_000200"
    return None, "double breach: no guard-passing checkpoint (amendment A1); nothing is evaluated"


def score_rows(rows: list[dict], results: dict[str, dict | None]) -> dict:
    """Settled terminal success per horizon. Infrastructure failures are NOT
    policy failures: they are counted separately and excluded from the rate."""
    out: dict = {}
    for horizon in sorted({int(r.get("horizon", 10)) for r in rows}):
        these = [r for r in rows if int(r.get("horizon", 10)) == horizon]
        valid, settled, ever, conditions, invalid = 0, 0, 0, [], []
        for row in these:
            res = results.get(row["id"])
            if not res:
                invalid.append(row["id"])
                continue
            valid += 1
            settled += int(bool(res["terminal_success"]))
            ever += int(bool(res["ever_success"]))
            conditions.append(res["conditions_passed"])
        out[f"h{horizon}"] = {
            "rows": len(these), "valid": valid, "settled": settled, "ever": ever,
            "mean_conditions": (sum(conditions) / len(conditions)) if conditions else None,
            "invalid": invalid,
        }
    return out


def rank_key(candidate: dict) -> tuple:
    """Higher is better: H10 settled, then H50 settled, then H10 conditions,
    then LOWER raster held-out loss."""
    dev = candidate["dev"]
    return (dev["h10"]["settled"], dev.get("h50", {}).get("settled", 0),
            dev["h10"]["mean_conditions"] or 0.0, -(candidate.get("heldout_loss") or 0.0))


def eligibility(candidate: dict, baseline_dev: dict) -> list[str]:
    """Reasons a candidate may NOT be selected; empty means eligible."""
    reasons = []
    if not candidate.get("heldout_gate"):
        reasons.append("raster retention guard breached")
    if not candidate.get("reload_ok"):
        reasons.append("checkpoint did not reload cleanly")
    b50 = baseline_dev.get("h50", {}).get("settled")
    c50 = candidate["dev"].get("h50", {}).get("settled")
    if b50 is None or c50 is None:
        reasons.append("H50 comparison unavailable")
    elif c50 < b50:
        reasons.append(f"H50 regression: {c50} < matched baseline {b50}")
    for h in ("h10", "h50"):
        block = candidate["dev"].get(h, {})
        if block.get("valid", 0) < block.get("rows", 0):
            reasons.append(f"{h}: {block.get('rows', 0) - block.get('valid', 0)} rows infrastructure-invalid")
    return reasons


def default_plan(k: int, manifest: dict, best: dict | None, prior: list[dict]) -> dict:
    """Preregistered fallback ladder, used ONLY if no explicit plan is committed
    within the grace period. One major factor changes per iteration."""
    # 1000 per iteration keeps default-ladder seeds clear of every preregistered
    # block: collection 4200s, expansion 4300s, development 200s, test 9000s.
    shift = 1000 * (k - 1)

    def rows(key: str, suffix: str) -> list[dict]:
        out = []
        for r in manifest[key]:
            r = dict(r)
            r["id"] = f"{r['id']}_{suffix}"
            r["seed"] = int(r["seed"]) + shift
            out.append(r)
        return out

    baseline = manifest["baseline_checkpoint"]["path"]
    training = {key: manifest["training"][key] for key in
                ("steps", "batch_size", "rollout_fraction", "lr", "checkpoint_every", "unfreeze", "seed")}
    if best is not None:
        return {
            "iteration": k, "source": "default", "factor_changed": "collection_policy",
            "justification": ("an eligible improved candidate exists, so the next iteration is the "
                              "canonical on-policy step: collect from it and train from it"),
            "collection_checkpoint": best["checkpoint"], "init_checkpoint": best["checkpoint"],
            "collection": rows("collection", f"it{k}"),
            "collection_expansion": rows("collection_expansion", f"it{k}x"),
            "training": training,
        }
    if k == 2:
        return {
            "iteration": k, "source": "default", "factor_changed": "collection_coverage",
            "justification": ("no eligible improvement; double the collection from 8 to 16 episodes "
                              "(the preregistered collection and expansion rows, fresh seeds)"),
            "collection_checkpoint": baseline, "init_checkpoint": baseline,
            "collection": rows("collection", f"it{k}") + rows("collection_expansion", f"it{k}c"),
            "collection_expansion": [],
            "training": training,
        }
    training = dict(training, rollout_fraction=0.75)
    return {
        "iteration": k, "source": "default", "factor_changed": "replay_balance",
        "justification": ("no eligible improvement after a coverage change; keep 16-episode coverage "
                          "and raise the rollout share of each batch from 0.5 to 0.75"),
        "collection_checkpoint": baseline, "init_checkpoint": baseline,
        "collection": rows("collection", f"it{k}") + rows("collection_expansion", f"it{k}c"),
        "collection_expansion": [],
        "training": training,
    }


# ----------------------------------------------------------------- Slurm
def relative_begin(when: dt.datetime) -> str:
    """Slurm's --begin reads an absolute timestamp in the CLUSTER's local time
    (UTC-7 here), so a UTC string lands seven hours late. The relative form has
    no time zone to get wrong."""
    return f"now+{max(0, int((when - now()).total_seconds()))}"


def clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}


def sbatch(args: list[str]) -> str:
    binary = REAL_SBATCH if Path(REAL_SBATCH).exists() else "sbatch"
    cmd = [binary, "--parsable", f"--mail-user={os.environ.get('USER', 'sanchej7')}", *args]
    out = subprocess.run(cmd, env=clean_env(), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed ({out.returncode}): {out.stderr.strip()} :: {cmd}")
    return out.stdout.strip().split(";")[0]


def scancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], env=clean_env(), capture_output=True)


def job_states(job_ids: list[str]) -> dict[str, list[str]]:
    if not job_ids:
        return {}
    out = subprocess.run(["sacct", "-n", "-X", "-P", "--format=JobID,State", "-j", ",".join(job_ids)],
                         capture_output=True, text=True).stdout
    states: dict[str, list[str]] = {j: [] for j in job_ids}
    for line in out.splitlines():
        if "|" not in line:
            continue
        jid, state = line.split("|", 1)
        states.setdefault(jid.split("_")[0], []).append(state)
    return states


def gpu_hours(job_ids: list[str]) -> float:
    if not job_ids:
        return 0.0
    out = subprocess.run(["sacct", "-n", "-X", "-P", "--format=JobID,ElapsedRaw,AllocTRES",
                          "-j", ",".join(job_ids)], capture_output=True, text=True).stdout
    total = 0.0
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or "gres/gpu" not in parts[2]:
            continue
        gpus = 1
        for tres in parts[2].split(","):
            if tres.startswith("gres/gpu="):
                gpus = int(tres.split("=")[1])
        total += gpus * int(parts[1] or 0) / 3600.0
    return total


# ----------------------------------------------------------------- driver
class Driver:
    def __init__(self, root: Path, dry: bool = False):
        self.root = root.resolve()
        self.repo = self.root.parents[1]
        self.dry = dry
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.amendment = json.loads((self.root / "amendments/A1-autonomous-mandate.json").read_text())
        self.caps = self.amendment["budget_caps"]
        self.state_path = self.root / "ledger/driver_state.json"
        self.ledger_path = self.root / "ledger/slurm-jobs.json"
        self.state = self._load()
        self.adopt()
        self.waiting: list[str] = []
        self.begin_at: dt.datetime | None = None

    # ---- persistence
    def _load(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {"schema": 1, "created_utc": stamp(), "stages": {}, "iterations": {},
                "baseline_dev": None, "best": None, "final": {}, "ticks": {},
                "done": False, "stop_reason": None, "log": []}

    def adopt(self) -> None:
        """Rebuild any stage the state lost from the ledger.

        The ledger is written the instant sbatch returns; the state is saved
        later. A tick killed in between (tick 21400711 was, by its own scancel)
        leaves a live job the state has never heard of, and the next tick would
        resubmit it. Adopting from the ledger closes that window.
        """
        for job in self.ledger()["jobs"]:
            phase = job.get("phase")
            key = ADOPT.get(phase) or (phase if job.get("submitted_by") == "driver" else None)
            if key and key not in self.state["stages"]:
                self.state["stages"][key] = {"job_ids": [job["job_id"]], "status": "submitted",
                                             "submitted_utc": job.get("submitted_utc"),
                                             "adopted_from_ledger": True}

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        os.replace(tmp, self.state_path)

    def event(self, message: str, **extra) -> None:
        self.state["log"].append({"utc": stamp(), "message": message, **extra})
        print(f"[driver {stamp()}] {message}", flush=True)

    def ledger(self) -> dict:
        return json.loads(self.ledger_path.read_text())

    def record(self, job: str, stage: str, script: str, gpu_tasks: int, **extra) -> None:
        data = self.ledger()
        data["jobs"].append({"job_id": job, "phase": stage, "script": script,
                             "gpu_tasks": gpu_tasks, "submitted_by": "driver",
                             "manifest_commit": self.manifest["git"]["commit"],
                             "submitted_utc": stamp(), "state": "submitted", **extra})
        self.ledger_path.write_text(json.dumps(data, indent=2) + "\n")

    # ---- budget
    def gpu_tasks_used(self) -> int:
        legacy = {"smoke": 1, "collect:iteration1": 8, "train:iteration1": 1}
        total = 0
        for job in self.ledger()["jobs"]:
            if "gpu_tasks" in job:
                total += int(job["gpu_tasks"])
            else:
                total += legacy.get(job["phase"], 0)
        return total

    def can_spend(self, tasks: int, *, final: bool = False) -> bool:
        reserve = 0 if final else int(self.caps["reserved_for_final_evaluation_gpu_tasks"])
        return self.gpu_tasks_used() + tasks + reserve <= int(self.caps["max_gpu_tasks_total"])

    def within_gpu_hours(self) -> bool:
        ids = [j["job_id"] for j in self.ledger()["jobs"]]
        return gpu_hours(ids) <= float(self.caps["max_gpu_hours_total"])

    def storage_ok(self) -> tuple[bool, str]:
        usage = subprocess.run(["du", "-sk", str(self.root)], capture_output=True, text=True).stdout
        used_gb = int(usage.split()[0]) / 1024 / 1024 if usage else 0.0
        st = os.statvfs(self.root)
        free_gb = st.f_bavail * st.f_frsize / 1024 ** 3
        cap = self.caps["storage"]
        if used_gb > cap["campaign_directory_max_gb"]:
            return False, f"campaign directory {used_gb:.1f} GB exceeds cap {cap['campaign_directory_max_gb']} GB"
        if free_gb < cap["stop_new_collection_if_filesystem_free_below_gb"]:
            return False, f"filesystem free {free_gb:.0f} GB below floor"
        return True, f"campaign {used_gb:.1f} GB, filesystem free {free_gb:.0f} GB"

    # ---- stages
    def stage(self, key: str) -> dict | None:
        return self.state["stages"].get(key)

    def stage_status(self, key: str) -> str:
        st = self.stage(key)
        if st is None:
            return "absent"
        if st.get("status") in ("completed", "failed", "skipped"):
            return st["status"]
        states = job_states(st["job_ids"])
        flat = [s for j in st["job_ids"] for s in states.get(j, [])]
        verdict = summarize(flat)
        if verdict == "unknown":
            verdict = "active"      # just submitted; sacct has not caught up
        if verdict in ("completed", "failed"):
            st["status"] = verdict
            st["finished_utc"] = stamp()
        else:
            self.waiting.extend(st["job_ids"])
        return verdict

    def submit(self, key: str, script: str, args: list[str], *, gpu_tasks: int,
               array: str | None = None, final: bool = False, **extra) -> str | None:
        if self.stage(key) is not None:
            return None                                   # never resubmit a key
        if gpu_tasks and not self.can_spend(gpu_tasks, final=final):
            self.event(f"budget refuses {key}: {gpu_tasks} GPU tasks")
            self.state["stages"][key] = {"job_ids": [], "status": "skipped",
                                         "reason": "gpu task budget"}
            return None
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        argv = [f"--chdir={self.root}",
                f"--output={logs}/{key.replace('/', '_')}-%A_%a.out" if array
                else f"--output={logs}/{key.replace('/', '_')}-%j.out"]
        if array:
            argv.append(f"--array={array}%{self.caps['gpu_array_concurrency']}" if gpu_tasks else f"--array={array}")
        argv += [str(self.root / "slurm" / script), *args]
        if self.dry:
            job = f"DRY-{key}"
        else:
            job = sbatch(argv)
            self.record(job, key, f"slurm/{script}", gpu_tasks, array=array, **extra)
        self.state["stages"][key] = {"job_ids": [job], "submitted_utc": stamp(),
                                     "status": "submitted", "gpu_tasks": gpu_tasks, **extra}
        self.waiting.append(job)
        self.event(f"submitted {key} -> {job}", gpu_tasks=gpu_tasks)
        if not self.dry:
            self.save()
        return job

    # ---- rollout bookkeeping
    def read_result(self, directory: Path) -> dict | None:
        status_path, result_path = directory / "status.json", directory / "rollout.json"
        if not status_path.is_file():
            return None
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed" or not result_path.is_file():
            return None
        result = json.loads(result_path.read_text())
        checker = result.get("terminal_checker", {})
        return {"terminal_success": bool(result.get("terminal_success")),
                "ever_success": bool(result.get("success")),
                "conditions_passed": checker.get("conditions_passed"),
                "conditions_total": checker.get("conditions_total"),
                "horizon": result.get("effective_n_action_steps")}

    def failed_rows(self, base: Path, rows: list[dict]) -> list[int]:
        return [i for i, row in enumerate(rows) if self.read_result(base / row["id"]) is None]

    def move_aside(self, base: Path, rows: list[dict], indices: list[int], tag: str) -> None:
        for i in indices:
            src = base / rows[i]["id"]
            if src.exists():
                dst = self.root / "attempts" / base.relative_to(self.root) / f"{rows[i]['id']}.{tag}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    def rollout_stage(self, key: str, script: str, args: list[str], base: Path,
                      rows: list[dict]) -> str:
        """Submit, wait, and retry infrastructure-failed rows once. Returns
        'wait', 'done' or 'incomplete'."""
        n = len(rows)
        if self.stage(key) is None:
            self.submit(key, script, args, gpu_tasks=n, array=f"0-{n - 1}")
            if self.stage(key).get("status") == "skipped":
                return "skipped"
            return "wait"
        if self.stage(key).get("status") == "skipped":
            return "skipped"
        if self.stage_status(key) == "active":
            return "wait"
        bad = self.failed_rows(base, rows)
        if not bad:
            return "done"
        retry = f"{key}.retry1"
        if self.stage(retry) is None:
            self.move_aside(base, rows, bad, "attempt1")
            self.event(f"{key}: retrying infrastructure-failed rows {bad}")
            self.submit(retry, script, args, gpu_tasks=len(bad), array=",".join(map(str, bad)))
            return "wait"
        if self.stage_status(retry) == "active":
            return "wait"
        still = self.failed_rows(base, rows)
        return "done" if not still else "incomplete"

    # ---- plans
    def committed(self, path: Path) -> bool:
        rel = str(path.relative_to(self.repo))
        tracked = subprocess.run(["git", "-C", str(self.repo), "ls-files", "--error-unmatch", rel],
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(self.repo), "diff", "--quiet", "HEAD", "--", rel],
                               capture_output=True).returncode == 0
        return tracked and clean

    def resolved_plan(self, k: int) -> dict | None:
        resolved = self.root / "plans" / f"iteration{k}.resolved.json"
        if resolved.is_file():
            return json.loads(resolved.read_text())
        if k == 1:
            plan = {"iteration": 1, "source": "manifest", "factor_changed": None,
                    "collection_checkpoint": self.manifest["baseline_checkpoint"]["path"],
                    "init_checkpoint": self.manifest["baseline_checkpoint"]["path"],
                    "collection": self.manifest["collection"],
                    "collection_expansion": self.manifest["collection_expansion"],
                    "training": {key: self.manifest["training"][key] for key in
                                 ("steps", "batch_size", "rollout_fraction", "lr",
                                  "checkpoint_every", "unfreeze", "seed")}}
        else:
            explicit = self.root / "plans" / f"iteration{k}.json"
            prev = self.state["iterations"].get(str(k - 1), {})
            if explicit.is_file() and self.committed(explicit):
                plan = json.loads(explicit.read_text())
                plan["source"] = "explicit"
            elif explicit.is_file():
                self.event(f"plan {explicit.name} exists but is not committed; waiting")
                return None
            else:
                ready = prev.get("concluded_utc")
                if ready is None:
                    return None
                due = parse(ready) + dt.timedelta(minutes=GRACE_MINUTES)
                if now() < due:
                    self.begin_at = due if self.begin_at is None else min(self.begin_at, due)
                    return None
                prior = [self.state["iterations"][str(i)] for i in range(1, k)]
                plan = default_plan(k, self.manifest, self.state.get("best"), prior)
                self.event(f"no explicit plan for iteration {k} after {GRACE_MINUTES} min grace; "
                           f"applying default ladder: {plan['factor_changed']}")
        plan["resolved_utc"] = stamp()
        if self.dry:
            # A dry run must leave no trace: a resolved plan written here would
            # be picked up by the next real tick without its hash recorded.
            return plan
        resolved.parent.mkdir(exist_ok=True)
        resolved.write_text(json.dumps(plan, indent=2) + "\n")
        self.state["iterations"].setdefault(str(k), {})["plan_sha256"] = \
            hashlib.sha256(resolved.read_bytes()).hexdigest()
        return plan

    # ---- one iteration
    def iteration(self, k: int) -> str:
        """Advance iteration k. Returns 'wait', 'concluded' or 'blocked'."""
        it = self.state["iterations"].setdefault(str(k), {"phase": "planning"})
        if it.get("concluded_utc"):
            return "concluded"
        plan = self.resolved_plan(k)
        if plan is None:
            return "wait"
        tag = f"iter{k}"
        group = "iteration1" if k == 1 else f"iteration{k}"
        base = self.root / "rollouts" / group
        resolved = self.root / "plans" / f"iteration{k}.resolved.json"
        C = str(self.root)

        # -- collection
        if k == 1:
            coll = self.rollout_stage(f"{tag}.collect", "collect.sbatch", [C], base, plan["collection"])
        else:
            if self.stage(f"{tag}.collect") is None:
                if now() > parse(self.caps["no_new_iteration_after_utc"]):
                    return self.conclude(k, "not started: past no_new_iteration_after_utc")
                ok, why = self.storage_ok()
                if not ok:
                    return self.conclude(k, f"not started: {why}")
                if not self.can_spend(len(plan["collection"]) + 1 + 16):
                    return self.conclude(k, "not started: GPU task budget")
            coll = self.rollout_stage(f"{tag}.collect", "collect_plan.sbatch",
                                      [C, str(resolved), "collection", plan["collection_checkpoint"]],
                                      base, plan["collection"])
        if coll == "wait":
            return "wait"
        if coll == "skipped":
            return self.conclude(k, "collection skipped: GPU task budget")
        if coll == "incomplete":
            return self.conclude(k, "collection rows failed twice on infrastructure")

        # -- compile, with the preregistered expansion on a degenerate signal
        gate_path = self.root / "audit" / f"collection_gate_{group}.json"
        scope = group
        plan_arg = [] if k == 1 else [str(resolved)]
        if self.stage(f"{tag}.compile") is None:
            self.submit(f"{tag}.compile", "compile.sbatch", [C, group, *plan_arg], gpu_tasks=0)
            return "wait"
        if self.stage_status(f"{tag}.compile") == "active":
            return "wait"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
        if gate is None:
            return self.conclude(k, "compile produced no gate report")
        if not gate["passed"]:
            exp_rows = plan.get("collection_expansion") or []
            if not exp_rows:
                it["degenerate"] = gate["unmet_requirements"]
                return self.conclude(k, f"degenerate signal, no expansion available: {gate['unmet_requirements']}")
            exp_group = "expansion" if k == 1 else f"iteration{k}-expansion"
            exp_base = self.root / "rollouts" / exp_group
            if k == 1:
                ex = self.rollout_stage(f"{tag}.expand", "expand.sbatch", [C], exp_base, exp_rows)
            else:
                ex = self.rollout_stage(f"{tag}.expand", "collect_plan.sbatch",
                                        [C, str(resolved), "collection_expansion",
                                         plan["collection_checkpoint"]], exp_base, exp_rows)
            if ex != "done":
                return "wait" if ex == "wait" else self.conclude(k, "expansion rows failed twice")
            scope = "expanded" if k == 1 else f"iteration{k}-expanded"
            gate_path = self.root / "audit" / f"collection_gate_{scope}.json"
            if self.stage(f"{tag}.compile_expanded") is None:
                self.submit(f"{tag}.compile_expanded", "compile.sbatch", [C, scope, *plan_arg],
                            gpu_tasks=0)
                return "wait"
            if self.stage_status(f"{tag}.compile_expanded") == "active":
                return "wait"
            gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
            if gate is None or not gate["passed"]:
                unmet = gate["unmet_requirements"] if gate else ["compile failed"]
                it["degenerate"] = unmet
                return self.conclude(k, f"degenerate signal after preregistered expansion: {unmet}")
        it["gate"] = {"scope": scope, "rewards": gate["terminal_rewards"],
                      "episode_ess": gate["episode_weight_summary"]["ess"],
                      "unmet": gate["unmet_requirements"]}

        # -- train
        train_key = f"{tag}.train"
        train_dir = self.root / "training" / scope
        if self.stage(train_key) is None:
            args = [C, scope] if k == 1 else [C, scope, str(resolved)]
            self.submit(train_key, "train.sbatch", args, gpu_tasks=1)
            return "wait"
        status = self.stage_status(train_key)
        if status == "active":
            return "wait"
        training_json = train_dir / "training.json"
        if not training_json.is_file():
            retry = f"{train_key}.retry1"
            if self.stage(retry) is None:
                if train_dir.exists():
                    dst = self.root / "attempts" / "training" / f"{scope}.attempt1"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(train_dir), str(dst))
                args = [C, scope] if k == 1 else [C, scope, str(resolved)]
                self.submit(retry, "train.sbatch", args, gpu_tasks=1)
                return "wait"
            if self.stage_status(retry) == "active":
                return "wait"
            if not training_json.is_file():
                return self.conclude(k, "training failed twice")
        training = json.loads(training_json.read_text())
        chosen, why = select_checkpoint(training)
        it["selection"] = why
        if chosen is None:
            it["candidate"] = None
            return self.conclude(k, why)
        label = f"iter{k}-step{int(chosen['step']):06d}"
        cand = it.setdefault("candidate", {})
        cand.update({"label": label, "checkpoint": chosen["path"], "step": int(chosen["step"]),
                     "heldout_gate": bool(chosen["heldout_gate"]),
                     "heldout_loss": chosen.get("heldout_loss"),
                     "heldout_loss_baseline": chosen.get("heldout_loss_baseline")})

        # -- reload exactly as the evaluator does
        reload_out = self.root / "audit" / f"reload_{label}.json"
        dataset = self.root / "datasets" / f"{scope}.npz"
        reload_args = [C, chosen["path"], str(dataset), str(reload_out)]
        if self.stage(f"{tag}.reload") is None:
            self.submit(f"{tag}.reload", "reload.sbatch", reload_args, gpu_tasks=1)
            return "wait"
        if self.stage_status(f"{tag}.reload") == "active":
            return "wait"
        if not reload_out.is_file():
            # No output means the job itself failed (node, CUDA, I/O) -- an
            # infrastructure fault, not evidence about the checkpoint. A
            # checkpoint that genuinely fails writes finite/config verdicts.
            retry = f"{tag}.reload.retry1"
            if self.stage(retry) is None:
                self.submit(retry, "reload.sbatch", reload_args, gpu_tasks=1)
                return "wait"
            if self.stage_status(retry) == "active":
                return "wait"
        reload = json.loads(reload_out.read_text()) if reload_out.is_file() else {}
        cand["reload_ok"] = bool(reload.get("finite") and reload.get("config_matches_baseline"))
        if not cand["reload_ok"]:
            return self.conclude(k, "candidate failed the evaluator-style reload")

        # -- development evaluation, matched to the baseline's 16 rows
        rows = self.manifest["benchmark"]
        ev = self.rollout_stage(f"{tag}.evaluate", "benchmark.sbatch", [C, chosen["path"], label],
                                self.root / "evaluation" / label / "benchmark", rows)
        if ev == "wait":
            return "wait"
        if ev == "skipped":
            return self.conclude(k, "evaluation skipped: GPU task budget; candidate unscored")
        results = {r["id"]: self.read_result(self.root / "evaluation" / label / "benchmark" / r["id"])
                   for r in rows}
        cand["dev"] = score_rows(rows, results)
        return self.conclude(k, "evaluated")

    def conclude(self, k: int, reason: str) -> str:
        it = self.state["iterations"].setdefault(str(k), {})
        it["concluded_utc"] = stamp()
        it["outcome"] = reason
        self.event(f"iteration {k} concluded: {reason}")
        return "concluded"

    # ---- baseline and selection
    def baseline_dev(self) -> dict | None:
        if self.state.get("baseline_dev"):
            return self.state["baseline_dev"]
        rows = self.manifest["benchmark"]
        base = self.root / "evaluation" / DEV_LABEL / "benchmark"
        st = self.rollout_stage("baseline.dev", "benchmark.sbatch",
                                [str(self.root), self.manifest["baseline_checkpoint"]["path"], DEV_LABEL],
                                base, rows)
        if st == "wait":
            return None
        results = {r["id"]: self.read_result(base / r["id"]) for r in rows}
        self.state["baseline_dev"] = score_rows(rows, results)
        self.event("matched baseline development measured",
                   h10=self.state["baseline_dev"]["h10"]["settled"],
                   h50=self.state["baseline_dev"]["h50"]["settled"])
        return self.state["baseline_dev"]

    def update_best(self) -> None:
        base = self.state.get("baseline_dev")
        if not base:
            return
        best = None
        for k, it in sorted(self.state["iterations"].items()):
            cand = it.get("candidate")
            if not cand or "dev" not in cand:
                continue
            cand["ineligible_because"] = eligibility(cand, base)
            improves = cand["dev"]["h10"]["settled"] > base["h10"]["settled"]
            cand["improves_h10_over_baseline"] = improves
            if cand["ineligible_because"] or not improves:
                continue
            if best is None or rank_key(cand) > rank_key(best):
                best = dict(cand, iteration=int(k))
        self.state["best"] = best

    # ---- final
    def finalize(self) -> str:
        fin = self.state["final"]
        best = self.state.get("best")
        chosen = best["checkpoint"] if best else self.manifest["baseline_checkpoint"]["path"]
        label = best["label"] if best else DEV_LABEL
        fin.setdefault("selected", {"label": label, "checkpoint": chosen,
                                    "reason": "best eligible candidate" if best else
                                    "no candidate qualified; the untouched baseline is retained"})
        rows = self.manifest["frozen_test"]
        C = str(self.root)
        pending = False
        outcome = {"baseline": self.rollout_stage(
            "final.test.baseline", "test.sbatch",
            [C, self.manifest["baseline_checkpoint"]["path"], DEV_LABEL],
            self.root / "evaluation" / DEV_LABEL / "frozen_test", rows)}
        pending |= outcome["baseline"] == "wait"
        if best:
            outcome["candidate"] = self.rollout_stage(
                "final.test.candidate", "test.sbatch", [C, chosen, label],
                self.root / "evaluation" / label / "frozen_test", rows)
            pending |= outcome["candidate"] == "wait"
            if self.stage("final.boundary") is None and self.can_spend(1, final=True):
                self.submit("final.boundary", "boundary.sbatch", [C, chosen, label],
                            gpu_tasks=1, final=True)
                pending = True
            elif self.stage("final.boundary") and self.stage_status("final.boundary") == "active":
                pending = True
        if pending:
            return "wait"
        for who, lab in (("baseline", DEV_LABEL), ("candidate", label if best else None)):
            if lab is None:
                continue
            if outcome.get(who) == "skipped":
                # A test that never ran has no score. Writing 0/0 here would let
                # the final report state a measurement that does not exist.
                fin[f"test_{who}"] = {"skipped": "GPU task budget"}
                continue
            results = {r["id"]: self.read_result(self.root / "evaluation" / lab / "frozen_test" / r["id"])
                       for r in rows}
            fin[f"test_{who}"] = score_rows(rows, results)
        fin["completed_utc"] = stamp()
        self.state["done"] = True
        self.write_final_report()
        self.event("campaign complete; final report written")
        return "done"

    def write_final_report(self) -> None:
        base = self.state.get("baseline_dev") or {}
        fin = self.state["final"]
        lines = ["# Final report — closed-loop training v2", "",
                 f"Generated {stamp()} by scripts/driver.py from ledger/driver_state.json.", "",
                 "## Development benchmark (8 poses; development data, used for selection)", "",
                 "| checkpoint | H10 settled | H50 settled | H10 mean conditions | eligible | notes |",
                 "|---|---|---|---|---|---|"]
        if base:
            lines.append(f"| baseline (matched, this campaign) | {base['h10']['settled']}/{base['h10']['valid']} | "
                         f"{base['h50']['settled']}/{base['h50']['valid']} | {base['h10']['mean_conditions']} | — | |")
        for k, it in sorted(self.state["iterations"].items()):
            cand = it.get("candidate") or {}
            dev = cand.get("dev")
            if not dev:
                lines.append(f"| iteration {k} | — | — | — | no | {it.get('outcome')} |")
                continue
            why = "; ".join(cand.get("ineligible_because") or []) or "eligible"
            lines.append(f"| {cand['label']} | {dev['h10']['settled']}/{dev['h10']['valid']} | "
                         f"{dev['h50']['settled']}/{dev['h50']['valid']} | {dev['h10']['mean_conditions']} | "
                         f"{'yes' if not cand.get('ineligible_because') else 'no'} | {why} |")
        sel = fin.get("selected", {})
        lines += ["", "## Selected checkpoint", "", f"`{sel.get('label')}` — {sel.get('reason')}", "",
                  f"`{sel.get('checkpoint')}`", "",
                  "## Frozen test set (reserved; excluded from all training and selection)", ""]
        for who in ("baseline", "candidate"):
            t = fin.get(f"test_{who}")
            if t and "skipped" in t:
                lines.append(f"- {who}: **not run** ({t['skipped']})")
            elif t:
                lines.append(f"- {who}: H10 settled {t['h10']['settled']}/{t['h10']['valid']} "
                             f"(mean conditions {t['h10']['mean_conditions']})")
        target = self.amendment["changes"]["targets"]["now"]
        best = self.state.get("best")
        h10 = best["dev"]["h10"]["settled"] if best else (base.get("h10", {}).get("settled") if base else None)
        lines += ["", "## Target", "", f"- primary: {target['primary']} — "
                  f"**{'MET' if h10 is not None and h10 >= 6 else 'NOT MET'}**",
                  f"- stretch: {target['stretch']} — **{'MET' if h10 == 8 else 'NOT MET'}**", ""]
        (self.root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")

    # ---- ticks and status
    def schedule_ticks(self) -> None:
        ticks = self.state["ticks"]
        waiting = sorted(set(self.waiting))
        me = os.environ.get("SLURM_JOB_ID")
        chain = ticks.get("chain")
        # Only a PENDING chain tick is "queued". A running one is this very
        # process: tick 21400711 compared itself against its new dependency
        # set, called scancel on its own job id, and died before saving.
        pending = chain and chain != me and \
            [s for v in job_states([chain]).values() for s in v][:1] == ["PENDING"]
        if pending and ticks.get("chain_deps") == waiting and \
                ticks.get("chain_begin") == (stamp(self.begin_at) if self.begin_at else None):
            pass                                       # identical tick already queued
        else:
            if pending:
                scancel(chain)
            chain = None
        if chain is None and (waiting or self.begin_at) and not self.state["done"]:
            argv = [f"--chdir={self.root}", f"--output={self.root}/ledger/ticks/tick-%j.out"]
            if waiting:
                # "?" is Slurm's OR: the tick runs when ANY waited job ends, so a
                # short job (a reload) is acted on without waiting for a long
                # array (a benchmark) that happens to be running alongside it.
                argv.append("--dependency=" + "?".join(f"afterany:{j}" for j in waiting))
            if self.begin_at:
                argv.append(f"--begin={relative_begin(self.begin_at)}")
            (self.root / "ledger/ticks").mkdir(parents=True, exist_ok=True)
            argv += [str(self.root / "slurm/tick.sbatch"), str(self.root)]
            if not self.dry:
                ticks["chain"] = sbatch(argv)
                ticks["chain_deps"] = waiting
                ticks["chain_begin"] = stamp(self.begin_at) if self.begin_at else None
        watchdog = ticks.get("watchdog")
        alive = watchdog and watchdog != me and \
            summarize([s for v in job_states([watchdog]).values() for s in v]) == "active"
        if not alive and not self.state["done"] and not self.dry:
            ticks["watchdog"] = sbatch([f"--chdir={self.root}",
                                        f"--output={self.root}/ledger/ticks/watchdog-%j.out",
                                        f"--begin={relative_begin(now() + dt.timedelta(hours=2))}",
                                        str(self.root / "slurm/tick.sbatch"), str(self.root)])

    def update_status(self, active: str, latest: str, blocker: str, nxt: str) -> None:
        path = self.root / "STATUS.md"
        text = path.read_text()
        begin, end = "<!-- driver:status:begin -->", "<!-- driver:status:end -->"
        if begin not in text or end not in text:
            return
        block = (f"{begin}\n| | |\n|---|---|\n| **Active job** | {active} |\n"
                 f"| **Latest result** | {latest} |\n| **Blocker** | {blocker} |\n"
                 f"| **Next milestone** | {nxt} |\n\n_Updated {stamp()} by scripts/driver.py._\n{end}")
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        path.write_text(head + block + tail)

    def describe(self) -> tuple[str, str, str, str]:
        live = [f"`{','.join(v['job_ids'])}` {k}" for k, v in self.state["stages"].items()
                if v.get("status") == "submitted"]
        active = ", ".join(live) if live else ("none — campaign complete" if self.state["done"] else "none")
        base = self.state.get("baseline_dev")
        best = self.state.get("best")
        parts = []
        if base:
            parts.append(f"matched baseline dev H10 {base['h10']['settled']}/8, H50 {base['h50']['settled']}/8")
        for k, it in sorted(self.state["iterations"].items()):
            cand = it.get("candidate") or {}
            if cand.get("dev"):
                parts.append(f"iter {k} `{cand['label']}` H10 {cand['dev']['h10']['settled']}/8, "
                             f"H50 {cand['dev']['h50']['settled']}/8")
            elif it.get("outcome"):
                parts.append(f"iter {k}: {it['outcome']}")
        latest = "; ".join(parts) or "iteration 1 in progress"
        blocker = self.state.get("stop_reason") or "none"
        if self.state["done"]:
            nxt = "none — see FINAL_REPORT.md"
        elif best:
            nxt = f"best so far `{best['label']}`; next: continue loop / final frozen test"
        else:
            nxt = "advance the current iteration; final frozen test after the loop"
        return active, latest, blocker, nxt

    # ---- main loop
    def tick(self) -> None:
        if self.state["done"]:
            self.event("tick: campaign already complete")
            return
        if now() > parse(self.caps["campaign_hard_stop_utc"]) and not self.state["final"]:
            self.state["stop_reason"] = "campaign hard stop reached"
        base = self.baseline_dev()
        max_iter = 3
        concluded = 0
        stop_loop = bool(self.state.get("stop_reason"))
        for k in range(1, max_iter + 1):
            if stop_loop:
                break
            it = self.state["iterations"].get(str(k), {})
            if not it.get("concluded_utc"):
                if k > 1 and base is None:
                    break                       # need the matched baseline before planning
                status = self.iteration(k)
                self.update_best()
                if status != "concluded":
                    break
            concluded = k
            best = self.state.get("best")
            if best and best["dev"]["h10"]["settled"] == 8:
                self.state["stop_reason"] = "stretch target 8/8 reached"
                stop_loop = True
            if not self.within_gpu_hours():
                self.state["stop_reason"] = "GPU-hour cap reached"
                stop_loop = True
        self.update_best()
        loop_done = stop_loop or concluded == max_iter or (
            concluded >= 1 and now() > parse(self.caps["no_new_iteration_after_utc"]))
        if loop_done and base is not None:
            self.finalize()
        if self.dry:
            return
        self.update_status(*self.describe())
        self.schedule_ticks()
        self.save()


class Lock:
    """Exclusive tick lock that recognises a dead holder immediately.

    A tick killed mid-run (tick 21400711 was, by SIGTERM) cannot remove its
    lock, and a purely time-based rule would then block every tick for the
    full staleness window. The lock records the holder's Slurm job id, or its
    host and pid when run by hand, and is broken as soon as that holder is
    provably gone.
    """

    def __init__(self, path: Path):
        self.path = path

    def holder_alive(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
        except (ValueError, OSError):
            return time.time() - self.path.stat().st_mtime < LOCK_STALE_SECONDS
        if time.time() - float(info.get("time", 0)) > LOCK_STALE_SECONDS:
            return False
        job = info.get("slurm_job_id")
        if job:
            verdict = summarize([s for v in job_states([job]).values() for s in v])
            return verdict in ("active", "unknown")
        if info.get("host") == socket.gethostname():
            try:
                os.kill(int(info["pid"]), 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
        return True

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self.holder_alive():
                raise SystemExit(f"driver locked by {self.path.read_text().strip()}; exiting")
            self.path.unlink(missing_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                 "time": time.time(), "utc": stamp()}).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("command", choices=("tick", "status"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.campaign.resolve()
    if args.command == "status":
        print(json.dumps(Driver(root, dry=True).state, indent=2))
        return 0
    with Lock(root / "ledger/.driver.lock"):
        driver = Driver(root, dry=args.dry_run)
        driver.tick()
    return 0


if __name__ == "__main__":
    sys.exit(main())
