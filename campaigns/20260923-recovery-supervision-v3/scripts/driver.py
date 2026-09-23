"""Restartable, duplicate-safe driver for the v3 recovery-supervision campaign.

v3 stage graph (every stage keyed, never resubmitted while its key holds a job):

    baseline.h10     fresh baseline H10 repeat on the 8 development poses
    search.collect   bounded recovery search on training-only garments
    search.compile   validated examples + coverage gate (exit 4 = no coverage)
    aK.train -> aK.reload -> aK.screen (H10 run 1)
             -> aK.confirm_h10 (H10 run 2) + aK.h50      [only if screen >= 4/8]
    final.*          untouched final set, candidate + baseline [only if improved]

Attempt 1 is fixed by the manifest. Attempt 2 exists only as an explicit,
committed plans/attempt2.json written from evidence; there is no default
ladder, so nothing unsuccessful is repeated by automation.

Infrastructure carried from v2, each fixed there after failing live:

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


def fisher_one_sided(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    """P(A's success count >= observed | margins), hypergeometric; stdlib only."""
    from math import comb
    total, successes = a_n + b_n, a_success + b_success
    denom = comb(total, a_n)
    return sum(comb(successes, k) * comb(total - successes, a_n - k)
               for k in range(a_success, min(successes, a_n) + 1)) / denom


def improvement_verdict(cand_runs: list[dict], base_runs: list[dict], cand_h50: dict,
                        base_h50: int, guard: bool, reload_ok: bool) -> dict:
    """The preregistered improvement rule, pure so it is testable."""
    cs = sum(r["settled"] for r in cand_runs); cn = sum(r["valid"] for r in cand_runs)
    bs = sum(r["settled"] for r in base_runs); bn = sum(r["valid"] for r in base_runs)
    p = fisher_one_sided(cs, cn, bs, bn) if cn and bn else 1.0
    checks = {
        "pooled_h10_margin_ge_4": cs - bs >= 4,
        "fisher_one_sided_p_lt_0.05": p < 0.05,
        "h50_not_below_baseline": cand_h50.get("settled", -1) >= base_h50,
        "retention_guard": bool(guard),
        "reload": bool(reload_ok),
        "all_rows_valid": all(r["valid"] == r["rows"] for r in cand_runs + [cand_h50]),
    }
    return {"improved": all(checks.values()), "checks": checks, "p_value": p,
            "candidate_h10": f"{cs}/{cn}", "baseline_h10": f"{bs}/{bn}",
            "target_6_of_8_both_runs": all(r["settled"] >= 6 for r in cand_runs) and len(cand_runs) == 2,
            "stretch_8_of_8_both_runs": all(r["settled"] == 8 for r in cand_runs) and len(cand_runs) == 2}


H10 = [2 * i for i in range(8)]        # development row indices at H10
H50 = [2 * i + 1 for i in range(8)]    # ... and at H50


class Driver:
    def __init__(self, root: Path, dry: bool = False):
        self.root = root.resolve()
        self.repo = self.root.parents[1]
        self.dry = dry
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.caps = self.manifest["budget"]
        self.state_path = self.root / "ledger/driver_state.json"
        self.ledger_path = self.root / "ledger/slurm-jobs.json"
        if not self.ledger_path.exists():
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.write_text(json.dumps({"campaign": self.manifest["campaign"], "jobs": []}) + "\n")
        self.state = self._load()
        self.adopt()
        self.waiting: list[str] = []
        self.begin_at: dt.datetime | None = None

    # ---- persistence (as v2)
    def _load(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {"schema": 1, "created_utc": stamp(), "stages": {}, "attempts": {},
                "baseline": None, "search": None, "final": {}, "ticks": {},
                "done": False, "stop_reason": None, "log": []}

    def adopt(self) -> None:
        """Rebuild any stage the state lost from the ledger (see v2 tick 21400711)."""
        for job in self.ledger()["jobs"]:
            key = job.get("phase")
            if job.get("submitted_by") == "driver" and key and key not in self.state["stages"]:
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

    def record(self, job, stage, script, gpu_tasks, **extra) -> None:
        data = self.ledger()
        data["jobs"].append({"job_id": job, "phase": stage, "script": script, "gpu_tasks": gpu_tasks,
                             "submitted_by": "driver", "manifest_commit": self.manifest["git"]["commit"],
                             "submitted_utc": stamp(), "state": "submitted", **extra})
        self.ledger_path.write_text(json.dumps(data, indent=2) + "\n")

    # ---- budget
    def gpu_tasks_used(self) -> int:
        return sum(int(j.get("gpu_tasks", 0) or 0) for j in self.ledger()["jobs"])

    def can_spend(self, tasks: int, *, reserve: int = 0) -> bool:
        return self.gpu_tasks_used() + tasks + reserve <= int(self.caps["gpu_tasks"])

    def within_gpu_hours(self) -> bool:
        return gpu_hours([j["job_id"] for j in self.ledger()["jobs"]]) <= float(self.caps["gpu_hours"])

    # ---- stages (as v2)
    def stage(self, key):
        return self.state["stages"].get(key)

    def stage_status(self, key: str) -> str:
        st = self.stage(key)
        if st is None:
            return "absent"
        if st.get("status") in ("completed", "failed", "skipped"):
            return st["status"]
        flat = [s for j in st["job_ids"] for s in job_states(st["job_ids"]).get(j, [])]
        verdict = summarize(flat)
        if verdict == "unknown":
            verdict = "active"
        if verdict in ("completed", "failed"):
            st["status"] = verdict
            st["finished_utc"] = stamp()
        else:
            self.waiting.extend(st["job_ids"])
        return verdict

    def submit(self, key, script, args, *, gpu_tasks, array=None, reserve=0, **extra):
        if self.stage(key) is not None:
            return None
        if gpu_tasks and not self.can_spend(gpu_tasks, reserve=reserve):
            self.event(f"budget refuses {key}: {gpu_tasks} GPU tasks")
            self.state["stages"][key] = {"job_ids": [], "status": "skipped", "reason": "gpu task budget"}
            return None
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        argv = [f"--chdir={self.root}",
                f"--output={logs}/{key}-%A_%a.out" if array else f"--output={logs}/{key}-%j.out"]
        if array:
            argv.append(f"--array={array}%8" if gpu_tasks else f"--array={array}")
        argv += [str(self.root / "slurm" / script), *args]
        if self.dry:
            job = f"DRY-{key}"
        else:
            job = sbatch(argv)
            self.record(job, key, f"slurm/{script}", gpu_tasks, array=array, **extra)
        self.state["stages"][key] = {"job_ids": [job], "submitted_utc": stamp(), "status": "submitted",
                                     "gpu_tasks": gpu_tasks, **extra}
        self.waiting.append(job)
        self.event(f"submitted {key} -> {job}", gpu_tasks=gpu_tasks)
        if not self.dry:
            self.save()
        return job

    def read_result(self, directory: Path) -> dict | None:
        sp, rp = directory / "status.json", directory / "rollout.json"
        if not sp.is_file() or json.loads(sp.read_text()).get("state") != "completed" or not rp.is_file():
            return None
        r = json.loads(rp.read_text())
        c = r.get("terminal_checker", {})
        return {"terminal_success": bool(r.get("terminal_success")), "ever_success": bool(r.get("success")),
                "conditions_passed": c.get("conditions_passed"), "conditions_total": c.get("conditions_total"),
                "horizon": r.get("effective_n_action_steps")}

    def rollout_stage(self, key, script, args, base: Path, rows: list[dict],
                      indices: list[int] | None = None, reserve: int = 0) -> str:
        """Submit a row subset, wait, retry infrastructure failures once.
        Returns wait / done / incomplete / skipped."""
        indices = list(range(len(rows))) if indices is None else indices
        array = ",".join(map(str, indices))
        if self.stage(key) is None:
            self.submit(key, script, args, gpu_tasks=len(indices), array=array, reserve=reserve)
            return "skipped" if self.stage(key).get("status") == "skipped" else "wait"
        if self.stage(key).get("status") == "skipped":
            return "skipped"
        if self.stage_status(key) == "active":
            return "wait"
        bad = [i for i in indices if self.read_result(base / rows[i]["id"]) is None]
        if not bad:
            return "done"
        retry = f"{key}.retry1"
        if self.stage(retry) is None:
            for i in bad:
                src = base / rows[i]["id"]
                if src.exists():
                    dst = self.root / "attempts" / src.relative_to(self.root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst) + ".attempt1")
            self.event(f"{key}: retrying infrastructure-failed rows {bad}")
            self.submit(retry, script, args, gpu_tasks=len(bad), array=",".join(map(str, bad)), reserve=reserve)
            return "wait"
        if self.stage_status(retry) == "active":
            return "wait"
        return "done" if all(self.read_result(base / rows[i]["id"]) for i in indices) else "incomplete"

    def score(self, label: str, phase: str, rows: list[dict], indices: list[int]) -> dict:
        sub = [rows[i] for i in indices]
        res = {r["id"]: self.read_result(self.root / "evaluation" / label / phase / r["id"]) for r in sub}
        horizon = int(sub[0].get("horizon", 10))
        return score_rows([dict(r, horizon=horizon) for r in sub], res)[f"h{horizon}"]

    def committed(self, path: Path) -> bool:
        rel = str(path.relative_to(self.repo))
        tracked = subprocess.run(["git", "-C", str(self.repo), "ls-files", "--error-unmatch", rel],
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(self.repo), "diff", "--quiet", "HEAD", "--", rel],
                               capture_output=True).returncode == 0
        return tracked and clean

    # ---- stages specific to v3
    def baseline_repeat(self) -> dict | None:
        if self.state.get("baseline"):
            return self.state["baseline"]
        rows = self.manifest["benchmark"]
        st = self.rollout_stage("baseline.h10", "benchmark.sbatch",
                                [str(self.root), self.manifest["baseline_checkpoint"]["path"], "baseline-v3"],
                                self.root / "evaluation/baseline-v3/benchmark", rows, H10)
        if st == "wait":
            return None
        v2 = json.loads((self.root.parent / "20260922-closed-loop-training-v2/ledger/driver_state.json").read_text())
        runs = [dict(v2["baseline_dev"]["h10"], source="v2 run"),
                dict(self.score("baseline-v3", "benchmark", rows, H10), source="v3 run")]
        self.state["baseline"] = {"h10_runs": runs, "h50": dict(v2["baseline_dev"]["h50"], source="v2 run")}
        self.event("baseline H10 repeat measured", v2=runs[0]["settled"], v3=runs[1]["settled"])
        return self.state["baseline"]

    def search(self) -> dict | None:
        if self.state.get("search"):
            return self.state["search"]
        rows = self.manifest["recovery_search"]
        st = self.rollout_stage("search.collect", "recovery.sbatch", [str(self.root)],
                                self.root / "recovery", rows, reserve=self.reserve_final())
        if st == "wait":
            return None
        if st in ("incomplete", "skipped"):
            ok = [r for r in rows if self.read_result(self.root / "recovery" / r["id"])]
            if len(ok) < len(rows) - 1:
                self.state["search"] = {"passed": False, "unmet_requirements": [f"search {st}"]}
                return self.state["search"]
        if self.stage("search.compile") is None:
            self.submit("search.compile", "compile_recovery.sbatch", [str(self.root)], gpu_tasks=0)
            return None
        if self.stage_status("search.compile") == "active":
            return None
        gate_path = self.root / "audit/recovery_coverage_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else \
            {"passed": False, "unmet_requirements": ["compile produced no gate report"]}
        self.state["search"] = gate
        self.event(f"recovery coverage gate passed={gate['passed']}",
                   successes=gate.get("successful_branches"), unmet=gate.get("unmet_requirements"))
        return gate

    def reserve_final(self) -> int:
        return int(self.caps["reserve"]["final_set"])

    def attempt_plan(self, k: int) -> dict | None:
        if k == 1:
            return {"attempt": 1, "dataset": "recovery", "init_checkpoint": self.manifest["baseline_checkpoint"]["path"],
                    "training": {}, "source": "manifest"}
        path = self.root / "plans" / f"attempt{k}.json"
        if path.is_file() and self.committed(path):
            return dict(json.loads(path.read_text()), source="explicit")
        return None

    def attempt(self, k: int, base: dict) -> str:
        """Returns wait / concluded."""
        at = self.state["attempts"].setdefault(str(k), {})
        if at.get("concluded_utc"):
            return "concluded"
        plan = self.attempt_plan(k)
        if plan is None:
            return "wait"
        at.setdefault("plan", plan)
        C, tag = str(self.root), f"a{k}"
        rows = self.manifest["benchmark"]
        plan_file = self.root / "plans" / f"attempt{k}.json"
        train_args = [C, plan["dataset"], tag] + ([str(plan_file)] if k > 1 else [])
        if self.stage(f"{tag}.train") is None:
            self.submit(f"{tag}.train", "train.sbatch", train_args, gpu_tasks=1, reserve=self.reserve_final())
            return "wait"
        if self.stage_status(f"{tag}.train") == "active":
            return "wait"
        tj = self.root / "training" / tag / "training.json"
        if not tj.is_file():
            if self.stage(f"{tag}.train.retry1") is None:
                if tj.parent.exists():
                    shutil.move(str(tj.parent), str(self.root / "attempts" / f"training-{tag}.attempt1"))
                self.submit(f"{tag}.train.retry1", "train.sbatch", train_args, gpu_tasks=1)
                return "wait"
            if self.stage_status(f"{tag}.train.retry1") == "active":
                return "wait"
            if not tj.is_file():
                return self.conclude(k, "training failed twice")
        chosen, why = select_checkpoint(json.loads(tj.read_text()))
        at["selection"] = why
        if chosen is None:
            return self.conclude(k, why)
        label = f"{tag}-step{int(chosen['step']):06d}"
        cand = at.setdefault("candidate", {})
        cand.update(label=label, checkpoint=chosen["path"], guard=bool(chosen["heldout_gate"]),
                    retention_loss=chosen.get("heldout_loss"), retention_baseline=chosen.get("heldout_loss_baseline"),
                    anchor_fit_loss=chosen.get("anchor_fit_loss"))
        out = self.root / "audit" / f"reload_{label}.json"
        reload_args = [C, chosen["path"], str(self.root / "datasets" / f"{plan['dataset']}.npz"), str(out)]
        if self.stage(f"{tag}.reload") is None:
            self.submit(f"{tag}.reload", "reload.sbatch", reload_args, gpu_tasks=1)
            return "wait"
        if self.stage_status(f"{tag}.reload") == "active":
            return "wait"
        if not out.is_file():
            if self.stage(f"{tag}.reload.retry1") is None:
                self.submit(f"{tag}.reload.retry1", "reload.sbatch", reload_args, gpu_tasks=1)
                return "wait"
            if self.stage_status(f"{tag}.reload.retry1") == "active":
                return "wait"
        reload = json.loads(out.read_text()) if out.is_file() else {}
        cand["reload_ok"] = bool(reload.get("finite") and reload.get("config_matches_baseline"))
        if not cand["reload_ok"]:
            return self.conclude(k, "candidate failed the evaluator-style reload")
        # screen: one H10 run
        r1 = f"{label}-r1"
        st = self.rollout_stage(f"{tag}.screen", "benchmark.sbatch", [C, chosen["path"], r1],
                                self.root / "evaluation" / r1 / "benchmark", rows, H10, reserve=self.reserve_final())
        if st == "wait":
            return "wait"
        if st == "skipped":
            return self.conclude(k, "screen skipped: budget")
        cand["h10_r1"] = self.score(r1, "benchmark", rows, H10)
        if cand["h10_r1"]["settled"] < 4:
            return self.conclude(k, f"screen failed: H10 {cand['h10_r1']['settled']}/8 < 4/8")
        # confirmation: second H10 run + one H50 run
        r2 = f"{label}-r2"
        s2 = self.rollout_stage(f"{tag}.confirm_h10", "benchmark.sbatch", [C, chosen["path"], r2],
                                self.root / "evaluation" / r2 / "benchmark", rows, H10, reserve=self.reserve_final())
        s3 = self.rollout_stage(f"{tag}.h50", "benchmark.sbatch", [C, chosen["path"], r1],
                                self.root / "evaluation" / r1 / "benchmark", rows, H50, reserve=self.reserve_final())
        if "wait" in (s2, s3):
            return "wait"
        if "skipped" in (s2, s3):
            return self.conclude(k, "confirmation skipped: budget")
        cand["h10_r2"] = self.score(r2, "benchmark", rows, H10)
        cand["h50"] = self.score(r1, "benchmark", rows, H50)
        cand["verdict"] = improvement_verdict([cand["h10_r1"], cand["h10_r2"]], base["h10_runs"],
                                              cand["h50"], base["h50"]["settled"], cand["guard"], cand["reload_ok"])
        return self.conclude(k, "confirmed" if cand["verdict"]["improved"] else "confirmation did not establish improvement")

    def conclude(self, k: int, reason: str) -> str:
        at = self.state["attempts"].setdefault(str(k), {})
        at["concluded_utc"] = stamp()
        at["outcome"] = reason
        self.event(f"attempt {k} concluded: {reason}")
        return "concluded"

    def finalize(self, winner: dict | None) -> str:
        fin = self.state["final"]
        rows = self.manifest["final_test"]
        C = str(self.root)
        if winner:
            fin.setdefault("selected", {"label": winner["label"], "checkpoint": winner["checkpoint"]})
            a = self.rollout_stage("final.candidate", "test.sbatch", [C, winner["checkpoint"], winner["label"]],
                                   self.root / "evaluation" / winner["label"] / "final_test", rows)
            b = self.rollout_stage("final.baseline", "test.sbatch",
                                   [C, self.manifest["baseline_checkpoint"]["path"], "baseline-v3"],
                                   self.root / "evaluation/baseline-v3/final_test", rows)
            if "wait" in (a, b):
                return "wait"
            for who, lab, st in (("candidate", winner["label"], a), ("baseline", "baseline-v3", b)):
                fin[who] = {"skipped": "GPU task budget"} if st == "skipped" else \
                    self.score(lab, "final_test", rows, list(range(len(rows))))
        else:
            fin.setdefault("selected", {"label": "baseline", "checkpoint": self.manifest["baseline_checkpoint"]["path"],
                                        "reason": "no attempt met the preregistered improvement rule; final set not spent"})
        fin["completed_utc"] = stamp()
        self.state["done"] = True
        self.event("campaign complete")
        return "done"

    # ---- tick
    def tick(self) -> None:
        if self.state["done"]:
            if not self.dry:
                self.update_status(*self.describe())
                self.save()
            return
        base = self.baseline_repeat()
        gate = self.search()
        winner, finished = None, False
        if gate is not None and not gate.get("passed"):
            self.state["stop_reason"] = (f"recovery coverage gate unmet: {gate.get('unmet_requirements')}; "
                                         "route 2 requires a committed plan")
            finished = not (self.root / "plans/attempt2.json").is_file()
        elif gate is not None and base is not None:
            for k in (1, 2):
                status = self.attempt(k, base)
                cand = self.state["attempts"].get(str(k), {}).get("candidate") or {}
                if cand.get("verdict", {}).get("improved"):
                    winner = cand
                    break
                if status != "concluded":
                    if k == 2 and self.attempt_plan(2) is None:
                        concluded1 = parse(self.state["attempts"]["1"]["concluded_utc"])
                        due = concluded1 + dt.timedelta(minutes=120)
                        if now() >= due:
                            finished = True
                        else:
                            self.begin_at = due
                    break
            else:
                finished = True
        if winner or finished:
            self.finalize(winner)
        if self.dry:
            return
        self.update_status(*self.describe())
        self.schedule_ticks()
        self.save()

    # ---- ticks and status (as v2, with the self-cancel and time-zone fixes)
    def schedule_ticks(self) -> None:
        ticks = self.state["ticks"]
        waiting = sorted(set(self.waiting))
        me = os.environ.get("SLURM_JOB_ID")
        chain = ticks.get("chain")
        pending = chain and chain != me and \
            [s for v in job_states([chain]).values() for s in v][:1] == ["PENDING"]
        if pending and ticks.get("chain_deps") == waiting and \
                ticks.get("chain_begin") == (stamp(self.begin_at) if self.begin_at else None):
            pass
        else:
            if pending:
                scancel(chain)
            chain = None
        if chain is None and (waiting or self.begin_at) and not self.state["done"]:
            argv = [f"--chdir={self.root}", f"--output={self.root}/ledger/ticks/tick-%j.out"]
            if waiting:
                argv.append("--dependency=" + "?".join(f"afterany:{j}" for j in waiting))
            if self.begin_at:
                argv.append(f"--begin={relative_begin(self.begin_at)}")
            (self.root / "ledger/ticks").mkdir(parents=True, exist_ok=True)
            ticks["chain"] = sbatch(argv + [str(self.root / "slurm/tick.sbatch"), str(self.root)])
            ticks["chain_deps"] = waiting
            ticks["chain_begin"] = stamp(self.begin_at) if self.begin_at else None
        watchdog = ticks.get("watchdog")
        alive = watchdog and watchdog != me and \
            summarize([s for v in job_states([watchdog]).values() for s in v]) == "active"
        if not alive and not self.state["done"]:
            ticks["watchdog"] = sbatch([f"--chdir={self.root}", f"--output={self.root}/ledger/ticks/watchdog-%j.out",
                                        f"--begin={relative_begin(now() + dt.timedelta(hours=2))}",
                                        str(self.root / "slurm/tick.sbatch"), str(self.root)])

    def update_status(self, active, latest, blocker, nxt) -> None:
        path = self.root / "STATUS.md"
        if not path.is_file():
            return
        text = path.read_text()
        begin, end = "<!-- driver:status:begin -->", "<!-- driver:status:end -->"
        if begin not in text or end not in text:
            return
        block = (f"{begin}\n| | |\n|---|---|\n| **Active job** | {active} |\n| **Current result** | {latest} |\n"
                 f"| **Limitation / blocker** | {blocker} |\n| **Next automatic action** | {nxt} |\n\n"
                 f"_Updated {stamp()} by scripts/driver.py._\n{end}")
        head, rest = text.split(begin, 1)
        path.write_text(head + block + rest.split(end, 1)[1])

    def describe(self):
        for key, st in self.state["stages"].items():
            if st.get("status") == "submitted" and st.get("job_ids"):
                self.stage_status(key)
        live = [f"`{','.join(v['job_ids'])}` {k}" for k, v in self.state["stages"].items()
                if v.get("status") == "submitted"]
        parts = []
        b = self.state.get("baseline")
        if b:
            parts.append("baseline H10 " + " + ".join(f"{r['settled']}/8" for r in b["h10_runs"]))
        g = self.state.get("search")
        if g:
            parts.append(f"recovery search {g.get('successful_branches', '?')} settled successes"
                         f" of {g.get('attempts_completed', '?')} attempts (gate {'pass' if g.get('passed') else 'FAIL'})")
        for k, at in sorted(self.state["attempts"].items()):
            c = at.get("candidate") or {}
            bits = [f"attempt {k}"]
            if "h10_r1" in c:
                bits.append(f"H10 r1 {c['h10_r1']['settled']}/8")
            if "h10_r2" in c:
                bits.append(f"r2 {c['h10_r2']['settled']}/8, H50 {c['h50']['settled']}/8")
            if at.get("outcome"):
                bits.append(at["outcome"])
            parts.append(", ".join(bits))
        if self.state["done"]:
            sel = self.state["final"].get("selected", {})
            parts.append(f"FINAL: `{sel.get('label')}` delivered")
        active = ", ".join(live) if live else ("none — campaign complete" if self.state["done"] else "none")
        blocker = self.state.get("stop_reason") or "none"
        nxt = ("none — campaign complete; see REPORT.md" if self.state["done"]
               else "driver advances the next stage when a waited job ends")
        return active, "; ".join(parts) or "starting", blocker, nxt


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
