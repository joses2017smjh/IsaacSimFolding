"""Restartable, duplicate-safe driver for v7: depth-qualified supervision.

Stage graph (every stage keyed in ledger/driver_state.json and never
resubmitted while its key holds a live or finished job):

    smoke           one short search row on the off-metadata large-garment
                    P_B placement: branch telemetry, spawn validity and the
                    compiler's pure functions (audit/smoke_telemetry.json)     1
    search.collect  fresh recovery search with branch telemetry: 28
                    training-only rows at P_A, P_B (small and large garments)
                    and P_C, baseline as student (v4's machinery)             28 tasks
    search.compile  mechanism check (landed-deep settles more often) and a
                    per-pose gate on DEPTH-QUALIFIED branches (exit 4: stop,
                    train nothing), then only those branches' labels,
                    pose-balanced weights; no earlier corpus is reused         CPU
    pb1.train       v4 attempt 2's recipe verbatim, from the untouched baseline 1
    selection       latest guard-passing checkpoint, else none (preregistered)
    pb1.reload      evaluator-style reload                                      1
    pb1.fit         repair verification + paired non-inferiority on labels      1
    run1, run2      v5's matched arrays: baseline and candidate interleaved on
                    the same 16 development rows, submitted by one tick         32 + 32
    verdict         matched_verdict(), unchanged from v5
    final           the untouched final set, both policies, ONLY if the rule
                    holds                                                       16

The baseline is kept unless the candidate meets the rule. N is fixed at two
runs per policy per horizon; nothing here can add a run or a second attempt.

Infrastructure is v4's and v5's, each piece fixed there after failing live:
chain ticks on Slurm `?` OR-dependencies, a 2-hour watchdog, a lock that
recognises a dead holder, stage adoption from the ledger, one retry of
infrastructure-failed rows, the real sbatch binary (the site wrapper
word-splits arguments), relative --begin (absolute times are cluster-local).

    driver.py --campaign ROOT tick      advance whatever can advance, schedule the next tick
    driver.py --campaign ROOT status    print the resumable state
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matched_task import POLICIES, checkpoint_block, destination, matched_index  # noqa: E402

REAL_SBATCH = "/apps/slurm/current/bin/sbatch"
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING",
          "RESIZING", "SUSPENDED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
          "STAGE_OUT"}
LOCK_STALE_SECONDS = 25 * 60
RUNS = ("r1", "r2")
HORIZONS = (10, 50)
TAG = "dq1"


# ------------------------------------------------------------------ time
def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp(t: dt.datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def latest_guard_passing(training: dict) -> tuple[dict | None, str]:
    """The LATEST checkpoint that passes the retention guard, else none."""
    passing = [c for c in training.get("checkpoints", []) if c.get("heldout_gate")]
    if not passing:
        return None, "no checkpoint passes the retention guard"
    chosen = max(passing, key=lambda c: int(c["step"]))
    return chosen, f"latest guard-passing checkpoint: step_{int(chosen['step']):06d}"


def fisher_one_sided(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    """P(A's success count >= observed | margins), hypergeometric; stdlib only."""
    from math import comb
    total, successes = a_n + b_n, a_success + b_success
    denom = comb(total, a_n)
    return sum(comb(successes, k) * comb(total - successes, a_n - k)
               for k in range(a_success, min(successes, a_n) + 1)) / denom


def pool(runs: list[dict]) -> tuple[int, int, int]:
    return (sum(int(r["settled"]) for r in runs), sum(int(r["valid"]) for r in runs),
            sum(int(r["rows"]) for r in runs))


def matched_verdict(cand: dict, base: dict, guard: bool, reload_ok: bool,
                    expected_rows: int = 16) -> dict:
    """v5's preregistered rule, unchanged, on runs measured in the same arrays:
    pooled H10 16 vs 16 with margin >= 4 AND one-sided Fisher p < 0.05; pooled
    candidate H50 >= pooled baseline H50; retention guard; reload; all rows
    valid."""
    cs, cn, cr = pool(cand["h10"])
    bs, bn, br = pool(base["h10"])
    c5, c5n, c5r = pool(cand["h50"])
    b5, b5n, b5r = pool(base["h50"])
    p = fisher_one_sided(cs, cn, bs, bn) if cn and bn else 1.0
    complete = all(n == r == expected_rows for n, r in ((cn, cr), (bn, br), (c5n, c5r), (b5n, b5r)))
    checks = {
        "pooled_h10_margin_ge_4": cs - bs >= 4,
        "fisher_one_sided_p_lt_0.05": p < 0.05,
        "pooled_h50_not_below_baseline": c5 >= b5,
        "retention_guard": bool(guard),
        "reload": bool(reload_ok),
        "all_rows_valid": complete,
    }
    return {"improved": all(checks.values()), "checks": checks, "p_value": p,
            "candidate_h10": f"{cs}/{cn}", "baseline_h10": f"{bs}/{bn}",
            "candidate_h50": f"{c5}/{c5n}", "baseline_h50": f"{b5}/{b5n}",
            "h10_margin": cs - bs, "h50_margin": c5 - b5}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------- Slurm
def relative_begin(when: dt.datetime) -> str:
    """Slurm's --begin reads an absolute timestamp in the CLUSTER's local time
    (UTC-7 here), so a UTC string lands seven hours late."""
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


class Driver:
    def __init__(self, root: Path, dry: bool = False):
        self.root = root.resolve()
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

    # ---- persistence
    def _load(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {"schema": 1, "created_utc": stamp(), "stages": {}, "search": None, "attempt": {},
                "scores": {}, "verdict": None, "final": {}, "ticks": {}, "done": False,
                "stop_reason": None, "log": []}

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

    # ---- budget (anti-runaway bounds)
    def gpu_tasks_used(self) -> int:
        return sum(int(j.get("gpu_tasks", 0) or 0) for j in self.ledger()["jobs"])

    def can_spend(self, tasks: int, *, reserve: int = 0) -> bool:
        return self.gpu_tasks_used() + tasks + reserve <= int(self.caps["gpu_tasks"])

    def within_gpu_hours(self) -> bool:
        return gpu_hours([j["job_id"] for j in self.ledger()["jobs"]]) <= float(self.caps["gpu_hours"])

    def reserve_final(self) -> int:
        return int(self.caps["reserve"]["final_set"])

    # ---- stages
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
        if gpu_tasks and not self.within_gpu_hours():
            self.event(f"budget refuses {key}: GPU-hour cap reached")
            self.state["stages"][key] = {"job_ids": [], "status": "skipped", "reason": "gpu hour budget"}
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
                                     "gpu_tasks": gpu_tasks, "array": array, **extra}
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

    def retry_or_wait(self, key, script, args, bad_dirs: list[Path], array: str, n_bad: int, reserve: int):
        """Move infrastructure-failed outputs aside and resubmit them once."""
        retry = f"{key}.retry1"
        if self.stage(retry) is None:
            for src in bad_dirs:
                if src.exists():
                    dst = self.root / "attempts" / src.relative_to(self.root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst) + ".attempt1")
            self.event(f"{key}: retrying infrastructure-failed tasks [{array}]")
            self.submit(retry, script, args, gpu_tasks=n_bad, array=array, reserve=reserve)
            return "wait"
        if self.stage_status(retry) == "active":
            return "wait"
        return "retried"

    def rollout_stage(self, key, script, args, base: Path, rows: list[dict], reserve: int = 0) -> str:
        """One array over plain rows (the search). Returns wait / done / incomplete / skipped."""
        indices = list(range(len(rows)))
        if self.stage(key) is None:
            self.submit(key, script, args, gpu_tasks=len(indices), array=",".join(map(str, indices)),
                        reserve=reserve)
            return "skipped" if self.stage(key).get("status") == "skipped" else "wait"
        if self.stage(key).get("status") == "skipped":
            return "skipped"
        if self.stage_status(key) == "active":
            return "wait"
        bad = [i for i in indices if self.read_result(base / rows[i]["id"]) is None]
        if not bad:
            return "done"
        st = self.retry_or_wait(key, script, args, [base / rows[i]["id"] for i in bad],
                                ",".join(map(str, bad)), len(bad), reserve)
        if st == "wait":
            return "wait"
        return "done" if all(self.read_result(base / rows[i]["id"]) for i in indices) else "incomplete"

    # ---- matched evaluation (v5)
    def label(self, which: str) -> str:
        return checkpoint_block(self.root, self.manifest, which)["label"]

    def dest(self, phase: str, index: int, run: str, rows: list[dict]) -> Path:
        which, r = matched_index(index)
        return destination(self.root, phase, rows[r], f"{self.label(which)}-{run}")

    def matched_stage(self, key: str, phase: str, run: str, rows: list[dict], reserve: int = 0) -> str:
        """ONE interleaved array over both policies. Returns wait / done / incomplete / skipped."""
        n = 2 * len(rows)
        indices = list(range(n))
        args = [str(self.root), phase, run]
        if self.stage(key) is None:
            self.submit(key, "matched.sbatch", args, gpu_tasks=n, array=f"0-{n - 1}", reserve=reserve)
            return "skipped" if self.stage(key).get("status") == "skipped" else "wait"
        if self.stage(key).get("status") == "skipped":
            return "skipped"
        if self.stage_status(key) == "active":
            return "wait"
        bad = [i for i in indices if self.read_result(self.dest(phase, i, run, rows)) is None]
        if not bad:
            return "done"
        st = self.retry_or_wait(key, "matched.sbatch", args, [self.dest(phase, i, run, rows) for i in bad],
                                ",".join(map(str, bad)), len(bad), reserve)
        if st == "wait":
            return "wait"
        return "done" if all(self.read_result(self.dest(phase, i, run, rows)) for i in indices) else "incomplete"

    def score_run(self, phase: str, run: str, rows: list[dict], which: str, horizon: int) -> dict:
        """Settled terminal success for one policy, run and horizon. Infrastructure
        failures are not policy failures: counted apart and excluded."""
        these = [(i, r) for i, r in enumerate(rows) if int(r.get("horizon", 10)) == horizon]
        valid, settled, ever, conditions, invalid = 0, 0, 0, [], []
        for r_index, row in these:
            res = self.read_result(self.dest(phase, 2 * r_index + POLICIES.index(which), run, rows))
            if not res:
                invalid.append(row["id"])
                continue
            valid += 1
            settled += int(res["terminal_success"])
            ever += int(res["ever_success"])
            conditions.append(res["conditions_passed"])
        return {"run": run, "rows": len(these), "valid": valid, "settled": settled, "ever": ever,
                "mean_conditions": (sum(conditions) / len(conditions)) if conditions else None,
                "invalid": invalid}

    def scores_now(self) -> dict:
        rows = self.manifest["benchmark"]
        return {which: {f"h{h}": [self.score_run("benchmark", run, rows, which, h) for run in RUNS]
                        for h in HORIZONS}
                for which in POLICIES}

    # ---- the campaign
    def finish(self, reason: str | None, winner: str = "baseline") -> str:
        fin = self.state["final"]
        block = checkpoint_block(self.root, self.manifest, winner)
        fin["selected"] = {"label": block["label"], "checkpoint": block["path"]}
        if reason:
            self.state["stop_reason"] = reason
            fin.setdefault("skipped", "no candidate met the preregistered rule; the final set stays unspent")
        fin["completed_utc"] = stamp()
        self.state["done"] = True
        self.event("campaign complete" + (f": {reason}" if reason else ""), selected=block["label"])
        return "done"

    def smoke(self) -> dict | None:
        """The launch gate for the changed runner and compiler interfaces."""
        if self.state.get("smoke"):
            return self.state["smoke"]
        report = self.root / "audit/smoke_telemetry.json"
        key = "smoke"
        if self.stage("smoke.retry1") is not None:
            key = "smoke.retry1"
        if self.stage(key) is None:
            self.submit(key, "smoke.sbatch", [str(self.root)], gpu_tasks=1, reserve=self.reserve_final())
            return None
        if self.stage_status(key) == "active":
            return None
        if not report.is_file() and key == "smoke":
            # infrastructure failure: move the partial output aside, retry once
            src = self.root / "smoke"
            if src.exists():
                dst = self.root / "attempts" / "smoke.attempt1"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            self.event("smoke produced no report; retrying once")
            self.submit("smoke.retry1", "smoke.sbatch", [str(self.root)], gpu_tasks=1, reserve=self.reserve_final())
            return None
        rep = json.loads(report.read_text()) if report.is_file() else \
            {"passed": False, "problems": ["smoke produced no report twice"]}
        self.state["smoke"] = {k: rep.get(k) for k in ("passed", "problems", "completed_attempts",
                                                        "start_margins_cm", "off_metadata_pose")}
        self.event(f"smoke passed={rep.get('passed')}", problems=rep.get("problems"),
                   start_margins_cm=rep.get("start_margins_cm"))
        return self.state["smoke"]

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
            self.submit("search.compile", "compile.sbatch", [str(self.root)], gpu_tasks=0)
            return None
        if self.stage_status("search.compile") == "active":
            return None
        gate_path = self.root / "audit/recovery_coverage_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else \
            {"passed": False, "unmet_requirements": ["compile produced no gate report"]}
        if gate.get("passed") and not (self.root / "datasets/recovery.npz").is_file():
            gate = dict(gate, passed=False, unmet_requirements=["gate passed but the corpus was not written"])
        self.state["search"] = gate
        mech = gate.get("mechanism_check") or {}
        self.event(f"mechanism check + per-pose depth gate passed={gate['passed']}",
                   mechanism={k: mech.get(k) for k in ("deep", "shallow", "p_one_sided")},
                   qualifying={p: g.get("successful_branches") for p, g in gate.get("per_pose", {}).items()},
                   unmet=gate.get("unmet_requirements"))
        return gate

    def train_select_verify(self) -> dict | str:
        """Train -> select -> reload -> fit gate. Returns the candidate record,
        'wait', or a conclusion string (prefixed 'stop:')."""
        at = self.state["attempt"]
        C = str(self.root)
        train_args = [C, "recovery", TAG]
        if self.stage(f"{TAG}.train") is None:
            self.submit(f"{TAG}.train", "train.sbatch", train_args, gpu_tasks=1, reserve=self.reserve_final())
            return "wait"
        if self.stage_status(f"{TAG}.train") == "active":
            return "wait"
        tj = self.root / "training" / TAG / "training.json"
        if not tj.is_file():
            if self.stage(f"{TAG}.train.retry1") is None:
                if tj.parent.exists():
                    shutil.move(str(tj.parent), str(self.root / "attempts" / f"training-{TAG}.attempt1"))
                self.submit(f"{TAG}.train.retry1", "train.sbatch", train_args, gpu_tasks=1)
                return "wait"
            if self.stage_status(f"{TAG}.train.retry1") == "active":
                return "wait"
            if not tj.is_file():
                return "stop:training failed twice"
        chosen, why = latest_guard_passing(json.loads(tj.read_text()))
        at["selection"] = why
        if chosen is None:
            return f"stop:{why}"
        label = f"{TAG}-step{int(chosen['step']):06d}"
        at.update(label=label, checkpoint=chosen["path"], guard=bool(chosen["heldout_gate"]),
                  retention_loss=chosen.get("heldout_loss"), retention_baseline=chosen.get("heldout_loss_baseline"))
        out = self.root / "audit" / f"reload_{label}.json"
        reload_args = [C, chosen["path"], str(self.root / "datasets/recovery.npz"), str(out)]
        if self.stage(f"{TAG}.reload") is None:
            self.submit(f"{TAG}.reload", "reload.sbatch", reload_args, gpu_tasks=1)
            return "wait"
        if self.stage_status(f"{TAG}.reload") == "active":
            return "wait"
        if not out.is_file():
            if self.stage(f"{TAG}.reload.retry1") is None:
                self.submit(f"{TAG}.reload.retry1", "reload.sbatch", reload_args, gpu_tasks=1)
                return "wait"
            if self.stage_status(f"{TAG}.reload.retry1") == "active":
                return "wait"
        reload = json.loads(out.read_text()) if out.is_file() else {}
        at["reload_ok"] = bool(reload.get("finite") and reload.get("config_matches_baseline"))
        if not at["reload_ok"]:
            return "stop:candidate failed the evaluator-style reload"
        fit_out = self.root / "audit" / f"fit_gate_{label}.json"
        base_ckpt = self.manifest["baseline_checkpoint"]["path"]
        fit_args = [C, f"baseline={base_ckpt},candidate={chosen['path']}", str(fit_out)]
        if self.stage(f"{TAG}.fit") is None:
            self.submit(f"{TAG}.fit", "offline_fit.sbatch", fit_args, gpu_tasks=1)
            return "wait"
        if self.stage_status(f"{TAG}.fit") == "active":
            return "wait"
        if not fit_out.is_file():
            if self.stage(f"{TAG}.fit.retry1") is None:
                self.submit(f"{TAG}.fit.retry1", "offline_fit.sbatch", fit_args, gpu_tasks=1)
                return "wait"
            if self.stage_status(f"{TAG}.fit.retry1") == "active":
                return "wait"
            if not fit_out.is_file():
                return "stop:fit gate job failed twice"
        gate = json.loads(fit_out.read_text()).get("gate") or {}
        at["fit_gate"] = {k: gate.get(k) for k in ("pass", "repair_ok", "noninferior_ok", "mean_paired_diff",
                                                   "diff_ci95", "baseline_seed_spread")}
        if not gate.get("pass"):
            bits = []
            if not gate.get("repair_ok"):
                bits.append("repair verification failed (bf16 expert still frozen)")
            if not gate.get("noninferior_ok"):
                bits.append("candidate is paired-inferior to the baseline beyond seed noise")
            return "stop:fit precondition failed: " + "; ".join(bits or ["no gate record"])
        record = self.root / "audit" / "candidate_checkpoint.json"
        if not record.is_file():
            ckpt = Path(chosen["path"])
            block = {"path": str(ckpt), "label": label, "step": int(chosen["step"]), "selection": why,
                     "sha256": {p.name: file_sha(p) for p in sorted(ckpt.iterdir()) if p.is_file()},
                     "written_utc": stamp()}
            if not self.dry:
                record.write_text(json.dumps(block, indent=2) + "\n")
                at["candidate_record_sha256"] = file_sha(record)
                self.event(f"candidate pinned: {label}", sha256_model=block["sha256"].get("model.safetensors"))
        return {"label": label, "guard": at["guard"], "reload_ok": at["reload_ok"]}

    def advance(self) -> str:
        sm = self.smoke()
        if sm is None:
            return "wait"
        if not sm.get("passed"):
            return self.finish(f"smoke failed: {sm.get('problems')}; nothing is searched")
        gate = self.search()
        if gate is None:
            return "wait"
        if not gate.get("passed"):
            return self.finish(f"depth gate unmet: {gate.get('unmet_requirements')}; landing depth is not "
                               "a usable lever on this search, so nothing is trained")
        cand = self.train_select_verify()
        if cand == "wait":
            return "wait"
        if isinstance(cand, str) and cand.startswith("stop:"):
            return self.finish(cand[5:])
        M = self.manifest
        dev = M["benchmark"]
        s1 = self.matched_stage("run1", "benchmark", "r1", dev, reserve=self.reserve_final())
        s2 = self.matched_stage("run2", "benchmark", "r2", dev, reserve=self.reserve_final())
        if "wait" in (s1, s2):
            return "wait"
        self.state["scores"] = self.scores_now()
        if self.state.get("verdict") is None:
            v = matched_verdict(self.state["scores"]["candidate"], self.state["scores"]["baseline"],
                                cand["guard"], cand["reload_ok"])
            v["run_status"] = {"run1": s1, "run2": s2}
            self.state["verdict"] = v
            self.event("verdict: " + ("IMPROVED under the preregistered rule" if v["improved"]
                                      else "not improved under the preregistered rule"),
                       candidate_h10=v["candidate_h10"], baseline_h10=v["baseline_h10"],
                       candidate_h50=v["candidate_h50"], baseline_h50=v["baseline_h50"],
                       p_value=round(v["p_value"], 4), failed=[k for k, ok in v["checks"].items() if not ok])
        v = self.state["verdict"]
        if not v["improved"]:
            return self.finish("the candidate did not meet the preregistered rule")
        fin = self.state["final"]
        sf = self.matched_stage("final", "final_test", "f1", M["final_test"])
        if sf == "wait":
            return "wait"
        fin["status"] = sf
        for which in POLICIES:
            fin[which] = self.score_run("final_test", "f1", M["final_test"], which, 10)
        return self.finish(None, "candidate")

    def tick(self) -> None:
        if not self.state["done"]:
            self.advance()
        if self.dry:
            return
        self.update_status(*self.describe())
        self.schedule_ticks()
        self.save()

    # ---- ticks and status (as v4/v5)
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
        sm = self.state.get("smoke")
        if sm:
            parts.append(f"smoke {'pass' if sm.get('passed') else 'FAIL'}")
        g = self.state.get("search")
        if g:
            per = ", ".join(f"{p} {x.get('successful_branches')}/{x.get('attempts_completed')}"
                            for p, x in g.get("per_pose", {}).items())
            mech = g.get("mechanism_check") or {}
            parts.append(f"search: depth-qualified {per or '-'}; landed-deep settled {mech.get('deep')} vs "
                         f"shallow {mech.get('shallow')} (gate {'pass' if g.get('passed') else 'FAIL'})")
        at = self.state.get("attempt") or {}
        if at.get("selection"):
            parts.append(f"{at['selection']}" + (f", reload {'ok' if at.get('reload_ok') else 'FAIL'}"
                                                  if "reload_ok" in at else "")
                         + (f", fit gate {'pass' if at['fit_gate'].get('pass') else 'FAIL'}" if at.get("fit_gate") else ""))
        if self.stage("run1") is not None and (self.root / "audit/candidate_checkpoint.json").is_file():
            sc = self.state.get("scores") or self.scores_now()
            for which in POLICIES:
                bits = [f"H{h} " + "+".join(f"{r['settled']}/{r['valid']}" for r in sc[which][f"h{h}"])
                        for h in HORIZONS]
                parts.append(f"{self.label(which)}: " + ", ".join(bits))
        v = self.state.get("verdict")
        if v:
            parts.append(("IMPROVED" if v["improved"] else "not improved")
                         + f" (H10 {v['candidate_h10']} vs {v['baseline_h10']}, p={v['p_value']:.3f}; "
                           f"H50 {v['candidate_h50']} vs {v['baseline_h50']})")
        if self.state["done"]:
            sel = self.state["final"].get("selected", {})
            parts.append(f"FINAL: `{sel.get('label')}` delivered")
        active = ", ".join(live) if live else ("none — campaign complete" if self.state["done"] else "none")
        blocker = self.state.get("stop_reason") or "none"
        nxt = ("none — campaign complete; see REPORT.md" if self.state["done"]
               else "driver advances the next stage when a waited job ends")
        return active, "; ".join(parts) or "smoke pending", blocker, nxt


class Lock:
    """Exclusive tick lock that recognises a dead holder immediately."""

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
