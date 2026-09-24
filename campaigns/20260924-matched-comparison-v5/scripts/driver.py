"""Restartable, duplicate-safe driver for the v5 matched comparison.

Stage graph (every stage keyed in ledger/driver_state.json and never
resubmitted while its key holds a live or finished job):

    smoke      both policies once on the 80-step smoke row       (2 tasks)
    run1       one matched array over the 16 development rows    (32 tasks)
    run2       a second, identical matched array                 (32 tasks)
    verdict    the preregistered rule -- matched_verdict(), pure
    final      the untouched final set, both policies, ONLY if the rule
               holds                                             (16 tasks)
    gallery    settled successes with provenance, failures counted

"Matched array": even indices run the baseline and odd indices the candidate
on row index // 2 (run_rollout_task.matched_index is the single source of
truth), so one array holds both policies and, throttled at 8 concurrent,
every batch runs four of each on the same queue at the same hour. run1 and
run2 are submitted by the same tick. N is fixed at two runs per policy per
horizon; nothing in this file can add a run.

Infrastructure is v4's, each piece fixed there after failing live: a chain
tick that fires when any waited job ends (Slurm `?` OR-dependency), a
2-hour watchdog tick, a lock that recognises a dead holder, stage adoption
from the ledger, one retry of infrastructure-failed rows, the real sbatch
binary (the site wrapper word-splits its arguments), and relative --begin
(absolute times are read in cluster-local time).

    driver.py --campaign ROOT tick      advance whatever can advance, schedule the next tick
    driver.py --campaign ROOT status    print the resumable state
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rollout_task import POLICIES, destination, matched_index  # noqa: E402

REAL_SBATCH = "/apps/slurm/current/bin/sbatch"
PYTHON = "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python"
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING",
          "RESIZING", "SUSPENDED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
          "STAGE_OUT"}
# A tick's Slurm allocation is 20 minutes, so any lock older than this is
# certainly abandoned even when its holder cannot be checked directly.
LOCK_STALE_SECONDS = 25 * 60
RUNS = ("r1", "r2")
HORIZONS = (10, 50)


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
    """The preregistered rule on runs measured in the SAME campaign.

    cand / base: {"h10": [run, run], "h50": [run, run]}, each run carrying
    settled / valid / rows. The H10 clauses are v2-v4's, unchanged: pooled
    16 vs 16, margin >= 4 AND one-sided Fisher p < 0.05. The H50 clause is
    v4's non-regression made matched -- pooled candidate H50 >= pooled
    baseline H50, both measured here -- in one wording fixed before any v5
    result. Retention guard and reload are v4's recorded facts about the
    checkpoint. Pure so it is testable.
    """
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
        return {"schema": 1, "created_utc": stamp(), "stages": {}, "scores": {}, "verdict": None,
                "final": {}, "gallery": None, "ticks": {}, "done": False, "stop_reason": None, "log": []}

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

    # ---- budget (anti-runaway bounds; the user lifted the hour constraint)
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

    # ---- results
    def read_result(self, directory: Path) -> dict | None:
        sp, rp = directory / "status.json", directory / "rollout.json"
        if not sp.is_file() or json.loads(sp.read_text()).get("state") != "completed" or not rp.is_file():
            return None
        r = json.loads(rp.read_text())
        c = r.get("terminal_checker", {})
        return {"terminal_success": bool(r.get("terminal_success")), "ever_success": bool(r.get("success")),
                "conditions_passed": c.get("conditions_passed"), "conditions_total": c.get("conditions_total"),
                "horizon": r.get("effective_n_action_steps")}

    def label(self, which: str) -> str:
        return self.manifest[f"{which}_checkpoint"]["label"]

    def dest(self, phase: str, index: int, run: str, rows: list[dict]) -> Path:
        which, r = matched_index(index)
        return destination(self.root, phase, rows[r], f"{self.label(which)}-{run}")

    def matched_stage(self, key: str, phase: str, run: str, rows: list[dict], reserve: int = 0) -> str:
        """Submit ONE interleaved array over both policies, wait, retry
        infrastructure failures once. Returns wait / done / incomplete / skipped."""
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
        retry = f"{key}.retry1"
        if self.stage(retry) is None:
            for i in bad:
                src = self.dest(phase, i, run, rows)
                if src.exists():
                    dst = self.root / "attempts" / src.relative_to(self.root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst) + ".attempt1")
            self.event(f"{key}: retrying infrastructure-failed indices {bad}")
            self.submit(retry, "matched.sbatch", args, gpu_tasks=len(bad),
                        array=",".join(map(str, bad)), reserve=reserve)
            return "wait"
        if self.stage_status(retry) == "active":
            return "wait"
        return "done" if all(self.read_result(self.dest(phase, i, run, rows)) for i in indices) else "incomplete"

    def score_run(self, phase: str, run: str, rows: list[dict], which: str, horizon: int) -> dict:
        """Settled terminal success for one policy, run and horizon.
        Infrastructure failures are NOT policy failures: counted apart, excluded."""
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

    def scores_now(self, phase: str = "benchmark", runs=RUNS) -> dict:
        rows = self.manifest[phase]
        return {which: {f"h{h}": [self.score_run(phase, run, rows, which, h) for run in runs]
                        for h in HORIZONS}
                for which in POLICIES}

    # ---- the campaign
    def stop(self, reason: str) -> str:
        self.state["stop_reason"] = reason
        self.state["done"] = True
        self.event("stopped: " + reason)
        return "stopped"

    def advance(self) -> str:
        M = self.manifest
        reserve = self.reserve_final()
        st = self.matched_stage("smoke", "smoke", "s1", M["smoke"], reserve=reserve)
        if st == "wait":
            return "wait"
        if st != "done":
            return self.stop(f"smoke did not complete ({st}); no development run is submitted")
        dev = M["benchmark"]
        s1 = self.matched_stage("run1", "benchmark", "r1", dev, reserve=reserve)
        s2 = self.matched_stage("run2", "benchmark", "r2", dev, reserve=reserve)
        if "wait" in (s1, s2):
            return "wait"
        self.state["scores"] = self.scores_now()
        if self.state.get("verdict") is None:
            prov = M["candidate_checkpoint"]["provenance"]
            v = matched_verdict(self.state["scores"]["candidate"], self.state["scores"]["baseline"],
                                prov["retention_guard"]["passed"], prov["reload"]["passed"])
            v["run_status"] = {"run1": s1, "run2": s2}
            self.state["verdict"] = v
            self.event("verdict: " + ("IMPROVED under the preregistered rule" if v["improved"]
                                      else "not improved under the preregistered rule"),
                       candidate_h10=v["candidate_h10"], baseline_h10=v["baseline_h10"],
                       candidate_h50=v["candidate_h50"], baseline_h50=v["baseline_h50"],
                       p_value=round(v["p_value"], 4), failed=[k for k, ok in v["checks"].items() if not ok])
        v = self.state["verdict"]
        fin = self.state["final"]
        winner = "candidate" if v["improved"] else "baseline"
        if v["improved"]:
            sf = self.matched_stage("final", "final_test", "f1", M["final_test"])
            if sf == "wait":
                return "wait"
            fin["status"] = sf
            for which in POLICIES:
                fin[which] = self.score_run("final_test", "f1", M["final_test"], which, 10)
        else:
            fin["skipped"] = "the preregistered rule did not hold; the final set stays unspent"
        fin["selected"] = {"label": self.label(winner), "checkpoint": M[f"{winner}_checkpoint"]["path"]}
        if not self.dry:
            self.gallery()
        fin["completed_utc"] = stamp()
        self.state["done"] = True
        self.event("campaign complete", selected=self.label(winner))
        return "done"

    def gallery(self) -> None:
        try:
            import build_gallery
            self.state["gallery"] = build_gallery.build(self.root)
            self.event("gallery built", settled=self.state["gallery"]["settled"],
                       valid=self.state["gallery"]["valid"])
        except Exception as exc:  # a gallery defect must not block the record
            self.state["gallery"] = {"error": repr(exc)}
            self.event(f"gallery failed: {exc!r}")

    def tick(self) -> None:
        if not self.state["done"]:
            self.advance()
        if self.dry:
            return
        self.update_status(*self.describe())
        self.schedule_ticks()
        self.save()

    # ---- ticks and status (as v4)
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
        if self.stage("run1") is not None:
            sc = self.state.get("scores") or self.scores_now()
            for which in POLICIES:
                bits = []
                for h in HORIZONS:
                    runs = sc[which][f"h{h}"]
                    bits.append(f"H{h} " + "+".join(f"{r['settled']}/{r['valid']}" for r in runs))
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
    """Exclusive tick lock that recognises a dead holder immediately (v2 tick
    21400711 died by SIGTERM and would otherwise have blocked every tick for
    the full staleness window)."""

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
