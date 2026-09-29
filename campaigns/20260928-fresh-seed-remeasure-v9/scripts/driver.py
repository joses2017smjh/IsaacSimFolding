"""Restartable, duplicate-safe driver for v9: the fresh-seed, GPU-pinned re-measurement.

Measurement only. No training, no checkpoint choice and no final set. The
untouched baseline stays the deliverable whatever the result.

Stage graph (every stage keyed in ledger/driver_state.json and never
resubmitted while its key holds a live or finished job):

    preflight       verify_sources.py over every pinned source, both pinned
                    checkpoints and the pose metadata, before any submission
    block:<model>   ONE interleaved matched array per platform block, pinned by
                    a Slurm GPU-model --constraint (a40 primary, rtx8000
                    secondary): 48 fresh-seed rows x {baseline, candidate},
                    even index = baseline, odd = candidate on row index // 2     96 each
    retry           one outcome-blind resubmission per block of every task whose
                    status is not 'completed' (no status/rollout, infrastructure
                    error incl. wrong GPU model or validation failure, timeout),
                    after re-running verify_sources
    results         scripts/analysis.py per block as soon as it ends, and the
                    preregistered classification once every block is terminal

The deadline (manifest protocol.block_deadline_utc) is enforced on each
episode's own completion time, not on tick time: an episode that completed
later is invalid, and a block still running at the deadline has its jobs
cancelled and is reported as incomplete. Any other cancellation is logged
as a protocol deviation.

Infrastructure is v4-v8's: chain ticks on Slurm `?` OR-dependencies, a
2-hour watchdog, a lock that recognises a dead holder, stage adoption from
the ledger, the real sbatch binary and a relative --begin.

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
import analysis  # noqa: E402

REAL_SBATCH = "/apps/slurm/current/bin/sbatch"
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING",
          "RESIZING", "SUSPENDED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
          "STAGE_OUT"}
LOCK_STALE_SECONDS = 25 * 60
PHASE = "remeasure"
TERMINAL = ("done", "incomplete", "skipped")


# ------------------------------------------------------------------ time
def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp(t: dt.datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


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


def sbatch_retry(args: list[str]) -> str:
    """sbatch once more after a short pause before giving up (transient controller errors)."""
    try:
        return sbatch(args)
    except RuntimeError:
        time.sleep(20)
        return sbatch(args)


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
        self.manifest_sha = file_sha(self.root / "manifest.json")
        self.caps = self.manifest["budget"]
        self.state_path = self.root / "ledger/driver_state.json"
        self.ledger_path = self.root / "ledger/slurm-jobs.json"
        if not self.ledger_path.exists() and not dry:
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
        return {"schema": 1, "created_utc": stamp(), "stages": {}, "preflight": None, "blocks": {},
                "retried": {}, "deviations": [], "results": None, "final": {}, "ticks": {}, "done": False,
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
        if not self.ledger_path.is_file():
            return {"campaign": self.manifest["campaign"], "jobs": []}
        return json.loads(self.ledger_path.read_text())

    def record(self, job, stage, script, gpu_tasks, **extra) -> None:
        data = self.ledger()
        data["jobs"].append({"job_id": job, "phase": stage, "script": script, "gpu_tasks": gpu_tasks,
                             "submitted_by": "driver", "manifest_commit": self.manifest["git"]["commit"],
                             "manifest_sha256": self.manifest_sha,
                             "submitted_utc": stamp(), "state": "submitted", **extra})
        self.ledger_path.write_text(json.dumps(data, indent=2) + "\n")

    # ---- budget (anti-runaway bounds)
    def gpu_tasks_used(self) -> int:
        return sum(int(j.get("gpu_tasks", 0) or 0) for j in self.ledger()["jobs"])

    def can_spend(self, tasks: int) -> bool:
        return self.gpu_tasks_used() + tasks <= int(self.caps["gpu_tasks"])

    def within_gpu_hours(self) -> bool:
        return gpu_hours([j["job_id"] for j in self.ledger()["jobs"]]) <= float(self.caps["gpu_hours"])

    # ---- stages
    def stage(self, key):
        return self.state["stages"].get(key)

    def stage_status(self, key: str) -> str:
        st = self.stage(key)
        if st is None:
            return "absent"
        if st.get("status") in ("completed", "failed", "skipped", "cancelled"):
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

    def submit(self, key, script, args, *, gpu_tasks, array=None, sbatch_extra=(), **extra):
        if self.stage(key) is not None:
            return None
        if gpu_tasks and not self.can_spend(gpu_tasks):
            self.event(f"budget refuses {key}: {gpu_tasks} GPU tasks")
            self.state["stages"][key] = {"job_ids": [], "status": "skipped", "reason": "gpu task budget"}
            return None
        if gpu_tasks and not self.within_gpu_hours():
            self.event(f"budget refuses {key}: GPU-hour cap reached")
            self.state["stages"][key] = {"job_ids": [], "status": "skipped", "reason": "gpu hour budget"}
            return None
        logs = self.root / "logs"
        name = key.replace(":", "_")
        argv = [f"--chdir={self.root}",
                f"--output={logs}/{name}-%A_%a.out" if array else f"--output={logs}/{name}-%j.out",
                *sbatch_extra]
        if array:
            argv.append(f"--array={array}%8" if gpu_tasks else f"--array={array}")
        argv += [str(self.root / "slurm" / script), *args]
        if self.dry:
            job = f"DRY-{key}"
            print(f"[dry-run] would submit {key}: sbatch {' '.join(argv)}", flush=True)
        else:
            logs.mkdir(exist_ok=True)
            job = sbatch(argv)
            self.record(job, key, f"slurm/{script}", gpu_tasks, array=array, sbatch_extra=list(sbatch_extra), **extra)
        self.state["stages"][key] = {"job_ids": [job], "submitted_utc": stamp(), "status": "submitted",
                                     "gpu_tasks": gpu_tasks, "array": array, **extra}
        self.waiting.append(job)
        self.event(f"submitted {key} -> {job}", gpu_tasks=gpu_tasks, sbatch_extra=list(sbatch_extra))
        if not self.dry:
            self.save()
        return job

    def read_result(self, directory: Path) -> dict | None:
        """A task counts only when it completed, wrote its rollout, and finished
        before the block deadline; the task itself marks a wrong-model run as
        an infrastructure error."""
        sp, rp = directory / "status.json", directory / "rollout.json"
        if not sp.is_file() or not rp.is_file():
            return None
        try:
            status = json.loads(sp.read_text())
        except ValueError:
            return None
        if status.get("state") != "completed" or not status.get("completed_utc"):
            return None
        if parse_utc(status["completed_utc"]) > parse_utc(self.manifest["protocol"]["block_deadline_utc"]):
            return None
        return {"ok": True}

    @staticmethod
    def task_state(directory: Path) -> dict:
        try:
            st = json.loads((directory / "status.json").read_text())
            return {"state": st.get("state"), "error": st.get("error")}
        except (OSError, ValueError):
            return {"state": "no status.json", "error": None}

    # ---- matched evaluation, one pinned GPU-model block
    def label(self, which: str) -> str:
        return checkpoint_block(self.root, self.manifest, which)["label"]

    def dest(self, index: int, block: dict, rows: list[dict]) -> Path:
        which, r = matched_index(index)
        return destination(self.root, PHASE, rows[r], f"{self.label(which)}-{block['key']}")

    def past_deadline(self) -> bool:
        return now() >= parse_utc(self.manifest["protocol"]["block_deadline_utc"])

    def cancel_block(self, keys: list[str]) -> None:
        for key in keys:
            st = self.stage(key)
            if st and st.get("status") == "submitted":
                for job in st["job_ids"]:
                    if not self.dry:
                        scancel(job)
                st["status"] = "cancelled"
                st["finished_utc"] = stamp()
                self.event(f"{key}: cancelled at the block deadline")

    def block_stage(self, block: dict, rows: list[dict]) -> str:
        """ONE interleaved array over both policies on one GPU model.
        Returns wait / done / incomplete / skipped."""
        key, retry = f"block:{block['key']}", f"block:{block['key']}.retry1"
        n = 2 * len(rows)
        indices = list(range(n))
        args = [str(self.root), PHASE, block["key"], block["key"], self.manifest_sha]
        extra = [f"--constraint={block['constraint']}"]
        if self.stage(key) is None:
            self.submit(key, "matched.sbatch", args, gpu_tasks=n, array=f"0-{n - 1}", sbatch_extra=extra,
                        block=block["key"])
            return "skipped" if self.stage(key).get("status") == "skipped" else "wait"
        if self.stage(key).get("status") == "skipped":
            return "skipped"
        live = [k for k in (key, retry) if self.stage(k) is not None and self.stage_status(k) == "active"]
        if live:
            if self.past_deadline():
                self.cancel_block(live)
            else:
                return "wait"
        self.note_external_cancellations([key, retry])
        bad = [i for i in indices if self.read_result(self.dest(i, block, rows)) is None]
        if not bad:
            return "done"
        if self.stage(retry) is None and not self.past_deadline():
            if not self.verify_before_retry(block["key"]):
                return "incomplete"
            record = {}
            for i in bad:
                src = self.dest(i, block, rows)
                record[str(i)] = {"dir": str(src.relative_to(self.root)), **self.task_state(src)}
                if src.exists():
                    dst = self.root / "attempts" / src.relative_to(self.root)
                    if self.dry:
                        print(f"[dry-run] would move {src} -> {dst}.attempt1", flush=True)
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst) + ".attempt1")
            self.state["retried"][block["key"]] = record
            self.event(f"{key}: one outcome-blind retry of {len(bad)} tasks whose status is not completed",
                       indices=bad, states=sorted({r["state"] or "?" for r in record.values()}))
            self.submit(retry, "matched.sbatch", args, gpu_tasks=len(bad), array=",".join(map(str, bad)),
                        sbatch_extra=extra, block=block["key"])
            return "wait"
        return "incomplete"

    def note_external_cancellations(self, keys: list[str]) -> None:
        """A CANCELLED task the driver did not cancel (at the deadline) is a protocol deviation."""
        for key in keys:
            st = self.stage(key)
            if not st or st.get("status") == "cancelled" or st.get("deviation_checked") or not st.get("job_ids"):
                continue
            if self.dry or str(st["job_ids"][0]).startswith("DRY"):
                continue
            states = [s for j in st["job_ids"] for s in job_states(st["job_ids"]).get(j, [])]
            cancelled = sum(s.split()[0].rstrip("+") == "CANCELLED" for s in states)
            st["deviation_checked"] = True
            if cancelled:
                self.state["deviations"].append({"utc": stamp(), "stage": key, "cancelled_tasks": cancelled,
                                                 "note": "cancelled outside the driver's deadline rule"})
                self.event(f"PROTOCOL DEVIATION: {cancelled} task(s) of {key} were cancelled outside the driver")

    def verify_before_retry(self, block_key: str) -> bool:
        """Re-run verify_sources before any retry submission; the preflight is days old by then."""
        if self.dry:
            return True
        out = self.root / "audit" / f"verify_sources_retry_{block_key}.json"
        proc = subprocess.run([sys.executable, str(self.root / "scripts/verify_sources.py"),
                               "--campaign", str(self.root), "--out", str(out)], capture_output=True, text=True)
        ok = proc.returncode == 0
        self.event(f"verify_sources before the {block_key} retry passed={ok}")
        if not ok:
            self.state["deviations"].append({"utc": stamp(), "stage": f"block:{block_key}.retry1",
                                             "note": "retry not submitted: sources or checkpoints drifted",
                                             "report": str(out)})
        return ok

    # ---- the campaign
    def preflight(self) -> bool:
        if self.state.get("preflight") is not None:
            return bool(self.state["preflight"]["passed"])
        out = self.root / "audit" / "verify_sources_launch.json"
        repo = self.root.parents[1]
        rel = str((self.root / "manifest.json").relative_to(repo))
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel],
                               capture_output=True).returncode == 0
        if not (tracked and clean):
            if self.dry:
                print(f"[dry-run] preflight would fail: manifest tracked={tracked} clean={clean}", flush=True)
                return True
            self.state["preflight"] = {"passed": False, "utc": stamp(), "manifest_sha256": self.manifest_sha,
                                       "problems": [f"manifest.json tracked={tracked} clean={clean}"]}
            self.event(f"preflight: manifest.json must be committed and clean (tracked={tracked}, clean={clean})")
            return False
        cmd = [sys.executable, str(self.root / "scripts/verify_sources.py"), "--campaign", str(self.root)]
        if not self.dry:
            cmd += ["--out", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        passed = proc.returncode == 0
        if self.dry:
            print(f"[dry-run] preflight verify_sources exit {proc.returncode}", flush=True)
            return passed
        report = json.loads(out.read_text()) if out.is_file() else {"problems": [proc.stderr[-2000:]]}
        self.state["preflight"] = {"passed": passed, "report": str(out), "utc": stamp(),
                                   "manifest_sha256": self.manifest_sha, "problems": report.get("problems", [])}
        self.event(f"preflight verify_sources passed={passed}")
        return passed

    def labels(self) -> dict[str, str]:
        return {which: self.label(which) for which in POLICIES}

    def record_block(self, block: dict, status: str) -> None:
        if self.state["blocks"].get(block["key"], {}).get("status") == status:
            return
        res = analysis.block_results(self.root, self.manifest, block, self.labels())
        out = self.root / "analysis" / "remeasure" / f"block-{block['key']}.json"
        if not self.dry:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1) + "\n")
        self.state["blocks"][block["key"]] = {
            "status": status, "results": str(out), "complete": res["complete"], "valid": res["valid"],
            "role": res["role"],
            "h10": {k: res["h10"][k] for k in ("candidate", "baseline", "margin", "fisher_one_sided_p",
                                              "row_stratified_exact_p", "row_sign_flip_p")},
            "h50": {k: res["h50"][k] for k in ("candidate", "baseline", "margin", "dev01")},
            "rule": res["rule"], "platforms": res["platforms"]}
        self.event(f"block {block['key']} ({res['role']}) {status}: H10 {res['h10']['candidate']} vs {res['h10']['baseline']} "
                   f"(p={res['h10']['fisher_one_sided_p']:.4f}); H50 {res['h50']['candidate']} vs "
                   f"{res['h50']['baseline']}; complete={res['complete']}")

    def finish(self, reason: str | None) -> str:
        fin = self.state["final"]
        block = checkpoint_block(self.root, self.manifest, "baseline")
        fin["deliverable"] = {"label": block["label"], "checkpoint": block["path"],
                              "why": "v9 is measurement only; the deliverable is unchanged by design"}
        fin["final_set"] = "unspent: v9 does not touch the final set"
        if reason:
            self.state["stop_reason"] = reason
        fin["completed_utc"] = stamp()
        self.state["done"] = True
        self.event("campaign complete" + (f": {reason}" if reason else ""))
        return "done"

    def advance(self) -> str:
        if not self.preflight():
            return self.finish("preflight source/checkpoint verification failed; nothing was submitted")
        rows = self.manifest["remeasure"]
        status = {}
        for block in self.manifest["blocks"]:
            status[block["key"]] = self.block_stage(block, rows)
            if status[block["key"]] in ("done", "incomplete"):
                self.record_block(block, status[block["key"]])
        if any(s not in TERMINAL for s in status.values()):
            return "wait"
        finished = [k for k, s in status.items() if s in ("done", "incomplete")]
        res = analysis.compute(self.root, self.manifest, self.labels(), finished)
        out = self.root / "analysis" / "remeasure" / "results.json"
        if not self.dry:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1) + "\n")
        self.state["results"] = {"path": str(out), "classification": res["classification"],
                                 "combined": res["combined"], "block_status": status}
        self.event(f"classification: {res['classification']['headline']}",
                   v5_rule_holds_on=res["classification"]["v5_rule_holds_on"])
        return self.finish(None)

    def tick(self) -> None:
        error = None
        if not self.state["done"]:
            try:
                self.advance()
            except Exception as exc:  # keep the chain alive; re-raise after saving
                error = exc
                self.event(f"tick error: {exc!r}")
        if self.dry:
            if error:
                raise error
            return
        try:
            self.update_status(*self.describe())
        finally:
            try:
                self.schedule_ticks()
            finally:
                self.save()
        if error:
            raise error

    # ---- ticks and status (as v4-v8)
    def schedule_ticks(self) -> None:
        ticks = self.state["ticks"]
        waiting = sorted(set(self.waiting))
        me = os.environ.get("SLURM_JOB_ID")
        chain = ticks.get("chain")
        pending = chain and chain != me and \
            [s for v in job_states([chain]).values() for s in v][:1] == ["PENDING"]
        keep = pending and ticks.get("chain_deps") == waiting and \
            ticks.get("chain_begin") == (stamp(self.begin_at) if self.begin_at else None)
        (self.root / "ledger/ticks").mkdir(parents=True, exist_ok=True)
        errors = []
        if not keep and (waiting or self.begin_at) and not self.state["done"]:
            argv = [f"--chdir={self.root}", f"--output={self.root}/ledger/ticks/tick-%j.out"]
            if waiting:
                argv.append("--dependency=" + "?".join(f"afterany:{j}" for j in waiting))
            if self.begin_at:
                argv.append(f"--begin={relative_begin(self.begin_at)}")
            try:   # the replacement is secured BEFORE the old chain is cancelled
                new = sbatch_retry(argv + [str(self.root / "slurm/tick.sbatch"), str(self.root)])
                if pending:
                    scancel(chain)
                ticks["chain"], ticks["chain_deps"] = new, waiting
                ticks["chain_begin"] = stamp(self.begin_at) if self.begin_at else None
            except RuntimeError as exc:
                errors.append(exc)
                self.event(f"chain tick submission failed: {exc}")
        watchdog = ticks.get("watchdog")
        alive = watchdog and watchdog != me and \
            summarize([s for v in job_states([watchdog]).values() for s in v]) == "active"
        if not alive and not self.state["done"]:
            try:
                ticks["watchdog"] = sbatch_retry([f"--chdir={self.root}",
                                                  f"--output={self.root}/ledger/ticks/watchdog-%j.out",
                                                  f"--begin={relative_begin(now() + dt.timedelta(hours=2))}",
                                                  str(self.root / "slurm/tick.sbatch"), str(self.root)])
            except RuntimeError as exc:
                errors.append(exc)
                self.event(f"watchdog submission failed: {exc}")
        if errors:
            raise errors[0]

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
        pf = self.state.get("preflight")
        if pf:
            parts.append("preflight " + ("passed" if pf["passed"] else "FAILED"))
        for block in self.manifest["blocks"]:
            b = self.state["blocks"].get(block["key"])
            if b:
                parts.append(f"{block['key']} [{b.get('role')}] ({b['status']}): H10 {b['h10']['candidate']} vs {b['h10']['baseline']} "
                             f"(p={b['h10']['fisher_one_sided_p']:.3f}), H50 {b['h50']['candidate']} vs "
                             f"{b['h50']['baseline']}")
            elif self.stage(f"block:{block['key']}") is not None:
                parts.append(f"{block['key']}: queued or running")
        res = self.state.get("results")
        if res:
            parts.append(res["classification"]["headline"])
        if self.state.get("deviations"):
            parts.append(f"{len(self.state['deviations'])} protocol deviation(s) logged")
        if self.state["done"]:
            parts.append("deliverable unchanged: baseline")
        active = ", ".join(live) if live else ("none — campaign complete" if self.state["done"] else "none")
        waiting = []
        for block in self.manifest["blocks"]:
            for key in (f"block:{block['key']}", f"block:{block['key']}.retry1"):
                st = self.stage(key)
                if st and st.get("status") == "submitted" and st.get("job_ids") and not self.dry:
                    out = subprocess.run(["squeue", "-h", "-j", st["job_ids"][0], "-o", "%T %r"],
                                         capture_output=True, text=True).stdout.split("\n")
                    pend = sorted({ln.strip() for ln in out if ln.strip().startswith("PENDING")})
                    run = sum(ln.strip().startswith("RUNNING") for ln in out)
                    waiting.append(f"{key}: {run} running" + (f", pending ({'; '.join(pend)})" if pend else ""))
        blocker = self.state.get("stop_reason") or ("; ".join(waiting) if waiting else "none")
        nxt = ("none — campaign complete; see REPORT.md" if self.state["done"]
               else "driver advances when a waited job ends (2-hour watchdog)")
        return active, "; ".join(parts) or "not started", blocker, nxt


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
    if args.dry_run:
        lock = Lock(root / "ledger/.driver.lock")
        if lock.path.exists() and lock.holder_alive():
            raise SystemExit("a live tick holds the driver lock; not dry-running concurrently")
        Driver(root, dry=True).tick()
        return 0
    (root / "ledger").mkdir(parents=True, exist_ok=True)
    with Lock(root / "ledger/.driver.lock"):
        driver = Driver(root)
        driver.tick()
    return 0


if __name__ == "__main__":
    sys.exit(main())
