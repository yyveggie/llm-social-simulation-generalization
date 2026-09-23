from __future__ import annotations

import asyncio
import codecs
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from . import llm_proxy, progress, settings
from .adapters.base import JobUnit
from .registry import get_adapter
from .schemas import JobInfo, JobProgress, JobState, JobUnitState, LaunchRequest, SampleStats, UnifiedLLM

_END_SENTINEL = "\x00__JOB_END__\x00"
_IS_POSIX = os.name == "posix"
_STATE_FILE = "job_state.json"
_ACTIVE_STATUSES = {"running", "paused"}
_COST_CIRCUIT_MARKERS = (
    "aborting to avoid runaway API cost",
    "circuit breaker tripped",
    "成本熔断保护",
)
_COST_CIRCUIT_MESSAGE = (
    "成本熔断保护已停止：历史结果中的空响应/API 请求失败超过阈值；"
    "触发后不会继续发送请求。"
)


_SAMPLE_STATS_TTL = 15.0
_SAMPLE_STATS_BACKFILL_INTERVAL = 20.0
_SAMPLE_STATS_BACKFILL_BATCH_SIZE = 8
_SAMPLE_STATS_BACKFILL_DELAY = 0.5
_SAMPLE_STATS_BACKFILL_TIMEOUT = 90.0


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return max(minimum, default)


_JOB_RECOVERY_SYNC_LIMIT = _env_int("ORCH_JOB_RECOVERY_SYNC_LIMIT", 24, 0)
_LLM_SNAPSHOT_FIELDS = (
    "provider_kind",
    "base_url",
    "api_key",
    "model",
    "temperature",
    "max_tokens",
    "system",
    "concurrency",
    "timeout",
    "max_retries",
)


class Job:
    def __init__(self, info: JobInfo) -> None:
        self.info = info
        self.proc: Optional[subprocess.Popen] = None
        self.lines: deque[str] = deque(maxlen=8000)
        self.lock = threading.Lock()
        self.subscribers: set[asyncio.Queue] = set()
        self.reader: Optional[threading.Thread] = None
        self._log_fh = None
        self._stopping = False
        self.state_lock = threading.Lock()
        self.persist_lock = threading.Lock()
        self.last_transient = False
        self.progress_base = 0
        self.progress_total = 0
        self.sample_stats_ts = 0.0
        self.log_backlog_loaded = False


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.launch_lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.sample_stats_backfill_lock = threading.Lock()
        self.sample_stats_backfill_thread: Optional[threading.Thread] = None
        self.sample_stats_backfill_last = 0.0
        self.sample_stats_backfill_skipped: set[str] = set()
        self.live_stats_lock = threading.Lock()
        self.live_stats_pending: set[str] = set()
        self.live_stats_thread: Optional[threading.Thread] = None
        self.recovery_partial = False
        self.known_job_ids: set[str] = set()
        self.recovery_thread: Optional[threading.Thread] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.load_persisted_jobs()
        self.ensure_sample_stats_backfill(force=True)

    def load_persisted_jobs(self) -> None:
        root = settings.RUNS_DIR
        if not root.exists():
            return
        restored: list[Job] = []


        persist_records: dict[str, tuple[JobUnit | None, dict[str, Any]]] = {}
        state_paths = list(root.glob(f"*/{_STATE_FILE}"))
        self.known_job_ids.update(p.parent.name for p in state_paths)
        state_paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        if _JOB_RECOVERY_SYNC_LIMIT and len(state_paths) > _JOB_RECOVERY_SYNC_LIMIT:
            self.recovery_partial = True
            remaining_state_paths = state_paths[_JOB_RECOVERY_SYNC_LIMIT:]
            state_paths = state_paths[:_JOB_RECOVERY_SYNC_LIMIT]
        else:
            remaining_state_paths = []
        for state_path in state_paths:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                repaired = _repair_moved_paths(state, state_path.parent)
                info = JobState.model_validate(state).info
            except Exception:
                continue
            info_before_recovery = info.model_dump()
            _normalize_cost_circuit_info(info)
            with self.lock:
                if info.id in self.jobs:
                    continue
            recovery_note = None
            if info.status in _ACTIVE_STATUSES:
                info.finished_at = None
                info.exit_code = None
                old_pid = info.pid
                if info.status == "running" and _pid_alive(old_pid) and _pid_looks_like_job(old_pid, info.command):
                    recovery_note = (
                        "[orchestrator] 后端重启：检测到原实验子进程仍在运行，保留为 running；"
                        "新后端继续提供 LLM proxy，日志流从重启后继续。"
                    )
                else:
                    info.pid = None
                    was_running = info.status == "running"
                    info.status = "paused"
                    if was_running or info.paused_at is None:
                        _mark_paused(info)
                    if was_running:
                        recovery_note = (
                            "[orchestrator] 后端重启：未检测到原 running 子进程，已恢复为 paused；"
                            "点「继续」按已有产物续跑。"
                        )
            job = Job(info)
            if repaired:
                repair_note = (
                    "[orchestrator] 检测到仓库目录已移动/重命名：已把任务状态中的旧绝对路径"
                    "改写到当前位置。"
                )
                job.lines.append(repair_note)
                _append_log_line(Path(info.log_path), repair_note)
                if state.get("unit"):
                    try:
                        repaired_unit = _unit_from_state(state)
                    except RuntimeError:
                        repaired_unit = None
                else:
                    repaired_unit = None
            else:
                repaired_unit = None
            if recovery_note:
                job.lines.append(recovery_note)
                _append_log_line(Path(info.log_path), recovery_note)
            if repaired or info.model_dump() != info_before_recovery:
                persist_records[info.id] = (repaired_unit, state)
            restored.append(job)

        if not restored:
            if remaining_state_paths:
                self._start_background_recovery(remaining_state_paths)
            return
        with self.lock:
            for job in restored:
                self.jobs.setdefault(job.info.id, job)
        for job in restored:
            record = persist_records.get(job.info.id)
            if record is None:
                continue


            unit, existing_state = record
            self._persist_job_state(job, unit=unit, existing_state=existing_state)
        if remaining_state_paths:
            self._start_background_recovery(remaining_state_paths)

    def _start_background_recovery(self, state_paths: list[Path]) -> None:
        if self.recovery_thread and self.recovery_thread.is_alive():
            return
        self.recovery_thread = threading.Thread(
            target=self._background_recover_jobs,
            args=(state_paths,),
            name="job-state-recovery",
            daemon=True,
        )
        self.recovery_thread.start()

    def _background_recover_jobs(self, state_paths: list[Path]) -> None:
        for state_path in state_paths:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                repaired = _repair_moved_paths(state, state_path.parent)
                info = JobState.model_validate(state).info
            except Exception:
                continue
            info_before_recovery = info.model_dump()
            _normalize_cost_circuit_info(info)
            with self.lock:
                if info.id in self.jobs:
                    continue
            if info.status in _ACTIVE_STATUSES:
                info.finished_at = None
                info.exit_code = None
                old_pid = info.pid
                if not (info.status == "running" and _pid_alive(old_pid) and _pid_looks_like_job(old_pid, info.command)):
                    info.pid = None
                    was_running = info.status == "running"
                    info.status = "paused"
                    if was_running or info.paused_at is None:
                        _mark_paused(info)
            job = Job(info)
            unit = None
            if repaired and state.get("unit"):
                try:
                    unit = _unit_from_state(state)
                except RuntimeError:
                    unit = None
            with self.lock:
                if info.id in self.jobs:
                    continue
                self.jobs[info.id] = job
            if repaired or info.model_dump() != info_before_recovery:
                self._persist_job_state(job, unit=unit, existing_state=state)
        self.recovery_partial = False


    def launch(self, req: LaunchRequest, llm: UnifiedLLM, raw_cfg: dict[str, Any]) -> list[JobInfo]:


        if self.recovery_partial:
            raise ValueError("历史任务正在后台恢复，请稍后再启动新实验。")
        adapter = get_adapter(req.project_id)


        with self.launch_lock:


            if not adapter.allow_concurrent and self._has_active_job(req.project_id):
                raise ValueError(
                    f"项目「{adapter.name}」已有任务正在运行或暂停中。该项目复用唯一的原生"
                    "配置文件，请等当前任务结束/停止后再启动新任务。"
                )

            units = adapter.plan_jobs(req.experiment_ids, req.params)
            if not units:
                raise ValueError("没有可启动的实验单元（请至少选择一个实验）。")
            py = settings.python_executable(raw_cfg)

            infos: list[JobInfo] = []
            for unit in units:
                jid = uuid4().hex[:12]
                run_dir = settings.RUNS_DIR / jid
                run_dir.mkdir(parents=True, exist_ok=True)

                adapter.render_config(llm, req.params, unit, work_dir=run_dir)
                job = self._spawn(adapter, unit, req, py, jid, run_dir, model=llm.model, llm=llm)
                infos.append(job.info)
            return infos

    def _has_active_job(self, project_id: str) -> bool:
        with self.lock:
            return any(
                j.info.project_id == project_id and j.info.status in ("running", "paused")
                for j in self.jobs.values()
            )

    def _has_other_active_job(self, project_id: str, job_id: str) -> bool:
        with self.lock:
            return any(
                j.info.id != job_id
                and j.info.project_id == project_id
                and j.info.status in ("running", "paused")
                for j in self.jobs.values()
            )

    def _spawn(self, adapter, unit, req: LaunchRequest, py: str, jid: str, run_dir: Path,
               model: str = "", llm: UnifiedLLM | None = None) -> Job:
        cwd = adapter.project_dir
        require_llm = unit.experiment_id != "analyze"
        argv = _prepare_spawn_argv(adapter, unit, require_llm=require_llm)
        unit_py = str(unit.extra.get("python_executable") or py)
        command = [unit_py, *argv]

        log_path = run_dir / "log.txt"

        info = JobInfo(
            id=jid,
            project_id=adapter.id,
            project_name=adapter.name,
            paper=getattr(adapter, "paper", ""),
            experiment_id=unit.experiment_id,
            label=unit.label,
            model=model,
            params={k: v for k, v in (req.params or {}).items() if k != "__nonce"},
            status="running",
            started_at=time.time(),
            command=command,
            cwd=str(cwd),
            results_dir=str(adapter.results_dir()),
            log_path=str(log_path),
        )

        job = Job(info)
        job.log_backlog_loaded = True
        self._persist_job_state(job, unit=unit, llm=llm)
        with self.lock:
            self.jobs[jid] = job

        self._start_process(job, unit, append_log=False)
        return job

    def _start_process(self, job: Job, unit: JobUnit, *, append_log: bool) -> None:
        info = job.info
        command = list(info.command)
        cwd = info.cwd
        llm_proxy.attach_tokens_to_job(
            [unit.extra.get("llm_proxy_token"), unit.extra.get("llm_proxy_token_judge")], info.id
        )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.update({str(k): str(v) for k, v in unit.extra.get("env", {}).items()})

        popen_kwargs: dict[str, Any] = dict(
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if _IS_POSIX:
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        log_path = Path(info.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append_log else "w"
        job._log_fh = open(log_path, mode, encoding="utf-8")
        if append_log:
            job._log_fh.write("\n# 恢复续跑，重新启动子进程\n")
        job._log_fh.write("$ " + " ".join(command) + "\n")
        job._log_fh.write(f"# cwd: {cwd}\n\n")
        job._log_fh.write(f"# job_id: {info.id}\n")
        job._log_fh.write(f"# project: {info.project_name} ({info.project_id})\n")
        job._log_fh.write(f"# experiment: {info.label} ({info.experiment_id})\n\n")
        job._log_fh.flush()

        try:
            job.proc = subprocess.Popen(command, **popen_kwargs)
        except Exception as exc:
            info.status = "failed"
            info.finished_at = time.time()
            self._handle_line(job, f"[orchestrator] 启动失败：{exc}", False)
            if job._log_fh:
                job._log_fh.close()
                job._log_fh = None
            self._persist_job_state(job)
            return

        info.pid = job.proc.pid
        info.status = "running"
        info.finished_at = None
        info.exit_code = None
        info.error = None
        self._persist_job_state(job, unit=unit)
        job.reader = threading.Thread(target=self._reader, args=(job,), daemon=True)
        job.reader.start()


    def _reader(self, job: Job) -> None:
        proc = job.proc
        assert proc is not None and proc.stdout is not None
        stream = proc.stdout
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        try:
            while True:
                chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
                if not chunk:
                    break
                buf += decoder.decode(chunk)
                segments, buf = _split_lines(buf)
                for seg, transient in segments:
                    self._handle_line(job, seg, transient)
            buf += decoder.decode(b"", final=True)
            if buf:
                self._handle_line(job, buf, False)
        except Exception as exc:
            self._handle_line(job, f"[orchestrator] 读取输出异常：{exc}", False)
        finally:
            code = proc.wait()
            job.info.exit_code = code
            _mark_finished(job.info)
            finished_ts = job.info.finished_at or time.time()
            with job.lock:
                lines_snapshot = list(job.lines)
            cost_circuit_breaker = code != 0 and _is_cost_circuit_breaker_lines(lines_snapshot)
            with job.state_lock:
                if job._stopping:
                    job.info.status = "stopped"
                elif code == 0:
                    job.info.status = "succeeded"
                elif cost_circuit_breaker:
                    job.info.status = "stopped"
                else:
                    job.info.status = "failed"


            job.info.sample_stats = self._compute_sample_stats(job.info)
            job.sample_stats_ts = time.monotonic()
            if job.info.status == "failed" or cost_circuit_breaker:
                job.info.error = _extract_error(lines_snapshot)
            if cost_circuit_breaker:
                job.info.error = _COST_CIRCUIT_MESSAGE
            self._persist_job_state(job)


            if job.proc is proc:
                llm_proxy.retire_job_tokens(job.info.id, created_before=finished_ts)
            try:
                if job._log_fh:
                    job._log_fh.write(f"\n# 进程结束，exit_code={code}, status={job.info.status}\n")
                    job._log_fh.flush()
                    job._log_fh.close()
            except Exception:
                pass
            self._write_run_meta(job)
            self._broadcast(job, _END_SENTINEL, False)

    def _write_run_meta(self, job: Job) -> None:
        info = job.info

        def _iso(ts: float | None) -> str | None:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None

        meta = {
            "schema": "run_meta/v1",
            "job_id": info.id,
            "project_id": info.project_id,
            "project": info.project_name,
            "paper": info.paper,
            "experiment_id": info.experiment_id,
            "experiment": info.label,
            "model": info.model,
            "params": info.params,
            "status": info.status,
            "exit_code": info.exit_code,
            "started_at": _iso(info.started_at),
            "finished_at": _iso(info.finished_at),
            "results_dir": info.results_dir,
            "log_path": info.log_path,
        }
        blob = json.dumps(meta, ensure_ascii=False, indent=2)

        run_dir = settings.RUNS_DIR / info.id
        try:
            (run_dir / "run_meta.json").write_text(blob, encoding="utf-8")
        except Exception:
            pass

        try:
            if info.results_dir:
                rd = Path(info.results_dir)
                if rd.exists():
                    (rd / "run_meta.json").write_text(blob, encoding="utf-8")
        except Exception:
            pass

        try:
            idx = {k: meta[k] for k in
                   ("job_id", "project", "experiment", "model", "status", "finished_at", "results_dir")}
            with (settings.RUNS_DIR / "runs_index.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(idx, ensure_ascii=False) + "\n")
            (settings.RUNS_DIR / "LAST_RUN.txt").write_text(
                f"{info.project_name} | {info.label} | {info.model} | "
                f"{info.status} | {meta['finished_at']} | {info.id}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _handle_line(self, job: Job, line: str, transient: bool) -> None:


        with job.lock:
            if transient and job.last_transient and job.lines:
                job.lines[-1] = line
            else:
                job.lines.append(line)
            job.last_transient = transient
            self._broadcast(job, line, transient)

        if job._log_fh and not transient:
            try:
                job._log_fh.write(line + "\n")
                job._log_fh.flush()
            except Exception:
                pass
        resumed = progress.parse_resume_line(line)
        if resumed:
            cur, total = resumed
            job.progress_base = cur
            job.progress_total = total
            job.info.progress.current = cur
            job.info.progress.total = total
            job.info.progress.percent = round(cur * 100 / total, 1) if total else 0.0
            self._persist_job_state(job)
            return

        parsed = progress.parse_line(line)
        if parsed:
            cur, total = parsed
            if total > 0 and job.progress_base and job.progress_total:
                full_total = job.progress_total
                remaining = max(0, full_total - job.progress_base)
                if total == remaining or total < full_total or (total == full_total and cur < job.progress_base):
                    cur = min(full_total, job.progress_base + cur)
                    total = full_total
            job.info.progress.current = cur
            job.info.progress.total = total
            job.info.progress.percent = round(cur * 100 / total, 1) if total else 0.0
            self._persist_job_state(job)

    def _broadcast(self, job: Job, line: str, transient: bool) -> None:
        loop = self.loop
        if loop is None:
            return
        for q in list(job.subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, (line, transient))
            except Exception:
                pass


    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is not None:
            return job
        job = self._load_job_on_demand(job_id)
        if job is not None:
            return job
        raise KeyError(f"未知任务：{job_id}")

    def _load_job_on_demand(self, job_id: str) -> Job | None:
        state_path = settings.RUNS_DIR / job_id / _STATE_FILE
        if not state_path.is_file():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            repaired = _repair_moved_paths(state, state_path.parent)
            info = JobState.model_validate(state).info
        except Exception:
            return None
        _normalize_cost_circuit_info(info)
        if info.status in _ACTIVE_STATUSES:
            info.finished_at = None
            info.exit_code = None
            old_pid = info.pid
            if not (info.status == "running" and _pid_alive(old_pid) and _pid_looks_like_job(old_pid, info.command)):
                info.pid = None
                was_running = info.status == "running"
                info.status = "paused"
                if was_running or info.paused_at is None:
                    _mark_paused(info)
        job = Job(info)
        unit = None
        if repaired and state.get("unit"):
            try:
                unit = _unit_from_state(state)
            except RuntimeError:
                unit = None
        with self.lock:
            existing = self.jobs.get(info.id)
            if existing is not None:
                return existing
            self.jobs[info.id] = job
        if repaired:
            self._persist_job_state(job, unit=unit)
        return job

    def job_ids_by_activity(self) -> tuple[set[str], set[str]]:
        with self.lock:
            all_ids = set(self.jobs)
            active = {j.info.id for j in self.jobs.values() if j.info.status in _ACTIVE_STATUSES}
        if self.recovery_partial:

            unknown = set(self.known_job_ids) - all_ids
            all_ids |= self.known_job_ids
            active |= unknown
        return all_ids, active

    def list_infos(self) -> list[JobInfo]:
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            job.info.api_calls = llm_proxy.calls_for_job(job.info.id)
            self._refresh_sample_stats(job)
        return sorted((j.info for j in jobs), key=lambda i: i.started_at, reverse=True)

    def _refresh_sample_stats(self, job: Job) -> None:
        info = job.info


        live = info.status == "running"
        if not _stats_need_refresh(info):
            if not live:
                return
            if time.monotonic() - job.sample_stats_ts < _SAMPLE_STATS_TTL:
                return
        if not live:


            self.ensure_sample_stats_backfill()
            return
        self._schedule_live_stats_refresh(job)

    def _schedule_live_stats_refresh(self, job: Job) -> None:
        with self.live_stats_lock:
            self.live_stats_pending.add(job.info.id)
            if self.live_stats_thread and self.live_stats_thread.is_alive():
                return
            self.live_stats_thread = threading.Thread(
                target=self._live_stats_refresh_loop,
                name="live-sample-stats",
                daemon=True,
            )
            self.live_stats_thread.start()

    def _live_stats_refresh_loop(self) -> None:
        while True:
            with self.live_stats_lock:
                if not self.live_stats_pending:
                    return
                job_id = self.live_stats_pending.pop()
            with self.lock:
                job = self.jobs.get(job_id)
            if job is None or job.info.status != "running":
                continue


            job.info.sample_stats = self._compute_sample_stats(job.info)
            job.sample_stats_ts = time.monotonic()

    @staticmethod
    def _compute_sample_stats(info: JobInfo) -> SampleStats | None:
        try:
            stats = get_adapter(info.project_id).sample_stats(info)
        except Exception:
            stats = None
        return SampleStats(**stats) if stats else None

    def ensure_sample_stats_backfill(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self.sample_stats_backfill_lock:
            if self.sample_stats_backfill_thread and self.sample_stats_backfill_thread.is_alive():
                return
            if not force and now - self.sample_stats_backfill_last < _SAMPLE_STATS_BACKFILL_INTERVAL:
                return
            self.sample_stats_backfill_last = now
            self.sample_stats_backfill_thread = threading.Thread(
                target=self._sample_stats_backfill_loop,
                name="sample-stats-backfill",
                daemon=True,
            )
            self.sample_stats_backfill_thread.start()

    def _sample_stats_backfill_loop(self) -> None:
        with self.lock:
            jobs = [
                job for job in self.jobs.values()
                if _stats_need_refresh(job.info)
                and job.info.status != "running"
                and job.info.id not in self.sample_stats_backfill_skipped
            ]
        jobs.sort(key=lambda j: j.info.started_at or 0, reverse=True)
        for job in jobs[:_SAMPLE_STATS_BACKFILL_BATCH_SIZE]:
            with job.state_lock:
                if not _stats_need_refresh(job.info) or job.info.status == "running":
                    continue
            stats = self._compute_sample_stats_in_subprocess(job.info)
            if stats is None:
                self.sample_stats_backfill_skipped.add(job.info.id)
                continue
            job.info.sample_stats = stats
            job.sample_stats_ts = time.monotonic()
            self._persist_job_state(job)
            time.sleep(_SAMPLE_STATS_BACKFILL_DELAY)

    @staticmethod
    def _compute_sample_stats_in_subprocess(info: JobInfo) -> SampleStats | None:
        state_path = settings.RUNS_DIR / info.id / _STATE_FILE
        if not state_path.is_file():
            return None
        child = r'''
import json
import sys
from pathlib import Path
from orchestrator.backend.registry import get_adapter
from orchestrator.backend.schemas import JobState

state = JobState.model_validate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
stats = get_adapter(state.info.project_id).sample_stats(state.info)
print(json.dumps(stats, ensure_ascii=False))
'''
        try:
            cp = subprocess.run(
                [sys.executable, "-c", child, str(state_path)],
                cwd=str(settings.REPO_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_SAMPLE_STATS_BACKFILL_TIMEOUT,
                preexec_fn=(lambda: os.nice(10)) if _IS_POSIX else None,
            )
        except Exception:
            return None
        if cp.returncode != 0:
            return None
        try:
            stats = json.loads(cp.stdout or "null")
        except json.JSONDecodeError:
            return None
        return SampleStats(**stats) if stats else None

    @staticmethod
    def _has_failed_samples(info: JobInfo) -> bool:
        try:
            stats = get_adapter(info.project_id).sample_stats(info)
        except Exception:
            stats = None
        return bool(stats and int(stats.get("fail", 0) or 0) > 0)

    @staticmethod
    def _has_incomplete_progress(info: JobInfo) -> bool:
        prog = info.progress
        if prog is None:
            return False
        try:
            total = int(prog.total or 0)
            current = int(prog.current or 0)
        except (TypeError, ValueError):
            return False
        return total > 0 and current < total


    def _signal_group(self, job: Job, sig: int) -> None:
        proc = job.proc
        if proc is None or proc.poll() is not None:
            return
        if _IS_POSIX:
            try:
                os.killpg(os.getpgid(proc.pid), sig)
                return
            except ProcessLookupError:
                return
            except Exception:
                pass
        proc.send_signal(sig)

    def _append_orchestrator_line(self, job: Job, line: str) -> None:
        with job.lock:
            job.lines.append(line)
            job.last_transient = False
            self._broadcast(job, line, False)
        try:
            if job.info.log_path:
                path = Path(job.info.log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    def _terminate_detached_process(self, job: Job, *, reason: str) -> None:
        killed = _kill_detached_by_marker(job.info.id, job.info.command)
        if killed:
            self._append_orchestrator_line(
                job,
                f"[orchestrator] {reason}：已按 job_id 终止 {killed} 个遗留进程组。",
            )
            job.info.pid = None
            return

        pid = job.info.pid
        if not _pid_alive(pid):
            job.info.pid = None
            return
        if not _pid_looks_like_job(pid, job.info.command):
            job.info.pid = None
            return

        self._append_orchestrator_line(
            job,
            f"[orchestrator] {reason} pid={pid}，先终止旧进程，再按已有产物断点续跑。",
        )
        if _IS_POSIX:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                job.info.pid = None
                return
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(0.5)
            if _pid_alive(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
        job.info.pid = None

    def pause(self, job_id: str) -> JobInfo:
        job = self.get(job_id)
        with job.state_lock:
            if job.info.status != "running":
                return job.info
            if not _IS_POSIX:
                raise RuntimeError("当前操作系统不支持暂停（仅 POSIX 支持 SIGSTOP）。")
            if job.proc is None or job.proc.poll() is not None:
                return job.info
            self._signal_group(job, signal.SIGSTOP)
            job.info.status = "paused"
            _mark_paused(job.info)
            self._persist_job_state(job)
        return job.info

    def resume(
        self,
        job_id: str,
        *,
        force_cost_circuit_resume: bool = False,
        params_override: dict[str, Any] | None = None,
    ) -> JobInfo:
        job = self.get(job_id)
        with self.launch_lock:
            with job.state_lock:
                status = job.info.status
                cost_circuit_overrides: dict[str, int] | None = None
                if status == "paused":
                    proc = job.proc
                    if proc is not None and proc.poll() is None:
                        self._signal_group(job, signal.SIGCONT)
                        _mark_resumed(job.info)
                        job.info.status = "running"
                        self._persist_job_state(job)
                        return job.info
                    self._terminate_detached_process(job, reason="续跑前检测到后端重启遗留的旧子进程")
                elif status in ("succeeded", "failed", "stopped"):
                    if status in ("failed", "stopped") and _is_cost_circuit_breaker_text(job.info.error):
                        if not force_cost_circuit_resume:
                            raise RuntimeError(
                                "该任务已被成本熔断保护停止：历史结果中的空响应/API 请求失败超过阈值，"
                                "为避免继续消耗费用，默认禁止直接续跑。确认要继续时，请调高或关闭"
                                " max_empty_responses / max_request_failures 后重跑。"
                            )
                        cost_circuit_overrides = _validate_cost_circuit_resume_overrides(params_override)


                    if status == "succeeded" and not (
                        self._has_failed_samples(job.info)
                        or self._has_incomplete_progress(job.info)
                    ):
                        return job.info
                else:
                    return job.info

                adapter = get_adapter(job.info.project_id)
                if not adapter.allow_concurrent and self._has_other_active_job(job.info.project_id, job.info.id):
                    raise RuntimeError(
                        f"项目「{adapter.name}」已有其它任务正在运行或暂停中，请先处理该任务后再继续。"
                    )


                run_dir = settings.RUNS_DIR / job.info.id
                state = _read_job_state(run_dir)
                unit = _unit_from_state(state)
                _assert_unit_matches_job(job.info, unit)
                if cost_circuit_overrides:
                    job.info.params = {**(job.info.params or {}), **cost_circuit_overrides}
                    self._append_orchestrator_line(
                        job,
                        "[orchestrator] 用户确认续跑成本熔断任务："
                        f"本次覆盖 max_empty_responses={cost_circuit_overrides['max_empty_responses']}，"
                        f"max_request_failures={cost_circuit_overrides['max_request_failures']}。",
                    )

                llm: UnifiedLLM | None = None
                if job.info.experiment_id != "analyze":
                    llm = self._llm_for_resume(job, state)
                    adapter.render_config(llm, job.info.params, unit, work_dir=run_dir)
                self._refresh_command(job, unit)
                job.info.results_dir = str(adapter.results_dir())
                _mark_resumed(job.info)
                self._terminate_detached_process(job, reason="续跑前检测到同一任务遗留的旧子进程")
                self._persist_job_state(job, unit=unit, llm=llm)
                self._start_process(job, unit, append_log=True)
        return job.info

    def rerun(self, job_id: str) -> JobInfo:
        job = self.get(job_id)
        with self.launch_lock:


            self._stop_for_delete(job)
            if job.reader and job.reader.is_alive():
                job.reader.join(timeout=5.0)
            self._close_log(job)

            with job.state_lock:
                job._stopping = False
                adapter = get_adapter(job.info.project_id)
                if not adapter.allow_concurrent and self._has_other_active_job(job.info.project_id, job.info.id):
                    raise RuntimeError(
                        f"项目「{adapter.name}」已有其它任务正在运行或暂停中，请先处理该任务后再重跑。"
                    )

                run_dir = settings.RUNS_DIR / job.info.id
                state = _read_job_state(run_dir)
                unit = _unit_from_state(state)
                _assert_unit_matches_job(job.info, unit)


                try:
                    adapter.reset_job_outputs(job.info)
                except Exception as exc:
                    raise RuntimeError(f"清空旧产物失败：{exc}") from exc

                llm: UnifiedLLM | None = None
                if job.info.experiment_id != "analyze":
                    llm = self._llm_for_resume(job, state)
                    adapter.render_config(llm, job.info.params, unit, work_dir=run_dir)
                self._refresh_command(job, unit)
                self._terminate_detached_process(job, reason="重跑前检测到同一任务遗留的旧子进程")


                job.info.results_dir = str(adapter.results_dir())
                job.info.progress = JobProgress()
                job.info.sample_stats = None
                job.info.api_calls = 0
                job.info.error = None
                job.info.exit_code = None
                job.info.paused_at = None
                job.info.paused_total = 0.0
                job.info.started_at = time.time()
                job.info.finished_at = None

                self._persist_job_state(job, unit=unit, llm=llm)
                self._start_process(job, unit, append_log=False)
        return job.info

    def stop(self, job_id: str) -> JobInfo:
        job = self.get(job_id)
        with job.state_lock:
            proc = job.proc
            if proc is None or proc.poll() is not None:


                if job.info.status in ("running", "paused"):
                    self._terminate_detached_process(job, reason="停止任务时检测到后端重启遗留的旧子进程")
                    _mark_finished(job.info)
                    job.info.status = "stopped"
                    job.info.exit_code = None
                    self._persist_job_state(job)
                    llm_proxy.retire_job_tokens(job.info.id)
                return job.info
            if job.info.status not in ("running", "paused"):
                return job.info
            job._stopping = True
            if job.info.status == "paused" and _IS_POSIX:
                _mark_resumed(job.info)
                self._signal_group(job, signal.SIGCONT)
            self._signal_group(job, signal.SIGTERM)
            self._terminate_detached_process(job, reason="停止任务时检测到同一任务遗留的旧子进程")
            job.info.status = "stopped"
            self._persist_job_state(job)
        threading.Thread(target=self._force_kill_later, args=(job, 8.0), daemon=True).start()
        return job.info

    def delete(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        info = job.info.model_copy(deep=True)
        run_dir = _safe_run_dir(info)
        result_dir: Path | None = None
        result_skip = ""
        try:
            adapter = get_adapter(info.project_id)
            result_dir, result_skip = _safe_result_dir(adapter, info)
        except Exception as exc:
            result_skip = f"无法解析结果目录：{exc}"

        self._stop_for_delete(job)
        if job.reader and job.reader.is_alive():
            job.reader.join(timeout=5.0)
        self._close_log(job)

        deleted: list[str] = []
        skipped: list[str] = []
        for target in (result_dir, run_dir):
            if target is None:
                continue
            try:
                if target.exists():
                    shutil.rmtree(target)
                    deleted.append(str(target))
                else:
                    skipped.append(f"{target}（不存在）")
            except Exception as exc:
                skipped.append(f"{target}（删除失败：{exc}）")
        if result_skip:
            skipped.append(result_skip)

        with self.lock:
            self.jobs.pop(job_id, None)
        llm_proxy.retire_job_tokens(job_id, drop_base=True)
        return {"job_id": job_id, "deleted": deleted, "skipped": skipped}

    def _stop_for_delete(self, job: Job) -> None:
        with job.state_lock:
            proc = job.proc
            if proc is None or proc.poll() is not None:
                if job.info.status in ("running", "paused"):
                    self._terminate_detached_process(job, reason="删除任务前检测到后端重启遗留的旧子进程")
                return
            if job.info.status == "paused" and _IS_POSIX:
                try:
                    self._signal_group(job, signal.SIGCONT)
                except Exception:
                    pass
            job._stopping = True
            self._signal_group(job, signal.SIGTERM)

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._signal_group(job, signal.SIGKILL if _IS_POSIX else signal.SIGTERM)
            try:
                proc.wait(timeout=3.0)
            except Exception:
                pass

    @staticmethod
    def _close_log(job: Job) -> None:
        try:
            if job._log_fh:
                job._log_fh.flush()
                job._log_fh.close()
                job._log_fh = None
        except Exception:
            pass

    def _llm_for_resume(self, job: Job, state: dict[str, Any]) -> UnifiedLLM:
        raw = settings.load_raw_config()
        fields = set(UnifiedLLM.model_fields)
        current_raw = settings.resolve_env(dict(raw.get("llm") or {}))
        snapshot = settings.resolve_env(dict(state.get("llm_snapshot") or {}))
        if snapshot:


            snap_key = str(snapshot.get("api_key") or "").strip()
            if not snap_key or snap_key.startswith("${"):
                snapshot["api_key"] = current_raw.get("api_key", "")
            llm = UnifiedLLM(**{k: v for k, v in snapshot.items() if k in fields})
        else:


            llm = UnifiedLLM(**{k: v for k, v in current_raw.items() if k in fields})
            if job.info.model:
                llm.model = job.info.model

        if job.info.model and llm.model and llm.model != job.info.model:
            raise RuntimeError(
                "不能续跑：job_state.json 中保存的模型与任务记录不一致，"
                f"状态快照={llm.model}，任务={job.info.model}。"
            )
        key = (llm.api_key or "").strip()
        if not key or key.startswith("${"):
            raise RuntimeError(
                "不能续跑：该任务的 api_key 为空或未解析。新任务会自动保存续跑快照；"
                "旧任务请先在统一配置/.env 中临时提供可用于原 endpoint 的 key 后再继续。"
            )
        return llm

    def _refresh_command(self, job: Job, unit: JobUnit) -> None:
        adapter = get_adapter(job.info.project_id)
        require_llm = job.info.experiment_id != "analyze"
        argv = _prepare_spawn_argv(adapter, unit, require_llm=require_llm)
        py = str(unit.extra.get("python_executable") or (job.info.command[0] if job.info.command else "python"))
        job.info.command = [py, *argv]

    def _persist_job_state(
        self,
        job: Job,
        *,
        unit: JobUnit | None = None,
        llm: UnifiedLLM | None = None,
        existing_state: dict[str, Any] | None = None,
    ) -> None:


        run_dir = settings.RUNS_DIR / job.info.id
        run_dir.mkdir(parents=True, exist_ok=True)

        state = dict(existing_state) if existing_state is not None else _read_job_state(run_dir)
        state["schema"] = "job_state/v1"
        state["info"] = job.info.model_dump()
        if unit is not None:
            state["unit"] = _unit_to_state(unit)
        if llm is not None:
            state["llm_snapshot"] = _llm_snapshot(llm)
        state = JobState.model_validate(state).model_dump(by_alias=True)
        tmp = run_dir / (_STATE_FILE + ".tmp")
        with job.persist_lock:
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(run_dir / _STATE_FILE)

    def _force_kill_later(self, job: Job, grace: float) -> None:
        time.sleep(grace)
        proc = job.proc
        if proc is not None and proc.poll() is None:
            self._signal_group(job, signal.SIGKILL if _IS_POSIX else signal.SIGTERM)


    @staticmethod
    def _ensure_log_backlog_loaded(job: Job) -> None:
        if job.log_backlog_loaded:
            return
        backlog_limit = 4000 if job.info.status in _ACTIVE_STATUSES else 800
        backlog_bytes = 512 * 1024 if job.info.status in _ACTIVE_STATUSES else 128 * 1024
        disk_lines = _read_log_backlog(
            Path(job.info.log_path), limit=backlog_limit, max_tail_bytes=backlog_bytes
        )
        with job.lock:
            if job.log_backlog_loaded:
                return
            existing = list(job.lines)
            job.lines.clear()
            for line in disk_lines:
                job.lines.append(line)
            for line in existing:
                job.lines.append(line)
            job.log_backlog_loaded = True

    async def stream_logs(self, job_id: str):
        job = self.get(job_id)
        self._ensure_log_backlog_loaded(job)
        q: asyncio.Queue = asyncio.Queue()


        with job.lock:
            backlog = list(job.lines)
            job.subscribers.add(q)
        try:
            for line in backlog:
                yield _sse(line, False)

            if job.info.status not in ("running", "paused") and q.empty():
                yield _sse_end(job.info)
                return
            while True:
                line, transient = await q.get()
                if line == _END_SENTINEL:
                    yield _sse_end(job.info)
                    break
                yield _sse(line, transient)
        finally:
            job.subscribers.discard(q)


def _stats_need_refresh(info: JobInfo) -> bool:
    if info.sample_stats is None:
        return True
    if _social_stats_look_stale(info):
        return True
    if info.sample_stats.failed_calls is None:
        try:
            return bool(getattr(get_adapter(info.project_id), "reports_failed_calls", False))
        except Exception:
            return False
    return False


def _social_stats_look_stale(info: JobInfo) -> bool:
    if info.project_id != "social" or info.sample_stats is None:
        return False
    try:
        current = int((info.progress.current if info.progress else 0) or 0)
        ok = int(info.sample_stats.ok or 0)
    except (TypeError, ValueError):
        return False
    return current > 0 and ok == 0


def _extract_error(lines: list[str], limit: int = 6) -> str:
    skip = ("# 进程结束", "$ ", "# cwd")
    out: list[str] = []
    for line in reversed(lines):
        s = line.strip()
        if not s or s.startswith(skip):
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return "\n".join(reversed(out))


def _is_cost_circuit_breaker_text(text: Any) -> bool:
    value = str(text or "")
    return any(marker in value for marker in _COST_CIRCUIT_MARKERS)


def _is_cost_circuit_breaker_lines(lines: list[str]) -> bool:
    return _is_cost_circuit_breaker_text("\n".join(lines[-80:]))


def _validate_cost_circuit_resume_overrides(overrides: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(overrides, dict):
        raise RuntimeError(
            "确认续跑成本熔断任务时，必须提供 max_empty_responses / max_request_failures 覆盖值。"
        )
    out: dict[str, int] = {}
    for key in ("max_empty_responses", "max_request_failures"):
        raw = overrides.get(key)
        if raw in (None, ""):
            raise RuntimeError(f"确认续跑成本熔断任务时，必须提供 {key}。")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{key} 必须是非负整数。") from exc
        if value < 0:
            raise RuntimeError(f"{key} 必须是非负整数。")
        out[key] = value
    return out


def _normalize_cost_circuit_info(info: JobInfo) -> None:
    if info.status in ("failed", "stopped") and _is_cost_circuit_breaker_text(info.error):
        info.status = "stopped"
        info.error = _COST_CIRCUIT_MESSAGE


def _mark_paused(info: JobInfo, now: float | None = None) -> None:
    if info.paused_at is None:
        info.paused_at = now or time.time()


def _mark_resumed(info: JobInfo, now: float | None = None) -> None:
    if info.paused_at is None:
        return
    ts = now or time.time()
    info.paused_total = max(0.0, float(info.paused_total or 0.0)) + max(0.0, ts - info.paused_at)
    info.paused_at = None


def _mark_finished(info: JobInfo, now: float | None = None) -> None:
    ts = now or time.time()
    _mark_resumed(info, ts)
    info.finished_at = ts


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_run_dir(info: JobInfo) -> Path | None:
    run_dir = settings.RUNS_DIR / info.id
    root = settings.RUNS_DIR.resolve()
    target = run_dir.resolve()
    if target.parent != root or target.name != info.id:
        return None
    return target


def _safe_result_dir(adapter: Any, info: JobInfo) -> tuple[Path | None, str]:
    target = adapter.results_dir_for_job(info).resolve()
    root = adapter.results_dir().resolve()
    if target == root:
        return None, f"结果目录指向项目级根目录，已跳过：{target}"
    if not _is_relative_to(target, root):
        return None, f"结果目录不在项目结果根目录下，已跳过：{target}"
    return target, ""


def _read_job_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / _STATE_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = JobState.model_validate(data)
    except Exception:
        return {}
    return state.model_dump(by_alias=True)


def _repair_moved_paths(state: dict[str, Any], state_dir: Path) -> bool:
    info = state.get("info")
    if not isinstance(info, dict):
        return False
    log_path = str(info.get("log_path") or "")
    if not log_path:
        return False
    old_run_dir = Path(log_path).parent
    if old_run_dir == state_dir:
        return False

    if (
        old_run_dir.name == state_dir.name
        and old_run_dir.parent.name == state_dir.parent.name
        and old_run_dir.parent.parent.name == state_dir.parent.parent.name
    ):
        old_prefix = str(old_run_dir.parent.parent.parent)
        new_prefix = str(state_dir.parent.parent.parent)
    else:
        old_prefix = str(old_run_dir)
        new_prefix = str(state_dir)

    def remap(value: Any) -> Any:
        if isinstance(value, str):
            if value == old_prefix or value.startswith(old_prefix + os.sep):
                return new_prefix + value[len(old_prefix):]
            return value
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value

    for key in ("log_path", "cwd", "results_dir", "command"):
        if info.get(key):
            info[key] = remap(info[key])
    unit = state.get("unit")
    if isinstance(unit, dict):
        for key in ("argv", "extra"):
            if unit.get(key):
                unit[key] = remap(unit[key])
    return True


def _append_log_line(path: Path, line: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0 or not _IS_POSIX:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_command(pid: int | None) -> str:
    if not pid or not _IS_POSIX:
        return ""
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            timeout=1.0,
        ).strip()
    except Exception:
        return ""


def _pid_looks_like_job(pid: int | None, command: list[str]) -> bool:
    proc_cmd = _pid_command(pid)
    if not proc_cmd or not command:
        return False

    exe_names = {Path(command[0]).name}
    try:
        exe_names.add(Path(os.path.realpath(command[0])).name)
    except Exception:
        pass
    if not any(name and name in proc_cmd for name in exe_names):
        return False

    markers: list[str] = []
    skip_next = False
    for arg in command[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--config", "-c", "-m"}:
            skip_next = arg == "--config"
            continue
        if arg.startswith("-"):
            continue
        marker = Path(arg).name if ("/" in arg or "\\" in arg) else arg
        if marker:
            markers.append(marker)
        if len(markers) >= 3:
            break
    return bool(markers) and any(marker in proc_cmd for marker in markers)


def _stop_detached_process_group(pid: int | None) -> None:
    if not pid or not _IS_POSIX:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGSTOP)
    except Exception:
        try:
            os.kill(pid, signal.SIGSTOP)
        except Exception:
            pass


def _kill_process_group(pid: int | None) -> int:
    if not pid or not _IS_POSIX:
        return 0
    try:
        pgid = os.getpgid(pid)
    except Exception:
        return 0
    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        return 0
    time.sleep(0.3)
    if _pid_alive(pid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    return 1


def _kill_detached_by_marker(marker: str, command: list[str]) -> int:
    if not marker or not _IS_POSIX:
        return 0
    try:
        out = subprocess.check_output(["pgrep", "-f", marker], text=True, timeout=2.0)
    except Exception:
        return 0
    self_pid = os.getpid()
    pgids: set[int] = set()
    for tok in out.split():
        if not tok.isdigit():
            continue
        pid = int(tok)
        if pid == self_pid or not _pid_looks_like_job(pid, command):
            continue
        try:
            pgids.add(os.getpgid(pid))
        except Exception:
            pass
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
    if pgids:
        time.sleep(0.5)
        for pgid in pgids:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
    return len(pgids)


def _unit_to_state(unit: JobUnit) -> dict[str, Any]:

    extra = {k: v for k, v in dict(unit.extra or {}).items()
             if k not in ("env", "llm_proxy_token", "llm_proxy_token_judge")}
    return JobUnitState(
        experiment_id=unit.experiment_id,
        label=unit.label,
        argv=list(unit.argv),
        selected=list(unit.selected),
        extra=extra,
    ).model_dump()


def _unit_from_state(state: dict[str, Any]) -> JobUnit:
    data = state.get("unit") or {}
    if not data:
        raise RuntimeError("不能续跑：缺少 job_state.json 中的启动单元信息。")
    try:
        unit = JobUnitState.model_validate(data)
    except Exception as exc:
        raise RuntimeError("不能续跑：job_state.json 中的启动单元信息格式无效。") from exc
    return JobUnit(
        experiment_id=unit.experiment_id,
        label=unit.label or unit.experiment_id,
        argv=unit.argv,
        selected=unit.selected,
        extra=unit.extra,
    )


def _assert_unit_matches_job(info: JobInfo, unit: JobUnit) -> None:
    if unit.experiment_id != info.experiment_id:
        raise RuntimeError(
            "不能续跑：job_state.json 中的实验 ID 与任务记录不一致，"
            f"任务={info.experiment_id}，启动单元={unit.experiment_id}。"
        )
    if unit.label and info.label and unit.label != info.label:
        raise RuntimeError(
            "不能续跑：job_state.json 中的实验名称与任务记录不一致，"
            f"任务={info.label}，启动单元={unit.label}。"
        )


def _llm_snapshot(llm: UnifiedLLM) -> dict[str, Any]:
    return {name: getattr(llm, name) for name in _LLM_SNAPSHOT_FIELDS}


def _read_log_backlog(path: Path, limit: int = 800, max_tail_bytes: int = 128 * 1024) -> list[str]:
    if not path.is_file():
        return []
    try:
        chunks: list[bytes] = []
        newlines = 0
        total_read = 0
        chunk_size = 64 * 1024
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            while pos > 0 and newlines <= limit and total_read < max_tail_bytes:
                read_size = min(chunk_size, pos, max_tail_bytes - total_read)
                if read_size <= 0:
                    break
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                chunks.append(chunk)
                total_read += len(chunk)
                newlines += chunk.count(b"\n")
        lines = b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-limit:]


def _with_config(argv: list[str], config_path: str) -> list[str]:
    out = list(argv)
    for i, a in enumerate(out):
        if a == "--config" and i + 1 < len(out):
            out[i + 1] = config_path
            return out
    return out + ["--config", config_path]


def _prepare_spawn_argv(adapter, unit: JobUnit, *, require_llm: bool = True) -> list[str]:
    cfg_override = unit.extra.get("config_path")
    if not cfg_override:
        raise ValueError(
            f"项目「{adapter.name}」未生成 per-job 配置（config_path 为空）。"
            "已拒绝启动，以免子进程读取子项目根目录模板配置。"
        )
    cfg_path = Path(str(cfg_override))
    if not cfg_path.is_file():
        raise ValueError(
            f"per-job 配置文件不存在：{cfg_path}。请重新从 orchestrator 发起实验。"
        )
    if require_llm and not unit.extra.get("llm_proxy_token"):
        raise ValueError(
            f"项目「{adapter.name}」缺少 LLM proxy token。"
            "已拒绝启动，以免未走 orchestrator 统一 API 注入。"
        )
    argv = _with_config(list(unit.argv), str(cfg_path))
    try:
        idx = argv.index("--config")
    except ValueError as exc:
        raise ValueError("内部错误：未能向子进程 argv 注入 --config。") from exc
    if idx + 1 >= len(argv) or argv[idx + 1] != str(cfg_path):
        raise ValueError("内部错误：--config 未指向 orchestrator 生成的配置文件。")
    if "--models-config" in argv:
        mi = argv.index("--models-config")
        if mi + 1 < len(argv):
            models_path = Path(argv[mi + 1])
            if not models_path.is_file():
                raise ValueError(f"per-job models 配置不存在：{models_path}")
    return argv


def _split_lines(buf: str) -> tuple[list[tuple[str, bool]], str]:
    out: list[tuple[str, bool]] = []
    start = 0
    i = 0
    n = len(buf)
    while i < n:
        c = buf[i]
        if c == "\n":
            out.append((buf[start:i], False))
            start = i + 1
        elif c == "\r":
            if i + 1 < n:
                if buf[i + 1] == "\n":
                    out.append((buf[start:i], False))
                    start = i + 2
                    i += 1
                else:
                    out.append((buf[start:i], True))
                    start = i + 1
            else:
                break
        i += 1
    return out, buf[start:]


def _sse(line: str, transient: bool) -> str:
    prefix = "event: progress\n" if transient else ""
    payload = "".join(f"data: {chunk}\n" for chunk in line.split("\n"))
    return prefix + payload + "\n"


def _sse_end(info: JobInfo) -> str:
    import json

    body = json.dumps({"status": info.status, "exit_code": info.exit_code})
    return f"event: end\ndata: {body}\n\n"


def list_result_files(results_dir: str, limit: int = 300) -> list[dict[str, Any]]:
    root = Path(results_dir)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
            except OSError:
                continue
            items.append({
                "path": str(p.relative_to(root)),
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]


manager = JobManager()
