from __future__ import annotations

import asyncio
import csv
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import llm_proxy, settings, token_ledger
from .jobs import list_result_files, manager
from .registry import get_adapter, list_projects
from .schemas import (
    ChatCompletionRequest,
    LaunchRequest,
    UnifiedConfig,
    UnifiedLLM,
)


_LLM_PROXY_EXECUTOR = ThreadPoolExecutor(max_workers=64, thread_name_prefix="llm-proxy")


@asynccontextmanager
async def _lifespan(_app: "FastAPI"):

    manager.attach_loop(asyncio.get_running_loop())
    llm_proxy.apply_from_raw_config(settings.load_raw_config())
    llm_proxy.load_tokens()

    llm_proxy.prune_tokens(*manager.job_ids_by_activity())
    token_ledger.load()
    try:
        yield
    finally:
        token_ledger.flush()
        llm_proxy.flush_tokens()
        _LLM_PROXY_EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="LLM 认知实验平台", lifespan=_lifespan)

FRONTEND_DIR = settings.ORCHESTRATOR_DIR / "frontend"


def _build_llm(raw_cfg: dict[str, Any], override: dict[str, Any] | None) -> UnifiedLLM:
    llm_raw = dict(raw_cfg.get("llm") or {})
    if override:
        llm_raw.update({k: v for k, v in override.items() if v is not None})
    llm_raw = settings.resolve_env(llm_raw)
    fields = set(UnifiedLLM.model_fields)
    return UnifiedLLM(**{k: v for k, v in llm_raw.items() if k in fields})


def _assert_authenticated(llm: UnifiedLLM) -> None:
    key = (llm.api_key or "").strip()
    if not key or key.startswith("${"):
        raise HTTPException(
            status_code=400,
            detail="api_key 为空或未解析（形如 ${VAR}）。请在统一配置里填写有效密钥，或在 .env/环境变量中设置后重试。",
        )


@app.post("/api/llm-proxy/v1/chat/completions")
async def post_llm_proxy_chat(req: Request, body: ChatCompletionRequest = Body(...)) -> dict[str, Any]:
    token = llm_proxy.token_from_request(req)
    payload = body.model_dump(mode="json", exclude_none=True)

    return await asyncio.get_running_loop().run_in_executor(
        _LLM_PROXY_EXECUTOR, llm_proxy.chat_completions_response, token, payload
    )


@app.get("/api/health")
def get_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/token-usage")
def get_token_usage() -> dict[str, Any]:
    return token_ledger.snapshot()


def _enrich_config_response(masked: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    llm = masked.get("llm") or {}
    concurrency = int(llm.get("concurrency") or 8)
    proxy = masked.get("proxy") or {}
    if not proxy.get("max_upstream_concurrency"):
        proxy = dict(proxy)
        proxy["max_upstream_concurrency"] = settings.proxy_max_from_raw(raw)
        masked["proxy"] = proxy
    proxy_max = int(proxy.get("max_upstream_concurrency") or 4)
    masked["api_key_configured"] = settings.api_key_is_literal(raw)
    masked["effective_upstream_concurrency"] = min(concurrency, proxy_max)
    masked["model_catalog"] = settings.load_model_catalog()
    return masked


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    raw = settings.load_raw_config()
    llm_proxy.apply_from_raw_config(raw)
    masked = settings.mask_config_for_client(raw)
    return _enrich_config_response(masked, raw)


@app.put("/api/config")
def put_config(cfg: UnifiedConfig) -> dict[str, Any]:
    existing = settings.load_raw_config()
    data = settings.merge_config_on_save(cfg.model_dump(), existing)
    exe = data.get("python_executable")
    if exe and not Path(str(exe)).is_file():
        raise HTTPException(status_code=400, detail=f"Python 解释器不存在：{exe}")
    settings.save_raw_config(data)
    llm_proxy.apply_from_raw_config(data)
    masked = settings.mask_config_for_client(data)
    return _enrich_config_response(masked, data)


@app.get("/api/projects")
def get_projects() -> list[dict[str, Any]]:
    return [p.model_dump() for p in list_projects()]


@app.post("/api/launch")
def post_launch(req: LaunchRequest) -> list[dict[str, Any]]:
    raw = settings.load_raw_config()
    try:
        llm = _build_llm(raw, req.llm_override)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"统一配置解析失败：{exc}") from exc
    _assert_authenticated(llm)
    try:
        infos = manager.launch(req, llm, raw)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [i.model_dump() for i in infos]


@app.get("/api/jobs")
def get_jobs() -> list[dict[str, Any]]:
    return [i.model_dump() for i in manager.list_infos()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return manager.get(job_id).info.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/{action}")
def post_job_action(job_id: str, action: str, body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = body or {}
    if action == "resume":
        try:
            return manager.resume(
                job_id,
                force_cost_circuit_resume=bool(payload.get("force_cost_circuit_resume")),
                params_override=payload.get("params_override") or None,
            ).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    ops = {"pause": manager.pause, "stop": manager.stop, "rerun": manager.rerun}
    if action not in ops:
        raise HTTPException(status_code=400, detail=f"不支持的操作：{action}")
    try:
        return ops[action](job_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    try:
        return manager.delete(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/logs/stream")
async def stream_job_logs(job_id: str) -> StreamingResponse:
    try:
        manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(
        manager.stream_logs(job_id), media_type="text/event-stream", headers=headers
    )


@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str) -> dict[str, Any]:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        adapter = get_adapter(job.info.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    root = adapter.results_dir_for_job(job.info)
    return {"results_dir": str(root), "files": list_result_files(str(root))}


@app.get("/api/jobs/{job_id}/transcript")
def get_job_transcript(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    only_failed: bool = Query(False),
) -> dict[str, Any]:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        adapter = get_adapter(job.info.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    data = adapter.load_transcript(offset, limit, job.info, only_failed=only_failed)
    if data is None:
        raise HTTPException(status_code=404, detail="该项目暂不支持对话查看，或尚无结果文件。")
    return data


@app.get("/api/jobs/{job_id}/results/download")
def download_job_result(job_id: str, path: str = Query(...)) -> FileResponse:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        adapter = get_adapter(job.info.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    root = adapter.results_dir_for_job(job.info).resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="非法路径。")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(str(target), filename=target.name)


@app.get("/api/projects/{project_id}/result-models")
def get_result_models(project_id: str) -> dict[str, Any]:
    try:
        adapter = get_adapter(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "project_id": project_id,
        "can_analyze": adapter.analyze_argv() is not None,
        "models": adapter.list_result_models(),
    }


@app.get("/api/projects/{project_id}/analysis")
def get_project_analysis(project_id: str, model: str | None = Query(default=None)) -> dict[str, Any]:
    try:
        adapter = get_adapter(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"analysis": adapter.load_analysis(model)}


_ANALYZE_LOCK = threading.Lock()


@app.post("/api/projects/{project_id}/analyze")
def post_project_analyze(project_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    try:
        adapter = get_adapter(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    model = (body or {}).get("model") or None
    argv = adapter.analyze_argv(model)
    if not argv:
        raise HTTPException(status_code=400, detail="该项目暂不支持一键结果分析。")

    if not _ANALYZE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="已有一个结果分析正在运行，请等它完成后再试（全局同一时刻仅允许一个分析）。",
        )
    try:

        commands = argv if isinstance(argv[0], list) else [argv]
        raw = settings.load_raw_config()
        py = settings.python_executable(raw)
        stdouts: list[str] = []
        for cmd in commands:
            try:
                proc = subprocess.run(
                    [py, *cmd],
                    cwd=str(adapter.project_dir),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail=f"分析脚本执行超时（>900s）：{cmd[0]}") from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"无法启动分析脚本：{exc}") from exc
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
                raise HTTPException(status_code=500, detail=f"分析脚本失败（{cmd[0]} exit={proc.returncode}）：\n{tail}")
            stdouts.append(proc.stdout or "")
        return {
            "ok": True,
            "stdout": "\n".join(stdouts)[-4000:],
            "analysis": adapter.load_analysis(model),
        }
    finally:
        _ANALYZE_LOCK.release()


@app.get("/api/analyze-status")
def get_analyze_status() -> dict[str, Any]:
    return {"analyzing": _ANALYZE_LOCK.locked()}


_TRENDS_DIR = settings.REPO_ROOT / "trends"


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


@app.get("/api/trends")
def get_trends() -> dict[str, Any]:
    metrics_path = _TRENDS_DIR / "metrics_long.csv"
    tests_path = _TRENDS_DIR / "trend_tests.csv"

    def _mtime(p: Path) -> str | None:
        try:
            return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            return None

    metrics = _read_csv_rows(metrics_path)

    undated = sorted({r.get("model", "") for r in metrics if not (r.get("release_date") or "").strip()})
    return {
        "available": metrics_path.is_file(),
        "metrics": metrics,
        "tests": _read_csv_rows(tests_path),
        "family_diffs": _read_csv_rows(_TRENDS_DIR / "family_diffs.csv"),
        "undated_models": [m for m in undated if m],
        "metrics_generated_at": _mtime(metrics_path),
        "tests_generated_at": _mtime(tests_path),
    }


@app.post("/api/trends/refresh")
def post_trends_refresh() -> dict[str, Any]:
    if not _ANALYZE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="已有一个结果分析/趋势汇总正在运行，请等它完成后再试。",
        )
    try:
        raw = settings.load_raw_config()
        py = settings.python_executable(raw)
        stdouts: list[str] = []
        for script in ("trends/collect_metrics.py", "trends/analyze_trends.py"):
            try:
                proc = subprocess.run(
                    [py, script],
                    cwd=str(settings.REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail=f"趋势脚本执行超时（>900s）：{script}") from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"无法启动趋势脚本：{exc}") from exc
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
                raise HTTPException(
                    status_code=500,
                    detail=f"趋势脚本失败（{script} exit={proc.returncode}）：\n{tail}",
                )
            stdouts.append(proc.stdout or "")
        return {"ok": True, "stdout": "\n".join(stdouts)[-4000:], **get_trends()}
    finally:
        _ANALYZE_LOCK.release()


if settings.FRONTEND_DIST.exists():


    _assets_dir = settings.FRONTEND_DIST / "assets"
    _assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#4f46e5"/>'
    '<path d="M11 12L21 11M11 12L16 22M21 11L16 22" stroke="#fff" stroke-width="1.6" fill="none" opacity="0.7"/>'
    '<circle cx="11" cy="12" r="3" fill="#fff"/>'
    '<circle cx="21" cy="11" r="3" fill="#fff"/>'
    '<circle cx="16" cy="22" r="3" fill="#fff"/>'
    '</svg>'
)


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index() -> str:

    dist_index = settings.FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return dist_index.read_text(encoding="utf-8")

    standalone = FRONTEND_DIR / "standalone.html"
    if standalone.exists():
        return standalone.read_text(encoding="utf-8")
    return (
        "<h1>前端尚未就绪</h1>"
        "<p>请在 orchestrator/frontend 下执行 <code>npm install &amp;&amp; npm run build</code>，"
        "或确认 standalone.html 存在。</p>"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
