"""HTTP API and static hosting of the built frontend."""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import platform
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

import funcprox
from funcprox import serialize
from funcprox.export import export_source

from . import catalog, registry
from .jobs import JobQueue
from .lab import LabManager, clean, preview
from .schemas import FunctionSpec, JobRequest, PredictRequest, RunSpec, SaveRequest, SearchRequest, SweepRequest
from .search import SearchManager
from .sweep import Sweep

STATIC = Path(__file__).resolve().parent / "static"
STREAM_TICK_S = 0.1
NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>funcprox</title>
<body style="font:16px system-ui;margin:3rem"><h1>Frontend not built</h1>
<p>Run <code>npm --prefix ui ci</code> and <code>npm --prefix ui run build</code>, then reload.
The API is up: <a href="/api/docs">/api/docs</a>.</p>"""

# The Windows registry can map .js to text/plain, and browsers then refuse module scripts
for _type, _ext in (("text/javascript", ".js"), ("text/css", ".css"), ("image/svg+xml", ".svg")):
    mimetypes.add_type(_type, _ext)


def default_jobs() -> JobQueue:
    def command(exp: str, profile: str, workers: int) -> list[str]:
        script = catalog.EXPERIMENTS / f"{catalog.experiment_scripts()[exp]}.py"
        return [sys.executable, str(script), "--profile", profile, "--workers", str(workers)]

    return JobQueue(command, after=[sys.executable, str(catalog.EXPERIMENTS / "run_all.py"), "--collect-only"],
                    cwd=str(catalog.EXPERIMENTS))


def sse(event: str, data) -> str:
    body = json.dumps(clean(data), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def python_file(name: str, source: str) -> PlainTextResponse:
    return PlainTextResponse(source, media_type="text/x-python", headers={"Content-Disposition": f'attachment; filename="{name}.py"'})


def event_stream(generator) -> StreamingResponse:
    return StreamingResponse(generator, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class SameOriginWrites:
    """A web page on another site must not be able to start runs on this machine."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            headers = dict(scope["headers"])
            origin = headers.get(b"origin")
            if origin and urlsplit(origin.decode("latin-1")).netloc != headers.get(b"host", b"").decode("latin-1"):
                await JSONResponse({"detail": "cross-site request refused"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(models_dir: Path | None = None, jobs: JobQueue | None = None) -> FastAPI:
    models = Path(models_dir) if models_dir else catalog.ROOT / "runs"
    lab = LabManager()
    searches = SearchManager(lab)
    sweeps = SearchManager(lab)
    queue = jobs or default_jobs()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        lab.shutdown()
        queue.shutdown()  # no experiment keeps running after the interface is closed

    app = FastAPI(title="funcprox interface", version=funcprox.__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.add_middleware(SameOriginWrites)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])  # blocks DNS rebinding

    # ---------------------------------------------------------------- catalog
    @app.get("/api/health")
    def health():
        return dict(status="ok", version=funcprox.__version__, python=platform.python_version(), numpy=np.__version__,
                    cpus=os.cpu_count() or 1, built=(STATIC / "index.html").exists())

    @app.get("/api/overview")
    def overview():
        return catalog.overview()

    @app.get("/api/targets")
    def targets():
        return catalog.targets()

    @app.get("/api/experiments")
    def experiments():
        return catalog.list_experiments()

    @app.get("/api/experiments/{exp}")
    def experiment(exp: str):
        detail = catalog.experiment_detail(exp)
        if detail is None:
            raise HTTPException(404, f"unknown experiment {exp!r}")
        return detail

    @app.get("/api/datasets")
    def datasets():
        return catalog.all_datasets()

    @app.get("/api/datasets/{exp}/{name}")
    def dataset(exp: str, name: str):
        data = catalog.load_dataset(exp, name)
        if data is None:
            raise HTTPException(404, f"no dataset {exp}/{name}")
        return data

    @app.get("/api/documents")
    def documents():
        return catalog.list_documents()

    @app.get("/api/documents/{slug}")
    def document(slug: str):
        doc = catalog.load_document(slug)
        if doc is None:
            raise HTTPException(404, f"unknown document {slug!r}")
        return doc

    @app.post("/api/functions/preview")
    def preview_function(body: FunctionSpec):
        return preview(body.resolved)

    # ----------------------------------------------------------------- search
    def get_search(search_id: str):
        search = searches.searches.get(search_id)
        if search is None:
            raise HTTPException(404, f"unknown search {search_id!r}")
        return search

    @app.post("/api/search", status_code=201)
    def start_search(body: SearchRequest):
        return searches.start(body).state()

    @app.get("/api/search/{search_id}")
    def read_search(search_id: str):
        return get_search(search_id).state()

    @app.post("/api/search/{search_id}/stop")
    def stop_search(search_id: str):
        search = get_search(search_id)
        search.stop()
        return search.state()

    # ------------------------------------------------------------------ sweep
    def get_sweep(sweep_id: str):
        sweep = sweeps.searches.get(sweep_id)
        if sweep is None:
            raise HTTPException(404, f"unknown sweep {sweep_id!r}")
        return sweep

    @app.post("/api/sweeps", status_code=201)
    def start_sweep(body: SweepRequest):
        return sweeps.start(body, Sweep).state()

    @app.get("/api/sweeps/{sweep_id}")
    def read_sweep(sweep_id: str):
        return get_sweep(sweep_id).state()

    @app.post("/api/sweeps/{sweep_id}/stop")
    def stop_sweep(sweep_id: str):
        sweep = get_sweep(sweep_id)
        sweep.stop()
        return sweep.state()

    # -------------------------------------------------------------------- lab
    def get_run(run_id: str):
        run = lab.get(run_id)
        if run is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return run

    @app.post("/api/lab/runs", status_code=201)
    def create_run(spec: RunSpec):
        return lab.submit(spec).summary()

    @app.get("/api/lab/runs")
    def list_runs():
        return [r.summary() for r in lab.all()]

    @app.get("/api/lab/runs/{run_id}")
    def read_run(run_id: str):
        return get_run(run_id).full()

    @app.post("/api/lab/runs/{run_id}/stop")
    def stop_run(run_id: str):
        run = get_run(run_id)
        run.stop_requested = True
        return run.summary()

    @app.delete("/api/lab/runs/{run_id}", status_code=204)
    def delete_run(run_id: str):
        if not lab.remove(run_id):
            raise HTTPException(404, f"unknown run {run_id!r}")
        return Response(status_code=204)

    @app.get("/api/lab/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        run = get_run(run_id)

        async def events():
            init, cursor = run.subscribe()
            yield sse("init", init)
            while not run.finished:
                await asyncio.sleep(STREAM_TICK_S)
                change = run.delta(cursor)
                if change:
                    yield sse("frame", change)
            change = run.delta(cursor)
            if change:
                yield sse("frame", change)
            yield sse("done", dict(run.summary(), final=run.final))

        return event_stream(events())

    def finished_run(run_id: str, what: str):
        run = get_run(run_id)
        if run.final is None:
            raise HTTPException(409, f"only a finished run has {what}")
        return run

    @app.get("/api/lab/runs/{run_id}/replay")
    def replay_run(run_id: str):
        return get_run(run_id).replay()

    @app.get("/api/lab/runs/{run_id}/landscape")
    def landscape_run(run_id: str):
        return finished_run(run_id, "a loss landscape").landscape()

    @app.get("/api/lab/runs/{run_id}/export")
    def export_run(run_id: str):
        run = finished_run(run_id, "a device to export")
        title = f"{run.spec.describe()}, trained on f = {run.target.formula}"
        return python_file(f"funcprox_{run.id}", export_source(run.net, run.xs, run.ys, title))

    @app.post("/api/lab/runs/{run_id}/predict")
    def predict_run(run_id: str, body: PredictRequest):
        run = get_run(run_id)
        if run.final is None:
            raise HTTPException(409, "only a finished run can answer")
        try:
            return registry.answer(run.net, run.xs, run.ys, run.target, body.inputs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/lab/runs/{run_id}/save", status_code=201)
    def save_run(run_id: str, body: SaveRequest):
        run = get_run(run_id)
        if run.final is None:
            raise HTTPException(409, "only a finished run can be saved")
        path = registry.path_for(models, body.name)
        if path.exists():
            raise HTTPException(409, f"a model named {body.name!r} already exists")
        registry.save(path, run)
        run.saved_as = body.name
        return dict(name=body.name, path=str(path))

    # ----------------------------------------------------------------- models
    def model_path(name: str) -> Path:
        path = registry.path_for(models, name)
        if path is None or not path.exists():
            raise HTTPException(404, f"unknown model {name!r}")
        return path

    @app.get("/api/models")
    def list_models():
        return registry.list_models(models)

    @app.get("/api/models/{name}")
    def read_model(name: str):
        return registry.model_detail(model_path(name))

    @app.get("/api/models/{name}/export")
    def export_model(name: str):
        net, xs, ys, meta = serialize.load(model_path(name))
        return python_file(name, export_source(net, xs, ys, f"{meta.get('label', name)}, saved as {name}"))

    @app.post("/api/models/{name}/predict")
    def predict(name: str, body: PredictRequest):
        try:
            return registry.predict(model_path(name), body.inputs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # ------------------------------------------------------------------- jobs
    def get_job(job_id: str):
        job = queue.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id!r}")
        return job

    @app.post("/api/jobs", status_code=201)
    def create_job(body: JobRequest):
        if body.experiment not in catalog.experiment_scripts():
            raise HTTPException(422, f"unknown experiment {body.experiment!r}")
        return queue.submit(body.experiment, body.profile, body.workers).summary()

    @app.get("/api/jobs")
    def list_jobs():
        return [j.summary() for j in reversed(list(queue.jobs.values()))]

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str):
        return get_job(job_id).full()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = get_job(job_id)
        queue.cancel(job)
        return job.summary()

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job(job_id: str):
        job = get_job(job_id)

        async def events():
            sent = 0
            yield sse("init", job.summary())
            while True:
                over = job.over  # read first: lines written before the end are still sent below
                lines = job.log[sent:]
                if lines:
                    sent += len(lines)
                    yield sse("lines", lines)
                if over:
                    break
                await asyncio.sleep(STREAM_TICK_S)
            yield sse("done", job.summary())

        return event_stream(events())

    # --------------------------------------------------------------- frontend
    for folder in ("figures", "screenshots"):
        if (catalog.DOCS / folder).exists():
            app.mount(f"/{folder}", StaticFiles(directory=catalog.DOCS / folder), name=folder)

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "unknown API route")
        file = (STATIC / path).resolve()
        if path and file.is_file() and file.is_relative_to(STATIC):
            cache = {"Cache-Control": "public, max-age=31536000, immutable"} if path.startswith("assets/") else None
            return FileResponse(file, headers=cache)
        index = STATIC / "index.html"
        if index.exists():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse(NOT_BUILT)

    return app

