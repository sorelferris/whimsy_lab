from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from image_gen_viz.config import RUNS_DIR, STATIC_DIR
from image_gen_viz.events import format_sse
from image_gen_viz.model import StableDiffusionModel
from image_gen_viz.storage import RunStorage
from image_gen_viz.tasks import GenerationManager
from image_gen_viz.validation import GenerationRequest


def create_app(manager: GenerationManager | None = None, runs_dir: Path = RUNS_DIR) -> FastAPI:
    app = FastAPI(title="Image Gen Viz")
    storage = manager.storage if manager is not None else RunStorage(runs_dir)
    generation_manager = manager or GenerationManager(storage=storage, model=StableDiffusionModel())

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    app.mount("/runs", StaticFiles(directory=runs_dir), name="runs")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/runs")
    async def create_run(request: GenerationRequest):
        try:
            run_id = await generation_manager.start(request)
            return {"run_id": run_id}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return storage.load_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid run_id") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        if run_id not in generation_manager.queues:
            raise HTTPException(status_code=404, detail="run not found")

        async def stream():
            async for event in generation_manager.events(run_id):
                yield format_sse(event)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
