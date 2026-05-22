import asyncio
from typing import Protocol

from PIL import Image

from image_gen_viz.events import GenerationEvent
from image_gen_viz.model import DecodedFrame
from image_gen_viz.storage import RunStorage
from image_gen_viz.validation import GenerationRequest


class ModelService(Protocol):
    def generate(self, request: GenerationRequest, on_frame) -> Image.Image:
        ...


class GenerationManager:
    def __init__(self, storage: RunStorage, model: ModelService) -> None:
        self.storage = storage
        self.model = model
        self.active_task: asyncio.Task | None = None
        self.queues: dict[str, asyncio.Queue[GenerationEvent]] = {}

    async def start(self, request: GenerationRequest) -> str:
        if self.active_task is not None and not self.active_task.done():
            raise RuntimeError("generation already running")
        run = self.storage.create_run(request)
        queue: asyncio.Queue[GenerationEvent] = asyncio.Queue()
        self.queues[run.run_id] = queue
        self.active_task = asyncio.create_task(self._run_generation(run.run_id, request, queue))
        return run.run_id

    async def events(self, run_id: str):
        queue = self.queues[run_id]
        while True:
            event = await queue.get()
            yield event
            if event.type in {"complete", "error"}:
                break

    async def _run_generation(self, run_id: str, request: GenerationRequest, queue: asyncio.Queue[GenerationEvent]) -> None:
        loop = asyncio.get_running_loop()
        self.storage.mark_running(run_id)
        await queue.put(GenerationEvent.started(run_id))

        def emit(event: GenerationEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def on_frame(frame: DecodedFrame) -> None:
            saved = self.storage.save_frame(run_id, frame.step, frame.image, frame.final)
            emit(GenerationEvent.frame(run_id, saved.step, saved.url, saved.final))

        try:
            final_image = await asyncio.to_thread(self.model.generate, request, on_frame)
            metadata = self.storage.load_run(run_id)
            if metadata["final_image"] is None:
                saved = self.storage.save_frame(run_id, request.steps, final_image, True)
                await queue.put(GenerationEvent.frame(run_id, saved.step, saved.url, saved.final))
            await queue.put(GenerationEvent.complete(run_id))
        except RuntimeError as exc:
            message = self._friendly_error(str(exc))
            self.storage.mark_error(run_id, message)
            await queue.put(GenerationEvent.error(run_id, message))
        except Exception as exc:
            message = f"Generation failed: {exc}"
            self.storage.mark_error(run_id, message)
            await queue.put(GenerationEvent.error(run_id, message))

    def _friendly_error(self, message: str) -> str:
        lowered = message.lower()
        if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
            return "GPU out of memory. Lower resolution, reduce steps, or close other GPU workloads."
        return f"Generation failed: {message}"
