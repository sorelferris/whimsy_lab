import asyncio

import pytest
from PIL import Image

from image_gen_viz.model import DecodedFrame
from image_gen_viz.storage import RunStorage
from image_gen_viz.tasks import GenerationManager
from image_gen_viz.validation import GenerationRequest


class FakeModel:
    def generate(self, request, on_frame, on_progress=None):
        for step in range(1, request.steps + 1):
            if on_progress is not None:
                on_progress(step)
            if step == 2:
                on_frame(DecodedFrame(step=2, image=Image.new("RGB", (8, 8), "red"), final=False))
            if step == 4:
                on_frame(DecodedFrame(step=4, image=Image.new("RGB", (8, 8), "blue"), final=True))
        return Image.new("RGB", (8, 8), "blue")


class OomModel:
    def generate(self, request, on_frame, on_progress=None):
        raise RuntimeError("CUDA out of memory")


async def collect_events(manager, run_id):
    events = []
    async with asyncio.timeout(1):
        async for event in manager.events(run_id):
            events.append(event)
            if event.type in {"complete", "error"}:
                break
    return events


@pytest.mark.asyncio
async def test_manager_streams_generation_events(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=FakeModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox", steps=4, decode_interval=2))

    events = await collect_events(manager, run_id)

    assert [event.type for event in events] == [
        "started",
        "progress",
        "progress",
        "frame",
        "progress",
        "progress",
        "frame",
        "complete",
    ]
    progress_events = [event for event in events if event.type == "progress"]
    assert [event.payload["step"] for event in progress_events] == [1, 2, 3, 4]
    assert all(event.payload["total_steps"] == 4 for event in progress_events)
    metadata = manager.storage.load_run(run_id)
    assert metadata["status"] == "completed"
    assert len(metadata["frames"]) == 2


@pytest.mark.asyncio
async def test_manager_removes_queue_after_terminal_event_is_consumed(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=FakeModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox", steps=4, decode_interval=2))

    await collect_events(manager, run_id)

    assert run_id not in manager.queues


@pytest.mark.asyncio
async def test_manager_converts_cuda_oom_to_friendly_error(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=OomModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox"))

    events = await collect_events(manager, run_id)

    assert events[-1].payload["message"] == "GPU out of memory. Lower resolution, reduce steps, or close other GPU workloads."
    assert manager.storage.load_run(run_id)["status"] == "error"


@pytest.mark.asyncio
async def test_manager_rejects_unknown_run_events(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=FakeModel())

    with pytest.raises(ValueError, match="unknown run_id"):
        async for _event in manager.events("missing"):
            pass


@pytest.mark.asyncio
async def test_manager_rejects_concurrent_generation(tmp_path):
    class SlowModel:
        def generate(self, request, on_frame, on_progress=None):
            import time
            time.sleep(0.2)
            return Image.new("RGB", (8, 8), "green")

    manager = GenerationManager(storage=RunStorage(tmp_path), model=SlowModel())
    first_run_id = await manager.start(GenerationRequest(prompt="first"))

    with pytest.raises(RuntimeError, match="already running"):
        await manager.start(GenerationRequest(prompt="second"))

    await collect_events(manager, first_run_id)
