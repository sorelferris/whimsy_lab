import asyncio

import pytest
from PIL import Image

from image_gen_viz.model import DecodedFrame
from image_gen_viz.storage import RunStorage
from image_gen_viz.tasks import GenerationManager
from image_gen_viz.validation import GenerationRequest


class FakeModel:
    def generate(self, request, on_frame):
        on_frame(DecodedFrame(step=2, image=Image.new("RGB", (8, 8), "red"), final=False))
        on_frame(DecodedFrame(step=4, image=Image.new("RGB", (8, 8), "blue"), final=True))
        return Image.new("RGB", (8, 8), "blue")


@pytest.mark.asyncio
async def test_manager_streams_generation_events(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=FakeModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox", steps=4, decode_interval=2))

    events = []
    async for event in manager.events(run_id):
        events.append(event)
        if event.type == "complete":
            break

    assert [event.type for event in events] == ["started", "frame", "frame", "complete"]
    metadata = manager.storage.load_run(run_id)
    assert metadata["status"] == "completed"
    assert len(metadata["frames"]) == 2


@pytest.mark.asyncio
async def test_manager_removes_queue_after_terminal_event_is_consumed(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=FakeModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox", steps=4, decode_interval=2))

    async for event in manager.events(run_id):
        if event.type == "complete":
            break

    assert run_id not in manager.queues


@pytest.mark.asyncio
async def test_manager_rejects_unknown_run_events(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=FakeModel())

    with pytest.raises(ValueError, match="unknown run_id"):
        async for _event in manager.events("missing"):
            pass


@pytest.mark.asyncio
async def test_manager_rejects_concurrent_generation(tmp_path):
    class SlowModel:
        def generate(self, request, on_frame):
            import time
            time.sleep(0.2)
            return Image.new("RGB", (8, 8), "green")

    manager = GenerationManager(storage=RunStorage(tmp_path), model=SlowModel())
    first_run_id = await manager.start(GenerationRequest(prompt="first"))

    with pytest.raises(RuntimeError, match="already running"):
        await manager.start(GenerationRequest(prompt="second"))

    async for event in manager.events(first_run_id):
        if event.type == "complete":
            break
