import json
import pytest

from PIL import Image

from image_gen_viz.storage import RunStorage
from image_gen_viz.validation import GenerationRequest


def test_create_run_writes_request_metadata(tmp_path):
    storage = RunStorage(tmp_path)
    request = GenerationRequest(prompt="a red fox", steps=12, decode_interval=3)

    run = storage.create_run(request)

    metadata_path = tmp_path / run.run_id / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["run_id"] == run.run_id
    assert metadata["request"]["prompt"] == "a red fox"
    assert metadata["frames"] == []
    assert metadata["status"] == "created"


def test_save_frame_updates_metadata(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run(GenerationRequest(prompt="a red fox"))
    image = Image.new("RGB", (8, 8), "red")

    frame = storage.save_frame(run.run_id, step=4, image=image, final=False)

    metadata = storage.load_run(run.run_id)
    assert frame.step == 4
    assert frame.url.endswith("/frames/step_0004.png")
    assert (tmp_path / run.run_id / "frames" / "step_0004.png").exists()
    assert metadata["frames"] == [{"step": 4, "url": frame.url, "final": False}]


def test_save_final_frame_marks_final_image(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run(GenerationRequest(prompt="a red fox"))
    image = Image.new("RGB", (8, 8), "blue")

    frame = storage.save_frame(run.run_id, step=10, image=image, final=True)

    metadata = storage.load_run(run.run_id)
    assert frame.final is True
    assert metadata["final_image"] == frame.url
    assert metadata["status"] == "completed"


def test_invalid_run_id_is_rejected(tmp_path):
    storage = RunStorage(tmp_path)
    image = Image.new("RGB", (8, 8), "red")

    with pytest.raises(ValueError, match="invalid run_id"):
        storage.load_run("../../etc/passwd")

    with pytest.raises(ValueError, match="invalid run_id"):
        storage.save_frame("not-a-run-id", step=1, image=image, final=False)


def test_mark_running_and_error_update_metadata(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run(GenerationRequest(prompt="a red fox"))

    storage.mark_running(run.run_id)
    assert storage.load_run(run.run_id)["status"] == "running"

    storage.mark_error(run.run_id, "boom")
    metadata = storage.load_run(run.run_id)
    assert metadata["status"] == "error"
    assert metadata["error"] == "boom"
