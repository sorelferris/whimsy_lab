from fastapi.testclient import TestClient
from PIL import Image

from image_gen_viz.model import DecodedFrame
from image_gen_viz.storage import RunStorage
from image_gen_viz.tasks import GenerationManager
from image_gen_viz.web import create_app


class FakeModel:
    def generate(self, request, on_frame):
        on_frame(DecodedFrame(step=request.decode_interval, image=Image.new("RGB", (8, 8), "red"), final=False))
        return Image.new("RGB", (8, 8), "green")


def test_create_run_returns_run_id(tmp_path):
    manager = GenerationManager(RunStorage(tmp_path), FakeModel())
    app = create_app(manager=manager, runs_dir=tmp_path)
    client = TestClient(app)

    response = client.post("/api/runs", json={"prompt": "a fox", "steps": 4, "decode_interval": 2})

    assert response.status_code == 200
    assert "run_id" in response.json()


def test_get_run_returns_saved_metadata(tmp_path):
    manager = GenerationManager(RunStorage(tmp_path), FakeModel())
    app = create_app(manager=manager, runs_dir=tmp_path)
    client = TestClient(app)
    run_id = client.post("/api/runs", json={"prompt": "a fox", "steps": 4, "decode_interval": 2}).json()["run_id"]

    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["run_id"] == run_id
    assert response.json()["request"]["prompt"] == "a fox"


def test_static_index_is_served(tmp_path):
    manager = GenerationManager(RunStorage(tmp_path), FakeModel())
    app = create_app(manager=manager, runs_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
