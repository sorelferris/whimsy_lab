import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from image_gen_viz.validation import GenerationRequest

RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: Path


@dataclass(frozen=True)
class FrameRecord:
    step: int
    url: str
    final: bool


class RunStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _run_path(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.match(run_id):
            raise ValueError("invalid run_id")
        return self.root / run_id

    def _load_metadata(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id) / "metadata.json"
        return json.loads(path.read_text())

    def _write_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        path = self._run_path(run_id) / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def create_run(self, request: GenerationRequest) -> RunRecord:
        run_id = uuid.uuid4().hex
        run_path = self._run_path(run_id)
        (run_path / "frames").mkdir(parents=True)
        metadata = {
            "run_id": run_id,
            "status": "created",
            "request": request.model_dump(),
            "frames": [],
            "final_image": None,
            "error": None,
        }
        self._write_metadata(run_id, metadata)
        return RunRecord(run_id=run_id, path=run_path)

    def save_frame(self, run_id: str, step: int, image: Image.Image, final: bool) -> FrameRecord:
        with self._lock:
            frame_name = f"step_{step:04d}.png"
            frame_path = self._run_path(run_id) / "frames" / frame_name
            image.save(frame_path)
            frame = FrameRecord(step=step, url=f"/runs/{run_id}/frames/{frame_name}", final=final)
            metadata = self._load_metadata(run_id)
            metadata["frames"].append({"step": step, "url": frame.url, "final": final})
            if final:
                metadata["final_image"] = frame.url
                metadata["status"] = "completed"
            self._write_metadata(run_id, metadata)
            return frame

    def mark_running(self, run_id: str) -> None:
        with self._lock:
            metadata = self._load_metadata(run_id)
            metadata["status"] = "running"
            self._write_metadata(run_id, metadata)

    def mark_error(self, run_id: str, message: str) -> None:
        with self._lock:
            metadata = self._load_metadata(run_id)
            metadata["status"] = "error"
            metadata["error"] = message
            self._write_metadata(run_id, metadata)

    def load_run(self, run_id: str) -> dict[str, Any]:
        return self._load_metadata(run_id)
