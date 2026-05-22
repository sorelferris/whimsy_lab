import json
from dataclasses import dataclass
from typing import Any, Literal

EventType = Literal["started", "progress", "frame", "complete", "error"]


@dataclass(frozen=True)
class GenerationEvent:
    type: EventType
    payload: dict[str, Any]

    @classmethod
    def started(cls, run_id: str) -> "GenerationEvent":
        return cls("started", {"run_id": run_id})

    @classmethod
    def progress(cls, run_id: str, step: int, total_steps: int) -> "GenerationEvent":
        return cls("progress", {"run_id": run_id, "step": step, "total_steps": total_steps})

    @classmethod
    def frame(cls, run_id: str, step: int, url: str, final: bool) -> "GenerationEvent":
        return cls("frame", {"run_id": run_id, "step": step, "url": url, "final": final})

    @classmethod
    def complete(cls, run_id: str) -> "GenerationEvent":
        return cls("complete", {"run_id": run_id})

    @classmethod
    def error(cls, run_id: str, message: str) -> "GenerationEvent":
        return cls("error", {"run_id": run_id, "message": message})


def should_decode_step(step: int, total_steps: int, decode_interval: int) -> bool:
    return step % decode_interval == 0 or step == total_steps


def format_sse(event: GenerationEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.payload)}\n\n"
