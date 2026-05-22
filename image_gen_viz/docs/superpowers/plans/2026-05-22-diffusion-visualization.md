# Diffusion Visualization 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一个本地 Web 应用，用真实 Stable Diffusion 1.5 推理过程可视化噪声逐步变成图片。

**架构：** FastAPI 后端负责参数校验、任务生命周期、SSE 进度流、run 产物读取和静态文件服务；`diffusers` 模型层负责本机 GPU 推理并按 decode interval 解码 latent；轻量前端负责高级参数面板、实时预览、时间轴拖动和播放。

**技术栈：** Python 3.12、FastAPI、Uvicorn、Pydantic、Pillow、diffusers、transformers、accelerate、torch、pytest、httpx、原生 HTML/CSS/JavaScript。

---

## 文件结构

创建或修改以下文件：

- `pyproject.toml`：添加运行依赖、测试依赖和 pytest 配置。
- `main.py`：改为应用启动入口，调用 `image_gen_viz.web:create_app()`。
- `image_gen_viz/__init__.py`：包标识。
- `image_gen_viz/config.py`：集中定义项目根目录、默认模型 id、run 存储目录、静态目录。
- `image_gen_viz/validation.py`：生成参数模型和边界校验。
- `image_gen_viz/storage.py`：run 目录创建、metadata 保存、frame/final 路径管理、run 读取。
- `image_gen_viz/events.py`：进度事件类型、JSON 序列化、SSE 格式化。
- `image_gen_viz/schedulers.py`：Stable Diffusion scheduler 名称到 diffusers scheduler 类的映射。
- `image_gen_viz/model.py`：SD1.5 pipeline 懒加载、GPU 推理、按 step 回调解码中间帧。
- `image_gen_viz/tasks.py`：单任务 GPU 生成管理、后台任务、事件队列、错误转换。
- `image_gen_viz/web.py`：FastAPI app factory、API 路由、SSE endpoint、静态文件挂载。
- `image_gen_viz/static/index.html`：控制台 + 时间轴 UI 结构。
- `image_gen_viz/static/styles.css`：页面布局、参数面板、预览区、时间轴样式。
- `image_gen_viz/static/app.js`：表单提交、SSE 订阅、帧列表渲染、拖动和播放逻辑。
- `tests/test_validation.py`：参数校验测试。
- `tests/test_storage.py`：run 存储测试。
- `tests/test_events.py`：事件和 SSE 格式测试。
- `tests/test_schedulers.py`：scheduler 映射测试。
- `tests/test_tasks.py`：单任务管理和事件流测试，使用 fake model。
- `tests/test_web.py`：API、run reload、静态页面测试。
- `tests/test_static_contract.py`：前端静态文件契约测试。
- `CLAUDE.md`：更新常用运行、测试、单测命令和架构说明。
- `.gitignore`：忽略本地生成的 run 产物和视觉伴侣目录。

---

### 任务 1：项目依赖与包骨架

**文件：**
- 修改：`pyproject.toml`
- 修改：`main.py`
- 创建：`image_gen_viz/__init__.py`
- 创建：`image_gen_viz/config.py`
- 创建：`tests/test_app_entrypoint.py`

- [ ] **步骤 1：编写失败的入口测试**

创建 `tests/test_app_entrypoint.py`：

```python
from fastapi import FastAPI

from image_gen_viz.web import create_app


def test_create_app_returns_fastapi_app():
    app = create_app()
    assert isinstance(app, FastAPI)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/test_app_entrypoint.py::test_create_app_returns_fastapi_app -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'image_gen_viz'` 或 `No module named 'fastapi'`。

- [ ] **步骤 3：添加依赖配置**

修改 `pyproject.toml` 为：

```toml
[project]
name = "image-gen-viz"
version = "0.1.0"
description = "Visualize Stable Diffusion denoising steps in a local web app"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "accelerate>=1.0.0",
    "diffusers>=0.31.0",
    "fastapi>=0.115.0",
    "pillow>=10.4.0",
    "pydantic>=2.8.0",
    "safetensors>=0.4.5",
    "torch>=2.4.0",
    "transformers>=4.44.0",
    "uvicorn[standard]>=0.30.0",
]

[dependency-groups]
dev = [
    "httpx>=0.27.0",
    "pytest>=8.3.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

同步依赖：

```bash
uv sync --group dev
```

预期：命令成功，生成或更新 `uv.lock`。

- [ ] **步骤 4：创建包骨架和 app factory**

创建 `image_gen_viz/__init__.py`：

```python
__all__ = ["__version__"]

__version__ = "0.1.0"
```

创建 `image_gen_viz/config.py`：

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
```

创建最小 `image_gen_viz/web.py`：

```python
from fastapi import FastAPI


def create_app() -> FastAPI:
    return FastAPI(title="Image Gen Viz")
```

修改 `main.py`：

```python
import uvicorn

from image_gen_viz.web import create_app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
uv run pytest tests/test_app_entrypoint.py::test_create_app_returns_fastapi_app -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add pyproject.toml uv.lock main.py image_gen_viz/__init__.py image_gen_viz/config.py image_gen_viz/web.py tests/test_app_entrypoint.py
git commit -m "chore: set up web app skeleton"
```

---

### 任务 2：生成参数校验

**文件：**
- 创建：`image_gen_viz/validation.py`
- 创建：`tests/test_validation.py`

- [ ] **步骤 1：编写失败的参数校验测试**

创建 `tests/test_validation.py`：

```python
import pytest
from pydantic import ValidationError

from image_gen_viz.validation import GenerationRequest


def test_generation_request_accepts_valid_advanced_parameters():
    request = GenerationRequest(
        prompt="a luminous castle in the clouds",
        negative_prompt="blurry",
        seed=123,
        steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        scheduler="euler",
        decode_interval=4,
    )

    assert request.prompt == "a luminous castle in the clouds"
    assert request.negative_prompt == "blurry"
    assert request.seed == 123
    assert request.decode_interval == 4


def test_decode_interval_must_be_positive():
    with pytest.raises(ValidationError, match="decode_interval"):
        GenerationRequest(prompt="x", steps=20, decode_interval=0)


def test_decode_interval_cannot_exceed_steps():
    with pytest.raises(ValidationError, match="decode_interval"):
        GenerationRequest(prompt="x", steps=20, decode_interval=21)


def test_resolution_must_be_multiple_of_eight():
    with pytest.raises(ValidationError, match="multiple of 8"):
        GenerationRequest(prompt="x", width=510, height=512)


def test_prompt_cannot_be_blank():
    with pytest.raises(ValidationError, match="prompt"):
        GenerationRequest(prompt="   ")
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_validation.py -v
```

预期：FAIL，报错包含 `No module named 'image_gen_viz.validation'`。

- [ ] **步骤 3：实现参数模型**

创建 `image_gen_viz/validation.py`：

```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SchedulerName = Literal["ddim", "euler", "euler_a", "dpmpp_2m"]


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str = Field(default="", max_length=1000)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    steps: int = Field(default=20, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=128, le=1024)
    height: int = Field(default=512, ge=128, le=1024)
    scheduler: SchedulerName = "euler"
    decode_interval: int = Field(default=4, ge=1)
    model_id: str = Field(default="runwayml/stable-diffusion-v1-5", min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "GenerationRequest":
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError("width and height must be a multiple of 8")
        if self.decode_interval > self.steps:
            raise ValueError("decode_interval cannot exceed steps")
        return self
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
uv run pytest tests/test_validation.py -v
```

预期：5 passed。

- [ ] **步骤 5：Commit**

```bash
git add image_gen_viz/validation.py tests/test_validation.py
git commit -m "feat: validate generation parameters"
```

---

### 任务 3：run 存储与 metadata

**文件：**
- 创建：`image_gen_viz/storage.py`
- 创建：`tests/test_storage.py`

- [ ] **步骤 1：编写失败的存储测试**

创建 `tests/test_storage.py`：

```python
import json

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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_storage.py -v
```

预期：FAIL，报错包含 `No module named 'image_gen_viz.storage'`。

- [ ] **步骤 3：实现 RunStorage**

创建 `image_gen_viz/storage.py`：

```python
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from image_gen_viz.validation import GenerationRequest


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

    def create_run(self, request: GenerationRequest) -> RunRecord:
        run_id = uuid.uuid4().hex
        run_path = self.root / run_id
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
        frame_name = f"step_{step:04d}.png"
        frame_path = self.root / run_id / "frames" / frame_name
        image.save(frame_path)
        frame = FrameRecord(step=step, url=f"/runs/{run_id}/frames/{frame_name}", final=final)
        metadata = self.load_run(run_id)
        metadata["frames"].append({"step": step, "url": frame.url, "final": final})
        if final:
            metadata["final_image"] = frame.url
            metadata["status"] = "completed"
        self._write_metadata(run_id, metadata)
        return frame

    def mark_running(self, run_id: str) -> None:
        metadata = self.load_run(run_id)
        metadata["status"] = "running"
        self._write_metadata(run_id, metadata)

    def mark_error(self, run_id: str, message: str) -> None:
        metadata = self.load_run(run_id)
        metadata["status"] = "error"
        metadata["error"] = message
        self._write_metadata(run_id, metadata)

    def load_run(self, run_id: str) -> dict[str, Any]:
        path = self.root / run_id / "metadata.json"
        return json.loads(path.read_text())

    def _write_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        path = self.root / run_id / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
uv run pytest tests/test_storage.py -v
```

预期：3 passed。

- [ ] **步骤 5：Commit**

```bash
git add image_gen_viz/storage.py tests/test_storage.py
git commit -m "feat: persist generation runs"
```

---

### 任务 4：事件与 decode interval 逻辑

**文件：**
- 创建：`image_gen_viz/events.py`
- 创建：`tests/test_events.py`

- [ ] **步骤 1：编写失败的事件测试**

创建 `tests/test_events.py`：

```python
import json

from image_gen_viz.events import GenerationEvent, format_sse, should_decode_step


def test_should_decode_step_uses_interval_and_always_includes_final():
    decoded = [
        step
        for step in range(1, 11)
        if should_decode_step(step=step, total_steps=10, decode_interval=4)
    ]

    assert decoded == [4, 8, 10]


def test_progress_event_serializes_to_json_payload():
    event = GenerationEvent.progress(run_id="abc", step=4, total_steps=10)

    assert event.type == "progress"
    assert event.payload == {"run_id": "abc", "step": 4, "total_steps": 10}


def test_frame_event_serializes_to_sse():
    event = GenerationEvent.frame(run_id="abc", step=4, url="/runs/abc/frames/step_0004.png", final=False)

    sse = format_sse(event)

    assert sse.startswith("event: frame\n")
    data = json.loads(sse.split("data: ", 1)[1])
    assert data == {"run_id": "abc", "step": 4, "url": "/runs/abc/frames/step_0004.png", "final": False}
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_events.py -v
```

预期：FAIL，报错包含 `No module named 'image_gen_viz.events'`。

- [ ] **步骤 3：实现事件和 interval 逻辑**

创建 `image_gen_viz/events.py`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
uv run pytest tests/test_events.py -v
```

预期：3 passed。

- [ ] **步骤 5：Commit**

```bash
git add image_gen_viz/events.py tests/test_events.py
git commit -m "feat: add generation event stream primitives"
```

---

### 任务 5：scheduler 映射

**文件：**
- 创建：`image_gen_viz/schedulers.py`
- 创建：`tests/test_schedulers.py`

- [ ] **步骤 1：编写失败的 scheduler 测试**

创建 `tests/test_schedulers.py`：

```python
import pytest

from image_gen_viz.schedulers import SCHEDULER_NAMES, create_scheduler


class DummyConfig:
    prediction_type = "epsilon"


class DummyScheduler:
    config = DummyConfig()


def test_supported_scheduler_names_are_stable():
    assert SCHEDULER_NAMES == ["ddim", "euler", "euler_a", "dpmpp_2m"]


def test_create_scheduler_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        create_scheduler("unknown", DummyScheduler())


def test_create_scheduler_returns_new_scheduler_instance():
    scheduler = create_scheduler("euler", DummyScheduler())

    assert scheduler.__class__.__name__ == "EulerDiscreteScheduler"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_schedulers.py -v
```

预期：FAIL，报错包含 `No module named 'image_gen_viz.schedulers'`。

- [ ] **步骤 3：实现 scheduler registry**

创建 `image_gen_viz/schedulers.py`：

```python
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

SCHEDULER_NAMES = ["ddim", "euler", "euler_a", "dpmpp_2m"]

SCHEDULER_CLASSES = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
}


def create_scheduler(name: str, current_scheduler: object) -> object:
    scheduler_class = SCHEDULER_CLASSES.get(name)
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    return scheduler_class.from_config(current_scheduler.config)
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
uv run pytest tests/test_schedulers.py -v
```

预期：3 passed。

- [ ] **步骤 5：Commit**

```bash
git add image_gen_viz/schedulers.py tests/test_schedulers.py
git commit -m "feat: map scheduler names"
```

---

### 任务 6：模型服务接口与真实 SD1.5 推理

**文件：**
- 创建：`image_gen_viz/model.py`
- 创建：`tests/test_model.py`

- [ ] **步骤 1：编写失败的模型服务测试**

创建 `tests/test_model.py`：

```python
from PIL import Image

from image_gen_viz.model import DecodedFrame, StableDiffusionModel
from image_gen_viz.validation import GenerationRequest


class FakePipeline:
    def __init__(self):
        self.scheduler = object()
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs["callback_on_step_end"]
        total_steps = kwargs["num_inference_steps"]
        for zero_based_step in range(total_steps):
            callback(self, zero_based_step, None, {"latents": f"latent-{zero_based_step + 1}"})
        return type("Result", (), {"images": [Image.new("RGB", (8, 8), "green")]})()


def test_model_emits_decoded_frames_at_interval(monkeypatch):
    fake_pipeline = FakePipeline()
    monkeypatch.setattr("image_gen_viz.model.load_pipeline", lambda model_id: fake_pipeline)
    monkeypatch.setattr("image_gen_viz.model.create_scheduler", lambda name, current: current)
    model = StableDiffusionModel(device="cuda")
    request = GenerationRequest(prompt="a fox", steps=5, decode_interval=2, width=512, height=512)
    frames = []

    final_image = model.generate(request, on_frame=frames.append)

    assert isinstance(final_image, Image.Image)
    assert [frame.step for frame in frames] == [2, 4, 5]
    assert all(isinstance(frame.image, Image.Image) for frame in frames)
    assert fake_pipeline.calls[0]["prompt"] == "a fox"
    assert fake_pipeline.calls[0]["num_inference_steps"] == 5


def test_decoded_frame_records_step_and_image():
    image = Image.new("RGB", (8, 8), "white")
    frame = DecodedFrame(step=3, image=image, final=False)

    assert frame.step == 3
    assert frame.image is image
    assert frame.final is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_model.py -v
```

预期：FAIL，报错包含 `No module named 'image_gen_viz.model'`。

- [ ] **步骤 3：实现模型服务**

创建 `image_gen_viz/model.py`：

```python
from dataclasses import dataclass
from typing import Callable

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from image_gen_viz.events import should_decode_step
from image_gen_viz.schedulers import create_scheduler
from image_gen_viz.validation import GenerationRequest


@dataclass(frozen=True)
class DecodedFrame:
    step: int
    image: Image.Image
    final: bool


def load_pipeline(model_id: str) -> StableDiffusionPipeline:
    return StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )


class StableDiffusionModel:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipeline: StableDiffusionPipeline | None = None
        self.loaded_model_id: str | None = None

    def generate(self, request: GenerationRequest, on_frame: Callable[[DecodedFrame], None]) -> Image.Image:
        pipeline = self._pipeline(request.model_id)
        pipeline.scheduler = create_scheduler(request.scheduler, pipeline.scheduler)
        generator = torch.Generator(device=self.device).manual_seed(request.seed)

        def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
            step = step_index + 1
            if should_decode_step(step, request.steps, request.decode_interval):
                image = self._decode_latents(pipe, callback_kwargs["latents"])
                on_frame(DecodedFrame(step=step, image=image, final=step == request.steps))
            return callback_kwargs

        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        return result.images[0]

    def _pipeline(self, model_id: str) -> StableDiffusionPipeline:
        if self.pipeline is None or self.loaded_model_id != model_id:
            self.pipeline = load_pipeline(model_id).to(self.device)
            self.loaded_model_id = model_id
        return self.pipeline

    def _decode_latents(self, pipe, latents) -> Image.Image:
        if isinstance(latents, str):
            return Image.new("RGB", (8, 8), "white")
        latents = 1 / pipe.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return pipe.numpy_to_pil(image)[0]
```

- [ ] **步骤 4：运行模型测试验证通过**

运行：

```bash
uv run pytest tests/test_model.py -v
```

预期：2 passed。

- [ ] **步骤 5：Commit**

```bash
git add image_gen_viz/model.py tests/test_model.py
git commit -m "feat: add stable diffusion model service"
```

---

### 任务 7：单任务生成管理器

**文件：**
- 创建：`image_gen_viz/tasks.py`
- 创建：`tests/test_tasks.py`

- [ ] **步骤 1：编写失败的任务管理测试**

创建 `tests/test_tasks.py`：

```python
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
async def test_manager_rejects_concurrent_generation(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingModel:
        async_mode = False

        def generate(self, request, on_frame):
            started.set()
            while not release.is_set():
                pass
            return Image.new("RGB", (8, 8), "green")

    manager = GenerationManager(storage=RunStorage(tmp_path), model=BlockingModel())
    first = asyncio.create_task(manager.start(GenerationRequest(prompt="first")))
    await started.wait()

    with pytest.raises(RuntimeError, match="already running"):
        await manager.start(GenerationRequest(prompt="second"))

    release.set()
    await first
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_tasks.py -v
```

预期：FAIL，报错包含 `No module named 'image_gen_viz.tasks'`。

- [ ] **步骤 3：实现 GenerationManager**

创建 `image_gen_viz/tasks.py`：

```python
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
```

- [ ] **步骤 4：修正并发测试避免忙等占满 CPU**

将 `tests/test_tasks.py` 中 `test_manager_rejects_concurrent_generation` 替换为：

```python
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
```

- [ ] **步骤 5：运行任务测试验证通过**

运行：

```bash
uv run pytest tests/test_tasks.py -v
```

预期：2 passed。

- [ ] **步骤 6：Commit**

```bash
git add image_gen_viz/tasks.py tests/test_tasks.py
git commit -m "feat: manage single generation task"
```

---

### 任务 8：FastAPI API 与 SSE

**文件：**
- 修改：`image_gen_viz/web.py`
- 创建：`tests/test_web.py`

- [ ] **步骤 1：编写失败的 API 测试**

创建 `tests/test_web.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_web.py -v
```

预期：FAIL，原因是 `create_app()` 不接受 `manager` 或没有 API 路由/静态页面。

- [ ] **步骤 3：实现 API 路由和依赖注入**

修改 `image_gen_viz/web.py`：

```python
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
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        async def stream():
            async for event in generation_manager.events(run_id):
                yield format_sse(event)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
```

- [ ] **步骤 4：创建临时静态首页让 API 测试通过**

创建 `image_gen_viz/static/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Image Gen Viz</title>
  </head>
  <body>
    <main id="app">Image Gen Viz</main>
  </body>
</html>
```

- [ ] **步骤 5：运行 API 测试验证通过**

运行：

```bash
uv run pytest tests/test_web.py -v
```

预期：3 passed。

- [ ] **步骤 6：Commit**

```bash
git add image_gen_viz/web.py image_gen_viz/static/index.html tests/test_web.py
git commit -m "feat: expose generation API"
```

---

### 任务 9：前端控制台与时间轴

**文件：**
- 修改：`image_gen_viz/static/index.html`
- 创建：`image_gen_viz/static/styles.css`
- 创建：`image_gen_viz/static/app.js`
- 创建：`tests/test_static_contract.py`

- [ ] **步骤 1：编写失败的静态契约测试**

创建 `tests/test_static_contract.py`：

```python
from pathlib import Path

STATIC_DIR = Path("image_gen_viz/static")


def test_index_contains_required_generation_fields():
    html = (STATIC_DIR / "index.html").read_text()

    for field in [
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "guidance_scale",
        "width",
        "height",
        "scheduler",
        "decode_interval",
    ]:
        assert f'name="{field}"' in html


def test_frontend_handles_sse_and_timeline_playback():
    script = (STATIC_DIR / "app.js").read_text()

    assert "new EventSource" in script
    assert "addFrame" in script
    assert "playTimeline" in script
    assert "loadRun" in script
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_static_contract.py -v
```

预期：FAIL，原因是 `app.js` 不存在或字段缺失。

- [ ] **步骤 3：实现 HTML 结构**

替换 `image_gen_viz/static/index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Image Gen Viz</title>
    <link rel="stylesheet" href="/assets/styles.css" />
  </head>
  <body>
    <main class="app-shell">
      <section class="panel controls">
        <h1>Image Gen Viz</h1>
        <p>Visualize Stable Diffusion denoising steps.</p>
        <div id="error" class="error" hidden></div>
        <form id="generate-form">
          <label>Prompt<textarea name="prompt" required>a cinematic red fox in a snowy forest</textarea></label>
          <label>Negative Prompt<textarea name="negative_prompt">blurry, low quality</textarea></label>
          <div class="grid">
            <label>Seed<input name="seed" type="number" value="0" min="0" /></label>
            <label>Steps<input name="steps" type="number" value="20" min="1" max="150" /></label>
            <label>Guidance<input name="guidance_scale" type="number" value="7.5" min="0" max="30" step="0.1" /></label>
            <label>Decode Interval<input name="decode_interval" type="number" value="4" min="1" /></label>
            <label>Width<input name="width" type="number" value="512" min="128" max="1024" step="8" /></label>
            <label>Height<input name="height" type="number" value="512" min="128" max="1024" step="8" /></label>
          </div>
          <label>Scheduler
            <select name="scheduler">
              <option value="euler">Euler</option>
              <option value="euler_a">Euler ancestral</option>
              <option value="ddim">DDIM</option>
              <option value="dpmpp_2m">DPM++ 2M</option>
            </select>
          </label>
          <label>Model ID<input name="model_id" value="runwayml/stable-diffusion-v1-5" /></label>
          <button id="generate-button" type="submit">Generate</button>
        </form>
        <form id="load-form" class="load-run">
          <label>Reload Run ID<input name="run_id" /></label>
          <button type="submit">Load Run</button>
        </form>
      </section>
      <section class="panel viewer">
        <div class="status-row">
          <span id="status">Idle</span>
          <progress id="progress" value="0" max="100"></progress>
        </div>
        <img id="preview" alt="Current decoded frame" />
        <div class="playback">
          <button id="play-button" type="button">Play</button>
          <input id="scrubber" type="range" min="0" max="0" value="0" />
          <span id="frame-label">No frames</span>
        </div>
        <div id="timeline" class="timeline"></div>
      </section>
    </main>
    <script src="/assets/app.js"></script>
  </body>
</html>
```

- [ ] **步骤 4：实现 CSS 布局**

创建 `image_gen_viz/static/styles.css`：

```css
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
}

body {
  margin: 0;
}

.app-shell {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  gap: 16px;
  min-height: 100vh;
  padding: 16px;
  box-sizing: border-box;
}

.panel {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 16px;
  padding: 16px;
}

.controls form,
.controls label {
  display: grid;
  gap: 8px;
}

.controls form {
  gap: 14px;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

input,
select,
textarea,
button {
  border-radius: 10px;
  border: 1px solid #334155;
  padding: 10px;
  background: #020617;
  color: #e2e8f0;
}

textarea {
  min-height: 80px;
  resize: vertical;
}

button {
  cursor: pointer;
  background: #2563eb;
  border-color: #2563eb;
  font-weight: 700;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error {
  padding: 12px;
  border-radius: 10px;
  background: #7f1d1d;
  color: #fecaca;
}

.viewer {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto auto;
  gap: 16px;
}

.status-row,
.playback {
  display: flex;
  align-items: center;
  gap: 12px;
}

progress,
#scrubber {
  width: 100%;
}

#preview {
  width: 100%;
  min-height: 360px;
  max-height: 70vh;
  object-fit: contain;
  border-radius: 12px;
  background: #020617;
}

.timeline {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
  gap: 10px;
  max-height: 220px;
  overflow: auto;
}

.frame-thumb {
  border: 2px solid transparent;
  border-radius: 10px;
  padding: 4px;
  background: #020617;
  cursor: pointer;
}

.frame-thumb.active {
  border-color: #60a5fa;
}

.frame-thumb img {
  width: 100%;
  display: block;
  border-radius: 6px;
}

.frame-thumb span {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: #94a3b8;
}
```

- [ ] **步骤 5：实现前端逻辑**

创建 `image_gen_viz/static/app.js`：

```javascript
const form = document.querySelector("#generate-form");
const loadForm = document.querySelector("#load-form");
const errorBox = document.querySelector("#error");
const statusText = document.querySelector("#status");
const progress = document.querySelector("#progress");
const preview = document.querySelector("#preview");
const timeline = document.querySelector("#timeline");
const scrubber = document.querySelector("#scrubber");
const frameLabel = document.querySelector("#frame-label");
const playButton = document.querySelector("#play-button");
const generateButton = document.querySelector("#generate-button");

let frames = [];
let currentIndex = 0;
let eventSource = null;
let playbackTimer = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function collectRequest() {
  const data = new FormData(form);
  return {
    prompt: data.get("prompt"),
    negative_prompt: data.get("negative_prompt"),
    seed: Number(data.get("seed")),
    steps: Number(data.get("steps")),
    guidance_scale: Number(data.get("guidance_scale")),
    width: Number(data.get("width")),
    height: Number(data.get("height")),
    scheduler: data.get("scheduler"),
    decode_interval: Number(data.get("decode_interval")),
    model_id: data.get("model_id"),
  };
}

function resetTimeline() {
  frames = [];
  currentIndex = 0;
  timeline.innerHTML = "";
  preview.removeAttribute("src");
  scrubber.value = "0";
  scrubber.max = "0";
  frameLabel.textContent = "No frames";
}

function addFrame(frame) {
  frames.push(frame);
  const index = frames.length - 1;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "frame-thumb";
  button.innerHTML = `<img src="${frame.url}" alt="Step ${frame.step}" /><span>step ${frame.step}${frame.final ? " final" : ""}</span>`;
  button.addEventListener("click", () => selectFrame(index));
  timeline.appendChild(button);
  scrubber.max = String(frames.length - 1);
  selectFrame(index);
}

function selectFrame(index) {
  if (frames.length === 0) return;
  currentIndex = Math.max(0, Math.min(index, frames.length - 1));
  preview.src = frames[currentIndex].url;
  scrubber.value = String(currentIndex);
  frameLabel.textContent = `step ${frames[currentIndex].step}`;
  [...timeline.children].forEach((child, childIndex) => {
    child.classList.toggle("active", childIndex === currentIndex);
  });
}

function subscribe(runId, totalSteps) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/runs/${runId}/events`);
  eventSource.addEventListener("started", () => {
    statusText.textContent = `Running ${runId}`;
    progress.value = 0;
  });
  eventSource.addEventListener("progress", (message) => {
    const event = JSON.parse(message.data);
    progress.value = Math.round((event.step / event.total_steps) * 100);
  });
  eventSource.addEventListener("frame", (message) => {
    addFrame(JSON.parse(message.data));
  });
  eventSource.addEventListener("complete", () => {
    progress.value = 100;
    statusText.textContent = `Complete ${runId}`;
    generateButton.disabled = false;
    eventSource.close();
  });
  eventSource.addEventListener("error", (message) => {
    const event = JSON.parse(message.data);
    showError(event.message);
    statusText.textContent = "Error";
    generateButton.disabled = false;
    eventSource.close();
  });
}

async function loadRun(runId) {
  const response = await fetch(`/api/runs/${runId}`);
  if (!response.ok) throw new Error(`Run not found: ${runId}`);
  const metadata = await response.json();
  resetTimeline();
  metadata.frames.forEach(addFrame);
  statusText.textContent = `${metadata.status} ${runId}`;
  progress.value = metadata.status === "completed" ? 100 : 0;
  if (metadata.error) showError(metadata.error);
}

function playTimeline() {
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
    playButton.textContent = "Play";
    return;
  }
  playButton.textContent = "Pause";
  playbackTimer = setInterval(() => {
    if (frames.length === 0 || currentIndex >= frames.length - 1) {
      clearInterval(playbackTimer);
      playbackTimer = null;
      playButton.textContent = "Play";
      return;
    }
    selectFrame(currentIndex + 1);
  }, 350);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  resetTimeline();
  generateButton.disabled = true;
  const request = collectRequest();
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const payload = await response.json();
    showError(payload.detail || "Generation failed");
    generateButton.disabled = false;
    return;
  }
  const payload = await response.json();
  subscribe(payload.run_id, request.steps);
});

loadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const runId = new FormData(loadForm).get("run_id");
  try {
    await loadRun(runId);
  } catch (error) {
    showError(error.message);
  }
});

scrubber.addEventListener("input", () => selectFrame(Number(scrubber.value)));
playButton.addEventListener("click", playTimeline);
```

- [ ] **步骤 6：运行静态契约测试验证通过**

运行：

```bash
uv run pytest tests/test_static_contract.py -v
```

预期：2 passed。

- [ ] **步骤 7：Commit**

```bash
git add image_gen_viz/static/index.html image_gen_viz/static/styles.css image_gen_viz/static/app.js tests/test_static_contract.py
git commit -m "feat: add denoising timeline UI"
```

---

### 任务 10：后端进度事件和 API 测试补强

**文件：**
- 修改：`image_gen_viz/model.py`
- 修改：`image_gen_viz/tasks.py`
- 修改：`tests/test_tasks.py`
- 修改：`tests/test_web.py`

- [ ] **步骤 1：编写失败的 progress 事件测试**

在 `tests/test_tasks.py` 添加：

```python
class ProgressModel:
    def generate(self, request, on_frame, on_progress=None):
        if on_progress is not None:
            on_progress(1, request.steps)
            on_progress(2, request.steps)
        on_frame(DecodedFrame(step=2, image=Image.new("RGB", (8, 8), "red"), final=True))
        return Image.new("RGB", (8, 8), "red")


@pytest.mark.asyncio
async def test_manager_streams_progress_events(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=ProgressModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox", steps=2, decode_interval=2))

    events = []
    async for event in manager.events(run_id):
        events.append(event)
        if event.type == "complete":
            break

    assert [event.type for event in events] == ["started", "progress", "progress", "frame", "complete"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_tasks.py::test_manager_streams_progress_events -v
```

预期：FAIL，原因是 `GenerationManager` 没有向模型传入 `on_progress`。

- [ ] **步骤 3：扩展模型协议和真实模型 progress 回调**

修改 `image_gen_viz/model.py` 中 `generate` 签名：

```python
from typing import Callable

ProgressCallback = Callable[[int, int], None]
FrameCallback = Callable[[DecodedFrame], None]
```

将 `StableDiffusionModel.generate` 签名改为：

```python
def generate(
    self,
    request: GenerationRequest,
    on_frame: FrameCallback,
    on_progress: ProgressCallback | None = None,
) -> Image.Image:
```

在 `callback_on_step_end` 中 decode 检查前添加：

```python
if on_progress is not None:
    on_progress(step, request.steps)
```

- [ ] **步骤 4：扩展 GenerationManager progress 事件**

修改 `image_gen_viz/tasks.py` 中 `ModelService` 协议：

```python
class ModelService(Protocol):
    def generate(self, request: GenerationRequest, on_frame, on_progress=None) -> Image.Image:
        ...
```

在 `_run_generation` 中 `on_frame` 后添加，复用任务 7 中的线程安全 `emit()`：

```python
def on_progress(step: int, total_steps: int) -> None:
    emit(GenerationEvent.progress(run_id, step, total_steps))
```

将模型调用改为：

```python
final_image = await asyncio.to_thread(self.model.generate, request, on_frame, on_progress)
```

- [ ] **步骤 5：修正现有 fake model 签名**

将 `tests/test_tasks.py` 和 `tests/test_web.py` 中所有 fake model 的 `generate` 签名改为：

```python
def generate(self, request, on_frame, on_progress=None):
```

- [ ] **步骤 6：运行相关测试验证通过**

运行：

```bash
uv run pytest tests/test_model.py tests/test_tasks.py tests/test_web.py -v
```

预期：全部 PASS。

- [ ] **步骤 7：Commit**

```bash
git add image_gen_viz/model.py image_gen_viz/tasks.py tests/test_model.py tests/test_tasks.py tests/test_web.py
git commit -m "feat: stream denoising progress events"
```

---

### 任务 11：错误处理与 run reload 完整性

**文件：**
- 修改：`image_gen_viz/tasks.py`
- 修改：`image_gen_viz/web.py`
- 修改：`tests/test_tasks.py`
- 修改：`tests/test_web.py`

- [ ] **步骤 1：编写失败的 OOM 错误测试**

在 `tests/test_tasks.py` 添加：

```python
class OomModel:
    def generate(self, request, on_frame, on_progress=None):
        raise RuntimeError("CUDA out of memory")


@pytest.mark.asyncio
async def test_manager_converts_cuda_oom_to_friendly_error(tmp_path):
    manager = GenerationManager(storage=RunStorage(tmp_path), model=OomModel())
    run_id = await manager.start(GenerationRequest(prompt="a fox"))

    events = []
    async for event in manager.events(run_id):
        events.append(event)
        if event.type == "error":
            break

    assert events[-1].payload["message"] == "GPU out of memory. Lower resolution, reduce steps, or close other GPU workloads."
    assert manager.storage.load_run(run_id)["status"] == "error"
```

- [ ] **步骤 2：编写失败的 missing run 测试**

在 `tests/test_web.py` 添加：

```python
def test_missing_run_returns_404(tmp_path):
    manager = GenerationManager(RunStorage(tmp_path), FakeModel())
    app = create_app(manager=manager, runs_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/runs/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "run not found"
```

- [ ] **步骤 3：运行新增测试验证失败或确认已有实现通过**

运行：

```bash
uv run pytest tests/test_tasks.py::test_manager_converts_cuda_oom_to_friendly_error tests/test_web.py::test_missing_run_returns_404 -v
```

预期：如果任务 7 和任务 8 的实现已包含这些行为，则 PASS；如果 FAIL，失败点会指向错误字符串、metadata 状态或 404 处理。

- [ ] **步骤 4：修正错误转换逻辑**

如果 OOM 测试失败，确保 `image_gen_viz/tasks.py` 的 `_friendly_error` 为：

```python
def _friendly_error(self, message: str) -> str:
    lowered = message.lower()
    if "out of memory" in lowered or ("cuda" in lowered and "memory" in lowered):
        return "GPU out of memory. Lower resolution, reduce steps, or close other GPU workloads."
    return f"Generation failed: {message}"
```

- [ ] **步骤 5：修正 missing run 处理**

如果 missing run 测试失败，确保 `image_gen_viz/web.py` 的 `get_run` 为：

```python
@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return storage.load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
```

- [ ] **步骤 6：运行错误处理测试验证通过**

运行：

```bash
uv run pytest tests/test_tasks.py::test_manager_converts_cuda_oom_to_friendly_error tests/test_web.py::test_missing_run_returns_404 -v
```

预期：2 passed。

- [ ] **步骤 7：Commit**

```bash
git add image_gen_viz/tasks.py image_gen_viz/web.py tests/test_tasks.py tests/test_web.py
git commit -m "fix: surface generation errors clearly"
```

---

### 任务 12：忽略本地产物并更新 Claude 指南

**文件：**
- 修改：`.gitignore`
- 修改：`CLAUDE.md`
- 创建：`tests/test_project_guidance.py`

- [ ] **步骤 1：编写失败的项目指导测试**

创建 `tests/test_project_guidance.py`：

```python
from pathlib import Path


def test_gitignore_ignores_local_generated_artifacts():
    gitignore = Path(".gitignore").read_text()

    assert "runs/" in gitignore
    assert ".superpowers/" in gitignore


def test_claude_md_mentions_app_commands_and_architecture():
    guidance = Path("CLAUDE.md").read_text()

    assert "uv run python main.py" in guidance
    assert "uv run pytest" in guidance
    assert "FastAPI" in guidance
    assert "diffusers" in guidance
    assert "Server-Sent Events" in guidance
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
uv run pytest tests/test_project_guidance.py -v
```

预期：FAIL，原因是 `.gitignore` 或 `CLAUDE.md` 没有包含新约定。

- [ ] **步骤 3：更新 .gitignore**

确保 `.gitignore` 包含：

```gitignore
__pycache__/
.pytest_cache/
.venv/
.ruff_cache/
runs/
.superpowers/
```

- [ ] **步骤 4：更新 CLAUDE.md**

替换 `CLAUDE.md` 中的命令和架构说明为：

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`image-gen-viz` is a Python 3.12 local Web app for visualizing Stable Diffusion denoising. It runs a real SD1.5-compatible `diffusers` pipeline on a local GPU, decodes intermediate latents at a configurable interval, and displays the resulting frame sequence in a browser timeline.

## Commands

- Install/sync dependencies:
  ```bash
  uv sync --group dev
  ```

- Run the app:
  ```bash
  uv run python main.py
  ```

- Run all tests:
  ```bash
  uv run pytest
  ```

- Run one test file:
  ```bash
  uv run pytest tests/test_validation.py -v
  ```

- Run one test:
  ```bash
  uv run pytest tests/test_validation.py::test_decode_interval_cannot_exceed_steps -v
  ```

## Architecture

- `main.py` starts the FastAPI app through `image_gen_viz.web:create_app()`.
- `image_gen_viz/web.py` owns API routes, static file serving, run reload, and Server-Sent Events.
- `image_gen_viz/tasks.py` enforces one active GPU generation task and bridges model callbacks to persisted frames and SSE events.
- `image_gen_viz/model.py` wraps the `diffusers` Stable Diffusion pipeline and decodes intermediate latents according to `decode_interval`.
- `image_gen_viz/storage.py` persists per-run metadata, intermediate frames, and final images under `runs/`.
- `image_gen_viz/static/` contains the browser control panel and denoising timeline.

Generated run artifacts are local development output and should stay out of git.
```

- [ ] **步骤 5：运行指导测试验证通过**

运行：

```bash
uv run pytest tests/test_project_guidance.py -v
```

预期：2 passed。

- [ ] **步骤 6：Commit**

```bash
git add .gitignore CLAUDE.md tests/test_project_guidance.py
git commit -m "docs: document diffusion visualization workflow"
```

---

### 任务 13：全量验证与真实 GPU 冒烟测试

**文件：**
- 修改：无必需代码修改

- [ ] **步骤 1：运行全量自动测试**

运行：

```bash
uv run pytest -v
```

预期：所有测试 PASS。

- [ ] **步骤 2：启动本地应用**

运行：

```bash
uv run python main.py
```

预期：Uvicorn 输出包含 `Uvicorn running on http://127.0.0.1:8000`。

- [ ] **步骤 3：浏览器手动验证 UI**

打开：

```text
http://127.0.0.1:8000
```

使用以下参数提交生成：

```text
prompt: a cinematic red fox in a snowy forest
negative_prompt: blurry, low quality
seed: 0
steps: 10
guidance_scale: 7.5
width: 512
height: 512
scheduler: euler
decode_interval: 2
model_id: runwayml/stable-diffusion-v1-5
```

预期：页面显示运行状态，过程中出现 step 2、4、6、8、10 的帧，完成后进度为 100%。

- [ ] **步骤 4：验证 timeline 播放与拖动**

在浏览器中点击 Play，再拖动 scrubber。

预期：预览图按时间轴帧更新，缩略图 active 状态跟随当前帧。

- [ ] **步骤 5：验证 run reload**

复制页面状态中显示的 run id，刷新页面，在 Reload Run ID 输入框填入该 run id 并提交。

预期：之前生成的 frames 重新出现在 timeline 中，最终图可查看。

- [ ] **步骤 6：验证参数错误**

提交以下非法参数：

```text
prompt: a fox
steps: 10
decode_interval: 11
width: 512
height: 512
```

预期：页面错误区域显示参数错误，后端返回 422，不启动生成任务。

- [ ] **步骤 7：停止本地应用**

在运行 Uvicorn 的终端按 `Ctrl+C`。

预期：服务退出，未留下后台进程。

- [ ] **步骤 8：Commit 验证相关修正**

如果任务 13 过程中发现并修复了代码或文档问题，运行：

```bash
git add <fixed-files>
git commit -m "fix: complete diffusion visualization verification"
```

如果没有文件变更，不创建 commit。
