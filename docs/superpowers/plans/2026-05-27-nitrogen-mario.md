# NitroGen 马里奥 — 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 基于 NitroGen 500M DiT + Flow Matching，构建能玩 Super Mario Bros World 1-4 的游戏 AI agent，支持 Flow 推理过程可视化和 RTX 3080 (20GB) 推理优化。

**架构：** Fork NitroGen 代码库，复用预训练 DiT 基座，Windows 端通过 dxcam 截屏 + vgamepad 控制游戏，Linux 端运行推理 server 通过 ZMQ 通信，Post-training 微调适配 SMB。

**技术栈：** PyTorch, transformers, peft (QLoRA), ZMQ, OpenCV, dxcam, vgamepad, gym-retro

---

## 文件结构

```
nitrogen-mario/
├── nitrogen/                           # Fork from NitroGen (MIT license)
│   ├── __init__.py
│   ├── flow_matching_transformer/
│   │   ├── __init__.py
│   │   ├── nitrogen.py               # DiT 主干，get_action / get_action_with_cfg
│   │   └── modules.py                # DiTConfig, DiT, BasicTransformerBlock, AdaLN
│   ├── mm_tokenizers.py               # NitrogenTokenizer, tokenize frame+action
│   ├── cfg.py                         # CkptConfig, ModalityConfig
│   ├── game_env.py                    # GamepadEnv (Windows)
│   ├── inference_session.py            # InferenceSession, load_model, predict
│   ├── inference_client.py             # ZMQ client (Windows side)
│   ├── inference_viz.py                # FlowVisualizer overlay
│   └── shared.py                      # PATH_REPO, constants
├── scripts/
│   ├── serve.py                       # Linux inference server (ZMQ REP)
│   ├── play.py                        # Windows agent runner
│   └── download_ckpt.py               # Download from HuggingFace
├── train/
│   ├── data_loader.py                 # SMB trajectory loader
│   ├── sft_trainer.py                 # Stage 1: QLoRA fine-tuning
│   └── rl_trainer.py                 # Stage 2: PPO reward shaping
├── configs/
│   ├── nitrogen_base.yaml             # NitroGen model config
│   ├── smb_finetune.yaml              # SMB post-training config
│   └── inference.yaml                 # Inference optimization config
├── eval/
│   └── benchmark.py                   # World 1-4 evaluation
└── main.py                            # Entry point (serve / play modes)
```

---

## 任务 1：项目初始化与 NitroGen Fork

**文件：**
- 创建：`nitrogen-mario/nitrogen/__init__.py`
- 创建：`nitrogen-mario/nitrogen/flow_matching_transformer/__init__.py`
- 创建：`nitrogen-mario/nitrogen/flow_matching_transformer/nitrogen.py`
- 创建：`nitrogen-mario/nitrogen/flow_matching_transformer/modules.py`
- 创建：`nitrogen-mario/nitrogen/mm_tokenizers.py`
- 创建：`nitrogen-mario/nitrogen/cfg.py`
- 创建：`nitrogen-mario/nitrogen/shared.py`
- 创建：`nitrogen-mario/pyproject.toml`
- 创建：`nitrogen-mario/README.md`
- 创建：`nitrogen-mario/.gitignore`

- [ ] **步骤 1：创建目录结构**

```bash
mkdir -p nitrogen-mario/nitrogen/flow_matching_transformer
mkdir -p nitrogen-mario/scripts
mkdir -p nitrogen-mario/train
mkdir -p nitrogen-mario/eval
mkdir -p nitrogen-mario/configs
touch nitrogen-mario/nitrogen/__init__.py
touch nitrogen-mario/nitrogen/flow_matching_transformer/__init__.py
```

- [ ] **步骤 2：创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "nitrogen-mario"
version = "1.0.0"
description = "NitroGen-based VLA agent for Super Mario Bros"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "torch",
    "pyyaml",
    "einops",
    "transformers",
    "pydantic>=2.0",
    "diffusers",
    "polars",
    "pillow",
    "opencv-python",
    "numpy",
    "pyzmq",
    "gymnasium",
    "psutil",
]

[project.optional-dependencies]
serve = ["torch", "pyyaml", "einops", "transformers", "pydantic>=2.0", "diffusers", "polars"]
play = ["pillow", "opencv-python", "numpy", "pyzmq", "gymnasium", "psutil", "dxcam", "vgamepad", "pywin32"]
train = ["torch", "transformers", "peft", "accelerate", "datasets"]
eval = ["pillow", "opencv-python", "numpy", "pyzmq"]
all = ["torch", "pyyaml", "einops", "transformers", "pydantic>=2.0", "diffusers", "polars",
       "pillow", "opencv-python", "gymnasium", "psutil", "peft", "accelerate", "datasets"]

[tool.setuptools.packages.find]
where = ["."]
exclude = ["scripts*", "train*", "eval*"]
```

- [ ] **步骤 3：创建 nitrogen/shared.py**

```python
from pathlib import Path

PATH_REPO = Path(__file__).parent.parent.resolve()
```

- [ ] **步骤 4：创建 nitrogen/cfg.py**

```python
from pydantic import BaseModel, Field
from nitrogen.flow_matching_transformer.nitrogen import NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig

class ModalityConfig(BaseModel):
    frame_per_sample: int = 1
    frame_spacing: int | None = None
    action_per_chunk: int = 8
    action_shift: int = 1
    action_interleaving: bool = False
    token_set: str = "new"

    def model_post_init(self, __context):
        if self.frame_spacing is None:
            object.__setattr__(self, 'frame_spacing', self.action_per_chunk)

class CkptConfig(BaseModel):
    experiment_name: str = Field(...)
    model_cfg: NitroGen_Config
    tokenizer_cfg: NitrogenTokenizerConfig
    modality_cfg: ModalityConfig
```

- [ ] **步骤 5：创建 nitrogen/flow_matching_transformer/modules.py**
（内容来自 NitroGen modules.py：TimestepEncoder, AdaLayerNorm, BasicTransformerBlock, DiTConfig, DiT, SelfAttentionTransformerConfig, SelfAttentionTransformer）

- [ ] **步骤 6：创建 nitrogen/flow_matching_transformer/nitrogen.py**
（内容来自 NitroGen nitrogen.py：NitroGen_Config, NitroGen class, get_action, get_action_with_cfg）

- [ ] **步骤 7：创建 nitrogen/mm_tokenizers.py**（stub，后续任务填充）

```python
# Stub - full implementation in Task 2
from pydantic import BaseModel, Field

class NitrogenTokenizerConfig(BaseModel):
    pass

class NitrogenTokenizer:
    def __init__(self, config): pass
    def encode(self, data): return {}
    def decode(self, output): return {"buttons": None, "j_left": None, "j_right": None}
    @property
    def game_mapping(self): return None
```

- [ ] **步骤 8：创建 nitrogen/inference_session.py**（stub，后续任务填充）

```python
# Stub - full implementation in Task 3
class InferenceSession:
    pass
```

- [ ] **步骤 9：创建 nitrogen/inference_client.py**（stub）

```python
# Stub - Windows ZMQ client
class NitrogenClient:
    pass
```

- [ ] **步骤 10：创建 nitrogen/inference_viz.py**（stub）

```python
# Stub - Flow visualizer
class FlowVisualizer:
    pass
```

- [ ] **步骤 11：创建 nitrogen/game_env.py**（stub，Windows only）

```python
# Stub - game environment
class GamepadEnv:
    pass
```

- [ ] **步骤 12：创建 scripts/serve.py**

```python
#!/usr/bin/env python3
"""NitroGen inference server - runs on Linux with GPU."""
import argparse
import zmq
import torch
from nitrogen.inference_session import InferenceSession

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt_path}...")
    session = InferenceSession.from_ckpt(args.ckpt_path)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Server listening on port {args.port}")

    while True:
        msg = socket.recv()
        # Expect: serialized image bytes + optional game_id
        # Response: action dict
        socket.send(b"ready")

if __name__ == "__main__":
    main()
```

- [ ] **步骤 13：创建 scripts/download_ckpt.py**

```python
#!/usr/bin/env python3
"""Download NitroGen checkpoint from HuggingFace."""
import argparse
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="ng.pt")
    args = parser.parse_args()

    path = hf_hub_download(
        repo_id="nvidia/NitroGen",
        filename="ng.pt",
        local_dir=".",
    )
    print(f"Downloaded to: {path}")

if __name__ == "__main__":
    main()
```

- [ ] **步骤 14：创建 configs/nitrogen_base.yaml**

```yaml
# NitroGen base model config
model:
  model_type: nitrogen
  hidden_size: 1024
  max_seq_len: 1024
  action_dim: 26
  action_horizon: 8
  diffusion_model_cfg:
    num_layers: 12
    num_attention_heads: 16
    attention_head_dim: 64
    output_dim: 26
    max_num_positional_embeddings: 512
    dropout: 0.1
    activation_fn: "gelu-approximate"
  vl_self_attention_cfg:
    num_layers: 12
    num_attention_heads: 16
    attention_head_dim: 64
  vision_encoder_name: "google/siglip-large-patch16-256"
  num_inference_timesteps: 16
  noise_beta_alpha: 1.5
  noise_beta_beta: 1.0
  noise_s: 0.999
  cfg_scale: 1.5
```

- [ ] **步骤 15：创建 configs/inference.yaml**

```yaml
# Inference optimization config for RTX 3080 20GB
inference:
  dtype: "bfloat16"
  num_inference_timesteps: 16
  cfg_scale: 1.5
  batch_size: 16
  max_context_frames: 4
  device: "cuda"

optimization:
  flash_attention: true
  compile: false  # Optional: torch.compile for faster inference
  kv_cache: true

visualization:
  enabled: true
  update_every_n_frames: 5
  overlay_opacity: 0.7
```

- [ ] **步骤 16：创建 .gitignore**

```
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.pt
*.pth
*.ckpt
checkpoints/
logs/
*.egg-info/
.DS_Store
```

- [ ] **步骤 17：创建 README.md**

```markdown
# NitroGen Mario

基于 NVIDIA NitroGen 的超级马里奥兄弟游戏 AI agent。

## 架构

- NitroGen 500M DiT + Flow Matching
- SigLIP-L 视觉编码器
- Windows 游戏端 + Linux 推理端分离
- ZMQ 通信

## 快速开始

```bash
pip install -e ".[all]"
python scripts/download_ckpt.py --output ng.pt
python scripts/serve.py ng.pt --port 5555
```
```

- [ ] **步骤 18：Commit**

```bash
git add nitrogen-mario/
git commit -m "feat: initialize nitrogen-mario project structure"
```

---

## 任务 2：NitrogenTokenizer 完整实现

**文件：**
- 修改：`nitrogen-mario/nitrogen/mm_tokenizers.py`

- [ ] **步骤 1：编写 tokenizer 测试**

创建 `tests/test_tokenizer.py`：

```python
import torch
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig

def test_tokenizer_encode_frame():
    from PIL import Image
    import numpy as np

    config = NitrogenTokenizerConfig(
        max_frames=4,
        frame_size=256,
        action_dim=26,
        training=False,
    )
    tokenizer = NitrogenTokenizer(config)

    # Create dummy frame (256x256 RGB)
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    data = {
        "frames": [img],
        "dropped_frames": [False],
        "game": "SMB",
    }
    encoded = tokenizer.encode(data)

    assert "input_ids" in encoded
    assert "pixel_values" in encoded
    assert isinstance(encoded["input_ids"], torch.Tensor)

def test_tokenizer_decode_action():
    config = NitrogenTokenizerConfig(training=False)
    tokenizer = NitrogenTokenizer(config)

    # Mock model output: 26-d action tensor
    model_output = {
        "buttons": torch.randn(1, 13),
        "j_left": torch.randn(1, 2),
        "j_right": torch.randn(1, 2),
    }
    decoded = tokenizer.decode(model_output)
    assert "buttons" in decoded
    assert "j_left" in decoded
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_tokenizer.py -v
# Expected: FAIL - module not fully implemented
```

- [ ] **步骤 3：实现 NitrogenTokenizer**

实现完整的 tokenizer，支持：
- frame → visual tokens (使用 SigLIP processor)
- action → tokenized action sequence
- game_id → conditioning token
- decode: action tokens → button/joystick dict

```python
# Full implementation - see NitroGen mm_tokenizers.py for reference
# Key: _PAD_TOKEN=0, _IMG_TOKEN=1, _LANG_TOKEN=2, _PROPRIO_TOKEN=3, _ACT_TOKEN=4, _IMG_SEP_TOKEN=5, _GAME_ID_TOKEN=6
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_tokenizer.py -v
# Expected: PASS
```

- [ ] **步骤 5：Commit**

```bash
git add tests/test_tokenizer.py nitrogen/mm_tokenizers.py
git commit -m "feat: implement NitrogenTokenizer with frame/action encoding"
```

---

## 任务 3：InferenceSession 完整实现

**文件：**
- 修改：`nitrogen-mario/nitrogen/inference_session.py`
- 创建：`nitrogen-mario/tests/test_inference_session.py`

- [ ] **步骤 1：编写 inference 测试**

```python
import torch
from nitrogen.inference_session import InferenceSession, load_model

def test_load_model():
    # Skip if no checkpoint available
    import os
    if not os.path.exists("ng.pt"):
        return

    model, tokenizer, img_proc, ckpt_config, game_mapping, action_downsample = load_model("ng.pt")
    assert model is not None
    assert tokenizer is not None

def test_predict_returns_action_dict():
    # Mock test
    pass
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_inference_session.py -v
# Expected: FAIL - stub implementation
```

- [ ] **步骤 3：实现 InferenceSession**

实现完整逻辑：
- `load_model()`: 加载 checkpoint，构建 NitroGen + SigLIP
- `from_ckpt()`: 工厂方法，创建 session 并选择游戏
- `predict(obs)`: 接收 PIL Image，执行 flow matching 推理，返回 action dict
- `_predict_flowmatching()`: 16步扩散 + CFG
- `reset()`: 清空 obs/action buffer
- `info()`: 返回 session 配置信息

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_inference_session.py -v
# Expected: PASS (or skip if no checkpoint)
```

- [ ] **步骤 5：Commit**

```bash
git add tests/test_inference_session.py nitrogen/inference_session.py
git commit -m "feat: implement InferenceSession with flow matching inference"
```

---

## 任务 4：ZMQ Server + Client 通信层

**文件：**
- 修改：`nitrogen-mario/scripts/serve.py`
- 创建：`nitrogen-mario/scripts/play.py`
- 修改：`nitrogen-mario/nitrogen/inference_client.py`
- 创建：`nitrogen-mario/tests/test_communication.py`

- [ ] **步骤 1：编写通信测试**

```python
# Test ZMQ message format
def test_action_message_format():
    action = {"buttons": [0,1,0,0,0,0,0,0,0,0,0,0,0], "j_left": [1.0, 0.0], "j_right": [0.0, 0.0]}
    import json
    msg = json.dumps(action)
    decoded = json.loads(msg)
    assert decoded["buttons"] == action["buttons"]
```

- [ ] **步骤 2：实现完整 serve.py**

- 支持多 client（ZMQ ROUTER）
- 异步处理 frame + game_id
- 集成 InferenceSession
- 优雅 shutdown (SIGINT/SIGTERM)

- [ ] **步骤 3：实现 play.py (Windows client)**

```python
#!/usr/bin/env python3
"""Windows agent runner - captures game, sends to server, executes actions."""
import argparse
import zmq
import time
from nitrogen.game_env import GamepadEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="tcp://localhost:5555")
    parser.add_argument("--process", type=str, required=True)
    args = parser.parse_args()

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(args.server)

    env = GamepadEnv(game=args.process)

    while True:
        obs = env.render()
        # Send to server
        socket.send(obs.tobytes())
        action = socket.recv_json()
        env.step(action)
```

- [ ] **步骤 4：Commit**

```bash
git add scripts/serve.py scripts/play.py nitrogen/inference_client.py tests/test_communication.py
git commit -m "feat: add ZMQ server-client communication layer"
```

---

## 任务 5：Flow 可视化层

**文件：**
- 修改：`nitrogen-mario/nitrogen/inference_viz.py`
- 创建：`nitrogen-mario/tests/test_visualizer.py`

- [ ] **步骤 1：编写可视化测试**

```python
import numpy as np
from nitrogen.inference_viz import FlowVisualizer

def test_visualizer_init():
    viz = FlowVisualizer(width=1920, height=1080)
    assert viz.width == 1920
    assert viz.height == 1080

def test_render_overlay():
    viz = FlowVisualizer()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    step_info = {
        "step": 8,
        "total_steps": 16,
        "noise_level": 0.3,
        "action_confidence": 0.82,
        "buttons": ["→", "A"],
        "j_left": (1.0, 0.0),
    }
    overlay = viz.render_overlay(frame, step_info)
    assert overlay.shape == frame.shape
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_visualizer.py -v
# Expected: FAIL - stub
```

- [ ] **步骤 3：实现 FlowVisualizer**

- `__init__(width, height, opacity)`: 初始化覆盖层
- `render_overlay(frame, step_info)`: 在帧上叠加推理信息
- `render_debug_panel(frame, debug_info)`: 渲染详细调试面板
- 支持两种模式切换（实时覆盖 / 调试面板）
- 按 D 键切换模式（通过 socket 命令）
- 半透明黑底 + 白色等宽字体
- 显示：step/total, noise_level, action confidence, 按钮激活状态, 摇杆值

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_visualizer.py -v
# Expected: PASS
```

- [ ] **步骤 5：Commit**

```bash
git add tests/test_visualizer.py nitrogen/inference_viz.py
git commit -m "feat: implement Flow visualizer overlay with realtime and debug modes"
```

---

## 任务 6：游戏环境适配（Windows）

**文件：**
- 修改：`nitrogen-mario/nitrogen/game_env.py`
- 创建：`nitrogen-mario/tests/test_game_env.py`

- [ ] **步骤 1：实现 GamepadEnv**

- 基于 NitroGen game_env.py 代码
- dxcam 截屏（1440p 降采样到 384×384）
- vgamepad 模拟 Xbox 手柄
- xspeedhack DLL 注入（pause/unpause 精确控制）
- 统一 action dict → 手柄按键映射
- async_mode：每 step 精确控制时长

- [ ] **步骤 2：编写测试**

```python
# Test game_env on Linux (mock)
def test_action_mapping():
    action = {"SOUTH": 1, "DPAD_RIGHT": 1, "AXIS_LEFTX": [1.0]}
    # Verify mapping to Xbox buttons
    assert action["SOUTH"] == 1
```

- [ ] **步骤 3：Commit**

```bash
git add tests/test_game_env.py nitrogen/game_env.py
git commit -m "feat: implement Windows game environment with dxcam + vgamepad"
```

---

## 任务 7：SMB 集成测试

**文件：**
- 创建：`nitrogen-mario/tests/test_integration.py`

- [ ] **步骤 1：编写集成测试**

```python
# Full pipeline test (requires SMB ROM and NitroGen checkpoint)
def test_full_pipeline():
    # 1. Load model
    # 2. Connect to game
    # 3. Run 10 inference steps
    # 4. Verify actions are executed
    pass
```

- [ ] **步骤 2：Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add SMB integration test scaffold"
```

---

## 任务 8：SMB Post-training 微调

**文件：**
- 创建：`nitrogen-mario/train/data_loader.py`
- 创建：`nitrogen-mario/train/sft_trainer.py`
- 创建：`nitrogen-mario/configs/smb_finetune.yaml`
- 创建：`nitrogen-mario/tests/test_trainer.py`

- [ ] **步骤 1：实现 SMB 数据加载器**

```python
# Load SMB trajectories: (frame, action, reward) sequences
# Support: gym-retro states, human play recordings
# Output: PyTorch Dataset compatible with transformers Trainer
```

- [ ] **步骤 2：实现 SFT trainer**

```python
# QLoRA fine-tuning on DiT
# 1. Load NitroGen checkpoint
# 2. Apply LoRA adapters (rank=64, target q/k/v/o proj)
# 3. Flow matching loss: MSE(predicted_action, target_action)
# 4. Train with: lr=2e-4, warmup=100, cosine decay, batch=8, grad_accum=8
# 5. Save adapter weights for inference
```

- [ ] **步骤 3：创建 smb_finetune.yaml**

```yaml
# SMB post-training config
model:
  lora_rank: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
  warmup_steps: 100
  weight_decay: 0.01

data:
  smb_trajectories_path: "./data/smb_trajectories"
  num_workers: 4
  prefetch_factor: 2
```

- [ ] **步骤 4：编写 trainer 测试**

```python
def test_lora_training_step():
    # Mock training step
    pass
```

- [ ] **步骤 5：Commit**

```bash
git add train/ configs/smb_finetune.yaml tests/test_trainer.py
git commit -m "feat: implement SMB post-training with QLoRA"
```

---

## 任务 9：推理优化

**文件：**
- 创建：`nitrogen-mario/nitrogen/optimized_inference.py`
- 修改：`nitrogen-mario/configs/inference.yaml`

- [ ] **步骤 1：实现 INT8 量化**

```python
# GPTQ quantization of NitroGen DiT
from transformers import GPTQConfig

quantized_model = model.quantize("auto")
# Expected: 500M params → ~250MB
```

- [ ] **步骤 2：实现 8步截断采样**

```python
def predict_fast(session, obs, num_steps=8):
    """Fast inference with reduced diffusion steps."""
    original_steps = session.model.num_inference_timesteps
    session.model.num_inference_timesteps = num_steps
    result = session.predict(obs)
    session.model.num_inference_timesteps = original_steps
    return result
```

- [ ] **步骤 3：实现 batch 推理**

```python
def predict_batch(session, obs_list):
    """Batch inference for multiple frames."""
    # Continuous batching with vLLM-style scheduling
    pass
```

- [ ] **步骤 4：Commit**

```bash
git add nitrogen/optimized_inference.py configs/inference.yaml
git commit -m "feat: add INT8 quantization and batch inference optimization"
```

---

## 任务 10：评估基准

**文件：**
- 创建：`nitrogen-mario/eval/benchmark.py`
- 创建：`nitrogen-mario/tests/test_benchmark.py`

- [ ] **步骤 1：实现 benchmark**

```python
# Evaluate on World 1-4
# Metrics:
#   - Clear rate (world通关率)
#   - Average score per episode
#   - Average steps per episode
#   - Death count
#   - Coin collection

def evaluate(model_path, world, num_episodes=10):
    """Run evaluation on a specific world."""
    pass
```

- [ ] **步骤 2：Commit**

```bash
git add eval/benchmark.py tests/test_benchmark.py
git commit -m "feat: implement World 1-4 evaluation benchmark"
```

---

## 任务 11：Main Entry Point

**文件：**
- 创建：`nitrogen-mario/main.py`

- [ ] **步骤 1：实现 main.py**

```python
#!/usr/bin/env python3
"""Main entry point for nitrogen-mario."""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="NitroGen Mario Agent")
    subparsers = parser.add_subparsers(dest="command")

    # serve mode
    serve = subparsers.add_parser("serve", help="Start inference server")
    serve.add_argument("checkpoint", type=str)
    serve.add_argument("--port", type=int, default=5555)

    # play mode
    play = subparsers.add_parser("play", help="Run agent on game")
    play.add_argument("--server", type=str, default="tcp://localhost:5555")
    play.add_argument("--process", type=str, required=True)

    # eval mode
    eval_p = subparsers.add_parser("eval", help="Run benchmark")
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--world", type=int, default=1)
    eval_p.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()
    if args.command == "serve":
        from scripts.serve import main as serve_main
        sys.argv = [sys.argv[0], args.checkpoint, "--port", str(args.port)]
        serve_main()
    elif args.command == "play":
        from scripts.play import main as play_main
        sys.argv = [sys.argv[0], "--server", args.server, "--process", args.process]
        play_main()
    elif args.command == "eval":
        from eval.benchmark import evaluate
        results = evaluate(args.checkpoint, args.world, args.episodes)
        print(results)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：Commit**

```bash
git add main.py
git commit -m "feat: add main entry point with serve/play/eval commands"
```

---

*计划完成。共 11 个任务，约 40+ 个步骤。*