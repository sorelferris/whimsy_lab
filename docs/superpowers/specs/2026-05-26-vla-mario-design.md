# VLA 超级马里奥兄弟 — 设计规格 (NitroGen 版)

> 设计版本: 2.0
> 日期: 2026-05-27
> 状态: 待用户审查

## 1. 背景与目标

基于 NVIDIA [NitroGen](https://github.com/MineDojo/NitroGen) 基础模型，实现能玩 Super Mario Bros World 1-4 的游戏 AI agent。

**核心目标**：
- 性能极限：追求最高通关率和得分
- 推理扩展性：专为 RTX 3080（10GB VRAM）优化推理部署
- 可视化：Flow Matching 推理过程可视化

**成功指标**：
- World 1-4 综合通关率 > 70%（NitroGen 基线能力 + post-training 适配）
- 单帧推理延迟 < 200ms (p95)（含 16 步扩散采样）
- 模型占用 < 8GB VRAM（INT8 量化后）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  Linux (RTX 3080)                      Windows (游戏机)         │
│  ┌────────────────────────┐          ┌──────────────┐     │
│  │  NitroGen DiT (500M)   │──ZMQ────►│ GamepadEmulator│    │
│  │  + SigLIP Vision Encoder│          │ (手柄模拟)    │    │
│  │  + Flow Matching        │          └───────┬──────┘    │
│  │  + CFG (cfg=1.5)       │◄─────────────── screenshot   │
│  └────────────────────────┘          dxcam 截屏            │
│       │                                              │      │
│       ▼                                              ▼      │
│  ┌─────────────────┐               ┌──────────────────┐  │
│  │ Flow 可视化层   │               │ xspeedhack 注入   │  │
│  │ (推理步数/隐变量│               │ (精确控制游戏节奏)│  │
│  │  实时展示)       │               └──────────────────┘  │
│  └─────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

**核心思路**：
- 复用 NitroGen 预训练 500M DiT 基座（已从互联网视频学习通用游戏行为）
- Windows 端运行游戏，通过 ZMQ 与 Linux 推理端通信
- Flow Matching 推理过程可视化（diffusion step 隐变量可视化）
- Post-training 微调适配 SMB

---

## 3. 组件规格

### 3.1 模型架构（直接复用 NitroGen）

**模型选择**：NitroGen 预训练模型 + SMB post-training 微调

**核心配置**：
```python
model_cfg = NitroGen_Config(
    hidden_size=1024,
    diffusion_model_cfg: DiTConfig(
        num_layers=12,
        num_attention_heads=16,
        attention_head_dim=64,
        output_dim=26,        # buttons(13) + joystick(13)
        max_num_positional_embeddings=512,
    ),
    vl_self_attention_cfg: SelfAttentionTransformerConfig(...),
    vision_encoder_name="google/siglip-large-patch16-256",
    num_inference_timesteps=16,  # 扩散步数
    noise_beta_alpha=1.5,
    noise_beta_beta=1.0,
    noise_s=0.999,
)
```

**动作空间（26维 continuous）**：
```
buttons: BACK, GUIDE, LEFT_SHOULDER, RIGHT_SHOULDER,
         WEST(A), SOUTH(B), EAST(X), NORTH(Y),
         START, DPAD_UP/DOWN/LEFT/RIGHT,
         LEFT_TRIGGER, RIGHT_TRIGGER

joystick: AXIS_LEFTX/Y, AXIS_RIGHTX/Y
```

**视觉编码器**：SigLIP-L（google/siglip-large-patch16-256）

### 3.2 推理流程

**Flow Matching 推理**（16 步）：
```
1. 编码当前帧 → SigLIP visual tokens
2. 编码 game conditioning token（SMB 专属）
3. 双路 CFG:
   - 有条件分支: frame + game_id → 预测 action
   - 无条件分支: frame (game=None) → 预测 null action
   - 合并: output = cfg * cond - (cfg-1) * uncond, cfg_scale=1.5
4. 16 步迭代去噪 → continuous action (26d)
5. 离散化手柄按钮 + 连续摇杆值
```

**每帧推理延迟拆解**：
```
SigLIP 编码:     ~30ms
DiT 16步扩散:    ~120ms (BF16, RTX 3080)
CFG 双路:        ×2 倍计算
---
总计:            ~150ms (可优化到 <200ms)
```

### 3.3 Post-training 微调（适配 SMB）

**Stage 1 - SMB 数据微调**（核心）：
```
数据: SMB World 1-4 人类通关轨迹
      + NitroGen 原有互联网游戏数据中涉及平台跳跃的片段
训练: QLoRA on DiT (rank=64, target q/k/v/o proj)
目标: 让模型适应 SMB 的特定动作模式（跳跃时长、敌人躲避等）
Loss: Flow Matching MSE + action head CE
Duration: 4-6h 单卡 RTX 3080
```

**Stage 2 - Reward Shaping**（可选）：
```
目标: 最大化金币收集 + 快速通关
Reward: +1 每秒存活, +5 每金币, +50 通关, -100 死亡
方法: 离线 RL（保守的 PPO），在微调模型基础上继续优化
```

### 3.4 Flow 可视化层

**实时覆盖模式**：
```
┌──────────────────────────────────────┐
│  [游戏画面]                          │
│                                       │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░ Flow Matching:                  ░
│  ░  Step 8/16 [████████░░░░] 50%  ░
│  ░  Noise σ=0.3 → Action cos=0.82 ░
│  ░  [→] [A] (RB+LT) confidence=88% ░
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└──────────────────────────────────────┘
```

- 显示当前 diffusion 步数 / 总步数
- 可选：展示隐变量 t-SNE 可视化（每步的隐空间轨迹）
- Action 按钮高亮（哪些按键被激活）
- 置信度（CFG 后的 action confidence）

**调试模式**（按 D 键切入）：
```
┌─────────────┬────────────────────────────────┐
│ 游戏画面    │ Flow Matching Debug:           │
│             │                                │
│             │ [1] Visual Encoding            │
│             │   SigLIP: 576 tokens           │
│             │   game_id: SMB (embedding: 0.3)│
│             │                                │
│             │ [2] CFG Branches               │
│             │   cond: [0.2, 0.8, 0.1, ...]   │
│             │   null:  [0.1, 0.4, 0.1, ...]  │
│             │   cfg=1.5 → blended            │
│             │                                │
│             │ [3] Diffusion Steps            │
│             │   t=0.8: noise=high, act=init  │
│             │   t=0.5: action emerging       │
│             │   t=0.2: action refining       │
│             │   t=0.0: final action          │
│             │                                │
│             │ [4] Action Output              │
│             │   buttons: [→, A, LB]          │
│             │   j_left: (1.0, 0.0)           │
│             │   confidence: 0.91             │
├─────────────┴────────────────────────────────┤
│ [暂停] [步进] [单帧前进] [重置] [CFG调参]   │
└──────────────────────────────────────────────┘
```

### 3.5 推理优化（RTX 3080 专项）

| 层次 | 优化项 | 预期效果 |
|------|--------|---------|
| 量化 | BF16 → INT8 GPTQ | 显存 500M→250M，延迟 -30% |
| 采样 | 16步 → 8步（截断） | 延迟 -50%，质量损失 <5% |
| Batch | Continuous batching | 吞吐 ×2-3 |
| CFG | cfg=1.5 → cfg=1.2 | 延迟 -15%，质量损失 <2% |
| Flash | FlashAttention-2 | 显存 -20%，延迟 -15% |

**预期性能**：
- 模型加载: ~500MB (INT8) - 1GB (FP16)
- 单帧推理延迟: 80-120ms (INT8 + 8步采样)
- 游戏帧率: 10+ FPS（原生）；可优化到 15-20 FPS

**RTX 3080 约束下的最优配置**：
```
INT8 量化 + 8步采样 + CFG=1.2 + FlashAttention
= ~100ms 延迟 + ~2GB VRAM
```

---

## 4. 技术栈

| 组件 | 技术选型 |
|------|---------|
| 基础模型 | NitroGen 500M DiT (GitHub/MineDojo) |
| 视觉编码器 | SigLIP-L (HuggingFace) |
| 游戏环境 | Windows + gym-retro 替代方案：dxcam 截屏 + vgamepad |
| 通信 | ZMQ (推理 server ←→ 游戏 client) |
| 微调 | transformers + peft (QLoRA) |
| 可视化 | OpenCV overlay |
| 训练 | 单卡 RTX 3080 |

---

## 5. 项目结构

```
nitrogen-mario/
├── nitrogen/                   # Forked from NitroGen
│   ├── flow_matching_transformer/
│   │   ├── nitrogen.py         # Core DiT model
│   │   └── modules.py          # DiT / Transformer blocks
│   ├── game_env.py             # Windows game interface
│   ├── inference_session.py    # Inference orchestration
│   ├── inference_client.py     # ZMQ client (Windows side)
│   ├── inference_viz.py        # Flow visualization
│   ├── mm_tokenizers.py        # Multi-modal tokenizer
│   ├── cfg.py                  # Config classes
│   └── shared.py               # Shared utilities
├── scripts/
│   ├── serve.py                # Linux: model inference server
│   ├── play.py                 # Windows: game agent runner
│   └── download_ckpt.py        # Download NitroGen checkpoint
├── train/
│   ├── sft_trainer.py          # Stage 1: SMB post-training
│   └── rl_trainer.py           # Stage 2: Reward shaping
├── eval/
│   └── benchmark.py            # World 1-4 evaluation
├── configs/
│   ├── nitrogen.yaml           # NitroGen model config
│   ├── smb_finetune.yaml       # SMB post-training config
│   └── inference.yaml          # Inference optimization config
└── main.py                     # Entry point
```

---

## 6. 实现计划（待 writing-plans 阶段展开）

1. **Fork & 环境搭建**：克隆 NitroGen，配置 Windows + Linux 双端环境
2. **基线验证**：下载 NitroGen 预训练模型，在 SMB 上跑通基线（无需微调）
3. **游戏环境适配**：用 dxcam 截取 SMB 画面，vmgamepad 模拟手柄输入
4. **Flow 可视化**：实现扩散步数/隐变量/Action 可视化覆盖层
5. **Post-training 微调**：SMB 数据 QLoRA 微调（4-6h）
6. **推理优化**：INT8 量化 + 8步采样 + CFG 调参
7. **Reward Shaping**：PPO 离线强化学习优化（可选）
8. **评估**：World 1-4 通关率测试

---

## 7. 与原版设计的关键差异

| | 原版（VLM 自回归）| 本版（NitroGen）|
|---|---|---|
| 架构 | MiniCPM-V + 自回归 | DiT + Flow Matching |
| 视觉 | MiniCPM-V 内置 | SigLIP-L（独立编码器）|
| CoT/可视化 | VLM reasoning text | Diffusion step 隐变量 |
| 游戏环境 | gym-retro（跨平台）| Windows + dxcam（依赖平台）|
| 泛化 | 微调后专精 SMB | 预训练泛化多游戏 + post-training |
| 训练数据 | 自己采集 | 互联网视频预训练 + SMB 微调 |
| 推理延迟 | 30-50ms (VLM) | 80-120ms (DiT 16步) |
| 模型大小 | 3B | 500M（更易部署）|

*NitroGen 的局限：只看当前帧（无 temporal planning），是 fast-reacting system-1 模型。SMB 的关卡跳转、隐藏砖块等需要长期规划的场景仍需额外处理。*

---

*设计已更新为 v2.0（基于 NitroGen），等待用户审查后进入实现计划阶段。*