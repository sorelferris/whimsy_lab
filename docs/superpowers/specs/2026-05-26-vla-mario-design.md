# VLA 超级马里奥兄弟 — 设计规格

> 设计版本: 1.0
> 日期: 2026-05-26
> 状态: 待用户审查

## 1. 背景与目标

实现一个基于 VLM（视觉语言模型）的 VLA（Vision-Language-Action）智能体，能够玩 Super Mario Bros World 1-4。

**核心目标**：
- 性能极限：追求最高通关率和得分
- 推理扩展性：专为 RTX 3080（10GB VRAM）优化推理部署
- 可视化：支持 Chain-of-Thought 推理过程可视化

**成功指标**：
- World 1-4 综合通关率 > 80%
- 单帧推理延迟 < 100ms (p95)
- 支持 8 并行 batch 推理

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  SMB Game (gym-retro)                                   │
│    └─► Observation (frame + state)                    │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────┐     ┌──────────────────────────┐ │
│  │  VLM (MiniCPM-V │────►│  Action + Reasoning Trace│ │
│  │   2.8B + QLoRA) │     └──────────┬───────────────┘ │
│  └─────────────────┘                │                  │
│                          ┌──────────▼──────────┐       │
│                          │  Fast Policy Head   │       │
│                          │  (parallel fallback) │       │
│                          └──────────┬──────────┘       │
│                                     ▼                   │
│                          ┌─────────────────────┐        │
│                          │  CoT 可视化层       │        │
│                          │  (实时 + 调试模式) │        │
│                          └─────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

**核心思路**：
- VLM 负责高level推理（场景理解、策略规划）
- Fast Policy Head 处理亚帧级快速反应
- CoT 层记录推理过程供可视化

---

## 3. 组件规格

### 3.1 VLM 模块

**模型选择**：MiniCPM-V 2.8B（本地可跑，量化友好）

**输入格式**：
```
文本: "You are a SMB expert. Given the screen, predict the action.
       Available actions: ←, →, ↑, ↓, A, B, ←+A, →+A, ↑+A, ↑+B, ↓+A, idle
       Also explain your reasoning briefly."

图像: 原始画面 resize 到 384×384
```

**输出格式**：
```
Action: →+A
Reasoning: "前方3格有坑，需要跳跃通过。同时跳起可以顶碎上方砖块获取金币。"
```

**QLoRA 配置**：
- LoRA rank: 64, alpha: 128, dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- 训练精度: BF16 主干 + INT4 LoRA
- 训练数据: 200K SMB 帧-动作对 + 30K 人类通关轨迹

### 3.2 Fast Policy Head

**目的**：处理需要亚帧级响应的场景（躲避炮弹、跳出包围等），延迟 <5ms。

**架构**：
```
输入: 原始画面 → 小型 CNN (3层) → 128-d embedding
      控制器状态 → 线性层 → 32-d embedding
      concat → 160-d → 3层 MLP → 12-d action logits

动作空间（12维）：
← → ↑ ↓ A B A+B ↑+A ↑+B ↓+A ↓+B （←+→禁止同按）
```

**训练**：从 VLM 的 action distribution 中蒸馏，KL散度 loss
**推理**：与 VLM 并行执行，最终 action = VLM_action × 0.7 + FastPolicy_action × 0.3

### 3.3 训练流程

**数据采集**：
- 人类玩家游玩 SMB World 1-4
- 记录 (frame, controller_state, action, reasoning)
- 约 30K 人类轨迹 + 200K 自动增强帧

**Stage 1 - CoT 微调（4-6小时，单卡 RTX 3080）**：
```
基础 VLM + QLoRA (rank=64)
训练目标: 最大化 P(action | frame, reasoning)
Loss = CE(action) + λ * CE(reasoning), λ=0.3
Learning rate: 2e-4, warmup 100步, cosine decay
Batch: 4, Gradient accumulation: 16 → effective 64
```

**Stage 2 - Reward Shaping（2-3小时）**：
```
PPO 风格优化
Reward: +10 通关, +1 存活秒, +5 收集金币, -100 死亡
同时用 VLM 自身的 reasoning 作为 auxiliary loss
```

### 3.4 CoT 可视化层

**实时覆盖模式**（默认）：
```
┌──────────────────────────────────────┐
│  [游戏画面]                          │
│                                       │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░ VLM推理:                          ░
│  ░ "前方3格有砖块，               ░
│  ░  跳起可顶出金币。              ░
│  ░  同时能越过前方敌人"          ░
│  ░  → Action: →+A (跳跃前进)     ░
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└──────────────────────────────────────┘
```
- 半透明黑底，白色字体，3行精简输出
- 更新频率: 每5帧更新一次（减少闪烁）

**调试模式**（按 D 键切入）：
```
┌─────────────┬──────────────────────────┐
│ 游戏画面    │ VLM Chain-of-Thought:    │
│             │                          │
│             │ [1] 场景分析              │
│             │   检测到: 坑(2格宽)       │
│             │         砖块(上方)       │
│             │         敌人(右方3格)    │
│             │                          │
│             │ [2] 策略生成              │
│             │   选项A: 跳过坑           │
│             │   选项B: 踩砖跳          │
│             │   ✓ 选项B (最高分)        │
│             │                          │
│             │ [3] 动作输出              │
│             │   → + A (持续0.5s)        │
│             │   confidence: 0.94        │
├─────────────┴──────────────────────────┤
│ [暂停] [步进] [单帧前进] [重置]        │
└─────────────────────────────────────────┘
```

### 3.5 推理优化（RTX 3080 专项）

**目标**：在 10GB VRAM 约束下实现最低延迟和最大 batch 吞吐。

**优化层次**：

| 层次 | 优化项 | 预期效果 |
|------|--------|---------|
| 量化 | FP16 → INT8 GPTQ → INT4 | 模型 3B→1.5GB，延迟 -40% |
| KV Cache | PagedAttention，帧间复用 | 峰值显存 -60% |
| Batch | 预填充 8并行，continuous batching | 吞吐 30+ FPS |
| CUDA | FlashAttention-2，4-bit 核函数 | 显存对齐减少碎片 |

**预期性能**：
- 模型加载: ~2GB (INT4)
- 单帧推理延迟: 30-50ms (VLM) + 2ms (Policy)
- 游戏帧率: 30+ FPS (可玩)
- Batch 吞吐: 8 并行请求，延迟 <100ms (p95)

---

## 4. 技术栈

| 组件 | 技术选型 |
|------|---------|
| 游戏环境 | gym-retro (SMB) |
| VLM 基座 | MiniCPM-V 2.8B |
| 微调框架 | transformers + peft (QLoRA) |
| 推理优化 | vLLM (PagedAttention, continuous batching) |
| 可视化 | Pygame overlay / OpenCV |
| 训练 | 单卡 RTX 3080, ~8-12 小时 |

---

## 5. 项目结构

```
vla-mario/
├── train/                  # 训练代码
│   ├── data_collection.py  # 数据采集
│   ├── sft_trainer.py      # Stage 1 微调
│   └── rl_trainer.py       # Stage 2 PPO
├── inference/              # 推理部署
│   ├── vlm_engine.py       # VLM 推理引擎
│   ├── policy_head.py      # Fast Policy
│   └── coT_visualizer.py   # CoT 可视化层
├── configs/                # 配置文件
│   ├── model.yaml          # 模型配置
│   ├── train.yaml          # 训练配置
│   └── inference.yaml      # 推理配置
├── eval/                   # 评估
│   └── benchmark.py        # World 1-4 评测
└── main.py                 # 入口
```

---

## 6. 实现计划（待 writing-plans 阶段展开）

1. 环境搭建：gym-retro + 游戏 ROM + 基线随机策略验证
2. 数据采集：人类玩家轨迹 + 自动增强
3. Stage 1 微调：VLM QLoRA + CoT
4. Fast Policy Head：蒸馏训练 + 并行推理
5. 推理优化：INT8/INT4 量化 + vLLM 集成
6. CoT 可视化：实时覆盖层 + 调试模式
7. Stage 2 RL：Reward Shaping 优化
8. 评估：World 1-4 全通关率测试

---

*设计已 commit，等待用户审查后进入实现计划阶段。*