# 项目遗漏项清单

本文档记录当前自动化测试之外仍需处理或人工确认的项目遗漏项。

当前自动化验证状态：`uv run pytest -v` 已通过，结果为 `41 passed`。

## 高优先级

### 1. 默认 `reload=True` 可能中断生成

位置：`main.py`

当前应用启动时使用 Uvicorn reload。生成过程中会写入 `runs/` 下的 metadata 和 PNG 文件，如果这些文件被 reload 监听到，服务可能重启，导致 GPU 生成任务和 SSE 流中断。

建议：

- 默认关闭 reload。
- 如需开发模式 reload，单独提供开发命令。
- 或显式排除 `runs/` 等生成目录。

### 2. SSE 事件顺序存在竞态风险

位置：`image_gen_viz/tasks.py`

当前 progress/frame 事件通过 `loop.call_soon_threadsafe(...)` 入队，`complete` 事件在 `asyncio.to_thread(...)` 返回后直接 `await queue.put(...)`。如果线程安全回调尚未全部执行，客户端可能先收到 `complete`，再收到最后的 frame/progress。

建议：

- 统一事件入队路径。
- 确保 terminal event 永远最后发送。
- 增加事件顺序回归测试。

### 3. metadata 写入不是原子操作

位置：`image_gen_viz/storage.py`

当前 `metadata.json` 直接重写。生成过程中如果前端 reload 或 API 读取刚好遇到半写入文件，可能出现 JSON 解析失败并返回 500。

建议：

- 写入临时文件后使用 `replace()` 原子替换。
- 或对 metadata 读写都使用同一把锁。
- 增加并发读写回归测试。

### 4. SSE 断线重连不够稳健

位置：`image_gen_viz/tasks.py`、`image_gen_viz/web.py`

如果浏览器在生成过程中刷新或断线，`events()` 可能在队列为空时移除 run queue。后续重新连接可能得到 404，即使生成任务仍在运行。

建议：

- active run 的 queue 保留到任务结束。
- 或支持从 metadata replay/reload 当前已生成帧。
- 增加断线后 reload/reconnect 行为测试。

### 5. 真实 GPU/浏览器 smoke 尚未执行

自动化测试已通过，但还没有实际完成以下人工验收：

- SD1.5 模型下载和加载。
- GPU 推理。
- timeline 实时展示。
- timeline 播放和 scrubber 拖动。
- run reload。
- 参数错误 UI。

手动步骤见：`docs/manual-gpu-validation.md`。

## 中优先级

### 6. CUDA 不可用时错误体验不完整

位置：`image_gen_viz/model.py`

当前 generator device 会在 CUDA 不可用时回退 CPU，但 pipeline 仍默认移动到 `cuda`，并使用 `float16`。在无 GPU 环境中，生成会失败。

建议：

- 在启动或生成前明确检测 CUDA。
- 给出清晰的用户错误提示。
- 或正式支持 CPU 模式，并调整 dtype/device 策略。

### 7. `model_id` 输入过于自由

位置：`image_gen_viz/validation.py`、`image_gen_viz/model.py`

用户可输入任意 `model_id`，会直接传给 `StableDiffusionPipeline.from_pretrained()`。这可能导致超大模型下载、磁盘占用过高、不兼容 pipeline 或本地路径加载。

建议：

- 默认限制为 SD1.5 兼容模型 allowlist。
- 或把自定义模型 ID 放入高级模式，并在 UI/README 中明确风险。

### 8. safety checker 被禁用但没有提示

位置：`image_gen_viz/model.py`

当前 `StableDiffusionPipeline.from_pretrained(...)` 使用 `safety_checker=None`。如果项目面向更广泛用户，需要明确说明生成内容未过滤。

建议：

- 启用 safety checker；或
- 在 README/UI 中明确提示本地生成内容未经过安全过滤。

### 9. 缺少取消生成能力

位置：`image_gen_viz/tasks.py`、前端 UI

当前后端只允许单个 active generation，但没有 cancel endpoint。如果生成卡住、下载过慢或用户想终止，只能重启服务。

建议：

- 增加取消生成 API。
- 前端增加 Cancel 按钮。
- 或至少在 README 中说明卡住时需要重启服务。

### 10. README 仍缺少用户级说明

位置：`README.md`

当前项目主要更新了 `CLAUDE.md` 和手动验收文档，但 README 仍缺少面向用户的安装和使用说明。

建议 README 包含：

- 项目简介。
- 依赖安装。
- 启动命令。
- 测试命令。
- GPU/CUDA 要求。
- 首次模型下载说明。
- 常见错误。
- safety checker 行为。
- 手动验收文档链接。

## 低优先级

### 11. UI 缺少专门可复制的 run id

位置：`image_gen_viz/static/app.js`、`image_gen_viz/static/index.html`

当前 run id 只混在状态文本中，例如 `Complete {runId}`。手动验收要求复制 run id，但 UI 不够直接。

建议：

- 增加 `Current Run ID` 展示区域。
- 可选增加复制按钮。

### 12. Web SSE 集成测试不够完整

位置：`tests/test_web.py`

当前测试覆盖 manager 事件和 SSE 格式化，但对 `/api/runs/{run_id}/events` 的真实 streaming 路由覆盖不足。

建议：

- 增加不会挂起的 `TestClient.stream()` 集成测试。
- 验证 started/progress/frame/complete 的事件顺序。
- 覆盖 terminal event 后 stream 结束行为。

### 13. 前端测试主要是静态字符串契约

位置：`tests/test_static_contract.py`

当前前端测试主要检查静态字符串存在，未真正执行 JS 行为。

建议：

- 增加浏览器级测试或 JS DOM 测试。
- 覆盖 request payload 构造、timeline playback、run reload、EventSource error path。

### 14. 最终提交前需整理工作树

当前工作树包含多个新增和修改文件，也存在上级目录 `../.gitignore` 修改。最终提交前需要确认哪些属于本项目，哪些是本地或无关改动。

建议：

- 使用 `git status --short` 检查所有变更。
- 只 stage 本项目相关文件。
- 不要误提交本地缓存、生成产物或无关上级目录改动。
