# 手动 GPU 验收指南

完成自动化测试后，还需要在本机 GPU 和浏览器中执行以下手动验收。

## 1. 启动本地应用

在项目根目录运行：

```bash
uv run python main.py
```

确认终端输出包含类似内容：

```text
Uvicorn running on http://127.0.0.1:8000
```

## 2. 打开页面

在浏览器打开：

```text
http://127.0.0.1:8000
```

确认页面显示生成参数面板、预览区和 timeline 区域。

## 3. 提交一次真实 GPU 生成

使用以下参数：

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

确认：

- 页面显示 running 状态。
- 进度条持续推进。
- timeline 出现 step 2、4、6、8、10 的帧。
- 生成完成后进度为 100%。

## 4. 验证 timeline 播放和拖动

操作：

1. 点击 `Play`。
2. 拖动 scrubber。

确认：

- 预览图随时间轴更新。
- 当前缩略图 active 状态跟随变化。

## 5. 验证 run reload

操作：

1. 复制当前 run id。
2. 刷新页面。
3. 在 `Reload Run ID` 输入框填入该 run id。
4. 点击 `Load Run`。

确认：

- 之前生成的 frames 重新出现在 timeline 中。
- 最终图可以查看。

## 6. 验证参数错误

提交以下非法参数：

```text
prompt: a fox
steps: 10
decode_interval: 11
width: 512
height: 512
```

确认：

- 页面错误区域显示参数错误。
- 后端返回 422。
- 不启动生成任务。

## 7. 停止服务

在运行服务的终端按：

```text
Ctrl+C
```

确认服务退出，没有留下后台进程。
