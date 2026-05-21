# Windows 安装和使用说明

本程序完全支持 Windows 系统！

## 前置要求

- Python 3.8 或更高版本
- 麦克风设备

## Windows 安装步骤

### 方法 1：使用预编译包（推荐）

1. 首先安装 Microsoft C++ Build Tools（如果还没有的话）：
   - 下载并安装 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

2. 安装依赖：
   ```bash
   pip install SpeechRecognition
   pip install pipwin
   pipwin install pyaudio
   ```

### 方法 2：使用 conda（如果你有 Anaconda/Miniconda）

```bash
conda install pyaudio
pip install SpeechRecognition
```

## 运行程序

```bash
python voice_assistant.py
```

## 使用说明

1. 程序启动后会监听唤醒词"你好助手"
2. 说出唤醒词后开始语音转文字
3. 说"停止"结束识别，回到等待状态
4. 按 `Ctrl+C` 完全退出

## 常见问题

### 1. PyAudio 安装失败
使用 `pipwin` 来安装预编译版本：
```bash
pip install pipwin
pipwin install pyaudio
```

### 2. 找不到麦克风
确保 Windows 系统中麦克风已正确连接并设为默认录音设备。

### 3. 识别准确率低
- 确保环境安静
- 麦克风靠近嘴边
- 说话清晰、语速适中
