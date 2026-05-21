import speech_recognition as sr
import time
import threading


class VoiceAssistant:
    def __init__(self, wake_word="你好助手", language="zh-CN"):
        self.wake_word = wake_word
        self.language = language
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.is_running = True

    def listen_for_wake_word(self):
        with sr.Microphone() as source:
            print(f"正在监听唤醒词: '{self.wake_word}'...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"听到: {text}")
            if self.wake_word in text:
                return True
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"语音服务错误: {e}")
        return False

    def transcribe_speech(self):
        with sr.Microphone() as source:
            print("请说话...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"识别结果: {text}")
            return text
        except sr.UnknownValueError:
            print("无法识别语音")
        except sr.RequestError as e:
            print(f"语音服务错误: {e}")
        except sr.WaitTimeoutError:
            print("等待超时")
        return None

    def run(self):
        print(f"语音助手已启动，唤醒词: '{self.wake_word}'")
        print("按 Ctrl+C 退出程序\n")

        while self.is_running:
            try:
                if self.listen_for_wake_word():
                    print("\n唤醒词已识别！现在开始语音转文字...")
                    self.is_listening = True
                    while self.is_listening:
                        result = self.transcribe_speech()
                        if result:
                            print(f"记录: {result}\n")
                        print("继续说话或说 '停止' 来结束...")
                        if result and "停止" in result:
                            print("停止监听，等待唤醒词...\n")
                            self.is_listening = False
                            break
            except KeyboardInterrupt:
                print("\n正在退出...")
                self.is_running = False
                break
            except Exception as e:
                print(f"发生错误: {e}")
                time.sleep(1)


if __name__ == "__main__":
    assistant = VoiceAssistant(wake_word="你好助手")
    assistant.run()
