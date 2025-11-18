import os, cv2, threading, speech_recognition as sr, pyttsx3
from ultralytics import YOLOWorld
from intent import IntentInference
from ocr import OCRProcessor

class YOLOWorldDetector:
    """YOLOWorld 实时检测 + 异步语音 + 即时意图推理 + OCR"""

    def __init__(self, classes=None, camera_index=1, show_window=False, verbose=False):
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        self.verbose = verbose
        self.show_window = show_window
        self.camera_index = camera_index
        self.running = True

        self.model = YOLOWorld("../config/yolov8s-worldv2.pt")
        try:
            self.model.set_classes(classes or ["door", "chair", "table", "stairs", "person", "bicycle", "car"])
        except Exception as e:
            print("CLIP 模型加载失败:", e)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 修复：移除 cv2.startWindowThread()，直接创建窗口
        if self.show_window:
            cv2.namedWindow("YOLOWorld - Detection", cv2.WINDOW_NORMAL)

        # === 初始化OCR处理器 ===
        self.ocr_processor = OCRProcessor(enabled=True, process_interval=5)

        # === 初始化意图推理引擎 ===
        self.intent_engine = IntentInference(output_func=self.speak_response)
        self.last_detections = []

        # === 启动语音监听线程 ===
        threading.Thread(target=self.listen_command, daemon=True).start()

    # ---------------------------- 语音监听 ----------------------------
    def listen_command(self):
        rec = sr.Recognizer()
        while self.running:
            try:
                with sr.Microphone(sample_rate=8000) as src:
                    rec.adjust_for_ambient_noise(src, duration=0.5)
                    print("🎤 正在监听语音指令...")
                    audio = rec.listen(src, timeout=5, phrase_time_limit=4)
                    cmd = rec.recognize_google(audio, language='en-US') #zh-CN是中文
                    cmd = cmd.strip().lower()  # 统一小写处理
                    print(f"🗣️ 识别到语音指令: {cmd}")
                    # 直接调用即时意图推理
                    self.intent_engine.infer_now(cmd)
            except sr.WaitTimeoutError:
                continue
            except Exception:
                continue

    # ---------------------------- 异步语音播报 ----------------------------
    def speak_response(self, response):
        def _speak():
            print(f"[🧠 意图结果] {response}")
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 170)
                engine.say(response)
                engine.runAndWait()
            except Exception as e:
                print(f"语音合成错误: {e}")

        threading.Thread(target=_speak, daemon=True).start()

    # ---------------------------- 主检测循环 ----------------------------
    def run(self):
        frame_count = 0
        skip_interval = 5  # 越大越流畅，越小越准

        while self.running:
            ret, frame = self.cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % skip_interval != 0:
                if self.show_window:
                    cv2.imshow("YOLOWorld - Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # YOLO目标检测
            results = self.model.predict(frame, imgsz=320, verbose=False)
            r = results[0]
            detections = [
                {"class": r.names[int(b.cls[0])],
                 "confidence": float(b.conf[0]),
                 "bbox": list(map(int, b.xyxy[0]))}
                for b in r.boxes
            ]
            self.last_detections = detections

            # 更新意图引擎的视觉信息（包含OCR结果）
            ocr_results = self.ocr_processor.get_ocr_results()
            self.intent_engine.update_vision(detections, ocr_results)

            # 异步处理OCR（不阻塞主进程）
            self.ocr_processor.process_frame_async(frame)

            if self.show_window:
                annotated = r.plot()

                # 在画面上显示OCR结果
                if ocr_results:
                    y_offset = 30
                    for i, ocr_item in enumerate(ocr_results[:3]):  # 只显示前3个
                        text = f"Text: {ocr_item['text']}"
                        cv2.putText(annotated, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += 25

                cv2.imshow("YOLOWorld - Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27是ESC键
                    break

        self.running = False
        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = YOLOWorldDetector(
        classes=[
            # 👤 人体整体
            "boy", "girl",

            # 🧠 人体部位
            "head", "face", "eye", "nose", "mouth", "ear", "hair",
            "hand", "arm", "shoulder", "elbow", "wrist",
            "leg", "knee", "foot", "ankle", "toe",
            "neck", "back", "chest", "belly", "waist",

            # 👕 穿戴
            "shirt", "jacket", "coat", "pants", "shorts", "dress", "skirt", "shoe",
            "hat", "cap", "helmet", "mask", "scarf", "gloves", "bag", "backpack", "watch",

            # 🪑 环境类（可保持完整列表）
            "door", "window", "stairs", "chair", "table", "sofa", "bed", "tv", "monitor",
            "laptop", "keyboard", "mouse", "phone", "cup", "bottle", "book", "lamp",
            "mirror", "refrigerator", "sink", "toilet", "microwave", "oven",

            # 🚗 户外/交通
            "car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane",
            "boat", "traffic light", "stop sign", "bench", "bridge", "crosswalk",

            # 🐶 动物
            "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "zebra", "giraffe",

            # ⚠️ 障碍与环境
            "wall", "floor", "pole", "cone", "barrier", "tree", "bush", "grass",

            # 🍽️ 日用品
            # "knife", "fork", "spoon", "plate"
            "bowl", "cup", "bottle", "can", "pen", "paper"

            # 🧰 其他
            "bag", "box", "bucket", "bin", "remote", "umbrella", "broom", "trash can"
        ],
        show_window=True,
        verbose=False
    )

    detector.run()