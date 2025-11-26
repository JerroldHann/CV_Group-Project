import os, cv2, threading, speech_recognition as sr, pyttsx3
from ultralytics import YOLOWorld
from intent_v7 import IntentInference
from OCR_v2 import OCRProcessor  # 新增OCR导入
from sklearn.cluster import KMeans
import numpy as np

# 🔽 新增：场景 caption 模型相关
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from PIL import Image

# 🔽 新增：Web 端展示相关
from flask import Flask, Response, jsonify, render_template_string
import time

# ==================== 全局 Web 状态 ====================
app = Flask(__name__)

FRAME_LOCK = threading.Lock()
LATEST_FRAME_BYTES = None  # 存 JPEG 编码后的最新一帧

EVENTS_LOCK = threading.Lock()
EVENTS = []  # 每条: {"time": "21:03:12", "command": "...", "response": "..."}

def register_command(cmd: str):
    """记录最新一条用户语音指令（Web 右侧用）"""
    t = time.strftime("%H:%M:%S")
    with EVENTS_LOCK:
        EVENTS.append({
            "time": t,
            "command": cmd,
            "response": ""
        })
        # 控制长度，最多保留最近 50 条
        if len(EVENTS) > 50:
            del EVENTS[:-50]


def register_response(resp: str):
    """把最新一条事件的 response 填上"""
    with EVENTS_LOCK:
        if EVENTS:
            EVENTS[-1]["response"] = resp


def set_latest_frame(frame_bgr):
    """把当前帧编码成 JPEG，供 /video_feed 使用"""
    global LATEST_FRAME_BYTES
    try:
        ret, jpeg = cv2.imencode(".jpg", frame_bgr)
        if not ret:
            return
        with FRAME_LOCK:
            LATEST_FRAME_BYTES = jpeg.tobytes()
    except Exception as e:
        print(f"[Web] 编码帧失败: {e}")


def gen_frames():
    """MJPEG 流生成器，用于 <img src='/video_feed'>"""
    global LATEST_FRAME_BYTES
    while True:
        with FRAME_LOCK:
            frame = LATEST_FRAME_BYTES
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)  # 控制刷新频率，大约 30fps 左右


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLOWorld Visual Assistant</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      background: radial-gradient(circle at top left, #1f2933, #020617);
      color: #e5e7eb;
      height: 100vh;
      overflow: hidden;
    }
    .container {
      display: flex;
      height: 100vh;
      padding: 16px;
      gap: 16px;
    }
    .video-panel {
      flex: 2;
      background: #020617;
      border-radius: 16px;
      padding: 8px;
      box-shadow: 0 20px 40px rgba(0,0,0,.45);
      display: flex;
      flex-direction: column;
    }
    .video-panel h2 {
      font-size: 18px;
      margin-bottom: 8px;
      color: #e5e7eb;
      letter-spacing: .03em;
      text-transform: uppercase;
      font-weight: 600;
    }
    .video-wrapper {
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .side-panel {
      flex: 1.2;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card {
      background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(15,23,42,.9));
      border-radius: 16px;
      padding: 12px 14px;
      box-shadow: 0 18px 40px rgba(0,0,0,.4);
      border: 1px solid rgba(148,163,184,.25);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .card-title {
      font-size: 16px;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #9ca3af;
    }
    .badge {
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 13px;
      background: rgba(56,189,248,.08);
      color: #7dd3fc;
      border: 1px solid rgba(125,211,252,.4);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,.35);
    }
    .command-text {
      font-size: 20px;
      color: #e5e7eb;
      margin-bottom: 6px;
      word-break: break-word;
    }
    .response-text {
      font-size: 18px;
      color: #cbd5f5;
      word-break: break-word;
    }
    .timestamp {
      font-size: 13px;
      color: #6b7280;
      margin-top: 4px;
    }
    .history-list {
      max-height: 60vh;
      overflow-y: auto;
      padding-right: 4px;
    }
    .history-item {
      padding: 8px 6px;
      border-radius: 10px;
      border: 1px solid transparent;
      margin-bottom: 6px;
    }
    .history-item.latest {
      border-color: rgba(94,234,212,.7);
      background: rgba(45,212,191,.08);
    }
    .history-item small {
      font-size: 13px;
      color: #9ca3af;
    }
    .history-command {
      font-size: 16px;
      color: #e5e7eb;
      margin-top: 2px;
      word-break: break-word;
    }
    .history-response {
      font-size: 15px;
      color: #cbd5f5;
      margin-top: 2px;
      word-break: break-word;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="video-panel">
      <h2>Live camera</h2>
      <div class="video-wrapper">
        <img src="/video_feed" alt="Video stream">
      </div>
    </div>
    <div class="side-panel">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Current interaction</div>
          <div class="badge">
            <span class="status-dot"></span>
            LIVE
          </div>
        </div>
        <div id="current-command" class="command-text">Waiting for voice command...</div>
        <div id="current-response" class="response-text"></div>
        <div id="current-time" class="timestamp"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">History</div>
        </div>
        <div id="history" class="history-list"></div>
      </div>
    </div>
  </div>

  <script>
    async function fetchEvents() {
      try {
        const res = await fetch('/events');
        if (!res.ok) return;
        const data = await res.json();
        renderEvents(data);
      } catch (e) {
        console.error(e);
      }
    }

    function renderEvents(events) {
      const historyEl = document.getElementById('history');
      const currentCmdEl = document.getElementById('current-command');
      const currentRespEl = document.getElementById('current-response');
      const currentTimeEl = document.getElementById('current-time');

      if (!events || events.length === 0) {
        currentCmdEl.textContent = 'Waiting for voice command...';
        currentRespEl.textContent = '';
        currentTimeEl.textContent = '';
        historyEl.innerHTML = '';
        return;
      }

      const latest = events[events.length - 1];
      currentCmdEl.textContent = latest.command || '(no command)';
      currentRespEl.textContent = latest.response || 'Thinking / waiting for response...';
      currentTimeEl.textContent = latest.time || '';

      historyEl.innerHTML = '';
      for (let i = events.length - 1; i >= 0; i--) {
        const ev = events[i];
        const div = document.createElement('div');
        div.className = 'history-item' + (i === events.length - 1 ? ' latest' : '');
        div.innerHTML = `
          <small>${ev.time || ''}</small>
          <div class="history-command">🗣 ${ev.command || ''}</div>
          <div class="history-response">💬 ${ev.response || ''}</div>
        `;
        historyEl.appendChild(div);
      }
    }

    fetchEvents();
    setInterval(fetchEvents, 1000);
  </script>
</body>
</html>
"""

# ==================== Flask 路由 ====================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def get_events():
    with EVENTS_LOCK:
        data = list(EVENTS)
    return jsonify(data)


# ==================== 原 YOLOWorldDetector ====================

class YOLOWorldDetector:
    """YOLOWorld 实时检测 + 异步语音 + 即时意图推理 + OCR"""

    def __init__(self, classes=None, camera_index=0, show_window=False, verbose=False):
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        self.verbose = verbose
        self.show_window = show_window
        self.camera_index = camera_index
        self.running = True
        self.k = 3  # K-means 聚类的簇数（最终颜色数量上限）

        # 场景推理相关
        self.scene_interval = 50          # 每 50 帧推理一次场景
        self.last_scene_text = ""         # 最近一次场景描述

        # 场景 caption 模型
        self.scene_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            scene_model_name = "nlpconnect/vit-gpt2-image-captioning"
            self.scene_model = VisionEncoderDecoderModel.from_pretrained(scene_model_name).to(self.scene_device)
            self.scene_processor = AutoImageProcessor.from_pretrained(scene_model_name)
            self.scene_tokenizer = AutoTokenizer.from_pretrained(scene_model_name)
            self.scene_model.eval()
            if self.verbose:
                print(f"[Scene] Loaded caption model: {scene_model_name} on {self.scene_device}")
        except Exception as e:
            print(f"[Scene] 场景 caption 模型加载失败: {e}")
            self.scene_model = None
            self.scene_processor = None
            self.scene_tokenizer = None

        self.model = YOLOWorld("yolov8s-worldv2.pt")
        try:
            self.model.set_classes(classes or ["door", "chair", "table", "stairs", "person", "bicycle", "car"])
        except Exception as e:
            print("CLIP 模型加载失败:", e)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if self.show_window:
            cv2.namedWindow("YOLOWorld - Detection", cv2.WINDOW_NORMAL)

        # === 初始化OCR处理器 ===
        self.ocr_processor = OCRProcessor(enabled=True, process_interval=5)

        # === 初始化意图推理引擎 ===
        self.intent_engine = IntentInference(output_func=self.speak_response)
        self.last_detections = []

        # === 启动语音监听线程 ===
        threading.Thread(target=self.listen_command, daemon=True).start()

        self.basic_colors = {
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "cyan": [0, 255, 255],
            "magenta": [255, 0, 255],
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "gray": [128, 128, 128],
            "brown": [165, 42, 42],
            "orange": [255, 165, 0],
            "pink": [255, 192, 203]
        }

    # ---------------------------- 语音监听 ----------------------------
    def listen_command(self):
        rec = sr.Recognizer()

        rec.energy_threshold = 300
        rec.dynamic_energy_threshold = True
        rec.pause_threshold = 1.2
        rec.non_speaking_duration = 0.6
        rec.phrase_threshold = 0.3

        mic = sr.Microphone(sample_rate=32000)

        with mic as src:
            print("正在校准环境噪声，请保持安静 2 秒...")
            rec.adjust_for_ambient_noise(src, duration=2.0)
            print(f"校准完成，当前能量阈值: {rec.energy_threshold}")

        while self.running:
            try:
                with mic as src:
                    print("可以开始说话了（最多 20 秒，停顿 >1.2 秒会自动结束）...")
                    audio = rec.listen(
                        src,
                        timeout=15,
                        phrase_time_limit=20
                    )

                try:
                    cmd = rec.recognize_google(audio, language='en-US')
                except sr.UnknownValueError:
                    print("没听清你说什么（UnknownValueError），忽略这轮。")
                    continue
                except sr.RequestError as e:
                    print(f"Google API 出错：{e}")
                    continue

                cmd = cmd.strip().lower()
                if len(cmd) < 3 or len(cmd.split()) <= 1:
                    print(f"识别结果太短，疑似没说完：{cmd!r}，本轮丢弃。")
                    continue

                print(f"识别到语音指令: {cmd}")
                # 🔽 同时记录到 Web 历史
                register_command(cmd)

                self.intent_engine.infer_now(cmd)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"语音监听异常：{e}")
                continue

    # ---------------------------- 异步语音播报 ----------------------------
    def speak_response(self, response):
        def _speak():
            print(f"[意图结果] {response}")
            # 🔽 写入 Web 状态
            register_response(response)

            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 170)
                engine.say(response)
                engine.runAndWait()
            except Exception as e:
                print(f"语音合成错误: {e}")

        threading.Thread(target=_speak, daemon=True).start()

    # ---------------------------- 场景 caption 推理 ----------------------------
    def _caption_scene_once(self, frame, max_length=20):
        if self.scene_model is None or self.scene_processor is None or self.scene_tokenizer is None:
            return ""

        if frame is None or frame.size == 0:
            return ""

        h, w = frame.shape[:2]
        scale = 256.0 / max(h, w)
        if scale < 1.0:
            img_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_small = frame

        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs = self.scene_processor(images=pil_img, return_tensors="pt").to(self.scene_device)

        with torch.no_grad():
            output_ids = self.scene_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        caption = self.scene_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    def infer_scene(self, frame):
        try:
            text = self._caption_scene_once(frame)
            if text:
                return text
        except Exception as e:
            if self.verbose:
                print(f"[Scene] 场景 caption 推理失败: {e}")
        return self.last_scene_text or ""

    # ---------------------------- 颜色相关 ----------------------------
    def color_distance(self, color1, color2):
        return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

    def get_closest_color(self, color):
        rgb = np.clip(np.array(color, dtype=np.float32), 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
        lab = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        H, S, V = hsv

        s_norm = S / 255.0
        v_norm = V / 255.0

        r, g, b = rgb.astype(np.float32) / 255.0
        channel_range = max(r, g, b) - min(r, g, b)

        SAT_GRAY_THR = 0.12
        RANGE_GRAY_THR = 0.06

        if s_norm < SAT_GRAY_THR and channel_range < RANGE_GRAY_THR:
            if v_norm < 0.20:
                return "black"
            elif v_norm > 0.65:
                return "white"
            else:
                return "gray"

        gray_like = {"black", "white", "gray"}
        candidates = [
            (name, np.clip(np.array(rgb_val, dtype=np.float32), 0, 255).astype(np.uint8))
            for name, rgb_val in self.basic_colors.items()
            if name not in gray_like
        ]
        if not candidates:
            candidates = [
                (name, np.clip(np.array(rgb_val, dtype=np.float32), 0, 255).astype(np.uint8))
                for name, rgb_val in self.basic_colors.items()
            ]

        h_deg = (H * 2.0) % 360.0

        best_name = None
        best_score = float("inf")

        for name, base_rgb in candidates:
            base_hsv = cv2.cvtColor(base_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
            base_lab = cv2.cvtColor(base_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

            d_lab = np.linalg.norm(lab - base_lab)

            H_b = base_hsv[0]
            h_b_deg = (H_b * 2.0) % 360.0
            dh = abs(h_deg - h_b_deg)
            dh = min(dh, 360.0 - dh)
            dh_norm = dh / 180.0

            score = d_lab + 30.0 * (dh_norm ** 2)

            if score < best_score:
                best_score = score
                best_name = name

        return best_name

    def extract_colors(self, image):
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if h <= 1 or w <= 1:
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)

        num_pixels = pixels.shape[0]
        if num_pixels == 0:
            return []

        ys, xs = np.indices((h, w))
        xs = xs.flatten().astype(np.float32)
        ys = ys.flatten().astype(np.float32)

        max_samples = 3000
        if num_pixels > max_samples:
            sample_idx = np.random.choice(num_pixels, max_samples, replace=False)
        else:
            sample_idx = np.arange(num_pixels)

        sample_pixels = pixels[sample_idx]
        sample_xs = xs[sample_idx]
        sample_ys = ys[sample_idx]

        if sample_pixels.shape[0] < 2:
            fg_pixels = pixels
        else:
            seg_k = min(4, sample_pixels.shape[0])
            seg_kmeans = KMeans(n_clusters=seg_k, n_init=4, max_iter=40)
            seg_labels = seg_kmeans.fit_predict(sample_pixels)

            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            half_w = w * 0.3
            half_h = h * 0.3
            central_mask = (
                (np.abs(sample_xs - cx) <= half_w) &
                (np.abs(sample_ys - cy) <= half_h)
            )

            cluster_scores = []
            for c in range(seg_k):
                mask_c = (seg_labels == c)
                size_c = np.sum(mask_c)
                if size_c == 0:
                    cluster_scores.append(0.0)
                    continue
                central_ratio = np.sum(mask_c & central_mask) / float(size_c)
                score = central_ratio * size_c
                cluster_scores.append(score)

            cluster_scores = np.array(cluster_scores, dtype=np.float32)
            best_score = float(cluster_scores.max()) if cluster_scores.size > 0 else 0.0

            if best_score <= 0:
                center_mask_all = (
                    (np.abs(xs - cx) <= half_w) &
                    (np.abs(ys - cy) <= half_h)
                )
                if np.any(center_mask_all):
                    fg_pixels = pixels[center_mask_all]
                else:
                    fg_pixels = pixels
            else:
                keep_clusters = np.where(cluster_scores >= 0.3 * best_score)[0]
                fg_mask_sample = np.isin(seg_labels, keep_clusters)
                fg_idx_global = sample_idx[fg_mask_sample]
                if fg_idx_global.size == 0:
                    fg_pixels = pixels
                else:
                    fg_pixels = pixels[fg_idx_global]

        num_fg = fg_pixels.shape[0]
        if num_fg == 0:
            return []

        n_clusters = min(self.k, num_fg)
        if n_clusters < 1:
            return []

        color_kmeans = KMeans(n_clusters=n_clusters, n_init=6, max_iter=50)
        color_labels = color_kmeans.fit_predict(fg_pixels)
        centers = color_kmeans.cluster_centers_
        counts = np.bincount(color_labels, minlength=n_clusters)
        proportions = counts / float(num_fg)

        name_to_prop = {}
        for i in range(n_clusters):
            prop = proportions[i]
            if prop <= 0:
                continue
            basic_name = self.get_closest_color(centers[i])
            name_to_prop[basic_name] = name_to_prop.get(basic_name, 0.0) + float(prop)

        filtered_items = [
            (name, p) for name, p in name_to_prop.items()
            if p >= 0.25
        ]

        if not filtered_items:
            filtered_items = sorted(name_to_prop.items(), key=lambda kv: kv[1], reverse=True)[:self.k]

        filtered_items = sorted(filtered_items, key=lambda kv: kv[1], reverse=True)
        color_names = [f"{name}: {int(p * 100)}%" for name, p in filtered_items]

        return color_names

    # ---------------------------- 主检测循环 ----------------------------
    def run(self):
        frame_count = 0
        skip_interval = 20  # 越大越流畅，越小越准

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLO 使用原始分辨率
            frame_for_yolo = frame

            # OCR 使用放大后的图像（例如放大 2 倍）
            ocr_frame = cv2.resize(
                frame, None,
                fx=2.0, fy=2.0,
                interpolation=cv2.INTER_CUBIC  # 双三次插值，对文字边缘友好
            )

            frame_for_stream = frame_for_yolo  # 默认给 Web 的还是原始/检测后的画面

            frame_count += 1
            frame_for_stream = frame  # 默认推原始帧

            if frame_count % self.scene_interval == 0:
                self.last_scene_text = self.infer_scene(frame)
            scene_text = self.last_scene_text

            if frame_count % skip_interval != 0:
                # 跳帧时依然更新 Web 端画面
                set_latest_frame(frame_for_stream)
                if self.show_window:
                    cv2.imshow("YOLOWorld - Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # YOLO目标检测
            results = self.model.predict(frame_for_yolo, imgsz=512, verbose=False)

            r = results[0]

            detections = [
                {
                    "class": r.names[int(b.cls[0])],
                    "confidence": float(b.conf[0]),
                    "bbox": list(map(int, b.xyxy[0])),
                    "scene": scene_text,
                }
                for b in r.boxes
            ]

            if not detections and scene_text:
                detections.append({
                    "class": "scene",
                    "confidence": 1.0,
                    "bbox": [0, 0, 0, 0],
                    "colors": [],
                    "scene": scene_text
                })

            for detection in detections:
                if detection["class"] == "scene":
                    detection.setdefault("colors", [])
                    continue

                if detection["class"] in ["boy", "girl"]:
                    detection["colors"] = "Not applicable for life forms"
                    continue

                x1, y1, x2, y2 = detection["bbox"]
                object_image = frame[y1:y2, x1:x2]
                colors = self.extract_colors(object_image)
                detection["colors"] = colors

            self.last_detections = detections
            print(detections)

            ocr_results = self.ocr_processor.get_ocr_results()
            self.intent_engine.update_vision(detections, ocr_results)
            self.ocr_processor.process_frame_async(ocr_frame)

            annotated = r.plot()  # 画框
            frame_for_stream = annotated  # Web 使用带框画面

            # 推给 Web
            set_latest_frame(frame_for_stream)

            if self.show_window:
                # 在画面上显示OCR结果
                if ocr_results:
                    y_offset = 30
                    for i, ocr_item in enumerate(ocr_results[:3]):
                        text = f"Text: {ocr_item['text']}"
                        cv2.putText(
                            annotated,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
                        y_offset += 25

                if scene_text:
                    cv2.putText(
                        annotated,
                        f"Scene: {scene_text}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow("YOLOWorld - Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        self.running = False
        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()


# ==================== 主入口：开 YOLO 线程 + Flask ====================

def start_detector():
    detector = YOLOWorldDetector(
        classes=[
            "boy", "girl",
            "shirt", "jacket", "coat", "pants", "shorts", "dress", "skirt", "shoe",
            "hat", "cap", "helmet", "mask", "scarf", "gloves", "backpack", "watch",
            "door", "window", "stairs", "chair", "table", "sofa", "bed",
            "keyboard", "mouse", "phone", "cup", "bottle",
            "sink", "toilet", "microwave", "oven",
            "car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane",
            "boat", "traffic light", "stop sign", "bench", "bridge", "crosswalk",
            "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "zebra", "giraffe",
            "obstacle", "wall", "floor", "pole", "cone", "barrier", "tree", "bush", "grass",
            "bowl", "cup", "bottle", "can", "pen", "paper",
            "bag", "bin", "umbrella", "broom", "trash can",
            "turnstile", "gate machine",
            "seagull",
            "road", "telegraph pole", "orange", "tangerine"
            "bollard",  # 石墩子，视为障碍物
            "tactile paving"  # 盲道，引导路径
        ],
        show_window=False,  # 主看网页，就关掉本地窗口；要窗口就改 True
        verbose=False
    )
    detector.run()


if __name__ == "__main__":
    # YOLO + 语音 + 推理 放在后台线程，主线程跑 Flask
    t = threading.Thread(target=start_detector, daemon=True)
    t.start()

    # Flask Web 服务
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
