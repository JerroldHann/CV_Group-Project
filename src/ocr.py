import cv2
import numpy as np
import threading
import time
import re
from collections import deque

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR 未安装，OCR功能将禁用。请运行: pip install easyocr")


class OCRProcessor:
    """快速英文OCR处理器，针对低分辨率图像优化"""

    def __init__(self, enabled=True, process_interval=10, confidence_threshold=0.6):
        self.enabled = enabled and EASYOCR_AVAILABLE
        self.process_interval = process_interval
        self.confidence_threshold = confidence_threshold
        self.reader = None

        # 最近一次判别结果（单帧）
        self.last_ocr_results = []

        # 最近 3 次判别结果（历史帧集合）
        # 超过 3 次，最老的自动覆盖
        self.history = deque(maxlen=3)

        self.frame_counter = 0
        self.lock = threading.Lock()
        self.last_processed_time = 0
        self.processing = False

        if self.enabled:
            self._initialize_reader()

    def _initialize_reader(self):
        """初始化EasyOCR阅读器（只加载英文模型）"""
        try:
            print("🔄 初始化OCR引擎...")
            self.reader = easyocr.Reader(
                ['en'],
                gpu=False,
                model_storage_directory=None,
                download_enabled=True
            )
            print("OCR引擎初始化完成")
        except Exception as e:
            print(f"OCR引擎初始化失败: {e}")
            self.enabled = False
            self.reader = None

    def preprocess_frame(self, frame):
        """预处理图像以提高OCR准确度"""
        if frame is None:
            return None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return denoised

    def clean_text(self, text):
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        return cleaned.strip() if cleaned.strip() else None

    @staticmethod
    def _bbox_to_rect(bbox):
        if not bbox or len(bbox) != 4:
            return 0, 0, 0, 0
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _is_inside(inner_rect, outer_rect, margin=0.1):
        ix1, iy1, ix2, iy2 = inner_rect
        ox1, oy1, ox2, oy2 = outer_rect

        w = max(1.0, ox2 - ox1)
        h = max(1.0, oy2 - oy1)
        dx = margin * w
        dy = margin * h

        return (
            ix1 >= ox1 - dx and
            iy1 >= oy1 - dy and
            ix2 <= ox2 + dx and
            iy2 <= oy2 + dy
        )

    def extract_text_from_frame(self, frame):
        """单帧 OCR + 去噪 + 单字母过滤 + 单词/字母避免重复"""
        if not self.enabled or self.reader is None:
            return []

        try:
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return []

            results = self.reader.readtext(
                processed_frame,
                detail=1,
                paragraph=False,
                min_size=5,
                text_threshold=0.5,
                low_text=0.3,
                link_threshold=0.4,
                canvas_size=2560,
                mag_ratio=2.0
            )

            # ---- 初筛 ----
            candidates = []
            for (bbox, text, confidence) in results:
                raw = text.strip()
                if not raw:
                    continue

                if not any(c.isalnum() for c in raw):
                    continue

                clean = self.clean_text(raw)
                if not clean:
                    continue

                # 单字符更高置信度
                if len(clean) == 1:
                    if confidence < max(self.confidence_threshold, 0.60):
                        continue
                else:
                    if confidence < self.confidence_threshold:
                        continue

                norm_bbox = [[int(x), int(y)] for x, y in bbox]
                x1, y1, x2, y2 = self._bbox_to_rect(norm_bbox)
                area = max(0, x2 - x1) * max(0, y2 - y1)

                candidates.append({
                    "text": clean,
                    "confidence": float(confidence),
                    "bbox": norm_bbox,
                    "rect": (x1, y1, x2, y2),
                    "area": float(area),
                })

            if not candidates:
                return []

            # ---- 排序：优先文本更长 + 更高置信度 ----
            candidates.sort(key=lambda r: (len(r["text"]), r["confidence"]), reverse=True)

            filtered_results = []
            for cand in candidates:
                txt = cand["text"]
                rect = cand["rect"]

                # 单字符被更长文本包含 → 噪声
                if len(txt) == 1:
                    skip = False
                    for kept in filtered_results:
                        if len(kept["text"]) > 1 and self._is_inside(rect, kept["rect"], margin=0.15):
                            skip = True
                            break
                    if skip:
                        continue

                # 文本+位置近似重复 → 忽略
                dup = False
                for kept in filtered_results:
                    if kept["text"].lower() == txt.lower():
                        kx1, ky1, kx2, ky2 = kept["rect"]
                        cx1 = (rect[0] + rect[2]) / 2
                        cy1 = (rect[1] + rect[3]) / 2
                        cx2 = (kx1 + kx2) / 2
                        cy2 = (ky1 + ky2) / 2
                        if abs(cx1 - cx2) < 10 and abs(cy1 - cy2) < 10:
                            dup = True
                            break
                if dup:
                    continue

                filtered_results.append(cand)

            # ---- 输出格式化 ----
            final_results = []
            for item in filtered_results:
                final_results.append({
                    "text": item["text"],
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                })

            return final_results

        except Exception as e:
            print(f"OCR处理异常: {e}")
            return []

    def process_frame_async(self, frame):
        """异步处理帧（非阻塞）"""
        if not self.enabled or self.reader is None:
            return

        current_time = time.time()
        self.frame_counter += 1

        if (
            self.frame_counter % self.process_interval != 0 or
            current_time - self.last_processed_time < 0.5 or
            self.processing
        ):
            return

        self.processing = True
        self.last_processed_time = current_time

        def _process():
            try:
                results = self.extract_text_from_frame(frame)
                with self.lock:
                    self.last_ocr_results = results
                    if results:
                        # 只保留最近 3 帧
                        self.history.append(results)

                if results:
                    texts = [r['text'] for r in results[:5]]
                    print(f"📝 OCR识别到文字: {texts}")

            except Exception as e:
                print(f"❌ OCR处理线程异常: {e}")

            finally:
                self.processing = False

        threading.Thread(target=_process, daemon=True).start()

    def get_ocr_results(self):
        """
        返回最近 3 次 OCR 结果的合并（去重后）。
        返回结构保持不变，可以直接喂给 IntentInference。
        """
        with self.lock:
            frames = list(self.history)
            last_single = list(self.last_ocr_results)

        if not frames:
            return last_single

        combined = []
        seen = set()

        # 只合并最近 3 次（history.maxlen = 3）
        for frame_results in frames:
            for r in frame_results:
                txt = r.get("text")
                bbox = r.get("bbox")
                if not txt or not bbox or len(bbox) != 4:
                    continue

                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))

                key = (txt.lower(), cx // 20, cy // 20)

                if key in seen:
                    continue

                seen.add(key)
                combined.append(r)

        return combined

    def is_enabled(self):
        return self.enabled and self.reader is not None
