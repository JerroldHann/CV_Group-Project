import cv2
import numpy as np
import threading
import time
import re

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
        self.last_ocr_results = []
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
            # 只加载英文模型，gpu=False使用CPU，减小内存占用
            self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=None, download_enabled=True)
            print("✅ OCR引擎初始化完成")
        except Exception as e:
            print(f"❌ OCR引擎初始化失败: {e}")
            self.enabled = False
            self.reader = None

    def preprocess_frame(self, frame):
        """预处理图像以提高OCR准确度"""
        if frame is None:
            return None

        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 图像增强 - 提高对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 轻微高斯模糊去噪
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return denoised

    # def preprocess_frame(self, frame):
    #     """预处理图像以提高OCR准确度
    #     strong=True 时会做更激进的文字增强（二值化 + 形态学），适合白底黑字/屏幕拍摄
    #     """
    #     strong = True
    #
    #     if frame is None:
    #         return None
    #
    #     # 1. 灰度
    #     if len(frame.shape) == 3:
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = frame
    #
    #     # 2. 适当放大（小分辨率时）
    #     h, w = gray.shape[:2]
    #     if max(h, w) < 720:
    #         gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    #
    #     # 3. CLAHE 提升局部对比度
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     enhanced = clahe.apply(gray)
    #
    #     # 4. 中值滤波去噪，保留边缘
    #     denoised = cv2.medianBlur(enhanced, 3)
    #
    #     if not strong:
    #         # 通用、偏保守的预处理：直接给 easyocr 灰度图/增强图用
    #         return denoised
    #
    #     # ====== 强化文字分支（可选） ======
    #     # 5. Otsu 二值化（黑白文字最清晰）
    #     _, binary = cv2.threshold(
    #         denoised, 0, 255,
    #         cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #     )
    #
    #     # 6. 形态学闭运算，让笔画更连贯
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #     strong_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    #     return strong_img

    def clean_text(self, text):
        """清理非英文字母和标点符号"""
        # 移除除了字母、数字、空格、常见标点外的所有字符
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        # 如果文本长度大于0，返回清理后的文本
        return cleaned.strip() if len(cleaned.strip()) > 0 else None

    def extract_text_from_frame(self, frame):
        """从帧中提取英文文本"""
        if not self.enabled or self.reader is None:
            return []

        try:
            # 预处理图像
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return []

            # 使用EasyOCR提取文本
            results = self.reader.readtext(processed_frame, detail=1, paragraph=False, min_size=10,
                                          text_threshold=0.5, low_text=0.3, link_threshold=0.4)

            # 过滤和整理结果
            filtered_results = []
            for (bbox, text, confidence) in results:
                if (confidence >= self.confidence_threshold and
                        len(text.strip()) >= 2 and  # 至少2个字符
                        any(c.isalnum() for c in text)):  # 包含字母或数字

                    # 清理文本
                    clean_text = self.clean_text(text)
                    if clean_text:
                        filtered_results.append({
                            'text': clean_text,
                            'confidence': float(confidence),
                            'bbox': [[int(x), int(y)] for x, y in bbox]  # 边界框坐标
                        })

            return filtered_results

        except Exception as e:
            print(f"❌ OCR处理异常: {e}")
            return []

    def process_frame_async(self, frame):
        """异步处理帧（非阻塞）"""
        if not self.enabled or self.reader is None:
            return

        current_time = time.time()
        self.frame_counter += 1

        # 控制处理频率
        if (self.frame_counter % self.process_interval != 0 or
                current_time - self.last_processed_time < 0.5 or  # 至少0.5秒间隔
                self.processing):
            return

        self.processing = True
        self.last_processed_time = current_time

        def _process():
            try:
                results = self.extract_text_from_frame(frame)
                with self.lock:
                    self.last_ocr_results = results
                if results and len(results) > 0:
                    texts = [r['text'] for r in results[:3]]  # 只显示前3个
                    print(f"📝 OCR识别到文字: {texts}")
            except Exception as e:
                print(f"❌ OCR处理线程异常: {e}")
            finally:
                self.processing = False

        # 在后台线程中处理
        threading.Thread(target=_process, daemon=True).start()

    def get_ocr_results(self):
        """获取最新的OCR结果"""
        with self.lock:
            return self.last_ocr_results.copy()

    def is_enabled(self):
        """检查OCR是否启用"""
        return self.enabled and self.reader is not None
