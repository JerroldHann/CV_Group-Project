# Visual Assist: YOLOWorld + OCR + Gemini 视觉无障碍助手

本项目是一个面向视障/低视力用户的 **实时视觉辅助系统** 原型。  
它通过摄像头画面，结合：

- **YOLOWorld 目标检测**（Ultralytics YOLOv8-Worldv2）
- **EasyOCR 文本识别（OCR）**
- **语音识别 + TTS 语音播报**
- **Google Gemini 大模型意图推理**

来实现一个可以“看环境、听指令、说反馈”的视觉助手。

---

## 功能特性

- **YOLOWorld 实时目标检测**
  - 支持丰富的检测类别：人物、人体部位、家具、交通工具、动物、障碍物、日用品等
  - 可自定义类别列表，在初始化 `YOLOWorldDetector` 时传入 `classes` 参数

- **Gemini 加持的意图推理（IntentInference）**
  - 基于用户语音指令（command）+ 视觉检测结果（objects）+ OCR 文本（ocr_texts）
  - 使用精心设计的英文 system prompt，让 Gemini 输出简洁、聚焦视觉的回答
  - 带超时保护（默认 4.5 秒），失败时支持回退到规则逻辑（预留接口）

- **快速英文 OCR 识别（OCRProcessor）**
  - 使用 EasyOCR，仅加载英文模型，适配低分辨率/噪声较大的画面
  - 内置图像预处理（灰度 + CLAHE 对比度增强 + 轻量去噪）
  - 异步处理帧：**不阻塞 YOLO 的实时检测**

- **语音交互**
  - 使用 `speech_recognition` 调用麦克风进行语音识别（Google STT）
  - 使用 `pyttsx3` 在本地进行 TTS 播报，免联网语音合成
  - 语音指令线程与视觉主循环并行，互不阻塞

- **多线程架构**
  - 主线程：摄像头读取 + YOLO 检测 + 画面显示
  - 后台线程1：语音指令监听
  - 后台线程2：异步 OCR 处理
  - 后台线程3：意图结果的语音播报

---

## 项目结构

```text
.
├── yolo.py         # YOLOWorldDetector：主入口，负责摄像头采集、YOLO 检测、语音线程启动、OCR 调度
├── intent.py       # IntentInference：意图推理引擎，封装 Gemini 调用与规则逻辑预留
└── ocr.py          # OCRProcessor：EasyOCR 英文文字识别，异步处理帧，提供最新 OCR 结果
