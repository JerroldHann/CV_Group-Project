import os, json, threading, time, configparser
from intent_rules import infer_intent_rules  # 新增：导入规则回退


class IntentInference:
    """即时意图推理（Gemini 主通路 + 本地 Qwen 备份 + 规则融合）"""

    def __init__(self, output_func=None, use_llm=True, llm_timeout=28.5, require_llm=False):
        self.output_func = output_func
        self.last_detections = []
        self.last_ocr_results = []  # 新增：存储OCR结果
        self.lock = threading.Lock()

        # 读取配置（同目录优先）
        cfg = self._load_config()

        self.use_llm = use_llm
        self.llm_timeout = llm_timeout

        # ========= Gemini 配置 =========
        self.gemini_model_name = (
            cfg.get("GEMINI_MODEL")
            or os.getenv("GEMINI_MODEL")
            or "gemini-1.5-flash"
        )
        self.gemini_key = (
            cfg.get("GEMINI_API_KEY")
            or cfg.get("api_key")  # 兼容写法
            or os.getenv("GEMINI_API_KEY")
        )
        self.gemini_model = None

        # ========= Qwen 本地备份配置 =========
        # 默认走 HuggingFace 上的 Qwen/Qwen2.5-7B-Instruct
        self.qwen_model_name = (
            cfg.get("QWEN_MODEL")
            or os.getenv("QWEN_MODEL")
            or "Qwen/Qwen2.5-3B-Instruct"
        )
        # 如果你想用本地路径（比如 D:/models/Qwen-7B），可以把 QWEN_MODEL 换成本地路径
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.qwen_device = "cuda"  # 初始化时再判断

        # ========== 一开始就尝试初始化 Gemini 和 Qwen ==========
        if self.use_llm:
            if self.gemini_key:
                self._init_gemini()
            else:
                print("[Gemini] 未启用：缺少 GEMINI_API_KEY。")

            # 无论有没有 Gemini key，都尝试初始化本地 Qwen（可能失败就打印错误）
            # self._init_qwen()

            if require_llm and (self.gemini_model is None and self.qwen_model is None):
                raise RuntimeError("[LLM] 初始化失败：Gemini 和 Qwen 都不可用。系统需要至少一个 LLM。")
        else:
            print("[LLM] use_llm=False，将只使用规则逻辑。")

    # -------- 读取同目录配置文件 ----------
    def _load_config(self):
        """
        支持以下同目录文件（按顺序优先）：
        - intent.config.json
        - gemini.config.json
        - intent.config.ini   (节名 [gemini] / [qwen])
        返回 dict，键包括：
        - GEMINI_API_KEY / api_key, GEMINI_MODEL / model
        - QWEN_MODEL（可选，本地路径或 HF 名称）
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "intent.config.json"),
            os.path.join(base_dir, "gemini.config.json"),
            os.path.join(base_dir, "intent.config.ini"),
        ]

        cfg = {}
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                if path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f) or {}
                    # 统一键名
                    cfg.update({
                        "GEMINI_API_KEY": data.get("GEMINI_API_KEY") or data.get("api_key"),
                        "GEMINI_MODEL": data.get("GEMINI_MODEL") or data.get("model"),
                        # 可选 Qwen 配置
                        "QWEN_MODEL": data.get("QWEN_MODEL") or data.get("qwen_model"),
                    })
                    print(f"[Gemini/Qwen] 已从配置文件读取：{os.path.basename(path)}")
                    break
                elif path.endswith(".ini"):
                    parser = configparser.ConfigParser()
                    parser.read(path, encoding="utf-8")
                    if parser.has_section("gemini"):
                        sec = parser["gemini"]
                        cfg.update({
                            "GEMINI_API_KEY": sec.get("GEMINI_API_KEY") or sec.get("api_key"),
                            "GEMINI_MODEL": sec.get("GEMINI_MODEL") or sec.get("model"),
                        })
                    if parser.has_section("qwen"):
                        sec_q = parser["qwen"]
                        cfg.update({
                            "QWEN_MODEL": sec_q.get("QWEN_MODEL") or sec_q.get("qwen_model"),
                        })
                    if "GEMINI_API_KEY" in cfg or "QWEN_MODEL" in cfg:
                        print(f"[Gemini/Qwen] 已从配置文件读取：{os.path.basename(path)}")
                        break
            except Exception as e:
                print(f"[Gemini/Qwen] 配置读取失败（{os.path.basename(path)}）：{e}")
        return {k: v for k, v in cfg.items() if v}

    def _get_system_prompt(self):
        """
        Unified system prompt for both Gemini and Qwen.
        Ensures consistent behavior and fluent, natural output without contradictions.
        """
        return (
            "You are a visual assistant for blind and low-vision users.\n"
            "You will receive a single JSON object with these keys:\n"
            "- command: the user's spoken request.\n"
            "- objects: a list of detected visual objects. Each item has:\n"
            "    • class: category name such as 'road', 'door', 'person', 'sign', 'arrow'.\n"
            "    • area: one of ['left','center','right','top','bottom',"
            "'top-left','top-right','bottom-left','bottom-right'] describing where it is.\n"
            "    • colors (optional): a small list of basic color names.\n"
            "    • scene (optional): a short description of the overall scene.\n"
            "- texts: OCR results (a list). Each item has:\n"
            "    • text: the recognized string (e.g., 'EXIT', 'Hong Kong Polytechnic University', 'A',"
            " 'Middle Road', 'All Districts').\n"
            "    • row: a line index (1, 2, 3, ...) where 1 is the top-most row in the image.\n"
            "    • col: a column index (1, 2, 3, ...) where 1 is the left-most column in the image.\n"
            "  Items with the same row belong to approximately the same horizontal line of text.\n"
            "  Items with the same col belong to approximately the same vertical alignment or column.\n"
            "- scene: an object with:\n"
            "    • main: one short sentence summarizing the scene.\n"
            "    • others: optional alternative descriptions.\n"
            "- summary: optional statistics such as unique_classes and has_text.\n"
            "\n"
            "Your goal is to produce a short, coherent, fluent, and helpful description of the surroundings.\n"
            "The reply must always read as a natural unified statement. Do NOT mention missing data, "
            "do NOT say phrases such as 'I don't see anything', 'nothing is detected', or anything similar. "
            "If information is limited, simply describe what is available in a positive, continuous way.\n"
            "\n"
            "Special safety rules for roads, vehicles and obstacles:\n"
            "1) If you detect vehicles such as 'car', 'bus', 'truck', 'bicycle' or 'motorcycle':\n"
            "   - On the user's right side (area includes 'right'): add a short warning like "
            "     'there is a car on your right, please be careful'.\n"
            "   - On the user's left side (area includes 'left'): add a warning like "
            "     'there is a car on your left, please be careful'.\n"
            "   - In front/center: say that a vehicle is ahead and the user should be careful.\n"
            "2) If you detect a 'road', 'street' or 'path':\n"
            "   - If it is mainly on the left, you may add a brief suggestion such as "
            "     'you may move a bit to your left along the road'.\n"
            "   - If it is mainly on the right, suggest moving a bit to the right along the road.\n"
            "   - If it is ahead or centered, you may say that the road continues ahead.\n"
            "3) If you detect obstacles such as 'pole', 'post', 'pillar', 'barrier', 'trash can', "
            "   'bollard', 'lamp post' or similar objects:\n"
            "   - Mention their approximate side (left, right, ahead) and clearly warn that "
            "     there is an obstacle there, for example "
            "     'there is an obstacle on your left, be careful not to bump into it'.\n"
            "4) Add at most one or two short safety sentences, so the overall answer stays concise.\n"
            "\n"
            "Special reasoning rules for OCR and signs:\n"
            "1) Use row and col to understand how texts are grouped:\n"
            "   - Same row → likely the same line on a sign or board.\n"
            "   - Same col → likely stacked lines in the same vertical panel or direction.\n"
            "2) If a row contains words like 'EXIT', a place name, and a letter (A, B, C...), "
            "   treat them as a single exit label (e.g., 'Exit A for the Hong Kong Polytechnic University').\n"
            "3) When different rows or columns describe different destinations (e.g., 'Middle Road' vs 'All Districts'), "
            "   explain clearly which side or direction each one refers to, using the detected objects and scene.\n"
            "4) When the user asks about exits or directions, prioritize:\n"
            "   - EXIT signs and letters (A, B, C...).\n"
            "   - Row/column groups that describe roads or places.\n"
            "   - Simple navigation such as 'in the top row, Exit A leads to the Hong Kong Polytechnic University'.\n"
            "5) Make only simple logical inferences supported by the data. Never invent details.\n"
            "\n"
            "General answering rules:\n"
            "1) Prioritize the 'objects' list. Describe the most relevant 3–5 objects using directional language "
            "   such as 'to your left' or 'ahead'.\n"
            "2) Use 'scene.main' only as extra context, and only if it adds new information beyond the objects.\n"
            "3) Use 'texts' not only to read words but also to REASON about exits, labels, and directions. "
            "   Mention only the key words needed and use their row/col relationships when helpful.\n"
            "4) If the user asks about a specific type of element (door, person, text, obstacles, exits, directions), "
            "   focus the output around it.\n"
            "5) Never mention JSON structure, data fields, coordinates, or numeric values.\n"
            "6) The output must be concise and natural. Normally keep it within 1–2 sentences (~40 words). "
            "   For direction-heavy questions, up to 3 short sentences are allowed.\n"
            "\n"
            "Most importantly: the final answer must always be a smooth, single, human-like description.\n"
            "Do NOT produce fragmented statements or contradictory phrases such as 'I don't see anything, but...'. "
            "Integrate all available information into one fluid explanation.\n"
        )

    # ========== Gemini 初始化 ==========
    def _init_gemini(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)

            system_prompt = self._get_system_prompt()

            try:
                self.gemini_model = genai.GenerativeModel(
                    model=self.gemini_model_name,
                    system_instruction=system_prompt,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 120,
                        "response_mime_type": "text/plain",
                    }
                )
            except TypeError:
                self.gemini_model = genai.GenerativeModel(
                    model_name=self.gemini_model_name,
                    system_instruction=system_prompt,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 120,
                        "response_mime_type": "text/plain",
                    }
                )

            print(f"[Gemini] 已启用模型：{self.gemini_model_name}")
        except Exception as e:
            print(f"[Gemini] 初始化失败：{e}")
            self.gemini_model = None

    # ========== Qwen 初始化（启动时一次性完成）==========
    def _init_qwen(self):
        """
        使用本地 Qwen2.5-7B-Instruct 作为备份 LLM。
        依赖：
            pip install transformers accelerate safetensors

        默认从 HuggingFace 加载 "Qwen/Qwen2.5-7B-Instruct"；
        也可以在环境变量/配置文件里把 QWEN_MODEL 换成本地路径。
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            print(f"[Qwen] 未安装 transformers / torch，跳过 Qwen 初始化：{e}")
            self.qwen_model = None
            self.qwen_tokenizer = None
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.qwen_device = device

            print(f"[Qwen] 正在加载备份模型 {self.qwen_model_name} 到 {device} ... "
                  f"（7B 在 CPU 上会比较慢，如有需要可自行改成 4bit 量化）")

            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model_name,
                trust_remote_code=True,
            )

            # GPU 用 bfloat16，CPU 用 float32（更稳）
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self.qwen_model.to(device)
            self.qwen_model.eval()

            print(f"[Qwen] 备份模型已加载：{self.qwen_model_name} ({device})")
        except Exception as e:
            print(f"[Qwen] 初始化失败，将跳过 Qwen 备份：{e}")
            self.qwen_model = None
            self.qwen_tokenizer = None

    def update_vision(self, detections, ocr_results=None):
        """更新视觉信息，包括目标检测和OCR结果"""
        with self.lock:
            self.last_detections = detections
            self.last_ocr_results = ocr_results or []

    def infer_now(self, cmd_text):
        with self.lock:
            dets = list(self.last_detections)
            ocr_results = list(self.last_ocr_results)

        # 优先尝试 LLM（Gemini -> 本地 Qwen），失败再回退规则
        result = None
        if self.use_llm:
            result = self._infer_intent_llm_timeout(cmd_text, dets, ocr_results, self.llm_timeout)
            if result:
                print("[LLM] 响应已生成（Gemini/Qwen）。")
            else:
                print("[LLM] LLM 超时或失败，回退规则。")

        if not result:
            result = infer_intent_rules(cmd_text, dets, ocr_results)

        if self.output_func:
            threading.Thread(target=self.output_func, args=(result,), daemon=True).start()
        else:
            print(f"[意图结果] {result}")

    # =============== LLM 版（带超时保护） ===============
    def _infer_intent_llm_timeout(self, cmd, vision, ocr_results, timeout_s):
        result_holder = {"text": None}

        def _worker():
            try:
                result_holder["text"] = self._infer_intent_llm(cmd, vision, ocr_results)
            except Exception as e:
                print(f"[LLM] 调用异常（总线）：{e}")
                result_holder["text"] = None

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_s)
        return result_holder["text"]

    # =============== 清洗 + 精简 JSON + Gemini/Qwen 双路 ===============
    def _infer_intent_llm(self, cmd, vision, ocr_results):
        """
        - vision: YOLOWorldDetector 传来的 detections 列表
          每项可能包含: class, confidence, bbox, colors, scene
        - ocr_results: OCRProcessor 的结果列表，预期有 text 和 bbox
        先尝试使用 Gemini；若失败或未配置，则使用本地 Qwen2.5-7B-Instruct 作为备份。
        """
        objs_raw = vision or []
        ocr_raw = ocr_results or []

        FRAME_W = 640.0
        FRAME_H = 480.0

        def _area_label(bbox):
            """根据 bbox 中心点，把位置映射到九宫格区域"""
            if not bbox or len(bbox) != 4:
                return "center"
            x1, y1, x2, y2 = bbox
            try:
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
            except Exception:
                return "center"

            # 特殊情况：全 0 视为默认中心
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                return "center"

            cx = max(0.0, min(FRAME_W, (x1 + x2) / 2.0))
            cy = max(0.0, min(FRAME_H, (y1 + y2) / 2.0))

            x_th1 = FRAME_W / 3.0
            x_th2 = 2.0 * FRAME_W / 3.0
            y_th1 = FRAME_H / 3.0
            y_th2 = 2.0 * FRAME_H / 3.0

            # 水平：left / center / right
            if cx < x_th1:
                horiz = "left"
            elif cx < x_th2:
                horiz = "center"
            else:
                horiz = "right"

            # 垂直：top / center / bottom
            if cy < y_th1:
                vert = "top"
            elif cy < y_th2:
                vert = "center"
            else:
                vert = "bottom"

            if horiz == "center" and vert == "center":
                return "center"
            if vert == "center":
                return horiz
            if horiz == "center":
                return "top" if vert == "top" else "bottom"
            return f"{vert}-{horiz}"  # top-left / bottom-right 等

        def _clean_colors(raw):
            """
            输入可能是:
              - 'pink: 76%'
              - ['pink: 76%', 'white: 10%']
              - set([...])
            输出: ['pink', 'white'] 这种纯色名列表（小写）
            """
            if raw is None:
                return []

            # 统一成列表
            if isinstance(raw, str):
                items = [raw]
            elif isinstance(raw, (list, tuple, set)):
                items = list(raw)
            else:
                return []

            clean = []
            for item in items:
                if not isinstance(item, str):
                    continue
                name = item.split(":")[0].strip().lower()
                if not name:
                    continue
                if name not in clean:
                    clean.append(name)
            return clean

        # ====== 1. 拆出 scene 文本，并清洗 YOLO 对象 ======
        scene_texts = []
        obj_with_area = []  # (area_px, obj_dict) 用来按面积排序

        for d in objs_raw:
            if not isinstance(d, dict):
                continue

            # 收集 scene 文本
            scene_val = d.get("scene")
            if isinstance(scene_val, str) and scene_val.strip():
                scene_texts.append(scene_val.strip())

            cls = d.get("class")
            # class == scene 的当“全局场景”，不当独立物体
            if not cls or cls == "scene":
                continue

            bbox = d.get("bbox") or [0, 0, 0, 0]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bbox = [0, 0, 0, 0]

            try:
                x1, y1, x2, y2 = bbox
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
            except Exception:
                x1 = y1 = 0.0
                x2 = y2 = 0.0

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area_px = max(0.0, w * h)

            area_label = _area_label([x1, y1, x2, y2])
            colors_clean = _clean_colors(d.get("colors"))

            obj = {
                "class": str(cls),
                "area": area_label,
            }
            if colors_clean:
                obj["colors"] = colors_clean

            obj_with_area.append((area_px, obj))

        # 按面积从大到小排，取前若干个
        obj_with_area.sort(key=lambda x: x[0], reverse=True)
        objects = [o for _, o in obj_with_area][:40]

        # ====== 2. scene 主 / 次 选择 ======
        main_scene = ""
        other_scenes = []
        if scene_texts:
            # 使用“最新”的一个作为 main，其余去重后放 others
            main_scene = scene_texts[-1]
            seen = set([main_scene])
            for s in scene_texts[:-1]:
                if s and (s not in seen):
                    seen.add(s)
                    other_scenes.append(s)

        # 把 main_scene 写入每个对象（方便 LLM 在每个物体上下文里用到场景）
        if main_scene:
            for obj in objects:
                obj["scene"] = main_scene

        # ====== 3. 清洗 OCR 结果（text + 行/列编号） ======
        # 基本思路：
        # 1) 从 bbox 提取矩形 (x1,y1,x2,y2)
        # 2) 按 y 方向聚类 → 行 row（1=最上面那行）
        # 3) 按 x 方向聚类 → 列 col（1=最左边那列）
        # 4) 输出结构：{"text": ..., "row": 1, "col": 2}

        # 可调系数：控制“近似”范围（相对于整幅图的比例）
        ROW_TOLERANCE = 0.08  # 行聚类阈值占图像高度的比例，可自行调大/调小
        COL_TOLERANCE = 0.08  # 列聚类阈值占图像宽度的比例，可自行调大/调小

        row_eps = ROW_TOLERANCE * FRAME_H
        col_eps = COL_TOLERANCE * FRAME_W

        # 先把原始 OCR 结果统一整理成带几何信息的列表
        ocr_items = []
        for item in ocr_raw:
            if not isinstance(item, dict):
                continue
            txt = item.get("text")
            if not txt or not isinstance(txt, str):
                continue

            bbox = item.get("bbox") or [0, 0, 0, 0]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bbox = [0, 0, 0, 0]

            try:
                x1, y1 = bbox[0]
                x2, y2 = bbox[2]
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
            except Exception:
                # 兼容 bbox=[x1,y1,x2,y2] 这种形式
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    x1 = y1 = 0.0
                    x2 = y2 = 0.0

            # 规范成 top-left/bottom-right
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0

            ocr_items.append({
                "text": txt.strip(),
                "bbox": [left, top, right, bottom],
                "cx": cx,
                "cy": cy,
            })

        # 如果没有 OCR 文本，直接给空列表
        if not ocr_items:
            texts = []
        else:
            # ---- 行聚类：按 top 排序，自上而下分配 row id ----
            rows = []  # 每个元素: {"top":..., "bottom":..., "indices":[...]}
            for idx, item in enumerate(sorted(ocr_items, key=lambda it: it["bbox"][1])):  # 按 top 升序
                top = item["bbox"][1]
                bottom = item["bbox"][3]

                assigned = False
                for row in rows:
                    r_top, r_bottom = row["top"], row["bottom"]
                    # 行近似条件：top 或 bottom 相差不大即可归为同一行
                    if (abs(top - r_top) <= row_eps) or (abs(bottom - r_bottom) <= row_eps):
                        row["top"] = min(r_top, top)
                        row["bottom"] = max(r_bottom, bottom)
                        row["indices"].append(idx)
                        assigned = True
                        break

                if not assigned:
                    rows.append({
                        "top": top,
                        "bottom": bottom,
                        "indices": [idx],
                    })

            # 行按中心 y 排序，并分配 row_id = 1,2,3...
            rows.sort(key=lambda r: (r["top"] + r["bottom"]) / 2.0)
            row_id_map = {}  # idx -> row_id
            for rid, row in enumerate(rows, start=1):
                for idx in row["indices"]:
                    row_id_map[idx] = rid

            # ---- 列聚类：按 left 排序，自左向右分配 col id ----
            cols = []  # 每个元素: {"left":..., "right":..., "indices":[...]}
            for idx, item in enumerate(sorted(ocr_items, key=lambda it: it["bbox"][0])):  # 按 left 升序
                left = item["bbox"][0]
                right = item["bbox"][2]

                assigned = False
                for col in cols:
                    c_left, c_right = col["left"], col["right"]
                    # 列近似条件：left 或 right 相差不大即可归为同一列
                    if (abs(left - c_left) <= col_eps) or (abs(right - c_right) <= col_eps):
                        col["left"] = min(c_left, left)
                        col["right"] = max(c_right, right)
                        col["indices"].append(idx)
                        assigned = True
                        break

                if not assigned:
                    cols.append({
                        "left": left,
                        "right": right,
                        "indices": [idx],
                    })

            # 列按中心 x 排序，并分配 col_id = 1,2,3...
            cols.sort(key=lambda c: (c["left"] + c["right"]) / 2.0)
            col_id_map = {}  # idx -> col_id
            for cid, col in enumerate(cols, start=1):
                for idx in col["indices"]:
                    col_id_map[idx] = cid

            # ---- 组装最终 texts 列表：只给 LLM text + row + col ----
            texts = []
            for idx, item in enumerate(ocr_items):
                row_id = row_id_map.get(idx, 1)
                col_id = col_id_map.get(idx, 1)
                texts.append({
                    "text": item["text"],
                    "row": int(row_id),
                    "col": int(col_id),
                })

            # 最多保留前 20 个，防止 JSON 太长
            texts = texts[:20]

        # ====== 4. 统计类目（给 LLM 一个 summary，方便避免重复） ======
        class_counts = {}
        for obj in objects:
            name = obj.get("class")
            if not name:
                continue
            class_counts[name] = class_counts.get(name, 0) + 1

        unique_classes = [
            {"class": k, "count": v}
            for k, v in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

        payload = {
            "command": cmd,
            "objects": objects,
            "texts": texts,
            "scene": {
                "main": main_scene,
                "others": other_scenes,
            },
            "summary": {
                "unique_classes": unique_classes,
                "has_text": bool(texts),
            }
        }

        user_prompt = (
            "You will receive the following JSON describing the user's spoken command, "
            "detected objects, OCR texts and overall scene. "
            "Use ONLY this JSON to answer according to the system instructions.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        # ========== 先尝试 Gemini ==========
        if self.gemini_model is not None:
            try:
                resp = self.gemini_model.generate_content(user_prompt)
                text = (getattr(resp, "text", "") or "").strip() if resp else ""
                if text:
                    return text
                else:
                    print("[Gemini] 返回内容为空，尝试本地 Qwen 备份模型...")
            except Exception as e:
                print(f"[Gemini] 调用异常：{e}，尝试本地 Qwen 备份模型...")

        # ========== 再尝试本地 Qwen 备份 ==========
        qwen_text = self._infer_intent_llm_qwen(user_prompt)
        if qwen_text:
            return qwen_text

        # Gemini + Qwen 都不可用，返回 None，让上层回退规则
        return None

    # =============== 本地 Qwen2.5-7B-Instruct 备份调用（只负责推理） ===============
    def _infer_intent_llm_qwen(self, user_prompt: str):
        """
        使用已经在 _init_qwen 中加载好的 Qwen2.5-7B-Instruct 进行推理。
        """
        if self.qwen_model is None or self.qwen_tokenizer is None:
            print("[Qwen] 备份模型未初始化，无法调用 Qwen。")
            return None

        try:
            import torch

            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]

            # Qwen 官方推荐用 chat template
            prompt_text = self.qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.qwen_tokenizer(
                prompt_text,
                return_tensors="pt"
            ).to(self.qwen_device)

            with torch.no_grad():
                output_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,   # 视觉助理更偏确定性
                    temperature=0.2,
                )

            # 只取生成的部分
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            text = self.qwen_tokenizer.decode(
                gen_ids,
                skip_special_tokens=True
            ).strip()

            if text:
                print("[Qwen] 本地备份模型响应已生成。")
            return text or None

        except Exception as e:
            print(f"[Qwen] 推理异常：{e}")
            return None
