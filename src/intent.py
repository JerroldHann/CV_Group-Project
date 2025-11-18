import os, json, threading, time, configparser

class IntentInference:
    """即时意图推理（Gemini + 规则融合；优先读取同目录配置文件）"""

    def __init__(self, output_func=None, use_llm=True, llm_timeout=4.5, require_llm=False):
        self.output_func = output_func
        self.last_detections = []
        self.last_ocr_results = []  # 新增：存储OCR结果
        self.lock = threading.Lock()

        # 读取配置（同目录优先）
        cfg = self._load_config()

        self.use_llm = use_llm
        self.llm_timeout = llm_timeout
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

        if self.use_llm and self.gemini_key:
            self._init_gemini()
        else:
            msg = "[Gemini] 未启用：缺少 GEMINI_API_KEY 或 use_llm=False，将使用规则逻辑。"
            print(msg)
            if require_llm and self.use_llm:
                raise RuntimeError(msg)

    # -------- 读取同目录配置文件 ----------
    def _load_config(self):

        config_path = "../config/intent.config.json"

        cfg = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}

            cfg = {
                "GEMINI_API_KEY": data.get("GEMINI_API_KEY") or data.get("api_key"),
                "GEMINI_MODEL": data.get("GEMINI_MODEL") or data.get("model"),
            }

            print(f"[Gemini] 已从配置文件读取：{config_path}")

        except FileNotFoundError:
            print(f"[Gemini] 配置文件未找到：{config_path}")
        except Exception as e:
            print(f"[Gemini] 配置读取失败：{e}")

        return {k: v for k, v in cfg.items() if v}


    def _init_gemini(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)

            system_prompt = (
                "You are a visual assistant for blind and low-vision users.\n"
                "You will receive a JSON object that contains the user's spoken command "
                "(command) and a list of currently detected objects (objects, including "
                "YOLO detection results and OCR text recognition results).\n"
                "Answer in clear, concise English in 1–2 short sentences. "
                "Do not be overly polite and do not refer to yourself as an AI.\n"
                "\n"
                "General requirements:\n"
                "1) Make full use of each object's pos (left / center / right / above / below), "
                "confidence, area (pixel area), and area_ratio (proportion of the frame; if "
                "missing, approximate with relative area size).\n"
                "2) If there is no relevant object or all confidences are below 0.35, explicitly "
                "say that you are unsure or that nothing was detected.\n"
                "3) Keep the response within about 40 English words; safety-related messages may be slightly longer.\n"
                "4) Do not invent objects that are not in the input. Do not output content "
                "unrelated to vision. Do not expose engineering terms or raw numeric values "
                "such as confidence scores or area ratios.\n"
                "5) You may offer specific, gentle suggestions or guesses based on detections. "
                "For example, if you detect small items like a phone, you may add something like "
                "“Do you want to find your phone?”; if an obstacle is detected, you may add "
                "“There is an obstacle ahead, please walk around it carefully.”; if time-like "
                "text is detected, you may say something like “This looks like a time, it says 21:30.”\n"
                "\n"
                "[OCR text recognition]\n"
                "- If there is clearly recognizable English text (complete words, signs, etc.) in the scene, "
                "briefly mention it in your answer when relevant.\n"
                "- When the user asks “What does it say?”, “What text is here?” or similar, "
                "prioritize describing the OCR-recognized text.\n"
                "- Use text information to help the user identify locations, product names, "
                "directions, signs, and similar cues.\n"
                "\n"
                "[Obstacle rules]\n"
                "A. Obstacle categories (examples, not exhaustive): door (closed / partially blocked), "
                "chair, table, sofa, bed, car, bus, truck, bicycle, motorcycle, wall, pole, cone, "
                "barrier, bench, trash can, box, bin, bucket, tree, bush, etc. "
                "stairs / steps are “level changes” and should be treated as “watch your step” risks.\n"
                "B. Area and approximate distance: primarily use area_ratio (or area if missing) "
                "to estimate how close and blocking an object is. Larger area_ratio means closer "
                "and more likely to block the path.\n"
                "C. Approximate area_ratio thresholds (use them loosely, not as hard rules):\n"
                "   - ≥ 0.12: large obstacle (high risk, likely blocking the way)\n"
                "   - 0.04–0.12: medium obstacle (should be noticed / walked around)\n"
                "   - 0.01–0.04: small obstacle (mind the available space)\n"
                "   If only relative area is known, treat objects with the largest areas in the list "
                "as closer and more important.\n"
                "\n"
                "[Answering strategy when the user asks about obstacles ahead]\n"
                "1) Among objects with pos indicating “in front” (center / front), first filter "
                "for obstacle categories. If any are present with confidence ≥ 0.35, use the "
                "area_ratio (or area) to judge their importance and say that there is an obstacle "
                "ahead, including its type and approximate direction (slightly left / right / straight ahead), "
                "optionally describing it as large / medium / small.\n"
                "2) If there is no obstacle in front:\n"
                "   - If there are people or other non-obstacle objects in the frame, say that the path "
                "ahead is clear but mention nearby people or objects and their general positions.\n"
                "   - If there are no clear objects or it is uncertain, say that the path ahead seems clear "
                "or that no obvious objects were detected.\n"
                "3) When the user asks specifically about a door / person / “what do you see”:\n"
                "   - Door: state whether a door is detected and where it is.\n"
                "   - Person: give approximate direction and count (left / center / right; group them when possible).\n"
                "   - “What do you see”: list about 3–5 main categories, ordered by area or importance.\n"
                "4) When the user asks about text:\n"
                "   - Prioritize describing the OCR-recognized text.\n"
                "   - If no text is recognized, say that no clear text was detected.\n"
                "\n"
                "[Style examples (for reference only; do not copy verbatim)]\n"
                "• “There is likely a large table ahead slightly to your right, it may block your way, please be careful.”\n"
                "• “The path ahead looks clear; I only see two people in front of you.”\n"
                "• “I do not see a door.” / “There is a door on your left.”\n"
                "• “In front of you I see a phone, a chair, and a table; the phone is on the table. "
                "Do you want to find the phone?”\n"
                "• “I can read the word ‘EXIT’; the exit seems to be on your left.”\n"
                "• “There is a STOP sign in view.”\n"
                "• “I cannot see any clear text here.”\n"
            )

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
            print(f"[Gemini] 初始化失败，将回退规则：{e}")
            self.gemini_model = None

    def update_vision(self, detections, ocr_results=None):
        """更新视觉信息，包括目标检测和OCR结果"""
        with self.lock:
            self.last_detections = detections
            self.last_ocr_results = ocr_results or []

    def infer_now(self, cmd_text):
        with self.lock:
            dets = list(self.last_detections)
            ocr_results = list(self.last_ocr_results)

        if self.gemini_model is not None:
            result = self._infer_intent_llm_timeout(cmd_text, dets, ocr_results, self.llm_timeout)
            if result:
                print("[Gemini] LLM 响应已生成。")
            if not result:
                print("[Gemini] LLM 超时或失败")
                # result = infer_intent_rules(cmd_text, dets, ocr_results)
        # else:
            # result = infer_intent_rules(cmd_text, dets, ocr_results)

        if self.output_func:
            threading.Thread(target=self.output_func, args=(result,), daemon=True).start()
        else:
            print(f"[🧠 意图结果] {result}")

    # =============== LLM 版（带超时保护） ===============
    def _infer_intent_llm_timeout(self, cmd, vision, ocr_results, timeout_s):
        result_holder = {"text": None}

        def _worker():
            try:
                result_holder["text"] = self._infer_intent_llm(cmd, vision, ocr_results)
            except Exception as e:
                print(f"[Gemini] 调用异常：{e}")
                result_holder["text"] = None

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_s)
        return result_holder["text"]

    def _infer_intent_llm(self, cmd, vision, ocr_results):
        objs_raw = vision
        max_x2 = max(o["bbox"][2] for o in objs_raw) if objs_raw else 1
        w_third = max_x2 / 3.0 if max_x2 > 0 else 1.0

        def _pos(o):
            x1, y1, x2, y2 = o["bbox"]
            cx = (x1 + x2) / 2
            if cx < w_third:
                return "left"
            elif cx < 2 * w_third:
                return "front"
            else:
                return "right"

        objs = []
        for o in objs_raw:
            x1, y1, x2, y2 = o["bbox"]
            area = max(1, (x2 - x1) * (y2 - y1))
            objs.append({
                "name": o.get("class", ""),
                "confidence": round(float(o.get("confidence", 0.0)), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "pos": _pos(o),
                "area": int(area)
            })

        objs = sorted(objs, key=lambda x: x["area"], reverse=True)
        agg = {}
        for o in objs:
            agg[o["name"]] = agg.get(o["name"], 0) + 1

        # 构建OCR结果
        ocr_texts = [{"text": item["text"], "confidence": item["confidence"]}
                     for item in ocr_results] if ocr_results else []

        user_payload = {
            "command": cmd,
            "objects": objs[:40],
            "ocr_texts": ocr_texts[:10],  # 最多10个OCR结果
            "summary": {
                "unique_classes": [{"name": k, "count": v} for k, v in
                                   sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))][:30],
                "has_text": len(ocr_texts) > 0
            }
        }
        user_prompt = "Infer and reply according to the JSON below：\n" + json.dumps(user_payload, ensure_ascii=False)

        # 这里也可加 request_options={"timeout": self.llm_timeout}
        resp = self.gemini_model.generate_content(user_prompt)
        text = (getattr(resp, "text", "") or "").strip() if resp else ""
        return text or None