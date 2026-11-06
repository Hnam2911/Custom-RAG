import json
import re
import time
import random
import requests
from typing import List, Dict, Any
from config import RAGConfig
from schema import (
    GenerationResult,
    EvalSample,
    ClaimVerification,
    GenerationEvalReport
)
from generation.prompt_builder import PromptBuilder
from generation.gemini_llm import SharedRateLimiter

class LLMGenerationJudge:
    """
    Bộ chấm điểm LLM-as-a-Judge tối ưu cho giới hạn RPM thấp:
    - Hỗ trợ chấm theo lô (Batch Evaluation): Gộp N câu trả lời vào 1 HTTP Request duy nhất.
    - Tích hợp SharedRateLimiter và tự động chuyển đổi sang fallback_llm_model khi gặp lỗi 503.
    """
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: RAGConfig):
        if not config.gemini_api_key or config.gemini_api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Vui lòng cung cấp GEMINI_API_KEY hợp lệ để chạy LLM Judge.")
        self.config = config
        self.api_key = config.gemini_api_key
        self.model_name = (
            config.llm_model
            if config.llm_model.startswith("models/")
            else f"models/{config.llm_model}"
        )
        fallback = getattr(config, "fallback_llm_model", "")
        self.fallback_model_name = (
            (fallback if fallback.startswith("models/") else f"models/{fallback}")
            if fallback else ""
        )
        self.max_rpm = getattr(config, "max_rpm", 10)
        self.batch_size = getattr(config, "judge_batch_size", 5)
        self.session = requests.Session()

    def _call_judge_api(self, system_prompt: str, user_prompt: str, max_retries: int = 4) -> dict:
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        active_model = self.model_name

        for attempt in range(max_retries):
            SharedRateLimiter.wait_for_slot(self.max_rpm)
            url = f"{self.BASE_URL}/{active_model}:generateContent"

            try:
                resp = self.session.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=90
                )
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    if resp.status_code == 503 and self.fallback_model_name and active_model != self.fallback_model_name:
                        print(f"[Judge 503] {active_model} quá tải, chuyển Judge sang {self.fallback_model_name}...")
                        active_model = self.fallback_model_name
                        time.sleep(2.0)
                        continue

                    base_wait = 15.0 * (attempt + 1) if resp.status_code == 429 else 8.0 * (attempt + 1)
                    wait_time = base_wait + random.uniform(0.5, 2.5)
                    print(f"[Judge HTTP {resp.status_code}] Tạm nghỉ {wait_time:.1f}s trước khi thử lại...")
                    time.sleep(wait_time)
                    SharedRateLimiter._last_request_time = time.time()
                    continue

                raise RuntimeError(f"Judge API lỗi ({resp.status_code}): {resp.text}")
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"Lỗi kết nối khi gọi Judge API: {e}") from e

        raise RuntimeError("Vượt quá số lần thử lại khi gọi Judge API.")

    def evaluate_single(self, sample: EvalSample, gen_result: GenerationResult) -> GenerationEvalReport:
        """Chấm 1 mẫu đơn lẻ (thông qua cơ chế batch 1 phần tử)."""
        return self._evaluate_batch([sample], [gen_result])[0]

    def evaluate_dataset(
        self,
        samples: List[EvalSample],
        gen_results: List[GenerationResult]
    ) -> Dict[str, Any]:
        """
        Chấm điểm toàn bộ tập kiểm thử theo lô (Batch).
        Với 5 câu hỏi mẫu, toàn bộ bước chấm điểm chỉ tốn DUY NHẤT 1 Request API.
        """
        if len(samples) != len(gen_results):
            raise ValueError("Số lượng mẫu kiểm thử và số lượng câu trả lời không khớp.")

        all_reports: List[GenerationEvalReport] = []
        for start_idx in range(0, len(samples), self.batch_size):
            batch_samples = samples[start_idx:start_idx + self.batch_size]
            batch_gens = gen_results[start_idx:start_idx + self.batch_size]
            all_reports.extend(self._evaluate_batch(batch_samples, batch_gens))

        n = max(len(all_reports), 1)
        return {
            "Mean_Faithfulness": round(sum(r.faithfulness for r in all_reports) / n, 4),
            "Mean_Answer_Relevance": round(sum(r.answer_relevance for r in all_reports) / n, 4),
            "Mean_Citation_Precision": round(sum(r.citation_precision for r in all_reports) / n, 4),
            "details": all_reports
        }

    def _evaluate_batch(
        self,
        samples: List[EvalSample],
        gen_results: List[GenerationResult]
    ) -> List[GenerationEvalReport]:
        reports_map: Dict[str, GenerationEvalReport] = {}
        pending_items: List[tuple[EvalSample, GenerationResult]] = []

        # Bước 1: Xử lý ngay lập tức các trường hợp Refusal bằng logic Python (Tốn 0 API call)
        for sample, gen_res in zip(samples, gen_results):
            if sample.expected_refusal:
                if gen_res.is_refusal:
                    reports_map[sample.sample_id] = GenerationEvalReport(
                        sample_id=sample.sample_id,
                        faithfulness=1.0,
                        answer_relevance=1.0,
                        citation_precision=1.0,
                        claims=[],
                        judge_reasoning="Hệ thống đã từ chối trả lời chính xác câu hỏi ngoài tài liệu."
                    )
                else:
                    reports_map[sample.sample_id] = GenerationEvalReport(
                        sample_id=sample.sample_id,
                        faithfulness=0.0,
                        answer_relevance=0.0,
                        citation_precision=0.0,
                        claims=[ClaimVerification(claim=gen_res.answer, supported=False)],
                        judge_reasoning="Lỗi Hallucination: Câu hỏi ngoài tài liệu nhưng mô hình vẫn trả lời."
                    )
            elif gen_res.is_refusal:
                reports_map[sample.sample_id] = GenerationEvalReport(
                    sample_id=sample.sample_id,
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    citation_precision=0.0,
                    claims=[],
                    judge_reasoning="Hệ thống từ chối trả lời sai (False Refusal) dù tài liệu có thông tin."
                )
            else:
                pending_items.append((sample, gen_res))

        if not pending_items:
            return [reports_map[s.sample_id] for s in samples]

        # Bước 2: Gom tất cả các mẫu cần chấm vào 1 Prompt duy nhất
        system_prompt = (
            "Bạn là Giám khảo Đánh giá Hệ thống RAG (RAG Evaluation Judge) khách quan và khắt khe.\n"
            "Bạn sẽ nhận danh sách nhiều mục cần chấm (<eval_item>). Với MỖI mục, hãy phân tích dựa trên "
            "<retrieved_context> của riêng mục đó và trả về kết quả trong mảng JSON 'evaluations':\n"
            "1. 'sample_id': Giữ nguyên mã sample_id của mục đó.\n"
            "2. 'claims': Tách câu trả lời thành các mệnh đề nguyên tử (atomic claims). Đánh dấu 'supported': true/false "
            "tùy thuộc mệnh đề có được chứng minh trực tiếp bởi <retrieved_context> hay không, kèm 'evidence_id'.\n"
            "3. 'answer_relevance': Điểm [0.0 -> 1.0] đánh giá mức độ giải quyết đúng câu hỏi.\n"
            "4. 'citation_precision': Điểm [0.0 -> 1.0] đánh giá độ chính xác của mã trích dẫn [chunk_id].\n"
            "5. 'reasoning': Giải thích ngắn gọn bằng tiếng Việt."
        )

        item_blocks: List[str] = []
        for sample, gen_res in pending_items:
            ctx_strs = [
                f"<chunk id=\"{r.chunk.chunk_id}\">\n{PromptBuilder.format_chunk_with_heading(r.chunk)}\n</chunk>"
                for r in gen_res.used_contexts
            ]
            ref_line = f"Đáp án tham chiếu: {sample.reference_answer}\n" if sample.reference_answer else ""
            item_blocks.append(
                f"<eval_item sample_id=\"{sample.sample_id}\">\n"
                f"<retrieved_context>\n{chr(10).join(ctx_strs)}\n</retrieved_context>\n"
                f"Câu hỏi: {sample.query}\n"
                f"{ref_line}"
                f"Câu trả lời cần chấm:\n{gen_res.answer}\n"
                f"</eval_item>"
            )

        user_prompt = (
            "\n\n".join(item_blocks) + "\n\n"
            "Hãy trả về duy nhất 1 JSON theo cấu trúc:\n"
            "{\n"
            "  \"evaluations\": [\n"
            "    {\n"
            "      \"sample_id\": \"...\",\n"
            "      \"claims\": [{\"claim\": \"...\", \"supported\": true, \"evidence_id\": \"...\"}],\n"
            "      \"answer_relevance\": 0.95,\n"
            "      \"citation_precision\": 1.0,\n"
            "      \"reasoning\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        raw_response = self._call_judge_api(system_prompt, user_prompt)
        parsed = self._extract_json_safely(raw_response)
        eval_list = parsed.get("evaluations", [])
        eval_by_id = {str(item.get("sample_id", "")): item for item in eval_list if isinstance(item, dict)}

        for sample, gen_res in pending_items:
            item_data = eval_by_id.get(sample.sample_id, {})
            raw_claims = item_data.get("claims", [])
            claim_objs = [
                ClaimVerification(
                    claim=str(c.get("claim", "")),
                    supported=bool(c.get("supported", False)),
                    evidence_id=str(c.get("evidence_id", ""))
                )
                for c in raw_claims if isinstance(c, dict) and "claim" in c
            ]

            if claim_objs:
                faithfulness_score = sum(1 for c in claim_objs if c.supported) / float(len(claim_objs))
            else:
                faithfulness_score = float(item_data.get("faithfulness", 0.0))

            relevance_score = max(0.0, min(1.0, float(item_data.get("answer_relevance", 0.0))))
            citation_score = max(0.0, min(1.0, float(item_data.get("citation_precision", 0.0))))
            if not gen_res.citations:
                citation_score = 0.0

            reports_map[sample.sample_id] = GenerationEvalReport(
                sample_id=sample.sample_id,
                faithfulness=round(faithfulness_score, 4),
                answer_relevance=round(relevance_score, 4),
                citation_precision=round(citation_score, 4),
                claims=claim_objs,
                judge_reasoning=str(item_data.get("reasoning", ""))
            )

        return [reports_map[s.sample_id] for s in samples]

    @staticmethod
    def _extract_json_safely(api_data: dict) -> dict:
        candidates = api_data.get("candidates", [])
        if not candidates:
            return {}
        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}