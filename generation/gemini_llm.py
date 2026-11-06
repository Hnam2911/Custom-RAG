import re
import time
import random
import requests
from typing import List, Dict, Optional
from config import RAGConfig
from schema import SearchResult, Citation, GenerationResult
from generation.prompt_builder import PromptBuilder

class SharedRateLimiter:
    """
    Bộ điều tiết nhịp độ gọi API toàn cục (Singleton Pacer).
    Đảm bảo khoảng cách giữa 2 request liên tiếp (dù từ Generator hay Judge) luôn >= 60 / max_rpm.
    """
    _last_request_time: float = 0.0

    @classmethod
    def wait_for_slot(cls, max_rpm: int) -> None:
        if max_rpm <= 0:
            return
        min_interval = 60.0 / float(max_rpm)
        elapsed = time.time() - cls._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        cls._last_request_time = time.time()


class GeminiLLMGenerator:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        config: RAGConfig,
        prompt_builder: Optional[PromptBuilder] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048
    ):
        if not config.gemini_api_key or config.gemini_api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Vui lòng cung cấp GEMINI_API_KEY hợp lệ trong RAGConfig.")
        self.config = config
        self.api_key = config.gemini_api_key
        self.model_name = self._normalize_model_name(config.llm_model)
        self.fallback_model_name = (
            self._normalize_model_name(config.fallback_llm_model)
            if getattr(config, "fallback_llm_model", "")
            else ""
        )
        self.max_rpm = getattr(config, "max_rpm", 10)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.session = requests.Session()

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return name if name.startswith("models/") else f"models/{name}"

    def _post_with_retry(self, payload: dict, max_retries: int = 4) -> dict:
        """
        Gửi HTTP POST có điều tiết RPM, chờ thông minh theo mã lỗi (429 vs 503)
        và tự động chuyển sang fallback_llm_model nếu model chính bị quá tải (503).
        """
        active_model = self.model_name

        for attempt in range(max_retries):
            # 1. Chờ đúng nhịp RPM trước khi bắn request
            SharedRateLimiter.wait_for_slot(self.max_rpm)
            url = f"{self.BASE_URL}/{active_model}:generateContent"

            try:
                response = self.session.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()

                # Xử lý quá tải cục bộ (503) hoặc chạm trần RPM (429)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    # Nếu gặp 503 (Model Overloaded) và có model dự phòng -> chuyển sang model dự phòng ở lần thử kế tiếp
                    if response.status_code == 503 and self.fallback_model_name and active_model != self.fallback_model_name:
                        print(f"[Cảnh báo 503] {active_model} đang quá tải, tự động chuyển sang {self.fallback_model_name}...")
                        active_model = self.fallback_model_name
                        time.sleep(2.0)
                        continue

                    # Nếu gặp 429 (Hết quota phút) -> nghỉ dài (15s, 30s, 45s) để chờ reset cửa sổ 1 phút
                    base_wait = 15.0 * (attempt + 1) if response.status_code == 429 else 8.0 * (attempt + 1)
                    jitter = random.uniform(0.5, 2.5)
                    wait_time = base_wait + jitter
                    print(f"[HTTP {response.status_code}] Đang tạm nghỉ {wait_time:.1f}s trước khi thử lại (lần {attempt + 1}/{max_retries - 1})...")
                    time.sleep(wait_time)
                    SharedRateLimiter. _last_request_time = time.time()
                    continue

                raise RuntimeError(f"Gemini LLM API lỗi ({response.status_code}): {response.text}")
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"Lỗi kết nối mạng khi gọi Gemini LLM API: {e}") from e

        raise RuntimeError("Đã vượt quá số lần thử lại khi gọi Gemini LLM API.")

    def generate(self, query: str, search_results: List[SearchResult]) -> GenerationResult:
        """
        Sinh câu trả lời từ danh sách SearchResult đã lọc ở Tầng 4.
        Nếu search_results rỗng, tự động ngắt mạch (short-circuit) mà không tốn request API.
        """
        # Trường hợp biên 1: Không có đoạn ngữ cảnh nào vượt qua bộ lọc Tầng 4
        if not search_results:
            return GenerationResult(
                query=query,
                answer=self.prompt_builder.REFUSAL_PHRASE,
                citations=[],
                used_contexts=[],
                system_prompt="",
                user_prompt="",
                finish_reason="NO_CONTEXT_SHORT_CIRCUIT",
                is_refusal=True
            )

        system_instruction, user_prompt = self.prompt_builder.build(query, search_results)

        url = f"{self.BASE_URL}/{self.model_name}:generateContent"
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens
            }
        }

        data = self._post_with_retry(payload)
        answer_text, finish_reason = self._parse_response_safely(data)

        # Kiểm tra xem LLM có từ chối trả lời do thiếu thông tin hay không
        is_refusal = self.prompt_builder.REFUSAL_PHRASE.lower() in answer_text.lower()

        # Bóc tách, làm sạch và ánh xạ các mã trích dẫn [chunk_id] về tọa độ gốc
        verified_citations, cleaned_answer = self._extract_and_verify_citations(
            answer_text, search_results
        )

        return GenerationResult(
            query=query,
            answer=cleaned_answer,
            citations=verified_citations,
            used_contexts=search_results,
            system_prompt=system_instruction,
            user_prompt=user_prompt,
            finish_reason=finish_reason,
            is_refusal=is_refusal
        )

    @staticmethod
    def _parse_response_safely(data: dict) -> tuple[str, str]:
        """Đọc kết quả JSON từ Gemini API an toàn, chống lỗi KeyError khi bị chặn Safety/MaxTokens."""
        candidates = data.get("candidates", [])
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
            return f"[Phản hồi bị chặn bởi hệ thống: {block_reason}]", block_reason

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason", "STOP")
        parts = first_candidate.get("content", {}).get("parts", [])

        text_segments = [p.get("text", "") for p in parts if "text" in p]
        full_text = "".join(text_segments).strip()

        if not full_text:
            return f"[Mô hình không sinh được văn bản. Lý do kết thúc: {finish_reason}]", finish_reason

        return full_text, finish_reason

    @staticmethod
    def _extract_and_verify_citations(
        answer_text: str,
        search_results: List[SearchResult]
    ) -> tuple[List[Citation], str]:
        """
        1. Tìm tất cả các thẻ trích dẫn dạng [chunk_id] hoặc [chunk_id_1, chunk_id_2] trong câu trả lời.
        2. Đối chiếu với danh sách chunk thực tế đã đưa vào Prompt.
        3. Loại bỏ các mã trích dẫn bịa đặt (phantom citations) khỏi văn bản nếu có.
        4. Trả về danh sách đối tượng Citation kèm start_char_idx & end_char_idx để UI highlight.
        """
        valid_map: Dict[str, SearchResult] = {r.chunk.chunk_id: r for r in search_results}

        # Hỗ trợ trường hợp LLM trích dẫn mã chunk con trước khi gộp (nằm trong metadata["merged_from"])
        sub_id_to_merged_id: Dict[str, str] = {}
        for r in search_results:
            merged_from = (r.chunk.metadata or {}).get("merged_from", [])
            for sub_id in merged_from:
                sub_id_to_merged_id[sub_id] = r.chunk.chunk_id

        cited_ids_ordered: List[str] = []

        # Bắt các cụm trong ngoặc vuông [...] không chứa khoảng trắng dài (tránh bắt nhầm câu văn)
        bracket_pattern = re.compile(r"\[([^\[\]\n]{1,60})\]")

        def validate_bracket(match: re.Match) -> str:
            inner = match.group(1).strip()
            # Tách nếu LLM gộp nhiều mã trong cùng 1 ngoặc vuông: [01_c1, 01_c2]
            raw_tokens = [
                re.sub(r"^(?:Nguồn|ID|Chunk)\s*:?\s*", "", tok.strip(), flags=re.IGNORECASE)
                for tok in re.split(r"[,;]\s*", inner)
            ]

            kept_ids: List[str] = []
            for tok in raw_tokens:
                resolved_id = None
                if tok in valid_map:
                    resolved_id = tok
                elif tok in sub_id_to_merged_id:
                    resolved_id = sub_id_to_merged_id[tok]

                if resolved_id:
                    kept_ids.append(f"[{resolved_id}]")
                    if resolved_id not in cited_ids_ordered:
                        cited_ids_ordered.append(resolved_id)

            # Nếu cụm trong ngoặc vuông là mã hợp lệ -> chuẩn hóa thành [id1][id2]
            if kept_ids:
                return "".join(kept_ids)
            # Nếu trông giống mã chunk bịa đặt (VD: [01_c99]) -> xóa bỏ khỏi câu trả lời
            if re.match(r"^[a-zA-Z0-9_+.-]+_c\d+", inner):
                return ""
            # Giữ nguyên nếu là ký hiệu kỹ thuật thông thường trong văn bản
            return match.group(0)

        cleaned_answer = bracket_pattern.sub(validate_bracket, answer_text)

        citations: List[Citation] = []
        for cid in cited_ids_ordered:
            c = valid_map[cid].chunk
            citations.append(Citation(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                page_number=c.page_number,
                start_char_idx=c.start_char_idx,
                end_char_idx=c.end_char_idx,
                block_type=c.block_type,
                parent_heading=(c.metadata or {}).get("parent_heading", "")
            ))

        return citations, cleaned_answer