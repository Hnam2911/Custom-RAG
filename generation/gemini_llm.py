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
    """
    Bộ sinh câu trả lời chuẩn OpenAI-Compatible REST API (Groq, OpenRouter, DeepSeek, OpenAI, vLLM...)
    Giữ nguyên tên class GeminiLLMGenerator để tương thích 100% với toàn bộ codebase.
    """
    def __init__(
        self,
        config: RAGConfig,
        prompt_builder: Optional[PromptBuilder] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048
    ):
        # Ưu tiên lấy llm_api_key mới, nếu chưa có thì fallback về gemini_api_key
        self.api_key = getattr(config, "llm_api_key", "") or getattr(config, "gemini_api_key", "")
        if not self.api_key or "YOUR_" in self.api_key:
            raise ValueError("Vui lòng cung cấp API Key hợp lệ trong RAGConfig.")

        self.config = config
        self.base_url = getattr(config, "llm_base_url", "https://api.groq.com/openai/v1").rstrip("/")
        self.model_name = config.llm_model
        self.fallback_model_name = getattr(config, "fallback_llm_model", "")
        self.max_rpm = getattr(config, "max_rpm", 20)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.session = requests.Session()

    def _post_with_retry(self, payload: dict, max_retries: int = 4) -> dict:
        active_model = self.model_name
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(max_retries):
            SharedRateLimiter.wait_for_slot(self.max_rpm)
            payload["model"] = active_model

            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    if response.status_code in (429, 503) and self.fallback_model_name and active_model != self.fallback_model_name:
                        print(f"\n[Cảnh báo {response.status_code}] Chuyển sang model dự phòng: {self.fallback_model_name}...")
                        active_model = self.fallback_model_name
                        time.sleep(2.0)
                        continue

                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after) + 1.5
                        except ValueError:
                            wait_time = 15.0 * (attempt + 1)
                    else:
                        m = re.search(r"try again in ([0-9.]+)s", response.text, re.IGNORECASE)
                        wait_time = (float(m.group(1)) + 2.0) if m else (12.0 * (attempt + 1) + random.uniform(0.5, 2.0))

                    print(f"\n[HTTP {response.status_code}] Đang tạm nghỉ {wait_time:.1f}s trước khi thử lại (lần {attempt + 1}/{max_retries - 1})...")
                    time.sleep(wait_time)
                    SharedRateLimiter._last_request_time = time.time()
                    continue

                raise RuntimeError(f"LLM API lỗi ({response.status_code}): {response.text}")
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(5.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"Lỗi kết nối mạng khi gọi LLM API: {e}") from e

        raise RuntimeError("Đã vượt quá số lần thử lại khi gọi LLM API.")

    def generate(self, query: str, search_results: List[SearchResult]) -> GenerationResult:
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

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens
        }

        data = self._post_with_retry(payload)
        answer_text, finish_reason = self._parse_response_safely(data)

        is_refusal = self.prompt_builder.REFUSAL_PHRASE.lower() in answer_text.lower()
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
        choices = data.get("choices", [])
        if not choices:
            return "[Phản hồi rỗng từ API]", "EMPTY_CHOICES"

        first_choice = choices[0]
        finish_reason = str(first_choice.get("finish_reason", "STOP")).upper()
        message = first_choice.get("message", {})
        full_text = (message.get("content") or "").strip()

        # Tự động lọc bỏ thẻ <think>...</think> nếu dùng model reasoning
        full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()

        if not full_text:
            return f"[Mô hình không sinh được văn bản. Lý do kết thúc: {finish_reason}]", finish_reason

        return full_text, finish_reason

    @staticmethod
    def _extract_and_verify_citations(
        answer_text: str,
        search_results: List[SearchResult]
    ) -> tuple[List[Citation], str]:
        valid_map: Dict[str, SearchResult] = {r.chunk.chunk_id: r for r in search_results}

        sub_id_to_merged_id: Dict[str, str] = {}
        for r in search_results:
            merged_from = (r.chunk.metadata or {}).get("merged_from", [])
            for sub_id in merged_from:
                sub_id_to_merged_id[sub_id] = r.chunk.chunk_id

        cited_ids_ordered: List[str] = []
        bracket_pattern = re.compile(r"\[([^\[\]\n]{1,60})\]")

        def validate_bracket(match: re.Match) -> str:
            inner = match.group(1).strip()
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

            if kept_ids:
                return "".join(kept_ids)
            if re.match(r"^[a-zA-Z0-9_+.-]+_c\d+", inner):
                return ""
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