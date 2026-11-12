import re
import time
import random
import requests
from typing import List, Dict, Optional, Tuple
from config import RAGConfig
from schema import SearchResult, Citation, GenerationResult
from generation.prompt_builder import PromptBuilder


class SharedRateLimiter:
    """
    Bộ điều tiết nhịp độ gọi API đa kênh:
    Tách riêng bộ đếm thời gian cho từng provider ('gemini', 'external')
    nhưng vẫn giữ tương thích ngược với cách gọi wait_for_slot(max_rpm) cũ.
    """
    _last_request_time: float = 0.0
    _provider_last_time: Dict[str, float] = {}

    @classmethod
    def wait_for_slot(cls, max_rpm: int, provider: str = "default") -> None:
        if max_rpm <= 0:
            return
        min_interval = 60.0 / float(max_rpm)
        last_t = cls._provider_last_time.get(provider, cls._last_request_time)
        elapsed = time.time() - last_t
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        now = time.time()
        cls._provider_last_time[provider] = now
        cls._last_request_time = now


class GeminiLLMGenerator:
    """
    Bộ sinh câu trả lời Hybrid Multi-Provider:
    Hỗ trợ gọi luân phiên (Round-Robin) và tự động chuyển kênh dự phòng (Failover)
    giữa Google Gemini API và bất kỳ OpenAI-Compatible API nào (Groq, OpenRouter, DeepSeek...).
    """
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        config: RAGConfig,
        prompt_builder: Optional[PromptBuilder] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024
    ):
        self.config = config
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.session = requests.Session()

        # 1. Cấu hình kênh Gemini
        self.gemini_key = getattr(config, "gemini_api_key", "")
        self.has_gemini = bool(self.gemini_key and "YOUR_" not in self.gemini_key)
        self.gemini_model = self._normalize_gemini_name(config.gemini_model)
        fallback = getattr(config, "fallback_llm_model", "")
        self.gemini_fallback = self._normalize_gemini_name(fallback) if fallback else ""
        self.gemini_rpm = getattr(config, "gemini_max_rpm", getattr(config, "max_rpm", 9))

        # 2. Cấu hình kênh External (OpenAI-Compatible)
        self.ext_key = getattr(config, "external_api_key", "")
        self.has_external = bool(self.ext_key and "YOUR_" not in self.ext_key)
        self.ext_base_url = getattr(config, "external_base_url", "https://api.groq.com/openai/v1").rstrip("/")
        self.ext_model = getattr(config, "external_llm_model", "llama-3.3-70b-versatile")
        self.ext_rpm = getattr(config, "external_max_rpm", 10)

        if not self.has_gemini and not self.has_external:
            raise ValueError("Phải cung cấp ít nhất 1 API Key hợp lệ (gemini_api_key hoặc external_api_key).")

        self.routing_mode = getattr(config, "routing_mode", "round_robin")
        self._call_counter = 0

    @staticmethod
    def _normalize_gemini_name(name: str) -> str:
        return name if name.startswith("models/") else f"models/{name}"

    def _choose_provider_order(self) -> List[str]:
        """Xác định thứ tự ưu tiên gọi API cho câu hỏi hiện tại."""
        if self.has_gemini and not self.has_external:
            return ["gemini"]
        if self.has_external and not self.has_gemini:
            return ["external"]

        # Khi cả 2 API đều sẵn sàng
        if self.routing_mode == "gemini_first":
            return ["gemini", "external"]
        if self.routing_mode == "external_first":
            return ["external", "gemini"]

        # Chế độ mặc định: round_robin (luân phiên chẵn/lẻ)
        self._call_counter += 1
        if self._call_counter % 2 == 1:
            return ["gemini", "external"]
        return ["external", "gemini"]

    def _call_gemini_once(self, system_instruction: str, user_prompt: str) -> Tuple[str, str]:
        """Gọi Google Gemini REST API (có tắt thinkingBudget để tránh cắt chữ)."""
        SharedRateLimiter.wait_for_slot(self.gemini_rpm, provider="gemini")
        url = f"{self.GEMINI_BASE_URL}/{self.gemini_model}:generateContent"

        gen_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens
        }
        # Nếu là dòng gemini-2.5-flash, tắt suy nghĩ ngầm để trả lời đầy đủ, không bị hụt token
        if "2.5-flash" in self.gemini_model:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_config
        }

        resp = self.session.post(
            url,
            params={"key": self.gemini_key},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
            return f"[Phản hồi bị chặn: {block_reason}]", block_reason

        first = candidates[0]
        finish_reason = first.get("finishReason", "STOP")
        parts = first.get("content", {}).get("parts", [])
        full_text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
        return (full_text or f"[Rỗng: {finish_reason}]"), f"GEMINI_{finish_reason}"

    def _call_external_once(self, system_instruction: str, user_prompt: str) -> Tuple[str, str]:
        """Gọi API bên ngoài theo chuẩn OpenAI Chat Completions."""
        SharedRateLimiter.wait_for_slot(self.ext_rpm, provider="external")
        url = f"{self.ext_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ext_key}"
        }
        payload = {
            "model": self.ext_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens
        }

        resp = self.session.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"External HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "[Phản hồi rỗng từ External API]", "EXT_EMPTY"

        first = choices[0]
        finish_reason = str(first.get("finish_reason", "STOP")).upper()
        full_text = (first.get("message", {}).get("content") or "").strip()
        full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        return (full_text or f"[Rỗng: {finish_reason}]"), f"EXT_{finish_reason}"

    def _generate_with_failover(self, system_instruction: str, user_prompt: str, max_rounds: int = 3) -> Tuple[str, str]:
        """
        Thử gọi theo thứ tự ưu tiên (ví dụ: Gemini -> External).
        Nếu kênh 1 bị 429/503, lập tức nhảy sang kênh 2 mà không phải chờ!
        Chỉ khi CẢ 2 kênh cùng bị 429 mới tạm nghỉ rồi thử lại.
        """
        provider_order = self._choose_provider_order()
        last_error = None

        for round_idx in range(max_rounds):
            for provider in provider_order:
                try:
                    if provider == "gemini":
                        return self._call_gemini_once(system_instruction, user_prompt)
                    else:
                        return self._call_external_once(system_instruction, user_prompt)
                except Exception as e:
                    last_error = e
                    print(f"\n[Chuyển kênh tự động] Kênh '{provider}' gặp lỗi ({e}) -> thử kênh kế tiếp...")
                    continue

            # Nếu cả 2 kênh đều lỗi trong vòng này, nghỉ trước khi vào vòng thử tiếp theo
            wait_s = 20.0 * (round_idx + 1)
            print(f"[*] Cả 2 kênh đều đang bận/chạm trần RPM. Tạm nghỉ {wait_s:.1f}s...")
            time.sleep(wait_s)

        raise RuntimeError(f"Tất cả các kênh API đều thất bại. Lỗi cuối: {last_error}")

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
        answer_text, finish_reason = self._generate_with_failover(system_instruction, user_prompt)

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