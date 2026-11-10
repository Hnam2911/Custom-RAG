# retrieval/query_rewriter.py
import re
import requests
from config import RAGConfig
from generation.gemini_llm import SharedRateLimiter

class QueryRewriter:
    """
    Bước 1 (Nhánh B): Chuyển câu hỏi dài (Query gốc q) thành từ khóa tìm kiếm Web (q_web).
    """
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    # Các cụm từ đệm nghi vấn phổ biến trong câu hỏi tự nhiên nhưng làm giảm độ chính xác khi tìm Google
    FILLER_PATTERNS = [
        r"\b(cho tôi hỏi|hãy giải thích|hãy cho biết|vui lòng cho biết|làm thế nào để|tại sao|như thế nào)\b",
        r"\b(là gì|có nghĩa là gì|chi tiết về|thông tin về|dựa trên tài liệu|theo tài liệu)\b",
        r"\b(can you explain|please tell me|what is the difference between|how does|what are the)\b"
    ]

    def __init__(self, config: RAGConfig):
        self.config = config
        self.session = requests.Session()

    def rewrite(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            return ""

        # Nếu câu truy vấn đã ngắn gọn (<= 8 từ), giữ nguyên để bảo toàn ngữ nghĩa
        if len(cleaned.split()) <= 8 and not self.config.use_llm_query_rewriter:
            return self._heuristic_rewrite(cleaned)

        if self.config.use_llm_query_rewriter:
            try:
                return self._llm_rewrite(cleaned)
            except Exception as e:
                print(f"[QueryRewriter Fallback] Chuyển về heuristic do lỗi API: {e}")
                return self._heuristic_rewrite(cleaned)

        return self._heuristic_rewrite(cleaned)

    def _heuristic_rewrite(self, query: str) -> str:
        """Trích xuất từ khóa tìm kiếm q_web tức thì (0ms, 0 API call), bảo toàn mã kỹ thuật và cụm ngoặc kép."""
        # Bảo toàn các cụm từ nằm trong dấu ngoặc kép "..."
        quoted_phrases = re.findall(r'"[^"]+"', query)

        working = query
        for pattern in self.FILLER_PATTERNS:
            working = re.sub(pattern, " ", working, flags=re.IGNORECASE)

        # Xóa dấu chấm hỏi, chấm than ở cuối câu nhưng giữ dấu chấm thập phân, gạch ngang mã kỹ thuật
        working = re.sub(r"[?!;]+", " ", working)
        working = re.sub(r"\s+", " ", working).strip()

        if not working:
            return query.strip()

        # Đảm bảo các cụm ngoặc kép quan trọng vẫn nằm trong truy vấn
        for qp in quoted_phrases:
            if qp not in working:
                working = f"{qp} {working}"

        # Giới hạn tối đa 18 từ khóa quan trọng nhất cho bộ máy tìm kiếm
        words = working.split()
        return " ".join(words[:18])

    def _llm_rewrite(self, query: str) -> str:
        model_name = (
            self.config.llm_model
            if self.config.llm_model.startswith("models/")
            else f"models/{self.config.llm_model}"
        )
        SharedRateLimiter.wait_for_slot(self.config.max_rpm)
        url = f"{self.BASE_URL}/{model_name}:generateContent"

        system_prompt = (
            "Bạn là chuyên gia tối ưu hóa truy vấn tìm kiếm Google (Search Query Optimizer).\n"
            "Nhiệm vụ: Chuyển câu hỏi dài của người dùng thành 1 chuỗi từ khóa tìm kiếm ngắn gọn (tối đa 12 từ), "
            "giữ nguyên tên riêng, thuật ngữ chuyên ngành, mã số kỹ thuật và năm.\n"
            "Chỉ trả về đúng chuỗi từ khóa tìm kiếm, không giải thích, không xuống dòng."
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 64}
        }
        resp = self.session.post(
            url,
            params={"key": self.config.gemini_api_key},
            json=payload,
            timeout=15
        )
        if resp.status_code == 200:
            candidates = resp.json().get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                rewritten = "".join(p.get("text", "") for p in parts).strip()
                if rewritten:
                    return rewritten.replace("\n", " ").strip()
        return self._heuristic_rewrite(query)