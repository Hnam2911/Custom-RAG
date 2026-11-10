# retrieval/reranker.py
import re
import json
import math
import requests
from typing import List, Dict
from config import RAGConfig
from schema import SearchResult
from indexing.bm25_index import BM25Index
from generation.gemini_llm import SharedRateLimiter

try:
    from sentence_transformers import CrossEncoder#type:ignore
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class CrossEncoderReranker:
    """
    Bộ chấm điểm chéo (Cross-Encoder Reranker) cho tập ứng viên hợp nhất (M_local + M_web).
    Đưa toàn bộ ứng viên từ Nhánh A (Local) và Nhánh B (Web) về chung thang điểm [0.0 -> 1.0].
    """
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: RAGConfig):
        self.config = config
        self.backend = getattr(config, "reranker_backend", "hybrid_calibrated")
        self._local_model = None
        self._tokenizer_helper = BM25Index(k1=config.bm25_k1, b=config.bm25_b)
        self.session = requests.Session()

        if self.backend == "local_cross_encoder" and HAS_SENTENCE_TRANSFORMERS:
            try:
                self._local_model = CrossEncoder(config.cross_encoder_model)#type:ignore
            except Exception as e:
                print(f"[Reranker Cảnh báo] Không tải được CrossEncoder '{config.cross_encoder_model}' ({e}), dùng hybrid_calibrated.")
                self.backend = "hybrid_calibrated"

    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        if not candidates or not query.strip():
            return []

        if self.backend == "local_cross_encoder" and self._local_model is not None:
            scored = self._rerank_with_local_cross_encoder(query, candidates)
        elif self.backend == "gemini_listwise":
            try:
                scored = self._rerank_with_gemini_listwise(query, candidates)
            except Exception as e:
                print(f"[Reranker Fallback] Gemini Listwise lỗi ({e}), chuyển sang hybrid_calibrated.")
                scored = self._rerank_with_hybrid_calibrated(query, candidates)
        else:
            scored = self._rerank_with_hybrid_calibrated(query, candidates)

        # Lọc theo ngưỡng điểm rerank_score_threshold và sắp xếp giảm dần
        threshold = getattr(self.config, "rerank_score_threshold", 0.25)
        filtered = [
            r for r in scored
            if (r.rerank_score is not None and r.rerank_score >= threshold)
        ]

        filtered.sort(
            key=lambda r: (
                round(r.rerank_score or 0.0, 6),
                r.dense_score if r.dense_score is not None else -1.0,
                r.score
            ),
            reverse=True
        )

        # Đồng bộ r.score = r.rerank_score để RetrievalEvaluator ở tầng Evaluation
        # sắp xếp đúng thứ tự xếp hạng sau Rerank.
        for r in filtered:
            if r.rerank_score is not None:
                r.score = r.rerank_score

        return filtered

    def _rerank_with_local_cross_encoder(
        self,
        query: str,
        candidates: List[SearchResult]
    ) -> List[SearchResult]:
        pairs = [(query, c.chunk.get_contextual_text()) for c in candidates]
        raw_logits = self._local_model.predict(pairs)#type:ignore

        for cand, logit in zip(candidates, raw_logits):
            val = float(logit)
            # Chuẩn hóa Sigmoid đưa logit thực về khoảng [0.0, 1.0]
            prob = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
            cand.rerank_score = round(prob, 6)
        return candidates

    def _rerank_with_hybrid_calibrated(
        self,
        query: str,
        candidates: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Chấm điểm chéo (Query-Chunk Interaction) trên CPU không phụ thuộc thư viện ngoài:
        Kết hợp đồng thời:
        - 55% Điểm tương đồng ngữ nghĩa Cosine (Dense Bi-Encoder) trên cùng không gian Gemini Embedding 2.
        - 35% Độ phủ tập token & bigram của câu hỏi bên trong văn bản chunk (Exact Lexical Coverage).
        - 10% Tín hiệu RRF chuẩn hóa.
        """
        q_tokens = set(self._tokenizer_helper.tokenize(query))
        max_rrf = (self.config.weight_dense + self.config.weight_bm25) / (self.config.rrf_k + 1.0)

        for cand in candidates:
            doc_text = cand.chunk.get_contextual_text()
            doc_tokens = set(self._tokenizer_helper.tokenize(doc_text))

            # 1. Điểm độ phủ từ khóa & cụm từ ghép (Bigram) trực tiếp giữa q và chunk
            if q_tokens:
                overlap_ratio = len(q_tokens.intersection(doc_tokens)) / float(len(q_tokens))
            else:
                overlap_ratio = 0.0

            # Thưởng thêm nếu cụm câu hỏi xuất hiện nguyên vẹn trong chunk
            exact_phrase_bonus = 0.10 if query.strip().lower() in doc_text.lower() else 0.0
            lexical_score = min(1.0, overlap_ratio + exact_phrase_bonus)

            # 2. Điểm ngữ nghĩa Cosine (đã nằm trên cùng thang [-1, 1] của GeminiEmbedder)
            semantic_score = max(0.0, float(cand.dense_score)) if cand.dense_score is not None else lexical_score

            # 3. Điểm RRF chuẩn hóa về [0, 1]
            norm_rrf = min(1.0, cand.score / max(max_rrf, 1e-6))

            combined = (0.55 * semantic_score) + (0.35 * lexical_score) + (0.10 * norm_rrf)
            cand.rerank_score = round(combined, 6)

        return candidates

    def _rerank_with_gemini_listwise(
        self,
        query: str,
        candidates: List[SearchResult]
    ) -> List[SearchResult]:
        """Chấm điểm toàn bộ ứng viên trong 1 HTTP Request duy nhất bằng Gemini JSON mode."""
        model_name = (
            self.config.llm_model
            if self.config.llm_model.startswith("models/")
            else f"models/{self.config.llm_model}"
        )
        SharedRateLimiter.wait_for_slot(self.config.max_rpm)
        url = f"{self.BASE_URL}/{model_name}:generateContent"

        passages_str = "\n\n".join(
            f"<candidate id=\"{c.chunk.chunk_id}\">\n{c.chunk.get_contextual_text()[:1200]}\n</candidate>"
            for c in candidates
        )
        system_prompt = (
            "Bạn là mô hình Cross-Encoder Reranker chính xác.\n"
            "Hãy chấm điểm mức độ liên quan trực tiếp của từng <candidate> đối với câu hỏi trên thang [0.0 -> 1.0].\n"
            "Trả về JSON theo cấu trúc: {\"scores\": [{\"id\": \"...\", \"score\": 0.85}]}"
        )
        user_prompt = f"Câu hỏi: {query}\n\nDanh sách ứng viên:\n{passages_str}"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
        }
        resp = self.session.post(
            url,
            params={"key": self.config.gemini_api_key},
            json=payload,
            timeout=45
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")

        raw_text = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE).strip()
        parsed = json.loads(cleaned)
        score_map: Dict[str, float] = {
            str(item.get("id", "")): float(item.get("score", 0.0))
            for item in parsed.get("scores", [])
            if isinstance(item, dict)
        }

        # Fallback về hybrid_calibrated cho bất kỳ ứng viên nào bị thiếu trong JSON
        self._rerank_with_hybrid_calibrated(query, candidates)
        for cand in candidates:
            if cand.chunk.chunk_id in score_map:
                cand.rerank_score = max(0.0, min(1.0, round(score_map[cand.chunk.chunk_id], 6)))

        return candidates