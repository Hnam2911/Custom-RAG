import time
import requests
import numpy as np
from typing import List, Optional
from schema import Chunk

class GeminiEmbedder:
    """
    Bộ tạo Vector Embedding gọi trực tiếp REST API của Google Gemini.
    Tương thích hoàn toàn với kiến trúc mới của 'models/gemini-embedding-2'
    (sử dụng In-prompt Task Prefix, Matryoshka outputDimensionality, và batchEmbedContents)
    cũng như tương thích ngược với 'models/gemini-embedding-001'.
    """
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MAX_BATCH_SIZE = 100  # Giới hạn số lượng request tối đa trong 1 batchEmbedContents

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-embedding-2",
        output_dimensionality: int = 768,
        task_prefix: str = "search result"
    ):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Vui lòng cung cấp GEMINI_API_KEY hợp lệ.")
        if not (128 <= output_dimensionality <= 3072):
            raise ValueError("output_dimensionality phải nằm trong khoảng từ 128 đến 3072.")

        self.api_key = api_key
        self.model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
        self.output_dimensionality = output_dimensionality
        self.task_prefix = task_prefix  # "search result" hoặc "question answering"
        self.is_v2_model = "gemini-embedding-2" in self.model_name
        self.session = requests.Session()
        self._query_cache: dict[str, np.ndarray] = {}

    def _post_with_retry(self, url: str, payload: dict, max_retries: int = 4) -> dict:
        """
        Gửi HTTP POST sử dụng header 'x-goog-api-key' chuẩn của Google,
        kèm Exponential Backoff + Jitter để vượt qua lỗi 429 (Rate Limit: 100 RPM / 30K TPM).
        """
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=45
                )
                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    # Với quota 100 RPM / 30K TPM trên Free/Tier 1, nghỉ tăng dần 2s, 4s, 8s
                    wait_time = (2 ** (attempt + 1)) + 0.5
                    time.sleep(wait_time)
                    continue

                raise RuntimeError(f"Gemini Embedding API lỗi ({response.status_code}): {response.text}")
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"Lỗi kết nối mạng khi gọi Gemini Embedding API: {e}") from e
        raise RuntimeError("Đã vượt quá số lần thử lại (max_retries) khi gọi Gemini API.")

    def _format_document_for_v2(self, chunk: Chunk) -> str:
        """
        Định dạng văn bản tài liệu chuẩn cho Gemini Embedding 2 (Asymmetric Retrieval):
        Cấu trúc bắt buộc: 'title: {title} | text: {content}'
        Nếu không có parent_heading thì dùng 'title: none'.
        Đồng thời tuân thủ quy ước không làm thay đổi chunk.content gốc.
        """
        parent_heading = (chunk.metadata or {}).get("parent_heading", "").strip()
        title_str = parent_heading if parent_heading else "none"
        body_text = chunk.content.strip() if chunk.content and chunk.content.strip() else "[Empty Chunk]"
        return f"title: {title_str} | text: {body_text}"

    def _format_query_for_v2(self, query: str) -> str:
        """
        Định dạng câu truy vấn chuẩn cho Gemini Embedding 2 (Asymmetric Retrieval):
        Cấu trúc bắt buộc: 'task: search result | query: {content}'
        hoặc 'task: question answering | query: {content}'
        """
        return f"task: {self.task_prefix} | query: {query.strip()}"

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Nhúng danh sách Chunk qua endpoint ':batchEmbedContents'.
        Mỗi Chunk được đóng gói trong một EmbedContentRequest độc lập để tránh bị
        Gemini Embedding 2 gộp chung (Embedding Aggregation).
        """
        if not chunks:
            return np.empty((0, self.output_dimensionality), dtype=np.float32)

        url = f"{self.BASE_URL}/{self.model_name}:batchEmbedContents"
        all_vectors: List[List[float]] = []

        for start_idx in range(0, len(chunks), self.MAX_BATCH_SIZE):
            batch_chunks = chunks[start_idx:start_idx + self.MAX_BATCH_SIZE]
            requests_payload = []

            for chunk in batch_chunks:
                if self.is_v2_model:
                    # Gemini Embedding 2: Đưa task & title vào thẳng prompt theo chuẩn mới, KHÔNG dùng trường taskType
                    formatted_text = self._format_document_for_v2(chunk)
                    req_item = {
                        "model": self.model_name,
                        "content": {"parts": [{"text": formatted_text}]},
                        "outputDimensionality": self.output_dimensionality
                    }
                else:
                    # Tương thích ngược cho gemini-embedding-001 (Gemini Embedding 1)
                    parent_heading = (chunk.metadata or {}).get("parent_heading", "").strip()
                    contextual_text = chunk.get_contextual_text() if hasattr(chunk, "get_contextual_text") else chunk.content
                    req_item = {
                        "model": self.model_name,
                        "content": {"parts": [{"text": contextual_text or "[Empty Chunk]"}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "outputDimensionality": self.output_dimensionality
                    }
                    if parent_heading:
                        req_item["title"] = parent_heading[:200]

                requests_payload.append(req_item)

            data = self._post_with_retry(url, {"requests": requests_payload})
            embeddings_list = data.get("embeddings", [])

            if len(embeddings_list) != len(batch_chunks):
                raise RuntimeError(
                    f"Số vector trả về ({len(embeddings_list)}) không khớp số chunk gửi đi ({len(batch_chunks)})."
                )

            for chunk, emb_obj in zip(batch_chunks, embeddings_list):
                vec = emb_obj.get("values", [])
                chunk.embedding = np.asarray(vec, dtype=np.float32)
                all_vectors.append(vec)

        return np.asarray(all_vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Tạo vector cho câu hỏi của người dùng với đúng tiền tố 'task: ... | query: ...'."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Câu truy vấn (query) không được để trống.")
        
        if cleaned_query in self._query_cache:
            return self._query_cache[cleaned_query]

        url = f"{self.BASE_URL}/{self.model_name}:embedContent"

        if self.is_v2_model:
            formatted_query = self._format_query_for_v2(cleaned_query)
            payload = {
                "model": self.model_name,
                "content": {"parts": [{"text": formatted_query}]},
                "outputDimensionality": self.output_dimensionality
            }
        else:
            payload = {
                "model": self.model_name,
                "content": {"parts": [{"text": cleaned_query}]},
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": self.output_dimensionality
            }

        data = self._post_with_retry(url, payload)
        vec = data.get("embedding", {}).get("values", [])
        if not vec:
            raise RuntimeError("API không trả về vector cho câu truy vấn.")
        vec_array = np.asarray(vec, dtype=np.float32)
        self._query_cache[cleaned_query] = vec_array
        return vec_array