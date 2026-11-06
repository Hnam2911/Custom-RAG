from typing import List, Dict, Optional
from config import RAGConfig
from schema import Chunk, SearchResult
from indexing.gemini_embedder import GeminiEmbedder
from indexing.dense_flat_index import DenseFlatIndex
from indexing.bm25_index import BM25Index

class HybridRetriever:
    """
    Bộ truy xuất lai (Hybrid Search) kết hợp Dense Vector Search (Cosine) và Sparse Search (BM25).
    Sử dụng thuật toán Reciprocal Rank Fusion (RRF) có kiểm soát ngưỡng tín hiệu gốc.
    """
    def __init__(
        self,
        embedder: GeminiEmbedder,
        dense_index: DenseFlatIndex,
        bm25_index: BM25Index,
        config: RAGConfig
    ):
        self.embedder = embedder
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.config = config

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Thực hiện truy vấn song song trên Dense Index và BM25 Index,
        sau đó hợp nhất bằng công thức RRF:
        RRF_Score(d) = w_dense / (k + rank_dense(d)) + w_bm25 / (k + rank_bm25(d))
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        k_candidates = top_k if top_k is not None else self.config.top_k_retrieval

        # 1. Truy xuất nhánh BM25 (Sparse) - Bản thân BM25Index chỉ trả về các chunk có score > 0.0
        bm25_hits = self.bm25_index.search(cleaned_query, top_k=k_candidates)

        # 2. Truy xuất nhánh Vector (Dense)
        dense_hits = []
        if self.dense_index.size > 0 and self.config.weight_dense > 0:
            query_vec = self.embedder.embed_query(cleaned_query)
            raw_dense_hits = self.dense_index.search(query_vec, top_k=k_candidates)
            # Lọc bỏ các vector có Cosine thấp hơn ngưỡng tối thiểu để tránh gán Rank 1 cho câu hỏi rác
            re_rank = 1
            for chunk, cos_score, _ in raw_dense_hits:
                if cos_score >= self.config.min_dense_cosine:
                    dense_hits.append((chunk, cos_score, re_rank))
                    re_rank += 1

        # 3. Hợp nhất bằng Reciprocal Rank Fusion (RRF)
        fused_map: Dict[str, SearchResult] = {}
        rrf_k = self.config.rrf_k
        w_dense = self.config.weight_dense
        w_bm25 = self.config.weight_bm25

        for chunk, d_score, d_rank in dense_hits:
            cid = chunk.chunk_id
            rrf_contrib = w_dense / (rrf_k + d_rank)
            fused_map[cid] = SearchResult(
                chunk=chunk,
                score=rrf_contrib,
                dense_rank=d_rank,
                bm25_rank=None,
                dense_score=d_score,
                bm25_score=None
            )

        for chunk, b_score, b_rank in bm25_hits:
            cid = chunk.chunk_id
            rrf_contrib = w_bm25 / (rrf_k + b_rank)
            if cid in fused_map:
                existing = fused_map[cid]
                existing.score += rrf_contrib
                existing.bm25_rank = b_rank
                existing.bm25_score = b_score
            else:
                fused_map[cid] = SearchResult(
                    chunk=chunk,
                    score=rrf_contrib,
                    dense_rank=None,
                    bm25_rank=b_rank,
                    dense_score=None,
                    bm25_score=b_score
                )

        # 4. Sắp xếp kết quả với quy tắc phá vỡ đồng hạng (Tie-breaking) tường minh:
        # Ưu tiên: Điểm RRF -> Xuất hiện ở cả 2 nhánh -> Điểm Cosine gốc -> Điểm BM25 gốc
        results = list(fused_map.values())
        results.sort(
            key=lambda r: (
                round(r.score, 8),
                1 if (r.dense_rank is not None and r.bm25_rank is not None) else 0,
                r.dense_score if r.dense_score is not None else -1.0,
                r.bm25_score if r.bm25_score is not None else -1.0
            ),
            reverse=True
        )

        return results