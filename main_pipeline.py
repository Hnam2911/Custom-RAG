# main_pipeline.py
from typing import List, Optional
from config import RAGConfig
from schema import Chunk, SearchResult, GenerationResult
from preprocess_file import process_file_to_chunks
from indexing.gemini_embedder import GeminiEmbedder
from indexing.dense_flat_index import DenseFlatIndex
from indexing.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.web_retriever import WebSearchRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.post_processor import RetrievalPostProcessor
from generation.gemini_llm import GeminiLLMGenerator


class HybridWebRAGPipeline:
    """
    Kiến trúc RAG tổng thể kết hợp:
    - NHÁNH A (Offline/Local): Tài liệu người dùng tự upload (PDF, MD, TXT, CSV, Excel).
    - NHÁNH B (Online/Web Search): Query Rewriter -> Macro Retriever -> Web Scraper -> Ephemeral Chunking -> Micro Retriever.
    - HỢP NHẤT & RERANK: Hợp nhất (M_local + M_web) -> Cross-Encoder Reranking -> Top-K -> Gemini LLM G_theta.
    """
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # Khởi tạo các thành phần lõi dùng chung
        self.embedder = GeminiEmbedder(
            api_key=self.config.gemini_api_key,
            model_name=self.config.embedding_model,
            output_dimensionality=self.config.embedding_dim,
            task_prefix=self.config.embedding_task
        )
        self.dense_index = DenseFlatIndex()
        self.bm25_index = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)

        # Khởi tạo Nhánh A (Local Hybrid Retriever)
        self.local_retriever = HybridRetriever(
            embedder=self.embedder,
            dense_index=self.dense_index,
            bm25_index=self.bm25_index,
            config=self.config
        )

        # Khởi tạo Nhánh B (Online Web Search Retriever)
        self.web_retriever = WebSearchRetriever(
            embedder=self.embedder,
            config=self.config
        )

        # Khởi tạo bộ Reranker, Post-Processor và LLM Generator
        self.reranker = CrossEncoderReranker(self.config)
        self.post_processor = RetrievalPostProcessor(self.config)
        self.generator = GeminiLLMGenerator(self.config)

    def ingest_local_file(self, file_path: str, doc_id: str) -> List[Chunk]:
        """Nhánh A (Offline): Đọc file, cắt chunk, tạo vector L2-Bi-Encoder và đưa vào BM25 + Dense Index."""
        chunks = process_file_to_chunks(file_path=file_path, doc_id=doc_id, config=self.config)
        if not chunks:
            return []

        embeddings = self.embedder.embed_chunks(chunks)
        self.dense_index.add_chunks(chunks, embeddings)
        self.bm25_index.add_chunks(chunks)
        return chunks

    def retrieve_combined(
        self,
        query: str,
        enable_web: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        Thực thi song hành Nhánh A (lấy M_local ứng viên) và Nhánh B (lấy M_web ứng viên),
        hợp nhất và chấm điểm chéo bằng Cross-Encoder Reranker để lấy Top-K Chunks tốt nhất.
        """
        use_web = self.config.enable_web_search if enable_web is None else enable_web

        # 1. Truy xuất Nhánh A (Local): Lấy M_local ứng viên
        m_local_candidates: List[SearchResult] = []
        if self.dense_index.size > 0 or self.bm25_index.size > 0:
            m_local_candidates = self.local_retriever.retrieve(
                query, top_k=self.config.top_k_retrieval
            )

        # 2. Truy xuất Nhánh B (Web Online): Lấy M_web ứng viên
        m_web_candidates: List[SearchResult] = []
        if use_web:
            m_web_candidates = self.web_retriever.retrieve(
                query, top_m_web=self.config.top_k_web_retrieval
            )

        # 3. Hợp nhất (M_local + M_web) -> Gộp liền kề -> Cross-Encoder Reranking -> Top-K & Lost-in-the-Middle
        return self.post_processor.merge_and_rerank(
            query=query,
            local_results=m_local_candidates,
            web_results=m_web_candidates,
            reranker=self.reranker,
            enable_merging=True,
            enable_lost_in_middle=True
        )

    def answer(self, query: str, enable_web: Optional[bool] = None) -> GenerationResult:
        """Luồng End-to-End: Truy xuất hợp nhất (Local + Web) -> Sinh câu trả lời kèm trích dẫn xác thực."""
        top_k_contexts = self.retrieve_combined(query, enable_web=enable_web)
        return self.generator.generate(query, top_k_contexts)