import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Cấu hình Google Gemini API
    llm_api_key: str = os.getenv("APIKEY", "YOUR_NEW_API_KEY_HERE")
    llm_base_url: str = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("MODEL","")       # Tên model của bên cung cấp mới
    fallback_llm_model: str = "llama-3.1-8b-instant"  # Model dự phòng khi quá tải
    max_rpm: int = 10                             # Số request/phút của nhà cung cấp mới

    # 2. Cấu hình Embedding (Giữ key Gemini cũ hoặc đổi sang Local / OpenAI Embedding)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dim: int = 768
    embedding_task: str = "search result"
    fallback_llm_model: str = "gemini-2.5-flash-lite"  # Tự động chuyển sang khi model chính bị 503
    gemini_model: str = "gemini-3.5-flash-lite"
    
    # Tham số Chunking
    chunk_size: int = 800          # Độ dài mục tiêu của 1 chunk (ký tự)
    chunk_overlap: int = 150       # Độ dài đoạn gối đầu giữa 2 chunk liên tiếp
    max_table_rows_per_chunk: int = 20  # Số dòng tối đa cho 1 chunk bảng
    
    # Tham số Indexing & Hybrid Retrieval
    bm25_k1: float = 1.5           # Độ bão hòa tần suất từ (TF saturation)
    bm25_b: float = 0.75           # Hệ số phạt độ dài văn bản
    rrf_k: int = 60                # Hằng số làm mượt Reciprocal Rank Fusion
    weight_dense: float = 0.5      # Trọng số nhánh Vector
    weight_bm25: float = 0.5       # Trọng số nhánh Từ khóa
    min_dense_cosine: float = 0.30
    
# Tham số Nhánh B: Web Search (Online)
    enable_web_search: bool = True
    google_search_api_key: str = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    google_cse_id: str = os.getenv("GOOGLE_CSE_ID", "")  # Programmable Search Engine ID (cx)
    use_llm_query_rewriter: bool = False  # True: dùng Gemini viết lại q_web; False: dùng thuật toán trích từ khóa nhanh (0 API call)
    web_top_u_urls: int = 5               # Top-U URLs lấy từ Macro Retriever
    web_scrape_timeout: float = 8.0       # Timeout tải mỗi trang web (giây)
    web_max_workers: int = 5              # Số luồng tải song song U trang web
    max_ephemeral_chunks: int = 100       # Giới hạn N_temp chunks trên RAM (<= 100 để vừa khít 1 batch embedding)

    # Tham số Hợp nhất ứng viên (M_local + M_web) & Cross-Encoder Reranking
    top_k_retrieval: int = 10             # M_local ứng viên lấy từ Nhánh A (có thể đặt = 20)
    top_k_web_retrieval: int = 10         # M_web ứng viên lấy từ Micro Retriever Nhánh B
    reranker_backend: str = "hybrid_calibrated"  # "local_cross_encoder" | "gemini_listwise" | "hybrid_calibrated"
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_score_threshold: float = 0.25  # Ngưỡng lọc điểm sau khi chấm chéo Rerank [0.0 -> 1.0]

    # Tham số Lọc sau truy xuất (Post-processing)
    top_k_final: int = 5                  # Top-K Chunks tốt nhất đưa vào LLM
    rrf_score_threshold: float = 0.0075
    max_merged_chars: int = 1800