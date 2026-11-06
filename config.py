import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Cấu hình Google Gemini API
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dim: int = 768       # Hỗ trợ linh hoạt 128 - 3072 (Khuyên dùng: 768, 1536, 3072)
    embedding_task: str = "search result"  # "search result" hoặc "question answering"
    llm_model: str = "gemini-3.5-flash"
    
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
    
    # Tham số Lọc sau truy xuất (Post-processing)
    top_k_retrieval: int = 10      # Số lượng lấy ra ở mỗi nhánh trước khi trộn
    top_k_final: int = 5           # Số lượng đoạn đưa vào Prompt cuối cùng
    rrf_score_threshold: float = 0.0075
    max_merged_chars: int = 1800