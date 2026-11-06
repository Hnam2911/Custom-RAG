from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

@dataclass
class RawBlock:
    """Đơn vị dữ liệu thô bóc tách trực tiếp từ file (1 đoạn văn hoặc 1 bảng)."""
    text: str
    block_type: str  # "text" hoặc "table"
    source_file: str
    page_number: Optional[int] = None
    base_char_idx: int = 0  # Vị trí ký tự bắt đầu của block tính trên toàn file
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """Đơn vị dữ liệu sau khi cắt nhỏ, sẵn sàng để đưa vào Index."""
    chunk_id: str
    doc_id: str
    content: str
    block_type: str  # "text" hoặc "table"
    source_file: str
    page_number: Optional[int] = None
    start_char_idx: int = 0
    end_char_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def get_contextual_text(self) -> str:
        """
        Ghép parent_heading vào trước content để phục vụ Embedding, BM25 và Prompting
        nhưng tự động loại bỏ các cấp heading đã nằm sẵn trong content để tránh nhân đôi TF.
        """
        parent_heading = (self.metadata or {}).get("parent_heading", "").strip()
        if not parent_heading:
            return self.content

        # Tách chuỗi phả hệ '# H1 > ## H2' và lọc bỏ cấp nào đã có sẵn trong chunk.content
        heading_parts = [h.strip() for h in parent_heading.split(" > ") if h.strip()]
        missing_headings = [h for h in heading_parts if h not in self.content]

        if not missing_headings:
            return self.content

        prefix = " > ".join(missing_headings)
        return f"[Ngữ cảnh: {prefix}]\n{self.content}"

@dataclass
class SearchResult:
    """Kết quả trả về từ tầng Retrieval."""
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    
@dataclass
class Citation:
    """Thông tin trích dẫn đã được đối chiếu ngược về vị trí ký tự trên tài liệu gốc."""
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    start_char_idx: int
    end_char_idx: int
    block_type: str
    parent_heading: str = ""

@dataclass
class GenerationResult:
    """Kết quả đầu ra hoàn chỉnh của Tầng 5 (Generation)."""
    query: str
    answer: str
    citations: List[Citation]                  # Danh sách nguồn thực sự được LLM trích dẫn hợp lệ
    used_contexts: List[SearchResult]          # Toàn bộ các đoạn ngữ cảnh đã đưa vào Prompt
    system_prompt: str = ""
    user_prompt: str = ""
    finish_reason: str = "STOP"
    is_refusal: bool = False                   # True nếu không tìm thấy thông tin trong tài liệu
    
@dataclass
class EvalSample:
    """Một mẫu kiểm thử chuẩn (Golden Sample) trong bộ Benchmark."""
    sample_id: str
    query: str
    ground_truth_chunk_ids: List[str] = field(default_factory=list)   # Danh sách chunk_id kỳ vọng
    expected_substrings: List[str] = field(default_factory=list)      # Các cụm từ/số liệu bắt buộc phải nằm trong chunk truy xuất được
    reference_answer: str = ""                                        # Câu trả lời chuẩn để đối chiếu
    expected_refusal: bool = False                                    # True nếu câu hỏi nằm ngoài tài liệu

@dataclass
class RetrievalEvalReport:
    """Kết quả đo lường nhánh Retrieval trên 1 câu truy vấn."""
    sample_id: str
    k: int
    hit_rate: float          # 0.0 hoặc 1.0 (Có ít nhất 1 mục tiêu đúng trong Top-K)
    recall: float            # Tỷ lệ mục tiêu đúng tìm được trên tổng số mục tiêu cần tìm
    mrr: float               # Mean Reciprocal Rank: 1 / rank_đúng_đầu_tiên
    ndcg: float              # Normalized Discounted Cumulative Gain [0.0 -> 1.0]
    retrieved_ids: List[str] = field(default_factory=list)
    matched_targets: List[str] = field(default_factory=list)

@dataclass
class ClaimVerification:
    """Kết quả kiểm chứng từng mệnh đề (claim) đơn lẻ trong câu trả lời của LLM."""
    claim: str
    supported: bool          # True nếu mệnh đề có căn cứ trực tiếp trong ngữ cảnh
    evidence_id: str = ""    # Mã chunk chứng minh mệnh đề đó

@dataclass
class GenerationEvalReport:
    """Kết quả chấm điểm nhánh Generation (LLM-as-a-Judge) trên 1 câu truy vấn."""
    sample_id: str
    faithfulness: float          # Số mệnh đề có căn cứ / Tổng số mệnh đề [0.0 -> 1.0]
    answer_relevance: float      # Mức độ giải quyết trọn vẹn câu hỏi [0.0 -> 1.0]
    citation_precision: float    # Tỷ lệ mã trích dẫn [chunk_id] thực sự chứa thông tin được nói tới [0.0 -> 1.0]
    claims: List[ClaimVerification] = field(default_factory=list)
    judge_reasoning: str = ""