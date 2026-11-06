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