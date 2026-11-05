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

@dataclass
class SearchResult:
    """Kết quả trả về từ tầng Retrieval."""
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None