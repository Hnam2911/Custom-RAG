import numpy as np
from typing import List, Dict, Tuple, Optional
from schema import Chunk

class DenseFlatIndex:
    """
    Chỉ mục Vector phẳng (Flat Index) lưu trực tiếp trên RAM bằng ma trận NumPy 2D (float32).
    Chuẩn hóa L2 toàn bộ vector ngay khi lưu để biến Cosine Similarity thành phép nhân vô hướng (Dot Product).
    """
    def __init__(self):
        self.matrix: Optional[np.ndarray] = None  # Kích thước (N, D), kiểu float32, đã chuẩn hóa L2
        self.chunks: List[Chunk] = []
        self._id_to_idx: Dict[str, int] = {}      # Ánh xạ chunk_id -> chỉ số dòng trong ma trận

    @property
    def size(self) -> int:
        return len(self.chunks)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa L2 theo từng dòng: v_norm = v / max(||v||_2, 1e-12).
        Tránh tuyệt đối lỗi chia cho 0 sinh ra NaN làm hỏng thứ tự argsort.
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            if norm < 1e-12 or np.isnan(norm):
                return np.zeros_like(vectors, dtype=np.float32)
            return (vectors / norm).astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where((norms < 1e-12) | np.isnan(norms), 1.0, norms)
        normalized = np.where((norms < 1e-12) | np.isnan(norms), 0.0, vectors / safe_norms)
        return normalized.astype(np.float32)

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[np.ndarray] = None) -> None:
        """
        Thêm danh sách Chunk và Vector vào Index.
        Nếu chunk_id đã tồn tại (do người dùng upload lại file), tự động cập nhật tại chỗ thay vì nhân bản.
        """
        if not chunks:
            return

        if embeddings is None:
            # Lấy trực tiếp từ thuộc tính chunk.embedding
            missing = [c.chunk_id for c in chunks if c.embedding is None]
            if missing:
                raise ValueError(f"Các chunk sau chưa có vector embedding: {missing[:5]}")
            raw_matrix = np.vstack([c.embedding for c in chunks]).astype(np.float32)#type:ignore
        else:
            raw_matrix = np.asarray(embeddings, dtype=np.float32)
            if raw_matrix.ndim == 1:
                raw_matrix = raw_matrix.reshape(1, -1)
            if raw_matrix.shape[0] != len(chunks):
                raise ValueError("Số dòng của ma trận embedding không khớp với số lượng chunks.")

        norm_matrix = self._l2_normalize(raw_matrix)

        new_chunks_to_append: List[Chunk] = []
        new_vecs_to_append: List[np.ndarray] = []

        for chunk, norm_vec in zip(chunks, norm_matrix):
            chunk.embedding = norm_vec
            if chunk.chunk_id in self._id_to_idx:
                # Ghi đè nếu chunk_id đã tồn tại
                existing_idx = self._id_to_idx[chunk.chunk_id]
                self.chunks[existing_idx] = chunk
                self.matrix[existing_idx] = norm_vec#type:ignore
            else:
                new_idx = len(self.chunks) + len(new_chunks_to_append)
                self._id_to_idx[chunk.chunk_id] = new_idx
                new_chunks_to_append.append(chunk)
                new_vecs_to_append.append(norm_vec)

        if new_chunks_to_append:
            self.chunks.extend(new_chunks_to_append)
            append_mat = np.vstack(new_vecs_to_append).astype(np.float32)
            if self.matrix is None or self.matrix.size == 0:
                self.matrix = append_mat
            else:
                if self.matrix.shape[1] != append_mat.shape[1]:
                    raise ValueError(
                        f"Số chiều vector mới ({append_mat.shape[1]}) khác với Index hiện tại ({self.matrix.shape[1]})."
                    )
                self.matrix = np.vstack([self.matrix, append_mat])

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[Chunk, float, int]]:
        """
        Tìm kiếm Top-K chunk có điểm Cosine Similarity cao nhất.
        Trả về danh sách tuple: (Chunk, cosine_score, rank_1_based).
        """
        if self.matrix is None or self.size == 0 or top_k <= 0:
            return []

        q_norm = self._l2_normalize(np.asarray(query_vector, dtype=np.float32).flatten())
        if np.all(q_norm == 0):
            return []

        # Vì cả matrix (N, D) và q_norm (D,) đều đã chuẩn hóa L2,
        # phép nhân ma trận np.dot chính là Cosine Similarity của toàn bộ N chunks
        scores = np.dot(self.matrix, q_norm)
        # Chặn mọi khả năng NaN lọt vào argsort
        scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)

        k = min(top_k, self.size)
        # Sắp xếp giảm dần
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[Chunk, float, int]] = []
        for rank_idx, doc_idx in enumerate(top_indices, start=1):
            results.append((self.chunks[int(doc_idx)], float(scores[int(doc_idx)]), rank_idx))

        return results

    def clear(self) -> None:
        """Giải phóng bộ nhớ Index của phiên hiện tại."""
        self.matrix = None
        self.chunks.clear()
        self._id_to_idx.clear()