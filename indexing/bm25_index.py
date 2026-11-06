import re
import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from schema import Chunk

class BM25Index:
    """
    Chỉ mục từ khóa thưa (Sparse Index) cài đặt thủ công theo thuật toán Okapi BM25.
    Tích hợp bộ Tokenizer bảo toàn mã kỹ thuật (INF-01, DB-01, TLS 1.3) và sinh Bigram cho tiếng Việt.
    """
    # Danh sách từ dừng (Stopwords) cơ bản tiếng Việt & tiếng Anh không mang nhiều ý nghĩa phân loại
    STOPWORDS: Set[str] = {
        "là", "và", "của", "các", "có", "được", "trong", "cho", "với", "để",
        "theo", "từ", "trên", "đến", "khi", "về", "như", "này", "đó", "một",
        "những", "bằng", "tại", "hoặc", "do", "ra", "vào", "lên", "sẽ", "đã",
        "the", "and", "of", "to", "in", "for", "on", "with", "as", "by", "is", "are"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_bigrams: bool = True):
        self.k1 = k1
        self.b = b
        self.use_bigrams = use_bigrams

        self.chunks: List[Chunk] = []
        self._id_to_idx: Dict[str, int] = {}

        # Cấu trúc dữ liệu lõi của BM25
        self.doc_Wait_tokens: List[List[str]] = []               # Danh sách token của từng chunk
        self.doc_lens: List[int] = []                            # Độ dài |D| của từng chunk
        self.avgdl: float = 0.0                                  # Độ dài trung bình văn bản
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)  # token -> {doc_idx: tf}
        self.idf: Dict[str, float] = {}                          # token -> giá trị IDF

    @property
    def size(self) -> int:
        return len(self.chunks)

    def tokenize(self, text: str) -> List[str]:
        """
        Bộ tách từ chuyên biệt cho văn bản kỹ thuật Tiếng Việt + Tiếng Anh:
        1. Chuẩn hóa dấu phẩy ngăn cách hàng nghìn trong số: '1,250' -> '1250'.
        2. Trích xuất các token đơn (giữ cả từ ghép có gạch ngang/dấu chấm như 'inf-01', '1.3', '99.95').
        3. Tách bổ sung các thành phần con của từ có gạch ngang ('inf-01' -> thêm 'inf', '01').
        4. Sinh thêm các cặp 2 từ liền kề (word bigrams) để bắt chính xác từ ghép tiếng Việt.
        """
        if not text:
            return []

        normalized = text.lower()
        # Gộp dấu phẩy ngăn cách hàng nghìn giữa các chữ số (VD: $1,800 -> $1800)
        normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)

        # Bắt các chuỗi chữ/số, cho phép nối bên trong bằng dấu '-', '.', '_' (VD: inf-01, aes-256, 99.95, 1.3)
        raw_tokens = re.findall(r"[a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:[-._][a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)*", normalized)

        unigrams: List[str] = []
        expanded_subtokens: List[str] = []

        for tok in raw_tokens:
            if tok in self.STOPWORDS:
                continue
            unigrams.append(tok)
            # Nếu là mã kỹ thuật có dấu '-' (như inf-01, db-02, multi-az), tách thêm từ con để hỗ trợ tìm kiếm rời
            if "-" in tok:
                for sub in tok.split("-"):
                    if sub and sub not in self.STOPWORDS and len(sub) > 1:
                        expanded_subtokens.append(sub)

        # Sinh Word Bigrams từ chuỗi unigrams liên tiếp để bắt từ ghép tiếng Việt (VD: "cơ_sở", "dữ_liệu")
        bigrams: List[str] = []
        if self.use_bigrams and len(unigrams) >= 2:
            for i in range(len(unigrams) - 1):
                bigrams.append(f"{unigrams[i]}_{unigrams[i + 1]}")

        return unigrams + expanded_subtokens + bigrams

    @staticmethod
    def _build_contextual_text(chunk: Chunk) -> str:
        """Lấy văn bản đã ghép parent_heading (không trùng lặp) để đánh chỉ mục BM25."""
        if hasattr(chunk, "get_contextual_text") and callable(chunk.get_contextual_text):
            return chunk.get_contextual_text()

        parent_heading = (chunk.metadata or {}).get("parent_heading", "").strip()
        if not parent_heading:
            return chunk.content

        heading_parts = [h.strip() for h in parent_heading.split(" > ") if h.strip()]
        missing_headings = [h for h in heading_parts if h not in chunk.content]
        if not missing_headings:
            return chunk.content

        return f"{' > '.join(missing_headings)}\n{chunk.content}"

    def _rebuild_statistics(self) -> None:
        """Tính toán lại Inverted Index, avgdl và bảng IDF trên toàn bộ tập tài liệu hiện có."""
        self.inverted_index.clear()
        self.idf.clear()

        n_docs = len(self.chunks)
        if n_docs == 0:
            self.avgdl = 0.0
            return

        total_len = sum(self.doc_lens)
        # Đảm bảo avgdl không bao giờ bằng 0 để tránh lỗi ZeroDivisionError
        self.avgdl = max(total_len / n_docs, 1.0)

        # Xây dựng Inverted Index: token -> {doc_idx: tần suất xuất hiện (TF)}
        for doc_idx, tokens in enumerate(self.doc_Wait_tokens):
            term_counts = Counter(tokens)
            for token, freq in term_counts.items():
                self.inverted_index[token][doc_idx] = freq

        # Tính IDF theo công thức chuẩn Lucene / Robertson-Spärck Jones (luôn dương nhờ + 1 trong log):
        # IDF(q_i) = ln( (N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1 )
        for token, doc_dict in self.inverted_index.items():
            df = len(doc_dict)
            self.idf[token] = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Thêm danh sách Chunk vào chỉ mục BM25 (hỗ trợ cập nhật nếu trùng chunk_id).
        """
        if not chunks:
            return

        for chunk in chunks:
            text_for_bm25 = self._build_contextual_text(chunk)
            tokens = self.tokenize(text_for_bm25)

            if chunk.chunk_id in self._id_to_idx:
                idx = self._id_to_idx[chunk.chunk_id]
                self.chunks[idx] = chunk
                self.doc_Wait_tokens[idx] = tokens
                self.doc_lens[idx] = len(tokens)
            else:
                idx = len(self.chunks)
                self._id_to_idx[chunk.chunk_id] = idx
                self.chunks.append(chunk)
                self.doc_Wait_tokens.append(tokens)
                self.doc_lens.append(len(tokens))

        self._rebuild_statistics()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float, int]]:
        """
        Tính điểm BM25 cho câu truy vấn sử dụng Inverted Index.
        CHỈ trả về các chunk có điểm BM25 > 0.0 để không làm nhiễu thuật toán trộn RRF ở Tầng 4.
        Trả về danh sách tuple: (Chunk, bm25_score, rank_1_based).
        """
        if self.size == 0 or top_k <= 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Đếm tần suất từ trong câu hỏi (nếu câu hỏi lặp từ khóa)
        query_term_weights = Counter(query_tokens)
        scores = np.zeros(self.size, dtype=np.float64)

        # Chỉ duyệt qua các tài liệu có chứa ít nhất 1 token của câu hỏi thông qua Inverted Index
        for token, q_freq in query_term_weights.items():
            if token not in self.inverted_index:
                continue

            token_idf = self.idf[token]
            posting_list = self.inverted_index[token]

            for doc_idx, tf in posting_list.items():
                doc_len = self.doc_lens[doc_idx]
                # Công thức BM25: IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (|D| / avgdl)))
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += token_idf * (numerator / denominator) * q_freq

        # Lọc bỏ hoàn toàn các tài liệu có điểm <= 0.0 (Tránh lỗi đầu độc RRF)
        positive_indices = np.where(scores > 0.0)[0]
        if len(positive_indices) == 0:
            return []

        # Sắp xếp các tài liệu có điểm > 0 theo thứ tự giảm dần
        sorted_pos_indices = positive_indices[np.argsort(scores[positive_indices])[::-1]]
        top_indices = sorted_pos_indices[:min(top_k, len(sorted_pos_indices))]

        results: List[Tuple[Chunk, float, int]] = []
        for rank_idx, doc_idx in enumerate(top_indices, start=1):
            results.append((self.chunks[int(doc_idx)], float(scores[int(doc_idx)]), rank_idx))

        return results

    def clear(self) -> None:
        self.chunks.clear()
        self._id_to_idx.clear()
        self.doc_Wait_tokens.clear()
        self.doc_lens.clear()
        self.inverted_index.clear()
        self.idf.clear()
        self.avgdl = 0.0