import re
from typing import List, Dict, Tuple, Optional
from config import RAGConfig
from schema import Chunk, SearchResult
from retrieval.reranker import CrossEncoderReranker

class RetrievalPostProcessor:
    """
    Bộ hậu xử lý kết quả truy xuất (Post-processor):
    1. Lọc theo ngưỡng điểm RRF (Score Thresholding).
    2. Gộp các chunk gối đầu (Overlap) hoặc liền kề trong cùng mục/bảng (Deduplication & Merging).
    3. Cắt Top-K cuối cùng và sắp xếp chống hiện tượng "Lost in the Middle".
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        
    def merge_and_rerank(
        self,
        query: str,
        local_results: List[SearchResult],
        web_results: List[SearchResult],
        reranker: Optional[CrossEncoderReranker] = None,
        enable_merging: bool = True,
        enable_lost_in_middle: bool = True
    ) -> List[SearchResult]:
        """
        Thực thi luồng Hợp nhất ứng viên (M_local + M_web) -> Cross-Encoder Reranking -> Top-K:
        1. Lọc sơ bộ các ứng viên dưới ngưỡng rrf_score_threshold ở từng nhánh.
        2. Gộp các chunk liền kề/gối đầu trong cùng 1 file hoặc cùng 1 URL để tạo ngữ cảnh hoàn chỉnh.
        3. Đưa toàn bộ ứng viên qua Cross-Encoder Reranker chấm điểm chéo với câu hỏi gốc q.
        4. Cắt lấy top_k_final chunks tốt nhất và đảo vị trí chống Lost-in-the-Middle.
        """
        combined = [
            r for r in (local_results + web_results)
            if r.score >= self.config.rrf_score_threshold
        ]
        if not combined:
            return []

        if enable_merging and len(combined) > 1:
            combined = self._merge_contiguous_results(combined)

        if reranker is not None:
            ranked_results = reranker.rerank(query, combined)
        else:
            ranked_results = sorted(combined, key=lambda r: r.score, reverse=True)

        if not ranked_results:
            return []

        top_final = ranked_results[:self.config.top_k_final]

        if enable_lost_in_middle and len(top_final) > 2:
            return self._reorder_lost_in_the_middle(top_final)

        return top_final

    def process(
        self,
        results: List[SearchResult],
        enable_merging: bool = True,
        enable_lost_in_middle: bool = True
    ) -> List[SearchResult]:
        if not results:
            return []

        # Bước 1: Lọc các đoạn có điểm RRF thấp hơn ngưỡng cấu hình
        filtered = [
            r for r in results
            if r.score >= self.config.rrf_score_threshold
        ]
        if not filtered:
            return []

        # Bước 2: Gộp các chunk liên tiếp / gối đầu (không sửa trực tiếp đối tượng Chunk trong Index)
        if enable_merging and len(filtered) > 1:
            processed_results = self._merge_contiguous_results(filtered)
        else:
            processed_results = filtered

        # Bước 3: Sắp xếp lại theo điểm RRF giảm dần và cắt lấy top_k_final
        processed_results.sort(key=lambda r: r.score, reverse=True)
        top_final = processed_results[:self.config.top_k_final]

        # Bước 4: Đảo vị trí chống hiện tượng "Lost in the Middle"
        if enable_lost_in_middle and len(top_final) > 2:
            return self._reorder_lost_in_the_middle(top_final)

        return top_final

    def _merge_contiguous_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Gom nhóm kết quả theo (source_file, doc_id), sắp xếp theo vị trí ký tự toàn cục (start_char_idx)
        và nối các chunk gối đầu hoặc liền kề trong cùng một mục/bảng.
        """
        grouped: Dict[Tuple[str, str], List[SearchResult]] = {}
        for res in results:
            key = (res.chunk.source_file, res.chunk.doc_id)
            grouped.setdefault(key, []).append(res)

        merged_all: List[SearchResult] = []

        for _, group in grouped.items():
            if len(group) == 1:
                merged_all.append(group[0])
                continue

            # Sắp xếp theo vị trí xuất hiện thực tế trong tài liệu gốc
            group.sort(key=lambda r: (r.chunk.start_char_idx, r.chunk.end_char_idx))

            current_res = group[0]
            for next_res in group[1:]:
                if self._can_merge(current_res.chunk, next_res.chunk):
                    current_res = self._merge_two_results(current_res, next_res)
                else:
                    merged_all.append(current_res)
                    current_res = next_res
            merged_all.append(current_res)

        return merged_all

    def _can_merge(self, chunk_a: Chunk, chunk_b: Chunk) -> bool:
        """Kiểm tra 2 chunk (A đứng trước B trên cùng tài liệu) có nên gộp lại hay không."""
        if chunk_a.block_type != chunk_b.block_type:
            return False

        # Kiểm tra giới hạn độ dài tối đa sau khi gộp
        estimated_len = len(chunk_a.content) + len(chunk_b.content)
        if estimated_len > self.config.max_merged_chars:
            return False

        gap = chunk_b.start_char_idx - chunk_a.end_char_idx

        # Trường hợp 1: Hai chunk bảng thuộc cùng một bảng lớn bị cắt làm nhiều phần
        if chunk_a.block_type == "table":
            # Cùng trang (nếu là PDF) và tiếp giáp nhau trên file gốc
            same_page = (chunk_a.page_number == chunk_b.page_number)
            return same_page and (-10 <= gap <= 30) and self._have_same_table_header(chunk_a.content, chunk_b.content)

        # Trường hợp 2: Hai chunk văn bản có vùng ký tự gối đầu nhau (Overlap thực sự)
        if gap < 0:
            return True

        # Trường hợp 3: Hai chunk văn bản nằm sát nhau (gap <= 25 ký tự do xóa dấu '---' / '\n\n')
        # CHỈ gộp nếu chúng cùng nằm trong một mục H2 (parent_heading có chung cấp '## ')
        # để nối lại các danh sách/giai đoạn bị xé nhỏ mà không làm dính 2 mục lớn độc lập (VD: ## 1 và ## 2)
        if 0 <= gap <= 25:
            h2_a = self._extract_h2_ancestor(chunk_a)
            h2_b = self._extract_h2_ancestor(chunk_b)
            if h2_a and h2_b and h2_a == h2_b:
                return True

        return False

    @staticmethod
    def _extract_h2_ancestor(chunk: Chunk) -> str:
        """Lấy tiêu đề cấp '## ' từ parent_heading hoặc từ dòng đầu của chunk.content."""
        parent = (chunk.metadata or {}).get("parent_heading", "")
        for part in parent.split(" > "):
            p = part.strip()
            if p.startswith("## ") and not p.startswith("### "):
                return p
        for line in chunk.content.splitlines():
            s = line.strip()
            if s.startswith("## ") and not s.startswith("### "):
                return s
        return ""

    @staticmethod
    def _have_same_table_header(table_a: str, table_b: str) -> bool:
        """Kiểm tra 2 chunk bảng có cùng dòng Header cột '| ... |' hay không."""
        rows_a = [line.strip() for line in table_a.splitlines() if line.strip().startswith("|")]
        rows_b = [line.strip() for line in table_b.splitlines() if line.strip().startswith("|")]
        if len(rows_a) >= 2 and len(rows_b) >= 2:
            return rows_a[0] == rows_b[0]
        return False

    def _merge_two_results(self, res_a: SearchResult, res_b: SearchResult) -> SearchResult:
        """
        Tạo một đối tượng Chunk và SearchResult MỚI từ res_a và res_b.
        Tuyệt đối không chỉnh sửa in-place vào res_a.chunk hay res_b.chunk.
        """
        ca, cb = res_a.chunk, res_b.chunk

        if ca.block_type == "table":
            merged_content = self._merge_table_contents(ca.content, cb.content)
        else:
            overlap_chars = max(0, ca.end_char_idx - cb.start_char_idx)
            merged_content = self._merge_text_contents(ca.content, cb.content, overlap_chars)

        # Kế thừa metadata an toàn
        merged_meta = dict(ca.metadata or {})
        if ca.block_type == "table" and "row_range" in merged_meta and "row_range" in (cb.metadata or {}):
            start_r = merged_meta["row_range"].split("-")[0]
            end_r = cb.metadata["row_range"].split("-")[-1]
            merged_meta["row_range"] = f"{start_r}-{end_r}"

        merged_meta["merged_from"] = merged_meta.get("merged_from", [ca.chunk_id]) + [cb.chunk_id]

        merged_chunk = Chunk(
            chunk_id=f"{ca.chunk_id}+{cb.chunk_id.split('_')[-1]}",
            doc_id=ca.doc_id,
            content=merged_content,
            block_type=ca.block_type,
            source_file=ca.source_file,
            page_number=ca.page_number,
            start_char_idx=min(ca.start_char_idx, cb.start_char_idx),
            end_char_idx=max(ca.end_char_idx, cb.end_char_idx),
            metadata=merged_meta,
            embedding=None
        )

        # Giữ mức điểm tốt nhất của cụm sau khi gộp
        best_dense_rank = min(
            [r for r in (res_a.dense_rank, res_b.dense_rank) if r is not None],
            default=None
        )
        best_bm25_rank = min(
            [r for r in (res_a.bm25_rank, res_b.bm25_rank) if r is not None],
            default=None
        )
        best_dense_score = max(
            [s for s in (res_a.dense_score, res_b.dense_score) if s is not None],
            default=None
        )
        best_bm25_score = max(
            [s for s in (res_a.bm25_score, res_b.bm25_score) if s is not None],
            default=None
        )
        
        best_rerank_score = max(
            [s for s in (res_a.rerank_score, res_b.rerank_score) if s is not None],
            default=None
        )
        
        return SearchResult(
            chunk=merged_chunk,
            score=max(res_a.score, res_b.score),
            dense_rank=best_dense_rank,
            bm25_rank=best_bm25_rank,
            dense_score=best_dense_score,
            bm25_score=best_bm25_score,
            rerank_score=best_rerank_score,
            source_branch=res_a.source_branch
        )

    @staticmethod
    def _merge_text_contents(text_a: str, text_b: str, overlap_hint: int) -> str:
        """
        Khử phần chuỗi trùng lặp (overlap) ở đuôi text_a và đầu text_b.
        Nếu không có chuỗi trùng, nối mạch lạc bằng ký tự xuống dòng kép.
        """
        if overlap_hint > 0:
            max_check = min(len(text_a), len(text_b), overlap_hint + 120)
            # Tìm đoạn khớp dài nhất giữa hậu tố của text_a và tiền tố của text_b
            for length in range(max_check, 5, -1):
                if text_a.endswith(text_b[:length]):
                    return text_a + text_b[length:]

            # Fallback: Tìm vị trí mỏ neo 30 ký tự đầu của text_b nằm ở cuối text_a
            anchor = text_b[:min(30, len(text_b))]
            pos = text_a.rfind(anchor, max(0, len(text_a) - max_check))
            if pos != -1:
                return text_a[:pos] + text_b

        return f"{text_a.rstrip()}\n\n{text_b.lstrip()}"

    @staticmethod
    def _merge_table_contents(table_a: str, table_b: str) -> str:
        """
        Nối 2 mảnh của cùng một bảng Markdown:
        Bỏ qua phần tiêu đề bảng và 2 dòng Header cột lặp lại ở đầu table_b, chỉ lấy các dòng dữ liệu.
        """
        lines_b = [line.strip() for line in table_b.splitlines() if line.strip()]
        table_rows_b = [line for line in lines_b if line.startswith("|")]
        data_rows_b = table_rows_b[2:] if len(table_rows_b) > 2 else []

        if not data_rows_b:
            return table_a
        return table_a.rstrip() + "\n" + "\n".join(data_rows_b)

    @staticmethod
    def _reorder_lost_in_the_middle(sorted_results: List[SearchResult]) -> List[SearchResult]:
        """
        Sắp xếp lại danh sách đã giảm dần theo điểm [R1, R2, R3, R4, R5]
        sao cho các đoạn điểm cao nhất nằm ở 2 biên (Đầu và Cuối), điểm thấp nhất nằm ở Giữa:
        Ví dụ 5 phần tử: [R1, R3, R5, R4, R2]
        """
        left_side: List[SearchResult] = []
        right_side: List[SearchResult] = []

        for idx, item in enumerate(sorted_results):
            if idx % 2 == 0:
                left_side.append(item)   # R1 (idx 0), R3 (idx 2), R5 (idx 4)...
            else:
                right_side.append(item)  # R2 (idx 1), R4 (idx 3)...

        # Đảo ngược nhánh phải để R2 nằm ở vị trí cuối cùng của mảng kết quả
        return left_side + right_side[::-1]