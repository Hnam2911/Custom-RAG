import math
from typing import List, Set, Dict, Any
from schema import SearchResult, EvalSample, RetrievalEvalReport

class RetrievalEvaluator:
    """
    Bộ tính toán các chỉ số đánh giá Retrieval thủ công bằng toán học thuần:
    - Hit Rate@K: Tỷ lệ truy vấn tìm thấy ít nhất 1 đoạn đúng trong Top-K.
    - Recall@K: Tỷ lệ số đoạn đúng tìm thấy trên tổng số đoạn đúng của câu hỏi.
    - MRR@K (Mean Reciprocal Rank): Nghịch đảo thứ hạng của đoạn đúng đầu tiên (1 / rank).
    - NDCG@K (Normalized Discounted Cumulative Gain): Đánh giá chất lượng sắp xếp thứ tự.
    """

    @staticmethod
    def _resolve_chunk_ids(res: SearchResult) -> Set[str]:
        """
        Trả về tập hợp tất cả chunk_id mà SearchResult này đại diện
        (bao gồm cả các chunk gốc nằm trong metadata['merged_from'] nếu đã qua bước gộp ở Tầng 4).
        """
        ids = {res.chunk.chunk_id}
        merged_from = (res.chunk.metadata or {}).get("merged_from", [])
        for sub_id in merged_from:
            ids.add(str(sub_id))
        return ids

    def evaluate_single(
        self,
        sample: EvalSample,
        results: List[SearchResult],
        k: int = 5
    ) -> RetrievalEvalReport:
        """
        Chấm điểm danh sách SearchResult cho 1 mẫu EvalSample tại ngưỡng Top-K.
        Lưu ý: Tự động sắp xếp lại theo điểm RRF/Relevance giảm dần để không bị ảnh hưởng
        bởi phép hoán vị Lost-in-the-Middle ở cuối Tầng 4.
        """
        # Trường hợp câu hỏi ngoài tài liệu (expected_refusal=True):
        # Hệ thống truy xuất tốt nhất khi danh sách sau lọc hoàn toàn RỖNG.
        if sample.expected_refusal:
            is_clean_rejection = (len(results) == 0)
            score = 1.0 if is_clean_rejection else 0.0
            return RetrievalEvalReport(
                sample_id=sample.sample_id,
                k=k,
                hit_rate=score,
                recall=score,
                mrr=score,
                ndcg=score,
                retrieved_ids=[r.chunk.chunk_id for r in results[:k]],
                matched_targets=[]
            )

        # Khôi phục thứ tự xếp hạng theo điểm số thực tế (trước bước đảo Lost-in-the-Middle)
        ranked_results = sorted(results, key=lambda r: r.score, reverse=True)[:k]
        retrieved_ids = [r.chunk.chunk_id for r in ranked_results]

        # Xác định danh sách mục tiêu cần tìm (ưu tiên ground_truth_chunk_ids, hỗ trợ thêm expected_substrings)
        use_substrings = len(sample.ground_truth_chunk_ids) == 0 and len(sample.expected_substrings) > 0
        targets: List[str] = (
            list(sample.expected_substrings)
            if use_substrings
            else list(sample.ground_truth_chunk_ids)
        )

        if not targets:
            raise ValueError(f"Mẫu '{sample.sample_id}' phải có ground_truth_chunk_ids hoặc expected_substrings.")

        total_targets = len(targets)
        credited_targets: Set[str] = set()
        relevance_at_rank: List[int] = []  # Mảng nhị phân [0, 1, 0...] tại từng vị trí 1..K
        first_hit_rank: int = 0

        for rank_idx, res in enumerate(ranked_results, start=1):
            is_new_relevant_hit = False

            if use_substrings:
                content_lower = res.chunk.content.lower()
                for sub in targets:
                    if sub not in credited_targets and sub.lower() in content_lower:
                        credited_targets.add(sub)
                        is_new_relevant_hit = True
            else:
                covered_ids = self._resolve_chunk_ids(res)
                for gt_id in targets:
                    if gt_id not in credited_targets and gt_id in covered_ids:
                        credited_targets.add(gt_id)
                        is_new_relevant_hit = True

            if is_new_relevant_hit:
                relevance_at_rank.append(1)
                if first_hit_rank == 0:
                    first_hit_rank = rank_idx
            else:
                relevance_at_rank.append(0)

        # 1. Hit Rate@K: Có ít nhất 1 mục tiêu đúng được tìm thấy trong Top-K
        hit_rate = 1.0 if len(credited_targets) > 0 else 0.0

        # 2. Recall@K: Số mục tiêu tìm thấy / Tổng số mục tiêu chuẩn
        recall = len(credited_targets) / float(total_targets)

        # 3. MRR@K (Mean Reciprocal Rank): 1 / thứ hạng của kết quả đúng đầu tiên
        mrr = (1.0 / float(first_hit_rank)) if first_hit_rank > 0 else 0.0

        # 4. NDCG@K: DCG@K / IDCG@K
        # Công thức DCG@K = sum_{i=1..K} (rel_i / log2(i + 1))
        dcg = 0.0
        for idx, rel in enumerate(relevance_at_rank, start=1):
            if rel > 0:
                dcg += float(rel) / math.log2(idx + 1)

        # Số lượng vị trí lý tưởng tối đa có thể đạt rel=1 là min(K, số mục tiêu thực tế tìm được hoặc tổng mục tiêu)
        ideal_ones = min(k, total_targets)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_ones + 1))
        ndcg = min(1.0, dcg / idcg) if idcg > 0 else 0.0

        return RetrievalEvalReport(
            sample_id=sample.sample_id,
            k=k,
            hit_rate=round(hit_rate, 4),
            recall=round(recall, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            retrieved_ids=retrieved_ids,
            matched_targets=sorted(list(credited_targets))
        )

    def evaluate_dataset(
        self,
        samples: List[EvalSample],
        results_per_sample: List[List[SearchResult]],
        k: int = 5
    ) -> Dict[str, Any]:
        """Tính trung bình cộng các chỉ số trên toàn bộ tập dữ liệu kiểm thử."""
        if len(samples) != len(results_per_sample):
            raise ValueError("Số lượng mẫu (samples) và số lượng kết quả truy xuất không khớp.")

        reports: List[RetrievalEvalReport] = [
            self.evaluate_single(sample, res_list, k=k)
            for sample, res_list in zip(samples, results_per_sample)
        ]

        n = max(len(reports), 1)
        summary = {
            f"Hit_Rate@{k}": round(sum(r.hit_rate for r in reports) / n, 4),
            f"Recall@{k}": round(sum(r.recall for r in reports) / n, 4),
            f"MRR@{k}": round(sum(r.mrr for r in reports) / n, 4),
            f"NDCG@{k}": round(sum(r.ndcg for r in reports) / n, 4),
            "details": reports
        }
        return summary