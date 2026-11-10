import os
import re
import json
import time
import string
import random
import pickle
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# Import trực tiếp từ các module trong repo Custom-RAG của bạn
from config import RAGConfig
from schema import Chunk, SearchResult, EvalSample, GenerationResult
from preprocess_file import process_file_to_chunks
from indexing.bm25_index import BM25Index
from indexing.dense_flat_index import DenseFlatIndex
from indexing.gemini_embedder import GeminiEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.post_processor import RetrievalPostProcessor
from generation.prompt_builder import PromptBuilder
from generation.gemini_llm import GeminiLLMGenerator
from evaluation.retrieval_metrics import RetrievalEvaluator
from evaluation.generation_judge import LLMGenerationJudge


# =====================================================================
# 1. TRIVIAQA PROMPT BUILDER (Kế thừa PromptBuilder gốc của bạn)
# =====================================================================
class TriviaQAPromptBuilder(PromptBuilder):
    """
    Kế thừa PromptBuilder của Custom-RAG để giữ nguyên cơ chế đóng gói XML <chunk>,
    parent_heading và kiểm tra trích dẫn [chunk_id], nhưng điều chỉnh System Instruction
    để LLM trả lời ngắn gọn (Short Extractive Answer) phục vụ chấm điểm EM/F1 trên TriviaQA.
    """
    def build_system_instruction(self) -> str:
        return (
            "You are a precise technical and encyclopedic question-answering assistant.\n"
            "YOUR RULES:\n"
            "1. Use ONLY the information inside <retrieved_context> to answer the user's question.\n"
            "2. Output ONLY the exact entity, name, number, date, or short phrase that directly answers the question, "
            "immediately followed by the valid source citation tag [chunk_id] (e.g., 'Paris [doc1_c0]').\n"
            "3. Do NOT write full sentences, introductory phrases, or explanations.\n"
            f"4. If <retrieved_context> does not contain the answer, output EXACTLY: \"{self.REFUSAL_PHRASE}\""
        )

    def build_user_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        base_prompt = super().build_user_prompt(query, search_results)
        # Thay dòng nhắc tiếng Việt cuối cùng bằng lời nhắc ngắn gọn cho TriviaQA
        return base_prompt.replace(
            "Hãy trả lời chi tiết, rõ ràng và đính kèm mã [chunk_id] chính xác sau từng thông tin:",
            "Concise Answer (exact entity/phrase + [chunk_id] only):"
        )


# =====================================================================
# 2. BỘ CHẤM ĐIỂM CHÍNH THỨC CỦA REPO MANDARJOSHI90/TRIVIAQA
# =====================================================================
def normalize_answer(s: str) -> str:
    """Hàm chuẩn hóa chuỗi nguyên bản từ evaluation/triviaqa_evaluation.py của mandarjoshi90."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s)))))


def f1_score_triviaqa(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score_triviaqa(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    scores_for_ground_truths = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0


def strip_citations_for_triviaqa(answer_text: str) -> str:
    """Xóa các thẻ [doc_id_c0] khỏi câu trả lời trước khi chấm Exact Match / F1."""
    cleaned = re.sub(r"\[[a-zA-Z0-9_+.-]+_c\d+(?:\+c\d+)*\]", "", answer_text)
    return cleaned.strip()


# =====================================================================
# 3. TẢI & CHUẨN BỊ DỮ LIỆU TRIVIAQA (500 CÂU + TÀI LIỆU MARKDOWN)
# =====================================================================
def prepare_triviaqa_subset(
    data_dir: str,
    num_samples: int = 500,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Tải tập validation của mandarjoshi/trivia_qa (nhánh rc.wikipedia),
    trích xuất ngẫu nhiên 500 câu hỏi và xuất các bài Wikipedia liên quan thành file .md
    có tiêu đề '# Title' để tận dụng bộ parser_text.py và parent_heading của bạn.
    """
    os.makedirs(data_dir, exist_ok=True)
    corpus_dir = os.path.join(data_dir, "corpus_md")
    os.makedirs(corpus_dir, exist_ok=True)
    dataset_json_path = os.path.join(data_dir, f"triviaqa_{num_samples}_dataset.json")

    if os.path.exists(dataset_json_path) and len(os.listdir(corpus_dir)) > 0:
        print(f"[*] Đã tìm thấy bộ dữ liệu {num_samples} câu tại '{dataset_json_path}'. Bỏ qua bước tải lại.")
        return dataset_json_path, corpus_dir

    
    from datasets import load_dataset
    print("[*] Đang stream dữ liệu 'mandarjoshi/trivia_qa' (rc.wikipedia) từ Hugging Face...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", streaming=True)

    # Thu thập các mẫu có chứa ít nhất 1 tài liệu Wikipedia đi kèm
    valid_pool = []
    target_pool_size = max(num_samples * 3, 1500)
    for item in ds:
        if len(item["entity_pages"]["wiki_context"]) > 0:
            valid_pool.append(item)
        if len(valid_pool) >= target_pool_size:
            break

    random.seed(seed)
    sampled_items = random.sample(valid_pool, min(num_samples, len(valid_pool)))

    triviaqa_data_entries = []
    doc_registry: Dict[str, str] = {}  # title -> doc_id

    for item in sampled_items:
        qid = item["question_id"]
        question = item["question"]
        ans = item["answer"]
        entity_pages = item["entity_pages"]

        sample_doc_ids = []
        combined_sections = []
        for title, wiki_ctx in zip(entity_pages["title"], entity_pages["wiki_context"]):
            # Lấy toàn bộ các trang liên quan đến câu hỏi để không bị thiếu đáp án
            combined_sections.append(f"# {title}\n\n{wiki_ctx}")

        if combined_sections:
            doc_id = f"doc{len(doc_registry) + 1:04d}"
            doc_registry[qid] = doc_id
            md_path = os.path.join(corpus_dir, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(combined_sections))
            sample_doc_ids.append(doc_id)

        triviaqa_data_entries.append({
            "QuestionId": qid,
            "Question": question,
            "TargetDocIds": sample_doc_ids,
            "Answer": {
                "Value": ans["value"],
                "NormalizedValue": ans["normalized_value"],
                "Aliases": ans["aliases"],
                "NormalizedAliases": ans["normalized_aliases"]
            }
        })

    official_format = {
        "Data": triviaqa_data_entries,
        "Domain": "Wikipedia",
        "VerifiedEval": False,
        "Version": 1.0
    }

    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(official_format, f, ensure_ascii=False, indent=2)

    print(f"[*] Hoàn tất chuẩn bị {len(triviaqa_data_entries)} câu hỏi và {len(doc_registry)} file tài liệu tại '{corpus_dir}'.")
    return dataset_json_path, corpus_dir


# =====================================================================
# 4. XÂY DỰNG INDEX & CACHE EMBEDDING TRÊN Ổ CỨNG (TIẾT KIỆM QUOTA)
# =====================================================================
def build_or_load_indices(
    corpus_dir: str,
    cache_dir: str,
    config: RAGConfig,
    embedder: GeminiEmbedder,
    questions: List[str],
    use_dense: bool = True
) -> Tuple[BM25Index, DenseFlatIndex, List[Chunk]]:
    os.makedirs(cache_dir, exist_ok=True)
    chunks_cache_path = os.path.join(cache_dir, "chunks_without_vec.pkl")
    vectors_cache_path = os.path.join(cache_dir, "chunk_embeddings.npz")

    if os.path.exists(chunks_cache_path):
        with open(chunks_cache_path, "rb") as f:
            all_chunks: List[Chunk] = pickle.load(f)
    else:
        md_files = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".md")])
        all_chunks = []
        for fname in tqdm(md_files, desc="Chunking"):
            doc_id = os.path.splitext(fname)[0]
            fpath = os.path.join(corpus_dir, fname)
            all_chunks.extend(process_file_to_chunks(file_path=fpath, doc_id=doc_id, config=config))
        with open(chunks_cache_path, "wb") as f:
            pickle.dump(all_chunks, f)

    # 1. BM25Index đánh chỉ mục TOÀN BỘ 24,000 chunks (miễn phí 100%)
    bm25_index = BM25Index(k1=config.bm25_k1, b=config.bm25_b, use_bigrams=True)
    bm25_index.add_chunks(all_chunks)

    dense_index = DenseFlatIndex()
    if not use_dense or config.weight_dense <= 0.0:
        return bm25_index, dense_index, all_chunks

    # 2. Lọc ra các Chunks ứng viên sáng giá nhất cho 500 câu hỏi (khoảng 600-900 chunks thay vì 24,000 chunks)
    candidate_map: Dict[str, Chunk] = {}
    for q in questions:
        for chunk, _, _ in bm25_index.search(q, top_k=2):
            candidate_map[chunk.chunk_id] = chunk
    target_chunks = list(candidate_map.values())
    print(f"[*] Đã rút gọn từ {len(all_chunks)} chunks xuống {len(target_chunks)} chunks trọng tâm để tiết kiệm Quota Embedding!")

    # 3. Nhúng các chunks trọng tâm với nhịp chờ 62s giữa mỗi batch 80 chunks
    embedded_map: Dict[str, np.ndarray] = {}
    if os.path.exists(vectors_cache_path):
        data = np.load(vectors_cache_path, allow_pickle=True)
        for cid, vec in zip(data["chunk_ids"], data["vectors"]):
            embedded_map[str(cid)] = vec

    missing_chunks = [c for c in target_chunks if c.chunk_id not in embedded_map]
    batch_size = 80  # Dưới ngưỡng 100 chunks/phút của Free Tier

    if missing_chunks:
        print(f"[*] Cần nhúng {len(missing_chunks)} chunks (~{(len(missing_chunks) + batch_size - 1) // batch_size} phút)...")
        for i in tqdm(range(0, len(missing_chunks), batch_size), desc="Batch Embedding"):
            batch = missing_chunks[i:i + batch_size]
            t_start = time.time()
            vecs = embedder.embed_chunks(batch)
            for c, v in zip(batch, vecs):
                embedded_map[c.chunk_id] = v

            np.savez_compressed(
                vectors_cache_path,
                chunk_ids=np.array(list(embedded_map.keys())),
                vectors=np.vstack(list(embedded_map.values())).astype(np.float32)
            )

            # BẮT BUỘC: Chờ cho hết cửa sổ 62 giây trước khi gửi batch 80 chunks tiếp theo
            if i + batch_size < len(missing_chunks):
                elapsed = time.time() - t_start
                sleep_needed = max(5.0, 62.0 - elapsed)
                time.sleep(sleep_needed)

    ready_chunks = [c for c in target_chunks if c.chunk_id in embedded_map]
    matrix = np.vstack([embedded_map[c.chunk_id] for c in ready_chunks]).astype(np.float32)
    dense_index.add_chunks(ready_chunks, matrix)
    return bm25_index, dense_index, all_chunks


# =====================================================================
# 5. CHẠY PIPELINE ĐÁNH GIÁ TOÀN DIỆN (RETRIEVAL + GENERATION)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Đánh giá hệ thống Custom-RAG trên 500 câu hỏi TriviaQA")
    parser.add_argument("--num-samples", type=int, default=500, help="Số lượng câu hỏi trích xuất (mặc định: 500)")
    parser.add_argument("--data-dir", type=str, default="triviaqa_eval_data", help="Thư mục lưu dữ liệu và cache")
    parser.add_argument("--bm25-only", action="store_true", help="Tắt nhánh Dense Vector để tiết kiệm 100% 500 requests/ngày cho LLM")
    parser.add_argument("--run-judge", action="store_true", help="Chạy thêm LLMGenerationJudge (tốn thêm 1 request / 5 câu)")
    parser.add_argument("--max-rpm", type=int, default=4, help="Giới hạn số request/phút cho Gemini Free Tier")
    args = parser.parse_args()

    # 1. Khởi tạo RAGConfig từ file config.py của bạn
    config = RAGConfig()
    #config.max_rpm = args.max_rpm  # Gắn thêm max_rpm cho GeminiLLMGenerator & SharedRateLimiter
    if args.bm25_only:
        config.weight_dense = 0.0
        config.weight_bm25 = 1.0

    # 2. Tải 500 câu hỏi TriviaQA & tạo corpus .md
    dataset_path, corpus_dir = prepare_triviaqa_subset(args.data_dir, num_samples=args.num_samples)
    with open(dataset_path, "r", encoding="utf-8") as f:
        triviaqa_items = json.load(f)["Data"]

    # 3. Khởi tạo Embedder + quản lý Cache cho Query Vector trên ổ cứng
    cache_dir = os.path.join(args.data_dir, "cache")
    query_cache_path = os.path.join(cache_dir, "query_embeddings.pkl")

    embedder = GeminiEmbedder(
        api_key=config.gemini_api_key,
        model_name=config.embedding_model,
        output_dimensionality=config.embedding_dim,
        task_prefix=config.embedding_task
    )
    if os.path.exists(query_cache_path):
        with open(query_cache_path, "rb") as f:
            embedder._query_cache = pickle.load(f)

    # 4. Xây dựng hoặc tải lại BM25Index & DenseFlatIndex
    bm25_index, dense_index, _ = build_or_load_indices(
        corpus_dir=corpus_dir,
        cache_dir=cache_dir,
        config=config,
        embedder=embedder,
        use_dense=(not args.bm25_only),
        questions=[it["Question"] for it in triviaqa_items]
    )

    # 5. Khởi tạo các tầng Retrieval, Post-Processor, Generator và Evaluator của bạn
    retriever = HybridRetriever(
        embedder=embedder,
        dense_index=dense_index,
        bm25_index=bm25_index,
        config=config
    )
    post_processor = RetrievalPostProcessor(config=config)
    prompt_builder = TriviaQAPromptBuilder()
    llm_generator = GeminiLLMGenerator(
        config=config,
        prompt_builder=prompt_builder,
        temperature=0.0,
        max_output_tokens=1024
    )
    retrieval_evaluator = RetrievalEvaluator()

    # File lưu tiến độ (Checkpoint) để nếu chạm 500 RPD thì hôm sau chạy tiếp không mất câu nào
    predictions_path = os.path.join(args.data_dir, "triviaqa_predictions.json")
    gen_cache_path = os.path.join(cache_dir, "generation_results_cache.pkl")

    predictions: Dict[str, str] = {}
    if os.path.exists(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    cached_gen_results: Dict[str, GenerationResult] = {}
    if os.path.exists(gen_cache_path):
        with open(gen_cache_path, "rb") as f:
            cached_gen_results = pickle.load(f)

    eval_samples: List[EvalSample] = []
    retrieved_results_per_sample: List[List[SearchResult]] = []
    gen_results_list: List[GenerationResult] = []

    print(f"\n[*] Bắt đầu chạy đánh giá trên {len(triviaqa_items)} câu hỏi TriviaQA...")
    quota_exhausted = False

    for item in tqdm(triviaqa_items, desc="RAG Evaluation"):
        qid = item["QuestionId"]
        query = item["Question"]
        ans_info = item["Answer"]

        # Tạo EvalSample tương thích với RetrievalEvaluator của bạn:
        # Dùng câu trả lời chuẩn (Value) làm expected_substrings để đo Hit_Rate@K, Recall@K, MRR@K, NDCG@K
        sample = EvalSample(
            sample_id=qid,
            query=query,
            ground_truth_chunk_ids=[],
            expected_substrings=[ans_info["Value"]],
            reference_answer=ans_info["Value"],
            expected_refusal=False
        )

        try:
            # Bước A: Truy xuất (Tầng 4)
            raw_search_results = retriever.retrieve(query, top_k=config.top_k_retrieval)
            final_search_results = post_processor.process(
                raw_search_results,
                enable_merging=True,
                enable_lost_in_middle=True
            )

            # Lưu lại query embedding cache để không tốn API nếu chạy lại
            if not args.bm25_only:
                with open(query_cache_path, "wb") as f:
                    pickle.dump(embedder._query_cache, f)

            # Bước B: Sinh câu trả lời bằng GeminiLLMGenerator (Tầng 5)
            if qid in cached_gen_results and qid in predictions:
                gen_res = cached_gen_results[qid]
            else:
                gen_res = llm_generator.generate(query, final_search_results)
                cached_gen_results[qid] = gen_res

                # Làm sạch thẻ trích dẫn [doc_c0] để lưu vào file predictions chuẩn của TriviaQA
                clean_pred = strip_citations_for_triviaqa(gen_res.answer)
                predictions[qid] = clean_pred

                # Ghi checkpoint ngay lập tức sau mỗi câu hỏi
                with open(predictions_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=2)
                with open(gen_cache_path, "wb") as f:
                    pickle.dump(cached_gen_results, f)

            eval_samples.append(sample)
            retrieved_results_per_sample.append(final_search_results)
            gen_results_list.append(gen_res)

        except Exception as e:
            print(f"\n[!] Dừng sớm tại câu hỏi '{qid}' do lỗi/chạm trần 500 RPD: {e}")
            print(f"[!] Đã lưu an toàn {len(predictions)} câu trả lời vào '{predictions_path}'.")
            quota_exhausted = True
            break

    if not eval_samples:
        print("[!] Chưa có mẫu nào hoàn tất để tính điểm.")
        return

    # =================================================================
    # 6. TỔNG HỢP BÁO CÁO ĐÁNH GIÁ (RETRIEVAL + TRIVIAQA EM/F1 + JUDGE)
    # =================================================================
    print("\n" + "=" * 65)
    print(f" BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CUSTOM-RAG ({len(eval_samples)}/{len(triviaqa_items)} CÂU)")
    print("=" * 65)

    # A. Chấm điểm Tầng Retrieval bằng RetrievalEvaluator của bạn (0 API call)
    ret_summary = retrieval_evaluator.evaluate_dataset(
        samples=eval_samples,
        results_per_sample=retrieved_results_per_sample,
        k=config.top_k_final
    )
    print("\n[1] CHỈ SỐ TRUY XUẤT (RETRIEVAL METRICS - TẦNG 4):")
    for metric_name, val in ret_summary.items():
        if metric_name != "details":
            print(f"  - {metric_name}: {val:.4f}")

    # B. Chấm điểm Exact Match (EM) và F1 chuẩn của mandarjoshi90/triviaqa (0 API call)
    em_total, f1_total = 0.0, 0.0
    item_by_qid = {it["QuestionId"]: it for it in triviaqa_items}

    for sample in eval_samples:
        qid = sample.sample_id
        pred_str = predictions.get(qid, "")
        gt_aliases = item_by_qid[qid]["Answer"]["Aliases"] + [item_by_qid[qid]["Answer"]["Value"]]

        em_total += metric_max_over_ground_truths(exact_match_score_triviaqa, pred_str, gt_aliases)
        f1_total += metric_max_over_ground_truths(f1_score_triviaqa, pred_str, gt_aliases)

    n_eval = len(eval_samples)
    print("\n[2] CHỈ SỐ TRIVIAQA OFFICIAL (GENERATION EM & F1 - TẦNG 5):")
    print(f"  - Exact Match (EM): {(em_total / n_eval) * 100:.2f}%")
    print(f"  - F1 Score        : {(f1_total / n_eval) * 100:.2f}%")

    # C. (Tùy chọn) Chấm điểm Faithfulness / Citation Precision bằng LLMGenerationJudge của bạn
    if args.run_judge and not quota_exhausted:
        print("\n[3] ĐANG CHẠY LLM-AS-A-JUDGE (THEO LÔ 5 CÂU / 1 REQUEST)...")
        judge = LLMGenerationJudge(config=config)
        judge_summary = judge.evaluate_dataset(eval_samples, gen_results_list)
        print(f"  - Mean Faithfulness      : {judge_summary['Mean_Faithfulness']:.4f}")
        print(f"  - Mean Answer Relevance  : {judge_summary['Mean_Answer_Relevance']:.4f}")
        print(f"  - Mean Citation Precision: {judge_summary['Mean_Citation_Precision']:.4f}")

    print("\n[*] File dự đoán chuẩn TriviaQA đã được xuất tại:", predictions_path)


if __name__ == "__main__":
    main()