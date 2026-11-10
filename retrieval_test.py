from config import RAGConfig
from indexing.gemini_embedder import GeminiEmbedder
from indexing.dense_flat_index import DenseFlatIndex
from indexing.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.post_processor import RetrievalPostProcessor
from preprocess_file import process_file_to_chunks

def test_tier4_retrieval(chunks, config: RAGConfig):
    # Khởi tạo theo đúng chữ ký mới của GeminiEmbedder
    embedder = GeminiEmbedder(
        config.gemini_api_key,
        config.embedding_model,
        config.embedding_dim,
        config.embedding_task
    )
    dense_index = DenseFlatIndex()
    bm25_index = BM25Index(k1=config.bm25_k1, b=config.bm25_b)

    # Nạp dữ liệu vào cả 2 Index
    emb_matrix = embedder.embed_chunks(chunks)
    dense_index.add_chunks(chunks, emb_matrix)
    bm25_index.add_chunks(chunks)

    retriever = HybridRetriever(embedder, dense_index, bm25_index, config)
    post_processor = RetrievalPostProcessor(config)

    # Câu hỏi kiểm thử: vừa hỏi lộ trình các giai đoạn (Mục 4), vừa hỏi chi phí DB-01 (Mục 3)
    query = "Tóm tắt các giai đoạn triển khai từ tháng 1 đến tháng 6 và chi phí của hạng mục DB-01?"
    raw_results = retriever.retrieve(query)

    print("=== 1. Trước khi qua Post-Processor (Raw RRF) ===")
    for r in raw_results:
        print(
            f"ID: {r.chunk.chunk_id:8s} | RRF: {r.score:.5f} | "
            f"DenseRank: {str(r.dense_rank):4s} | BM25Rank: {str(r.bm25_rank):4s} | "
            f"Offset: {r.chunk.start_char_idx}-{r.chunk.end_char_idx}"
        )

    final_results = post_processor.process(raw_results)

    print("\n=== 2. Sau khi qua Post-Processor (Đã lọc, gộp Chunk & đảo Lost-in-the-Middle) ===")
    for pos, r in enumerate(final_results, start=1):
        print(
            f"Vị trí Prompt #{pos} | ID: {r.chunk.chunk_id:12s} | RRF: {r.score:.5f} | "
            f"Offset: {r.chunk.start_char_idx}-{r.chunk.end_char_idx} | "
            f"Độ dài: {len(r.chunk.content)} chars"
        )
        
if __name__ == "__main__":
    # Khởi tạo cấu hình
    config = RAGConfig()
    chunks = process_file_to_chunks("tailieu.md",'01', config)
    test_tier4_retrieval(chunks, config)