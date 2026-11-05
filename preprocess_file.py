import os
from typing import List
from config import RAGConfig
from schema import Chunk
from ingestion.parser_text import parse_text_or_markdown
from ingestion.parser_pdf import parse_pdf
from ingestion.parser_table import parse_tabular_file
from chunking.recursive_chunker import ManualRecursiveChunker
from chunking.table_chunker import TableAwareChunker
from pprint import pprint

def process_file_to_chunks(file_path: str, doc_id: str, config: RAGConfig) -> List[Chunk]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        raw_blocks = parse_text_or_markdown(file_path)
    elif ext == ".pdf":
        raw_blocks = parse_pdf(file_path)
    elif ext in [".csv", ".xlsx", ".xls"]:
        raw_blocks = parse_tabular_file(file_path)
    else:
        raise ValueError(f"Định dạng file chưa hỗ trợ: {ext}")

    text_chunker = ManualRecursiveChunker(config.chunk_size, config.chunk_overlap)
    table_chunker = TableAwareChunker(config.max_table_rows_per_chunk)

    all_chunks: List[Chunk] = []
    seq_counter = 0

    for block in raw_blocks:
        if block.block_type == "table":
            new_chunks = table_chunker.chunk_table_block(block, doc_id, seq_counter)
        else:
            new_chunks = text_chunker.chunk_block(block, doc_id, seq_counter)
        all_chunks.extend(new_chunks)
        seq_counter += len(new_chunks)

    return all_chunks

if __name__ == "__main__":
    file_path="C:/Users/Admin/Project/Custom-RAG/tailieu.md"
    config=RAGConfig()
    chunks=process_file_to_chunks(file_path=file_path,doc_id='01',config=config)
    for chunk in chunks:
        pprint(chunk)
    