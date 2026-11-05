from typing import List
from schema import RawBlock, Chunk

class TableAwareChunker:
    def __init__(self, max_rows_per_chunk: int = 20):
        self.max_rows = max_rows_per_chunk

    def chunk_table_block(self, block: RawBlock, doc_id: str, start_seq: int = 0) -> List[Chunk]:
        lines = [line.strip() for line in block.text.strip().split("\n") if line.strip()]
        
        # Tách các dòng tiêu đề đứng trước bảng (không bắt đầu bằng '|') và các dòng nội dung bảng
        title_lines = [line for line in lines if not line.startswith("|")]
        table_lines = [line for line in lines if line.startswith("|")]

        if len(table_lines) <= 2:
            return [Chunk(
                chunk_id=f"{doc_id}_c{start_seq}",
                doc_id=doc_id,
                content=block.text,
                block_type="table",
                source_file=block.source_file,
                page_number=block.page_number,
                start_char_idx=block.base_char_idx,
                end_char_idx=block.base_char_idx + len(block.text),
                metadata=dict(block.metadata)
            )]

        header_lines = table_lines[:2]  # Dòng tên cột và dòng | --- |
        data_rows = table_lines[2:]

        chunks: List[Chunk] = []
        chunk_idx = 0
        search_cursor = 0

        for i in range(0, len(data_rows), self.max_rows):
            batch_rows = data_rows[i:i + self.max_rows]
            # Luôn gắn: Tiêu đề bảng (nếu có) + Header cột + Các dòng dữ liệu
            table_piece = "\n".join(title_lines + header_lines + batch_rows)
            
            # Tính vị trí ký tự thực tế trên tài liệu gốc
            first_row_pos = block.text.find(batch_rows[0], search_cursor)
            last_row_pos = block.text.find(batch_rows[-1], first_row_pos if first_row_pos != -1 else search_cursor)
            
            local_start = 0 if i == 0 else (first_row_pos if first_row_pos != -1 else search_cursor)
            local_end = (last_row_pos + len(batch_rows[-1])) if last_row_pos != -1 else len(block.text)
            search_cursor = local_end
            
            init_headings = block.metadata.get("initial_headings", {})
            # Bảng đang có tiêu đề ## 3 (cấp 2), nên chỉ kế thừa tiêu đề cấp 1 (# H1)
            parent_headers = [h for lvl, h in sorted(init_headings.items()) if lvl < 2 and h not in table_piece]
            meta = {k: v for k, v in block.metadata.items() if k != "initial_headings"}
            meta["row_range"] = f"{i + 1}-{i + len(batch_rows)}"
            if parent_headers:
                meta["parent_heading"] = " > ".join(parent_headers)

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{start_seq + chunk_idx}",
                doc_id=doc_id,
                content=table_piece,
                block_type="table",
                source_file=block.source_file,
                page_number=block.page_number,
                start_char_idx=block.base_char_idx + local_start,
                end_char_idx=block.base_char_idx + local_end,
                metadata=meta
            ))
            chunk_idx += 1

        return chunks