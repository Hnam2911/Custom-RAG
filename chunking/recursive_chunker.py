import re
from typing import List

from matplotlib.pylab import block
from schema import RawBlock, Chunk

class ManualRecursiveChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
        self.active_headings = {}  # Lưu trạng thái tiêu đề tuần tự xuyên qua các Block

    def reset_state(self):
        self.active_headings = {}

    @staticmethod
    def _is_divider(text: str) -> bool:
        return text.strip() in ["---", "***", "___"]

    @staticmethod
    def _is_heading_or_intro(text: str) -> bool:
        s = text.strip()
        if not s:
            return False
        # Nhận diện Markdown heading (#), tiêu đề đánh số/Giai đoạn, hoặc câu dẫn kết thúc bằng ':'
        if re.match(r"^(#{1,6}\s+|Giai đoạn\s+\d+|Phần\s+\d+|Mục\s+\d+)", s, re.IGNORECASE):
            return True
        if s.endswith(":") and len(s) < 180 and "\n" not in s:
            return True
        return False
    
    @staticmethod
    def _extract_heading_level(line: str) -> tuple[int, str]:
        """Trả về (cấp_độ, nội_dung_tiêu_đề). Nếu không phải tiêu đề trả về (0, '')."""
        s = line.strip()
        m = re.match(r"^(#{1,6})\s+(.*)", s)
        if m:
            return len(m.group(1)), s
        return 0, ""

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        chosen_sep = separators[-1]
        next_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                next_separators = separators[i + 1:]
                break

        if chosen_sep == "":
            splits = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        else:
            raw_parts = text.split(chosen_sep)
            splits = [
                part + (chosen_sep if idx < len(raw_parts) - 1 else "")
                for idx, part in enumerate(raw_parts)
                if part
            ]

        final_atomic_splits: List[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                final_atomic_splits.append(s)
            else:
                final_atomic_splits.extend(self._split_text_recursive(s, next_separators))

        return final_atomic_splits

    def _merge_splits_with_overlap(self, atomic_splits: List[str]) -> List[str]:
        # Lọc bỏ các dòng kẻ '---' đứng riêng lẻ làm nhiễu ranh giới chunk
        cleaned_splits = [s for s in atomic_splits if not self._is_divider(s)]
        
        chunks: List[str] = []
        current_window: List[str] = []
        current_len = 0
        n = len(cleaned_splits)

        for i, split in enumerate(cleaned_splits):
            split_len = len(split)

            # QUY TẮC CHỐNG ORPHAN HEADING (Look-ahead 1 bước):
            # Nếu mảnh hiện tại là Tiêu đề/Câu dẫn, kiểm tra xem mảnh nội dung ngay sau nó
            # có còn vừa trong chunk hiện tại không. Nếu không vừa -> ngắt chunk ngay trước tiêu đề!
            next_len = len(cleaned_splits[i + 1]) if (i + 1 < n and self._is_heading_or_intro(split)) else 0
            
            if current_window and (current_len + split_len + next_len > self.chunk_size):
                # Trước khi xuất chunk, nếu phần tử cuối của current_window lỡ là tiêu đề mồ côi thì rút ra
                popped_orphans: List[str] = []
                while len(current_window) > 1 and self._is_heading_or_intro(current_window[-1]):
                    orphan = current_window.pop()
                    current_len -= len(orphan)
                    popped_orphans.insert(0, orphan)

                chunks.append("".join(current_window).strip())

                # Trượt cửa sổ overlap
                while current_window and current_len > self.chunk_overlap:
                    removed = current_window.pop(0)
                    current_len -= len(removed)

                # Nếu mảnh chuẩn bị thêm vào là một Heading mới (## hoặc ###),
                # xóa bỏ phần overlap thừa của mục cũ để chunk mới bắt đầu sạch sẽ từ Heading mới
                if self._is_heading_or_intro(split) and split.lstrip().startswith("#"):
                    current_window.clear()
                    current_len = 0

                # Đưa lại các tiêu đề mồ côi (nếu có) vào đầu cửa sổ mới
                for orphan in popped_orphans:
                    current_window.append(orphan)
                    current_len += len(orphan)

            current_window.append(split)
            current_len += split_len

        if current_window:
            last_chunk = "".join(current_window).strip()
            if last_chunk:
                chunks.append(last_chunk)

        return chunks

    def chunk_block(self, block: RawBlock, doc_id: str, start_seq: int = 0) -> List[Chunk]:
            atomic_splits = self._split_text_recursive(block.text, self.separators)
            merged_texts = self._merge_splits_with_overlap(atomic_splits)

            result_chunks: List[Chunk] = []
            search_cursor = 0
            
            # Theo dõi cây tiêu đề từ đầu block đến vị trí của chunk hiện tại
            for idx, content in enumerate(merged_texts):
                anchor = content[:min(40, len(content))]
                pos = block.text.find(anchor, search_cursor)
                if pos == -1:
                    pos = block.text.find(anchor)
                local_start = pos if pos != -1 else search_cursor

                tail_anchor = content[max(0, len(content) - 40):]
                tail_pos = block.text.find(tail_anchor, local_start)
                local_end = (tail_pos + len(tail_anchor)) if tail_pos != -1 else (local_start + len(content))
                search_cursor = max(search_cursor, local_start + 1)
                
                prev_scan_pos = 0
                for line in block.text[prev_scan_pos:local_start].splitlines():
                    lvl, h_text = self._extract_heading_level(line)
                    if lvl > 0:
                        self.active_headings[lvl] = h_text
                        for k in [k for k in self.active_headings if k > lvl]:
                            del self.active_headings[k]
                prev_scan_pos = local_start

                #Quét thêm dòng đầu tiên của chính chunk hiện tại để biết nó đang thuộc mục nào
                temp_headings = dict(self.active_headings)
                for line in content.splitlines():
                    lvl, h_text = self._extract_heading_level(line)
                    if lvl > 0:
                        temp_headings[lvl] = h_text
                        for k in [k for k in temp_headings if k > lvl]:
                            del temp_headings[k]

                meta = dict(block.metadata)
                if temp_headings:
                    meta["parent_heading"] = " > ".join(
                        h for _, h in sorted(temp_headings.items())
                    )
                    
                result_chunks.append(Chunk(
                    chunk_id=f"{doc_id}_c{start_seq + idx}",
                    doc_id=doc_id,
                    content=content,
                    block_type="text",
                    source_file=block.source_file,
                    page_number=block.page_number,
                    start_char_idx=block.base_char_idx + local_start,
                    end_char_idx=block.base_char_idx + local_end,
                    metadata=meta
                ))
            return result_chunks