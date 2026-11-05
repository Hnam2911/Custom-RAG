import os
import re
from typing import List, Dict
from schema import RawBlock

def parse_text_or_markdown(file_path: str) -> List[RawBlock]:
    """Đọc file TXT/MD, bảo toàn offset toàn cục, gắn tiêu đề bảng và truyền phả hệ Heading xuyên qua các Block."""
    source_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    lines = full_text.splitlines(keepends=True)

    blocks: List[RawBlock] = []
    current_lines: List[str] = []
    current_type = "text"
    block_start_offset = 0
    cursor_offset = 0

    # Theo dõi cây tiêu đề toàn cục xuyên suốt cả file
    global_headings: Dict[int, str] = {}
    block_initial_headings: Dict[int, str] = {}

    def is_markdown_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2

    def is_heading_or_intro(line: str) -> bool:
        stripped = line.strip()
        return bool(re.match(r"^#{1,6}\s+", stripped)) or stripped.endswith(":")

    def update_heading_tracker(line: str, tracker: Dict[int, str]):
        m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if m:
            lvl = len(m.group(1))
            tracker[lvl] = line.strip()
            for k in list(tracker.keys()):
                if k > lvl:
                    del tracker[k]

    def flush_block(lines_buf: List[str], b_type: str, raw_start: int, init_headings: Dict[int, str]):
        raw_str = "".join(lines_buf)
        stripped_str = raw_str.strip()
        if not stripped_str:
            return
        leading_ws = len(raw_str) - len(raw_str.lstrip())
        blocks.append(RawBlock(
            text=stripped_str,
            block_type=b_type,
            source_file=source_name,
            page_number=1,
            base_char_idx=raw_start + leading_ws,
            metadata={"initial_headings": dict(init_headings)}
        ))

    for line in lines:
        line_type = "table" if is_markdown_table_line(line) else "text"

        if line_type != current_type and current_lines:
            transferred_lines: List[str] = []
            if current_type == "text" and line_type == "table":
                temp_trailing = []
                while current_lines and current_lines[-1].strip() == "":
                    temp_trailing.insert(0, current_lines.pop())
                if current_lines and is_heading_or_intro(current_lines[-1]):
                    transferred_lines = [current_lines.pop()] + temp_trailing
                else:
                    current_lines.extend(temp_trailing)

            while current_lines and current_lines[-1].strip() in ["", "---", "***"]:
                current_lines.pop()

            flush_block(current_lines, current_type, block_start_offset, block_initial_headings)

            # Cập nhật lại global_headings từ các dòng đã đóng gói vào block vừa rồi
            for committed_line in current_lines:
                update_heading_tracker(committed_line, global_headings)

            # Khởi tạo trạng thái tiêu đề kế thừa cho block mới
            block_initial_headings = dict(global_headings)

            transferred_len = sum(len(x) for x in transferred_lines)
            current_lines = transferred_lines
            block_start_offset = cursor_offset - transferred_len

        current_type = line_type
        current_lines.append(line)
        cursor_offset += len(line)

    if current_lines:
        flush_block(current_lines, current_type, block_start_offset, block_initial_headings)

    return blocks