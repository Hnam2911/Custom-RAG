import os
import re
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Tuple
from schema import RawBlock


def _is_valid_markdown_table(extracted_rows: List[List[str]]) -> bool:
    """
    Quy tắc phổ quát cho mọi bảng dữ liệu:
    1. Phải có ít nhất 2 dòng và ít nhất 2 cột (loại bỏ các hộp hình vẽ 1 cột hoặc cột số bị cắt lẻ).
    2. Dòng Header (dòng đầu) phải chứa chữ cái (không nhận bảng chỉ toàn số trơ trọi).
    """
    if not extracted_rows or len(extracted_rows) < 2:
        return False
    num_cols = len(extracted_rows[0])
    if num_cols < 2:
        return False

    # Dòng đầu tiên (Header) phải có ít nhất 3 ký tự chữ cái
    header_text = " ".join(str(c or "") for c in extracted_rows[0])
    if len(re.findall(r"[a-zA-ZÀ-ỹ]", header_text)) < 3:
        return False

    # Tối thiểu 30% số ô trong bảng phải có dữ liệu
    total_cells = len(extracted_rows) * num_cols
    non_empty = sum(1 for row in extracted_rows for c in row if c and str(c).strip())
    return (non_empty / total_cells) >= 0.3


def _is_noise_block(text: str, y0: float, y1: float, page_height: float) -> bool:
    """
    Lọc 3 loại nhiễu phổ biến nhất trong PDF bằng quy tắc đơn giản:
    1. Số trang ở mép trên/dưới trang.
    2. Các cụm nhãn trục biểu đồ (chỉ toàn các số ngắn hoặc ký tự rời rạc như '10 8 6 4 2 0').
    3. Dòng Caption/Footnote ngắn nằm lơ lửng nếu muốn tách riêng.
    """
    s = text.strip()
    if not s:
        return True
    # 1. Số trang ở 8% đầu hoặc cuối trang
    if re.match(r"^\d{1,6}$", s) and (y0 < page_height * 0.08 or y1 > page_height * 0.92):
        return True
    # 2. Khối rác từ trục biểu đồ: tỷ lệ chữ cái < 20% và không phải câu văn hoàn chỉnh
    letters = len(re.findall(r"[a-zA-ZÀ-ỹ]", s))
    if len(s) < 40 and letters < 3:
        return True
    return False

def _sort_blocks_multi_column(blocks: List[Tuple[float, float, float, float, str]], page_width: float) -> List[str]:
    """
    Thuật toán sắp xếp thứ tự đọc (Reading Order) cho cả trang 1 cột và 2 cột:
    Mỗi phần tử trong blocks là: (x0, y0, x1, y1, text)
    """
    mid_x = page_width / 2.0
    margin_tol = page_width * 0.04  # Dung sai 4% độ rộng trang ở vùng ranh giới giữa 2 cột

    # Chia làm các dải (bands) theo chiều dọc: mỗi khi gặp 1 block trải ngang cả 2 cột (Full-width
    # như Title, Abstract, Section lớn), ta chốt và gomcột trái + cột phải phía trên nó trước.
    sorted_texts: List[str] = []
    left_col: List[Tuple[float, float, str]] = []
    right_col: List[Tuple[float, float, str]] = []

    def flush_columns():
        nonlocal left_col, right_col
        if left_col:
            left_col.sort(key=lambda b: (round(b[0], 1), b[1]))  # Sort theo y0, rồi x0
            sorted_texts.extend([b[2] for b in left_col])
            left_col = []
        if right_col:
            right_col.sort(key=lambda b: (round(b[0], 1), b[1]))
            sorted_texts.extend([b[2] for b in right_col])
            right_col = []

    # Sắp xếp sơ bộ từ trên xuống dưới theo y0 để duyệt tuần tự
    blocks_by_y = sorted(blocks, key=lambda b: b[1])

    for (x0, y0, x1, y1, text) in blocks_by_y:
        block_width = x1 - x0
        # 1. Thuộc cột trái: nằm bên trái đường chia giữa (mid_x + margin_tol)
        if x1 <= mid_x + margin_tol:
            left_col.append((y0, x0, text))
        # 2. Thuộc cột phải: bắt đầu từ bên phải đường chia giữa (mid_x - margin_tol)
        elif x0 >= mid_x - margin_tol:
            right_col.append((y0, x0, text))
        # 3. Block trải ngang cả 2 cột (Full-width: Title, Abstract, Figure caption lớn)
        else:
            # Nếu block full-width nằm thấp hơn các cột đang tích lũy -> xả 2 cột trước
            flush_columns()
            sorted_texts.append(text)

    flush_columns()
    return sorted_texts

def parse_pdf(file_path: str, skip_references: bool = True) -> List[RawBlock]:
    source_name = os.path.basename(file_path)
    blocks: List[RawBlock] = []
    global_char_cursor = 0
    reached_references = False

    doc_fitz = fitz.open(file_path)
    pdf_plumber = pdfplumber.open(file_path)

    try:
        for page_idx in range(len(doc_fitz)):
            if reached_references and skip_references:
                break

            page_num = page_idx + 1
            fitz_page = doc_fitz[page_idx]
            plumber_page = pdf_plumber.pages[page_idx]
            page_w, page_h = float(fitz_page.rect.width), float(fitz_page.rect.height)

            # 1. Tách bảng hợp lệ (>= 2 cột, có header chữ)
            valid_tables: List[Tuple[Tuple[float, float, float, float], str]] = []
            for t in plumber_page.find_tables():
                rows = t.extract()
                if not _is_valid_markdown_table(rows):#type:ignore
                    continue
                md_lines = []
                for r_idx, row in enumerate(rows):
                    cleaned = [str(c or "").replace("\n", " ").strip() for c in row]
                    md_lines.append("| " + " | ".join(cleaned) + " |")
                    if r_idx == 0:
                        md_lines.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
                valid_tables.append((t.bbox, "\n".join(md_lines)))

            table_bboxes = [bbox for bbox, _ in valid_tables]

            # 2. Đọc các block chữ ngoài vùng bảng và lọc rác biểu đồ
            raw_blocks = fitz_page.get_text("blocks", sort=False)
            body_blocks = []
            floating_captions = []  # Tách riêng Figure/Table caption để không cắt ngang câu

            for b in raw_blocks:
                x0, y0, x1, y1, raw_text, _, b_type = b
                if b_type != 0:
                    continue
                # Bỏ qua nếu nằm trong vùng bảng đã trích xuất
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0#type:ignore
                if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for (bx0, by0, bx1, by1) in table_bboxes):
                    continue
                if _is_noise_block(raw_text, y0, y1, page_h):#type:ignore
                    continue

                # Làm sạch dòng trong block
                clean_txt = " ".join(line.strip() for line in raw_text.splitlines() if line.strip())

                # Nếu là Caption (Figure X:, Table X:) hoặc Footnote cuối trang -> gom riêng xuống cuối trang
                if re.match(r"^(Figure|Table|Algorithm)\s+\d+[:\.]", clean_txt) or (
                    y0 > page_h * 0.85 and re.match(r"^\d+[A-Z]", clean_txt)#type:ignore
                ):
                    floating_captions.append(clean_txt)
                else:
                    body_blocks.append((x0, y0, x1, y1, clean_txt))

            # 3. Sắp xếp 2 cột (Cột trái -> Cột phải)
            ordered_paras = _sort_blocks_multi_column(body_blocks, page_w)

            # 4. Đánh dấu tiêu đề mục (vd: "2 Related Work", "4.1 Overview") và kiểm tra References
            final_paras = []
            for p in ordered_paras:
                if skip_references and re.match(r"^(\d+\.?\s+)?References\b", p, re.IGNORECASE) and len(p) < 35:
                    reached_references = True
                    break
                if re.match(r"^(\d+(\.\d+)*)\s+[A-Z][A-Za-z0-9\s\-:]{2,55}$", p):
                    lvl = p.split()[0].count(".") + 2
                    p = f"{'#' * min(lvl, 4)} {p}"
                final_paras.append(p)

            # Đưa các Caption xuống cuối trang để không bao giờ cắt đôi câu văn chính
            final_paras.extend(floating_captions)

            if final_paras:
                page_text = "\n\n".join(final_paras)
                # Nối các từ bị gạch nối ngắt dòng (kể cả ngắt qua 2 đoạn/cột khác nhau '\n\n')
                page_text = re.sub(r"([a-zA-Z]{2,})-\s+\n*\s*([a-z]{2,})", r"\1\2", page_text)

                blocks.append(RawBlock(
                    text=page_text,
                    block_type="text",
                    source_file=source_name,
                    page_number=page_num,
                    base_char_idx=global_char_cursor
                ))
                global_char_cursor += len(page_text) + 2

            for t_idx, (_, md_table) in enumerate(valid_tables):
                blocks.append(RawBlock(
                    text=md_table,
                    block_type="table",
                    source_file=source_name,
                    page_number=page_num,
                    base_char_idx=global_char_cursor,
                    metadata={"table_index": t_idx}
                ))
                global_char_cursor += len(md_table) + 2

    finally:
        doc_fitz.close()
        pdf_plumber.close()

    return blocks