import os
import pandas as pd
from typing import List
from schema import RawBlock

def parse_tabular_file(file_path: str) -> List[RawBlock]:
    """Đọc CSV hoặc Excel, chuyển từng sheet thành chuỗi bảng Markdown chuẩn."""
    source_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    blocks: List[RawBlock] = []

    if ext == ".csv":
        df = pd.read_csv(file_path).fillna("")
        md_table = dataframe_to_markdown(df)
        blocks.append(RawBlock(
            text=md_table,
            block_type="table",
            source_file=source_name,
            page_number=1,
            metadata={"sheet_name": "default"}
        ))
    elif ext in [".xlsx", ".xls"]:
        excel_file = pd.ExcelFile(file_path)
        for idx, sheet in enumerate(excel_file.sheet_names, start=1):
            df = excel_file.parse(sheet).fillna("")
            if not df.empty:
                md_table = dataframe_to_markdown(df)
                blocks.append(RawBlock(
                    text=md_table,
                    block_type="table",
                    source_file=source_name,
                    page_number=idx,
                    metadata={"sheet_name": sheet}
                ))
    return blocks

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame thành bảng Markdown mà không cần cài thêm thư viện tabulate."""
    headers = [str(col).replace("|", "\\|").strip() for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    row_lines = []
    for _, row in df.iterrows():
        cells = [str(val).replace("\n", " ").replace("|", "\\|").strip() for val in row]
        row_lines.append("| " + " | ".join(cells) + " |")
        
    return "\n".join([header_line, separator_line] + row_lines)