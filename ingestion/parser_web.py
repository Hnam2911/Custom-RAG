# ingestion/parser_web.py
import re
import html
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from schema import RawBlock

try:
    from bs4 import BeautifulSoup, Tag
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class WebScraperCleaner:
    """
    Bước 3 (Nhánh B): Tải song song U trang web và làm sạch HTML Boilerplate.
    Tái tạo cấu trúc Heading (# H1..H6) và bảng Markdown (| ... |) thành danh sách RawBlock
    để tương thích 100% với ManualRecursiveChunker và TableAwareChunker của hệ thống.
    """
    BOILERPLATE_TAGS = [
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "form", "svg", "iframe", "button", "input", "meta", "link"
    ]
    NOISE_CLASS_OR_ID_PATTERN = re.compile(
        r"(nav|navbar|footer|sidebar|menu|cookie|banner|popup|advert|ads|social|share|comment|breadcrumb)",
        re.IGNORECASE
    )
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    def __init__(self, timeout: float = 8.0, max_workers: int = 5):
        self.timeout = timeout
        self.max_workers = max_workers

    def scrape_urls_parallel(
        self,
        urls_with_snippets: List[Dict[str, str]]
    ) -> List[Tuple[str, List[RawBlock]]]:
        """
        Tải song song danh sách URLs.
        Mỗi phần tử đầu vào là dict: {"url": "...", "title": "...", "snippet": "..."}.
        Nếu 1 trang chặn bot (403/Timeout), tự động fallback về Title + Snippet của bộ máy tìm kiếm
        để không làm đứt gãy luồng dữ liệu.
        Trả về danh sách: [(url, [RawBlock, ...]), ...] theo đúng thứ tự xếp hạng của Macro Retriever.
        """
        if not urls_with_snippets:
            return []

        ordered_results: Dict[int, Tuple[str, List[RawBlock]]] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(urls_with_snippets))) as executor:
            future_to_idx = {
                executor.submit(self._fetch_and_parse_single, item): idx
                for idx, item in enumerate(urls_with_snippets)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                item = urls_with_snippets[idx]
                url = item.get("url", f"web_source_{idx}")
                try:
                    blocks = future.result()
                    if blocks:
                        ordered_results[idx] = (url, blocks)
                except Exception:
                    fallback_blocks = self._build_snippet_fallback(item)
                    if fallback_blocks:
                        ordered_results[idx] = (url, fallback_blocks)

        return [ordered_results[i] for i in sorted(ordered_results.keys())]

    def _fetch_and_parse_single(self, item: Dict[str, str]) -> List[RawBlock]:
        url = item.get("url", "").strip()
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()

        if not url:
            return self._build_snippet_fallback(item)

        try:
            resp = requests.get(url, headers=self.DEFAULT_HEADERS, timeout=self.timeout)
            content_type = resp.headers.get("Content-Type", "").lower()
            if resp.status_code != 200 or ("text/html" not in content_type and "application/xhtml" not in content_type):
                return self._build_snippet_fallback(item)

            resp.encoding = resp.apparent_encoding or "utf-8"
            raw_html = resp.text
        except requests.RequestException:
            return self._build_snippet_fallback(item)

        blocks = self.parse_html_to_blocks(raw_html, source_url=url, fallback_title=title)
        if not blocks:
            return self._build_snippet_fallback(item)
        return blocks

    def _build_snippet_fallback(self, item: Dict[str, str]) -> List[RawBlock]:
        """Fallback an toàn khi trang web dùng Cloudflare/JS chặn cào trực tiếp."""
        url = item.get("url", "unknown_url")
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        if not snippet and not title:
            return []

        heading_prefix = f"# {title}\n\n" if title else ""
        full_text = f"{heading_prefix}{snippet}".strip()
        return [
            RawBlock(
                text=full_text,
                block_type="text",
                source_file=url,
                page_number=None,
                base_char_idx=0,
                metadata={"initial_headings": {1: f"# {title}"} if title else {}, "is_snippet_fallback": True}
            )
        ]

    def parse_html_to_blocks(
        self,
        raw_html: str,
        source_url: str,
        fallback_title: str = ""
    ) -> List[RawBlock]:
        if HAS_BS4:
            return self._parse_with_bs4(raw_html, source_url, fallback_title)
        return self._parse_with_regex_fallback(raw_html, source_url, fallback_title)

    def _parse_with_bs4(self, raw_html: str, source_url: str, fallback_title: str) -> List[RawBlock]:
        soup = BeautifulSoup(raw_html, "html.parser")#type:ignore

        # 1. Trích xuất tiêu đề trang làm Heading cấp 1 (# H1)
        page_title = fallback_title
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()

        # 2. Loại bỏ các thẻ rác (Boilerplate tags)
        for tag in soup.find_all(self.BOILERPLATE_TAGS):
            tag.decompose()

        # 3. Loại bỏ các khối quảng cáo, menu, sidebar, footer theo class/id
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag) or tag.attrs is None:#type:ignore
                continue
            # Không xóa nhầm thẻ gốc chứa toàn bộ bài viết (body, main, article)
            if tag.name in ("html", "body", "main", "article"):
                continue
            attr_str = " ".join(
                str(tag.attrs.get("id", "")) + " " + " ".join(tag.attrs.get("class", []))
            )
            if self.NOISE_CLASS_OR_ID_PATTERN.search(attr_str):
                tag.decompose()

        main_container = soup.find("article") or soup.find("main") or soup.body or soup

        # 4. Chuyển đổi DOM sang chuỗi Markdown trung gian (giữ nguyên # Heading và bảng | ... |)
        md_lines: List[str] = []
        if page_title:
            clean_title = self._clean_inline_noise(page_title)
            md_lines.append(f"# {clean_title}\n")

        for elem in main_container.find_all(["h1", "h2", "h3", "h4", "table", "p", "li", "pre"]):
            # Bỏ qua các thẻ p/li nằm bên trong table để không nhân đôi dữ liệu bảng
            if elem.name in ("p", "li") and elem.find_parent("table") is not None:
                continue

            if elem.name in ("h1", "h2", "h3", "h4"):
                lvl = int(elem.name[1])
                # Nếu đã có page_title là # H1 thì đẩy h1 trong bài xuống ## H2 để giữ cây phả hệ chuẩn
                effective_lvl = min(lvl + 1, 6) if (page_title and lvl == 1) else lvl
                h_text = self._clean_inline_noise(elem.get_text(" ", strip=True))
                if h_text and len(h_text) < 200:
                    md_lines.append(f"\n{'#' * effective_lvl} {h_text}\n")

            elif elem.name == "table":
                md_table = self._html_table_to_markdown(elem)
                if md_table:
                    md_lines.append(f"\n{md_table}\n")

            elif elem.name == "li":
                li_text = self._clean_inline_noise(elem.get_text(" ", strip=True))
                if len(li_text) >= 15:
                    md_lines.append(f"- {li_text}")

            else:
                p_text = self._clean_inline_noise(elem.get_text(" ", strip=True))
                if len(p_text) >= 25:
                    md_lines.append(f"{p_text}\n")

        full_markdown = "\n".join(md_lines).strip()
        if not full_markdown:
            return []

        return self._split_markdown_into_raw_blocks(full_markdown, source_url)

    @staticmethod
    def _clean_inline_noise(text: str) -> str:
        """Làm sạch ký tự HTML entities, chú thích Wikipedia [1], [sửa] và khoảng trắng thừa."""
        s = html.unescape(text)
        s = re.sub(r"\[\s*(?:\d+|sửa|edit|citation needed|cần dẫn nguồn)\s*\]", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _html_table_to_markdown(self, table_tag) -> str:
        rows_data: List[List[str]] = []
        for tr in table_tag.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            row = [
                self._clean_inline_noise(c.get_text(" ", strip=True)).replace("|", "\\|")
                for c in cells
            ]
            if any(row):
                rows_data.append(row)

        if len(rows_data) < 2:
            return ""
        num_cols = max(len(r) for r in rows_data)
        if num_cols < 2:
            return ""

        # Chuẩn hóa số cột trên từng dòng
        normalized_rows = [r + [""] * (num_cols - len(r)) for r in rows_data]
        header = "| " + " | ".join(normalized_rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * num_cols) + " |"
        body = ["| " + " | ".join(r) + " |" for r in normalized_rows[1:]]
        return "\n".join([header, sep] + body)

    def _parse_with_regex_fallback(self, raw_html: str, source_url: str, fallback_title: str) -> List[RawBlock]:
        """Fallback bằng Regex thuần nếu môi trường chưa cài đặt BeautifulSoup4."""
        cleaned = re.sub(r"<(script|style|noscript|nav|footer|header|svg)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<h([1-4])[^>]*>(.*?)</h\1>", lambda m: f"\n{'#' * int(m.group(1))} {re.sub(r'<[^>]+>', '', m.group(2)).strip()}\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?(p|div|br|li|tr)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)

        lines = []
        if fallback_title:
            lines.append(f"# {self._clean_inline_noise(fallback_title)}")
        for line in cleaned.splitlines():
            s = self._clean_inline_noise(line)
            if len(s) >= 25 or s.startswith("#"):
                lines.append(s)

        full_markdown = "\n\n".join(lines).strip()
        if not full_markdown:
            return []
        return self._split_markdown_into_raw_blocks(full_markdown, source_url)

    @staticmethod
    def _split_markdown_into_raw_blocks(full_markdown: str, source_url: str) -> List[RawBlock]:
        """
        Tách văn bản Markdown của trang web thành các RawBlock (text / table)
        với bộ đếm base_char_idx và phả hệ initial_headings giống hệt ingestion/parser_text.py.
        """
        lines = full_markdown.splitlines(keepends=True)
        blocks: List[RawBlock] = []
        current_lines: List[str] = []
        current_type = "text"
        block_start_offset = 0
        cursor_offset = 0
        global_headings: Dict[int, str] = {}
        block_initial_headings: Dict[int, str] = {}

        def is_table_line(line: str) -> bool:
            s = line.strip()
            return s.startswith("|") and s.endswith("|") and len(s) > 2

        def update_headings(line: str, tracker: Dict[int, str]):
            m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
            if m:
                lvl = len(m.group(1))
                tracker[lvl] = line.strip()
                for k in [k for k in tracker if k > lvl]:
                    del tracker[k]

        def flush(buf: List[str], b_type: str, raw_start: int, init_h: Dict[int, str]):
            raw_str = "".join(buf)
            stripped = raw_str.strip()
            if not stripped:
                return
            leading_ws = len(raw_str) - len(raw_str.lstrip())
            blocks.append(RawBlock(
                text=stripped,
                block_type=b_type,
                source_file=source_url,
                page_number=None,
                base_char_idx=raw_start + leading_ws,
                metadata={"initial_headings": dict(init_h), "source_type": "web"}
            ))

        for line in lines:
            l_type = "table" if is_table_line(line) else "text"
            if l_type != current_type and current_lines:
                flush(current_lines, current_type, block_start_offset, block_initial_headings)
                for committed in current_lines:
                    update_headings(committed, global_headings)
                block_initial_headings = dict(global_headings)
                current_lines = []
                block_start_offset = cursor_offset

            current_type = l_type
            current_lines.append(line)
            cursor_offset += len(line)

        if current_lines:
            flush(current_lines, current_type, block_start_offset, block_initial_headings)

        return blocks