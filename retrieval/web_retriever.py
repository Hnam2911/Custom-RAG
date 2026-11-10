# retrieval/web_retriever.py
import re
import urllib.parse
import requests
from typing import List, Dict, Optional
from config import RAGConfig
from schema import Chunk, SearchResult
from chunking.recursive_chunker import ManualRecursiveChunker
from chunking.table_chunker import TableAwareChunker
from indexing.gemini_embedder import GeminiEmbedder
from indexing.dense_flat_index import DenseFlatIndex
from indexing.bm25_index import BM25Index
from ingestion.parser_web import WebScraperCleaner
from retrieval.query_rewriter import QueryRewriter


class WebSearchRetriever:
    """
    Nhánh B (Online Web Search Pipeline):
    Query Rewriter -> Macro Retriever (Top-U URLs) -> Parallel Web Scraper & Cleaner
    -> Ephemeral In-Memory Chunking (N_temp chunks) -> Micro Retriever trên RAM (M_web chunks).
    """
    GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
    DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, embedder: GeminiEmbedder, config: RAGConfig):
        self.embedder = embedder
        self.config = config
        self.query_rewriter = QueryRewriter(config)
        self.scraper = WebScraperCleaner(
            timeout=config.web_scrape_timeout,
            max_workers=config.web_max_workers
        )
        self.text_chunker = ManualRecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.table_chunker = TableAwareChunker(
            max_rows_per_chunk=config.max_table_rows_per_chunk
        )
        self.session = requests.Session()

    def retrieve(self, query: str, top_m_web: Optional[int] = None) -> List[SearchResult]:
        """
        Thực thi toàn bộ 5 bước của Nhánh B và trả về M_web ứng viên SearchResult.
        """
        if not self.config.enable_web_search or not query.strip():
            return []

        m_web = top_m_web if top_m_web is not None else self.config.top_k_web_retrieval

        # BƯỚC 1: Query Rewriter (Chuyển câu hỏi dài q -> từ khóa tìm kiếm q_web)
        q_web = self.query_rewriter.rewrite(query)

        # BƯỚC 2: BỘ TRUY XUẤT 1 - MACRO RETRIEVER (Lấy Top-U URLs)
        top_u_items = self._macro_retrieve_urls(q_web, top_u=self.config.web_top_u_urls)
        if not top_u_items:
            return []

        # BƯỚC 3: Web Scraper & HTML Boilerplate Cleaner (Tải song song U trang)
        scraped_pages = self.scraper.scrape_urls_parallel(top_u_items)
        if not scraped_pages:
            return []

        # BƯỚC 4: Ephemeral In-Memory Chunking (Cắt U trang thành N_temp chunks trên RAM)
        ephemeral_chunks = self._ephemeral_chunk_pages(scraped_pages)
        if not ephemeral_chunks:
            return []

        # BƯỚC 5: BỘ TRUY XUẤT 2 - MICRO RETRIEVER (Chấm điểm N_temp chunks trên RAM lấy M_web ứng viên)
        return self._micro_retrieve_on_ram(query, ephemeral_chunks, top_k=m_web)

    def _macro_retrieve_urls(self, q_web: str, top_u: int = 5) -> List[Dict[str, str]]:
        """
        Ưu tiên gọi Google Custom Search JSON API (đúng theo sơ đồ).
        Nếu chưa cấu hình API Key / CSE ID của Google Search, tự động fallback sang DuckDuckGo HTML.
        """
        if self.config.google_search_api_key and self.config.google_cse_id:
            try:
                results = self._search_google_api(q_web, top_u)
                if results:
                    return results
            except Exception as e:
                print(f"[MacroRetriever Cảnh báo] Google Search API lỗi ({e}), chuyển sang DuckDuckGo fallback...")

        return self._search_duckduckgo_fallback(q_web, top_u)

    def _search_google_api(self, q_web: str, top_u: int) -> List[Dict[str, str]]:
        params = {
            "key": self.config.google_search_api_key,
            "cx": self.config.google_cse_id,
            "q": q_web,
            "num": min(max(top_u, 1), 10)
        }
        resp = self.session.get(self.GOOGLE_SEARCH_ENDPOINT, params=params, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        items = resp.json().get("items", [])
        results: List[Dict[str, str]] = []
        for item in items[:top_u]:
            link = item.get("link", "").strip()
            if link:
                results.append({
                    "url": link,
                    "title": item.get("title", "").strip(),
                    "snippet": item.get("snippet", "").strip()
                })
        return results

    def _search_duckduckgo_fallback(self, q_web: str, top_u: int) -> List[Dict[str, str]]:
        """Fallback miễn phí không cần API key khi chạy thử nghiệm cục bộ."""
        try:
            resp = self.session.post(
                self.DDG_HTML_ENDPOINT,
                data={"q": q_web},
                headers=WebScraperCleaner.DEFAULT_HEADERS,
                timeout=10
            )
            if resp.status_code != 200:
                return []

            html_text = resp.text
            # Bắt các thẻ kết quả <a class="result__a" href="...">Title</a> và snippet
            link_matches = re.findall(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                html_text,
                flags=re.DOTALL | re.IGNORECASE
            )
            snippet_matches = re.findall(
                r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                html_text,
                flags=re.DOTALL | re.IGNORECASE
            )

            results: List[Dict[str, str]] = []
            for idx, (raw_href, raw_title) in enumerate(link_matches):
                if len(results) >= top_u:
                    break
                url = self._clean_ddg_redirect_url(raw_href)
                if not url.startswith("http"):
                    continue
                clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                clean_snippet = (
                    re.sub(r"<[^>]+>", "", snippet_matches[idx]).strip()
                    if idx < len(snippet_matches) else ""
                )
                results.append({"url": url, "title": clean_title, "snippet": clean_snippet})
            return results
        except requests.RequestException:
            return []

    @staticmethod
    def _clean_ddg_redirect_url(raw_url: str) -> str:
        if "uddg=" in raw_url:
            parsed = urllib.parse.urlparse(raw_url)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return urllib.parse.unquote(qs["uddg"][0])
        return raw_url

    def _ephemeral_chunk_pages(self, scraped_pages: List[tuple]) -> List[Chunk]:
        """
        Bước 4: Cắt U trang web thành N_temp chunks trên RAM.
        Đảm bảo gọi reset_state() giữa mỗi trang web để không rò rỉ parent_heading.
        Phân bổ đồng đều hạn ngạch chunk giữa các URL để 1 trang dài không độc chiếm N_temp.
        """
        max_total = self.config.max_ephemeral_chunks
        max_per_page = max(10, max_total // max(len(scraped_pages), 1) + 5)
        ephemeral_chunks: List[Chunk] = []

        for url_idx, (url, raw_blocks) in enumerate(scraped_pages, start=1):
            # QUAN TRỌNG: Reset cây tiêu đề trước khi cắt trang web mới
            self.text_chunker.reset_state()
            # Đặt doc_id = "web01", "web02"... để sinh chunk_id = "web01_c0" khớp chuẩn Regex Citation
            doc_id = f"web{url_idx:02d}"
            seq_counter = 0
            page_chunks: List[Chunk] = []

            for block in raw_blocks:
                if block.block_type == "table":
                    new_chunks = self.table_chunker.chunk_table_block(block, doc_id, seq_counter)
                else:
                    new_chunks = self.text_chunker.chunk_block(block, doc_id, seq_counter)

                for c in new_chunks:
                    c.metadata["source_branch"] = "web"
                    c.metadata["url"] = url
                page_chunks.extend(new_chunks)
                seq_counter += len(new_chunks)

                if len(page_chunks) >= max_per_page:
                    page_chunks = page_chunks[:max_per_page]
                    break

            ephemeral_chunks.extend(page_chunks)
            if len(ephemeral_chunks) >= max_total:
                ephemeral_chunks = ephemeral_chunks[:max_total]
                break

        return ephemeral_chunks

    def _micro_retrieve_on_ram(
        self,
        original_query: str,
        ephemeral_chunks: List[Chunk],
        top_k: int
    ) -> List[SearchResult]:
        """
        Bước 5: Chấm điểm N_temp chunks bằng Bi-Encoder (DenseFlatIndex) + BM25 tạm thời trên RAM
        so với câu truy vấn gốc q, lấy ra M_web ứng viên tốt nhất rồi giải phóng Index tạm.
        """
        temp_dense_index = DenseFlatIndex()
        temp_bm25_index = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)

        try:
            # 1. Đánh chỉ mục BM25 tạm thời trên RAM
            temp_bm25_index.add_chunks(ephemeral_chunks)
            bm25_hits = temp_bm25_index.search(original_query, top_k=top_k)

            # 2. Nhúng N_temp chunks bằng Bi-Encoder (Tối đa 100 chunks = 1 batch request duy nhất)
            dense_hits = []
            if self.config.weight_dense > 0:
                embeddings = self.embedder.embed_chunks(ephemeral_chunks)
                temp_dense_index.add_chunks(ephemeral_chunks, embeddings)

                # embed_query(original_query) tự động lấy từ _query_cache nếu Nhánh A đã nhúng q!
                q_vec = self.embedder.embed_query(original_query)
                raw_dense_hits = temp_dense_index.search(q_vec, top_k=top_k)

                re_rank = 1
                for chunk, cos_score, _ in raw_dense_hits:
                    if cos_score >= self.config.min_dense_cosine:
                        dense_hits.append((chunk, cos_score, re_rank))
                        re_rank += 1

            # 3. Hợp nhất điểm trong nội bộ Nhánh B theo chuẩn RRF của hệ thống
            fused_map: Dict[str, SearchResult] = {}
            rrf_k = self.config.rrf_k
            w_dense = self.config.weight_dense
            w_bm25 = self.config.weight_bm25

            for chunk, d_score, d_rank in dense_hits:
                cid = chunk.chunk_id
                fused_map[cid] = SearchResult(
                    chunk=chunk,
                    score=w_dense / (rrf_k + d_rank),
                    dense_rank=d_rank,
                    bm25_rank=None,
                    dense_score=d_score,
                    bm25_score=None,
                    source_branch="web"
                )

            for chunk, b_score, b_rank in bm25_hits:
                cid = chunk.chunk_id
                rrf_contrib = w_bm25 / (rrf_k + b_rank)
                if cid in fused_map:
                    existing = fused_map[cid]
                    existing.score += rrf_contrib
                    existing.bm25_rank = b_rank
                    existing.bm25_score = b_score
                else:
                    fused_map[cid] = SearchResult(
                        chunk=chunk,
                        score=rrf_contrib,
                        dense_rank=None,
                        bm25_rank=b_rank,
                        dense_score=None,
                        bm25_score=b_score,
                        source_branch="web"
                    )

            results = list(fused_map.values())
            results.sort(
                key=lambda r: (
                    round(r.score, 8),
                    1 if (r.dense_rank is not None and r.bm25_rank is not None) else 0,
                    r.dense_score if r.dense_score is not None else -1.0,
                    r.bm25_score if r.bm25_score is not None else -1.0
                ),
                reverse=True
            )
            return results[:top_k]

        finally:
            # Giải phóng hoàn toàn bộ nhớ RAM tạm thời ngay sau khi lấy được M_web ứng viên
            temp_dense_index.clear()
            temp_bm25_index.clear()