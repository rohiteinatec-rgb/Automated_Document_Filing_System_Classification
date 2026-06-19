import re
import fitz
import pymupdf4llm
from rapidocr_onnxruntime import RapidOCR
from config import Config
from quality import QualityGate


class PDFReader:

    MAX_PAGES_TO_SCAN = 50       # Fast digital text extraction up to 50 pages
    MAX_OCR_PAGES = 3            # OCR is CPU heavy; strictly cap it to the first 3 pages
    MAX_IMAGE_DIMENSION = 4000   # Pixel limit to prevent OOM on massive blueprints
    METADATA_CHARS = 200

    # ── Line-item detection patterns ──────────────────────────────────
    LINE_ITEM_PATTERNS = [
        r'^\s*\d+[\s\t]+\S+.*\d+[.,]\d{2}\s*€?\s*$',
        r'^\s*\d+\s+[\d.,]+\s+[\d.,]+\s+[\d.,]+\s*$',
        r'^\|.*\|.*\|.*\|',
        r'^[-|]+$',
        r'^\s*Línia\s+Descripció',
        r'^\s*Línea\s+Descripción',
        r'^\s*Line\s+Description',
    ]

    # ── Junk metadata values to ignore ────────────────────────────────
    METADATA_JUNK = {
        "", "(anonymous)", "(unspecified)", "unknown", "n/a",
        "untitled", "microsoft word", "adobe", "none",
    }

    @classmethod
    def _contains_images(cls, doc: fitz.Document) -> bool:
        """Checks if the PDF contains any images, bounded to the page limit."""
        for i in range(min(doc.page_count, cls.MAX_PAGES_TO_SCAN)):
            page = doc[i]
            if page.get_images():
                return True
            # Check for Large Image-only pages (common in scans)
            if len(page.get_text("text").strip()) < 10 and page.get_pixmap():
                return True
        return False

    @classmethod
    def _ocr_extract(cls, doc: fitz.Document, debug: bool = False) -> str:
        """Renders PDF pages to images and runs RapidOCR with strict bounds."""
        engine = RapidOCR()
        ocr_text = []

        # Limit to MAX_OCR_PAGES to prevent massive CPU spikes
        for i in range(min(cls.MAX_OCR_PAGES, doc.page_count)):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            if pix.width > cls.MAX_IMAGE_DIMENSION or pix.height > cls.MAX_IMAGE_DIMENSION:
                if debug: print(f"  [Resource Guard] Skipping massive image page {i} to prevent OOM.")
                continue

            img_bytes = pix.tobytes("png")
            result, elapse = engine(img_bytes)

            if result:
                page_text = "\n".join([line[1] for line in result])
                ocr_text.append(page_text)

        return "\n".join(ocr_text)

    @classmethod
    def _is_line_item(cls, line: str) -> bool:
        return any(re.match(p, line) for p in cls.LINE_ITEM_PATTERNS)

    @classmethod
    def _extract_pdf_metadata(cls, doc: fitz.Document) -> str:
        """Free classification signal from PDF metadata. Junk values filtered."""
        try:
            meta  = doc.metadata or {}
            parts = []
            for key in ("title", "subject", "author", "keywords"):
                val = meta.get(key, "").strip()
                if val.lower() not in cls.METADATA_JUNK:
                    parts.append(f"{key}: {val}")
            return "\n".join(parts)
        except Exception:
            return ""

    @classmethod
    def _clean_text_extract(cls, doc: fitz.Document) -> str:
        """
        Extracts FULL continuous text up to MAX_PAGES_TO_SCAN.
        Replaces the legacy head/tail skipping logic so the AI sees everything.
        """
        sections = []

        for i in range(min(doc.page_count, cls.MAX_PAGES_TO_SCAN)):
            page_lines = doc[i].get_text("text").splitlines()
            # Strip noisy data to save LLM context window space, but keep all structure
            clean_lines = [l for l in page_lines if not cls._is_line_item(l)]
            sections.append(f"[PAGE {i+1}]\n" + "\n".join(clean_lines))

        metadata = cls._extract_pdf_metadata(doc)
        if metadata:
            sections.append(f"[PDF METADATA]\n{metadata[:cls.METADATA_CHARS]}")

        return "\n\n".join(sections)

    @classmethod
    def _sanitize_document_text(cls, text: str) -> str:
        """Removes invisible control characters and zero-width spaces."""
        if not text:
            return ""
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        return text.strip()

    @classmethod
    def extract_for_classification(cls, pdf_path: str, debug: bool = False) -> tuple[str, str]:
        gate = QualityGate(debug)

        try:
            # Determine safe page boundaries for this specific file
            with fitz.open(pdf_path) as doc_temp:
                safe_page_count = min(doc_temp.page_count, cls.MAX_PAGES_TO_SCAN)
            safe_pages = list(range(safe_page_count))

            # ── Tier 1: Markdown ──────────────────────────────────
            # If pymupdf4llm crashes, it will print a warning and naturally fall down to Tier 2.
            try:
                markdown_text = pymupdf4llm.to_markdown(pdf_path, pages=safe_pages)
                eval_result   = gate.evaluate(markdown_text)

                if eval_result.get("passed", False):
                    final = cls._sanitize_document_text(markdown_text)
                    if debug: cls._print_extraction_stats(final, "digital-markdown")
                    return final, "digital-markdown"

                if debug: print(f"  [Reader] Tier 1 weak ({eval_result.get('issues', 'Unknown')}). Falling back...")
            except Exception as tier1_error:
                if debug: print(f"  [Reader] ⚠️ Tier 1 (Markdown) crashed: {tier1_error}. Seamlessly falling back to Tier 2...")

            # ── Tier 2: Clean Sequential ──────────────────────────
            with fitz.open(pdf_path) as doc:
                # UPDATE 3: Wrapped Tier 2 in its own try/except block.
                try:
                    smart_text = cls._clean_text_extract(doc)
                    eval2 = gate.evaluate(smart_text)

                    if eval2.get("passed", False):
                        final = cls._sanitize_document_text(smart_text)
                        if debug: cls._print_extraction_stats(final, "clean-text")
                        return final, "clean-text"
                except Exception as tier2_error:
                    if debug: print(f"  [Reader] ⚠️ Tier 2 crashed: {tier2_error}. Seamlessly falling back to Tier 3...")

                # ── Tier 3: Raw fallback ──────────────────────────────
                if debug: print("  [Reader] Tier 2 weak/failed. Raw fallback...")

                try:
                    raw_text_parts = []
                    for i in range(safe_page_count):
                        raw_text_parts.append(doc[i].get_text("text"))
                    raw_text = "\n".join(raw_text_parts)

                    eval3 = gate.evaluate(raw_text)

                    if eval3.get("passed", False) or len(raw_text.strip()) > 50:
                        final = cls._sanitize_document_text(raw_text)
                        if debug: cls._print_extraction_stats(final, "raw-fallback")
                        return final, "raw-fallback"
                except Exception as tier3_error:
                    if debug: print(f"  [Reader] ⚠️ Tier 3 crashed: {tier3_error}. Seamlessly falling back to Tier 4...")

                if cls._contains_images(doc):
                    if debug: print("  [Reader] No text found, images detected. Triggering OCR...")

                    ocr_text = cls._ocr_extract(doc, debug=debug)

                    if len(ocr_text.strip()) > 20:
                        final = cls._sanitize_document_text(ocr_text)
                        if debug: cls._print_extraction_stats(final, "ocr-fallback")
                        return final, "ocr-fallback"
                    else:
                        if debug: print("  [Reader] OCR returned no meaningful text.")

            return "", "failed-all"

        except Exception as e:
            if debug: print(f"  [Reader] Fatal extraction error: {e}")
            return "", "error"

    @classmethod
    def _print_extraction_stats(cls, text: str, method: str):
        lines    = text.count('\n')
        chars    = len(text)
        print(f"  [Reader] Method={method} | {chars} chars | {lines} lines")
