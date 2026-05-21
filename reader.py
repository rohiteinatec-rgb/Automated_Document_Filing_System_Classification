# reader.py
import re
import fitz
import pymupdf4llm
from rapidocr_onnxruntime import RapidOCR
from config import Config
from quality import QualityGate


class PDFReader:

    # ── Tunable budgets ───────────────────────────────────────────────
    PAGE1_CHARS    = 800
    TAIL_CHARS     = 400
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
        """Checks if the PDF contains any images, indicating it might be a scan."""
        for page in doc:
            if page.get_images():
                return True
            # Check for Large Image-only pages (common in scans)
            if len(page.get_text("text").strip()) < 10 and page.get_pixmap():
                return True
        return False

    @classmethod
    def _ocr_extract(cls, doc: fitz.Document) -> str:
        """Renders PDF pages to images and runs RapidOCR."""
        engine = RapidOCR()
        ocr_text = []

        # Limit to first 2 pages for OCR to prevent massive CPU spikes
        for i in range(min(2, len(doc))):
            page = doc[i]
            # Render page to an image at 150 DPI (good balance of speed/accuracy)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            # ---> CHANGE 3: Convert PyMuPDF pixmap directly to PNG bytes
            # RapidOCR natively handles raw image bytes, no need for the PIL library
            img_bytes = pix.tobytes("png")

            # ---> CHANGE 4: Run RapidOCR and parse the tuple output
            # RapidOCR returns: result (list of lists) and elapse (time taken)
            result, elapse = engine(img_bytes)

            # ---> CHANGE 5: Extract just the text from the result payload
            # 'result' format is: [ [[box_coords], "Text line", confidence_score], ... ]
            if result:
                # Extract the string at index 1 for every line found
                page_text = "\n".join([line[1] for line in result])
                ocr_text.append(page_text)

        return "\n".join(ocr_text)

    @classmethod
    def _is_line_item(cls, line: str) -> bool:
        return any(re.match(p, line) for p in cls.LINE_ITEM_PATTERNS)

    @classmethod
    def _extract_pdf_metadata(cls, doc: fitz.Document) -> str:
        """
        Free classification signal from PDF metadata.
        Junk placeholder values are filtered out.
        """
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
    def _truncate_to_line(cls, text: str, max_chars: int) -> str:
        """
        Truncate at max_chars but snap back to the nearest complete
        line — never cut mid-word or mid-company-name.
        """
        if len(text) <= max_chars:
            return text
        cutoff = text.rfind('\n', 0, max_chars)
        return text[:cutoff] if cutoff > 0 else text[:max_chars]

    @classmethod
    def _page_aware_extract(cls, doc: fitz.Document) -> str:
        """
        Smart extraction:
          - Page 1        : full text, line items stripped, snapped to line (TOP)
          - Middle pages  : skipped entirely
          - Last page     : footer only, snap-forward from tail start
          - PDF metadata  : free signal, junk filtered (BOTTOM)
        """
        pages  = list(doc)
        total  = len(pages)
        sections = []

        # ── 1. Page 1 (This is the primary signal for ChromaDB) ─────
        page1_lines = pages[0].get_text("text").splitlines()
        clean_lines = [l for l in page1_lines if not cls._is_line_item(l)]
        page1_text  = cls._truncate_to_line(
            "\n".join(clean_lines), cls.PAGE1_CHARS
        )
        sections.append(f"[PAGE 1]\n{page1_text}")

        # ── 2. Middle pages: skipped ───────────────────────────────
        if total > 2:
            sections.append(f"[PAGES 2-{total - 1}: LINE ITEMS OMITTED]")

        # ── 3. Last page footer ────────────────────────────────────
        if total > 1:
            last_lines = pages[-1].get_text("text").splitlines()
            clean_last = [l for l in last_lines if not cls._is_line_item(l)]
            last_text  = "\n".join(clean_last)

            if len(last_text) > cls.TAIL_CHARS:
                tail_start = len(last_text) - cls.TAIL_CHARS
                snap       = last_text.find('\n', tail_start)
                tail       = last_text[snap + 1:] if snap > 0 \
                    else last_text[tail_start:]
            else:
                tail = last_text

            sections.append(f"[LAST PAGE FOOTER]\n{tail}")

        # ── 4. PDF metadata (MOVED TO BOTTOM) ──────────────────────
        # This prevents metadata from "poisoning" the top-500 char fingerprint
        metadata = cls._extract_pdf_metadata(doc)
        if metadata:
            sections.append(f"[PDF METADATA]\n{metadata[:cls.METADATA_CHARS]}")

        return "\n\n".join(sections)

    @classmethod
    def _sanitize_document_text(cls, text: str) -> str:
        """
        Production-grade text sanitizer.
        Removes invisible control characters and zero-width spaces used in OCR errors or obfuscation attacks.
        Avoids brittle regex whack-a-mole for English/Spanish injection phrases.
        """
        if not text:
            return ""

        # 1. Strip control characters (keep newlines and tabs)
        # This prevents terminal injection and layout manipulation
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # 2. Strip Zero-Width characters (Unicode obfuscation)
        # Attackers use these to hide instructions from regex but keep them visible to the LLM
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

        return text.strip()

    @classmethod
    def extract_for_classification(cls, pdf_path: str,
                                   debug: bool = False) -> tuple[str, str]:
        gate = QualityGate(debug)

        try:
            # ── Tier 1: Markdown ──────────────────────────────────
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            eval_result   = gate.evaluate(markdown_text)

            if eval_result["passed"]:
                final = cls._smart_truncate(markdown_text)
                if debug:
                    cls._print_extraction_stats(final, "digital-markdown")
                return cls._sanitize_document_text(final), "digital-markdown"

            if debug:
                print(f"  [Reader] Tier 1 weak "
                      f"({eval_result['issues']}). Falling back...")

            # ── Tier 2: Page-aware ────────────────────────────────
            doc        = fitz.open(pdf_path)
            smart_text = cls._page_aware_extract(doc)
            doc.close()

            eval2 = gate.evaluate(smart_text)
            if eval2["passed"]:
                if debug:
                    cls._print_extraction_stats(smart_text, "page-aware")
                return cls._sanitize_document_text(smart_text), "page-aware"

            # ── Tier 3: Raw fallback ──────────────────────────────
            if debug:
                print("  [Reader] Tier 2 weak. Raw fallback...")
            doc      = fitz.open(pdf_path)
            raw_text = "\n".join(p.get_text("text") for p in doc)
            # Evaluate Tier 3 before blindly returning it
            eval3 = gate.evaluate(raw_text)

            # If Tier 3 has decent text, return it
            if eval3["passed"] or len(raw_text.strip()) > 50:
                doc.close()
                final = cls._smart_truncate(raw_text)
                if debug:
                    cls._print_extraction_stats(final, "raw-fallback")
                return cls._sanitize_document_text(final), "raw-fallback"

            # ⭐ TIER 4: OCR (Last resort, very slow) ⭐
            # Trigger if all above fail AND image content detected
            if cls._contains_images(doc):
                if debug:
                    print("  [Reader] No text found, but images detected. Triggering OCR...")

                ocr_text = cls._ocr_extract(doc)
                doc.close()

                if len(ocr_text.strip()) > 20:
                    final = cls._smart_truncate(ocr_text)
                    if debug:
                        cls._print_extraction_stats(final, "ocr-fallback")
                    return cls._sanitize_document_text(final), "ocr-fallback"
                else:
                    if debug:
                        print("  [Reader] OCR returned no meaningful text.")
            else:
                doc.close()

            # If all 4 Tiers fail
            return "", "failed-all"

        except Exception as e:
            if debug:
                print(f"  [Reader] Extraction failed: {e}")
            return "", "error"

    @classmethod
    def _smart_truncate(cls, text: str) -> str:
        """Head + tail truncation for Tier 1 markdown path."""
        head_len = Config.CHARS_FOR_CLASSIFICATION
        tail_len = 500

        if len(text) <= (head_len + tail_len):
            return text

        # Use line-snap for head so we never cut mid-company-name
        head = cls._truncate_to_line(text, head_len)
        tail = text[-tail_len:]
        return f"{head}\n\n[MIDDLE OMITTED]\n\n{tail}"

    @classmethod
    def _print_extraction_stats(cls, text: str, method: str):
        lines    = text.count('\n')
        chars    = len(text)
        has_meta = "[PDF METADATA]" in text
        has_p1   = "[PAGE 1]"       in text
        has_tail = "[LAST PAGE"     in text
        print(f"  [Reader] Method={method} | {chars} chars | {lines} lines")
        print(f"  [Reader] Sections: metadata={has_meta} "
              f"page1={has_p1} footer={has_tail}")