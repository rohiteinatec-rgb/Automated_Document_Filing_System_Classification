"""
GESDOC Document Classifier
main.py — orchestrator and CLI entry point

USAGE:
  # Single PDF
  python main.py --pdf ..\\input\\invoice.pdf

  # Single PDF with debug output
  python main.py --pdf ..\\input\\invoice.pdf --debug

  # Process folder — all PDFs
  python main.py --folder ..\\input
"""

import os
import sys
import time
import argparse
import asyncio
from pathlib import Path
from errors import PDFProcessingError, AlertManager
from observability import ObservabilityManager
from database import DatabaseArchiver
from schemas import ProcessResult
from logger import adfs_logger
from config import Config
from reader import PDFReader
from classifier import Classifier
from filer import Filer

class DocumentAutoFiler:
    def __init__(self, debug: bool = False, dry_run: bool = False):
        self.debug = debug
        self.dry_run = dry_run
        self.classifier = Classifier(debug)
        self.filer = Filer(debug)

        self.db_archiver = DatabaseArchiver()
        self.task_queue = asyncio.Queue()
        self.results = [] # Store results for batch summary

    async def process_queue(self):
        """
        Background worker that processes PDFs from the queue sequentially.
        """
        print("  [SYSTEM] Background worker started. Processing tasks...")
        adfs_logger.info("Background queue worker started", extra={"stage": "queue"})
        while True:
            pdf_path = await self.task_queue.get()
            try:
                result = await self.process(pdf_path)
                self.results.append(result) # Save for the final CLI summary
                await asyncio.to_thread(self.db_archiver.archive_filing, result)

                if hasattr(self, 'callback') and self.callback:
                    if asyncio.iscoroutinefunction(self.callback):
                        await self.callback(result)
                    else:
                        self.callback(result)

            except PDFProcessingError as pe:
                self._handle_structured_error(pe)
                error_result = {"success": False, "file": pe.file_path, "action": "error", "message": str(pe)}
                self.results.append(error_result)
                await asyncio.to_thread(self.db_archiver.archive_filing, error_result)

            # Catch standard unexpected crashes
            except Exception as e:
                AlertManager.send_alert("UNHANDLED_CRASH", str(e), severity="CRITICAL")
                crash_result = {"success": False, "file": pdf_path, "action": "error", "message": "unhandled crash"}
                self.results.append(crash_result)
                await asyncio.to_thread(self.db_archiver.archive_filing, crash_result)
            finally:
                self.task_queue.task_done()

    def _handle_structured_error(self, error: PDFProcessingError):
        """Escalates errors based on their specific classification."""
        if error.error_type == PDFProcessingError.DISK_FULL:
            AlertManager.send_alert(error.error_type, error.args[0], severity="CRITICAL")
            # You could even write logic here to pause the queue until space is cleared!

        elif error.error_type == PDFProcessingError.PERMISSION_DENIED:
            AlertManager.send_alert(error.error_type, error.args[0], severity="CRITICAL")

        elif error.error_type == PDFProcessingError.CLASSIFICATION_TIMEOUT:
            AlertManager.send_alert(error.error_type, f"Ollama unresponsive for {error.file_path}", severity="HIGH")

        else:
            # For extraction failures or file collisions, a warning is fine.
            AlertManager.send_alert(error.error_type, f"{error.file_path} - {error.args[0]}", severity="WARNING")

    async def process(self, pdf_path: str) -> ProcessResult:
        t_total_start = time.time()
        pdf_path = str(pdf_path)
        fname = Path(pdf_path).name

        adfs_logger.info(f"Started processing document", extra={"stage": "extraction", "doc": fname})

        print(f"\n{'─'*55}")
        print(f"  📄 {fname}")
        print(f"{'─'*55}")

        # -----------------------------------------
        # Step 1: Read (Extraction)
        # -----------------------------------------
        t_ext_start = time.time()
        text, method = await asyncio.to_thread(PDFReader.extract_for_classification, pdf_path, self.debug)
        t_ext_ms = (time.time() - t_ext_start) * 1000

        print(f"\n  [DEBUG] The exact text fed to Qwen:\n  {repr(text[:300])}\n")

        if not text or len(text.strip()) < 20:
            # print(f"  ❌ Extraction failed (0 chars) — skipping")
            raise PDFProcessingError("Extraction failed (0 chars)", PDFProcessingError.EXTRACTION_FAILED, pdf_path)
            # return {"success": False, "file": fname, "action": "error", "message": "unreadable PDF"}

        if self.debug:
            print(f"  [Read] Method={method} | {len(text)} chars extracted")

        # -----------------------------------------
        # Step 2: Classify (LLM + ChromaDB)
        # -----------------------------------------
        t_cls_start = time.time()
        classification = await self.classifier.classify(text, fname)
        t_cls_ms = (time.time() - t_cls_start) * 1000
        classification["original_filename"] = fname
        # Quick sanitizer for conversational LLM outputs
        if len(classification.get("company", "")) > 30 or "not found" in classification.get("company", "").lower():
            classification["company"] = "Unknown"
        # -----------------------------------------
        # Step 3: File (Rename & Move)
        # -----------------------------------------
        t_file_start = time.time()
        if self.dry_run:
            new_filename = self.filer.build_new_filename(
                classification["tag"],
                classification.get("company", "unknown"),
                fname
            )
            t_file_ms = (time.time() - t_file_start) * 1000
            t_total_ms = (time.time() - t_total_start) * 1000

            print(f"\n  [DRY RUN — no files moved]")
            print(f"  Tag        : {classification['tag']}")
            print(f"  Confidence : {classification['confidence']:.2f}")
            print(f"  New name   : {new_filename}")

            self._print_metrics(t_ext_ms, t_cls_ms, t_file_ms, t_total_ms)
            return {"success": True, "file": fname, "action": "dry_run", "tag": classification["tag"], "company": classification.get("company", "Unknown")}

        # Actual filing
        result = await asyncio.to_thread(self.filer.file_document, pdf_path, classification)
        t_file_ms = (time.time() - t_file_start) * 1000
        t_total_ms = (time.time() - t_total_start) * 1000

        adfs_logger.info(
            f"Successfully filed as {classification['tag']}",
            extra={
                "stage": "filing",
                "doc": fname,
                "latency_ms": round(t_total_ms, 2)
            }
        )

        # Summary Output
        # print(f"\n  {result['message']}")
        print(f"\n  {result.get('message', 'File moved successfully')}")
        self._print_metrics(t_ext_ms, t_cls_ms, t_file_ms, t_total_ms)

        return result

    def _print_metrics(self, ext_ms, cls_ms, file_ms, total_ms):
        """Helper to print formatted timing metrics."""
        print(f"\n  ⏱️ PERFORMANCE METRICS:")
        print(f"    Extraction : {ext_ms:8.2f} ms")
        print(f"    Classifier : {cls_ms:8.2f} ms")
        print(f"    Filing     : {file_ms:8.2f} ms")
        print(f"    -------------------------")
        print(f"    TOTAL TIME : {total_ms:8.2f} ms")
        print(f"\n{'-'*55}")

async def process_folder(folder_path: str, debug: bool, dry_run: bool):
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))

    if not pdfs:
        print(f"[INFO] No PDFs found in: {folder_path}")
        return

    print(f"\n{'='*55}")
    print(f"  BATCH MODE — {len(pdfs)} PDF(s)")
    if dry_run: print(f"  Mode   : DRY RUN")
    print(f"{'='*55}")

    auto_filer = DocumentAutoFiler(debug, dry_run)
    if not auto_filer.classifier.metrics.check_health():
        print("\n  ❌ SYSTEM UNHEALTHY: Aborting batch process to prevent data loss.")
        return
    # 1. Start the background worker
    worker_task = asyncio.create_task(auto_filer.process_queue())

    # 2. Instantly queue all PDFs
    for pdf in pdfs:
        await auto_filer.task_queue.put(str(pdf))

    # 3. Wait for the queue to finish processing all items
    await auto_filer.task_queue.join()

    # 4. Cancel the worker so the script can finish
    worker_task.cancel()

    # results = [auto_filer.process(str(pdf)) for pdf in pdfs]
    results = auto_filer.results

    print(f"\n{'='*55}")
    print(f"  BATCH COMPLETE")
    print(f"  Filed     : {sum(1 for r in results if r.get('action') == 'filed')}")
    print(f"  Uncertain : {sum(1 for r in results if r.get('action') == 'uncertain')}")
    print(f"  Errors    : {sum(1 for r in results if r.get('action') == 'error')}")
    auto_filer.classifier.metrics.print_batch_sli_summary()
    print(f"{'='*55}")
    aggregator = MetricsAggregator()
    aggregator.generate_html_dashboard()

async def run():
    Config.validate_security_patterns()
    parser = argparse.ArgumentParser(description="Invoice Document Classifier")
    parser.add_argument("--pdf", default=None, help="Single PDF to classify")
    parser.add_argument("--folder", default=None, help="Process all PDFs in folder")
    parser.add_argument("--dry-run", action="store_true", help="Classify but do NOT move")
    parser.add_argument("--debug", action="store_true", help="Detailed output")
    args = parser.parse_args()

    if args.folder:
        await process_folder(args.folder, args.debug, args.dry_run)
        sys.exit(0)

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"[ERROR] File not found: {args.pdf}")
            sys.exit(1)
        auto_filer = DocumentAutoFiler(args.debug, args.dry_run)
        result = await auto_filer.process(args.pdf)
        sys.exit(0 if result.get("success") else 1)

    obs = ObservabilityManager()
    if not obs.check_health():
        print("  [System] 🛑 Health check failed. Aborting batch.")
        return

    parser.print_help()

if __name__ == "__main__":
    asyncio.run(run())