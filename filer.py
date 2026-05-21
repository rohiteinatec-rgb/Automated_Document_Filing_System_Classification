import os
import shutil
import re
import json
import unicodedata
import secrets
import errno
from errors import PDFProcessingError
from pathlib import Path
from datetime import datetime
from config import Config


class Filer:

    def __init__(self, debug: bool = False):
        self.debug = debug

    # ─────────────────────────────────────────────────────────────────
    # Filename builder  →  {tag}_{company}_{original}_{datetime}.pdf
    # ─────────────────────────────────────────────────────────────────
    def build_new_filename(self, tag: str, company: str,
                           original_filename: str) -> str:

        clean_stem = self._strip_existing_tag(original_filename)
        ext        = Path(original_filename).suffix.lower() or ".pdf"
        dt_stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

        tag     = Filer.sanitise(tag.lower())
        company = Filer.sanitise(company)

        # Format: {tag}_{company}_{original}_{datetime}
        if company and company.lower() not in ("unknown", ""):
            new_stem = f"{tag}_{company}_{clean_stem}_{dt_stamp}"
        else:
        # No company extracted — keep format consistent, use placeholder
            new_stem = f"{tag}_unknown_{clean_stem}_{dt_stamp}"

        # Remove forbidden characters
        for ch in Config.FILENAME_FORBIDDEN_CHARS:
            new_stem = new_stem.replace(ch, "")

        while "__" in new_stem:
            new_stem = new_stem.replace("__", "_")

        return new_stem + ext

    @staticmethod
    def sanitise(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r'[<>:"/\\|?*,.]', '_', s).strip('_')  # added comma and dot
        s = re.sub(r'\s+', '_', s)
        while '__' in s:
            s = s.replace('__', '_')
        return s

    def _strip_existing_tag(self, filename: str) -> str:
        """Prevent double-tagging: invoice_test.pdf → test"""
        stem = Path(filename).stem
        for prefix in Config.KNOWN_TAG_PREFIXES:
            if stem.lower().startswith(prefix + "_"):
                return stem[len(prefix) + 1:]
        return stem

    # ─────────────────────────────────────────────────────────────────
    # Main filing method
    # ─────────────────────────────────────────────────────────────────
    def file_document(self, source_path: str, classification: dict) -> dict:
        source = Path(source_path)
        if not source.exists():
            return self._result(False, source_path, None,
                                "error", "Source file not found")

        # 1. Extract core variables
        tag               = classification.get("tag", "uncertain")
        company           = classification.get("company", "Unknown")
        confidence        = float(classification.get("confidence", 0.0))
        original_filename = classification.get("original_filename", source.name)

        # 🔴 NEW: The Operational Logic Gate
        # If confidence is low, or the AI explicitly rejected the file via the prompt rules
        needs_review = (
                confidence < 0.75 or
                tag.lower() == "uncertain" or
                company.lower() in ("unknown", "")
        )

        # Build new filename with datetime stamp
        new_filename  = self.build_new_filename(tag, company, original_filename)

        # 🔴 NEW: Dynamic Routing
        if needs_review:
            target_folder = Path(Config.OUTPUT_ROOT) / "_HUMAN_REVIEW"
            action = "review"
        else:
            # Enforce the rigorous Company/Tag taxonomy for clean files
            clean_company = self.sanitise(company)
            clean_tag = self.sanitise(tag)
            target_folder = Path(Config.OUTPUT_ROOT) / clean_company / clean_tag
            action = "filed"

        target_folder.mkdir(parents=True, exist_ok=True)
        target_path   = self._resolve_conflict(target_folder / new_filename)

        temp_path = target_path.with_suffix('.tmp')
        pdf_path_for_errors = str(source) # For custom exception tracking

        try:
            # Atomic OS move
            shutil.copy2(str(source), str(temp_path))
            temp_path.replace(target_path)
            source.unlink()

            # 🔴 NEW: Generate Sidecar JSON for Human Auditor
            if needs_review:
                sidecar_path = target_path.with_suffix('.json')
                with open(sidecar_path, 'w', encoding='utf-8') as f:
                    json.dump(classification, f, indent=2, ensure_ascii=False)

            message = (
                f"{'⚠️ SENT TO REVIEW' if needs_review else '✅ FILED'}: "
                f"{new_filename} → {target_folder.name}/ "
                f"(confidence={confidence:.2f})"
            )

            self._log_action(str(source), str(target_path), action, message)

            if self.debug:
                print(f"  [Filer] {message}")

            return self._result(True, str(source), str(target_path),
                                action, message, new_filename)

        except OSError as e:
            # Cleanup temp files if needed
            if temp_path.exists():
                temp_path.unlink()

            # e.errno contains the exact system reason for the failure
            if e.errno == errno.ENOSPC:
                raise PDFProcessingError(f"Output drive is out of space.", PDFProcessingError.DISK_FULL, pdf_path_for_errors)
            elif e.errno == errno.EACCES:
                raise PDFProcessingError(f"Missing write permissions for output folder.", PDFProcessingError.PERMISSION_DENIED, pdf_path_for_errors)
            elif e.errno == errno.EEXIST:
                raise PDFProcessingError(f"File collision. Document already exists.", PDFProcessingError.FILE_ALREADY_EXISTS, pdf_path_for_errors)
            else:
                raise PDFProcessingError(f"OS Move failed: {e}", PDFProcessingError.UNKNOWN_SYSTEM, pdf_path_for_errors)

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()

            msg = f"Unexpected filing error: {e}"
            self._log_action(str(source), None, "error", msg)
            raise PDFProcessingError(msg, PDFProcessingError.UNKNOWN_SYSTEM, pdf_path_for_errors)

    def _resolve_conflict(self, target: Path) -> Path:
        """If file already exists, add milliseconds to avoid collision."""
        original_target = target
        while target.exists():
            # Generate a 4-character random hex (e.g., 'a3f9') instead of milliseconds
            random_suffix = secrets.token_hex(2)
            target = original_target.parent / f"{original_target.stem}_{random_suffix}{original_target.suffix}"

        return target

    def _log_action(self, source, destination, action, message):
        """Writes the action directly to the hard drive immediately."""
        entry = {
            "timestamp":   datetime.now().isoformat(),
            "action":      action,
            "source":      source,
            "destination": destination,
            "message":     message,
        }

        log_path = Path(Config.OUTPUT_ROOT) / "filing_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _result(success, source, destination, action,
                message, new_filename=None) -> dict:
        return {
            "success":      success,
            "source":       source,
            "destination":  destination,
            "new_filename": new_filename,
            "action":       action,
            "message":      message,
        }