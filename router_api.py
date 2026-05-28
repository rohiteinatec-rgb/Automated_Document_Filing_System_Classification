import os
import shutil
import fitz  # PyMuPDF
import re
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, Depends, Security, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from database import DatabaseArchiver

# Execute dotenv so the API key is loaded into memory
dotenv_path = os.path.join(os.getcwd(), 'dev.env')
print(f"DEBUG: Looking for .env at: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

# Import your existing ADFS modules
from classifier import TagMemory
from config import Config
from sanitizer import sanitise_filename
from filer import Filer

# ---------------------------------------------------------
#  NEW: CPU/MEMORY RESOURCE LIMITS & SAFE HELPERS
# ---------------------------------------------------------
MAX_PAGES_TO_SCAN = 5
MAX_PREVIEW_TIMEOUT = 5.0  # seconds
MAX_EXTRACTION_TIMEOUT = 10.0 # seconds

def _safe_render_preview(filepath: str) -> bytes:
    """CPU-bound task: Renders page 0 with safety limits."""
    try:
        with fitz.open(filepath) as doc:
            page = doc.load_page(0)

            # Use a matrix to limit the dimension to prevent OOM
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Hard cutoff for maliciously massive documents (e.g., blueprints)
            if pix.width > 4000 or pix.height > 4000:
                raise ValueError("Document dimensions exceed maximum allowed safe size.")

            return pix.tobytes("png")
    except Exception as e:
        raise RuntimeError(f"Preview generation failed: {e}")

def _safe_extract_text(filepath: str) -> str:
    """CPU-bound task: Extracts text up to a strict page limit."""
    text_blocks = []
    try:
        with fitz.open(filepath) as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PAGES_TO_SCAN:
                    print(f"[Resource Guard] Stopped extraction at page {MAX_PAGES_TO_SCAN}")
                    break
                text_blocks.append(page.get_text())
        return "".join(text_blocks)
    except Exception as e:
        raise RuntimeError(f"Text extraction failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifecycle] Booting up ADFS API...")

    # Initialize heavy IO tasks safely during startup
    try:
        app.state.memory = TagMemory(debug=True)
        print("[Lifecycle] ✅ TagMemory initialized.")
    except Exception as e:
        print(f"[Lifecycle] ⚠️ TagMemory degraded/unavailable: {e}")
        app.state.memory = None # Graceful degradation

    yield # The API runs while yielded

    print("[Lifecycle] Shutting down ADFS API. Cleaning up resources...")

app = FastAPI(title="ADFS Human-In-The-Loop API", lifespan=lifespan)

# ---------------------------------------------------------
# SECURITY: CORS CONFIGURATION
# ---------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://localhost:63342"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# SECURITY: API AUTHENTICATION & PATH VALIDATION
# ---------------------------------------------------------
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    """Ensures requests contain the valid secret key from .env"""
    expected = os.getenv("ADFS_API_KEY")

    if not expected:
        print("[CRITICAL] ADFS_API_KEY is not set in .env!")
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    if api_key == expected:
        return api_key

    raise HTTPException(status_code=401, detail="Unauthorized")

def _safe_join_and_check(base_dir: str, filename: str) -> str:
    """Cryptographically secure path resolution to prevent directory traversal."""
    if '\x00' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename: Null byte detected.")

    candidate = os.path.realpath(os.path.join(base_dir, filename))
    base_real = os.path.realpath(base_dir)

    if os.path.commonpath([base_real, candidate]) != base_real:
        print(f"[SECURITY] 🛑 Traversal blocked: {filename}")
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal attempt blocked.")

    if not os.path.exists(candidate):
        raise HTTPException(status_code=404, detail="File not found")

    return candidate


# ---------------------------------------------------------
# DATA SCHEMAS
# ---------------------------------------------------------
class ResolveRequest(BaseModel):
    filename: str
    tag: str
    company: str

# ---------------------------------------------------------
# ENDPOINT 1: The Queue
# ---------------------------------------------------------
@app.get("/api/queue")
async def get_uncertain_queue(api_key: str = Depends(get_api_key)):
    """Returns a list of all PDFs currently sitting in the UNCERTAIN folder."""
    relative_path = Config.get_folder("./output/UNCERTAIN")
    absolute_uncertain_dir = os.path.abspath(relative_path)

    if not os.path.exists(absolute_uncertain_dir):
        return {"queue": [], "count": 0}

    try:
        files = [f for f in os.listdir(absolute_uncertain_dir) if f.lower().endswith('.pdf')]
        return {"queue": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read queue directory")

# ---------------------------------------------------------
# ENDPOINT 2: Privacy-First Document Viewer
# ---------------------------------------------------------
@app.get("/api/preview/{filename}")
async def get_document_preview(filename: str, api_key: str = Depends(get_api_key)):
    """Renders the first page of the PDF as a PNG image safely."""
    uncertain_dir = os.path.abspath(Config.get_folder("./output/UNCERTAIN"))
    filepath = _safe_join_and_check(uncertain_dir, filename)

    try:
        img_bytes = await asyncio.wait_for(
            asyncio.to_thread(_safe_render_preview, filepath),
            timeout=MAX_PREVIEW_TIMEOUT
        )
        print("[DEBUG] ✅ Image rendered successfully.")
        return Response(content=img_bytes, media_type="image/png")

    except asyncio.TimeoutError:
        print(f"[DEBUG] ❌ 504: Preview rendering timed out for {filename}")
        raise HTTPException(status_code=504, detail="Preview rendering timed out (file too complex).")
    except Exception as e:
        print(f"[DEBUG] ❌ 500: PyMuPDF crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to render preview: {str(e)}")

# ---------------------------------------------------------
# ENDPOINT 3: The Resolution & Learning Engine
# ---------------------------------------------------------
@app.post("/api/resolve")
async def resolve_document(req: ResolveRequest, request: Request, bg_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    """Moves the file, RENAMES it to the correct format, and teaches the AI."""

    uncertain_dir = os.path.abspath(Config.get_folder("./output/UNCERTAIN"))
    source_path = _safe_join_and_check(uncertain_dir, req.filename)

    try:
        raw_text = await asyncio.wait_for(
            asyncio.to_thread(_safe_extract_text, source_path),
            timeout=MAX_EXTRACTION_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Text extraction timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    clean_tag = sanitise_filename(req.tag)
    clean_company = sanitise_filename(req.company)

    if request.app.state.memory:
        request.app.state.memory.store_tag(
            tag=clean_tag,
            company=clean_company,
            source="Human_Verified",
            text=raw_text
        )

    pure_stem = Path(req.filename).stem
    match = re.match(r'^(?:[a-zA-Z-]+)_(?:unknown)_(.*?)_\d{8}_\d{6}$', pure_stem, re.IGNORECASE)
    core_original_stem = match.group(1) if match else pure_stem

    filer_instance = Filer(debug=True)
    new_filename = filer_instance.build_new_filename(clean_tag, clean_company, core_original_stem)

    final_dir = os.path.abspath(os.path.join(Config.OUTPUT_ROOT, clean_tag))
    os.makedirs(final_dir, exist_ok=True)
    destination_path = os.path.join(final_dir, new_filename)

    try:
        shutil.move(source_path, destination_path)
        print("[DEBUG] ✅ Move and Rename successful!")

        log_message = f"✅ HUMAN RESOLVED: {req.filename} → {clean_tag}/{new_filename}"
        filer_instance.log_action(source_path, destination_path, "human_resolved", log_message)

        db = DatabaseArchiver()
        bg_tasks.add_task(
            db.archive_filing,
            record={
                "file": core_original_stem,
                "tag": clean_tag,
                "company": clean_company,
                "action": "human_resolved",
                "message": log_message,
                "success": True
            }
        )
        print("[DEBUG] ✅ Action queued for database archival.")

    except Exception as e:
        print(f"[DEBUG] ❌ 500: OS move failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File move failed: {str(e)}")

    return {
        "status": "success",
        "message": f"File routed and renamed to {clean_tag}/{new_filename}"
    }