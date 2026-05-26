import os
import shutil
import fitz  # PyMuPDF
import re
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing ADFS modules
from classifier import TagMemory
from config import Config
from sanitizer import sanitise_filename
from filer import Filer

app = FastAPI(title="ADFS Human-In-The-Loop API")

# Allow your local frontend UI to talk to this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow any local port during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI's Semantic Memory
memory = TagMemory(debug=True)

# ---------------------------------------------------------
# DATA SCHEMAS (Clean API Design)
# ---------------------------------------------------------
class ResolveRequest(BaseModel):
    filename: str
    tag: str
    company: str

# ---------------------------------------------------------
# ENDPOINT 1: The Queue (List all Uncertain files)
# ---------------------------------------------------------
@app.get("/api/queue")
async def get_uncertain_queue():
    """Returns a list of all PDFs currently sitting in the UNCERTAIN folder."""
    # 1. Get the path
    uncertain_dir = Config.get_folder("./output/UNCERTAIN")

    # 2. Print it to the terminal so we can debug where FastAPI is looking
    print(f"DEBUG: Looking for uncertain files in -> {os.path.abspath(uncertain_dir)}")

    # 3. Safe fallback WITH the count included to fix the 'undefined' UI bug
    if not os.path.exists(uncertain_dir):
        print("DEBUG: Folder does not exist!")
        return {"queue": [], "count": 0}

    # 4. Fetch the files
    files = [f for f in os.listdir(uncertain_dir) if f.lower().endswith('.pdf')]
    print(f"DEBUG: Found {len(files)} PDFs.")

    return {"queue": files, "count": len(files)}

# ---------------------------------------------------------
# ENDPOINT 2: Privacy-First Document Viewer (Air-Gapped)
# ---------------------------------------------------------
@app.get("/api/preview/{filename}")
async def get_document_preview(filename: str):
    """
    Renders the first page of the PDF as a PNG image.
    """
    print(f"\n[DEBUG] Frontend requested preview for: {filename}")

    # Security: Prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # FORCE THE ABSOLUTE PATH (Matching the Queue logic)
    uncertain_dir = os.path.abspath(Config.get_folder("./output/UNCERTAIN"))
    filepath = os.path.join(uncertain_dir, filename)

    print(f"[DEBUG] Backend looking at exact path: {filepath}")

    if not os.path.exists(filepath):
        print("[DEBUG] ❌ 404: File does not exist at that path!")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        doc = fitz.open(filepath)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        print("[DEBUG] ✅ Image rendered successfully.")
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        print(f"[DEBUG] ❌ 500: PyMuPDF crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to render preview: {str(e)}")

# ---------------------------------------------------------
# ENDPOINT 3: The Resolution & Learning Engine
# ---------------------------------------------------------
@app.post("/api/resolve")
async def resolve_document(req: ResolveRequest):
    """
    Moves the file, RENAMES it to the correct format, and teaches the AI.
    """
    print(f"\n[DEBUG] Resolving document: {req.filename}")

    # 1. Security Check
    if ".." in req.filename or "/" in req.filename or "\\" in req.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 2. Exact Path Match for the Source
    uncertain_dir = os.path.abspath(Config.get_folder("./output/UNCERTAIN"))
    source_path = os.path.join(uncertain_dir, req.filename)

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Source file no longer exists.")

    # 3. Extract the raw text for the AI's memory
    try:
        doc = fitz.open(source_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    # 4. Update ChromaDB
    clean_tag = sanitise_filename(req.tag)
    clean_company = sanitise_filename(req.company)

    memory.store_tag(
        tag=clean_tag,
        company=clean_company,
        source="Human_Verified",
        text=raw_text
    )

    # ---------------------------------------------------------
    # 5. NEW: SMART RENAMING LOGIC
    # ---------------------------------------------------------
    # Use Regex to strip out the old AI mistakes (e.g., "factura_unknown_")
    # and the old datetime stamp to find the true original filename.
    match = re.match(r'^(?:[a-zA-Z-]+)_(?:unknown)_(.*?)_\d{8}_\d{6}(\.[a-zA-Z0-9]+)$', req.filename, re.IGNORECASE)

    if match:
        core_original = match.group(1) + match.group(2) # e.g., "anonima_fundacion.pdf"
    else:
        core_original = req.filename # Fallback if the regex doesn't match

    # Use your Filer class to build the perfect new name
    filer_instance = Filer(debug=True)
    new_filename = filer_instance.build_new_filename(clean_tag, clean_company, core_original)

    # ---------------------------------------------------------
    # 6. Move and Rename
    # ---------------------------------------------------------
    final_dir = os.path.abspath(os.path.join(Config.OUTPUT_ROOT, clean_tag))
    os.makedirs(final_dir, exist_ok=True)

    destination_path = os.path.join(final_dir, new_filename)
    print(f"[DEBUG] Renaming & Moving to: {destination_path}")

    try:
        ## 1. Execute the OS Move
        shutil.move(source_path, destination_path)
        print("[DEBUG] ✅ Move and Rename successful!")

        # 2. NEW: Write to the persistent Audit Log (filing_log.jsonl)
        log_message = f"✅ HUMAN RESOLVED: {req.filename} → {clean_tag}/{new_filename}"
        filer_instance._log_action(
            source=source_path,
            destination=destination_path,
            action="human_resolved",
            message=log_message
        )
        print("[DEBUG] ✅ Action written to audit log.")

    except Exception as e:
        print(f"[DEBUG] ❌ 500: OS move failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File move failed: {str(e)}")

    return {
        "status": "success",
        "message": f"File routed and renamed to {clean_tag}/{new_filename}"
    }