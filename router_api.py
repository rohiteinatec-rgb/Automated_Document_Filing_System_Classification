import os
import shutil
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing ADFS modules
from classifier import TagMemory
from config import Config
from sanitizer import sanitise_filename

app = FastAPI(title="ADFS Human-In-The-Loop API")

# Allow your local frontend UI to talk to this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Restrict this to your UI's local port
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
    uncertain_dir = Config.get_folder("uncertain")

    if not os.path.exists(uncertain_dir):
        return {"queue": []}

    files = [f for f in os.listdir(uncertain_dir) if f.lower().endswith('.pdf')]
    return {"queue": files, "count": len(files)}

# ---------------------------------------------------------
# ENDPOINT 2: Privacy-First Document Viewer (Air-Gapped)
# ---------------------------------------------------------
@app.get("/api/preview/{filename}")
async def get_document_preview(filename: str):
    """
    Renders the first page of the PDF as a PNG image.
    Crucial for Privacy: Prevents browser extensions from reading raw financial text.
    """
    safe_filename = sanitise_filename(filename)
    filepath = os.path.join(Config.get_folder("uncertain"), safe_filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Open PDF and render Page 1 to an image buffer
        doc = fitz.open(filepath)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150) # 150 DPI is a good balance of readability and speed
        img_bytes = pix.tobytes("png")
        doc.close()

        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render preview: {str(e)}")

# ---------------------------------------------------------
# ENDPOINT 3: The Resolution & Learning Engine
# ---------------------------------------------------------
@app.post("/api/resolve")
async def resolve_document(req: ResolveRequest):
    """
    Moves the file to its correct final folder AND teaches the AI to
    recognize this document layout in the future.
    """
    safe_filename = sanitise_filename(req.filename)
    source_path = os.path.join(Config.get_folder("uncertain"), safe_filename)

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Source file no longer exists. Might have been resolved by another user.")

    # 1. Extract the raw text for the AI's memory
    try:
        doc = fitz.open(source_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    # 2. Update ChromaDB (The Continuous Learning Loop)
    clean_tag = sanitise_filename(req.tag)
    clean_company = sanitise_filename(req.company)

    memory.store_tag(
        tag=clean_tag,
        company=clean_company,
        source="Human_Verified",
        text=raw_text
    )

    # 3. Move the file to the correct final destination (Thread-safe OS operation)
    final_dir = Config.get_folder(clean_tag)
    company_dir = os.path.join(final_dir, clean_company)
    os.makedirs(company_dir, exist_ok=True)

    destination_path = os.path.join(company_dir, safe_filename)

    try:
        shutil.move(source_path, destination_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File move failed: {str(e)}")

    return {
        "status": "success",
        "message": f"File routed to {clean_tag}/{clean_company} and AI Memory updated."
    }