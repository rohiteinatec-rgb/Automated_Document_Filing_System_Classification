import os
from dotenv import load_dotenv

# 1. Determine the environment (default to development)
ENV = os.getenv("ADFS_ENV", "development")

# 2. THE CASCADE PATTERN
# First, load the base .env to establish the foundational defaults.
load_dotenv(".env")

# Second, load the environment-specific file (if it exists) and OVERRIDE the defaults.
# This way, your .env.production only needs to contain the variables that actually change!
env_file = f".env.{ENV}"
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)

class Config:
    # -- Ollama --
    ENVIRONMENT = ENV
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    OLLAMA_TIMEOUT_FAST = int(os.getenv("OLLAMA_TIMEOUT_FAST", "90"))   # 1.5 minute for fast scans
    OLLAMA_TIMEOUT_DEEP = int(os.getenv("OLLAMA_TIMEOUT_DEEP", "600"))  # 10 minutes for complex thinking

    APPRISE_URL = os.getenv("APPRISE_URL", None)

    # Fast path: tag known, company extraction only, think:OFF
    OLLAMA_OPTIONS_FAST  = {
        "num_ctx":     2048,
        "num_predict": 300,
        "temperature": 0.1,
        "num_thread":  20,
    }
    # DEEP Scan path: tag unknown, full document scan reasoning, think:ON
    OLLAMA_OPTIONS_DEEP  = {
        "num_ctx":     6096,
        "num_predict": 2000,
        "temperature": 0.1,
        "num_thread":  20,
    }

    # ── Database & Monitoring (from .env) ──
    DATABASE_URL = os.getenv("DATABASE_URL", None)
    DATABASE_ARCHIVAL_ENABLED = DATABASE_URL is not None
    SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", None)
    EMAIL_ALERTS = os.getenv("EMAIL_ALERTS", None)
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))

    # -- Paths --
    BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DB_PATH = os.path.join(BASE_DIR, "chromadb")
    OUTPUT_ROOT    = os.path.join(BASE_DIR, "output")

    # -- Classification --
    CHARS_FOR_CLASSIFICATION = 1200
    CONFIDENCE_THRESHOLD     = 0.70
    MAX_TAG_LENGTH           = 30
    FALLBACK_CONFIDENCE      = float(os.getenv("FALLBACK_CONFIDENCE", "0.75"))
    FASTRACK_CONFIDENCE      = 0.80
    MEMORY_TRUST_THRESHOLD   = float(os.getenv("MEMORY_TRUST_THRESHOLD", "0.85"))    # PAGE1_CHARS              = 800
    # TAIL_CHARS               = 400
    # METADATA_CHARS           = 200
    VAGUE_TAGS               = {"other", "uncertain"}

    # -- Known tags --
    KNOWN_TAG_PREFIXES = [
        "factura", "invoice", "nomina", "work-contract",
        "m111", "pressupost", "contracte", "albarans",
        "informe", "other", "uncertain"
    ]

    # Enterprise Data Contract: Forces the AI to return this exact structure
    OLLAMA_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "tag": {"type": "string"},
            "company": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["tag", "company", "confidence"]
    }

    # -- Filer --
    FILENAME_FORBIDDEN_CHARS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # -- Folder mapping --
    TAG_FOLDERS = {
        "factura":       "school-financial",
        "invoice":       "school-financial",
        "nomina":        "hr-payroll",
        "work-contract": "hr-contracts",
        "m111":          "tax-M111",
        "pressupost":    "school-financial",
        "contracte":     "hr-contracts",
        "albarans":      "logistics",
        "other":         "unclassified",
        "uncertain":     "UNCERTAIN",
        "carta":         "correspondence",
        "menu":          "menus",
        "comunicat":     "corporate-communications",
    }
    @classmethod
    def get_folder(cls, tag: str) -> str:
        return cls.TAG_FOLDERS.get(tag.lower(), tag.lower())

    @classmethod
    def get_all_tags(cls) -> list:
        return list(cls.TAG_FOLDERS.keys())

    # Add to the very bottom of config.py
    # print(f"  [Config] Booting in {Config.ENVIRONMENT.upper()} mode "
    #   f"(Model: {Config.OLLAMA_MODEL} | DB: {'Enabled' if Config.DATABASE_ARCHIVAL_ENABLED else 'Disabled'})")
