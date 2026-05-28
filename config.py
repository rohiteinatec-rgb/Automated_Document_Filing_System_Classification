import os
import re
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
    # OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", "qwen3:8b") # Fits 100% in 8GB VRAM
    OLLAMA_MODEL_DEEP = os.getenv("OLLAMA_MODEL_DEEP", "qwen3:14b")  # Uses VRAM + 32GB RAM spillover
    OLLAMA_TIMEOUT_FAST = int(os.getenv("OLLAMA_TIMEOUT_FAST", "90"))   # 1.5 minute for fast scans
    OLLAMA_TIMEOUT_DEEP = int(os.getenv("OLLAMA_TIMEOUT_DEEP", "600"))  # 10 minutes for complex thinking

    APPRISE_URL = os.getenv("APPRISE_URL", None)

    # ─────────────────────────────────────────────────────────────────
    # PRODUCTION GUARDRAIL POLICIES (Decoupled from core engine)
    # ─────────────────────────────────────────────────────────────────

    # 1. Prompt Injection & Adversarial Text (Relational & Meta-Data Shield)
    SECURITY_INJECTION_PATTERN = (
        r'(?i)'
        # Natural language overrides
        r'\b(ignore|disregard|forget|override|bypass).{0,30}(instruction|prompt|rule|system|context|above|anterior)\b'
        r'|\b(ignora|olvida|omite|descarta).{0,30}(instrucci|prompt|regla|sistema)\b'
        # Meta-references to AI/prompt internals
        r'|\b(system prompt|language model|you are (an ai|a bot)|classif(y|ica) as)\b'
        # JSON key injection
        r'|("tag"\s*:|tag\s*:|"company"\s*:|company\s*:|"confidence"\s*:|confidence\s*:)'
        # Jailbreak phrases
        r'|\b(jailbreak|DAN mode|developer mode|unrestricted mode)\b'
        # Role-switching attempts
        r'|\b(pretend (you are|to be)|act as|roleplay as|from now on (you|respond))\b'
    )

    # 2. Customs, Logistics, and Prohibited Legal Types
    SECURITY_DUA_PATTERN = r'\b(dua|documento [uú]nico administrativo|aduana|bill of lading)\b'
    SECURITY_LEGAL_REJECT_PATTERN = r'\b(amendment|anexo|nda|non[- ]?disclosure|procurement terms|terms and conditions|due diligence|archive|litigation|collection|arbitration)\b'

    # 3. Structural Validations
    SECURITY_EMAIL_CHAIN_PATTERN = r'\b(from:|to:|subject:|fw:|fwd:|de:|para:|asunto:)\b'
    SECURITY_TAX_ID_PATTERN = r'\b(?:[A-Z]{2})?[- ]?[A-Z]?[- ]?\d{6,9}[- ]?[A-Z0-9]{1,2}\b'

    # 4. Payroll, ERP, and GL export rejection pattern
    SECURITY_PAYROLL_ERP_PATTERN = (
        r'\b(n[oó]mina|payroll|salary\s+slip|wage\s+slip'
        r'|gl\s+export|g\.l\.\s+export'
        r'|general\s+ledger|balance\s+de\s+comprobaci[oó]n'
        r'|variance\s+report|informe\s+de\s+variaci[oó]n'
        r'|erp\s+export|sap\s+export|oracle\s+export'
        r'|bank\s+statement|extracto\s+bancario|account\s+statement'
        r'|extracto\s+de\s+cuenta)\b'
    )

    # Memory quarantine window in hours (Protects ChromaDB from bad data)
    MEMORY_QUARANTINE_HOURS = int(os.getenv("MEMORY_QUARANTINE_HOURS", "24"))

    # Fast path: tag known, company extraction only, think:OFF
    OLLAMA_OPTIONS_FAST  = {
        "num_ctx":     2048,
        "num_predict": 300,
        "temperature": 0.1,
        "num_gpu":     -1,
    }
    # DEEP Scan path: tag unknown, full document scan reasoning, think:ON
    OLLAMA_OPTIONS_DEEP  = {
        "num_ctx":     6096,
        "num_predict": 2000,
        "temperature": 0.1,
        "num_gpu":     -1,
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
    MEMORY_TRUST_THRESHOLD   = float(os.getenv("MEMORY_TRUST_THRESHOLD", "0.85"))
    MAX_PAGES_TO_SCAN = 2
    VAGUE_TAGS               = {"other", "uncertain"}

    # -- Known tags --
    KNOWN_TAG_PREFIXES = [
        "factura", "invoice", "nomina", "work-contract",
        "m111", "pressupost", "contracte", "albarans", "albara",
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
        "albara":        "logistics",
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

    @classmethod
    def validate_security_patterns(cls):
        """
        Validates all regex patterns on startup.
        Prevents the application from running if a guardrail is broken.
        """
        patterns = {
            "SECURITY_INJECTION_PATTERN":   cls.SECURITY_INJECTION_PATTERN,
            "SECURITY_DUA_PATTERN":         cls.SECURITY_DUA_PATTERN,
            "SECURITY_LEGAL_REJECT_PATTERN":cls.SECURITY_LEGAL_REJECT_PATTERN,
            "SECURITY_EMAIL_CHAIN_PATTERN": cls.SECURITY_EMAIL_CHAIN_PATTERN,
            "SECURITY_TAX_ID_PATTERN":      cls.SECURITY_TAX_ID_PATTERN,
            "SECURITY_PAYROLL_ERP_PATTERN": cls.SECURITY_PAYROLL_ERP_PATTERN,
        }

        errors = []
        for name, pattern in patterns.items():
            if not pattern:
                continue # Allow empty patterns if someone disables a rule intentionally
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                errors.append(f"  INVALID: {name} — {e}")

        if errors:
            raise ValueError("Security pattern validation failed:\n" + "\n".join(errors))

        print(f"  [Config] ✅ All {len(patterns)} security patterns compiled and validated.")