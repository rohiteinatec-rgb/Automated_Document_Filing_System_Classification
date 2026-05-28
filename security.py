import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from config import Config

@dataclass
class SecurityCheckResult:
    passed:         bool
    threat_type:    Optional[str]  = None
    threat_score:   float          = 0.0
    sanitized_text: Optional[str]  = None

_INJECTION_SIGNATURES = [
    (r'\b(ignore|disregard|forget|override|bypass).{0,40}(instruction|prompt|rule|system|context|above|anterior)\b', "nl_override"),
    (r'\b(ignora|olvida|omite|descarta).{0,40}(instrucci|prompt|regla|sistema|anterior)\b', "nl_override_es"),
    (r'\b(system\s+prompt|language\s+model|large\s+language|llm\b|gpt\b|claude\b|qwen\b)\b', "meta_reference"),
    (r'\b(jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode|pretend\s+(you\s+are|to\s+be)|act\s+as\s+a|roleplay\s+as)\b', "role_switch"),
    (r'["\']tag["\']\s*:\s*["\']?\w+["\']?', "json_key_injection"),
    (r'["\']confidence["\']\s*:\s*[\d.]+', "json_key_injection"),
    (r'["\']company["\']\s*:\s*["\']', "json_key_injection"),
    (r'\b(classif(y|ica)\s+this\s+(document\s+)?as\b|label\s+this\s+as\b|the\s+tag\s+(should\s+be|is|must\s+be)\b|output\s+tag\b)', "classification_command"),
    (r'(ADVISORY|NOTICE|INSTRUCTION|NOTE TO AI|AI DIRECTIVE)\s*[:：]', "embedded_directive"),
]

_COMPILED_SIGNATURES = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), threat_type)
    for pattern, threat_type in _INJECTION_SIGNATURES
]

def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

# --- LAYER A: PRE-LLM SANITIZATION ---
def detect_and_sanitize(text: str, debug: bool = False) -> SecurityCheckResult:
    """
    Scans for injection patterns. If ANY threat signature is found, the check FAILS.
    We do not silently pass redacted text to the LLM (Zero-Trust Policy).
    """
    normalized = _nfkc(text)
    sanitized = text
    hit_detected = False
    top_threat = None

    for pattern, threat_type in _COMPILED_SIGNATURES:
        if pattern.search(normalized):
            hit_detected = True
            top_threat = threat_type
            if debug:
                print(f"  [Security] 🛡️ Neutralizing threat: {threat_type}")
            # Replace the malicious block with a harmless placeholder
            sanitized = pattern.sub(" [REDACTED_SECURITY_THREAT] ", sanitized)

    if hit_detected:
        return SecurityCheckResult(passed=False, threat_type=top_threat, threat_score=1.0, sanitized_text=sanitized)

    return SecurityCheckResult(passed=True, sanitized_text=text)

# --- LAYER B: STRUCTURAL VALIDATION ---
def validate_document_structure(text: str, debug: bool = False) -> SecurityCheckResult:
    stripped = text.strip()

    # 1. Embedded JSON block in document text
    if re.search(r'\{[^{}]{0,200}"(tag|company|confidence)"\s*:', stripped, re.IGNORECASE):
        if debug: print("  [Security] 🚨 Structural: embedded JSON block found.")
        return SecurityCheckResult(passed=False, threat_type="embedded_json_block", threat_score=1.0)

    # 2. Control character density
    ctrl_chars = sum(1 for c in stripped if ord(c) < 32 and c not in '\n\r\t')
    ctrl_ratio = ctrl_chars / max(len(stripped), 1)
    if ctrl_ratio > 0.02:
        if debug: print(f"  [Security] ⚠️ Structural: high control character density ({ctrl_ratio:.2%}).")
        return SecurityCheckResult(passed=False, threat_type="control_char_injection", threat_score=0.8)

    return SecurityCheckResult(passed=True)

# --- LAYER C: POST-LLM VALIDATION ---
def validate_llm_output(parsed: dict, original_text: str, debug: bool = False) -> SecurityCheckResult:
    # 1. Check required keys
    required = {"tag", "company", "confidence"}
    if not required.issubset(parsed.keys()):
        if debug: print(f"  [Security] ❌ Output validation: missing keys")
        return SecurityCheckResult(passed=False, threat_type="schema_violation", threat_score=1.0)

    # 2. 🔴 FIXED: Dynamically validate tags against Config's MASTER routing table
    allowed_tags = {tag.lower() for tag in Config.get_all_tags()}

    tag = str(parsed.get("tag", "")).lower().strip()
    if tag not in allowed_tags:
        if debug: print(f"  [Security] ❌ Output validation: unknown tag '{tag}'")
        return SecurityCheckResult(passed=False, threat_type="unknown_tag", threat_score=0.9)

    # 3. Validate Confidence Bounds
    try:
        conf_val = float(parsed.get("confidence", 0.0))
        if not (0.0 <= conf_val <= 1.0):
            if debug: print(f"  [Security] ❌ Output validation: confidence out of bounds ({conf_val})")
            return SecurityCheckResult(passed=False, threat_type="invalid_confidence", threat_score=1.0)
    except (ValueError, TypeError):
        if debug: print(f"  [Security] ❌ Output validation: confidence must be a float")
        return SecurityCheckResult(passed=False, threat_type="invalid_confidence", threat_score=1.0)

    # 4. Validate Company injection echo
    company = str(parsed.get("company", "")).strip()
    if company.lower() not in ("unknown", ""):
        normalized_company = _nfkc(company)
        for pattern, threat_type in _COMPILED_SIGNATURES:
            if pattern.search(normalized_company):
                if debug: print(f"  [Security] 🚨 Output validation: company field contains injection echo")
                return SecurityCheckResult(passed=False, threat_type="company_injection_echo", threat_score=1.0)

    return SecurityCheckResult(passed=True)

# --- THE MASTER SECURITY GATE ---
def run_full_security_check(text: str, parsed_output: Optional[dict] = None, debug: bool = False) -> tuple[bool, Optional[str], str]:
    # Phase 1: Pre-LLM Checks
    if parsed_output is None:
        # Layer A: Sanitize
        res_a = detect_and_sanitize(text, debug=debug)
        clean_text = res_a.sanitized_text

        # If Layer A fails, immediately halt and reject the document
        if not res_a.passed:
            return False, res_a.threat_type, clean_text

        # Layer B: Structure
        res_b = validate_document_structure(clean_text, debug=debug)
        if not res_b.passed and res_b.threat_score >= 0.8:
            return False, res_b.threat_type, clean_text

        return True, None, clean_text

    # Phase 2: Post-LLM Checks
    else:
        res_c = validate_llm_output(parsed_output, text, debug=debug)
        if not res_c.passed:
            return False, res_c.threat_type, text

        return True, None, text