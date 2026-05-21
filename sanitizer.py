import unicodedata
import re

def sanitise_filename(s: str) -> str:
    """Sanitize strings for safe filenames."""
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r'[<>:"/\\|?*,.]', '_', s).strip('_')
    s = re.sub(r'\s+', '_', s)
    while '__' in s:
        s = s.replace('__', '_')
    return s