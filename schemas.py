from typing import TypedDict, Optional

class ClassificationResult(TypedDict):
    """Strict schema for the output of the Classifier."""
    tag: str
    confidence: float
    company: str
    folder: str
    is_uncertain: bool
    original_filename: str

class ProcessResult(TypedDict, total=False):
    """
    Strict schema for the final pipeline output.
    total=False allows some keys to be optional (like tag/company on failure)
    """
    success: bool
    file: str
    action: str
    message: Optional[str]
    tag: Optional[str]
    company: Optional[str]