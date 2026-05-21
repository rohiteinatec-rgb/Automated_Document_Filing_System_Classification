import re

class QualityGate:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.MIN_VALID_WORDS = 5
        self.MIN_DENSITY_RATIO = 0.15 # At least 15% of the characters must belong to actual words

    def evaluate(self, text: str) -> dict:
        """Evaluates if the extracted text is high-quality enough for the AI."""
        issues = []
        score = 100

        if not text or not text.strip():
            return {"passed": False, "score": 0, "issues": "empty_payload", "density": 0.0}

        # 1. Check for usable content (Alphanumeric density)
        valid_words = re.findall(r'\b[a-zA-ZÀ-ÿ0-9]{3,}\b', text)
        word_count = len(valid_words)

        text_no_spaces = re.sub(r'\s+', '', text)
        raw_length = len(text_no_spaces) if len(text_no_spaces) > 0 else 1
        valid_chars = sum(len(w) for w in valid_words)
        density_ratio = valid_chars / raw_length

        if word_count < self.MIN_VALID_WORDS:
            issues.append(f"Insufficient words ({word_count} < {self.MIN_VALID_WORDS})")
            score = 0  # Hard fail
        elif density_ratio < self.MIN_DENSITY_RATIO:
            issues.append(f"Low density ({density_ratio:.2f} < {self.MIN_DENSITY_RATIO})")
            score = 0  # Hard fail

        # 2. Check for PDF font corruption (CID blocks)
        # If the document is littered with (cid:xx), the LLM will hallucinate.
        cid_count = text.count("(cid:")
        if cid_count > 0:
            if cid_count > 5:
                issues.append(f"Significant font corruption ({cid_count} CIDs)")
                score -= 60
            else:
                score -= 10 # Minor corruption, maybe still readable

        # 3. Detect Markdown Table Explosion
        # Massive grids of pipes often confuse the LLM's spatial reasoning.
        if "||||" in text or "|---|---|" in text:
            issues.append("Markdown table explosion")
            score -= 40

        # 4. Detect extreme repetition (UN-NESTED)
        # Catches loops in extraction where one word repeats hundreds of times.
        words = text.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.25: # If more than 75% are duplicates
                issues.append(f"Extreme repetition (Unique ratio: {unique_ratio:.2f})")
                score -= 70

        # 5. Gibberish Detection (Long words)
        # Scrambled PDFs often produce words with 50+ characters.
        if any(len(word) > 50 for word in words[:100]):
            issues.append("Gibberish detected (ultra-long words)")
            score -= 30

        passed = score >= 50
        # Consolidate issues into a string for downstream logging
        issues_str = " | ".join(issues) if issues else "none"
        if self.debug and issues:
            print(f"  [QualityGate] Score: {score} | Density: {density_ratio:.2f} | Issues: {issues_str}")

        return {"passed": passed, "score": score, "issues": issues_str, "density": density_ratio}