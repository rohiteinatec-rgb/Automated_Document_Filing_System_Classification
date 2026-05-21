import re

class SemanticChunker:
    """
    Enterprise-grade recursive text splitter.
    Splits text by natural semantic boundaries (\n\n, \n, space)
    while preserving an overlapping window of context.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, debug: bool = False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.debug = debug
        # The separators in order of semantic importance
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        # Clean up excessive newlines from OCR
        text = re.sub(r'\n{3,}', '\n\n', text)
        chunks = self._split(text, self.separators)

        if self.debug:
            print(f"  [Chunker] Sliced document into {len(chunks)} semantic chunks "
                  f"(Max {self.chunk_size} chars each)")
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        # Base case: if the text fits perfectly, return it as one chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Find the best semantic separator to use
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                continue
            if sep in text:
                separator = sep
                break

        # Split the text by the chosen separator
        splits = text.split(separator) if separator else list(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split) + (len(separator) if current_length > 0 else 0)

            # If adding this split exceeds the chunk size, save the chunk
            if current_length + split_len > self.chunk_size and current_length > 0:
                chunks.append(separator.join(current_chunk))

                # Build the overlap for the next chunk
                overlap_length = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_length + len(s) > self.overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_length += len(s) + len(separator)

                current_chunk = overlap_chunk
                current_length = sum(len(s) for s in current_chunk) + (len(current_chunk) - 1) * len(separator)

            current_chunk.append(split)
            current_length += len(split) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        # If a single chunk is somehow STILL too big, recurse with a harsher separator
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size and len(separators) > 1:
                final_chunks.extend(self._split(chunk, separators[1:]))
            else:
                final_chunks.append(chunk)

        return final_chunks