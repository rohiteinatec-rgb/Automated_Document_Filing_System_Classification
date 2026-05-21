from chunker import SemanticChunker
from reader import PDFReader

# Initialize the chunker (1000 characters per block, 200 characters of overlap)
chunker = SemanticChunker(chunk_size=1000, overlap=200, debug=True)

# Extract text from one of your long, complex PDFs
text, method = PDFReader.extract_for_classification("./test_suite/test_hard_platform_aws_cloud.pdf", debug=False)

# Slice it
chunks = chunker.split_text(text)

# Print the chunks to see the magic
for i, chunk in enumerate(chunks):
    print(f"\n{'='*50}")
    print(f" 📦 CHUNK {i + 1} (Length: {len(chunk)})")
    print(f"{'='*50}")
    print(chunk)