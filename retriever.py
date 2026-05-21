import chromadb
import uuid
import asyncio

class DocumentRetriever:
    """
    Creates a temporary, in-memory vector database for a single document.
    Used to find the exact semantic chunks containing target entities.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        # EphemeralClient lives entirely in RAM and vanishes when the script ends.
        # It is incredibly fast and requires zero disk I/O.
        self.client = chromadb.EphemeralClient()

    async def get_relevant_chunks(self, chunks: list[str], query: str, top_k: int = 3) -> str:
        """
        Vectorizes the chunks, searches them, and returns a single merged string
        of the most relevant paragraphs.
        """
        if not chunks:
            return ""

        # Run the synchronous ChromaDB operations in a separate thread
        return await asyncio.to_thread(self._sync_retrieve, chunks, query, top_k)

    def _sync_retrieve(self, chunks: list[str], query: str, top_k: int) -> str:
        # 1. Strict Isolation: Generate a unique ID for this specific document
        # Create a unique temporary collection for this specific document
        collection_name = f"temp_doc_{uuid.uuid4().hex[:8]}"
        collection = self.client.create_collection(name=collection_name)

        try:
            # 2. Vectorize and Add
            ids = [str(i) for i in range(len(chunks))]
            collection.add(documents=chunks, ids=ids)

            # 3. Query the semantic intent
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, len(chunks))
            )

            best_chunks = results['documents'][0]

            if self.debug:
                print(f"  [Retriever] 🎯 Retrieved top {len(best_chunks)} highly relevant chunks.")

            return "\n\n...[SNIPPED]...\n\n".join(best_chunks)

        finally:
            # 4. Guaranteed Cleanup: Destroy the sandbox to free RAM immediately,
            # even if the query above throws an unexpected error.
            self.client.delete_collection(name=collection_name)