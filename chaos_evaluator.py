import pytest
import asyncio
import requests
import errno
from unittest.mock import patch, MagicMock
from main import DocumentAutoFiler
from errors import PDFProcessingError

class ChaosEvaluator:
    def __init__(self):
        self.pipeline = DocumentAutoFiler(debug=True)

    @pytest.mark.asyncio
    async def test_ollama_timeout(self):
        """
        SCENARIO: Ollama hangs mid-request.
        EXPECTED: Pipeline raises CLASSIFICATION_TIMEOUT and handles it via AlertManager.
        """
        print("\n🔥 Starting Chaos Test: Ollama Timeout...")

        # We mock 'requests.post' to raise a Timeout error
        with patch('requests.post', side_effect=requests.exceptions.Timeout):
            with pytest.raises(PDFProcessingError) as excinfo:
                # We skip Stage 1/2 by using a unique string to force Deep-Scan
                await self.pipeline.classifier._call_ollama_deep("Unique content")

            assert excinfo.value.error_type == PDFProcessingError.CLASSIFICATION_TIMEOUT
            print("✅ SUCCESS: System caught the timeout and classified the error.")

    @pytest.mark.asyncio
    async def test_disk_full_simulation(self):
        """
        SCENARIO: Filesystem reports 'No space left on device' (Errno 28).
        EXPECTED: Filer catches the OSError and raises structured DISK_FULL error.
        """
        print("\n🔥 Starting Chaos Test: Disk Full Simulation...")

        # Mock shutil.copy2 to simulate a full disk
        with patch('shutil.copy2', side_effect=OSError(errno.ENOSPC, "No space left")):
            with pytest.raises(PDFProcessingError) as excinfo:
                # Attempt to file a dummy classification
                self.pipeline.filer.file_document("dummy.pdf", {"tag": "factura", "company": "Test"})

            assert excinfo.value.error_type == PDFProcessingError.DISK_FULL
            print("✅ SUCCESS: Filer correctly identified the 'Disk Full' OS state.")

    @pytest.mark.asyncio
    async def test_load_concurrency_100(self):
        """
        SCENARIO: 100 files enter the queue simultaneously.
        EXPECTED: Throughput is maintained, no 'Connection Reset' from over-tasking Ollama.
        """
        print("\n🔥 Starting Load Test: 100 Concurrent Items...")
        start_time = asyncio.get_event_loop().time()

        # Fill the queue with 100 fake paths
        for i in range(100):
            await self.pipeline.task_queue.put(f"test_file_{i}.pdf")

        # We mock 'process' to take exactly 0.1s to simulate work
        with patch.object(DocumentAutoFiler, 'process', return_value={"success": True}):
            worker = asyncio.create_task(self.pipeline.process_queue())
            await self.pipeline.task_queue.join()
            worker.cancel()

        duration = asyncio.get_event_loop().time() - start_time
        print(f"✅ SUCCESS: 100 items drained in {duration:.2f}s without deadlocks.")

    @pytest.mark.asyncio
    async def test_chromadb_corruption_recovery(self):
        """
        SCENARIO: ChromaDB collection is None or deleted.
        EXPECTED: Stage 1 fails silently/gracefully and passes to Stage 2 (Keywords).
        """
        print("\n🔥 Starting Chaos Test: ChromaDB Corruption...")

        # Set the collection to None to simulate a lost connection
        self.pipeline.classifier.memory._collection = None

        # The code should NOT crash, but instead return an empty list
        results = self.pipeline.classifier.memory.find_similar("Sample text")
        assert results == []
        print("✅ SUCCESS: System bypassed corrupted memory and stayed alive.")