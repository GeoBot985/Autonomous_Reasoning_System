import logging
from pathlib import Path
from pypdf import PdfReader
import textwrap
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.llm_summarizer import summarize_with_local_llm
# We only import these for the default case
from Autonomous_Reasoning_System.memory.embeddings import EmbeddingModel
from Autonomous_Reasoning_System.memory.vector_store import DuckVSSVectorStore

logger = logging.getLogger(__name__)

class PDFIngestor:
    """
    Loads a PDF, extracts text, chunks it, and stores each part as a memory.
    Optionally creates an overall summary.
    """
    def __init__(self, memory_storage=None):
        # If an existing storage (Tyrone's brain) is passed, use it.
        if memory_storage:
            self.memory = memory_storage
            # We assume the storage already has an embedder/vector_store attached
            self.embedder = memory_storage.embedder
        else:
            # Fallback: Create a standalone stack (Legacy behavior, fixes the warning too)
            logger.info("PDFIngestor: Initializing standalone memory stack...")
            self.embedder = EmbeddingModel()
            self.vector_store = DuckVSSVectorStore()
            self.memory = MemoryStorage(
                embedding_model=self.embedder, 
                vector_store=self.vector_store
            )

    def ingest(self, file_path: str, chunk_size: int = 1000, summarize: bool = True):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"📄 Reading PDF: {path.name}")
        try:
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            return

        if not text.strip():
            logger.warning("⚠️ No text extracted.")
            return

        # Split into chunks
        chunks = textwrap.wrap(text, chunk_size)
        logger.info(f"🧩 Splitting into {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks, 1):
            title = f"{path.stem} (Part {i}/{len(chunks)})"
            self.memory.add_memory(
                text=f"{title}\n\n{chunk}",
                memory_type="document",
                importance=0.7,
                source=path.name,
            )

        if summarize:
            logger.info("🧠 Summarizing content...")
            summary = summarize_with_local_llm(text[:6000])  # limit for speed
            self.memory.add_memory(
                text=f"Summary of {path.name}:\n{summary}",
                memory_type="document_summary",
                importance=0.9,
                source="PDFIngestor"
            )
            logger.info("🧾 Summary added to memory.")

        logger.info(f"✅ Ingestion complete: {len(chunks)} chunks + summary stored.")

if __name__ == "__main__":
    # Standalone test usage
    import sys
    from Autonomous_Reasoning_System.infrastructure.logging_utils import setup_logging
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage: python -m Autonomous_Reasoning_System.io.pdf_ingestor <pdf_path>")
        sys.exit(1)
        
    ingestor = PDFIngestor() # Will use fallback init
    ingestor.ingest(sys.argv[1])