import logging
from pathlib import Path
from pypdf import PdfReader
from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.memory.llm_summarizer import summarize_with_local_llm
import textwrap

logger = logging.getLogger(__name__)

class PDFIngestor:
    """
    Loads a PDF, extracts text, chunks it, and stores each part as a memory.
    Optionally creates an overall summary.
    """
    def __init__(self):
        # In standalone script usage, we might need to instantiate new Storage
        self.memory = MemoryStorage()

    def ingest(self, file_path: str, chunk_size: int = 1000, summarize: bool = True):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"📄 Reading PDF: {path.name}")
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
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


# ==========================================================
# MAIN EXECUTION ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    import sys
    import traceback
    from Autonomous_Reasoning_System.infrastructure.logging_utils import setup_logging

    setup_logging()

    if len(sys.argv) < 2:
        logger.error("Usage: python -m Autonomous_Reasoning_System.io.pdf_ingestor <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    logger.info(f"⚙️ Starting ingestion for: {pdf_path}")

    try:
        ingestor = PDFIngestor()
        ingestor.ingest(pdf_path)
        logger.info("✅ Done.")
    except Exception as e:
        logger.error("❌ Error during ingestion:")
        logger.error(traceback.format_exc())
