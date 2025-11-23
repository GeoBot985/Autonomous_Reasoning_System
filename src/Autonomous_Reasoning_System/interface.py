import gradio as gr
import logging
import sys
import threading
import time
import textwrap
import signal
import os
from pathlib import Path

_processing_files = set()   # ← Global deduplication lock

# Try importing pypdf
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

from .brain import get_brain

# --- CTRL+C HANDLER ---
def force_exit(signum, frame):
    os._exit(0)
signal.signal(signal.SIGINT, force_exit)
signal.signal(signal.SIGTERM, force_exit)

# --- Logs ---
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_buffer = []
        self.lock = threading.Lock()
    def emit(self, record):
        try:
            msg = self.format(record)
            with self.lock:
                self.log_buffer.append(msg)
                if len(self.log_buffer) > 200: self.log_buffer.pop(0)
        except: pass
    def get_logs_as_str(self):
        with self.lock: return "\n".join(reversed(self.log_buffer))

root = logging.getLogger()
root.setLevel(logging.INFO)
if root.handlers:
    for h in root.handlers: root.removeHandler(h)
log_capture = ListHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
log_capture.setFormatter(formatter)
root.addHandler(log_capture)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
root.addHandler(console)
logger = logging.getLogger("Interface")

print("🧠 Initializing Refactored Brain...")
brain = get_brain()

def chat_interaction(user_message, history):
    print(f"\n[UI] 📨 Received: '{user_message}'")
    if not user_message: return "", history
    if history is None: history = []
    history.append({"role": "user", "content": user_message})
    
    print(f"[UI] ⏳ Sending to Brain...")
    start_t = time.time()
    try:
        response_text = brain.think(user_message)
        print(f"[UI] ✅ Brain responded in {time.time() - start_t:.2f}s")
    except Exception as e:
        logger.error(f"Brain Error: {e}", exc_info=True)
        response_text = f"⚠️ Error: {e}"
    
    history.append({"role": "assistant", "content": response_text})
    return "", history

# --- INGESTION FIX ---
def ingest_files(file_objs):
    global _processing_files
    
    if not file_objs:
        return "No files uploaded."

    results = []

    for file_obj in file_objs:
        path = Path(file_obj.name)

        # ←←← DEDUPLICATION: skip if already processing this exact file
        if path in _processing_files:
            results.append(f"Already processing → skipped: {path.name}")
            continue

        _processing_files.add(path)
        batch_texts = []        # ←←← ALWAYS define it here (this was the bug!)
        
        try:
            print(f"[UI] Ingesting {path.name}...")

            # ── 1. Extract text ──
            text_content = ""
            if path.suffix.lower() == ".pdf" and HAS_PDF:
                reader = PdfReader(path)
                text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif path.suffix.lower() in {".txt", ".md", ".py", ".json"}:
                text_content = Path(path).read_text(encoding="utf-8", errors="ignore")
            else:
                results.append(f"Unsupported format: {path.name}")
                _processing_files.discard(path)
                continue

            if not text_content.strip():
                results.append(f"Empty document: {path.name}")
                _processing_files.discard(path)
                continue

            # ── 2. Chunk ──
            chunks = textwrap.wrap(
                text_content,
                width=500,
                break_long_words=False,
                replace_whitespace=False
            )
            print(f"[UI] Splitting {path.name} into {len(chunks)} chunks...")

            for chunk in chunks:
                if chunk.strip():
                    batch_texts.append(chunk.strip())

            # ── 3. Save ──
            if batch_texts:
                print(f"[Memory] Batch processing {len(batch_texts)} chunks...")
                brain.memory.remember_batch(
                    batch_texts,
                    memory_type="document_chunk",
                    importance=0.5,
                    source=path.name
                )
                results.append(f"Completed: {path.name} ({len(batch_texts)} chunks)")
            else:
                results.append(f"No usable text found in: {path.name}")

        except Exception as e:
            logger.error(f"Ingest failed for {path.name}: {e}", exc_info=True)
            results.append(f"Failed: {path.name} ({str(e)})")
        finally:
            _processing_files.discard(path)   # ← always release the lock

    return "\n".join(results)

def refresh_logs(): return log_capture.get_logs_as_str()

with gr.Blocks(title="Tyrone ARS") as demo:
    gr.Markdown("# 🧠 Tyrone ARS")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="Interaction")
            msg = gr.Textbox(label="Command", placeholder="Type here...")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Quick Memory Ingest")
            files = gr.File(file_count="multiple", label="Upload Documents")
            upload_status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### 🖥️ Live Logs")
            logs = gr.Code(language="shell", interactive=False, lines=20, label="System Activity")
            timer = gr.Timer(1)

    msg.submit(chat_interaction, [msg, chatbot], [msg, chatbot])
    send.click(chat_interaction, [msg, chatbot], [msg, chatbot])
    files.upload(ingest_files, files, upload_status)
    timer.tick(refresh_logs, outputs=logs)
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    print("\n[UI] 🚀 Launching Gradio. Press Ctrl+C to stop.")
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)