import gradio as gr
import logging
import sys
import threading
import time
import textwrap
import signal # <--- ADDED
import os
from pathlib import Path

# --- CTRL+C HANDLER ---
# This ensures the app quits instantly when you press Ctrl+C in the terminal
def force_exit(signum, frame):
    print("\n[System] 🛑 Ctrl+C detected. Exiting cleanly...")
    os._exit(0)

signal.signal(signal.SIGINT, force_exit)
signal.signal(signal.SIGTERM, force_exit)
# -----------------------

# Try importing pypdf
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

from .brain import get_brain

# --- Log Capture ---
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
        except Exception: self.handleError(record)
    def get_logs_as_str(self):
        with self.lock: return "\n".join(reversed(self.log_buffer))

root = logging.getLogger()
root.setLevel(logging.INFO)
if root.handlers:
    for handler in root.handlers: root.removeHandler(handler)
log_capture = ListHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
log_capture.setFormatter(formatter)
root.addHandler(log_capture)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
root.addHandler(console)
logger = logging.getLogger("Interface")

# --- Init ---
print("🧠 Initializing Refactored Brain (FastEmbed Edition)...")
brain = get_brain()

# --- Chat Handler ---
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
        print(f"[UI] ❌ Brain Error: {e}")

    history.append({"role": "assistant", "content": response_text})
    return "", history

# --- Ingestion ---
def ingest_files(file_objs):
    if not file_objs: return "No files selected."
    results = []
    print(f"[UI] 📂 Ingesting {len(file_objs)} files...")
    
    for f in file_objs:
        path = Path(f.name)
        try:
            text_content = ""
            if path.suffix.lower() == ".pdf":
                if HAS_PDF:
                    reader = PdfReader(f.name)
                    text_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                else:
                    results.append(f"❌ Skipped {path.name} (pypdf not installed)")
                    continue
            elif path.suffix.lower() in [".txt", ".md", ".py", ".json", ".log"]:
                with open(f.name, "r", encoding="utf-8") as txt_file:
                    text_content = txt_file.read()
            else:
                results.append(f"⚠️ Unknown type: {path.name}")
                continue

            if text_content.strip():
                chunks = textwrap.wrap(text_content, 500)
                print(f"[UI] ✂️ Splitting {path.name} into {len(chunks)} chunks...")
                for i, chunk in enumerate(chunks):
                    brain.memory.remember(chunk, memory_type="document_chunk", importance=0.5, source=f"{path.name} (Part {i+1})")
                results.append(f"✅ Ingested: {path.name} ({len(chunks)} chunks)")
            else:
                results.append(f"⚠️ Empty file: {path.name}")

        except Exception as e:
            logger.error(f"Ingest failed for {path.name}: {e}")
            results.append(f"❌ Failed: {path.name} ({str(e)})")

    return "\n".join(results)

def refresh_logs(): return log_capture.get_logs_as_str()

# --- UI Layout ---
with gr.Blocks(title="Tyrone ARS (Refactored)") as demo:
    gr.Markdown("# 🧠 Tyrone ARS (Refactored)")
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