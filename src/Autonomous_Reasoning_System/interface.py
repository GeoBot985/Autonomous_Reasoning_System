import gradio as gr
import logging
import sys
import threading
import time
import textwrap
import signal
import os
import re
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
    # Retrieve the brain instance (initializes only once per process)
    local_brain = get_brain()
    
    print(f"\n[UI] 📨 Received: '{user_message}'")
    if not user_message: return "", history
    if history is None: history = []
    history.append({"role": "user", "content": user_message})
    
    print(f"[UI] ⏳ Sending to Brain...")
    start_t = time.time()
    try:
        response_text = local_brain.think(user_message, history)
        print(f"[UI] ✅ Brain responded in {time.time() - start_t:.2f}s")
    except Exception as e:
        logger.error(f"Brain Error: {e}", exc_info=True)
        response_text = f"⚠️ Error: {e}"
    
    history.append({"role": "assistant", "content": response_text})
    return "", history

# Global headers regex for CV-style documents
# We look for common headers like EXPERIENCE, SKILLS, FMI, etc.
HEADER_REGEX = re.compile(
    r'^\s*(EXPERIENCE|SKILLS|EDUCATION|SUMMARY|PROFILE|FMI|PROJECTS|AWARDS|CONTACTS|HISTORY|GOALS|GOAL)\s*$', 
    re.IGNORECASE
)

def ingest_files(file_objs):
    # Retrieve the brain instance (initializes only once per process)
    local_brain = get_brain()

    global _processing_files
    
    if not file_objs:
        return "No files uploaded."

    results = []

    for file_obj in file_objs:
        path = Path(file_obj.name)

        if path in _processing_files:
            results.append(f"Already processing → skipped: {path.name}")
            continue

        _processing_files.add(path)
        full_text = ""
        
        try:
            print(f"[UI] Ingesting {path.name}...")

            # ── 1. Extract text ──
            if path.suffix.lower() == ".pdf" and HAS_PDF:
                reader = PdfReader(file_obj.name)
                full_text = "".join(page.extract_text() for page in reader.pages)
            else:
                with open(file_obj.name, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = f.read()
            
            if not full_text:
                results.append(f"No usable text found in: {path.name}")
                continue

            # ── 2. Segment by Header (NEW LOGIC) ──
            section_segments = [] # Stores: [{'text': '...', 'section': '...'}, ...]
            current_section = "DOCUMENT HEADER"
            current_text_block = ""
            
            lines = full_text.split('\n')
            
            for line in lines:
                header_match = HEADER_REGEX.match(line)
                
                # Heuristic: Check if line matches a header pattern and is short
                if header_match and 5 < len(line.strip()) < 30: 
                    # End of previous section
                    if current_text_block.strip():
                        section_segments.append({'text': current_text_block.strip(), 'section': current_section})
                    
                    # Start of new section
                    current_section = header_match.group(1).upper()
                    current_text_block = line + "\n"
                else:
                    current_text_block += line + "\n"
            
            # Add the final block
            if current_text_block.strip():
                section_segments.append({'text': current_text_block.strip(), 'section': current_section})
            
            # ── 3. Chunk Segments and Prepare Metadata ──
            batch_texts = []
            metadata_list = []
            
            for segment in section_segments:
                # Chunk the section block (using the standard 500 width)
                # Setting replace_whitespace=False to prevent paragraphs merging poorly
                chunks = textwrap.wrap(segment['text'], width=500, replace_whitespace=False)
                
                for chunk in chunks:
                    batch_texts.append(chunk)
                    # Metadata now includes the section header
                    metadata_list.append({'section': segment['section']}) 

            # ── 4. Save ──
            if batch_texts:
                print(f"[Memory] Batch processing {len(batch_texts)} chunks...")
                local_brain.memory.remember_batch(
                    batch_texts,
                    memory_type="document_chunk",
                    importance=0.5,
                    source=path.name,
                    metadata_list=metadata_list # <-- Pass the metadata list
                )
                results.append(f"Completed: {path.name} ({len(batch_texts)} chunks)")
            else:
                results.append(f"No usable text found in: {path.name}")

        except Exception as e:
            logger.error(f"Ingest failed for {path.name}: {e}", exc_info=True)
            results.append(f"Failed: {path.name} ({str(e)})")
        finally:
            _processing_files.discard(path)

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
    # ... (all gr.Blocks definition and component bindings remain the same) ...

    # The final launch command
    print("\n[UI] 🚀 Launching Gradio. Press Ctrl+C to stop.")
    
    # --- CHANGE THIS LINE ---
    # ADDED 'show_api=False' and 'inbrowser=False' for cleaner startup, but most importantly:
    # ADDED 'prevent_thread_lock=True' and removed the reliance on the automatic reloader.
    demo.queue().launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False, 
        prevent_thread_lock=False, # Recommended for complex multi-threaded/process apps
        inbrowser=False,
        # The key to stop reloading is to ensure you are not using the development server 
        # which often relies on reloading, or running it with the specific `__name__ == '__main__'` guard
        # which we already did. This should fix the final import loop.
    )
