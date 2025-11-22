import gradio as gr
import logging
import sys
import threading
import time
from pathlib import Path

# --- 1. Log Capture for UI ---
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_buffer = []
        self.lock = threading.Lock()

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.lock.acquire(timeout=0.5):
                try:
                    self.log_buffer.append(msg)
                    if len(self.log_buffer) > 500:
                        self.log_buffer.pop(0)
                finally:
                    self.lock.release()
        except Exception:
            self.handleError(record)

    def get_logs_as_str(self):
        if self.lock.acquire(timeout=0.5):
            try:
                return "\n".join(reversed(self.log_buffer))
            finally:
                self.lock.release()
        return "Log buffer locked..."

# Setup logging
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)

log_capture = ListHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_capture.setFormatter(formatter)
root.addHandler(log_capture)
root.setLevel(logging.INFO)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
root.addHandler(console)

logger = logging.getLogger("Interface")

# Import Core Modules
try:
    from Autonomous_Reasoning_System.control.core_loop import CoreLoop
    from Autonomous_Reasoning_System.io.pdf_ingestor import PDFIngestor
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import core modules. {e}\n")
    sys.exit(1)

# Initialize System
print("Initializing CoreLoop...")
# Warning thread for model download
def hang_warning():
    time.sleep(10)
    print("\n[Note: If waiting here, system is likely downloading the embedding model...]\n")

t = threading.Thread(target=hang_warning, daemon=True)
t.start()

tyrone = CoreLoop(verbose=True) 
ingestor = PDFIngestor()

# Interaction Functions
def chat_interaction(user_message, history):
    if not user_message: return "", history
    history = history + [[user_message, None]]
    try:
        result = tyrone.run_once(user_message)
        response_text = result.get("summary", "(No response)")
        if result.get("decision"):
            intent = result['decision'].get('intent', 'unknown')
            pipeline = result['decision'].get('pipeline', [])
            response_text += f"\n\n*(Intent: {intent} | Pipeline: {pipeline})*"
        history[-1][1] = response_text
    except Exception as e:
        logger.error(f"UI Error: {e}", exc_info=True)
        history[-1][1] = f"⚠️ Error: {e}"
    return "", history

def ingest_files(file_objs):
    if not file_objs: return "No files."
    results = []
    for f in file_objs:
        try:
            path = f.name
            logger.info(f"UI: Ingesting {path}...")
            ingestor.ingest(path, summarize=True)
            results.append(f"✅ Ingested: {Path(path).name}")
        except Exception as e:
            results.append(f"❌ Failed: {Path(path).name} ({str(e)})")
    return "\n".join(results)

def refresh_logs():
    return log_capture.get_logs_as_str()

# --- Gradio Layout (Simplified) ---
# Removing theme argument to ensure compatibility
with gr.Blocks(title="Tyrone ARS") as demo:
    gr.Markdown("# 🧠 Tyrone ARS")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Command")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            gr.Markdown("### 📂 RAG")
            files = gr.File(file_count="multiple")
            status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### 🖥️ Logs")
            logs = gr.Code(language="shell", interactive=False, lines=20)
            timer = gr.Timer(1)

    msg.submit(chat_interaction, [msg, chatbot], [msg, chatbot])
    send.click(chat_interaction, [msg, chatbot], [msg, chatbot])
    files.upload(ingest_files, files, status)
    timer.tick(refresh_logs, outputs=logs)
    clear.click(lambda: None, None, chatbot, queue=False)

# Initialize context
try:
    tyrone.initialize_context()
except Exception:
    pass

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)