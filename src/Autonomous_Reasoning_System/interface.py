import gradio as gr
import logging
import sys
import threading
from pathlib import Path

# Import ARS components
from Autonomous_Reasoning_System.control.core_loop import CoreLoop
from Autonomous_Reasoning_System.io.pdf_ingestor import PDFIngestor
from Autonomous_Reasoning_System.infrastructure.logging_utils import setup_logging

# --- 1. Log Capture for UI ---
class ListHandler(logging.Handler):
    """
    Custom logging handler that sends logs to a list 
    so Gradio can display them in real-time.
    """
    def __init__(self):
        super().__init__()
        self.log_buffer = []
        self.lock = threading.Lock()

    def emit(self, record):
        try:
            msg = self.format(record)
            with self.lock:
                self.log_buffer.append(msg)
                # Keep buffer size manageable
                if len(self.log_buffer) > 500:
                    self.log_buffer.pop(0)
        except Exception:
            self.handleError(record)

    def get_logs_as_str(self):
        with self.lock:
            return "\n".join(reversed(self.log_buffer)) # Newest on top

# Setup logging first, add our custom handler
setup_logging()
log_capture = ListHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_capture.setFormatter(formatter)
logging.getLogger().addHandler(log_capture)
logger = logging.getLogger("Interface")

# --- 2. Initialize System ---
logger.info("Initializing Tyrone Core Loop...")
tyrone = CoreLoop(verbose=True)
ingestor = PDFIngestor()

# --- 3. Interaction Functions ---

def chat_interaction(user_message, history):
    """
    Passes user input to CoreLoop and returns response.
    """
    if not user_message:
        return "", history
    
    # Add user message to history immediately
    history = history + [[user_message, None]]
    
    try:
        # Run the core loop
        result = tyrone.run_once(user_message)
        response_text = result.get("summary", "(No response generated)")
        
        # Append reflection/plan info if available (Optional debugging aid)
        if result.get("decision"):
            intent = result['decision'].get('intent', 'unknown')
            pipeline = result['decision'].get('pipeline', [])
            debug_info = f"\n\n*(Intent: {intent} | Pipeline: {pipeline})*"
            response_text += debug_info

        history[-1][1] = response_text
        
    except Exception as e:
        logger.error(f"UI Error: {e}", exc_info=True)
        history[-1][1] = f"⚠️ System Error: {e}"

    return "", history

def ingest_files(file_objs):
    """
    Handles file drag-and-drop for RAG.
    """
    if not file_objs:
        return "No files selected."
    
    results = []
    for f in file_objs:
        try:
            path = f.name # Gradio passes a temp file path
            logger.info(f"UI: Ingesting {path}...")
            ingestor.ingest(path, summarize=True)
            results.append(f"✅ Ingested: {Path(path).name}")
        except Exception as e:
            logger.error(f"Ingestion failed for {f.name}: {e}")
            results.append(f"❌ Failed: {Path(path).name} ({str(e)})")
            
    return "\n".join(results)

def refresh_logs():
    """Returns the current log buffer string."""
    return log_capture.get_logs_as_str()

# --- 4. Gradio Layout ---

with gr.Blocks(title="Tyrone ARS Debugger", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Tyrone Autonomous Reasoning System")
    
    with gr.Row():
        # LEFT COLUMN: Chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="Interaction")
            msg = gr.Textbox(
                show_label=False, 
                placeholder="Enter your command or query...",
                container=False,
                scale=7
            )
            
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Memory context (UI only)", scale=1)

        # RIGHT COLUMN: Logs & RAG
        with gr.Column(scale=1):
            # File Upload Section
            gr.Markdown("### 📂 RAG Ingestion")
            file_upload = gr.File(
                file_count="multiple", 
                file_types=[".pdf", ".txt"],
                label="Drag & Drop Documents"
            )
            upload_status = gr.Textbox(label="Ingestion Status", interactive=False)
            
            # Log Section
            gr.Markdown("### 🖥️ Live System Logs")
            log_display = gr.Code(
                language="shell", 
                label="Console Output", 
                interactive=False,
                lines=20
            )
            # Auto-refresh logs every 1 second
            log_timer = gr.Timer(1)

    # --- Wiring ---
    
    # Chat
    msg.submit(chat_interaction, [msg, chatbot], [msg, chatbot])
    send_btn.click(chat_interaction, [msg, chatbot], [msg, chatbot])
    
    # Files
    file_upload.upload(ingest_files, file_upload, upload_status)
    
    # Logs
    log_timer.tick(refresh_logs, outputs=log_display)
    
    # Clear UI history (Core memory persists)
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# Launch
if __name__ == "__main__":
    # Initialize context (Feet)
    tyrone.initialize_context()
    
    # Launch on all interfaces so you can access from other devices if needed
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)