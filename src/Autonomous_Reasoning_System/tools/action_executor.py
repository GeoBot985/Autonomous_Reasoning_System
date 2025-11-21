# tools/action_executor.py
"""
ActionExecutor
---------------
Bridges Tyrone's planning system to external tools or cognitive functions.
Given a step description, it resolves an appropriate tool and executes it.
"""

from Autonomous_Reasoning_System.memory.storage import MemoryStorage
from Autonomous_Reasoning_System.tools import ocr  # example tool module


class ActionExecutor:
    def __init__(self, memory_storage=None):
        self.memory = memory_storage or MemoryStorage()

    # ---------------------- Dispatcher ----------------------
    def execute_step(self, step_description: str, workspace) -> dict:
        """
        Attempt to execute a single plan step based on its description.
        Returns a structured result dict.
        """
        text = step_description.lower()
        result_text = "No matching tool found."
        success = False

        try:
            if "ocr" in text or "extract text" in text:
                # Example: use OCR module (placeholder)
                image_path = workspace.get("image_path", "data/sample_image.jpg")
                extracted = ocr.run(image_path)
                workspace.set("extracted_text", extracted)
                result_text = f"OCR extracted text of length {len(extracted)}"
                success = True

            elif "load image" in text:
                # Placeholder for an image load step
                workspace.set("image_path", "data/sample_image.jpg")
                result_text = "Loaded sample image successfully."
                success = True

            elif "store" in text and "text" in text:
                text_to_store = workspace.get("extracted_text", "")
                if text_to_store:
                    self.memory.add_memory(
                        text=f"Stored OCR text snippet: {text_to_store[:80]}",
                        memory_type="ocr_result",
                        importance=0.5,
                        source="ActionExecutor",
                    )
                    result_text = "Stored OCR text in long-term memory."
                    success = True

        except Exception as e:
            result_text = f"Error executing step: {e}"

        # Log the attempt
        self.memory.add_memory(
            text=f"Action executed: '{step_description}' → {result_text}",
            memory_type="action_log",
            importance=0.3,
            source="ActionExecutor",
        )

        return {"success": success, "result": result_text}
