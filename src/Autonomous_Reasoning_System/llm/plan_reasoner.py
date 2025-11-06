"""
PlanReasoner
------------
A lightweight LLM wrapper dedicated to goal decomposition and planning.
Extends ReflectionInterpreter but requests raw, structured responses
(numbered steps or JSON lists) for reliable plan generation.
"""

from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter


class PlanReasoner(ReflectionInterpreter):
    def __init__(self):
        super().__init__()

    def generate_steps(self, goal_text: str) -> str:
        """
        Use the underlying LLM to generate a numbered list of steps.
        Returns the raw text output directly.
        """
        prompt = (
            f"You are Tyrone's planning assistant.\n\n"
            f"Break down the goal \"{goal_text}\" into exactly 3 to 6 concise numbered steps.\n\n"
            "Output format:\n"
            "1. Step one\n"
            "2. Step two\n"
            "3. Step three\n"
            "...\n\n"
            "Rules:\n"
            "- Do NOT add explanations, reflections, or commentary.\n"
            "- Return ONLY the numbered list."
        )

        # Call the parent interpreter but request raw LLM output
        try:
            if hasattr(super(), "interpret"):
                # Assume interpret() accepts 'raw' or just return raw text if available
                result = super().interpret(prompt, raw=True)
            else:
                # Fallback to generate() if direct call
                result = self.llm.generate(prompt)
            return str(result)
        except Exception as e:
            print(f"[WARN] PlanReasoner failed: {e}")
            return "1. Define objective\n2. Execute objective\n3. Verify outcome"
