from Autonomous_Reasoning_System.llm.reflection_interpreter import ReflectionInterpreter
import re


class PlanReasoner(ReflectionInterpreter):
    def __init__(self, memory_storage=None, embedding_model=None):
        super().__init__(memory_storage=memory_storage, embedding_model=embedding_model)

    def generate_steps(self, goal_text: str) -> list[str]:
        prompt = f"Decompose this goal into 3-7 concrete, actionable steps:\n\n{goal_text}"
        raw = self.interpret(prompt)
        steps = re.split(r'\n\d+\.|\n-|\n•', str(raw))
        steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 10]
        return steps[:10] or ["Review and clarify the goal."]
