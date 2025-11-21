import unittest
from unittest.mock import MagicMock, patch
from Autonomous_Reasoning_System.control.core_loop import CoreLoop
from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter
from Autonomous_Reasoning_System.tools.system_tools import get_current_time, get_current_location

class TestStartupContext(unittest.TestCase):

    @patch('Autonomous_Reasoning_System.control.core_loop.get_current_time')
    @patch('Autonomous_Reasoning_System.control.core_loop.get_current_location')
    def test_initialize_context(self, mock_location, mock_time):
        # Mock system tools
        mock_time.return_value = "2025-01-01 12:00:00"
        mock_location.return_value = "Test City, Test Country"

        # Initialize CoreLoop (mocking dependencies to speed up)
        with patch('Autonomous_Reasoning_System.control.core_loop.Dispatcher'), \
             patch('Autonomous_Reasoning_System.control.core_loop.EmbeddingModel'), \
             patch('Autonomous_Reasoning_System.control.core_loop.VectorStore'), \
             patch('Autonomous_Reasoning_System.control.core_loop.MemoryStorage'), \
             patch('Autonomous_Reasoning_System.control.core_loop.MemoryInterface'), \
             patch('Autonomous_Reasoning_System.control.core_loop.PlanBuilder'), \
             patch('Autonomous_Reasoning_System.control.core_loop.ReflectionInterpreter'), \
             patch('Autonomous_Reasoning_System.control.core_loop.LearningManager'), \
             patch('Autonomous_Reasoning_System.control.core_loop.ConfidenceManager'), \
             patch('Autonomous_Reasoning_System.control.core_loop.start_heartbeat_with_plans'):

            tyrone = CoreLoop()

            # Initially startup context should be empty
            self.assertEqual(tyrone.context_adapter.startup_context, {})

            # Run initialize_context
            tyrone.initialize_context()

            # Verify context
            self.assertEqual(tyrone.context_adapter.startup_context.get("Current Time"), "2025-01-01 12:00:00")
            self.assertEqual(tyrone.context_adapter.startup_context.get("Current Location"), "Test City, Test Country")

    def test_context_adapter_prompt(self):
        adapter = ContextAdapter()
        startup_context = {"Location": "Mars"}
        adapter.set_startup_context(startup_context)

        # Mock retriever to return nothing
        adapter.retriever = MagicMock()
        adapter.retriever.retrieve.return_value = []

        # Mock call_llm
        with patch('Autonomous_Reasoning_System.llm.context_adapter.call_llm') as mock_llm:
            adapter.run("Hello")

            # Verify system prompt contains location
            args, kwargs = mock_llm.call_args
            system_prompt = args[0] if args else kwargs.get('system_prompt')
            self.assertIn("Location: Mars", str(system_prompt))

if __name__ == '__main__':
    unittest.main()
