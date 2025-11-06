from Autonomous_Reasoning_System.llm.context_adapter import ContextAdapter


adapter = ContextAdapter()
response = adapter.run("What was I working on yesterday?")
print(response)
