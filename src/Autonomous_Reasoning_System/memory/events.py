from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class MemoryCreatedEvent:
    text: str
    timestamp: str
    source: str
    memory_id: str
    metadata: Dict[str, Any]
