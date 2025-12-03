from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    FACT = "fact"
    PLAN_SUMMARY = "plan_summary"
    PERSONAL_FACT = "personal_fact"

@dataclass
class KGTriple:
    subject: str
    relation: str
    object: str

@dataclass
class MemoryItem:
    id: str
    text: str
    memory_type: Union[MemoryType, str]
    created_at: datetime
    importance: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional embedding if needed in the model, usually kept separate or in storage
    embedding: Optional[List[float]] = None

class PlanStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress" # generic in progress

@dataclass
class Plan:
    id: str
    goal: str
    steps: List[str]
    status: Union[PlanStatus, str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
