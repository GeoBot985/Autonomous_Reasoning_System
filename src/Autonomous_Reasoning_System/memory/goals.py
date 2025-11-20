from dataclasses import dataclass, field, asdict
from datetime import datetime
from uuid import uuid4
from typing import List, Optional, Dict
import json

@dataclass
class Goal:
    text: str
    priority: int = 1
    status: str = "pending"  # pending, active, paused, completed, failed
    steps: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        # Handle steps and metadata if they are JSON strings (from DB)
        if isinstance(data.get('steps'), str):
            try:
                data['steps'] = json.loads(data['steps'])
            except:
                data['steps'] = []
        if isinstance(data.get('metadata'), str):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except:
                data['metadata'] = {}

        return cls(**data)
