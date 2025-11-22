import re
from typing import List, Tuple, Set
from Autonomous_Reasoning_System.memory.sanitizer import MemorySanitizer

class KGValidator:
    """
    Strict semantic gatekeeper for the Knowledge Graph.
    """

    def __init__(self):
        # Simple keyword lists for rejection
        self.subjective_keywords = {"feel", "think", "believe", "prefer", "like", "dislike", "love", "hate"}
        self.ephemeral_keywords = {"today", "yesterday", "tomorrow", "now", "soon", "later"}

        # Basic schema for validation
        self.valid_relations = {
            "controls": {("person", "device"), ("device", "device")},
            "owns": {("person", "object"), ("person", "device")},
            "knows": {("person", "person")},
            "located_in": {("person", "location"), ("object", "location"), ("device", "location")},
            "is_a": {("object", "concept"), ("device", "concept"), ("person", "concept")},
            "has_birthday": {("person", "date"), ("person", "unknown"), ("unknown", "date"), ("unknown", "unknown")}
        }

    def is_valid_content(self, text: str) -> bool:
        """Check if text is valid source for KG."""
        return MemorySanitizer.is_valid_for_kg(text)

    def validate_triple(self, subject: str, relation: str, object_: str, subject_type: str = None, object_type: str = None) -> bool:
        """
        Validates a candidate triple.
        """
        if not subject or not relation or not object_:
            return False

        relation_lower = relation.lower()

        # Reject opinions/feelings (unless stable, but this is a simple heuristic)
        # Check strict match or basic pluralization (e.g., "likes" -> "like")
        if relation_lower in self.subjective_keywords:
             return False

        # Simple stemming for common cases
        if relation_lower.endswith('s') and relation_lower[:-1] in self.subjective_keywords:
             return False

        # Reject ephemeral events
        if any(w in subject.lower() or w in object_.lower() for w in self.ephemeral_keywords):
             return False

        # Deduplication is handled by the unique constraint in DB, but we can check locally if needed.
        # Here we focus on semantic validity.

        # Entity type constraints
        if subject_type and object_type and subject_type != 'unknown' and object_type != 'unknown':
            if relation_lower in self.valid_relations:
                allowed = self.valid_relations[relation_lower]
                # Allow if (s_type, o_type) is in allowed set
                # Also we should be lenient if schema is not exhaustive, but instruction says "Strict semantic gatekeeper"
                # "Reject low-confidence extractions" - maybe implying strictness?
                # "Enforce entity_type constraints (device → controls → device, etc.)"

                # Check if there is a match
                if (subject_type, object_type) not in allowed:
                    # Could be too strict if extraction is noisy, but sticking to requirements
                    # Let's print warning and return False
                    # Or maybe we just return False
                    return False
            else:
                # If relation is not in valid_relations map, do we reject?
                # The user said "Add KG validation rules". If I restrict only to this set, I might miss things.
                # But "Strict semantic gatekeeper" implies whitelist.
                # I'll assume unknown relations are rejected unless added.
                # But for now, I'll allow 'has_birthday' and keep strictness.
                # If relation is not known, maybe we should reject it?
                # "Right now it’s never being called."
                # I'll reject unknown relations to be safe/strict.
                return False

        return True

    def canonicalize(self, name: str) -> str:
        """
        Canonicalize entity names (e.g., lowercase, strip).
        """
        return name.strip().lower()

    def validate_batch(self, triples: List[Tuple]) -> List[Tuple]:
        """
        Validate and canonicalize a batch of triples.
        Triples can be (s, r, o) or (s, s_type, r, o, o_type).
        """
        valid_triples = []
        seen = set()

        for triple in triples:
            if len(triple) == 5:
                s, s_type, r, o, o_type = triple
            elif len(triple) == 3:
                s, r, o = triple
                s_type = "unknown"
                o_type = "unknown"
            else:
                continue

            s_can = self.canonicalize(s)
            r_can = self.canonicalize(r)
            o_can = self.canonicalize(o)

            # Validate with types
            if self.validate_triple(s_can, r_can, o_can, s_type, o_type):
                 # Return consistent 5-tuple
                 triple_key = (s_can, r_can, o_can)
                 if triple_key not in seen:
                     valid_triples.append((s_can, s_type, r_can, o_can, o_type))
                     seen.add(triple_key)

        return valid_triples
