from dataclasses import dataclass, asdict
from typing import Any
import json


@dataclass
class CharacterSpec:
    base_figure: str = "Genesis 9"
    gender: str = "female"
    age: int = 28
    height: float = 0.0
    weight: float = 0.0
    muscularity: float = 0.0
    body_shape: str = "natural"
    skin_tone: str = "medium"
    hair: str = "default"
    outfit: str = "casual"
    outfit_color: str = "black"
    shoes: str = "default"
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterSpec":
        allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for key in ("height", "weight", "muscularity"):
            value = allowed.get(key)
            if isinstance(value, str):
                import re
                match = re.search(r"-?\d+(?:[.,]\d+)?", value)
                if match:
                    allowed[key] = float(match.group(0).replace(",", "."))
        if isinstance(allowed.get("age"), str):
            import re
            match = re.search(r"\d+", allowed["age"])
            if match: allowed["age"] = int(match.group(0))
        return cls(**allowed)
