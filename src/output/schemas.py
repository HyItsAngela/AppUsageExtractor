from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class AppData:
    name: str
    usage: str
    debug_info: Optional[Dict] = None

@dataclass
class IDResult:
    id: str
    apps: List[AppData]
    total_seconds: int = 0