from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Detection:
    label: str
    x: int
    y: int
    w: int  
    h: int  
    confidence: float

@dataclass
class MatchResult:
    app_name: Optional['Detection'] = None
    app_usage: Optional['Detection'] = None
    app_icon: Optional['Detection'] = None