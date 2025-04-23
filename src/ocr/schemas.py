from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class OCRResult:
    text: str
    confidence: float
    corrections: List[str]
    debug_info: dict

@dataclass
class TextLine:
    text: str
    confidence: float
    bbox: Tuple[Tuple[float, float], ...] 
    center: Tuple[float, float]