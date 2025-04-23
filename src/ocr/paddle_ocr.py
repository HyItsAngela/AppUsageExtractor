import numpy as np
from paddleocr import PaddleOCR
from typing import Tuple, List
from .schemas import TextLine, OCRResult

class PaddleOCRWrapper:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    def extract_text(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> OCRResult:
        """Improved text extraction with region handling and confidence calculation"""
        x, y, w, h = region
        cropped = image[y:y+h, x:x+w]
        
        result = self.ocr.ocr(cropped, cls=True)
        
        text_lines = []
        total_confidence = 0
        valid_blocks = 0
        
        for block in (result[0] or []):
            text, confidence = block[1]
            points = [(int(x + p[0]), int(y + p[1])) for p in block[0]]
            center = (
                sum(p[0] for p in points) / 4,
                sum(p[1] for p in points) / 4
            )
            
            text_lines.append(TextLine(
                text=text,
                confidence=confidence,
                bbox=tuple(points),
                center=center
            ))
            
            if confidence > 0.3:
                total_confidence += confidence
                valid_blocks += 1
                
        avg_confidence = total_confidence / valid_blocks if valid_blocks else 0
        full_text = ' '.join([tl.text for tl in text_lines])
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            corrections=[],
            debug_info={'text_blocks': text_lines}
        )