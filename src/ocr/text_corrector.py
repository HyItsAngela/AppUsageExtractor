from fuzzywuzzy import fuzz, process
import re
from typing import List, Optional, Tuple
from .constants import OCR_CHAR_REPLACEMENTS
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TextCorrector:
    def __init__(
        self, 
        app_names: Optional[List[str]] = None, 
        json_path: Path = Path("data/scraped_app_names-cleaned.json"),
        min_match_score: int = 80
    ):
        # Preserve original case in self.app_names
        self.app_names = app_names or self.load_app_names(json_path)
        
        # Lowercase versions for matching only
        self._processed_names = [name.lower() for name in self.app_names]
        
        self.min_match_score = min_match_score
        
    @staticmethod
    def load_app_names(json_path: Path) -> List[str]:
        """Load app names from JSON file with validation"""
        try:
            if not json_path.exists():
                raise FileNotFoundError(f"App names JSON not found: {json_path}")
                
            with open(json_path) as f:
                names = json.load(f)
                
            if not isinstance(names, list):
                raise ValueError("JSON file should contain a list of strings")
                
            return [str(name).strip() for name in names if name.strip()]
            
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load app names: {str(e)}")
            return []
    
    def correct_text(
        self,
        text: str,
        region_type: str
    ) -> Tuple[str, List[str]]:
        """Maintain original fuzzy matching for names, character replacement only for usage"""
        text = text.strip()
        corrections = []
        
        if region_type == "app_name" and self.app_names:
            corrected, match_score = self._correct_app_name(text)
            if corrected != text:
                corrections.append(f"Name correction ({match_score}): {text}→{corrected}")
            return corrected, corrections
        
        elif region_type == "app_usage":
            # Apply character replacements only to usage
            corrected = self._correct_usage(text)
            if corrected != text:
                corrections.append(f"Usage correction: {text}→{corrected}")
            return corrected, corrections
            
        return text, corrections

    def _correct_app_name(self, text: str) -> Tuple[str, int]:
        """Correct app name using fuzzy matching"""
        text_lower = text.lower()
        result = process.extractOne(
            text_lower,
            self._processed_names,
            scorer=fuzz.token_sort_ratio
        )
        
        if not result or result[1] < self.min_match_score:
            return text, 0
            
        # Get original case version
        matched_idx = self._processed_names.index(result[0])
        return self.app_names[matched_idx], result[1]

    def _correct_usage(self, text: str) -> str:
        """Apply character replacements and format validation"""
        # Apply replacements first
        corrected = ''.join([OCR_CHAR_REPLACEMENTS.get(c, c) for c in text])
        
        if re.fullmatch(r"^(\d+h)?(\d+m)?(\d+s)?$", corrected):
            return corrected
            
        numbers = re.findall(r"\d+", corrected)
        if not numbers:
            return text  # Return original if no numbers
            
        return ''.join(f"{n}{'hms'[i]}" for i, n in enumerate(numbers[-3:]))