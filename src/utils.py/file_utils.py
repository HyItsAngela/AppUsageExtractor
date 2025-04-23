import os
import re
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

ID_PATTERN = re.compile(
    r'(?i)^.*?(OWL\d{4,})'
    r'(?:Phone.*?)?'  
    r'(?:\..+)?$' 
)

def extract_id_from_filename(image_path: Path) -> Optional[str]:
    try:
        filename = image_path.name
        match = ID_PATTERN.match(filename)
        
        if not match:
            logger.debug(f"No ID found in filename: {filename}")
            return None
            
        raw_id = match.group(1).upper()  # Normalize to uppercase
        clean_id = raw_id.replace(" ", "").strip()
        
        if not re.match(r'^OWL\d{4,}$', clean_id):
            logger.warning(f"Invalid ID format: {clean_id} from {filename}")
            return None
            
        return clean_id
        
    except Exception as e:
        logger.error(f"ID extraction failed for {image_path}: {str(e)}")
        return None