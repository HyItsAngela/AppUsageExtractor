from tqdm import tqdm
from typing import Iterable, Optional
import yaml
from pathlib import Path
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_progress_config() -> dict:
    config_path = Path("configs/settings.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f).get('progress', {})
    except Exception as e:
        logger.warning(f"Failed to load progress config: {str(e)}")
        return {'enabled': True, 'style': 'detailed'}

def get_progress_bar(
    iterable: Iterable, 
    description: str = "Processing",
    unit: str = "items"
) -> Iterable:
    """Configurable progress bar with fallback"""
    config = _load_progress_config()
    
    if not config.get('enabled', True):
        return iterable
        
    style_config = {
        'basic': {
            'bar_format': "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        },
        'detailed': {
            'bar_format': (
                "{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
        }
    }
    
    return tqdm(
        iterable,
        desc=description,
        unit=unit,
        **style_config.get(config.get('style', 'detailed'), {})
    )