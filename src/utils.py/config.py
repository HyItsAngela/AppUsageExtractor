from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class OutputConfig:
    formats: List[str]
    csv_path: Path
    database_uri: Optional[str] = None
    debug: bool = False

@dataclass
class TextCorrectionConfig:
    app_names_json: Path
    min_match_score: int = 80
    char_replacements: Dict[str, str] = None

@dataclass
class AppConfig:
    output: OutputConfig
    text_correction: TextCorrectionConfig
    yolo_model_path: Path

def load_config(path: Path) -> AppConfig:
    """Load and validate full application configuration"""
    try:
        with open(path) as f:
            config_data = yaml.safe_load(f)

        # Output Config
        output_section = config_data.get('output', {})
        output_config = OutputConfig(
            formats=output_section.get('formats', ['csv']),
            csv_path=Path(output_section.get('csv_path', 'output/results.csv')),
            database_uri=output_section.get('database_uri'),
            debug=output_section.get('debug', False)
        )

        # Text Correction Config
        textcorr_section = config_data.get('text_correction', {})
        textcorr_config = TextCorrectionConfig(
            app_names_json=Path(textcorr_section.get(
                'app_names_json', 
                'data/app_names.json'
            )),
            min_match_score=textcorr_section.get('min_match_score', 80),
            char_replacements=textcorr_section.get('char_replacements', {})
        )

        # YOLO Config
        yolo_config = Path(config_data.get('yolo', {}).get(
            'model_path', 
            'models/yolo/weights/best.pt'
        ))

        return AppConfig(
            output=output_config,
            text_correction=textcorr_config,
            yolo_model_path=yolo_config
        )

    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing required config section: {e}")
        raise