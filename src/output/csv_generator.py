import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
from .time_utils import parse_time, format_time  

logger = logging.getLogger(__name__)

class CSVGenerator:
    def __init__(self, config: Dict):
        self.required_columns = config.get('required_columns', ["id", "total_usage"])
        self.error_value = config.get('error_value', "-1")
        self.time_format = config.get('time_format', "compact")
        
    def generate(self, extracted_data: Dict[str, List[Dict]], output_path: str) -> None:
        try:
            df = self._create_dataframe(extracted_data)
            self._validate_columns(df)
            self._save_to_disk(df, output_path)
        except Exception as e:
            logger.error(f"CSV generation failed: {str(e)}")
            raise
            
    def _create_dataframe(self, data: Dict) -> pd.DataFrame:
        rows = []
        for session_id, entries in data.items():
            row = {"session_id": session_id}
            total_seconds = 0

            for app in entries:
                seconds = parse_time(app['app_usage'])
                if seconds >= 0:
                    row[app['app_name']] = format_time(seconds, self.time_format)
                    total_seconds += seconds
                else:
                    row[app['app_name']] = self.error_value

            row['total_usage'] = format_time(total_seconds, "compact")
            rows.append(row)

        return pd.DataFrame(rows)
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
    
    def _save_to_disk(self, df: pd.DataFrame, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')