from datetime import datetime
from pathlib import Path
import hashlib
from typing import List, Dict

class FileManager:
    def organize_by_id(self, id_text: str, image_path: str, data: List[Dict]) -> Path:
        clean_id = self._sanitize_id(id_text)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{clean_id}_{self._file_hash(image_path)}"
        id_folder = self.id_folders_root / unique_id
        
        id_folder.mkdir(parents=True, exist_ok=True)
        self._save_versioned_copy(image_path, id_folder)
        self._save_metadata(id_folder, data, timestamp)
        
        return id_folder
    
    def _file_hash(self, path: str) -> str:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    
    def _save_versioned_copy(self, src: str, dest_folder: Path) -> None:
        stem = Path(src).stem
        timestamp = datetime.now().strftime("%H%M%S")
        dest_path = dest_folder / f"{stem}_{timestamp}{Path(src).suffix}"
        cv2.imwrite(str(dest_path), cv2.imread(src))
