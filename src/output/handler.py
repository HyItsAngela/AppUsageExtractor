from typing import List, Dict
import logging

class OutputHandler:
    def __init__(self, config: Dict):
        self.generators = []
        self._configure_generators(config)
        
    def _configure_generators(self, config: Dict):
        if config.get('csv'):
            self.generators.append(CSVGenerator(config['csv']))
            
        if config.get('database'):
            db_handler = DatabaseHandler(config['database']['uri'])
            db_handler.configure_schema(config['database'].get('schema'))
            self.generators.append(db_handler)
            
    def save(self, data: Dict) -> None:
        for generator in self.generators:
            try:
                if isinstance(generator, CSVGenerator):
                    generator.generate(data, config['csv']['path'])
                elif isinstance(generator, DatabaseHandler):
                    self._save_to_database(generator, data)
            except Exception as e:
                logging.error(f"Output failed for {type(generator).__name__}: {str(e)}")
                continue
