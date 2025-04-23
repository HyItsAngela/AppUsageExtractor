from sqlalchemy import create_engine, event, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import List
import logging

Base = declarative_base()
logger = logging.getLogger(__name__)

class AppUsageRecord(Base):
    __tablename__ = 'app_usage'

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_id = Column(String, nullable=False)
    app_name = Column(String, nullable=False)
    usage_time = Column(String, nullable=False)
    source_image = Column(String, nullable=True)

    def __repr__(self):
        return f"<AppUsageRecord(phone_id={self.phone_id}, app_name={self.app_name}, usage_time={self.usage_time})>"

class DatabaseHandler:
    def __init__(self, db_uri: str):
        self.engine = create_engine(db_uri)
        event.listen(self.engine, 'connect', self._set_sqlite_pragma)
        self.Session = sessionmaker(bind=self.engine)  
        self.session = self.Session() 
        self._create_tables()

    def _set_sqlite_pragma(self, dbapi_connection, connection_record):
        """Enable foreign key constraints for SQLite"""
        if 'sqlite' in self.engine.url.drivername:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    def _create_tables(self):
        """Safe table creation"""
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.error(f"Table creation failed: {str(e)}")

    def bulk_save(self, records: List[AppUsageRecord]) -> None:
        """Efficient bulk insert"""
        try:
            self.session.bulk_save_objects(records)
            self.session.commit()
            logger.info("Bulk save successful.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Bulk save failed: {str(e)}")
