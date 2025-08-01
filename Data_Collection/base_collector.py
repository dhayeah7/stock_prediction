import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from config import *

class BaseCollector(ABC):
    def __init__(self, source_name):
        self.logger = logging.getLogger(source_name)
        self.source_name = source_name
    
    @abstractmethod
    def collect(self):
        """Implement this method in each collector"""
        pass
    
    def get_cutoff_date(self, lookback_days=None):
        """Get cutoff date for data collection"""
        days = lookback_days or DEFAULT_LOOKBACK_DAYS
        return datetime.now() - timedelta(days=days)
    
    def save_to_csv(self, df, prefix):
        """Save data to CSV file"""
        output_file = get_output_filename(prefix)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {len(df)} items to {output_file}")
        return output_file
