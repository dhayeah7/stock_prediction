import pandas as pd
import logging
import os
import glob
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.validation_results = {}
    
    def validate_file(self, filepath, required_columns=None):
        """Validate a single CSV file's structure and content"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False
            
            df = pd.read_csv(filepath)
            
            # Use default columns if none specified
            if required_columns is None:
                required_columns = ['date', 'content', 'source']
            
            # Check columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {filepath}: {missing_cols}")
                return False
            
            # Check for empty dataframe
            if len(df) == 0:
                logger.error(f"Empty dataset in {filepath}")
                return False
            
            # Validate date format
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                logger.error(f"Invalid date format in {filepath}: {e}")
                return False
            
            # Validate content
            if df['content'].isna().any():
                logger.warning(f"Found {df['content'].isna().sum()} rows with missing content in {filepath}")
            
            # Validate source
            if df['source'].isna().any():
                logger.error(f"Found {df['source'].isna().sum()} rows with missing source in {filepath}")
                return False
            
            # Store validation result
            self.validation_results[filepath] = {
                'status': 'valid',
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': f"{df['date'].min()} to {df['date'].max()}"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")
            self.validation_results[filepath] = {
                'status': 'invalid',
                'error': str(e)
            }
            return False
    
    def validate_all_data(self):
        """Validate all data files in the pipeline"""
        all_valid = True
        
        # Validate source files
        for source, columns in REQUIRED_COLUMNS.items():
            pattern = os.path.join(OUTPUT_DIR, f"{source}*.csv")
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                logger.warning(f"No files found for {source}")
                all_valid = False
                continue
            
            # Validate most recent file
            latest_file = max(matching_files, key=os.path.getctime)
            if not self.validate_file(latest_file, columns):
                all_valid = False
        
        # Validate processed files
        processed_patterns = {
            'merged': 'merged_data_*.csv',
            'sentiment': 'sentiment_analyzed_*.csv'
        }
        
        for stage, pattern in processed_patterns.items():
            files = glob.glob(os.path.join(OUTPUT_DIR, pattern))
            if files:
                latest_file = max(files, key=os.path.getctime)
                if not self.validate_file(latest_file):
                    all_valid = False
            else:
                logger.warning(f"No {stage} files found")
                all_valid = False
        
        return all_valid
    
    def get_validation_report(self):
        """Generate a validation report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_files': len(self.validation_results),
                'valid_files': sum(1 for r in self.validation_results.values() if r['status'] == 'valid'),
                'invalid_files': sum(1 for r in self.validation_results.values() if r['status'] == 'invalid')
            },
            'details': self.validation_results
        }
        return report

def main():
    validator = DataValidator()
    if validator.validate_all_data():
        logger.info("All data files validated successfully")
        report = validator.get_validation_report()
        logger.info(f"Validation Report:\n{report}")
        sys.exit(0)
    else:
        logger.error("Data validation failed")
        report = validator.get_validation_report()
        logger.error(f"Validation Report:\n{report}")
        sys.exit(1)

if __name__ == "__main__":
    main()
