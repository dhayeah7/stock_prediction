import pandas as pd
import os
import glob
import logging
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.required_columns = ['date', 'content', 'source']
    
    def clean_text(self, text):
        """Clean text by removing line breaks and extra whitespace"""
        if pd.isna(text):
            return ""
        # Replace line breaks with space and remove extra whitespace
        return ' '.join(str(text).replace('\n', ' ').split())
    
    def process_all_data(self):
        """Process and merge all CSV files from Output directory"""
        try:
            print("Starting data processing...")
            
            # Get all CSV files
            csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*.csv'))
            print(f"Found {len(csv_files)} CSV files")
            
            # List to store all dataframes
            all_dfs = []
            
            for file_path in csv_files:
                try:
                    # Skip system files
                    if file_path.startswith('.') or '~$' in file_path or '.DS_Store' in file_path:
                        continue
                    
                    filename = os.path.basename(file_path)
                    print(f"Processing {filename}...")
                    
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"Skipping empty file: {filename}")
                        continue
                    
                    # Standardize column names
                    df.columns = [col.lower().strip() for col in df.columns]
                    
                    # Ensure required columns exist
                    if not all(col in df.columns for col in self.required_columns):
                        print(f"Skipping {filename} - missing required columns")
                        continue
                    
                    # Keep only required columns
                    df = df[self.required_columns]
                    
                    # Clean content
                    df['content'] = df['content'].apply(self.clean_text)
                    
                    # Remove rows with empty content
                    df = df[df['content'].str.len() > 0]
                    
                    all_dfs.append(df)
                    print(f"Successfully processed {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
            
            if not all_dfs:
                raise ValueError("No valid data found to merge")
            
            # Merge all dataframes
            print("Merging all data...")
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            # Remove duplicates based on content
            print("Removing duplicates...")
            merged_df = merged_df.drop_duplicates(subset=['content'], keep='first')
            
            # Sort by date
            print("Sorting by date...")
            merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
            merged_df = merged_df.sort_values('date')
            
            # Save merged data
            output_file = os.path.join(OUTPUT_DIR, 'merged_cleaned_data.csv')
            print(f"Saving merged data to {output_file}")
            merged_df.to_csv(output_file, index=False)
            
            print(f"Successfully merged {len(merged_df)} rows of data")
            return output_file
            
        except Exception as e:
            print(f"Error in data processing: {e}")
            raise

def main():
    try:
        processor = DataProcessor()
        output_file = processor.process_all_data()
        print("Data processing completed successfully")
    except Exception as e:
        print(f"Data processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
