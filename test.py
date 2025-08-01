import os
import pandas as pd
from config import OUTPUT_DIR

def test_output_files():
    """Test if all expected output files exist and have data"""
    # Get latest files for each source
    def get_latest_file(pattern):
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
        return files[-1] if files else None
    
    sources = {
        'NewsAPI': '*newsapi*.csv',
        'Economic Times': '*et_collector*.csv',
        'StockTwits': '*stocktwits*.csv',
        'Yahoo Finance': '*yahoo_finance*.csv',
        'Reuters': '*reuters*.csv',
        'MarketWatch': '*marketwatch*.csv',
        'Merged Data': 'merged_data*.csv',
        'Sentiment Analysis': 'sentiment_analyzed*.csv'
    }
    
    results = {}
    for source, pattern in sources.items():
        file = get_latest_file(pattern)
        if file:
            df = pd.read_csv(file)
            results[source] = {
                'file': os.path.basename(file),
                'rows': len(df),
                'columns': list(df.columns)
            }
        else:
            results[source] = {'error': 'File not found'}
    
    return results

def main():
    print("Testing pipeline outputs...")
    results = test_output_files()
    
    print("\nResults:")
    for source, data in results.items():
        print(f"\n{source}:")
        if 'error' in data:
            print(f"  Error: {data['error']}")
        else:
            print(f"  File: {data['file']}")
            print(f"  Rows: {data['rows']}")
            print(f"  Columns: {', '.join(data['columns'])}")

if __name__ == "__main__":
    main()
