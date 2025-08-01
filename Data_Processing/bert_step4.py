import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import logging
from config import *
import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sentiment_score(label, confidence):
    """
    Convert sentiment label and confidence to a score between -1 and 1
    - Positive sentiment: score between 0 and 1
    - Negative sentiment: score between -1 and 0
    - Neutral sentiment: score close to 0
    """
    if label.lower() == 'positive':
        return confidence
    elif label.lower() == 'negative':
        return -confidence
    else:  # neutral
        return 0

def batch_process_texts(texts, nlp, batch_size=32):
    """Process texts in batches for better performance"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate texts to 512 tokens and remove empty strings
        batch = [str(text)[:512] for text in batch if str(text).strip()]
        if batch:
            try:
                batch_results = nlp(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # If batch fails, process one by one
                for text in batch:
                    try:
                        result = nlp([text])[0]
                        results.append(result)
                    except:
                        results.append({"label": "neutral", "score": 0})
        else:
            results.extend([{"label": "neutral", "score": 0} for _ in batch])
    return results

def calculate_weighted_sentiment(row):
    """Calculate weighted sentiment score based on source credibility"""
    # Get source weight
    source_weight = SOURCE_WEIGHTS.get(row['source'], 0.5)
    
    # Use the raw sentiment score (-1 to 1) and multiply by source weight
    return row['raw_sentiment_score'] * source_weight

def process_sentiment(input_file=None, output_file=None):
    try:
        # Use config paths if none provided
        input_file = input_file or os.path.join(OUTPUT_DIR, 'merged_cleaned_data.csv')
        if input_file.endswith('*.csv'):
            files = sorted(glob.glob(input_file))
            if files:
                input_file = files[-1]
            else:
                raise FileNotFoundError("No merged data file found")
        
        output_file = output_file or os.path.join(OUTPUT_DIR, 'sentiment_analyzed_news.csv')
        
        logger.info("Loading FinBERT model...")
        try:
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Load model and tokenizer
            model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL_NAME,
                num_labels=3
            ).to(device)
            tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            
            # Create pipeline with batch size
            nlp = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device if device.type == 'cuda' else -1
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        logger.info("Loading news data...")
        df = pd.read_csv(input_file)
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows...")
        
        # Process in batches
        texts = df['content'].fillna("").tolist()
        batch_size = 32
        
        logger.info("Processing sentiments in batches...")
        results = batch_process_texts(texts, nlp, batch_size)
        
        # Add sentiment labels
        df['sentiment'] = [r['label'].lower() for r in results]
        
        # Calculate raw sentiment scores (-1 to 1)
        df['raw_sentiment_score'] = [
            get_sentiment_score(r['label'], r['score']) 
            for r in results
        ]
        
        # Calculate weighted sentiment scores
        logger.info("Calculating weighted sentiment scores...")
        df['source_weight'] = df['source'].map(SOURCE_WEIGHTS).fillna(0.5)
        df['weighted_sentiment'] = df.apply(calculate_weighted_sentiment, axis=1)
        
        # Add sentiment strength category
        df['sentiment_strength'] = pd.cut(
            df['raw_sentiment_score'].abs(),
            bins=[0, 0.33, 0.66, 1],
            labels=['weak', 'moderate', 'strong']
        )
        
        # Reorder columns
        columns_order = [
            'date', 
            'content', 
            'source',
            'source_weight',
            'sentiment',
            'sentiment_strength',
            'raw_sentiment_score',
            'weighted_sentiment'
        ]
        df = df[columns_order]
        
        logger.info("Saving results...")
        df.to_csv(output_file, index=False)
        
        # Print summary
        print("\nSentiment Analysis Summary:")
        print(f"Total processed: {len(df)}")
        
        print("\nSentiment Distribution:")
        sentiment_dist = df['sentiment'].value_counts()
        print(sentiment_dist)
        
        print("\nSentiment Strength Distribution:")
        strength_dist = df['sentiment_strength'].value_counts()
        print(strength_dist)
        
        print("\nAverage Scores by Source:")
        summary = df.groupby('source').agg({
            'raw_sentiment_score': ['mean', 'count'],
            'weighted_sentiment': 'mean',
            'source_weight': 'first'
        }).round(3)
        print(summary)
        
        # Save summary to a separate file
        summary_file = os.path.join(OUTPUT_DIR, 'sentiment_summary.csv')
        summary.to_csv(summary_file)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise

if __name__ == "__main__":
    process_sentiment()
