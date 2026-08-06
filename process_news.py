import pandas as pd
import nltk
import os
import ssl

# Bypass SSL certificate verification for downloads (common macOS Python issue)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK datasets
for dataset in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(dataset, quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download {dataset}: {e}")

from preprocessing.clean_text import clean_text
from preprocessing.tokenization import tokenize_text
from preprocessing.stopword_removal import remove_stopwords
from sentiment.vader_sentiment import get_sentiment
from event_detection.detect_events import detect_event

def process_news():
    input_path = "data/news_data.csv"
    output_path = "data/news_processed.csv"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the news downloader first.")
        return

    print("Loading news data...")
    df = pd.read_csv(input_path)

    # Fill NaN values in content
    df['content'] = df['content'].fillna('')

    print("Cleaning, tokenizing, and removing stopwords...")
    # Clean text
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Tokenize
    df['tokens'] = df['cleaned_content'].apply(tokenize_text)
    
    # Remove stopwords
    df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
    # Join filtered tokens back to string
    df['processed_text'] = df['filtered_tokens'].apply(lambda x: ' '.join(x))

    print("Analyzing sentiment and detecting events...")
    # Calculate sentiment (using cleaned text as VADER handles compound scores well)
    df['sentiment'] = df['cleaned_content'].apply(get_sentiment)
    
    # Detect events
    df['event'] = df['cleaned_content'].apply(detect_event)

    # Save to news_processed.csv
    processed_df = df[['title', 'content', 'date', 'sentiment', 'event']]
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed news data and saved to {output_path}")
    print(processed_df.head())

if __name__ == "__main__":
    process_news()
