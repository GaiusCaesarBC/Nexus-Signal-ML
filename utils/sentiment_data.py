"""
sentiment_data.py — Fetch sentiment data for ML model features

Sources:
1. Fear & Greed Index (Alternative.me API — free, no key needed)
2. News sentiment from Finnhub (if API key available)

These are used as additional features during model training and prediction.
"""

import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache to avoid repeated API calls during training
_sentiment_cache = {}


def get_fear_greed_index(days=30):
    """
    Fetch Fear & Greed Index history from Alternative.me (crypto-focused but useful for all markets).
    Free API, no key needed.

    Returns:
        DataFrame with columns: Date, fear_greed_value, fear_greed_class
        Values: 0 = Extreme Fear, 100 = Extreme Greed
    """
    cache_key = f'fear_greed_{days}'
    if cache_key in _sentiment_cache:
        cached = _sentiment_cache[cache_key]
        if (datetime.now() - cached['ts']).total_seconds() < 3600:  # 1 hour cache
            return cached['data']

    try:
        url = f'https://api.alternative.me/fng/?limit={days}&format=json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'data' not in data:
            return None

        rows = []
        for entry in data['data']:
            rows.append({
                'Date': pd.to_datetime(datetime.fromtimestamp(int(entry['timestamp']))),
                'fear_greed_value': int(entry['value']),
                'fear_greed_class': entry['value_classification']
            })

        df = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)
        _sentiment_cache[cache_key] = {'data': df, 'ts': datetime.now()}
        return df

    except Exception as e:
        logger.warning(f'Failed to fetch Fear & Greed Index: {e}')
        return None


def get_news_sentiment(symbol, api_key=None):
    """
    Fetch news sentiment from Finnhub for a specific symbol.

    Returns:
        Dict with: sentiment_score (-1 to 1), buzz_score, articles_count
    """
    if not api_key:
        api_key = _get_finnhub_key()
    if not api_key:
        return None

    cache_key = f'news_{symbol}'
    if cache_key in _sentiment_cache:
        cached = _sentiment_cache[cache_key]
        if (datetime.now() - cached['ts']).total_seconds() < 1800:  # 30 min cache
            return cached['data']

    try:
        today = datetime.now().strftime('%Y-%m-%d')
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        url = 'https://finnhub.io/api/v1/news-sentiment'
        params = {'symbol': symbol, 'token': api_key}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data or 'sentiment' not in data:
            return None

        result = {
            'sentiment_score': data['sentiment'].get('bullishPercent', 0.5) - 0.5,  # Center around 0
            'buzz_score': data.get('buzz', {}).get('buzz', 0),
            'articles_count': data.get('buzz', {}).get('articlesInLastWeek', 0),
            'weekly_average': data.get('sentiment', {}).get('weeklyAverage', 0),
        }

        _sentiment_cache[cache_key] = {'data': result, 'ts': datetime.now()}
        return result

    except Exception as e:
        logger.warning(f'Failed to fetch news sentiment for {symbol}: {e}')
        return None


def get_social_sentiment(symbol):
    """
    Estimate social sentiment from price action momentum.
    (Proxy — no external API needed)

    This creates sentiment-like features from price data:
    - Volume spike detection
    - Momentum acceleration
    - Gap frequency

    Returns values that can be used as features alongside real sentiment data.
    """
    # This is a derived feature, not an API call
    # The actual calculation is done in engineer_features
    return None


def build_sentiment_features(symbol, data_length=500):
    """
    Build sentiment feature arrays that can be merged with price data.

    Returns:
        Dict of arrays/values that can be added to the feature matrix
    """
    features = {}

    # 1. Fear & Greed Index
    fg = get_fear_greed_index(days=min(data_length, 365))
    if fg is not None and len(fg) > 0:
        # Use the most recent value as a static feature
        # (Fear & Greed doesn't change intraday, so this is fine for daily models)
        latest = fg.iloc[-1]['fear_greed_value']
        avg_30d = fg.tail(30)['fear_greed_value'].mean() if len(fg) >= 30 else latest
        avg_7d = fg.tail(7)['fear_greed_value'].mean() if len(fg) >= 7 else latest

        features['fear_greed'] = latest
        features['fear_greed_avg_7d'] = avg_7d
        features['fear_greed_avg_30d'] = avg_30d
        features['fear_greed_change'] = latest - avg_30d  # Sentiment momentum
        features['extreme_fear'] = 1 if latest <= 20 else 0
        features['extreme_greed'] = 1 if latest >= 80 else 0
    else:
        features['fear_greed'] = 50
        features['fear_greed_avg_7d'] = 50
        features['fear_greed_avg_30d'] = 50
        features['fear_greed_change'] = 0
        features['extreme_fear'] = 0
        features['extreme_greed'] = 0

    # 2. News sentiment (if Finnhub key available)
    news = get_news_sentiment(symbol)
    if news:
        features['news_sentiment'] = news['sentiment_score']
        features['news_buzz'] = min(news['buzz_score'], 5)  # Cap at 5
        features['news_volume'] = min(news['articles_count'] / 10, 5)  # Normalize
    else:
        features['news_sentiment'] = 0
        features['news_buzz'] = 0
        features['news_volume'] = 0

    return features


def _get_finnhub_key():
    """Get Finnhub API key from environment."""
    import os
    return os.environ.get('FINNHUB_API_KEY')


def clear_cache():
    """Clear the sentiment cache."""
    global _sentiment_cache
    _sentiment_cache = {}
