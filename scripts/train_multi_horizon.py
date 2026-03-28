#!/usr/bin/env python3
"""
train_multi_horizon.py - Train ML models for multiple prediction horizons

This script trains separate models for short-term (7d), medium-term (30d),
and long-term (90d) predictions.

Usage:
    python scripts/train_multi_horizon.py                    # Train all horizons on all 100 symbols
    python scripts/train_multi_horizon.py --horizons 7 30    # Train specific horizons
    python scripts/train_multi_horizon.py --stocks-only      # Train only on stocks
    python scripts/train_multi_horizon.py --symbols AAPL MSFT  # Train specific symbols
"""

import sys
import os
import argparse
import logging
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predictor import StockPredictor
from utils.technical_indicators import TechnicalIndicators
from utils.market_data import fetch_stock_data, is_crypto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Top 50 stocks by market cap and trading volume
DEFAULT_STOCKS = [
    # Tech Giants (15)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CRM',
    'AMD', 'ADBE', 'INTC', 'CSCO', 'NFLX',
    # Finance (10)
    'JPM', 'BAC', 'GS', 'V', 'MA', 'WFC', 'C', 'BLK', 'AXP', 'MS',
    # Healthcare (8)
    'JNJ', 'UNH', 'PFE', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT',
    # Consumer (7)
    'WMT', 'KO', 'PEP', 'MCD', 'COST', 'HD', 'NKE',
    # Energy (4)
    'XOM', 'CVX', 'COP', 'SLB',
    # Industrial (3)
    'BA', 'CAT', 'GE',
    # ETFs (3)
    'SPY', 'QQQ', 'IWM'
]

# Top 50 cryptocurrencies by market cap
DEFAULT_CRYPTO = [
    # Top tier (10)
    'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'TRX', 'AVAX', 'DOT',
    # High cap altcoins (15)
    'LINK', 'MATIC', 'SHIB', 'LTC', 'BCH', 'UNI', 'ATOM', 'XLM', 'ETC', 'NEAR',
    'HBAR', 'FIL', 'VET', 'ALGO', 'ICP',
    # DeFi & Layer 2 (10)
    'AAVE', 'MKR', 'GRT', 'CRV', 'COMP', 'ARB', 'OP', 'INJ', 'SUI', 'APT',
    # Mid cap with volume (10)
    'SAND', 'MANA', 'AXS', 'ENJ', 'BAT', 'ZRX', 'SUSHI', '1INCH', 'YFI', 'SNX',
    # Meme coins with liquidity (5)
    'PEPE', 'WIF', 'BONK', 'FLOKI', 'TRUMP'
]

# Combined default (all 100)
DEFAULT_SYMBOLS = DEFAULT_STOCKS + DEFAULT_CRYPTO

# Default prediction horizons
DEFAULT_HORIZONS = [7, 30, 90]

# Horizon descriptions
HORIZON_DESCRIPTIONS = {
    7: 'Short-term (7 days) - Best for swing trades',
    30: 'Medium-term (30 days) - Best for position trades',
    90: 'Long-term (90 days) - Best for trend following'
}


def train_horizon(symbols, days, model_dir):
    """
    Train ML models for a specific prediction horizon.

    Args:
        symbols: List of stock symbols
        days: Prediction horizon in days
        model_dir: Directory to save models

    Returns:
        Dict with training results
    """
    logger.info(f'\n{"="*60}')
    logger.info(f'Training {days}-day prediction models')
    logger.info(f'Model directory: {model_dir}')
    logger.info(f'{"="*60}')

    # Create model directory for this horizon
    os.makedirs(model_dir, exist_ok=True)

    # Initialize predictor with horizon-specific directory
    predictor = StockPredictor(
        model_dir=model_dir,
        use_ml=True,
        use_ensemble=True
    )

    tech_indicators = TechnicalIndicators()

    results = {
        'horizon': days,
        'successful': [],
        'failed': [],
        'metrics': []
    }

    for i, symbol in enumerate(symbols):
        logger.info(f'[{i+1}/{len(symbols)}] Training {symbol} ({days}d)...')

        try:
            # Fetch historical data (2 years for better training)
            stock_data = fetch_stock_data(symbol, period='2y')

            # Rate limiting
            if is_crypto(symbol):
                time.sleep(0.5)
            else:
                time.sleep(1)

            if stock_data is None:
                logger.warning(f'{symbol}: No data available')
                results['failed'].append({'symbol': symbol, 'error': 'No data available'})
                continue

            if len(stock_data) < 200:
                logger.warning(f'{symbol}: Insufficient data ({len(stock_data)} points)')
                results['failed'].append({
                    'symbol': symbol,
                    'error': f'Insufficient data: {len(stock_data)} points'
                })
                continue

            # Calculate indicators
            indicators = tech_indicators.calculate_all(stock_data)

            # Train the model with specified horizon
            metrics = predictor.train(symbol, stock_data, indicators, days)

            if metrics:
                logger.info(f'{symbol}: Training successful')
                if isinstance(metrics, dict):
                    if 'xgboost' in metrics:
                        xgb_acc = metrics['xgboost'].get('direction_accuracy', 0) if metrics['xgboost'] else 0
                        lgb_acc = metrics['lightgbm'].get('direction_accuracy', 0) if metrics['lightgbm'] else 0
                        logger.info(f'  XGBoost: {xgb_acc:.2%}, LightGBM: {lgb_acc:.2%}')
                    else:
                        acc = metrics.get('direction_accuracy', 0)
                        logger.info(f'  Accuracy: {acc:.2%}')

                results['successful'].append(symbol)
                results['metrics'].append({'symbol': symbol, 'metrics': metrics})
            else:
                logger.warning(f'{symbol}: Training failed')
                results['failed'].append({'symbol': symbol, 'error': 'Training returned None'})

        except Exception as e:
            logger.error(f'{symbol}: Error - {str(e)}')
            results['failed'].append({'symbol': symbol, 'error': str(e)})

    return results


def calculate_average_accuracy(results):
    """Calculate average direction accuracy from results."""
    accuracies = []
    for m in results['metrics']:
        if isinstance(m['metrics'], dict):
            if 'xgboost' in m['metrics']:
                xgb = m['metrics']['xgboost']
                lgb = m['metrics']['lightgbm']
                if xgb and lgb:
                    avg = (xgb.get('direction_accuracy', 0) + lgb.get('direction_accuracy', 0)) / 2
                    accuracies.append(avg)
            elif 'direction_accuracy' in m['metrics']:
                accuracies.append(m['metrics']['direction_accuracy'])

    return sum(accuracies) / len(accuracies) if accuracies else 0


def main():
    parser = argparse.ArgumentParser(description='Train ML models for multiple prediction horizons')
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=None,
        help='Specific symbols to train on'
    )
    parser.add_argument(
        '--stocks-only',
        action='store_true',
        help='Train only on stocks (50 symbols)'
    )
    parser.add_argument(
        '--crypto-only',
        action='store_true',
        help='Train only on crypto (50 symbols)'
    )
    parser.add_argument(
        '--horizons', '-H',
        nargs='+',
        type=int,
        default=DEFAULT_HORIZONS,
        help='Prediction horizons in days (default: 7 30 90)'
    )
    parser.add_argument(
        '--no-ensemble',
        action='store_true',
        help='Use single XGBoost model instead of ensemble'
    )

    args = parser.parse_args()

    # Determine which symbols to train
    if args.symbols:
        symbols = args.symbols
    elif args.stocks_only:
        symbols = DEFAULT_STOCKS
    elif args.crypto_only:
        symbols = DEFAULT_CRYPTO
    else:
        symbols = DEFAULT_SYMBOLS

    start_time = datetime.now()
    logger.info('=' * 60)
    logger.info('Multi-Horizon ML Model Training')
    logger.info('=' * 60)

    stock_count = len([s for s in symbols if s in DEFAULT_STOCKS])
    crypto_count = len([s for s in symbols if s in DEFAULT_CRYPTO])
    logger.info(f'Symbols: {len(symbols)} ({stock_count} stocks, {crypto_count} crypto)')
    logger.info(f'Horizons: {args.horizons} days')

    # Train each horizon
    all_results = {}
    for horizon in args.horizons:
        model_dir = f'trained_models/{horizon}d'
        results = train_horizon(symbols, horizon, model_dir)
        all_results[horizon] = results

    # Print summary
    duration = datetime.now() - start_time
    logger.info('\n' + '=' * 60)
    logger.info('TRAINING COMPLETE - SUMMARY')
    logger.info('=' * 60)
    logger.info(f'Total duration: {duration}')
    logger.info('')

    # Summary table
    logger.info(f'{"Horizon":<12} {"Successful":<12} {"Failed":<10} {"Avg Accuracy":<15}')
    logger.info('-' * 50)

    for horizon in args.horizons:
        results = all_results[horizon]
        avg_acc = calculate_average_accuracy(results)
        logger.info(
            f'{horizon} days{"":<6} '
            f'{len(results["successful"]):<12} '
            f'{len(results["failed"]):<10} '
            f'{avg_acc:.2%}'
        )

    logger.info('')
    logger.info('Model directories:')
    for horizon in args.horizons:
        logger.info(f'  {horizon}d: trained_models/{horizon}d/')

    logger.info('')
    logger.info('To use a specific horizon in predictions, specify the model_dir:')
    logger.info('  predictor = StockPredictor(model_dir="trained_models/30d")')

    return 0


if __name__ == '__main__':
    sys.exit(main())
