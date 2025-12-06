#!/usr/bin/env python3
"""
train_models.py - Script to train ML models on popular stocks

Usage:
    python scripts/train_models.py                    # Train on default popular stocks
    python scripts/train_models.py --symbols AAPL MSFT GOOGL   # Train on specific symbols
    python scripts/train_models.py --days 7           # Set prediction horizon
    python scripts/train_models.py --generic          # Train a single generic model
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predictor import StockPredictor
from utils.technical_indicators import TechnicalIndicators
from utils.market_data import fetch_stock_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Popular stocks for training
DEFAULT_SYMBOLS = [
    # Tech giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    # Finance
    'JPM', 'BAC', 'GS', 'V', 'MA',
    # Healthcare
    'JNJ', 'UNH', 'PFE',
    # Consumer
    'WMT', 'KO', 'PEP', 'MCD',
    # Energy
    'XOM', 'CVX',
    # Industrial
    'BA', 'CAT',
    # Popular ETFs
    'SPY', 'QQQ', 'IWM'
]


def train_models(symbols, days=7, use_ensemble=True):
    """
    Train ML models for the given symbols.

    Args:
        symbols: List of stock symbols
        days: Prediction horizon in days
        use_ensemble: Whether to use ensemble (XGBoost + LightGBM)

    Returns:
        Dict with training results
    """
    logger.info(f'Starting training for {len(symbols)} symbols')
    logger.info(f'Prediction horizon: {days} days, Ensemble: {use_ensemble}')

    # Initialize predictor
    predictor = StockPredictor(
        model_dir='trained_models',
        use_ml=True,
        use_ensemble=use_ensemble
    )

    tech_indicators = TechnicalIndicators()

    results = {
        'successful': [],
        'failed': [],
        'metrics': []
    }

    for i, symbol in enumerate(symbols):
        logger.info(f'[{i+1}/{len(symbols)}] Training {symbol}...')

        try:
            # Fetch historical data (1 year)
            stock_data = fetch_stock_data(symbol, period='1y')

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

            # Train the model
            metrics = predictor.train(symbol, stock_data, indicators, days)

            if metrics:
                logger.info(f'{symbol}: Training successful')
                if isinstance(metrics, dict):
                    if 'xgboost' in metrics:
                        # Ensemble metrics
                        xgb_acc = metrics['xgboost'].get('direction_accuracy', 0) if metrics['xgboost'] else 0
                        lgb_acc = metrics['lightgbm'].get('direction_accuracy', 0) if metrics['lightgbm'] else 0
                        logger.info(f'  XGBoost accuracy: {xgb_acc:.2%}')
                        logger.info(f'  LightGBM accuracy: {lgb_acc:.2%}')
                    else:
                        acc = metrics.get('direction_accuracy', 0)
                        logger.info(f'  Direction accuracy: {acc:.2%}')

                results['successful'].append(symbol)
                results['metrics'].append({'symbol': symbol, 'metrics': metrics})
            else:
                logger.warning(f'{symbol}: Training failed')
                results['failed'].append({'symbol': symbol, 'error': 'Training returned None'})

        except Exception as e:
            logger.error(f'{symbol}: Error - {str(e)}')
            results['failed'].append({'symbol': symbol, 'error': str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description='Train ML models for stock prediction')
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help='Stock symbols to train on'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=7,
        help='Prediction horizon in days (default: 7)'
    )
    parser.add_argument(
        '--no-ensemble',
        action='store_true',
        help='Use single XGBoost model instead of ensemble'
    )
    parser.add_argument(
        '--generic',
        action='store_true',
        help='Train a single generic model on all data'
    )

    args = parser.parse_args()

    start_time = datetime.now()
    logger.info('=' * 60)
    logger.info('ML Model Training Script')
    logger.info('=' * 60)

    # Create trained_models directory
    os.makedirs('trained_models', exist_ok=True)

    # Run training
    results = train_models(
        symbols=args.symbols,
        days=args.days,
        use_ensemble=not args.no_ensemble
    )

    # Print summary
    duration = datetime.now() - start_time
    logger.info('=' * 60)
    logger.info('Training Complete')
    logger.info('=' * 60)
    logger.info(f'Total symbols: {len(args.symbols)}')
    logger.info(f'Successful: {len(results["successful"])}')
    logger.info(f'Failed: {len(results["failed"])}')
    logger.info(f'Duration: {duration}')

    if results['failed']:
        logger.info('\nFailed symbols:')
        for f in results['failed']:
            logger.info(f"  {f['symbol']}: {f['error']}")

    # Calculate average accuracy
    if results['metrics']:
        accuracies = []
        for m in results['metrics']:
            if isinstance(m['metrics'], dict):
                if 'xgboost' in m['metrics']:
                    # Ensemble - average both
                    xgb = m['metrics']['xgboost']
                    lgb = m['metrics']['lightgbm']
                    if xgb and lgb:
                        avg = (xgb.get('direction_accuracy', 0) + lgb.get('direction_accuracy', 0)) / 2
                        accuracies.append(avg)
                elif 'direction_accuracy' in m['metrics']:
                    accuracies.append(m['metrics']['direction_accuracy'])

        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            logger.info(f'\nAverage direction accuracy: {avg_accuracy:.2%}')

    logger.info('\nModels saved to: trained_models/')

    return 0 if len(results['successful']) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
