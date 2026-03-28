#!/usr/bin/env python3
"""
train_models.py - Script to train ML models on stocks and crypto

Usage:
    python scripts/train_models.py                    # Train on all 100 symbols (50 stocks + 50 crypto)
    python scripts/train_models.py --stocks-only      # Train only on 50 stocks
    python scripts/train_models.py --crypto-only      # Train only on 50 cryptos
    python scripts/train_models.py --symbols AAPL MSFT BTC ETH  # Train on specific symbols
    python scripts/train_models.py --days 7           # Set prediction horizon
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
            # Fetch historical data (2 years for better training)
            stock_data = fetch_stock_data(symbol, period='2y')

            # Rate limiting: Alpha Vantage Pro = 75/min, CoinGecko Pro = 500/min
            if is_crypto(symbol):
                time.sleep(0.5)  # 0.5 second for crypto (CoinGecko Pro has high limits)
            else:
                time.sleep(1)  # 1 second for stocks (Alpha Vantage Pro = 75 calls/min)

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
    parser = argparse.ArgumentParser(description='Train ML models for stocks and crypto')
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=None,
        help='Specific symbols to train on (overrides --stocks-only/--crypto-only)'
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

    args = parser.parse_args()

    # Determine which symbols to train
    if args.symbols:
        symbols = args.symbols
    elif args.stocks_only:
        symbols = DEFAULT_STOCKS
    elif args.crypto_only:
        symbols = DEFAULT_CRYPTO
    else:
        symbols = DEFAULT_SYMBOLS  # All 100

    start_time = datetime.now()
    logger.info('=' * 60)
    logger.info('ML Model Training Script')
    logger.info('=' * 60)

    # Show what we're training
    stock_count = len([s for s in symbols if s in DEFAULT_STOCKS])
    crypto_count = len([s for s in symbols if s in DEFAULT_CRYPTO])
    logger.info(f'Training {len(symbols)} symbols: {stock_count} stocks, {crypto_count} crypto')

    # Create trained_models directory
    os.makedirs('trained_models', exist_ok=True)

    # Run training
    results = train_models(
        symbols=symbols,
        days=args.days,
        use_ensemble=not args.no_ensemble
    )

    # Print summary
    duration = datetime.now() - start_time
    logger.info('=' * 60)
    logger.info('Training Complete')
    logger.info('=' * 60)
    logger.info(f'Total symbols: {len(symbols)}')
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
