# ml-service/app.py - Flask API Server for ML Predictions

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import os
import logging
from datetime import datetime
import json

# Import our prediction modules
from models.predictor import StockPredictor
from utils.technical_indicators import TechnicalIndicators
from utils.market_data import fetch_stock_data
from utils.ai_insights import generate_insights

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with allowed origins from environment variable
allowed_origins = os.getenv('CORS_ALLOWED_ORIGINS', '*').split(',')
CORS(app, origins=allowed_origins)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Authentication
ML_API_KEY = os.getenv('ML_API_KEY')

def require_api_key(f):
    """Decorator to require API key for protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth if no API key is configured (development mode)
        if not ML_API_KEY:
            return f(*args, **kwargs)

        # Check for API key in header
        provided_key = request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')

        if not provided_key:
            logger.warning('API request without API key')
            return jsonify({'error': 'API key required', 'code': 'UNAUTHORIZED'}), 401

        if provided_key != ML_API_KEY:
            logger.warning('API request with invalid API key')
            return jsonify({'error': 'Invalid API key', 'code': 'FORBIDDEN'}), 403

        return f(*args, **kwargs)
    return decorated_function

# Available prediction horizons
AVAILABLE_HORIZONS = {
    7: {'name': 'short', 'description': 'Short-term (7 days)', 'model_dir': 'trained_models/7d'},
    30: {'name': 'medium', 'description': 'Medium-term (30 days)', 'model_dir': 'trained_models/30d'},
    90: {'name': 'long', 'description': 'Long-term (90 days)', 'model_dir': 'trained_models/90d'}
}

# Initialize predictors for each horizon
predictors = {}
for horizon, config in AVAILABLE_HORIZONS.items():
    model_dir = config['model_dir']
    if os.path.exists(model_dir):
        predictors[horizon] = StockPredictor(model_dir=model_dir)
        logger.info(f'Loaded {horizon}-day predictor from {model_dir}')
    else:
        logger.info(f'No trained models found for {horizon}-day horizon at {model_dir}')

# Default predictor (7-day or fallback to trained_models)
if 7 in predictors:
    default_predictor = predictors[7]
elif os.path.exists('trained_models'):
    default_predictor = StockPredictor(model_dir='trained_models')
    predictors[7] = default_predictor
    logger.info('Using default trained_models directory')
else:
    default_predictor = StockPredictor()
    predictors[7] = default_predictor
    logger.info('No trained models found, using untrained predictor')


def get_predictor_for_days(days):
    """Get the appropriate predictor for the given horizon."""
    # Find closest available horizon
    if days in predictors:
        return predictors[days], days

    # Map to nearest horizon
    if days <= 14:
        horizon = 7
    elif days <= 60:
        horizon = 30
    else:
        horizon = 90

    if horizon in predictors:
        return predictors[horizon], horizon

    # Fallback to default
    return default_predictor, 7

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ML Prediction Service',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/horizons', methods=['GET'])
def get_horizons():
    """Get available prediction horizons"""
    horizons = []
    for days, config in AVAILABLE_HORIZONS.items():
        horizons.append({
            'days': days,
            'name': config['name'],
            'description': config['description'],
            'available': days in predictors
        })
    return jsonify({
        'horizons': horizons,
        'default': 7
    })


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Predict price movement for a single stock

    Request body:
    {
        "symbol": "AAPL",
        "days": 7  (optional, default 7 - supported: 7, 30, 90)
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        days = data.get('days', 7)

        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 365:
            days = 7  # Reset to default if invalid

        # Get the appropriate predictor for this horizon
        predictor, actual_horizon = get_predictor_for_days(days)

        logger.info(f'Prediction request for {symbol} ({actual_horizon}d horizon)')

        # Fetch historical data (more for long-term predictions)
        period = '1y' if days > 30 else '6mo'
        stock_data = fetch_stock_data(symbol, period=period)

        if stock_data is None or len(stock_data) < 30:
            return jsonify({
                'error': f'Insufficient data for {symbol}',
                'symbol': symbol
            }), 400

        # Calculate technical indicators
        tech_indicators = TechnicalIndicators()
        indicators = tech_indicators.calculate_all(stock_data)

        # Make prediction using horizon-specific model
        prediction_result = predictor.predict(symbol, stock_data, indicators, actual_horizon)

        # Add horizon info to response
        if prediction_result and isinstance(prediction_result, dict):
            prediction_result['horizon'] = {
                'requested': days,
                'actual': actual_horizon,
                'name': AVAILABLE_HORIZONS.get(actual_horizon, {}).get('name', 'custom')
            }

        return jsonify(prediction_result)
        
    except Exception as e:
        logger.error(f'Error in prediction: {str(e)}')
        return jsonify({'error': 'An error occurred while processing your prediction request'}), 500

@app.route('/predict/batch', methods=['POST'])
@require_api_key
def predict_batch():
    """
    Predict price movements for multiple stocks

    Request body:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "days": 7  (optional, default 7 - supported: 7, 30, 90)
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        days = data.get('days', 7)

        if not symbols or not isinstance(symbols, list):
            return jsonify({'error': 'Symbols array is required'}), 400

        # Validate days parameter
        if not isinstance(days, int) or days < 1 or days > 365:
            days = 7  # Reset to default if invalid

        # Get the appropriate predictor for this horizon
        predictor, actual_horizon = get_predictor_for_days(days)

        logger.info(f'Batch prediction request for {len(symbols)} symbols ({actual_horizon}d horizon)')

        # Reuse single TechnicalIndicators instance for efficiency
        tech_indicators = TechnicalIndicators()

        # Fetch more data for long-term predictions
        period = '1y' if days > 30 else '6mo'

        results = []
        for symbol in symbols:
            try:
                symbol = symbol.upper()
                stock_data = fetch_stock_data(symbol, period=period)

                if stock_data is None or len(stock_data) < 30:
                    results.append({
                        'symbol': symbol,
                        'error': 'Insufficient data'
                    })
                    continue

                indicators = tech_indicators.calculate_all(stock_data)
                prediction_result = predictor.predict(symbol, stock_data, indicators, actual_horizon)

                # Add horizon info
                if prediction_result and isinstance(prediction_result, dict):
                    prediction_result['horizon'] = actual_horizon

                results.append(prediction_result)

            except Exception as e:
                logger.error(f'Error predicting {symbol}: {str(e)}')
                results.append({
                    'symbol': symbol,
                    'error': 'Prediction failed for this symbol'
                })

        return jsonify({
            'predictions': results,
            'total': len(results),
            'horizon': actual_horizon,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f'Error in batch prediction: {str(e)}')
        return jsonify({'error': 'An error occurred while processing batch predictions'}), 500

@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    """
    Deep analysis of a stock with AI-powered insights

    Request body:
    {
        "symbol": "AAPL"
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()

        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        logger.info(f'Analysis request for {symbol}')

        # Fetch data
        stock_data = fetch_stock_data(symbol, period='1y')

        if stock_data is None or len(stock_data) < 30:
            return jsonify({
                'error': f'Insufficient data for {symbol}',
                'symbol': symbol
            }), 400

        # Calculate indicators
        tech_indicators = TechnicalIndicators()
        indicators = tech_indicators.calculate_all(stock_data)

        # Get AI insights
        insights = generate_insights(symbol, stock_data, indicators)

        # Get prediction
        prediction = predictor.predict(symbol, stock_data, indicators, 7)

        return jsonify({
            'symbol': symbol,
            'prediction': prediction,
            'insights': insights,
            'technical_indicators': {
                'rsi': float(indicators['rsi'].iloc[-1]) if 'rsi' in indicators else None,
                'macd': float(indicators['macd'].iloc[-1]) if 'macd' in indicators else None,
                'signal': float(indicators['signal'].iloc[-1]) if 'signal' in indicators else None,
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f'Error in analysis: {str(e)}')
        return jsonify({'error': 'An error occurred while analyzing the stock'}), 500


@app.route('/train', methods=['POST'])
@require_api_key
def train_model():
    """
    Train ML model for a specific stock

    Request body:
    {
        "symbol": "AAPL",
        "days": 7  (optional, prediction horizon)
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        days = data.get('days', 7)

        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        logger.info(f'Training request for {symbol}')

        # Fetch historical data (1 year for training)
        stock_data = fetch_stock_data(symbol, period='1y')

        if stock_data is None or len(stock_data) < 200:
            return jsonify({
                'error': f'Insufficient data for training {symbol}. Need at least 200 data points.',
                'symbol': symbol,
                'data_points': len(stock_data) if stock_data is not None else 0
            }), 400

        # Calculate indicators
        tech_indicators = TechnicalIndicators()
        indicators = tech_indicators.calculate_all(stock_data)

        # Train the model
        metrics = predictor.train(symbol, stock_data, indicators, days)

        if metrics is None:
            return jsonify({
                'error': 'Training failed. ML may not be available.',
                'symbol': symbol
            }), 500

        return jsonify({
            'symbol': symbol,
            'status': 'trained',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f'Error in training: {str(e)}')
        return jsonify({'error': 'An error occurred while training the model'}), 500


@app.route('/train/batch', methods=['POST'])
@require_api_key
def train_batch():
    """
    Train ML models for multiple stocks

    Request body:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "days": 7  (optional)
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        days = data.get('days', 7)

        if not symbols or not isinstance(symbols, list):
            return jsonify({'error': 'Symbols array is required'}), 400

        logger.info(f'Batch training request for {len(symbols)} symbols')

        tech_indicators = TechnicalIndicators()
        results = []

        for symbol in symbols:
            try:
                symbol = symbol.upper()
                stock_data = fetch_stock_data(symbol, period='1y')

                if stock_data is None or len(stock_data) < 200:
                    results.append({
                        'symbol': symbol,
                        'status': 'failed',
                        'error': 'Insufficient data'
                    })
                    continue

                indicators = tech_indicators.calculate_all(stock_data)
                metrics = predictor.train(symbol, stock_data, indicators, days)

                if metrics:
                    results.append({
                        'symbol': symbol,
                        'status': 'trained',
                        'metrics': metrics
                    })
                else:
                    results.append({
                        'symbol': symbol,
                        'status': 'failed',
                        'error': 'Training returned no metrics'
                    })

            except Exception as e:
                logger.error(f'Error training {symbol}: {str(e)}')
                results.append({
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'Training failed for this symbol'
                })

        successful = len([r for r in results if r.get('status') == 'trained'])

        return jsonify({
            'total': len(symbols),
            'successful': successful,
            'failed': len(symbols) - successful,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f'Error in batch training: {str(e)}')
        return jsonify({'error': 'An error occurred while processing batch training'}), 500


@app.route('/model/info', methods=['GET'])
@require_api_key
def model_info():
    """Get information about the ML model"""
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f'Error getting model info: {str(e)}')
        return jsonify({'error': 'An error occurred while retrieving model info'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)