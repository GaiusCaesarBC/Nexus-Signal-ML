# ml-service/app.py - Flask API Server for ML Predictions

from flask import Flask, request, jsonify
from flask_cors import CORS
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

# Initialize predictor
predictor = StockPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ML Prediction Service',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict price movement for a single stock
    
    Request body:
    {
        "symbol": "AAPL",
        "days": 7  (optional, default 7)
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

        logger.info(f'Prediction request for {symbol}')
        
        # Fetch historical data
        stock_data = fetch_stock_data(symbol, period='6mo')
        
        if stock_data is None or len(stock_data) < 30:
            return jsonify({
                'error': f'Insufficient data for {symbol}',
                'symbol': symbol
            }), 400
        
        # Calculate technical indicators
        tech_indicators = TechnicalIndicators()
        indicators = tech_indicators.calculate_all(stock_data)
        
        # Make prediction
        prediction_result = predictor.predict(symbol, stock_data, indicators, days)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        logger.error(f'Error in prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict price movements for multiple stocks
    
    Request body:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "days": 7  (optional, default 7)
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

        logger.info(f'Batch prediction request for {len(symbols)} symbols')

        # Reuse single TechnicalIndicators instance for efficiency
        tech_indicators = TechnicalIndicators()

        results = []
        for symbol in symbols:
            try:
                symbol = symbol.upper()
                stock_data = fetch_stock_data(symbol, period='6mo')

                if stock_data is None or len(stock_data) < 30:
                    results.append({
                        'symbol': symbol,
                        'error': 'Insufficient data'
                    })
                    continue

                indicators = tech_indicators.calculate_all(stock_data)
                prediction_result = predictor.predict(symbol, stock_data, indicators, days)
                results.append(prediction_result)
                
            except Exception as e:
                logger.error(f'Error predicting {symbol}: {str(e)}')
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f'Error in batch prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
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
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
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
        return jsonify({'error': str(e)}), 500


@app.route('/train/batch', methods=['POST'])
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
                    'error': str(e)
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
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the ML model"""
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f'Error getting model info: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)