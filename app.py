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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for requests from Node.js backend

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
        
        logger.info(f'Batch prediction request for {len(symbols)} symbols')
        
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
                
                tech_indicators = TechnicalIndicators()
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
        from utils.ai_insights import generate_insights
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
    Retrain the ML model with latest data
    
    Request body:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"]  (optional)
    }
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
        
        logger.info(f'Training model with {len(symbols)} symbols')
        
        # Train the model
        success = predictor.train_model(symbols)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model training failed'
            }), 500
            
    except Exception as e:
        logger.error(f'Error in training: {str(e)}')
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)