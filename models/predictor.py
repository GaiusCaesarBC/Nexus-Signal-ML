# ml-service/models/predictor.py - ENHANCED with ML predictions + rule-based fallback

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# Try to import ML model
try:
    from models.ml_model import StockMLModel, EnsemblePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning('ML models not available, using rule-based predictions only')


def safe_float(val, default=0):
    """Safely convert value to float, handling pandas Series and None values"""
    if val is None:
        return default
    if hasattr(val, 'iloc'):
        return float(val.iloc[-1]) if len(val) > 0 else default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class StockPredictor:
    def __init__(self, model_dir='trained_models', use_ml=True, use_ensemble=True):
        """
        Initialize the predictor with ML models.

        Args:
            model_dir: Directory containing trained models
            use_ml: Whether to use ML predictions (falls back to rule-based if False or no model)
            use_ensemble: If True, use ensemble of XGBoost+LightGBM; else just XGBoost
        """
        self.model_dir = model_dir
        self.use_ml = use_ml and ML_AVAILABLE
        self.use_ensemble = use_ensemble
        self.ml_model = None
        self._loaded_symbols = set()

        # Initialize ML model
        if self.use_ml:
            try:
                if use_ensemble:
                    self.ml_model = EnsemblePredictor(model_dir)
                else:
                    self.ml_model = StockMLModel(model_dir, use_lightgbm=False)
                logger.info(f'ML model initialized (ensemble={use_ensemble})')
            except Exception as e:
                logger.warning(f'Failed to initialize ML model: {e}')
                self.use_ml = False

    def _try_load_model(self, symbol):
        """Try to load a trained model for the symbol."""
        if not self.use_ml or self.ml_model is None:
            return False

        if symbol in self._loaded_symbols:
            return True

        # Try symbol-specific model first, then generic
        if hasattr(self.ml_model, 'load_models'):
            loaded = self.ml_model.load_models(symbol)
        else:
            loaded = self.ml_model.load_model(symbol)

        if loaded:
            self._loaded_symbols.add(symbol)

        return loaded
    
    def predict(self, symbol, data, indicators, days=7):
        """
        Make a prediction for a stock using ML model or rule-based fallback.

        Args:
            symbol: Stock ticker
            data: Historical price data (DataFrame)
            indicators: Technical indicators (dict)
            days: Prediction timeframe in days

        Returns:
            Dictionary with prediction results
        """
        try:
            current_price = float(data['Close'].iloc[-1])
            ml_prediction = None
            prediction_method = 'rule_based'

            # Try ML prediction first
            if self.use_ml and self.ml_model is not None:
                self._try_load_model(symbol)
                try:
                    ml_prediction = self.ml_model.predict(data, indicators, days)
                    if ml_prediction is not None:
                        prediction_method = ml_prediction.get('model_type', 'ml')
                except Exception as e:
                    logger.warning(f'ML prediction failed for {symbol}: {e}')
                    ml_prediction = None

            # Use ML prediction or fall back to rule-based
            if ml_prediction is not None:
                direction = ml_prediction['direction']
                confidence = ml_prediction['confidence']
                price_change_percent = ml_prediction['predicted_change_percent']

                # Calculate target price from predicted change
                if direction == 'UP':
                    target_price = current_price * (1 + abs(price_change_percent) / 100)
                else:
                    target_price = current_price * (1 - abs(price_change_percent) / 100)
            else:
                # Fall back to rule-based prediction
                direction, confidence, target_price = self._rule_based_prediction(
                    current_price, indicators, days
                )
                price_change_percent = ((target_price - current_price) / current_price) * 100

            # Generate signals
            signals = self._generate_signals(indicators)

            # Build technical analysis summary - ensure all values are scalars
            technical_analysis = {
                'rsi': safe_float(indicators.get('rsi', 50), 50),
                'macd_signal': str(indicators.get('macd_signal', 'Neutral')),
                'volatility': safe_float(indicators.get('volatility', 0), 0),
                'volume_status': str(indicators.get('volume_status', 'Normal'))
            }

            result = {
                'symbol': symbol,
                'current_price': current_price,
                'prediction': {
                    'direction': direction,
                    'confidence': confidence,
                    'target_price': target_price,
                    'price_change_percent': price_change_percent,
                    'timeframe_days': days
                },
                'signals': signals,
                'technical_analysis': technical_analysis,
                'prediction_method': prediction_method,
                'timestamp': datetime.now().isoformat()
            }

            # Include ML-specific data if available
            if ml_prediction is not None and 'probabilities' in ml_prediction:
                result['ml_probabilities'] = ml_prediction['probabilities']

            return result

        except Exception as e:
            logger.error(f'Error making prediction for {symbol}: {str(e)}')
            raise
    
    def _rule_based_prediction(self, current_price, indicators, days):
        """
        Enhanced rule-based prediction with better confidence scoring
        """
        # Initialize scores
        bullish_score = 0
        bearish_score = 0
        confidence_factors = []
        
        # RSI Analysis (strong signal)
        rsi = indicators.get('rsi', 50)
        # Ensure it's a scalar value
        if hasattr(rsi, 'iloc'):
            rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50
        else:
            rsi = float(rsi) if rsi is not None else 50
            
        if rsi < 30:
            bullish_score += 30
            confidence_factors.append(15)
        elif rsi < 40:
            bullish_score += 15
            confidence_factors.append(10)
        elif rsi > 70:
            bearish_score += 30
            confidence_factors.append(15)
        elif rsi > 60:
            bearish_score += 15
            confidence_factors.append(10)
        else:
            confidence_factors.append(5)  # Neutral RSI = low confidence
        
        # MACD Analysis (strong signal)
        macd_signal = indicators.get('macd_signal', 'Neutral')
        if 'bullish' in macd_signal.lower():
            bullish_score += 25
            confidence_factors.append(12)
        elif 'bearish' in macd_signal.lower():
            bearish_score += 25
            confidence_factors.append(12)
        else:
            confidence_factors.append(3)
        
        # Moving Average Trend (medium signal)
        ma_trend = indicators.get('ma_trend', 'Neutral')
        if 'bullish' in ma_trend.lower():
            bullish_score += 20
            confidence_factors.append(10)
        elif 'bearish' in ma_trend.lower():
            bearish_score += 20
            confidence_factors.append(10)
        else:
            confidence_factors.append(5)
        
        # Volume Analysis (medium signal)
        volume_status = indicators.get('volume_status', 'Normal')
        volume_trend = indicators.get('volume_trend', 'Neutral')
        if volume_status == 'High':
            if 'increasing' in volume_trend.lower():
                # High volume with increase confirms trend
                if bullish_score > bearish_score:
                    bullish_score += 15
                    confidence_factors.append(8)
                elif bearish_score > bullish_score:
                    bearish_score += 15
                    confidence_factors.append(8)
            else:
                confidence_factors.append(5)
        else:
            confidence_factors.append(3)  # Low volume = lower confidence
        
        # Bollinger Bands (medium signal)
        bb_signal = indicators.get('bb_signal', 'Neutral')
        if 'oversold' in bb_signal.lower():
            bullish_score += 15
            confidence_factors.append(8)
        elif 'overbought' in bb_signal.lower():
            bearish_score += 15
            confidence_factors.append(8)
        else:
            confidence_factors.append(4)
        
        # Stochastic (weak signal but additive)
        stoch = indicators.get('stoch_k', 50)
        if hasattr(stoch, 'iloc'):
            stoch = float(stoch.iloc[-1]) if len(stoch) > 0 else 50
        else:
            stoch = float(stoch) if stoch is not None else 50
            
        if stoch < 20:
            bullish_score += 10
            confidence_factors.append(5)
        elif stoch > 80:
            bearish_score += 10
            confidence_factors.append(5)
        
        # ADX Trend Strength (confidence modifier)
        adx = indicators.get('adx', 25)
        if hasattr(adx, 'iloc'):
            adx = float(adx.iloc[-1]) if len(adx) > 0 else 25
        else:
            adx = float(adx) if adx is not None else 25
            
        if adx > 40:
            # Strong trend - increases confidence
            confidence_factors.append(10)
        elif adx > 25:
            confidence_factors.append(5)
        else:
            # Weak trend - decreases confidence
            confidence_factors.append(-5)
        
        # Determine direction and base confidence
        total_score = bullish_score - bearish_score
        
        if total_score > 15:
            direction = 'UP'
            base_confidence = min(bullish_score, 85)  # Cap at 85%
        elif total_score < -15:
            direction = 'DOWN'
            base_confidence = min(bearish_score, 85)
        else:
            direction = 'NEUTRAL'
            base_confidence = 40 + abs(total_score)  # 40-55% for neutral
        
        # Calculate final confidence from factors
        confidence = base_confidence + (sum(confidence_factors) / len(confidence_factors))
        confidence = max(30, min(95, confidence))  # Clamp between 30-95%
        
        # Calculate target price with volatility consideration
        volatility = indicators.get('volatility', 20)
        if hasattr(volatility, 'iloc'):
            volatility = float(volatility.iloc[-1]) if len(volatility) > 0 else 20
        else:
            volatility = float(volatility) if volatility is not None else 20
        volatility = volatility / 100
        
        if direction == 'UP':
            # Bullish target
            base_change = 0.02 + (confidence / 100) * 0.08  # 2-10% based on confidence
            volatility_factor = 1 + (volatility * 0.5)  # More volatile = bigger moves
            price_change_pct = base_change * volatility_factor
            target_price = current_price * (1 + price_change_pct)
            
        elif direction == 'DOWN':
            # Bearish target
            base_change = 0.02 + (confidence / 100) * 0.08
            volatility_factor = 1 + (volatility * 0.5)
            price_change_pct = base_change * volatility_factor
            target_price = current_price * (1 - price_change_pct)
            
        else:
            # Neutral - use small deterministic adjustment based on recent trend
            # Instead of random, use a small percentage of volatility
            price_change_pct = volatility * 0.1  # Small move based on volatility
            target_price = current_price * (1 + price_change_pct)
        
        return direction, round(confidence, 1), target_price
    
    def _generate_signals(self, indicators):
        """
        Generate human-readable trading signals
        """
        signals = []

        # RSI signals
        rsi = safe_float(indicators.get('rsi', 50), 50)
        if rsi < 30:
            signals.append(f"RSI oversold at {rsi:.1f} (bullish reversal signal)")
        elif rsi > 70:
            signals.append(f"RSI overbought at {rsi:.1f} (bearish reversal signal)")
        elif rsi < 40:
            signals.append(f"RSI at {rsi:.1f} (moderately bullish)")
        elif rsi > 60:
            signals.append(f"RSI at {rsi:.1f} (moderately bearish)")
        
        # MACD signals
        macd_signal = indicators.get('macd_signal', 'Neutral')
        if 'bullish' in macd_signal.lower():
            signals.append("MACD bullish crossover detected")
        elif 'bearish' in macd_signal.lower():
            signals.append("MACD bearish crossover detected")
        
        # Moving Average signals
        ma_trend = indicators.get('ma_trend', 'Neutral')
        if 'bullish' in ma_trend.lower():
            signals.append("Price above key moving averages (bullish)")
        elif 'bearish' in ma_trend.lower():
            signals.append("Price below key moving averages (bearish)")
        
        # Volume signals
        volume_status = indicators.get('volume_status', 'Normal')
        volume_trend = indicators.get('volume_trend', 'Neutral')
        if volume_status == 'High' and 'increasing' in volume_trend.lower():
            signals.append("High volume with increasing trend (confirms momentum)")
        elif volume_status == 'Low':
            signals.append("Below average volume (weak conviction)")
        
        # Bollinger Bands
        bb_signal = indicators.get('bb_signal', 'Neutral')
        if 'oversold' in bb_signal.lower():
            signals.append("Price at lower Bollinger Band (potential bounce)")
        elif 'overbought' in bb_signal.lower():
            signals.append("Price at upper Bollinger Band (potential pullback)")
        
        # Stochastic
        stoch = safe_float(indicators.get('stoch_k', 50), 50)
        if stoch < 20:
            signals.append(f"Stochastic oversold at {stoch:.1f} (bullish)")
        elif stoch > 80:
            signals.append(f"Stochastic overbought at {stoch:.1f} (bearish)")
        
        # ADX Trend Strength
        adx = safe_float(indicators.get('adx', 25), 25)
        if adx > 40:
            signals.append(f"Strong trend detected (ADX: {adx:.1f})")
        elif adx < 20:
            signals.append(f"Weak trend - ranging market (ADX: {adx:.1f})")
        
        # Volatility warning
        volatility = safe_float(indicators.get('volatility', 20), 20)
        if volatility > 40:
            signals.append(f"High volatility ({volatility:.1f}%) - increased risk")
        
        return signals if signals else ['No strong signals detected']
    
    def batch_predict(self, predictions_data, days=7):
        """
        Make predictions for multiple stocks

        Args:
            predictions_data: List of (symbol, data, indicators) tuples
            days: Prediction timeframe

        Returns:
            List of prediction dictionaries
        """
        results = []

        for symbol, data, indicators in predictions_data:
            try:
                prediction = self.predict(symbol, data, indicators, days)
                results.append(prediction)
            except Exception as e:
                logger.error(f'Error predicting {symbol}: {str(e)}')
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })

        return results

    def train(self, symbol, data, indicators, days=7):
        """
        Train the ML model for a specific stock.

        Args:
            symbol: Stock ticker
            data: Historical OHLCV data
            indicators: Technical indicators
            days: Prediction horizon

        Returns:
            Training metrics or None if training failed
        """
        if not self.use_ml or self.ml_model is None:
            logger.warning('ML not available, cannot train')
            return None

        try:
            if hasattr(self.ml_model, 'train'):
                # Single model training
                metrics = self.ml_model.train(symbol, data, indicators, days)
            else:
                # Ensemble training (trains both XGBoost and LightGBM)
                metrics = self.ml_model.train(symbol, data, indicators, days)

            if metrics:
                self._loaded_symbols.add(symbol)

            return metrics
        except Exception as e:
            logger.error(f'Training failed for {symbol}: {e}')
            return None

    def train_generic(self, training_data, days=7):
        """
        Train a generic model on multiple stocks' data.

        Args:
            training_data: List of (symbol, data, indicators) tuples
            days: Prediction horizon

        Returns:
            Combined training metrics
        """
        if not self.use_ml or self.ml_model is None:
            logger.warning('ML not available, cannot train')
            return None

        all_metrics = []
        for symbol, data, indicators in training_data:
            try:
                metrics = self.train(symbol, data, indicators, days)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                logger.error(f'Training failed for {symbol}: {e}')

        return {
            'total_trained': len(all_metrics),
            'symbols': [m.get('symbol') if isinstance(m, dict) else 'unknown' for m in all_metrics],
            'metrics': all_metrics
        }

    def get_model_info(self):
        """Get information about the loaded ML model."""
        info = {
            'ml_available': ML_AVAILABLE,
            'ml_enabled': self.use_ml,
            'ensemble_mode': self.use_ensemble,
            'loaded_symbols': list(self._loaded_symbols),
            'model_dir': self.model_dir
        }

        if self.use_ml and self.ml_model is not None:
            if hasattr(self.ml_model, 'get_feature_importance'):
                info['feature_importance'] = self.ml_model.get_feature_importance()

        return info