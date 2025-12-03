# ml-service/models/predictor.py - ENHANCED with better predictions

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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
    def __init__(self, model_path=None):
        """
        Initialize the predictor
        """
        self.model_path = model_path
        self.model = None
        
        # Try to load pre-trained model if available
        if model_path:
            try:
                import joblib
                self.model = joblib.load(model_path)
                logger.info('Pre-trained model loaded successfully')
            except Exception as e:
                logger.info('No pre-trained model found, will use rule-based predictions')
    
    def predict(self, symbol, data, indicators, days=7):
        """
        Make a prediction for a stock
        
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
            
            # Calculate prediction using enhanced rule-based system
            direction, confidence, target_price = self._rule_based_prediction(
                current_price, indicators, days
            )
            
            price_change = target_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Generate signals
            signals = self._generate_signals(indicators)
            
            # Build technical analysis summary - ensure all values are scalars
            technical_analysis = {
                'rsi': safe_float(indicators.get('rsi', 50), 50),
                'macd_signal': str(indicators.get('macd_signal', 'Neutral')),
                'volatility': safe_float(indicators.get('volatility', 0), 0),
                'volume_status': str(indicators.get('volume_status', 'Normal'))
            }
            
            return {
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
                'timestamp': datetime.now().isoformat()
            }
            
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
        
        return direction, round(confidence, 1), round(target_price, 2)
    
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