# ml-service/utils/technical_indicators.py - Technical Analysis Indicators

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Calculate technical indicators for stock analysis
    """
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f'Error calculating RSI: {str(e)}')
            return pd.Series([50] * len(data))
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        MACD = 12-day EMA - 26-day EMA
        Signal = 9-day EMA of MACD
        """
        try:
            exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
            
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            return macd, signal_line
        except Exception as e:
            logger.error(f'Error calculating MACD: {str(e)}')
            return pd.Series([0] * len(data)), pd.Series([0] * len(data))
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Middle Band = 20-day SMA
        Upper Band = Middle Band + (2 × standard deviation)
        Lower Band = Middle Band - (2 × standard deviation)
        """
        try:
            middle_band = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return upper_band, middle_band, lower_band
        except Exception as e:
            logger.error(f'Error calculating Bollinger Bands: {str(e)}')
            return (pd.Series([0] * len(data)), 
                   pd.Series([0] * len(data)), 
                   pd.Series([0] * len(data)))
    
    def calculate_sma(self, data, period=20):
        """Calculate Simple Moving Average"""
        try:
            return data['Close'].rolling(window=period).mean()
        except Exception as e:
            logger.error(f'Error calculating SMA: {str(e)}')
            return pd.Series([0] * len(data))
    
    def calculate_ema(self, data, period=20):
        """Calculate Exponential Moving Average"""
        try:
            return data['Close'].ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f'Error calculating EMA: {str(e)}')
            return pd.Series([0] * len(data))
    
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR) - measures volatility
        """
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean()
            
            return atr
        except Exception as e:
            logger.error(f'Error calculating ATR: {str(e)}')
            return pd.Series([0] * len(data))
    
    def calculate_obv(self, data):
        """
        Calculate On-Balance Volume (OBV)
        
        OBV measures buying and selling pressure
        """
        try:
            obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
            return obv
        except Exception as e:
            logger.error(f'Error calculating OBV: {str(e)}')
            return pd.Series([0] * len(data))
    
    def calculate_stochastic(self, data, period=14):
        """
        Calculate Stochastic Oscillator
        
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) × 100
        """
        try:
            low_min = data['Low'].rolling(window=period).min()
            high_max = data['High'].rolling(window=period).max()
            
            k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=3).mean()
            
            return k_percent, d_percent
        except Exception as e:
            logger.error(f'Error calculating Stochastic: {str(e)}')
            return pd.Series([50] * len(data)), pd.Series([50] * len(data))
    
    def calculate_all(self, data):
        """
        Calculate all technical indicators
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Dictionary of indicators
        """
        try:
            indicators = {}
            
            # Trend indicators
            indicators['sma_20'] = self.calculate_sma(data, 20)
            indicators['sma_50'] = self.calculate_sma(data, 50)
            indicators['ema_12'] = self.calculate_ema(data, 12)
            indicators['ema_26'] = self.calculate_ema(data, 26)
            
            # Momentum indicators
            indicators['rsi'] = self.calculate_rsi(data)
            indicators['macd'], indicators['signal'] = self.calculate_macd(data)
            indicators['stoch_k'], indicators['stoch_d'] = self.calculate_stochastic(data)
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.calculate_bollinger_bands(data)
            indicators['atr'] = self.calculate_atr(data)
            
            # Volume indicators
            indicators['obv'] = self.calculate_obv(data)
            
            return indicators
            
        except Exception as e:
            logger.error(f'Error calculating indicators: {str(e)}')
            return {}