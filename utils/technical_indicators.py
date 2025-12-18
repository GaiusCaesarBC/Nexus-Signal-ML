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

            # Avoid division by zero - when loss is 0, RS is infinite, so RSI = 100
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            # Fill NaN values (where loss was 0) with 100 (max RSI)
            rsi = rsi.fillna(100)

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

            # Avoid division by zero when high == low (flat price)
            price_range = high_max - low_min
            price_range = price_range.replace(0, np.nan)

            k_percent = 100 * ((data['Close'] - low_min) / price_range)
            # Fill NaN values with 50 (neutral) when price range is 0
            k_percent = k_percent.fillna(50)
            d_percent = k_percent.rolling(window=3).mean()

            return k_percent, d_percent
        except Exception as e:
            logger.error(f'Error calculating Stochastic: {str(e)}')
            return pd.Series([50] * len(data)), pd.Series([50] * len(data))

    def calculate_adx(self, data, period=14):
        """
        Calculate Average Directional Index (ADX) - measures trend strength

        ADX > 25: Strong trend
        ADX < 20: Weak trend / ranging
        """
        try:
            # Calculate True Range
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)

            # Calculate directional movement
            plus_dm = data['High'].diff()
            minus_dm = -data['Low'].diff()

            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            # Smoothed values
            atr = true_range.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            # Calculate DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()

            return adx.fillna(25), plus_di.fillna(25), minus_di.fillna(25)
        except Exception as e:
            logger.error(f'Error calculating ADX: {str(e)}')
            return (pd.Series([25] * len(data)),
                    pd.Series([25] * len(data)),
                    pd.Series([25] * len(data)))

    def calculate_williams_r(self, data, period=14):
        """
        Calculate Williams %R - momentum indicator

        Range: -100 to 0
        Below -80: Oversold
        Above -20: Overbought
        """
        try:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()

            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, np.nan)

            williams_r = -100 * ((highest_high - data['Close']) / price_range)
            return williams_r.fillna(-50)
        except Exception as e:
            logger.error(f'Error calculating Williams %R: {str(e)}')
            return pd.Series([-50] * len(data))

    def calculate_cci(self, data, period=20):
        """
        Calculate Commodity Channel Index (CCI)

        Above +100: Overbought
        Below -100: Oversold
        """
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )

            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci.fillna(0)
        except Exception as e:
            logger.error(f'Error calculating CCI: {str(e)}')
            return pd.Series([0] * len(data))

    def calculate_mfi(self, data, period=14):
        """
        Calculate Money Flow Index (MFI) - volume-weighted RSI

        Range: 0 to 100
        Above 80: Overbought
        Below 20: Oversold
        """
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            money_flow = typical_price * data['Volume']

            delta = typical_price.diff()
            positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
            negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()

            mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
            mfi = 100 - (100 / (1 + mfi_ratio))
            return mfi.fillna(50)
        except Exception as e:
            logger.error(f'Error calculating MFI: {str(e)}')
            return pd.Series([50] * len(data))

    def calculate_roc(self, data, period=10):
        """
        Calculate Rate of Change (ROC) - momentum

        Positive: Price rising
        Negative: Price falling
        """
        try:
            roc = ((data['Close'] - data['Close'].shift(period)) /
                   data['Close'].shift(period)) * 100
            return roc.fillna(0)
        except Exception as e:
            logger.error(f'Error calculating ROC: {str(e)}')
            return pd.Series([0] * len(data))

    def calculate_vwap(self, data):
        """
        Calculate Volume Weighted Average Price (VWAP)
        """
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap
        except Exception as e:
            logger.error(f'Error calculating VWAP: {str(e)}')
            return data['Close']

    def calculate_ichimoku(self, data):
        """
        Calculate Ichimoku Cloud components

        Returns:
            tenkan_sen (conversion line): 9-period
            kijun_sen (base line): 26-period
            senkou_span_a (leading span A)
            senkou_span_b (leading span B): 52-period
        """
        try:
            # Tenkan-sen (Conversion Line): 9-period high/low average
            high_9 = data['High'].rolling(window=9).max()
            low_9 = data['Low'].rolling(window=9).min()
            tenkan_sen = (high_9 + low_9) / 2

            # Kijun-sen (Base Line): 26-period high/low average
            high_26 = data['High'].rolling(window=26).max()
            low_26 = data['Low'].rolling(window=26).min()
            kijun_sen = (high_26 + low_26) / 2

            # Senkou Span A (Leading Span A): Average of Tenkan-sen and Kijun-sen
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            # Senkou Span B (Leading Span B): 52-period high/low average
            high_52 = data['High'].rolling(window=52).max()
            low_52 = data['Low'].rolling(window=52).min()
            senkou_span_b = ((high_52 + low_52) / 2).shift(26)

            return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
        except Exception as e:
            logger.error(f'Error calculating Ichimoku: {str(e)}')
            price = data['Close']
            return price, price, price, price

    def calculate_pivot_points(self, data):
        """
        Calculate Pivot Points for support/resistance levels
        """
        try:
            # Use previous day's data
            high = data['High'].shift(1)
            low = data['Low'].shift(1)
            close = data['Close'].shift(1)

            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)

            return pivot, r1, s1, r2, s2
        except Exception as e:
            logger.error(f'Error calculating Pivot Points: {str(e)}')
            price = data['Close']
            return price, price, price, price, price
    
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
            indicators['sma_200'] = self.calculate_sma(data, 200)  # Long-term trend
            indicators['ema_12'] = self.calculate_ema(data, 12)
            indicators['ema_26'] = self.calculate_ema(data, 26)
            indicators['ema_50'] = self.calculate_ema(data, 50)

            # Momentum indicators
            indicators['rsi'] = self.calculate_rsi(data)
            indicators['macd'], indicators['signal'] = self.calculate_macd(data)
            indicators['stoch_k'], indicators['stoch_d'] = self.calculate_stochastic(data)

            # NEW: Additional momentum indicators
            indicators['williams_r'] = self.calculate_williams_r(data)
            indicators['cci'] = self.calculate_cci(data)
            indicators['roc'] = self.calculate_roc(data)
            if 'Volume' in data.columns:
                indicators['mfi'] = self.calculate_mfi(data)

            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.calculate_bollinger_bands(data)
            indicators['atr'] = self.calculate_atr(data)

            # Volume indicators
            indicators['obv'] = self.calculate_obv(data)
            if 'Volume' in data.columns:
                indicators['vwap'] = self.calculate_vwap(data)

            # Trend strength
            indicators['adx'], indicators['plus_di'], indicators['minus_di'] = self.calculate_adx(data)

            # NEW: Ichimoku Cloud
            indicators['tenkan_sen'], indicators['kijun_sen'], indicators['senkou_span_a'], indicators['senkou_span_b'] = self.calculate_ichimoku(data)

            # NEW: Pivot Points
            indicators['pivot'], indicators['r1'], indicators['s1'], indicators['r2'], indicators['s2'] = self.calculate_pivot_points(data)

            # Add derived signals for rule-based prediction
            self._add_derived_signals(data, indicators)

            return indicators

        except Exception as e:
            logger.error(f'Error calculating indicators: {str(e)}')
            return {}

    def _add_derived_signals(self, data, indicators):
        """Add derived signals for rule-based prediction compatibility."""
        try:
            # MACD signal (bullish/bearish crossover)
            macd = indicators.get('macd')
            signal = indicators.get('signal')
            if macd is not None and signal is not None:
                macd_last = float(macd.iloc[-1])
                signal_last = float(signal.iloc[-1])
                macd_prev = float(macd.iloc[-2]) if len(macd) > 1 else macd_last
                signal_prev = float(signal.iloc[-2]) if len(signal) > 1 else signal_last

                if macd_last > signal_last and macd_prev <= signal_prev:
                    indicators['macd_signal'] = 'Bullish Crossover'
                elif macd_last < signal_last and macd_prev >= signal_prev:
                    indicators['macd_signal'] = 'Bearish Crossover'
                elif macd_last > signal_last:
                    indicators['macd_signal'] = 'Bullish'
                elif macd_last < signal_last:
                    indicators['macd_signal'] = 'Bearish'
                else:
                    indicators['macd_signal'] = 'Neutral'

            # Moving average trend
            close = data['Close'].iloc[-1]
            sma_20 = float(indicators['sma_20'].iloc[-1])
            sma_50 = float(indicators['sma_50'].iloc[-1])

            if close > sma_20 and close > sma_50:
                indicators['ma_trend'] = 'Bullish'
            elif close < sma_20 and close < sma_50:
                indicators['ma_trend'] = 'Bearish'
            else:
                indicators['ma_trend'] = 'Neutral'

            # Bollinger Band signal
            bb_upper = float(indicators['bb_upper'].iloc[-1])
            bb_lower = float(indicators['bb_lower'].iloc[-1])

            if close <= bb_lower:
                indicators['bb_signal'] = 'Oversold'
            elif close >= bb_upper:
                indicators['bb_signal'] = 'Overbought'
            else:
                indicators['bb_signal'] = 'Neutral'

            # Volume analysis
            if 'Volume' in data.columns:
                vol = data['Volume']
                vol_avg = vol.rolling(window=20).mean().iloc[-1]
                vol_current = vol.iloc[-1]

                if vol_current > vol_avg * 1.5:
                    indicators['volume_status'] = 'High'
                elif vol_current < vol_avg * 0.5:
                    indicators['volume_status'] = 'Low'
                else:
                    indicators['volume_status'] = 'Normal'

                # Volume trend
                vol_5d_avg = vol.tail(5).mean()
                vol_20d_avg = vol.tail(20).mean()
                if vol_5d_avg > vol_20d_avg:
                    indicators['volume_trend'] = 'Increasing'
                elif vol_5d_avg < vol_20d_avg:
                    indicators['volume_trend'] = 'Decreasing'
                else:
                    indicators['volume_trend'] = 'Neutral'
            else:
                indicators['volume_status'] = 'Normal'
                indicators['volume_trend'] = 'Neutral'

            # Volatility percentage
            atr = indicators.get('atr')
            if atr is not None:
                atr_last = float(atr.iloc[-1])
                indicators['volatility'] = (atr_last / close) * 100
            else:
                indicators['volatility'] = 20  # Default

        except Exception as e:
            logger.error(f'Error adding derived signals: {str(e)}')
            indicators['macd_signal'] = 'Neutral'
            indicators['ma_trend'] = 'Neutral'
            indicators['bb_signal'] = 'Neutral'
            indicators['volume_status'] = 'Normal'
            indicators['volume_trend'] = 'Neutral'
            indicators['volatility'] = 20