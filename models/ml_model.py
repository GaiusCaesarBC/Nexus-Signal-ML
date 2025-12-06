# ml-service/models/ml_model.py - XGBoost/LightGBM ML Models for Stock Prediction

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

logger = logging.getLogger(__name__)

# Initialize ML library availability flags
XGB_AVAILABLE = False
LGB_AVAILABLE = False
xgb = None
lgb = None

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    logger.info('XGBoost loaded successfully')
except ImportError as e:
    logger.warning(f'XGBoost not available: {e}')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    logger.info('LightGBM loaded successfully')
except ImportError as e:
    logger.warning(f'LightGBM not available: {e}')


class StockMLModel:
    """
    XGBoost/LightGBM-based ML model for stock price prediction.
    Uses technical indicators as features to predict:
    1. Direction (UP/DOWN) - Classification
    2. Price change magnitude - Regression
    """

    def __init__(self, model_dir='trained_models', use_lightgbm=False):
        """
        Initialize the ML model.

        Args:
            model_dir: Directory to save/load trained models
            use_lightgbm: If True, use LightGBM instead of XGBoost
        """
        self.model_dir = model_dir
        self.use_lightgbm = use_lightgbm and LGB_AVAILABLE

        # Models
        self.direction_model = None  # Classification: UP/DOWN
        self.magnitude_model = None  # Regression: price change %
        self.scaler = StandardScaler()

        # Feature configuration
        self.feature_columns = [
            'rsi', 'rsi_change', 'rsi_ma',
            'macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
            'bb_position', 'bb_width', 'bb_squeeze',
            'stoch_k', 'stoch_d', 'stoch_cross',
            'atr', 'atr_percent',
            'obv_change', 'obv_ma_ratio',
            'sma_20_ratio', 'sma_50_ratio', 'ema_12_ratio',
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'volume_ratio', 'volume_ma_ratio',
            'high_low_range', 'close_position',
            'adx', 'plus_di', 'minus_di',
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_10d', 'volatility_20d'
        ]

        # Training config
        self.lookback_days = 5  # Days to look ahead for target
        self.min_training_samples = 200

        # Create model directory if needed
        os.makedirs(model_dir, exist_ok=True)

    def _get_model_path(self, symbol, model_type):
        """Get path for saving/loading model."""
        return os.path.join(self.model_dir, f'{symbol}_{model_type}.joblib')

    def _get_generic_model_path(self, model_type):
        """Get path for generic (all-stock) model."""
        return os.path.join(self.model_dir, f'generic_{model_type}.joblib')

    def engineer_features(self, data, indicators):
        """
        Create feature matrix from price data and technical indicators.

        Args:
            data: DataFrame with OHLCV data
            indicators: Dict with calculated technical indicators

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        features = pd.DataFrame(index=df.index)

        # RSI features
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            features['rsi'] = rsi
            features['rsi_change'] = rsi.diff()
            features['rsi_ma'] = rsi.rolling(window=5).mean()
        else:
            features['rsi'] = 50
            features['rsi_change'] = 0
            features['rsi_ma'] = 50

        # MACD features
        if 'macd' in indicators and 'signal' in indicators:
            macd = indicators['macd']
            signal = indicators['signal']
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = macd - signal
            features['macd_hist_change'] = features['macd_hist'].diff()
        else:
            features['macd'] = 0
            features['macd_signal'] = 0
            features['macd_hist'] = 0
            features['macd_hist_change'] = 0

        # Bollinger Bands features
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            bb_middle = indicators.get('bb_middle', (bb_upper + bb_lower) / 2)

            # Position within bands (0=lower, 1=upper)
            bb_range = bb_upper - bb_lower
            features['bb_position'] = (df['Close'] - bb_lower) / bb_range.replace(0, np.nan)
            features['bb_width'] = bb_range / bb_middle
            features['bb_squeeze'] = features['bb_width'].rolling(window=20).apply(
                lambda x: 1 if x.iloc[-1] == x.min() else 0
            )
        else:
            features['bb_position'] = 0.5
            features['bb_width'] = 0
            features['bb_squeeze'] = 0

        # Stochastic features
        if 'stoch_k' in indicators:
            features['stoch_k'] = indicators['stoch_k']
            features['stoch_d'] = indicators.get('stoch_d', indicators['stoch_k'].rolling(3).mean())
            # Stochastic crossover signal
            features['stoch_cross'] = np.where(
                features['stoch_k'] > features['stoch_d'], 1,
                np.where(features['stoch_k'] < features['stoch_d'], -1, 0)
            )
        else:
            features['stoch_k'] = 50
            features['stoch_d'] = 50
            features['stoch_cross'] = 0

        # ATR features
        if 'atr' in indicators:
            features['atr'] = indicators['atr']
            features['atr_percent'] = indicators['atr'] / df['Close'] * 100
        else:
            features['atr'] = df['High'] - df['Low']
            features['atr_percent'] = features['atr'] / df['Close'] * 100

        # OBV features
        if 'obv' in indicators:
            obv = indicators['obv']
            features['obv_change'] = obv.pct_change()
            features['obv_ma_ratio'] = obv / obv.rolling(window=20).mean()
        else:
            features['obv_change'] = 0
            features['obv_ma_ratio'] = 1

        # Moving average ratios
        features['sma_20_ratio'] = df['Close'] / df['Close'].rolling(window=20).mean()
        features['sma_50_ratio'] = df['Close'] / df['Close'].rolling(window=50).mean()
        features['ema_12_ratio'] = df['Close'] / df['Close'].ewm(span=12).mean()

        # Price momentum
        features['price_momentum_5'] = df['Close'].pct_change(5)
        features['price_momentum_10'] = df['Close'].pct_change(10)
        features['price_momentum_20'] = df['Close'].pct_change(20)

        # Volume features
        if 'Volume' in df.columns:
            vol_ma = df['Volume'].rolling(window=20).mean()
            features['volume_ratio'] = df['Volume'] / vol_ma
            features['volume_ma_ratio'] = vol_ma / vol_ma.rolling(window=50).mean()
        else:
            features['volume_ratio'] = 1
            features['volume_ma_ratio'] = 1

        # Price range features
        features['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)

        # ADX features (trend strength)
        if 'adx' in indicators:
            features['adx'] = indicators['adx']
            features['plus_di'] = indicators.get('plus_di', 25)
            features['minus_di'] = indicators.get('minus_di', 25)
        else:
            features['adx'] = 25
            features['plus_di'] = 25
            features['minus_di'] = 25

        # Returns
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_10d'] = df['Close'].pct_change(10)

        # Volatility
        features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()

        # Fill NaN values
        features = features.ffill().bfill()
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        return features

    def create_targets(self, data, days=5):
        """
        Create target variables for training.

        Args:
            data: DataFrame with OHLCV data
            days: Number of days ahead for prediction

        Returns:
            direction (1=UP, 0=DOWN), magnitude (% change)
        """
        # Future price change
        future_close = data['Close'].shift(-days)
        current_close = data['Close']

        pct_change = (future_close - current_close) / current_close * 100

        # Direction: 1 = UP, 0 = DOWN
        direction = (pct_change > 0).astype(int)

        return direction, pct_change

    def train(self, symbol, data, indicators, days=5, save_model=True):
        """
        Train ML models for a specific stock.

        Args:
            symbol: Stock ticker
            data: Historical OHLCV data (DataFrame)
            indicators: Calculated technical indicators
            days: Prediction horizon in days
            save_model: Whether to save trained models

        Returns:
            Training metrics dict
        """
        # Check if ML libraries are available
        if not self.use_lightgbm and not XGB_AVAILABLE:
            logger.error('XGBoost not available and LightGBM not enabled')
            return None
        if self.use_lightgbm and not LGB_AVAILABLE:
            logger.error('LightGBM not available')
            return None

        logger.info(f'Training ML model for {symbol} with {len(data)} samples')

        # Engineer features
        features = self.engineer_features(data, indicators)

        # Create targets
        direction, magnitude = self.create_targets(data, days)

        # Align and clean data
        valid_idx = ~(direction.isna() | magnitude.isna())
        X = features.loc[valid_idx]
        y_direction = direction.loc[valid_idx]
        y_magnitude = magnitude.loc[valid_idx]

        if len(X) < self.min_training_samples:
            logger.warning(f'Insufficient data for {symbol}: {len(X)} samples')
            return None

        # Use only available feature columns
        available_cols = [c for c in self.feature_columns if c in X.columns]
        X = X[available_cols]

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Train Direction Model (Classification)
        if self.use_lightgbm:
            self.direction_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            )
        else:
            self.direction_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )

        # Train Magnitude Model (Regression)
        if self.use_lightgbm:
            self.magnitude_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            )
        else:
            self.magnitude_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

        # Cross-validation scores
        cv_direction_scores = []
        cv_magnitude_scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_dir_train, y_dir_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
            y_mag_train, y_mag_val = y_magnitude.iloc[train_idx], y_magnitude.iloc[val_idx]

            # Train direction model
            self.direction_model.fit(X_train, y_dir_train)
            dir_pred = self.direction_model.predict(X_val)
            cv_direction_scores.append(accuracy_score(y_dir_val, dir_pred))

            # Train magnitude model
            self.magnitude_model.fit(X_train, y_mag_train)
            mag_pred = self.magnitude_model.predict(X_val)
            cv_magnitude_scores.append(mean_absolute_error(y_mag_val, mag_pred))

        # Final training on all data
        self.direction_model.fit(X_scaled, y_direction)
        self.magnitude_model.fit(X_scaled, y_magnitude)

        metrics = {
            'symbol': symbol,
            'samples': len(X),
            'features': len(available_cols),
            'direction_accuracy': np.mean(cv_direction_scores),
            'direction_std': np.std(cv_direction_scores),
            'magnitude_mae': np.mean(cv_magnitude_scores),
            'magnitude_std': np.std(cv_magnitude_scores),
            'trained_at': datetime.now().isoformat()
        }

        logger.info(f'Training complete for {symbol}: Direction accuracy={metrics["direction_accuracy"]:.2%}, MAE={metrics["magnitude_mae"]:.2f}%')

        # Save models
        if save_model:
            self.save_model(symbol)

        return metrics

    def predict(self, data, indicators, days=5):
        """
        Make prediction using trained ML models.

        Args:
            data: Recent OHLCV data (DataFrame)
            indicators: Calculated technical indicators
            days: Prediction horizon

        Returns:
            Dict with direction, confidence, predicted_change
        """
        if self.direction_model is None or self.magnitude_model is None:
            logger.warning('Models not trained, falling back to rule-based')
            return None

        # Engineer features
        features = self.engineer_features(data, indicators)

        # Get last row (current state)
        X = features.iloc[[-1]]

        # Use only available feature columns
        available_cols = [c for c in self.feature_columns if c in X.columns]
        X = X[available_cols]

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # Predict direction
        direction_prob = self.direction_model.predict_proba(X_scaled)[0]
        direction_pred = self.direction_model.predict(X_scaled)[0]

        # Predict magnitude
        magnitude_pred = self.magnitude_model.predict(X_scaled)[0]

        # Calculate confidence from probability
        confidence = max(direction_prob) * 100

        # Determine direction
        direction = 'UP' if direction_pred == 1 else 'DOWN'

        # Adjust magnitude sign based on direction
        if direction == 'DOWN' and magnitude_pred > 0:
            magnitude_pred = -abs(magnitude_pred)
        elif direction == 'UP' and magnitude_pred < 0:
            magnitude_pred = abs(magnitude_pred)

        return {
            'direction': direction,
            'confidence': round(confidence, 1),
            'predicted_change_percent': round(magnitude_pred, 2),
            'probabilities': {
                'up': round(direction_prob[1] * 100, 1),
                'down': round(direction_prob[0] * 100, 1)
            },
            'model_type': 'lightgbm' if self.use_lightgbm else 'xgboost'
        }

    def save_model(self, symbol='generic'):
        """Save trained models to disk."""
        if self.direction_model is None:
            return

        model_data = {
            'direction_model': self.direction_model,
            'magnitude_model': self.magnitude_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'use_lightgbm': self.use_lightgbm,
            'saved_at': datetime.now().isoformat()
        }

        path = self._get_model_path(symbol, 'full')
        joblib.dump(model_data, path)
        logger.info(f'Model saved to {path}')

    def load_model(self, symbol='generic'):
        """Load trained models from disk."""
        path = self._get_model_path(symbol, 'full')

        if not os.path.exists(path):
            # Try generic model
            path = self._get_generic_model_path('full')
            if not os.path.exists(path):
                logger.info(f'No trained model found for {symbol}')
                return False

        try:
            model_data = joblib.load(path)
            self.direction_model = model_data['direction_model']
            self.magnitude_model = model_data['magnitude_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            logger.info(f'Model loaded from {path}')
            return True
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            return False

    def get_feature_importance(self):
        """Get feature importance from trained models."""
        if self.direction_model is None:
            return None

        importance = pd.DataFrame({
            'feature': self.feature_columns[:len(self.direction_model.feature_importances_)],
            'importance': self.direction_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.to_dict('records')


class EnsemblePredictor:
    """
    Ensemble model combining XGBoost and LightGBM predictions.
    """

    def __init__(self, model_dir='trained_models'):
        self.xgb_model = StockMLModel(model_dir, use_lightgbm=False)
        self.lgb_model = StockMLModel(model_dir, use_lightgbm=True)
        self.model_dir = model_dir

    def train(self, symbol, data, indicators, days=5):
        """Train both models."""
        xgb_metrics = self.xgb_model.train(symbol, data, indicators, days)
        lgb_metrics = self.lgb_model.train(symbol, data, indicators, days)

        return {
            'xgboost': xgb_metrics,
            'lightgbm': lgb_metrics
        }

    def predict(self, data, indicators, days=5):
        """
        Get ensemble prediction by averaging both models.
        """
        xgb_pred = self.xgb_model.predict(data, indicators, days)
        lgb_pred = self.lgb_model.predict(data, indicators, days)

        if xgb_pred is None and lgb_pred is None:
            return None

        if xgb_pred is None:
            return lgb_pred
        if lgb_pred is None:
            return xgb_pred

        # Average probabilities
        avg_up = (xgb_pred['probabilities']['up'] + lgb_pred['probabilities']['up']) / 2
        avg_down = (xgb_pred['probabilities']['down'] + lgb_pred['probabilities']['down']) / 2

        # Average magnitude
        avg_magnitude = (xgb_pred['predicted_change_percent'] + lgb_pred['predicted_change_percent']) / 2

        # Determine direction
        direction = 'UP' if avg_up > avg_down else 'DOWN'
        confidence = max(avg_up, avg_down)

        return {
            'direction': direction,
            'confidence': round(confidence, 1),
            'predicted_change_percent': round(avg_magnitude, 2),
            'probabilities': {
                'up': round(avg_up, 1),
                'down': round(avg_down, 1)
            },
            'model_type': 'ensemble',
            'components': {
                'xgboost': xgb_pred,
                'lightgbm': lgb_pred
            }
        }

    def load_models(self, symbol='generic'):
        """Load both models."""
        xgb_loaded = self.xgb_model.load_model(symbol)
        lgb_loaded = self.lgb_model.load_model(symbol)
        return xgb_loaded or lgb_loaded
