# ml-service/models/ml_model.py - XGBoost/LightGBM ML Models for Stock Prediction
# ENHANCED: Feature selection, hyperparameter tuning, improved targets

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

logger = logging.getLogger(__name__)

# Try to import Optuna for hyperparameter tuning
OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    logger.info('Optuna loaded successfully for hyperparameter tuning')
except ImportError:
    logger.warning('Optuna not available - using default hyperparameters')

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

    def __init__(self, model_dir='trained_models', use_lightgbm=False,
                 n_features=30, target_threshold=0.3, use_tuning=True):
        """
        Initialize the ML model.

        Args:
            model_dir: Directory to save/load trained models
            use_lightgbm: If True, use LightGBM instead of XGBoost
            n_features: Number of top features to select (default 30)
            target_threshold: Minimum % change to classify as UP/DOWN (default 0.3%)
            use_tuning: Whether to use Optuna hyperparameter tuning
        """
        self.model_dir = model_dir
        self.use_lightgbm = use_lightgbm and LGB_AVAILABLE

        # NEW: Feature selection config
        self.n_features = n_features
        self.feature_selector = None
        self.selected_feature_names = None

        # NEW: Target threshold to filter noise
        self.target_threshold = target_threshold

        # NEW: Hyperparameter tuning
        self.use_tuning = use_tuning and OPTUNA_AVAILABLE
        self.best_params = None

        # Models
        self.direction_model = None  # Classification: UP/DOWN
        self.magnitude_model = None  # Regression: price change %
        self.scaler = StandardScaler()

        # Feature configuration - EXPANDED with new indicators
        self.feature_columns = [
            # RSI features
            'rsi', 'rsi_change', 'rsi_ma', 'rsi_divergence',
            # MACD features
            'macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
            # Bollinger Bands features
            'bb_position', 'bb_width', 'bb_squeeze', 'bb_trend',
            # Stochastic features
            'stoch_k', 'stoch_d', 'stoch_cross',
            # NEW: Williams %R
            'williams_r', 'williams_r_change',
            # NEW: CCI
            'cci', 'cci_change',
            # NEW: MFI
            'mfi', 'mfi_change',
            # NEW: ROC
            'roc', 'roc_change',
            # ATR features
            'atr', 'atr_percent', 'atr_change',
            # OBV features
            'obv_change', 'obv_ma_ratio',
            # NEW: VWAP features
            'vwap_ratio', 'vwap_distance',
            # Moving average ratios - expanded
            'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
            'ema_12_ratio', 'ema_50_ratio',
            'ma_cross_20_50', 'ma_cross_50_200',
            # Price momentum - short term
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            # LONG-TERM: Extended momentum for 30d/90d predictions
            'price_momentum_30', 'price_momentum_60', 'price_momentum_90',
            # Volume features
            'volume_ratio', 'volume_ma_ratio', 'volume_trend',
            # Price range features
            'high_low_range', 'close_position', 'gap',
            # Trend strength (ADX)
            'adx', 'plus_di', 'minus_di', 'di_cross',
            # Returns - short term
            'returns_1d', 'returns_5d', 'returns_10d',
            # LONG-TERM: Extended returns
            'returns_20d', 'returns_30d', 'returns_60d',
            # Volatility - short term
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            # LONG-TERM: Extended volatility
            'volatility_30d', 'volatility_60d',
            # NEW: Ichimoku features
            'ichimoku_signal', 'cloud_thickness', 'price_vs_cloud',
            # NEW: Pivot point features
            'pivot_distance', 'near_support', 'near_resistance',
            # NEW: Lagged features
            'rsi_lag_1', 'rsi_lag_5', 'macd_lag_1',
            # Day of week (for patterns)
            'day_of_week',
            # LONG-TERM: Seasonality features
            'month_of_year', 'quarter',
            # LONG-TERM: 52-week high/low distance
            'distance_52w_high', 'distance_52w_low',
            # LONG-TERM: Trend persistence
            'trend_strength_30d', 'trend_consistency',
            # LONG-TERM: Mean reversion signals
            'mean_reversion_20d', 'mean_reversion_50d',
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
        EXPANDED with new indicators for improved accuracy.

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
            # RSI divergence (price up but RSI down = bearish divergence)
            price_change = df['Close'].pct_change(5)
            rsi_change_5 = rsi.diff(5)
            features['rsi_divergence'] = np.where(
                (price_change > 0) & (rsi_change_5 < 0), -1,  # Bearish
                np.where((price_change < 0) & (rsi_change_5 > 0), 1, 0)  # Bullish
            )
            # Lagged RSI
            features['rsi_lag_1'] = rsi.shift(1)
            features['rsi_lag_5'] = rsi.shift(5)
        else:
            features['rsi'] = 50
            features['rsi_change'] = 0
            features['rsi_ma'] = 50
            features['rsi_divergence'] = 0
            features['rsi_lag_1'] = 50
            features['rsi_lag_5'] = 50

        # MACD features
        if 'macd' in indicators and 'signal' in indicators:
            macd = indicators['macd']
            signal = indicators['signal']
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = macd - signal
            features['macd_hist_change'] = features['macd_hist'].diff()
            features['macd_lag_1'] = macd.shift(1)
        else:
            features['macd'] = 0
            features['macd_signal'] = 0
            features['macd_hist'] = 0
            features['macd_hist_change'] = 0
            features['macd_lag_1'] = 0

        # Bollinger Bands features
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            bb_middle = indicators.get('bb_middle', (bb_upper + bb_lower) / 2)

            bb_range = bb_upper - bb_lower
            features['bb_position'] = (df['Close'] - bb_lower) / bb_range.replace(0, np.nan)
            features['bb_width'] = bb_range / bb_middle
            features['bb_squeeze'] = features['bb_width'].rolling(window=20).apply(
                lambda x: 1 if len(x) > 0 and x.iloc[-1] == x.min() else 0, raw=False
            )
            # BB trend (expanding or contracting)
            features['bb_trend'] = features['bb_width'].diff()
        else:
            features['bb_position'] = 0.5
            features['bb_width'] = 0
            features['bb_squeeze'] = 0
            features['bb_trend'] = 0

        # Stochastic features
        if 'stoch_k' in indicators:
            features['stoch_k'] = indicators['stoch_k']
            features['stoch_d'] = indicators.get('stoch_d', indicators['stoch_k'].rolling(3).mean())
            features['stoch_cross'] = np.where(
                features['stoch_k'] > features['stoch_d'], 1,
                np.where(features['stoch_k'] < features['stoch_d'], -1, 0)
            )
        else:
            features['stoch_k'] = 50
            features['stoch_d'] = 50
            features['stoch_cross'] = 0

        # NEW: Williams %R features
        if 'williams_r' in indicators:
            wr = indicators['williams_r']
            features['williams_r'] = wr
            features['williams_r_change'] = wr.diff()
        else:
            features['williams_r'] = -50
            features['williams_r_change'] = 0

        # NEW: CCI features
        if 'cci' in indicators:
            cci = indicators['cci']
            features['cci'] = cci
            features['cci_change'] = cci.diff()
        else:
            features['cci'] = 0
            features['cci_change'] = 0

        # NEW: MFI features
        if 'mfi' in indicators:
            mfi = indicators['mfi']
            features['mfi'] = mfi
            features['mfi_change'] = mfi.diff()
        else:
            features['mfi'] = 50
            features['mfi_change'] = 0

        # NEW: ROC features
        if 'roc' in indicators:
            roc = indicators['roc']
            features['roc'] = roc
            features['roc_change'] = roc.diff()
        else:
            features['roc'] = 0
            features['roc_change'] = 0

        # ATR features
        if 'atr' in indicators:
            atr = indicators['atr']
            features['atr'] = atr
            features['atr_percent'] = atr / df['Close'] * 100
            features['atr_change'] = atr.diff()
        else:
            features['atr'] = df['High'] - df['Low']
            features['atr_percent'] = features['atr'] / df['Close'] * 100
            features['atr_change'] = features['atr'].diff()

        # OBV features
        if 'obv' in indicators:
            obv = indicators['obv']
            features['obv_change'] = obv.pct_change()
            features['obv_ma_ratio'] = obv / obv.rolling(window=20).mean()
        else:
            features['obv_change'] = 0
            features['obv_ma_ratio'] = 1

        # NEW: VWAP features
        if 'vwap' in indicators:
            vwap = indicators['vwap']
            features['vwap_ratio'] = df['Close'] / vwap
            features['vwap_distance'] = (df['Close'] - vwap) / vwap * 100
        else:
            features['vwap_ratio'] = 1
            features['vwap_distance'] = 0

        # Moving average ratios - expanded
        sma_20 = df['Close'].rolling(window=20).mean()
        sma_50 = df['Close'].rolling(window=50).mean()
        sma_200 = df['Close'].rolling(window=200).mean()
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_50 = df['Close'].ewm(span=50).mean()

        features['sma_20_ratio'] = df['Close'] / sma_20
        features['sma_50_ratio'] = df['Close'] / sma_50
        features['sma_200_ratio'] = df['Close'] / sma_200
        features['ema_12_ratio'] = df['Close'] / ema_12
        features['ema_50_ratio'] = df['Close'] / ema_50

        # MA crossover signals
        features['ma_cross_20_50'] = np.where(sma_20 > sma_50, 1, -1)
        features['ma_cross_50_200'] = np.where(sma_50 > sma_200, 1, -1)

        # Price momentum - short term
        features['price_momentum_5'] = df['Close'].pct_change(5)
        features['price_momentum_10'] = df['Close'].pct_change(10)
        features['price_momentum_20'] = df['Close'].pct_change(20)

        # LONG-TERM: Extended momentum for 30d/90d predictions
        features['price_momentum_30'] = df['Close'].pct_change(30)
        features['price_momentum_60'] = df['Close'].pct_change(60)
        features['price_momentum_90'] = df['Close'].pct_change(90)

        # Volume features
        if 'Volume' in df.columns:
            vol_ma_20 = df['Volume'].rolling(window=20).mean()
            vol_ma_50 = df['Volume'].rolling(window=50).mean()
            features['volume_ratio'] = df['Volume'] / vol_ma_20
            features['volume_ma_ratio'] = vol_ma_20 / vol_ma_50
            features['volume_trend'] = df['Volume'].rolling(5).mean() / vol_ma_20
        else:
            features['volume_ratio'] = 1
            features['volume_ma_ratio'] = 1
            features['volume_trend'] = 1

        # Price range features
        features['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
        features['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # ADX features
        if 'adx' in indicators:
            features['adx'] = indicators['adx']
            features['plus_di'] = indicators.get('plus_di', 25)
            features['minus_di'] = indicators.get('minus_di', 25)
            # DI crossover signal
            features['di_cross'] = np.where(
                indicators.get('plus_di', 25) > indicators.get('minus_di', 25), 1, -1
            )
        else:
            features['adx'] = 25
            features['plus_di'] = 25
            features['minus_di'] = 25
            features['di_cross'] = 0

        # Returns - short term
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_10d'] = df['Close'].pct_change(10)

        # LONG-TERM: Extended returns
        features['returns_20d'] = df['Close'].pct_change(20)
        features['returns_30d'] = df['Close'].pct_change(30)
        features['returns_60d'] = df['Close'].pct_change(60)

        # Volatility - short term
        features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()

        # LONG-TERM: Extended volatility
        features['volatility_30d'] = df['Close'].pct_change().rolling(30).std()
        features['volatility_60d'] = df['Close'].pct_change().rolling(60).std()

        # NEW: Ichimoku features
        if 'tenkan_sen' in indicators and 'kijun_sen' in indicators:
            tenkan = indicators['tenkan_sen']
            kijun = indicators['kijun_sen']
            span_a = indicators.get('senkou_span_a', tenkan)
            span_b = indicators.get('senkou_span_b', kijun)

            # Ichimoku signal: 1=bullish, -1=bearish
            features['ichimoku_signal'] = np.where(tenkan > kijun, 1, -1)
            # Cloud thickness
            features['cloud_thickness'] = (span_a - span_b) / df['Close'] * 100
            # Price vs cloud
            cloud_top = np.maximum(span_a, span_b)
            cloud_bottom = np.minimum(span_a, span_b)
            features['price_vs_cloud'] = np.where(
                df['Close'] > cloud_top, 1,
                np.where(df['Close'] < cloud_bottom, -1, 0)
            )
        else:
            features['ichimoku_signal'] = 0
            features['cloud_thickness'] = 0
            features['price_vs_cloud'] = 0

        # NEW: Pivot point features
        if 'pivot' in indicators:
            pivot = indicators['pivot']
            r1 = indicators.get('r1', pivot)
            s1 = indicators.get('s1', pivot)

            features['pivot_distance'] = (df['Close'] - pivot) / pivot * 100
            features['near_support'] = (df['Close'] - s1).abs() / df['Close'] < 0.02
            features['near_resistance'] = (df['Close'] - r1).abs() / df['Close'] < 0.02
        else:
            features['pivot_distance'] = 0
            features['near_support'] = 0
            features['near_resistance'] = 0

        # NEW: Day of week (for patterns - Monday=0, Friday=4)
        if hasattr(df.index, 'dayofweek'):
            features['day_of_week'] = df.index.dayofweek
        else:
            features['day_of_week'] = 2  # Default to Wednesday

        # LONG-TERM: Seasonality features (month and quarter)
        if hasattr(df.index, 'month'):
            features['month_of_year'] = df.index.month
            features['quarter'] = df.index.quarter
        else:
            features['month_of_year'] = 6  # Default to June
            features['quarter'] = 2

        # LONG-TERM: 52-week (252 trading days) high/low distance
        rolling_high_252 = df['High'].rolling(window=252, min_periods=20).max()
        rolling_low_252 = df['Low'].rolling(window=252, min_periods=20).min()
        features['distance_52w_high'] = (df['Close'] - rolling_high_252) / rolling_high_252 * 100
        features['distance_52w_low'] = (df['Close'] - rolling_low_252) / rolling_low_252 * 100

        # LONG-TERM: Trend persistence (how consistent is the trend over 30 days)
        # Count how many of the last 30 days were positive
        daily_returns = df['Close'].pct_change()
        features['trend_strength_30d'] = daily_returns.rolling(30).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5, raw=True
        )
        # Trend consistency (std of direction, lower = more consistent)
        features['trend_consistency'] = daily_returns.rolling(30).apply(
            lambda x: np.sign(x).std() if len(x) > 0 else 1, raw=True
        )

        # LONG-TERM: Mean reversion signals (distance from rolling mean)
        sma_20_mr = df['Close'].rolling(window=20).mean()
        sma_50_mr = df['Close'].rolling(window=50).mean()
        features['mean_reversion_20d'] = (df['Close'] - sma_20_mr) / sma_20_mr * 100
        features['mean_reversion_50d'] = (df['Close'] - sma_50_mr) / sma_50_mr * 100

        # Fill NaN values — ffill only (bfill leaks future data into past rows)
        features = features.ffill()
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        return features

    def create_targets(self, data, days=5, threshold=None):
        """
        Create target variables for training with noise filtering.

        Args:
            data: DataFrame with OHLCV data
            days: Number of days ahead for prediction (matches prediction horizon)
            threshold: Minimum % change to classify as UP/DOWN (filters noise)

        Returns:
            direction (1=UP, 0=DOWN, -1=NEUTRAL), magnitude (% change), valid_mask
        """
        if threshold is None:
            threshold = self.target_threshold

        # Future price change
        future_close = data['Close'].shift(-days)
        current_close = data['Close']

        pct_change = (future_close - current_close) / current_close * 100

        # Direction with threshold: Only classify significant moves
        # This filters out noise where price barely moved
        direction = pd.Series(index=data.index, dtype=int)
        direction[:] = -1  # Default to NEUTRAL

        direction[pct_change >= threshold] = 1   # UP: significant positive move
        direction[pct_change <= -threshold] = 0  # DOWN: significant negative move
        # Moves between -threshold and +threshold stay as -1 (NEUTRAL/noise)

        # Create mask for valid samples (exclude neutral/noise)
        valid_mask = direction != -1

        logger.info(f'Target distribution - UP: {(direction==1).sum()}, DOWN: {(direction==0).sum()}, '
                   f'Filtered (noise): {(direction==-1).sum()} ({threshold}% threshold)')

        return direction, pct_change, valid_mask

    def _tune_hyperparameters(self, X_train, y_train, tscv, n_trials=30):
        """
        Use Optuna to find optimal hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels
            tscv: TimeSeriesSplit cross-validator
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters dict
        """
        if not OPTUNA_AVAILABLE:
            return None

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'gamma': trial.suggest_float('gamma', 0.0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
            }

            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    verbosity=0
                )
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
                pred = model.predict(X_v)
                scores.append(accuracy_score(y_v, pred))

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(f'Best hyperparameters found: accuracy={study.best_value:.2%}')
        return study.best_params

    def train(self, symbol, data, indicators, days=7, save_model=True):
        """
        Train ML models for a specific stock.
        ENHANCED: Feature selection, hyperparameter tuning, noise filtering.

        Args:
            symbol: Stock ticker
            data: Historical OHLCV data (DataFrame)
            indicators: Calculated technical indicators
            days: Prediction horizon in days (now properly used for target creation)
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

        logger.info(f'Training ML model for {symbol} with {len(data)} samples, {days}-day horizon')

        # Engineer features
        features = self.engineer_features(data, indicators)

        # Create targets using the ACTUAL prediction horizon (not fixed 5 days)
        direction, magnitude, valid_mask = self.create_targets(data, days=days)

        # Align and clean data - filter out noise samples
        valid_idx = valid_mask & ~direction.isna() & ~magnitude.isna()
        X = features.loc[valid_idx]
        y_direction = direction.loc[valid_idx]
        y_magnitude = magnitude.loc[valid_idx]

        if len(X) < self.min_training_samples:
            logger.warning(f'Insufficient data for {symbol}: {len(X)} samples (need {self.min_training_samples})')
            return None

        # Use only available feature columns
        available_cols = [c for c in self.feature_columns if c in X.columns]
        X = X[available_cols]

        # Scale features BEFORE feature selection
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # FEATURE SELECTION: Use tree-based importance (captures nonlinear relationships)
        n_select = min(self.n_features, len(available_cols))
        if XGB_AVAILABLE:
            # Train quick model to get feature importances
            quick_model = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric='logloss', verbosity=0
            )
            quick_model.fit(X_scaled, y_direction)
            importances = quick_model.feature_importances_
            top_indices = np.argsort(importances)[-n_select:]
            self.selected_feature_names = [available_cols[i] for i in top_indices]
            self.feature_selector = None  # Not using SelectKBest
        else:
            # Fallback to mutual information (nonlinear, better than f_classif)
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_select)
            self.feature_selector.fit(X_scaled, y_direction)
            selected_mask = self.feature_selector.get_support()
            self.selected_feature_names = [col for col, selected in zip(available_cols, selected_mask) if selected]

        logger.info(f'Feature selection: {len(available_cols)} → {len(self.selected_feature_names)} features')
        logger.info(f'Top features: {self.selected_feature_names[:10]}')

        # Select columns
        X_selected = X_scaled[self.selected_feature_names]

        # Time series split for validation (increased to 10 folds)
        tscv = TimeSeriesSplit(n_splits=min(10, len(X_selected) // 50))

        # Calculate class weights for imbalanced data
        class_counts = y_direction.value_counts()
        total = len(y_direction)
        class_weight_dict = {0: total / (2 * class_counts.get(0, 1)),
                             1: total / (2 * class_counts.get(1, 1))}

        # HYPERPARAMETER TUNING with Optuna
        if self.use_tuning and len(X_selected) >= 300:
            logger.info('Running hyperparameter optimization...')
            self.best_params = self._tune_hyperparameters(X_selected, y_direction, tscv, n_trials=30)
        else:
            self.best_params = None

        # Build model with tuned or default hyperparameters
        if self.best_params:
            base_params = self.best_params.copy()
        else:
            # Optimized defaults (less aggressive than before)
            base_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }

        # Train Direction Model (Classification) with EARLY STOPPING
        if self.use_lightgbm:
            self.direction_model = lgb.LGBMClassifier(
                n_estimators=base_params.get('n_estimators', 200),
                max_depth=base_params.get('max_depth', 5),
                learning_rate=base_params.get('learning_rate', 0.05),
                num_leaves=min(31, 2**base_params.get('max_depth', 5) - 1),
                min_child_samples=base_params.get('min_child_weight', 5),
                reg_alpha=base_params.get('reg_alpha', 0.1),
                reg_lambda=base_params.get('reg_lambda', 0.1),
                class_weight=class_weight_dict,
                random_state=42,
                verbose=-1
            )
        else:
            self.direction_model = xgb.XGBClassifier(
                n_estimators=base_params.get('n_estimators', 200),
                max_depth=base_params.get('max_depth', 5),
                learning_rate=base_params.get('learning_rate', 0.05),
                min_child_weight=base_params.get('min_child_weight', 5),
                subsample=base_params.get('subsample', 0.8),
                colsample_bytree=base_params.get('colsample_bytree', 0.8),
                gamma=base_params.get('gamma', 0.1),
                reg_alpha=base_params.get('reg_alpha', 0.1),
                reg_lambda=base_params.get('reg_lambda', 0.1),
                scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=30,
                verbosity=0
            )

        # Train Magnitude Model (Regression)
        if self.use_lightgbm:
            self.magnitude_model = lgb.LGBMRegressor(
                n_estimators=base_params.get('n_estimators', 200),
                max_depth=base_params.get('max_depth', 5),
                learning_rate=base_params.get('learning_rate', 0.05),
                num_leaves=min(31, 2**base_params.get('max_depth', 5) - 1),
                min_child_samples=base_params.get('min_child_weight', 5),
                reg_alpha=base_params.get('reg_alpha', 0.1),
                reg_lambda=base_params.get('reg_lambda', 0.1),
                random_state=42,
                verbose=-1
            )
        else:
            self.magnitude_model = xgb.XGBRegressor(
                n_estimators=base_params.get('n_estimators', 200),
                max_depth=base_params.get('max_depth', 5),
                learning_rate=base_params.get('learning_rate', 0.05),
                min_child_weight=base_params.get('min_child_weight', 5),
                subsample=base_params.get('subsample', 0.8),
                colsample_bytree=base_params.get('colsample_bytree', 0.8),
                gamma=base_params.get('gamma', 0.1),
                reg_alpha=base_params.get('reg_alpha', 0.1),
                reg_lambda=base_params.get('reg_lambda', 0.1),
                random_state=42,
                early_stopping_rounds=30,
                verbosity=0
            )

        # Cross-validation with early stopping
        cv_direction_scores = []
        cv_magnitude_scores = []

        for train_idx, val_idx in tscv.split(X_selected):
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_dir_train, y_dir_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
            y_mag_train, y_mag_val = y_magnitude.iloc[train_idx], y_magnitude.iloc[val_idx]

            # Train direction model with early stopping
            if self.use_lightgbm:
                self.direction_model.fit(X_train, y_dir_train)
            else:
                self.direction_model.fit(
                    X_train, y_dir_train,
                    eval_set=[(X_val, y_dir_val)],
                    verbose=False
                )
            dir_pred = self.direction_model.predict(X_val)
            cv_direction_scores.append(accuracy_score(y_dir_val, dir_pred))

            # Train magnitude model with early stopping
            if self.use_lightgbm:
                self.magnitude_model.fit(X_train, y_mag_train)
            else:
                self.magnitude_model.fit(
                    X_train, y_mag_train,
                    eval_set=[(X_val, y_mag_val)],
                    verbose=False
                )
            mag_pred = self.magnitude_model.predict(X_val)
            cv_magnitude_scores.append(mean_absolute_error(y_mag_val, mag_pred))

        # Final training on all data (with a holdout for early stopping)
        split_idx = int(len(X_selected) * 0.9)
        X_train_final = X_selected.iloc[:split_idx]
        X_val_final = X_selected.iloc[split_idx:]
        y_dir_train_final = y_direction.iloc[:split_idx]
        y_dir_val_final = y_direction.iloc[split_idx:]
        y_mag_train_final = y_magnitude.iloc[:split_idx]
        y_mag_val_final = y_magnitude.iloc[split_idx:]

        if self.use_lightgbm:
            self.direction_model.fit(X_train_final, y_dir_train_final)
            self.magnitude_model.fit(X_train_final, y_mag_train_final)
        else:
            self.direction_model.fit(
                X_train_final, y_dir_train_final,
                eval_set=[(X_val_final, y_dir_val_final)],
                verbose=False
            )
            self.magnitude_model.fit(
                X_train_final, y_mag_train_final,
                eval_set=[(X_val_final, y_mag_val_final)],
                verbose=False
            )

        # Calibrate confidence on holdout set (Platt scaling)
        try:
            from sklearn.calibration import CalibratedClassifierCV
            val_probs = self.direction_model.predict_proba(X_val_final)
            val_preds = self.direction_model.predict(X_val_final)
            val_acc = accuracy_score(y_dir_val_final, val_preds)
            logger.info(f'Holdout accuracy before calibration: {val_acc:.2%}')

            # Wrap the trained model with isotonic calibration
            self.calibrator = CalibratedClassifierCV(
                self.direction_model, method='isotonic', cv='prefit'
            )
            self.calibrator.fit(X_val_final, y_dir_val_final)
            logger.info(f'Confidence calibration fitted on {len(X_val_final)} samples')
        except Exception as cal_err:
            logger.warning(f'Calibration failed: {cal_err}, using raw probabilities')
            self.calibrator = None

        metrics = {
            'symbol': symbol,
            'samples': len(X_selected),
            'samples_filtered': len(features) - len(X_selected),
            'features_original': len(available_cols),
            'features_selected': len(self.selected_feature_names),
            'selected_features': self.selected_feature_names,
            'direction_accuracy': np.mean(cv_direction_scores),
            'direction_std': np.std(cv_direction_scores),
            'magnitude_mae': np.mean(cv_magnitude_scores),
            'magnitude_std': np.std(cv_magnitude_scores),
            'hyperparameters': self.best_params or base_params,
            'target_threshold': self.target_threshold,
            'prediction_horizon_days': days,
            'trained_at': datetime.now().isoformat()
        }

        logger.info(f'Training complete for {symbol}: Direction accuracy={metrics["direction_accuracy"]:.2%}, '
                   f'MAE={metrics["magnitude_mae"]:.2f}%, Features={len(self.selected_feature_names)}')

        # Save models
        if save_model:
            self.save_model(symbol)

        return metrics

    def predict(self, data, indicators, days=5):
        """
        Make prediction using trained ML models.
        ENHANCED: Uses feature selection from training.

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

        # Apply feature selection if available
        if self.selected_feature_names is not None:
            # Use selected feature names directly (works for both tree-based and SelectKBest)
            available = [col for col in self.selected_feature_names if col in X_scaled.columns]
            X_selected = X_scaled[available]
        elif self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            X_selected = pd.DataFrame(X_selected, columns=self.selected_feature_names, index=X.index)
        else:
            X_selected = X_scaled

        # Predict direction (use calibrated probabilities if available)
        if hasattr(self, 'calibrator') and self.calibrator is not None:
            direction_prob = self.calibrator.predict_proba(X_selected)[0]
            direction_pred = self.calibrator.predict(X_selected)[0]
        else:
            direction_prob = self.direction_model.predict_proba(X_selected)[0]
            direction_pred = self.direction_model.predict(X_selected)[0]

        # Predict magnitude
        magnitude_pred = self.magnitude_model.predict(X_selected)[0]

        # Calculate confidence from calibrated probability
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
            'model_type': 'lightgbm' if self.use_lightgbm else 'xgboost',
            'features_used': len(self.selected_feature_names) if self.selected_feature_names else len(available_cols)
        }

    def save_model(self, symbol='generic'):
        """Save trained models to disk including feature selector."""
        if self.direction_model is None:
            return

        model_data = {
            'direction_model': self.direction_model,
            'magnitude_model': self.magnitude_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'use_lightgbm': self.use_lightgbm,
            # Feature selection components
            'feature_selector': self.feature_selector,
            'selected_feature_names': self.selected_feature_names,
            'n_features': self.n_features,
            'target_threshold': self.target_threshold,
            'best_params': self.best_params,
            # Confidence calibration
            'calibrator': getattr(self, 'calibrator', None),
            'saved_at': datetime.now().isoformat()
        }

        path = self._get_model_path(symbol, 'full')
        joblib.dump(model_data, path)
        logger.info(f'Model saved to {path} (features: {len(self.selected_feature_names) if self.selected_feature_names else "all"})')

    def load_model(self, symbol='generic'):
        """Load trained models from disk including feature selector."""
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
            # NEW: Load feature selection components
            self.feature_selector = model_data.get('feature_selector', None)
            self.selected_feature_names = model_data.get('selected_feature_names', None)
            self.n_features = model_data.get('n_features', 25)
            self.target_threshold = model_data.get('target_threshold', 1.0)
            self.best_params = model_data.get('best_params', None)
            self.calibrator = model_data.get('calibrator', None)

            features_info = f", features: {len(self.selected_feature_names)}" if self.selected_feature_names else ""
            logger.info(f'Model loaded from {path}{features_info}')
            return True
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            return False

    def get_feature_importance(self):
        """Get feature importance from trained models using selected features."""
        if self.direction_model is None:
            return None

        # Use selected feature names if available
        feature_names = self.selected_feature_names if self.selected_feature_names else self.feature_columns

        # Ensure we have the right number of features
        n_importances = len(self.direction_model.feature_importances_)
        feature_names = feature_names[:n_importances]

        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.direction_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.to_dict('records')


class EnsemblePredictor:
    """
    Ensemble model combining XGBoost and LightGBM predictions.
    ENHANCED: Now uses feature selection and hyperparameter tuning.
    """

    def __init__(self, model_dir='trained_models', n_features=30, target_threshold=0.3, use_tuning=True):
        self.xgb_model = StockMLModel(model_dir, use_lightgbm=False,
                                       n_features=n_features, target_threshold=target_threshold,
                                       use_tuning=use_tuning)
        self.lgb_model = StockMLModel(model_dir, use_lightgbm=True,
                                       n_features=n_features, target_threshold=target_threshold,
                                       use_tuning=False)  # Only tune XGBoost to save time
        self.model_dir = model_dir
        self.n_features = n_features
        self.target_threshold = target_threshold

    def train(self, symbol, data, indicators, days=7):
        """Train both models with improved settings."""
        xgb_metrics = self.xgb_model.train(symbol, data, indicators, days)
        lgb_metrics = self.lgb_model.train(symbol, data, indicators, days)

        # Calculate ensemble accuracy estimate
        ensemble_accuracy = None
        if xgb_metrics and lgb_metrics:
            ensemble_accuracy = (xgb_metrics.get('direction_accuracy', 0) +
                               lgb_metrics.get('direction_accuracy', 0)) / 2

        return {
            'xgboost': xgb_metrics,
            'lightgbm': lgb_metrics,
            'ensemble_accuracy_estimate': ensemble_accuracy
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

        # Disagreement detection: if models disagree on direction, slash confidence
        models_agree = xgb_pred['direction'] == lgb_pred['direction']

        # Average probabilities
        avg_up = (xgb_pred['probabilities']['up'] + lgb_pred['probabilities']['up']) / 2
        avg_down = (xgb_pred['probabilities']['down'] + lgb_pred['probabilities']['down']) / 2

        # Average magnitude
        avg_magnitude = (xgb_pred['predicted_change_percent'] + lgb_pred['predicted_change_percent']) / 2

        # Determine direction
        direction = 'UP' if avg_up > avg_down else 'DOWN'
        confidence = max(avg_up, avg_down)

        # Penalize confidence when models disagree (this is a weak signal)
        if not models_agree:
            confidence = min(confidence, 52.0)  # Cap at barely above random
            logger.info(f'Ensemble disagreement: XGB={xgb_pred["direction"]}, LGB={lgb_pred["direction"]} → confidence capped at {confidence}%')

        return {
            'direction': direction,
            'confidence': round(confidence, 1),
            'predicted_change_percent': round(avg_magnitude, 2),
            'probabilities': {
                'up': round(avg_up, 1),
                'down': round(avg_down, 1)
            },
            'model_type': 'ensemble',
            'models_agree': models_agree,
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
