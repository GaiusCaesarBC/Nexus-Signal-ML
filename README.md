# 🤖 Nexus Signal ML Prediction Service

AI-powered stock prediction engine using machine learning and technical analysis.

## 🚀 Features

- **Real-time Stock Predictions** - Predict price movements with confidence scores
- **Technical Analysis** - RSI, MACD, Bollinger Bands, and more
- **AI Insights** - Powered by Google Gemini for intelligent market commentary
- **Batch Processing** - Analyze entire portfolios at once
- **Historical Tracking** - Track prediction accuracy over time

## 📋 Prerequisites

- Python 3.8+ (you have 3.13.9 ✓)
- pip (package installer)
- Google Gemini API key (already configured ✓)

## ⚡ Quick Start

### 1. Setup (First Time Only)

```bash
cd C:\Users\2cody\source\repos
git clone <your-ml-service-repo>  # Or create new folder
cd Nexus-Signal-ML

# Run setup script (Windows)
setup.bat
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies (TensorFlow, Flask, etc.)
- Create necessary directories

### 2. Start the Service

```bash
# Activate virtual environment
venv\Scripts\activate.bat

# Start the Flask server
python app.py
```

The service will start on **http://localhost:5001**

### 3. Test the Service

Open your browser and go to:
```
http://localhost:5001/health
```

You should see:
```json
{
  "status": "healthy",
  "service": "ML Prediction Service",
  "version": "1.0.0"
}
```

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Single Stock Prediction
```
POST /predict
Content-Type: application/json

{
  "symbol": "AAPL",
  "days": 7
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "current_price": 178.32,
  "prediction": {
    "direction": "UP",
    "confidence": 72.5,
    "target_price": 182.15,
    "price_change_percent": 2.15,
    "timeframe_days": 7
  },
  "signals": [
    "RSI oversold (bullish)",
    "MACD bullish crossover"
  ],
  "technical_analysis": {
    "rsi": 42.3,
    "macd_signal": "Bullish",
    "volatility": 28.5,
    "volume_status": "High"
  }
}
```

### Batch Predictions
```
POST /predict/batch
Content-Type: application/json

{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "days": 7
}
```

### Deep Analysis (with AI Insights)
```
POST /analyze
Content-Type: application/json

{
  "symbol": "AAPL"
}
```

**Response includes:**
- Prediction
- AI-generated insights from Google Gemini
- Detailed technical analysis
- Trading recommendations

## 🔧 Configuration

Edit `.env` file:

```env
PORT=5001
GOOGLE_API_KEY=your_api_key_here
MODEL_PATH=./models/saved_models
LOG_LEVEL=INFO
```

## 🧠 How It Works

### 1. Data Fetching
- Uses `yfinance` to get real-time stock data
- Fetches 6 months of historical data by default

### 2. Technical Analysis
- Calculates 10+ technical indicators
- RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic

### 3. Prediction Engine
- Rule-based predictions using technical indicators
- Confidence scoring system
- Can be upgraded to LSTM/TensorFlow models

### 4. AI Insights
- Google Gemini generates human-readable insights
- Explains "why" behind predictions
- Provides actionable recommendations

## 📊 Prediction Algorithm

The prediction engine uses a weighted scoring system:

```
Score = RSI_signal + MACD_signal + BB_signal + Momentum_signal + Volume_signal

If score >= 2:  Direction = UP,   Confidence = 60-85%
If score <= -2: Direction = DOWN, Confidence = 60-85%
Else:           Direction = NEUTRAL, Confidence = ~50%
```

### Signals Explained:

- **RSI < 30**: Oversold (Bullish) +2 points
- **RSI > 70**: Overbought (Bearish) -2 points
- **MACD > Signal**: Bullish +1 point
- **Price near lower BB**: Bullish +1 point
- **High volume**: Momentum +0.5 points

## 🔌 Integration with Node.js Backend

Add to your `Nexus-Signal-Server`:

```javascript
// server/routes/predictionsRoutes.js
const express = require('express');
const router = express.Router();
const axios = require('axios');
const auth = require('../middleware/authMiddleware');

const ML_SERVICE_URL = 'http://localhost:5001';

router.post('/predict', auth, async (req, res) => {
    try {
        const { symbol } = req.body;
        const response = await axios.post(`${ML_SERVICE_URL}/predict`, { symbol });
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Prediction service unavailable' });
    }
});

module.exports = router;
```

## 🚀 Next Steps

1. **Upgrade to Deep Learning**
   - Implement LSTM neural networks
   - Train on larger datasets
   - Add sentiment analysis from news

2. **Add More Features**
   - Options pricing models
   - Portfolio optimization
   - Risk analysis

3. **Improve Accuracy**
   - Collect prediction results
   - Calculate actual accuracy
   - Retrain models based on performance

## 📝 Development

### Project Structure
```
ml-service/
├── app.py                      # Flask API server
├── requirements.txt            # Python dependencies
├── .env                        # Configuration
├── models/
│   ├── predictor.py           # ML prediction engine
│   └── saved_models/          # Trained models
├── utils/
│   ├── technical_indicators.py # Technical analysis
│   ├── market_data.py         # Data fetching
│   └── ai_insights.py         # Gemini AI integration
└── logs/                      # Service logs
```

### Adding New Indicators

Edit `utils/technical_indicators.py`:

```python
def calculate_your_indicator(self, data):
    # Your calculation here
    return indicator_values
```

### Training Custom Models

```python
# models/trainer.py (create this file)
from predictor import StockPredictor

predictor = StockPredictor()
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
predictor.train_model(symbols)
```

## 🐛 Troubleshooting

### "Module not found" Error
```bash
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### "Port already in use"
Change port in `.env`:
```env
PORT=5002
```

### Google API Key Issues
1. Get key from: https://aistudio.google.com/apikey
2. Update `.env` file
3. Restart service

## 📚 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Google Gemini API](https://ai.google.dev/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

## 🤝 Support

Need help? Check the logs:
```bash
tail -f logs/ml_service.log
```

## 📄 License

MIT License - Built for Nexus Signal

---

**Ready to predict the future? Let's go! 🚀**