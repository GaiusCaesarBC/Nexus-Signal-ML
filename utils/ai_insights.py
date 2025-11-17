# ml-service/utils/ai_insights.py - FIXED Gemini Model

import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def generate_insights(symbol, prediction_data, technical_data):
    """
    Generate AI insights using Google Gemini
    
    Args:
        symbol: Stock ticker symbol
        prediction_data: Dictionary with prediction results
        technical_data: Dictionary with technical indicators
    
    Returns:
        Dictionary with AI-generated insights
    """
    if not GOOGLE_API_KEY:
        logger.warning('Google API key not configured')
        return get_fallback_insights(prediction_data)
    
    try:
        # Use the correct model name for Gemini 1.5
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create detailed prompt
        prompt = f"""
Analyze this stock prediction for {symbol} and provide investment insights:

**Current Price:** ${prediction_data.get('current_price', 'N/A')}

**Prediction:**
- Direction: {prediction_data.get('prediction', {}).get('direction', 'NEUTRAL')}
- Confidence: {prediction_data.get('prediction', {}).get('confidence', 0):.1f}%
- Target Price (7 days): ${prediction_data.get('prediction', {}).get('target_price', 0):.2f}
- Expected Change: {prediction_data.get('prediction', {}).get('price_change_percent', 0):+.2f}%

**Technical Indicators:**
- RSI: {technical_data.get('rsi', 'N/A')}
- MACD Signal: {technical_data.get('macd_signal', 'N/A')}
- Volatility: {technical_data.get('volatility', 'N/A')}%
- Volume Status: {technical_data.get('volume_status', 'N/A')}

**Trading Signals:**
{chr(10).join('- ' + signal for signal in prediction_data.get('signals', []))}

Please provide:
1. A brief summary (2-3 sentences) explaining the overall outlook
2. 3-5 key insights about why this prediction makes sense
3. Overall sentiment (bullish/neutral/bearish)
4. Investment recommendation (buy/hold/sell)
5. Risk level (low/medium/high)
6. Suggested time horizon (short-term/medium-term/long-term)
7. Brief reasoning for your recommendation

Format your response as JSON with these exact keys:
{{
    "summary": "Brief 2-3 sentence overview",
    "key_insights": ["insight 1", "insight 2", "insight 3"],
    "sentiment": "bullish|neutral|bearish",
    "recommendation": "buy|hold|sell",
    "risk_level": "low|medium|high",
    "time_horizon": "short-term|medium-term|long-term",
    "reasoning": "Why this recommendation makes sense"
}}

Return ONLY the JSON, no additional text.
"""
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Parse response
        import json
        insights_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if insights_text.startswith('```json'):
            insights_text = insights_text.replace('```json', '').replace('```', '').strip()
        elif insights_text.startswith('```'):
            insights_text = insights_text.replace('```', '').strip()
        
        insights = json.loads(insights_text)
        
        logger.info(f'Successfully generated AI insights for {symbol}')
        return insights
        
    except Exception as e:
        logger.error(f'Error generating AI insights: {e}')
        return get_fallback_insights(prediction_data)

def get_fallback_insights(prediction_data):
    """
    Generate fallback insights when AI is unavailable
    """
    direction = prediction_data.get('prediction', {}).get('direction', 'NEUTRAL')
    confidence = prediction_data.get('prediction', {}).get('confidence', 50)
    change_pct = prediction_data.get('prediction', {}).get('price_change_percent', 0)
    
    # Determine sentiment
    if direction == 'UP' and confidence >= 60:
        sentiment = 'bullish'
        recommendation = 'buy' if confidence >= 70 else 'hold'
    elif direction == 'DOWN' and confidence >= 60:
        sentiment = 'bearish'
        recommendation = 'sell' if confidence >= 70 else 'hold'
    else:
        sentiment = 'neutral'
        recommendation = 'hold'
    
    # Determine risk level
    volatility = prediction_data.get('technical_analysis', {}).get('volatility', 20)
    if volatility > 30:
        risk_level = 'high'
    elif volatility > 20:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Determine time horizon
    if abs(change_pct) > 5:
        time_horizon = 'short-term'
    elif abs(change_pct) > 2:
        time_horizon = 'medium-term'
    else:
        time_horizon = 'long-term'
    
    # Build summary
    if direction == 'UP':
        summary = f"The technical indicators suggest bullish momentum with a {confidence:.1f}% confidence level. "
        summary += f"Expected price increase of {change_pct:+.2f}% over the next 7 days. "
    elif direction == 'DOWN':
        summary = f"The technical analysis indicates bearish pressure with {confidence:.1f}% confidence. "
        summary += f"Expected price decrease of {change_pct:.2f}% in the coming week. "
    else:
        summary = f"The stock shows neutral signals with mixed technical indicators. "
        summary += f"Minimal price movement expected ({change_pct:+.2f}%) over the next 7 days. "
    
    summary += f"Current market volatility is {volatility:.1f}%."
    
    # Build key insights
    key_insights = []
    signals = prediction_data.get('signals', [])
    
    if signals:
        key_insights.append(f"Technical signals indicate: {', '.join(signals[:2])}")
    
    tech = prediction_data.get('technical_analysis', {})
    rsi = tech.get('rsi', 50)
    if rsi > 70:
        key_insights.append("RSI indicates overbought conditions - potential pullback risk")
    elif rsi < 30:
        key_insights.append("RSI shows oversold levels - possible bounce opportunity")
    else:
        key_insights.append(f"RSI at {rsi:.1f} suggests neutral momentum")
    
    macd = tech.get('macd_signal', '')
    if 'bullish' in macd.lower():
        key_insights.append("MACD shows bullish momentum building")
    elif 'bearish' in macd.lower():
        key_insights.append("MACD indicates bearish pressure increasing")
    
    volume = tech.get('volume_status', '')
    if volume == 'High':
        key_insights.append("Higher than average volume confirms the current trend")
    elif volume == 'Low':
        key_insights.append("Low volume suggests caution as trend may lack conviction")
    
    key_insights.append(f"Recommended for {time_horizon} traders with {risk_level} risk tolerance")
    
    # Build reasoning
    if recommendation == 'buy':
        reasoning = f"Strong {sentiment} signals with {confidence:.0f}% confidence justify a buy position. The technical setup favors upward movement."
    elif recommendation == 'sell':
        reasoning = f"Bearish indicators with {confidence:.0f}% confidence suggest reducing exposure. Risk/reward ratio favors selling."
    else:
        reasoning = f"Mixed signals and {confidence:.0f}% confidence suggest waiting for clearer direction before making moves."
    
    return {
        'summary': summary,
        'key_insights': key_insights[:5],  # Max 5 insights
        'sentiment': sentiment,
        'recommendation': recommendation,
        'risk_level': risk_level,
        'time_horizon': time_horizon,
        'reasoning': reasoning
    }

def batch_generate_insights(predictions_list):
    """
    Generate insights for multiple predictions
    
    Args:
        predictions_list: List of (symbol, prediction_data, technical_data) tuples
    
    Returns:
        Dictionary mapping symbols to insights
    """
    results = {}
    
    for symbol, prediction_data, technical_data in predictions_list:
        try:
            insights = generate_insights(symbol, prediction_data, technical_data)
            results[symbol] = insights
        except Exception as e:
            logger.error(f'Error generating insights for {symbol}: {e}')
            results[symbol] = get_fallback_insights(prediction_data)
    
    return results