# ml-service/utils/market_data.py - Alpha Vantage + CoinGecko

import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# API keys must be set via environment variables - no defaults for security
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')

if not ALPHA_VANTAGE_API_KEY:
    logger.warning('ALPHA_VANTAGE_API_KEY not set - stock data fetching will fail')
if not COINGECKO_API_KEY:
    logger.warning('COINGECKO_API_KEY not set - crypto data fetching will fail')

# CoinGecko symbol mapping
CRYPTO_MAPPING = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'SOL': 'solana',
    'DOT': 'polkadot',
    'DOGE': 'dogecoin',
    'MATIC': 'matic-network',
    'LTC': 'litecoin',
    'AVAX': 'avalanche-2',
    'SHIB': 'shiba-inu',
    'TRX': 'tron',
    'UNI': 'uniswap',
    'LINK': 'chainlink',
    'ATOM': 'cosmos',
    'ETC': 'ethereum-classic',
    'XLM': 'stellar',
    'BCH': 'bitcoin-cash',
    'ALGO': 'algorand',
    'PEPE': 'pepe',
    'ARB': 'arbitrum',
    'OP': 'optimism',
    'SUI': 'sui',
    'APT': 'aptos',
    'INJ': 'injective-protocol',
    'SEI': 'sei-network',
    'WIF': 'dogwifcoin',
    'BONK': 'bonk',
    'TRUMP': 'official-trump',  # Trump coin
    'ZEC': 'zcash',  # Zcash
    'XMR': 'monero',
    'HBAR': 'hedera-hashgraph',
    'VET': 'vechain',
    'FIL': 'filecoin',
    'NEAR': 'near',
    'AAVE': 'aave',
    'MKR': 'maker',
    'GRT': 'the-graph',
    'SAND': 'the-sandbox',
    'MANA': 'decentraland',
    'AXS': 'axie-infinity',
    'CRV': 'curve-dao-token',
    'COMP': 'compound-governance-token',
    'SUSHI': 'sushi',
    'YFI': 'yearn-finance',
    '1INCH': '1inch',
    'ENJ': 'enjincoin',
    'BAT': 'basic-attention-token',
    'ZRX': '0x',
    'KNC': 'kyber-network-crystal',
    'SNX': 'havven',
    'RUNE': 'thorchain',
    'LUNA': 'terra-luna-2',
    'FTM': 'fantom',
    'ONE': 'harmony',
    'ZIL': 'zilliqa',
    'DASH': 'dash',
    'WAVES': 'waves',
    'XTZ': 'tezos',
    'EOS': 'eos',
    'IOTA': 'iota',
    'NEO': 'neo',
    'QTUM': 'qtum',
    'ICX': 'icon',
    'ONT': 'ontology',
    'ZEN': 'horizen',
    'DCR': 'decred',
    'DGB': 'digibyte',
    'RVN': 'ravencoin',
    'SC': 'siacoin',
    'LSK': 'lisk',
    'STEEM': 'steem',
    'XEM': 'nem',
    'BTS': 'bitshares'
}

def is_crypto(symbol):
    """Check if symbol is a cryptocurrency"""
    clean_symbol = symbol.upper().replace('USD', '').replace('USDT', '').replace('-USD', '')
    return clean_symbol in CRYPTO_MAPPING

def fetch_crypto_data(symbol, period='6mo'):
    """
    Fetch cryptocurrency data using CoinGecko Pro API
    
    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        period: Time period
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        # Clean symbol and get CoinGecko ID
        clean_symbol = symbol.upper().replace('USD', '').replace('USDT', '').replace('-USD', '')
        
        if clean_symbol not in CRYPTO_MAPPING:
            logger.error(f'Crypto {clean_symbol} not supported. Available: {list(CRYPTO_MAPPING.keys())}')
            return None
        
        coin_id = CRYPTO_MAPPING[clean_symbol]
        logger.info(f'Fetching crypto data for {clean_symbol} ({coin_id}) from CoinGecko')
        
        # Calculate days for period
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, 'max': 'max'
        }
        days = period_map.get(period, 180)
        
        # CoinGecko Pro API endpoint
        url = f'https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
        
        headers = {
            'x-cg-pro-api-key': COINGECKO_API_KEY
        }
        
        params = {
            'vs_currency': 'usd',
            'days': days if days != 'max' else 'max',
            'interval': 'daily'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'prices' not in data:
            logger.error(f'No price data in CoinGecko response for {clean_symbol}')
            return None
        
        # Parse CoinGecko data (prices, volumes)
        prices = data['prices']
        volumes = data.get('total_volumes', [])
        
        if not prices:
            logger.error(f'Empty price data for {clean_symbol}')
            return None
        
        # Build DataFrame with OHLC estimates
        # CoinGecko gives us daily close prices, we estimate OHLC
        rows = []
        for i, (timestamp, close) in enumerate(prices):
            date = pd.to_datetime(timestamp, unit='ms')
            volume = volumes[i][1] if i < len(volumes) else 0
            
            # Estimate OHLC from close price (±0.5% for daily range)
            high = close * 1.005
            low = close * 0.995
            open_price = close * (1 + (0.002 if i == 0 else (prices[i][1] - prices[i-1][1]) / prices[i-1][1] * 0.3))
            
            rows.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove any duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')
        
        if df.empty:
            logger.warning(f'No crypto data found for {clean_symbol}')
            return None
        
        logger.info(f'Successfully fetched {len(df)} rows for crypto {clean_symbol} from CoinGecko')
        return df
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error(f'CoinGecko rate limit exceeded for {symbol}')
        else:
            logger.error(f'CoinGecko HTTP error for {symbol}: {e}')
        return None
    except Exception as e:
        logger.error(f'Error fetching crypto data for {symbol}: {str(e)}')
        return None

def fetch_stock_data(symbol, period='6mo', interval='1d'):
    """
    Fetch historical stock/crypto data
    Routes to appropriate API based on symbol type
    
    Args:
        symbol: Stock ticker or crypto symbol
        period: Time period
        interval: Data interval
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        # Route to crypto API if it's a crypto symbol
        if is_crypto(symbol):
            return fetch_crypto_data(symbol, period)
        
        # Otherwise use Alpha Vantage for stocks
        logger.info(f'Fetching stock data for {symbol} using Alpha Vantage')
        
        outputsize = 'full' if period in ['1y', '2y', '5y', 'max'] else 'compact'
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': outputsize,
            'datatype': 'json'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
            return None
        
        if 'Time Series (Daily)' not in data:
            logger.error(f"No time series data for {symbol}")
            return None
        
        # Parse the time series data
        time_series = data['Time Series (Daily)']
        
        rows = []
        for date_str, values in time_series.items():
            rows.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter by period
        if period != 'max':
            period_map = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
            }
            days = period_map.get(period, 180)
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['Date'] >= cutoff_date]
        
        if df.empty:
            logger.warning(f'No data found for {symbol}')
            return None
        
        logger.info(f'Successfully fetched {len(df)} rows for {symbol} from Alpha Vantage')
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f'Network error fetching {symbol}: {str(e)}')
        return None
    except Exception as e:
        logger.error(f'Error fetching data for {symbol}: {str(e)}')
        return None

def fetch_current_price(symbol):
    """
    Fetch current price for a stock or crypto
    
    Args:
        symbol: Stock ticker or crypto symbol
    
    Returns:
        Current price as float
    """
    try:
        # Use CoinGecko for crypto
        if is_crypto(symbol):
            clean_symbol = symbol.upper().replace('USD', '').replace('USDT', '').replace('-USD', '')
            coin_id = CRYPTO_MAPPING.get(clean_symbol)
            
            if not coin_id:
                return None
            
            url = f'https://pro-api.coingecko.com/api/v3/simple/price'
            headers = {'x-cg-pro-api-key': COINGECKO_API_KEY}
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id in data and 'usd' in data[coin_id]:
                return float(data[coin_id]['usd'])
            
            return None
        
        # Use Alpha Vantage for stocks
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            return float(data['Global Quote']['05. price'])
        
        return None
        
    except Exception as e:
        logger.error(f'Error fetching current price for {symbol}: {str(e)}')
        return None

def fetch_stock_info(symbol):
    """
    Fetch detailed stock/crypto information
    
    Args:
        symbol: Stock ticker or crypto symbol
    
    Returns:
        Dictionary of info
    """
    try:
        # CoinGecko for crypto
        if is_crypto(symbol):
            clean_symbol = symbol.upper().replace('USD', '').replace('USDT', '').replace('-USD', '')
            coin_id = CRYPTO_MAPPING.get(clean_symbol)
            
            if not coin_id:
                return None
            
            url = f'https://pro-api.coingecko.com/api/v3/coins/{coin_id}'
            headers = {'x-cg-pro-api-key': COINGECKO_API_KEY}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': clean_symbol,
                'name': data.get('name', clean_symbol),
                'sector': 'Cryptocurrency',
                'industry': data.get('categories', ['Crypto'])[0] if data.get('categories') else 'Crypto',
                'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                'pe_ratio': 0,
                'dividend_yield': 0,
                'fifty_two_week_high': data.get('market_data', {}).get('high_24h', {}).get('usd', 0),
                'fifty_two_week_low': data.get('market_data', {}).get('low_24h', {}).get('usd', 0),
            }
        
        # Alpha Vantage for stocks
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        info = response.json()
        
        if not info or 'Symbol' not in info:
            return None
        
        return {
            'symbol': symbol,
            'name': info.get('Name', symbol),
            'sector': info.get('Sector', 'Unknown'),
            'industry': info.get('Industry', 'Unknown'),
            'market_cap': int(info.get('MarketCapitalization', 0)),
            'pe_ratio': float(info.get('PERatio', 0)) if info.get('PERatio') != 'None' else 0,
            'dividend_yield': float(info.get('DividendYield', 0)) if info.get('DividendYield') else 0,
            'fifty_two_week_high': float(info.get('52WeekHigh', 0)),
            'fifty_two_week_low': float(info.get('52WeekLow', 0)),
        }
        
    except Exception as e:
        logger.error(f'Error fetching info for {symbol}: {str(e)}')
        return None

def fetch_multiple_stocks(symbols, period='6mo'):
    """
    Fetch data for multiple stocks/cryptos
    
    Args:
        symbols: List of stock/crypto symbols
        period: Time period
    
    Returns:
        Dictionary of {symbol: DataFrame}
    """
    try:
        result = {}
        
        for symbol in symbols:
            data = fetch_stock_data(symbol, period)
            if data is not None:
                result[symbol] = data

            # Rate limiting
            time.sleep(0.5)
        
        return result
        
    except Exception as e:
        logger.error(f'Error fetching multiple stocks: {str(e)}')
        return {}