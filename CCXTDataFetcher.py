import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import time
import warnings
warnings.filterwarnings('ignore')


class CCXTDataFetcher:
    """
    Class to fetch price data from centralized exchanges
    Supports Binance spot and Bitget futures
    """
    
    def __init__(self, data_dir='./data/centralized_prices'):
        self.data_dir = data_dir
        self.spot_data_dir = os.path.join(data_dir, 'spot')
        self.futures_data_dir = os.path.join(data_dir, 'futures')
        
        # Create directories
        for directory in [self.spot_data_dir, self.futures_data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # Initialize exchanges
        self.binance = ccxt.binance()
        self.bitget = ccxt.bitget()
        
        # Token mappings for different exchanges
        self.token_mappings = {
            'binance': {
                'ETH': 'ETH/USDT',
                'WETH': 'ETH/USDT',  # WETH same as ETH
                'USDC': 'USDC/USDT',
                'INJ': 'INJ/USDT',
                'WTAO': 'TAO/USDT',
                'MORPHO': 'MORPHO/USDT',
                'LINK': 'LINK/USDT'
            },
            'bitget': {
                'ETH': 'ETH/USDT:USDT',  # Perpetual futures
                'WETH': 'ETH/USDT:USDT',
                'INJ': 'INJ/USDT:USDT',
                'WTAO': 'TAO/USDT:USDT',
                'MORPHO': 'MORPHO/USDT:USDT',
                'LINK': 'LINK/USDT:USDT'
            }
        }
    
    def fetch_spot_data(self, symbol, timeframe='5m', start_date=None, end_date=None):
        """
        Fetch spot data from Binance
        """
        try:
            # Get symbol mapping
            binance_symbol = self.token_mappings['binance'].get(symbol, f"{symbol}/USDT")
            
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime(2024, 6, 1)
            if end_date is None:
                end_date = datetime.now()
            
            print(f"Fetching {timeframe} spot data for {symbol} from Binance...")
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while current_since < end_timestamp:
                try:
                    # Fetch data in chunks
                    ohlcv = self.binance.fetch_ohlcv(
                        binance_symbol, 
                        timeframe, 
                        since=current_since, 
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    
                    # Update timestamp for next fetch
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error fetching chunk: {e}")
                    break
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Remove duplicates and sort
                df = df.drop_duplicates().sort_index()
                
                # Save to file
                filename = f"{symbol.lower()}_binance_{timeframe.replace('m', 'm')}.csv"
                filepath = os.path.join(self.spot_data_dir, filename)
                df.to_csv(filepath)
                
                print(f"✓ Downloaded {len(df)} records for {symbol}")
                print(f"  Saved to: {filepath}")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                
                return df
            else:
                print(f"✗ No data found for {symbol}")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching spot data for {symbol}: {e}")
            return None
    
    def fetch_futures_data(self, symbol, timeframe='5m', start_date=None, end_date=None):
        """
        Fetch futures data from Bitget
        """
        try:
            # Get symbol mapping
            bitget_symbol = self.token_mappings['bitget'].get(symbol, f"{symbol}/USDT:USDT")
            
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime(2024, 6, 1)
            if end_date is None:
                end_date = datetime.now()
            
            print(f"Fetching {timeframe} futures data for {symbol} from Bitget...")
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while current_since < end_timestamp:
                try:
                    # Fetch data in chunks
                    ohlcv = self.bitget.fetch_ohlcv(
                        bitget_symbol, 
                        timeframe, 
                        since=current_since, 
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    
                    # Update timestamp for next fetch
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error fetching chunk: {e}")
                    break
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Remove duplicates and sort
                df = df.drop_duplicates().sort_index()
                
                # Save to file
                filename = f"{symbol.lower()}_bitget_futures_{timeframe.replace('m', 'm')}.csv"
                filepath = os.path.join(self.futures_data_dir, filename)
                df.to_csv(filepath)
                
                print(f"✓ Downloaded {len(df)} records for {symbol}")
                print(f"  Saved to: {filepath}")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                
                return df
            else:
                print(f"✗ No data found for {symbol}")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching futures data for {symbol}: {e}")
            return None
    
    def fetch_all_tokens_data(self, tokens=None, timeframe='5m', start_date=None, end_date=None):
        """
        Fetch data for all tokens from both spot and futures
        """
        if tokens is None:
            tokens = list(self.token_mappings['binance'].keys())
        
        print(f"Fetching data for {len(tokens)} tokens...")
        print("=" * 60)
        
        # Fetch spot data
        print("SPOT DATA (Binance):")
        print("-" * 30)
        spot_results = {}
        for token in tokens:
            if token != 'USDC':  # Skip USDC as it's stable
                result = self.fetch_spot_data(token, timeframe, start_date, end_date)
                if result is not None:
                    spot_results[token] = result
        
        print("\n" + "=" * 60)
        
        # Fetch futures data
        print("FUTURES DATA (Bitget):")
        print("-" * 30)
        futures_results = {}
        for token in tokens:
            if token != 'USDC':  # Skip USDC as it's stable
                result = self.fetch_futures_data(token, timeframe, start_date, end_date)
                if result is not None:
                    futures_results[token] = result
        
        print("\n" + "=" * 60)
        print("Download complete!")
        
        return {
            'spot': spot_results,
            'futures': futures_results
        }
    
    def get_available_symbols(self):
        """
        Get available symbols from both exchanges
        """
        print("Available symbols:")
        print("=" * 40)
        
        print("Binance Spot:")
        for token, symbol in self.token_mappings['binance'].items():
            print(f"  {token} -> {symbol}")
        
        print("\nBitget Futures:")
        for token, symbol in self.token_mappings['bitget'].items():
            print(f"  {token} -> {symbol}")
    
    def check_data_status(self):
        """
        Check what data files exist and their status
        """
        print("Data Status Check:")
        print("=" * 40)
        
        # Check spot data
        print("Spot Data (Binance):")
        spot_files = os.listdir(self.spot_data_dir) if os.path.exists(self.spot_data_dir) else []
        for file in sorted(spot_files):
            if file.endswith('.csv'):
                filepath = os.path.join(self.spot_data_dir, file)
                df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
                print(f"  {file}: {len(df)} records, {df.index.min()} to {df.index.max()}")
        
        print("\nFutures Data (Bitget):")
        futures_files = os.listdir(self.futures_data_dir) if os.path.exists(self.futures_data_dir) else []
        for file in sorted(futures_files):
            if file.endswith('.csv'):
                filepath = os.path.join(self.futures_data_dir, file)
                df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
                print(f"  {file}: {len(df)} records, {df.index.min()} to {df.index.max()}")


# Example usage and testing
if __name__ == "__main__":
    fetcher = CCXTDataFetcher()
    
    # Show available symbols
    fetcher.get_available_symbols()
    
    # Check existing data
    fetcher.check_data_status()