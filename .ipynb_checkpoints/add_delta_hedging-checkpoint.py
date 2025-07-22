import pandas as pd
import ccxt
import time
from datetime import datetime
import os

# Configuration
SYMBOLS = ['UNIUSDT', 'ETHUSDT']  # Bitget perpetual futures
TIMEFRAME = '1h'  # 1-hour candles
OUTPUT_DIR = 'bitget_futures_data'
LIMIT = 1000  # Max candles per request
FEE_RATE = 0.0002  # Bitget maker fee: 0.02%

# Initialize Bitget exchange
exchange = ccxt.bitget({'enableRateLimit': True})

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def milliseconds_to_datetime(ms):
    """Convert milliseconds to datetime."""
    return datetime.utcfromtimestamp(ms / 1000)

def fetch_ohlcv_data(symbol, timeframe, since, limit):
    """Fetch OHLCV data for a symbol."""
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched {len(ohlcv)} OHLCV candles for {symbol} from {milliseconds_to_datetime(since)}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            time.sleep(5)
            continue
    return all_ohlcv

def save_ohlcv_to_csv(symbol, ohlcv_data):
    """Save OHLCV data to CSV."""
    if not ohlcv_data:
        print(f"No OHLCV data to save for {symbol}")
        return
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    symbol_clean = symbol.replace('/', '_').replace(':', '_')
    df.to_csv(f"{OUTPUT_DIR}/{symbol_clean}_ohlcv.csv", index=False)
    print(f"Saved OHLCV data for {symbol} to {OUTPUT_DIR}/{symbol_clean}_ohlcv.csv")

# Load strategy results
try:
    results_df = pd.read_csv('strategy_results.csv', parse_dates=['time_pd'])
    results_df['time_pd'] = pd.to_datetime(results_df['time_pd'], utc=True)
    results_df.set_index('time_pd', inplace=True)
except FileNotFoundError:
    print("Error: strategy_results.csv not found")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Verify required columns
required_columns = ['reset_point', 'token_0_total', 'token_1_total', 'price', 'price_0_usd', 'value_position_usd', 'value_hold_usd']
if not all(col in results_df.columns for col in required_columns):
    print(f"Error: Missing required columns. Found: {results_df.columns.tolist()}")
    exit(1)

# Debug: Print data summary
print("Strategy Results Shape:", results_df.shape)
print("Head:\n", results_df[required_columns].head())

# Fetch Bitget futures data
start_time = results_df.index.min()
end_time = results_df.index.max()
since = int(start_time.timestamp() * 1000)

for symbol in SYMBOLS:
    ohlcv_data = fetch_ohlcv_data(symbol, TIMEFRAME, since, LIMIT)
    save_ohlcv_to_csv(symbol, ohlcv_data)

# Load futures data
uni_futures = pd.read_csv(f"{OUTPUT_DIR}/UNI_USDT_USDT_ohlcv.csv", parse_dates=['datetime'])
eth_futures = pd.read_csv(f"{OUTPUT_DIR}/ETH_USDT_USDT_ohlcv.csv", parse_dates=['datetime'])
uni_futures.set_index('datetime', inplace=True)
eth_futures.set_index('datetime', inplace=True)

# Merge futures prices with results_df
results_df = results_df.join(
    uni_futures[['close']].rename(columns={'close': 'uni_futures_price'}),
    how='left'
).join(
    eth_futures[['close']].rename(columns={'close': 'weth_futures_price'}),
    how='left'
).ffill()

# Initialize hedging columns
results_df['uni_hedge_qty'] = 0.0
results_df['weth_hedge_qty'] = 0.0
results_df['uni_hedge_entry_price'] = 0.0
results_df['weth_hedge_entry_price'] = 0.0
results_df['hedge_realized_pnl_usd'] = 0.0
results_df['hedge_unrealized_pnl_usd'] = 0.0
results_df['value_position_hedged_usd'] = results_df['value_position_usd']

# Initialize first hedge at t=0
if not results_df.empty:
    results_df.iloc[0, results_df.columns.get_loc('uni_hedge_qty')] = -results_df.iloc[0]['token_0_total']
    results_df.iloc[0, results_df.columns.get_loc('weth_hedge_qty')] = -results_df.iloc[0]['token_1_total']
    results_df.iloc[0, results_df.columns.get_loc('uni_hedge_entry_price')] = results_df.iloc[0]['uni_futures_price']
    results_df.iloc[0, results_df.columns.get_loc('weth_hedge_entry_price')] = results_df.iloc[0]['weth_futures_price']

# Process hedging
for i in range(1, len(results_df)):
    # Carry forward hedge positions
    results_df.iloc[i, results_df.columns.get_loc('uni_hedge_qty')] = results_df.iloc[i-1]['uni_hedge_qty']
    results_df.iloc[i, results_df.columns.get_loc('weth_hedge_qty')] = results_df.iloc[i-1]['weth_hedge_qty']
    results_df.iloc[i, results_df.columns.get_loc('uni_hedge_entry_price')] = results_df.iloc[i-1]['uni_hedge_entry_price']
    results_df.iloc[i, results_df.columns.get_loc('weth_hedge_entry_price')] = results_df.iloc[i-1]['weth_hedge_entry_price']
    
    # Calculate unrealized PnL
    uni_unrealized = results_df.iloc[i]['uni_hedge_qty'] * (results_df.iloc[i]['uni_futures_price'] - results_df.iloc[i]['uni_hedge_entry_price'])
    weth_unrealized = results_df.iloc[i]['weth_hedge_qty'] * (results_df.iloc[i]['weth_futures_price'] - results_df.iloc[i]['weth_hedge_entry_price'])
    results_df.iloc[i, results_df.columns.get_loc('hedge_unrealized_pnl_usd')] = uni_unrealized + weth_unrealized
    
    # Check for rebalance
    if results_df.iloc[i]['reset_point']:
        # Calculate realized PnL from closing previous hedge
        uni_qty = -results_df.iloc[i-1]['uni_hedge_qty']
        weth_qty = -results_df.iloc[i-1]['weth_hedge_qty']
        uni_pnl = uni_qty * (results_df.iloc[i-1]['uni_hedge_entry_price'] - results_df.iloc[i]['uni_futures_price'])
        weth_pnl = weth_qty * (results_df.iloc[i-1]['weth_hedge_entry_price'] - results_df.iloc[i]['weth_futures_price'])
        
        # Transaction fees for closing
        uni_fee = abs(uni_qty * results_df.iloc[i]['uni_futures_price'] * FEE_RATE)
        weth_fee = abs(weth_qty * results_df.iloc[i]['weth_futures_price'] * FEE_RATE)
        
        # Total realized PnL
        total_pnl = uni_pnl + weth_pnl - uni_fee - weth_fee
        results_df.iloc[i, results_df.columns.get_loc('hedge_realized_pnl_usd')] = results_df.iloc[i-1]['hedge_realized_pnl_usd'] + total_pnl
        
        # Open new hedge
        results_df.iloc[i, results_df.columns.get_loc('uni_hedge_qty')] = -results_df.iloc[i]['token_0_total']
        results_df.iloc[i, results_df.columns.get_loc('weth_hedge_qty')] = -results_df.iloc[i]['token_1_total']
        results_df.iloc[i, results_df.columns.get_loc('uni_hedge_entry_price')] = results_df.iloc[i]['uni_futures_price']
        results_df.iloc[i, results_df.columns.get_loc('weth_hedge_entry_price')] = results_df.iloc[i]['weth_futures_price']
        
        # Transaction fees for opening
        uni_fee_open = abs(results_df.iloc[i]['uni_hedge_qty'] * results_df.iloc[i]['uni_futures_price'] * FEE_RATE)
        weth_fee_open = abs(results_df.iloc[i]['weth_hedge_qty'] * results_df.iloc[i]['weth_futures_price'] * FEE_RATE)
        results_df.iloc[i, results_df.columns.get_loc('hedge_realized_pnl_usd')] -= (uni_fee_open + weth_fee_open)
    
    # Update hedged portfolio value
    results_df.iloc[i, results_df.columns.get_loc('value_position_hedged_usd')] = (
        results_df.iloc[i]['value_position_usd'] + results_df.iloc[i]['hedge_realized_pnl_usd']
    )

# Save updated results
results_df.to_csv('strategy_results_hedged.csv')

# Updated plot_position_value
def plot_position_value(data_strategy):
    import plotly.graph_objects as go
    CHART_SIZE = 300
    required_columns = ['time_pd', 'value_position_usd', 'value_hold_usd', 'value_position_hedged_usd']
    if not all(col in data_strategy.columns for col in required_columns):
        print("Error: Missing required columns in data_strategy")
        return None

    print("time_pd dtype:", data_strategy['time_pd'].dtype)
    print("time_pd head:", data_strategy['time_pd'].head().to_list())

    data_strategy = data_strategy.copy()
    data_strategy['time_pd'] = pd.to_datetime(data_strategy['time_pd'], utc=True, errors='coerce')
    if data_strategy['time_pd'].isna().any():
        print("Warning: NaN values in time_pd after conversion")
        return None

    fig_strategy = go.Figure()
    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time_pd'], 
        y=data_strategy['value_position_usd'],
        name='Value of LP Position',
        line=dict(width=2, color='red')
    ))
    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time_pd'], 
        y=data_strategy['value_hold_usd'],
        name='Value of Holding',
        line=dict(width=2, color='blue')
    ))
    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time_pd'], 
        y=data_strategy['value_position_hedged_usd'],
        name='Value of Hedged LP Position',
        line=dict(width=2, color='green')
    ))

    fig_strategy.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=CHART_SIZE,
        title='Strategy Simulation — LP Position vs. Holding vs. Hedged',
        xaxis_title="Date",
        yaxis_title='Position Value (USD)',
        xaxis_type='date',
        xaxis=dict(tickformat='%Y-%m-%d', tickmode='auto')
    )

    fig_strategy.show(renderer="png")
    return fig_strategy

# Plot results
plot_position_value(results_df)

# Print total PnL
initial_value = results_df.iloc[0]['value_position_usd']
final_value_unhedged = results_df.iloc[-1]['value_position_usd']
final_value_hedged = results_df.iloc[-1]['value_position_hedged_usd']
print(f"Initial Portfolio Value: ${initial_value:.2f}")
print(f"Final Unhedged Portfolio Value: ${final_value_unhedged:.2f}")
print(f"Unhedged PnL: ${(final_value_unhedged - initial_value):.2f}")
print(f"Final Hedged Portfolio Value: ${final_value_hedged:.2f}")
print(f"Hedged PnL: ${(final_value_hedged - initial_value):.2f}")