import pandas as pd
import numpy as np
from typing import Optional


def _load_futures_prices(symbol_slug: str, futures_dir: str) -> pd.DataFrame:
    """
    Load Bitget perpetual futures OHLCV for a token symbol slug.

    Expects CSV with columns: datetime, timestamp, open, high, low, close, volume
    and UTC timestamps in the 'datetime' column.
    """
    file_path = f"{futures_dir.rstrip('/')}/{symbol_slug}_bitget_futures_5m.csv"
    df = pd.read_csv(file_path)
    if 'datetime' not in df.columns or 'close' not in df.columns:
        raise ValueError(f"Futures file missing required columns: {file_path}")
    df['time_pd'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('time_pd').sort_index()
    return df[['close']].rename(columns={'close': f'{symbol_slug}_futures_price'})


def apply_delta_hedging(
    strategy_df: pd.DataFrame,
    token0_symbol_slug: str,
    token1_symbol_slug: str,
    futures_dir: str = '/Users/nicolaschiavo/Dev/tesi/univ3-strategies/data/centralized_prices/futures',
    fee_rate: float = 0.0002,
    time_col: str = 'time_pd',
    token0_total_col: str = 'token_0_total',
    token1_total_col: str = 'token_1_total',
    reset_flag_col: str = 'reset_point',
    value_usd_col: str = 'value_position_usd'
) -> pd.DataFrame:
    """
    Apply simple delta hedging using Bitget perpetual futures close prices.

    - Opens hedge at first row sized to -token holdings.
    - On each reset (reset_flag_col == True), closes previous hedge (realized PnL
      captured), then opens a new hedge to current exposures.
    - Adds columns for realized/unrealized PnL and value_position_hedged_usd.

    Returns a new DataFrame with hedging columns.
    """
    required_cols = [time_col, token0_total_col, token1_total_col, reset_flag_col]
    for col in required_cols:
        if col not in strategy_df.columns and (col != time_col or strategy_df.index.name != time_col):
            raise ValueError(f"Missing required column in strategy_df: {col}")

    # Ensure DatetimeIndex for merge_asof alignment
    df = strategy_df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Strategy DataFrame must be indexed by datetime or include time_col.")

    # Load futures and align by time (nearest earlier bar)
    fut0 = _load_futures_prices(token0_symbol_slug, futures_dir)
    fut1 = _load_futures_prices(token1_symbol_slug, futures_dir)

    # Use merge_asof to align 5m futures to strategy timestamps
    df_sorted = df.sort_index().copy()
    fut0_sorted = fut0.sort_index().copy()
    fut1_sorted = fut1.sort_index().copy()

    # Merge token0
    tmp = df_sorted.reset_index().rename(columns={df_sorted.index.name: 'time_pd'})
    fut0_reset = fut0_sorted.reset_index().rename(columns={'time_pd': 'fut_time_0'})
    fut1_reset = fut1_sorted.reset_index().rename(columns={'time_pd': 'fut_time_1'})
    tmp = pd.merge_asof(tmp.sort_values('time_pd'), fut0_reset.sort_values('fut_time_0'), left_on='time_pd', right_on='fut_time_0', direction='backward')
    tmp = pd.merge_asof(tmp.sort_values('time_pd'), fut1_reset.sort_values('fut_time_1'), left_on='time_pd', right_on='fut_time_1', direction='backward')
    tmp = tmp.set_index('time_pd').sort_index()

    price_col_0 = f'{token0_symbol_slug}_futures_price'
    price_col_1 = f'{token1_symbol_slug}_futures_price'
    if price_col_0 not in tmp.columns or price_col_1 not in tmp.columns:
        raise ValueError("Aligned futures price columns missing after merge.")

    # Forward-fill to handle any small gaps
    tmp[price_col_0] = tmp[price_col_0].ffill()
    tmp[price_col_1] = tmp[price_col_1].ffill()

    # Coerce reset flag to boolean explicitly (CSV can have 'True'/'False' strings)
    if reset_flag_col in tmp.columns:
        if tmp[reset_flag_col].dtype != bool:
            tmp[reset_flag_col] = tmp[reset_flag_col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
            tmp[reset_flag_col] = tmp[reset_flag_col].fillna(False).astype(bool)

    # Initialize hedge state columns
    tmp['hedge_qty_token0'] = 0.0
    tmp['hedge_qty_token1'] = 0.0
    tmp['hedge_entry_price_token0'] = 0.0
    tmp['hedge_entry_price_token1'] = 0.0
    tmp['hedge_realized_pnl_usd'] = 0.0
    tmp['hedge_unrealized_pnl_usd'] = 0.0

    # Open initial hedge at first row
    if len(tmp) == 0:
        return tmp
    first_idx = tmp.index[0]
    tmp.loc[first_idx, 'hedge_qty_token0'] = -float(tmp.loc[first_idx, token0_total_col])
    tmp.loc[first_idx, 'hedge_qty_token1'] = -float(tmp.loc[first_idx, token1_total_col])
    tmp.loc[first_idx, 'hedge_entry_price_token0'] = float(tmp.loc[first_idx, price_col_0])
    tmp.loc[first_idx, 'hedge_entry_price_token1'] = float(tmp.loc[first_idx, price_col_1])

    # Iterate forward computing PnL and re-hedging on reset
    idx_list = list(tmp.index)
    for i in range(1, len(idx_list)):
        t_prev = idx_list[i - 1]
        t_now = idx_list[i]

        # Carry hedge positions
        tmp.loc[t_now, 'hedge_qty_token0'] = float(tmp.loc[t_prev, 'hedge_qty_token0'])
        tmp.loc[t_now, 'hedge_qty_token1'] = float(tmp.loc[t_prev, 'hedge_qty_token1'])
        tmp.loc[t_now, 'hedge_entry_price_token0'] = float(tmp.loc[t_prev, 'hedge_entry_price_token0'])
        tmp.loc[t_now, 'hedge_entry_price_token1'] = float(tmp.loc[t_prev, 'hedge_entry_price_token1'])
        tmp.loc[t_now, 'hedge_realized_pnl_usd'] = float(tmp.loc[t_prev, 'hedge_realized_pnl_usd'])

        # Re-hedge at reset points
        if bool(tmp.loc[t_now, reset_flag_col]):
            # Close previous hedge -> realized PnL
            close_qty_0 = -tmp.loc[t_prev, 'hedge_qty_token0']
            close_qty_1 = -tmp.loc[t_prev, 'hedge_qty_token1']
            pnl_0 = close_qty_0 * (tmp.loc[t_prev, 'hedge_entry_price_token0'] - tmp.loc[t_now, price_col_0])
            pnl_1 = close_qty_1 * (tmp.loc[t_prev, 'hedge_entry_price_token1'] - tmp.loc[t_now, price_col_1])
            fee_0_close = abs(close_qty_0 * tmp.loc[t_now, price_col_0] * fee_rate)
            fee_1_close = abs(close_qty_1 * tmp.loc[t_now, price_col_1] * fee_rate)
            realized = pnl_0 + pnl_1 - fee_0_close - fee_1_close
            tmp.loc[t_now, 'hedge_realized_pnl_usd'] = float(tmp.loc[t_prev, 'hedge_realized_pnl_usd'] + realized)

            # Open new hedge at current exposure
            new_qty_0 = -float(tmp.loc[t_now, token0_total_col])
            new_qty_1 = -float(tmp.loc[t_now, token1_total_col])
            tmp.loc[t_now, 'hedge_qty_token0'] = new_qty_0
            tmp.loc[t_now, 'hedge_qty_token1'] = new_qty_1
            tmp.loc[t_now, 'hedge_entry_price_token0'] = float(tmp.loc[t_now, price_col_0])
            tmp.loc[t_now, 'hedge_entry_price_token1'] = float(tmp.loc[t_now, price_col_1])

            # Opening fees
            fee_0_open = abs(new_qty_0 * tmp.loc[t_now, price_col_0] * fee_rate)
            fee_1_open = abs(new_qty_1 * tmp.loc[t_now, price_col_1] * fee_rate)
            tmp.loc[t_now, 'hedge_realized_pnl_usd'] = float(tmp.loc[t_now, 'hedge_realized_pnl_usd'] - fee_0_open - fee_1_open)

            # Unrealized PnL right after re-hedge is zero by construction
            tmp.loc[t_now, 'hedge_unrealized_pnl_usd'] = 0.0
        else:
            # Unrealized PnL for information (no reset this bar)
            u_pnl_0 = tmp.loc[t_now, 'hedge_qty_token0'] * (tmp.loc[t_now, price_col_0] - tmp.loc[t_now, 'hedge_entry_price_token0'])
            u_pnl_1 = tmp.loc[t_now, 'hedge_qty_token1'] * (tmp.loc[t_now, price_col_1] - tmp.loc[t_now, 'hedge_entry_price_token1'])
            tmp.loc[t_now, 'hedge_unrealized_pnl_usd'] = float(u_pnl_0 + u_pnl_1)

    # Hedged portfolio value (only realized PnL included)
    if value_usd_col in tmp.columns:
        tmp['value_position_hedged_usd'] = tmp[value_usd_col] + tmp['hedge_realized_pnl_usd']
    else:
        # If USD value missing, still return hedging columns; caller can compute their own value
        tmp['value_position_hedged_usd'] = np.nan

    return tmp


