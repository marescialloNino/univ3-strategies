import pandas as pd
import numpy as np
from typing import Optional


def _load_futures_prices(symbol_slug: str, futures_dir: str) -> pd.DataFrame:
    """
    Load Bitget perpetual futures OHLCV for a token symbol slug.

    Expects CSV with columns: datetime, timestamp, open, high, low, close, volume
    and UTC timestamps in the 'datetime' column.
    """
    file_path = f"{futures_dir.rstrip('/')}/{symbol_slug}_bitget_futures_1h.csv"
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
    value_usd_col: str = 'value_position_usd',
    # Price source options
    price_source: str = 'futures',  # 'futures' or 'usd'
    token0_usd_col: str = 'price_0_usd',
    token1_usd_col: Optional[str] = None,
    quote_price_col: str = 'price',
    # Hedging control
    hedge_only: str = 'both',  # 'both' or 'token0'
    # Reporting unit extension
    reporting_unit: str = 'usd'  # 'usd' or 'token1'
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

    df_sorted = df.sort_index().copy()

    if price_source.lower() == 'usd':
        # Use USD price columns from the strategy_df directly
        tmp = df_sorted.copy()
        # Ensure token0 USD column exists
        if token0_usd_col not in tmp.columns:
            raise ValueError(f"token0_usd_col '{token0_usd_col}' not found in strategy_df")
        # Create/validate token1 USD column
        if token1_usd_col is None or token1_usd_col not in tmp.columns:
            if quote_price_col not in tmp.columns:
                raise ValueError("token1_usd_col not provided and quote_price_col missing to derive it")
            # token1_usd = token0_usd / quote_price (price = token1 per token0)
            tmp['__token1_usd'] = (tmp[token0_usd_col].astype(float) / tmp[quote_price_col].astype(float)).replace([np.inf, -np.inf], np.nan)
            token1_usd_col_internal = '__token1_usd'
        else:
            token1_usd_col_internal = token1_usd_col

        price_col_0 = token0_usd_col
        price_col_1 = token1_usd_col_internal
        # Forward-fill any holes
        tmp[price_col_0] = tmp[price_col_0].astype(float).ffill()
        tmp[price_col_1] = tmp[price_col_1].astype(float).ffill()
        # Standardized helper columns for plotting/conversions
        tmp['token0_usd_price'] = tmp[price_col_0].astype(float)
        tmp['token1_usd_price'] = tmp[price_col_1].astype(float)
    else:
        # Load futures and align by time (nearest earlier bar)
        fut0 = _load_futures_prices(token0_symbol_slug, futures_dir)
        fut1 = _load_futures_prices(token1_symbol_slug, futures_dir)

        fut0_sorted = fut0.sort_index().copy()
        fut1_sorted = fut1.sort_index().copy()

        # Merge token0 and token1 futures via asof (backward)
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
        tmp['token0_usd_price'] = tmp[price_col_0].astype(float)
        tmp['token1_usd_price'] = tmp[price_col_1].astype(float)

    # Coerce reset flag to boolean explicitly (CSV can have 'True'/'False' strings)
    if reset_flag_col in tmp.columns:
        if tmp[reset_flag_col].dtype != bool:
            tmp[reset_flag_col] = tmp[reset_flag_col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
            tmp[reset_flag_col] = tmp[reset_flag_col].fillna(False).astype(bool)

    # Initialize hedge state columns
    tmp['hedge_qty_token0'] = 0.0
    tmp['hedge_qty_token1'] = 0.0
    tmp['hedge_mean_price_token0'] = 0.0  # Mean acquisition price
    tmp['hedge_mean_price_token1'] = 0.0  # Mean acquisition price
    tmp['hedge_realized_pnl_usd'] = 0.0
    tmp['hedge_unrealized_pnl_usd'] = 0.0

    # Open initial hedge at first row
    if len(tmp) == 0:
        return tmp
    first_idx = tmp.index[0]
    tmp.loc[first_idx, 'hedge_qty_token0'] = -float(tmp.loc[first_idx, token0_total_col])
    tmp.loc[first_idx, 'hedge_qty_token1'] = -float(tmp.loc[first_idx, token1_total_col])
    tmp.loc[first_idx, 'hedge_mean_price_token0'] = float(tmp.loc[first_idx, price_col_0])
    tmp.loc[first_idx, 'hedge_mean_price_token1'] = float(tmp.loc[first_idx, price_col_1])

    # Iterate forward computing PnL and re-hedging on reset
    idx_list = list(tmp.index)
    for i in range(1, len(idx_list)):
        t_prev = idx_list[i - 1]
        t_now = idx_list[i]

        # Carry hedge positions and mean prices
        tmp.loc[t_now, 'hedge_qty_token0'] = float(tmp.loc[t_prev, 'hedge_qty_token0'])
        tmp.loc[t_now, 'hedge_qty_token1'] = float(tmp.loc[t_prev, 'hedge_qty_token1'])
        tmp.loc[t_now, 'hedge_mean_price_token0'] = float(tmp.loc[t_prev, 'hedge_mean_price_token0'])
        tmp.loc[t_now, 'hedge_mean_price_token1'] = float(tmp.loc[t_prev, 'hedge_mean_price_token1'])
        tmp.loc[t_now, 'hedge_realized_pnl_usd'] = float(tmp.loc[t_prev, 'hedge_realized_pnl_usd'])

        # Re-hedge at reset points
        if bool(tmp.loc[t_now, reset_flag_col]):
            # Target quantities for new hedge
            target_qty_0 = -float(tmp.loc[t_now, token0_total_col])
            target_qty_1 = -float(tmp.loc[t_now, token1_total_col])
            if hedge_only.lower() == 'token0':
                # Only hedge token0; do not short token1
                target_qty_1 = 0.0
            elif hedge_only.lower() == 'token1':
                # Only hedge token1; do not short token0
                target_qty_0 = 0.0
            curr_qty_0 = tmp.loc[t_now, 'hedge_qty_token0']
            curr_qty_1 = tmp.loc[t_now, 'hedge_qty_token1']
            
            # Token 0 hedge adjustment
            if abs(target_qty_0) < abs(curr_qty_0):  # Reducing short position
                close_qty_0 = curr_qty_0 - target_qty_0  # Positive number (reducing short)
                # Realized PnL on closed portion
                pnl_0 = abs(close_qty_0) * (tmp.loc[t_now, 'hedge_mean_price_token0'] - tmp.loc[t_now, price_col_0])
                fee_0 = abs(close_qty_0 * tmp.loc[t_now, price_col_0] * fee_rate)
                tmp.loc[t_now, 'hedge_realized_pnl_usd'] += float(pnl_0 - fee_0)
                # Keep same mean price for remaining position
                tmp.loc[t_now, 'hedge_qty_token0'] = target_qty_0
            else:  # Increasing short position or same
                if abs(target_qty_0) > abs(curr_qty_0):  # Only if actually increasing
                    add_qty_0 = target_qty_0 - curr_qty_0  # Negative when adding to a short
                    # Update mean acquisition price with weighted average
                    denom_0 = abs(curr_qty_0) + abs(add_qty_0)
                    if denom_0 > 0:
                        new_mean_0 = (
                            (abs(curr_qty_0) * tmp.loc[t_now, 'hedge_mean_price_token0']) +
                            (abs(add_qty_0) * tmp.loc[t_now, price_col_0])
                        ) / denom_0
                    else:
                        new_mean_0 = float(tmp.loc[t_now, price_col_0])
                    tmp.loc[t_now, 'hedge_mean_price_token0'] = float(new_mean_0)
                    # Apply fees on added quantity
                    fee_0 = abs(add_qty_0 * tmp.loc[t_now, price_col_0] * fee_rate)
                    tmp.loc[t_now, 'hedge_realized_pnl_usd'] -= float(fee_0)
                tmp.loc[t_now, 'hedge_qty_token0'] = target_qty_0

            # Token 1 hedge adjustment (same logic as token 0)
            if abs(target_qty_1) < abs(curr_qty_1):  # Reducing short position
                close_qty_1 = curr_qty_1 - target_qty_1  # Positive number (reducing short)
                # Realized PnL on closed portion
                pnl_1 = abs(close_qty_1) * (tmp.loc[t_now, 'hedge_mean_price_token1'] - tmp.loc[t_now, price_col_1])
                fee_1 = abs(close_qty_1 * tmp.loc[t_now, price_col_1] * fee_rate)
                tmp.loc[t_now, 'hedge_realized_pnl_usd'] += float(pnl_1 - fee_1)
                # Keep same mean price for remaining position
                tmp.loc[t_now, 'hedge_qty_token1'] = target_qty_1
            else:  # Increasing short position or same
                if abs(target_qty_1) > abs(curr_qty_1):  # Only if actually increasing
                    add_qty_1 = target_qty_1 - curr_qty_1  # Negative when adding to a short
                    # Update mean acquisition price with weighted average
                    denom_1 = abs(curr_qty_1) + abs(add_qty_1)
                    if denom_1 > 0:
                        new_mean_1 = (
                            (abs(curr_qty_1) * tmp.loc[t_now, 'hedge_mean_price_token1']) +
                            (abs(add_qty_1) * tmp.loc[t_now, price_col_1])
                        ) / denom_1
                    else:
                        new_mean_1 = float(tmp.loc[t_now, price_col_1])
                    tmp.loc[t_now, 'hedge_mean_price_token1'] = float(new_mean_1)
                    # Apply fees on added quantity
                    fee_1 = abs(add_qty_1 * tmp.loc[t_now, price_col_1] * fee_rate)
                    tmp.loc[t_now, 'hedge_realized_pnl_usd'] -= float(fee_1)
                tmp.loc[t_now, 'hedge_qty_token1'] = target_qty_1

        # Calculate unrealized PnL
        u_pnl_0 = abs(tmp.loc[t_now, 'hedge_qty_token0']) * (tmp.loc[t_now, 'hedge_mean_price_token0'] - tmp.loc[t_now, price_col_0])
        u_pnl_1 = abs(tmp.loc[t_now, 'hedge_qty_token1']) * (tmp.loc[t_now, 'hedge_mean_price_token1'] - tmp.loc[t_now, price_col_1])
        tmp.loc[t_now, 'hedge_unrealized_pnl_usd'] = float(u_pnl_0 + u_pnl_1)

    # Calculate hedge value in USD using futures prices
    tmp['hedge_value_usd'] = (
        tmp['hedge_qty_token0'] * tmp[price_col_0] + 
        tmp['hedge_qty_token1'] * tmp[price_col_1]
    )

    # Hedged portfolio value (MTM): LP value + hedge MTM value
    if value_usd_col in tmp.columns:
        tmp['value_position_hedged_usd'] = tmp[value_usd_col] + tmp['hedge_value_usd']
    else:
        tmp['value_position_hedged_usd'] = np.nan

    # Add compatibility columns expected by plotting functions
    tmp['value_position_hedged_usd_mtm'] = tmp['value_position_hedged_usd']
    tmp['total_hedged_value_usd_mtm'] = tmp['value_position_hedged_usd']
    tmp['hedge_realized_pnl_total_usd'] = tmp.get('hedge_realized_pnl_usd', 0.0)
    tmp['hedge_unrealized_pnl_total_usd'] = tmp.get('hedge_unrealized_pnl_usd', 0.0)

    # Ensure time_pd is a column, not just index
    if tmp.index.name == 'time_pd':
        tmp = tmp.reset_index()

    # Optional: reporting in token1 units (requires token1 USD price column)
    if price_source.lower() == 'usd' and reporting_unit.lower() == 'token1':
        conv = tmp[price_col_1].astype(float).replace(0.0, np.nan)
        def _div_safe(series):
            return series.astype(float) / conv

        if value_usd_col in tmp.columns:
            tmp['value_position_token1'] = _div_safe(tmp[value_usd_col])
        if 'value_hold_usd' in tmp.columns:
            tmp['value_hold_token1'] = _div_safe(tmp['value_hold_usd'])
        tmp['hedge_value_token1'] = _div_safe(tmp['hedge_value_usd'])
        tmp['value_position_hedged_token1'] = _div_safe(tmp['value_position_hedged_usd'])

        if 'hedge_realized_pnl_usd' in tmp.columns:
            tmp['hedge_realized_pnl_total_token1'] = _div_safe(tmp['hedge_realized_pnl_usd'])
        if 'hedge_unrealized_pnl_usd' in tmp.columns:
            tmp['hedge_unrealized_pnl_total_token1'] = _div_safe(tmp['hedge_unrealized_pnl_usd'])

    return tmp


