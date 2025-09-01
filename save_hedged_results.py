import sys
import os
import pandas as pd
import numpy as np
from hedging import apply_delta_hedging


FUTURES_DIR_DEFAULT = '/Users/nicolaschiavo/Dev/tesi/univ3-strategies/data/centralized_prices/futures'
OUTPUT_DEFAULT = '/Users/nicolaschiavo/Dev/tesi/univ3-strategies/results_opt_hedged.csv'


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Prefer explicit time column if present
    if 'time_pd' in df.columns:
        df['time_pd'] = pd.to_datetime(df['time_pd'], utc=True)
    elif 'time' in df.columns:
        df['time_pd'] = pd.to_datetime(df['time'], utc=True)
    else:
        raise ValueError('results CSV missing time or time_pd column')
    return df


def align_futures(series_time: pd.Series, futures_path: str, price_col_name: str) -> pd.Series:
    fut = pd.read_csv(futures_path)
    fut['time_pd'] = pd.to_datetime(fut['datetime'], utc=True)
    fut = fut[['time_pd', 'close']].sort_values('time_pd')
    base = pd.DataFrame({'time_pd': pd.to_datetime(series_time, utc=True)})
    merged = pd.merge_asof(base.sort_values('time_pd'), fut, on='time_pd', direction='backward')
    return merged['close'].rename(price_col_name)


def infer_pool_tokens(df: pd.DataFrame, futures_dir: str) -> tuple[str, str]:
    candidates = ['link', 'inj', 'morpho', 'wtao']
    eth_path = os.path.join(futures_dir, 'eth_bitget_futures_5m.csv')
    if not os.path.exists(eth_path):
        raise FileNotFoundError(f'ETH futures not found at {eth_path}')

    # Preload ETH futures aligned to df timestamps
    eth_prices = align_futures(df['time_pd'], eth_path, 'eth')
    best = (None, None, -np.inf, False)  # token_slug, order_token0, corr, inverted

    for token in candidates:
        token_path = os.path.join(futures_dir, f'{token}_bitget_futures_5m.csv')
        if not os.path.exists(token_path):
            continue
        tok_prices = align_futures(df['time_pd'], token_path, token)
        # price in strategy is token1/token0
        # Try mapping 1: token0 = eth, token1 = token  => ratio = token/eth
        ratio_1 = tok_prices / eth_prices
        corr_1 = np.corrcoef(df['price'].astype(float).values, ratio_1.astype(float).values)[0, 1]
        # Try mapping 2: token0 = token, token1 = eth => ratio = eth/token
        ratio_2 = eth_prices / tok_prices
        corr_2 = np.corrcoef(df['price'].astype(float).values, ratio_2.astype(float).values)[0, 1]

        if np.isfinite(corr_1) and corr_1 > best[2]:
            best = (token, True, corr_1, False)  # token0=eth, token1=token
        if np.isfinite(corr_2) and corr_2 > best[2]:
            best = (token, False, corr_2, True)  # token0=token, token1=eth

    if best[0] is None or best[2] < 0.5:
        # Fallback to LINK/ETH assumption
        return ('eth', 'link')

    token = best[0]
    if best[1]:
        # token0 = eth, token1 = token
        return ('eth', token)
    else:
        # token0 = token, token1 = eth
        return (token, 'eth')


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/nicolaschiavo/Dev/tesi/univ3-strategies/results_opt.csv'
    futures_dir = sys.argv[2] if len(sys.argv) > 2 else FUTURES_DIR_DEFAULT
    output_path = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DEFAULT

    df = load_results(results_path)
    # Ensure numeric columns present
    required = ['token_0_total', 'token_1_total', 'reset_point', 'value_position_usd', 'price']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in results: {missing}')

    token0_slug, token1_slug = infer_pool_tokens(df, futures_dir)
    print(f'Inferred pool mapping -> token0: {token0_slug}, token1: {token1_slug}')

    hedged = apply_delta_hedging(
        strategy_df=df,
        token0_symbol_slug=token0_slug,
        token1_symbol_slug=token1_slug,
        futures_dir=futures_dir,
        fee_rate=0.0002
    )

    hedged.reset_index().to_csv(output_path, index=False)
    init_val = float(hedged.iloc[0]['value_position_usd']) if 'value_position_usd' in hedged.columns else np.nan
    final_unhedged = float(hedged.iloc[-1]['value_position_usd']) if 'value_position_usd' in hedged.columns else np.nan
    final_hedged = float(hedged.iloc[-1]['value_position_hedged_usd']) if 'value_position_hedged_usd' in hedged.columns else np.nan
    print(f'Saved hedged results to: {output_path}')
    print(f'Initial value: {init_val:.2f} | Final unhedged: {final_unhedged:.2f} | Final hedged: {final_hedged:.2f}')


if __name__ == '__main__':
    main()





