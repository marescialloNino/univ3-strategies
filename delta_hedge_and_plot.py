import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from hedging import apply_delta_hedging
from save_hedged_results import infer_pool_tokens


def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'time_pd' in df.columns:
        df['time_pd'] = pd.to_datetime(df['time_pd'], utc=True)
    elif 'time' in df.columns:
        df['time_pd'] = pd.to_datetime(df['time'], utc=True)
    else:
        raise ValueError('results CSV missing time or time_pd column')
    return df


def plot_pnl(hedged_df: pd.DataFrame, output_path: str) -> None:
    if 'value_position_usd' not in hedged_df.columns:
        raise ValueError('value_position_usd column missing in results for plotting')
    if 'value_position_hedged_usd' not in hedged_df.columns:
        raise ValueError('value_position_hedged_usd column missing after hedging')

    initial_value = float(hedged_df.iloc[0]['value_position_usd'])
    if isinstance(hedged_df.index, pd.DatetimeIndex):
        t = hedged_df.index.to_pydatetime()
    else:
        t = pd.to_datetime(hedged_df['time_pd'], utc=True).to_pydatetime()

    unhedged_pnl = (hedged_df['value_position_usd'] - initial_value).to_numpy()
    hedged_pnl = (hedged_df['value_position_hedged_usd'] - initial_value).to_numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(t, unhedged_pnl, label='Unhedged PnL', linewidth=1.8)
    plt.plot(t, hedged_pnl, label='Hedged PnL', linewidth=1.8)
    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.title('PnL: Unhedged vs Hedged')
    plt.xlabel('Time')
    plt.ylabel('PnL (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Apply delta hedging using local futures and plot PnL comparison.')
    parser.add_argument('--results', type=str, default='/Users/nicolaschiavo/Dev/tesi/univ3-strategies/results_opt.csv', help='Path to results CSV (input)')
    parser.add_argument('--futures-dir', type=str, default='/Users/nicolaschiavo/Dev/tesi/univ3-strategies/data/centralized_prices/futures', help='Directory containing local futures CSVs')
    parser.add_argument('--token0', type=str, default=None, help='Token0 slug matching <slug>_bitget_futures_5m.csv (e.g., eth, link, inj, morpho, wtao)')
    parser.add_argument('--token1', type=str, default=None, help='Token1 slug matching <slug>_bitget_futures_5m.csv')
    parser.add_argument('--output', type=str, default='/Users/nicolaschiavo/Dev/tesi/univ3-strategies/results_opt_hedged.csv', help='Output CSV path (ignored if --inplace)')
    parser.add_argument('--inplace', action='store_true', help='Overwrite the input results CSV with hedging columns')
    parser.add_argument('--plot', type=str, default='/Users/nicolaschiavo/Dev/tesi/univ3-strategies/pnl_hedged_vs_unhedged.png', help='Output PNG path for the PnL plot')

    args = parser.parse_args()

    df = load_results_csv(args.results)

    # Validate required columns exist
    required_cols = ['token_0_total', 'token_1_total', 'reset_point', 'value_position_usd', 'price']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in results: {missing}')

    # Infer tokens if not provided
    if args.token0 is None or args.token1 is None:
        token0_slug, token1_slug = infer_pool_tokens(df, args.futures_dir)
    else:
        token0_slug, token1_slug = args.token0, args.token1

    print(f'Hedging with token0={token0_slug}, token1={token1_slug} (short futures)')

    hedged = apply_delta_hedging(
        strategy_df=df,
        token0_symbol_slug=token0_slug,
        token1_symbol_slug=token1_slug,
        futures_dir=args.futures_dir,
        fee_rate=0.0002
    )

    # Save CSV
    out_path = args.results if args.inplace else args.output
    # Ensure consistent order with index as time
    hedged_out = hedged.reset_index()
    hedged_out.to_csv(out_path, index=False)
    print(f'Saved hedged results to: {out_path}')

    # Plot PnL comparison
    plot_pnl(hedged, args.plot)
    print(f'Saved PnL comparison plot to: {args.plot}')


if __name__ == '__main__':
    main()


