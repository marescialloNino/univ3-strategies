import pandas as pd
import numpy as np
import math
import UNI_v3_funcs
import copy

class StrategyObservation:
    def __init__(self, timepoint, current_price, strategy_in, liquidity_in_0, liquidity_in_1, fee_tier,
                 decimals_0, decimals_1, token_0_left_over=0.0, token_1_left_over=0.0,
                 token_0_fees_uncollected=0.0, token_1_fees_uncollected=0.0, liquidity_ranges=None,
                 strategy_info=None, swaps=None, simulate_strat=True, price_0_usd=None):
        """Initialize the StrategyObservation with additional price_0_usd parameter."""
        self.time = timepoint
        self.price = current_price
        self.liquidity_in_0 = liquidity_in_0
        self.liquidity_in_1 = liquidity_in_1
        self.fee_tier = fee_tier
        self.decimals_0 = decimals_0
        self.decimals_1 = decimals_1
        self.token_0_left_over = token_0_left_over
        self.token_1_left_over = token_1_left_over
        self.token_0_fees_uncollected = token_0_fees_uncollected
        self.token_1_fees_uncollected = token_1_fees_uncollected
        self.reset_point = False
        self.compound_point = False
        self.reset_reason = ''
        self.decimal_adjustment = 10**(self.decimals_1 - self.decimals_0)
        self.tickSpacing = int(self.fee_tier * 2 * 10000) if self.fee_tier > (100 / 1e6) else int(self.fee_tier * 10000)
        self.token_0_fees = 0.0
        self.token_1_fees = 0.0
        self.simulate_strat = simulate_strat
        self.strategy_info = copy.deepcopy(strategy_info)
        self.price_0_usd = price_0_usd  # USD price of token_0

        TICK_P_PRE = math.log(self.decimal_adjustment * self.price, 1.0001)
        self.price_tick = math.floor(TICK_P_PRE / self.tickSpacing) * self.tickSpacing
        self.price_tick_current = math.floor(TICK_P_PRE)

        if liquidity_ranges is None:
            self.liquidity_ranges, self.strategy_info = strategy_in.set_liquidity_ranges(self)
        else:
            self.liquidity_ranges = copy.deepcopy(liquidity_ranges)
            for i in range(len(self.liquidity_ranges)):
                self.liquidity_ranges[i]['time'] = self.time
                if self.simulate_strat:
                    amount_0, amount_1 = UNI_v3_funcs.get_amounts(self.price_tick_current,
                                                                 self.liquidity_ranges[i]['lower_bin_tick'],
                                                                 self.liquidity_ranges[i]['upper_bin_tick'],
                                                                 self.liquidity_ranges[i]['position_liquidity'],
                                                                 self.decimals_0, self.decimals_1)
                    self.liquidity_ranges[i]['token_0'] = amount_0
                    self.liquidity_ranges[i]['token_1'] = amount_1

            if swaps is not None:
                fees_token_0, fees_token_1 = self.accrue_fees(swaps)
                self.token_0_fees = fees_token_0
                self.token_1_fees = fees_token_1

            self.liquidity_ranges, self.strategy_info = strategy_in.check_strategy(self)

    def accrue_fees(self, relevant_swaps):
        fees_earned_token_0 = 0.0
        fees_earned_token_1 = 0.0
        if len(relevant_swaps) > 0:
            for s in range(len(relevant_swaps)):
                for i in range(len(self.liquidity_ranges)):
                    in_range = (self.liquidity_ranges[i]['lower_bin_tick'] <= relevant_swaps.iloc[s]['tick_swap']) and \
                               (self.liquidity_ranges[i]['upper_bin_tick'] >= relevant_swaps.iloc[s]['tick_swap'])
                    token_0_in = relevant_swaps.iloc[s]['token_in'] == 'token0'
                    fraction_fees_earned_position = 1 if relevant_swaps.iloc[s]['virtual_liquidity'] < 1e-9 else \
                        self.liquidity_ranges[i]['position_liquidity'] / (self.liquidity_ranges[i]['position_liquidity'] + relevant_swaps.iloc[s]['virtual_liquidity'])
                    fees_earned_token_0 += in_range * token_0_in * self.fee_tier * fraction_fees_earned_position * relevant_swaps.iloc[s]['traded_in']
                    fees_earned_token_1 += in_range * (1 - token_0_in) * self.fee_tier * fraction_fees_earned_position * relevant_swaps.iloc[s]['traded_in']
        self.token_0_fees_uncollected += fees_earned_token_0
        self.token_1_fees_uncollected += fees_earned_token_1
        return fees_earned_token_0, fees_earned_token_1

    def remove_liquidity(self):
        removed_amount_0 = 0.0
        removed_amount_1 = 0.0
        for i in range(len(self.liquidity_ranges)):
            position_liquidity = self.liquidity_ranges[i]['position_liquidity']
            TICK_A = self.liquidity_ranges[i]['lower_bin_tick']
            TICK_B = self.liquidity_ranges[i]['upper_bin_tick']
            token_amounts = UNI_v3_funcs.get_amounts(self.price_tick, TICK_A, TICK_B,
                                                     position_liquidity, self.decimals_0, self.decimals_1)
            removed_amount_0 += token_amounts[0]
            removed_amount_1 += token_amounts[1]
        self.liquidity_in_0 = removed_amount_0 + self.token_0_left_over + self.token_0_fees_uncollected
        self.liquidity_in_1 = removed_amount_1 + self.token_1_left_over + self.token_1_fees_uncollected
        self.token_0_left_over = 0.0
        self.token_1_left_over = 0.0
        self.token_0_fees_uncollected = 0.0
        self.token_1_fees_uncollected = 0.0

def simulate_strategy(price_data, swap_data, strategy_in, liquidity_in_0, liquidity_in_1, fee_tier, decimals_0, decimals_1, token_0_usd_data=None):
    strategy_results = []
    if token_0_usd_data is not None:
        token_0_usd_data = token_0_usd_data.sort_index()
        price_data_with_usd = pd.merge_asof(price_data.to_frame(name='price'), token_0_usd_data[['quotePrice']],
                                            left_index=True, right_index=True, direction='backward')
        price_data_with_usd = price_data_with_usd.rename(columns={'quotePrice': 'price_0_usd'})
    else:
        price_data_with_usd = price_data.to_frame(name='price')
        price_data_with_usd['price_0_usd'] = None

    for i in range(len(price_data)):
        price_0_usd = price_data_with_usd.iloc[i]['price_0_usd'] if 'price_0_usd' in price_data_with_usd.columns else None
        if i == 0:
            strategy_results.append(StrategyObservation(price_data.index[i], price_data[i], strategy_in,
                                                        liquidity_in_0, liquidity_in_1, fee_tier, decimals_0, decimals_1,
                                                        price_0_usd=price_0_usd))
        else:
            relevant_swaps = swap_data[price_data.index[i-1]:price_data.index[i]]
            strategy_results.append(StrategyObservation(price_data.index[i], price_data[i], strategy_in,
                                                        strategy_results[i-1].liquidity_in_0,
                                                        strategy_results[i-1].liquidity_in_1,
                                                        strategy_results[i-1].fee_tier,
                                                        strategy_results[i-1].decimals_0,
                                                        strategy_results[i-1].decimals_1,
                                                        strategy_results[i-1].token_0_left_over,
                                                        strategy_results[i-1].token_1_left_over,
                                                        strategy_results[i-1].token_0_fees_uncollected,
                                                        strategy_results[i-1].token_1_fees_uncollected,
                                                        strategy_results[i-1].liquidity_ranges,
                                                        strategy_results[i-1].strategy_info,
                                                        relevant_swaps, price_0_usd=price_0_usd))
    return strategy_results

def generate_simulation_series(simulations, strategy_in, token_0_usd_data=None):
    data_strategy = pd.DataFrame([strategy_in.dict_components(i) for i in simulations])
    data_strategy = data_strategy.set_index('time', drop=False)
    data_strategy = data_strategy.sort_index()
    
    token_0_initial = simulations[0].liquidity_ranges[0]['token_0'] + simulations[0].liquidity_ranges[1]['token_0'] + simulations[0].token_0_left_over
    token_1_initial = simulations[0].liquidity_ranges[0]['token_1'] + simulations[0].liquidity_ranges[1]['token_1'] + simulations[0].token_1_left_over
    
    if token_0_usd_data is None:
        data_strategy['value_position_usd'] = data_strategy['value_position_in_token_0']
        data_strategy['base_position_value_usd'] = data_strategy['base_position_value_in_token_0']
        data_strategy['limit_position_value_usd'] = data_strategy['limit_position_value_in_token_0']
        data_strategy['cum_fees_usd'] = data_strategy['token_0_fees'].cumsum() + (data_strategy['token_1_fees'] / data_strategy['price']).cumsum()
        data_strategy['token_0_hold_usd'] = token_0_initial
        data_strategy['token_1_hold_usd'] = token_1_initial / data_strategy['price']
        data_strategy['value_hold_usd'] = data_strategy['token_0_hold_usd'] + data_strategy['token_1_hold_usd']
        data_return = data_strategy
    else:
        token_0_usd_data['price_0_usd'] = token_0_usd_data['quotePrice']
        token_0_usd_data['time_pd'] = token_0_usd_data.index
        token_0_usd_data = token_0_usd_data.set_index('time_pd').sort_index()
        
        data_strategy['time_pd'] = pd.to_datetime(data_strategy['time'], utc=True)
        data_strategy = data_strategy.set_index('time_pd').sort_index()
        data_return = pd.merge_asof(data_strategy, token_0_usd_data['price_0_usd'], on='time_pd', direction='backward', allow_exact_matches=True)
        
        data_return['value_position_usd'] = data_return['value_position_in_token_0'] * data_return['price_0_usd']
        data_return['base_position_value_usd'] = data_return['base_position_value_in_token_0'] * data_return['price_0_usd']
        data_return['limit_position_value_usd'] = data_return['limit_position_value_in_token_0'] * data_return['price_0_usd']
        data_return['cum_fees_0'] = data_return['token_0_fees'].cumsum() + (data_return['token_1_fees'] / data_return['price']).cumsum()
        data_return['cum_fees_usd'] = data_return['cum_fees_0'] * data_return['price_0_usd']
        data_return['token_0_hold_usd'] = token_0_initial * data_return['price_0_usd']
        data_return['token_1_hold_usd'] = token_1_initial * data_return['price_0_usd'] / data_return['price']
        data_return['value_hold_usd'] = data_return['token_0_hold_usd'] + data_return['token_1_hold_usd']
        
    return data_return



########################################################
# Calculates % returns over a minutes frequency
########################################################

def fill_time(data):
    price_range               = pd.DataFrame({'time_pd': pd.date_range(data.index.min(),data.index.max(),freq='1 min',tz='UTC')})
    price_range               = price_range.set_index('time_pd')
    new_data                  = price_range.merge(data,left_index=True,right_index=True,how='left').ffill()    
    return new_data

def aggregate_price_data(data, frequency):
    if frequency == 'M':
        resample_option = '1 min'
    elif frequency == 'H':
        resample_option = '1H'
    elif frequency == 'D':
        resample_option = '1D'
    
    data_floored_min = data.copy()
    data_floored_min.index = data_floored_min.index.floor('Min')    
    price_range = pd.DataFrame({
        'time_pd': pd.date_range(
            data_floored_min.index.min(),
            data_floored_min.index.max(),
            freq='1 min',
            tz='UTC'
        )
    })
    price_range = price_range.set_index('time_pd')
    new_data = price_range.merge(data_floored_min, left_index=True, right_index=True, how='left')
    new_data['quotePrice'] = new_data['quotePrice'].ffill()
    
    # Ensure the index is a DatetimeIndex before resampling
    new_data.index = pd.to_datetime(new_data.index, utc=True)
    
    price_data_aggregated = new_data.resample(resample_option).last().copy()
    price_data_aggregated['price_return'] = price_data_aggregated['quotePrice'].pct_change()
    return price_data_aggregated

def aggregate_swap_data(data, frequency):
    if frequency == 'M':
        resample_option = '1 min'
    elif frequency == 'H':
        resample_option = '1H'
    elif frequency == 'D':
        resample_option = '1D'
    
    # Define aggregation methods for all columns
    agg_dict = {}
    for col in data.columns:
        if col in ['amount0', 'amount1', 'amount0_adj', 'amount1_adj']:
            agg_dict[col] = np.sum  # Sum swap amounts
        elif col in ['virtual_liquidity', 'virtual_liquidity_adj']:
            agg_dict[col] = np.median  # Median for liquidity
        elif col in ['tick_swap', 'token0', 'token1', 'sqrtPriceX96', 'pool', 'fee_tier', 'quotePrice']:
            agg_dict[col] = 'last'  # Last for state/categorical columns
        else:
            agg_dict[col] = 'last'  # Default to last for unknown columns
    
    # Resample and aggregate
    swap_data_tmp = data.resample(resample_option).agg(agg_dict)
    
    # Derive token_in based on net swap direction
    swap_data_tmp['token_in'] = np.where(swap_data_tmp['amount0_adj'] > 0, 'token1', 'token0')
    
    # Forward-fill to handle missing values
    return swap_data_tmp.ffill()

def analyze_strategy(data_usd,frequency = 'M'):
    
    if   frequency == 'M':
            annualization_factor = 365*24*60
    elif frequency == 'H':
            annualization_factor = 365*24
    elif frequency == 'D':
            annualization_factor = 365

    days_strategy           = (data_usd['time'].max()-data_usd['time'].min()).days    
    strategy_last_obs       = data_usd.tail(1)
    strategy_last_obs       = strategy_last_obs.reset_index(drop=True)
    initial_position_value  = data_usd.iloc[0]['value_hold_usd']
    net_apr                 = float((strategy_last_obs['value_position_usd']/initial_position_value - 1) * 365 / days_strategy)
    

    summary_strat = {
                        'days_strategy'        : days_strategy,
                        'gross_fee_apr'        : float((strategy_last_obs['cum_fees_usd']/initial_position_value) * 365 / days_strategy),
                        'gross_fee_return'     : float(strategy_last_obs['cum_fees_usd']/initial_position_value),
                        'net_apr'              : net_apr,
                        'net_return'           : float(strategy_last_obs['value_position_usd']/initial_position_value  - 1),
                        'rebalances'           : data_usd['reset_point'].sum(),
                        'compounds'            : data_usd['compound_point'].sum(),
                        'max_drawdown'         : ( data_usd['value_position_usd'].max() - data_usd['value_position_usd'].min() ) / data_usd['value_position_usd'].max(),
                        'volatility'           : ((data_usd['value_position_usd'].pct_change().var())**(0.5)) * ((annualization_factor)**(0.5)),
                        'sharpe_ratio'         : float(net_apr / (((data_usd['value_position_usd'].pct_change().var())**(0.5)) * ((annualization_factor)**(0.5)))),
                        'impermanent_loss'     : ((strategy_last_obs['value_position_usd'] - strategy_last_obs['value_hold_usd']) / strategy_last_obs['value_hold_usd'])[0],
                        'mean_base_position'   : (data_usd['base_position_value_in_token_0']/ \
                                                  (data_usd['base_position_value_in_token_0']+data_usd['limit_position_value_in_token_0']+data_usd['value_left_over_in_token_0'])).mean(),        
                        'median_base_position' : (data_usd['base_position_value_in_token_0']/ \
                                                  (data_usd['base_position_value_in_token_0']+data_usd['limit_position_value_in_token_0']+data_usd['value_left_over_in_token_0'])).median(),
                        'mean_base_width'      : ((data_usd['base_range_upper']-data_usd['base_range_lower'])/data_usd['price_at_reset']).mean(),
                        'median_base_width'    : ((data_usd['base_range_upper']-data_usd['base_range_lower'])/data_usd['price_at_reset']).median(),        
                        'final_value'          : data_usd['value_position_usd'].iloc[-1]
                    }
    
    return summary_strat


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_strategy(data_strategy, y_axis_label, base_color='#ff0000', flip_price_axis=False):
    CHART_SIZE = (10, 4)
    
    if flip_price_axis:
        data_strategy_here = data_strategy.copy()
        data_strategy_here['base_range_lower'] = 1 / data_strategy_here['base_range_lower']
        data_strategy_here['base_range_upper'] = 1 / data_strategy_here['base_range_upper']
        data_strategy_here['reset_range_lower'] = 1 / data_strategy_here['reset_range_lower']
        data_strategy_here['reset_range_upper'] = 1 / data_strategy_here['reset_range_upper']
        data_strategy_here['price'] = 1 / data_strategy_here['price']
    else:
        data_strategy_here = data_strategy.copy()
    
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    
    # Base position filled area
    ax.fill_between(
        data_strategy_here['time_pd'], 
        data_strategy_here['base_range_lower'],
        data_strategy_here['base_range_upper'],
        alpha=0.3,
        color=base_color,
        label='Base Position'
    )
    
    # Base position bounds
    ax.plot(
        data_strategy_here['time_pd'], 
        data_strategy_here['base_range_lower'],
        color=base_color,
        linewidth=1
    )
    ax.plot(
        data_strategy_here['time_pd'], 
        data_strategy_here['base_range_upper'],
        color=base_color,
        linewidth=1
    )
    
    # Reset range bounds
    ax.plot(
        data_strategy_here['time_pd'], 
        data_strategy_here['reset_range_lower'],
        color='black',
        linewidth=2,
        linestyle='--',
        label='Strategy Reset Bound'
    )
    ax.plot(
        data_strategy_here['time_pd'], 
        data_strategy_here['reset_range_upper'],
        color='black',
        linewidth=2,
        linestyle='--'
    )
    
    # Price
    ax.plot(
        data_strategy_here['time_pd'], 
        data_strategy_here['price'],
        color='black',
        linewidth=2,
        label='Price'
    )
    
    # Formatting
    ax.set_title('Strategy Simulation', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_position_value(data_strategy):
    CHART_SIZE = (10, 4)

    # Validate required columns
    required_columns = ['time_pd', 'value_position_usd', 'value_hold_usd']
    if not all(col in data_strategy.columns for col in required_columns):
        print("Error: Missing required columns in data_strategy")
        return None

    fig, ax = plt.subplots(figsize=CHART_SIZE)
    
    ax.plot(
        data_strategy['time_pd'], 
        data_strategy['value_position_usd'],
        color='red',
        linewidth=2,
        label='Value of LP Position'
    )

    ax.plot(
        data_strategy['time_pd'], 
        data_strategy['value_hold_usd'],
        color='blue',
        linewidth=2,
        label='Value of Holding'
    )

    # Formatting
    ax.set_title('LP Position vs. Holding', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Position Value USD', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_position_return_decomposition(data_strategy):
    CHART_SIZE = (10, 4)

    # Validate required columns
    required_columns = ['time_pd', 'cum_fees_usd', 'value_hold_usd', 'value_position_usd']
    if not all(col in data_strategy.columns for col in required_columns):
        print("Error: Missing required columns in data_strategy")
        return None

    if data_strategy.empty:
        print("Error: data_strategy is empty")
        return None

    INITIAL_POSITION_VALUE = data_strategy.iloc[0]['value_position_usd']

    fig, ax = plt.subplots(figsize=CHART_SIZE)
    
    ax.plot(
        data_strategy['time_pd'], 
        data_strategy['cum_fees_usd'] / INITIAL_POSITION_VALUE,
        color='blue',
        linewidth=2,
        label='Accumulated Fees'
    )

    ax.plot(
        data_strategy['time_pd'], 
        (data_strategy['value_hold_usd'] - data_strategy['value_position_usd']) / INITIAL_POSITION_VALUE,
        color='black',
        linewidth=2,
        label='Value Hold - Position'
    )
    
    ax.plot(
        data_strategy['time_pd'], 
        (data_strategy['value_hold_usd'] / INITIAL_POSITION_VALUE) - 1,
        color='green',
        linewidth=2,
        label='Value Hold'
    )

    ax.plot(
        data_strategy['time_pd'], 
        (data_strategy['value_position_usd'] / INITIAL_POSITION_VALUE) - 1,
        color='#ff0000',
        linewidth=2,
        label='Net Position Value'
    )

    # Formatting
    ax.set_title('Position Value Decomposition', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Position returns %', fontsize=12)
    ax.legend(loc='best', title='Component')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return fig