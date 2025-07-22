import pandas as pd
import numpy as np
import math
from collections import deque
import UNI_v3_funcs
import copy
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

class SimpleResetStrategy:
    def __init__(self, model_data, alpha_param=0.95, tau_param=0.99, base_width=0.05, vol_window=24, 
                 vol_threshold_very_low=0.01, vol_threshold_low=0.02, 
                 vol_threshold_high=0.05, vol_threshold_very_high=0.1):
        self.alpha_param = alpha_param
        self.tau_param = tau_param
        self.base_width = base_width
        self.vol_window = vol_window
        self.vol_threshold_very_low = vol_threshold_very_low
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_very_high = vol_threshold_very_high
        self.alpha_ewma = 2 / (24 + 1.0)
        self.ewma = None
        self.price_history = deque(maxlen=vol_window)
        ecdf = ECDF(model_data['price_return'].to_numpy())
        self.inverse_ecdf = monotone_fn_inverter(
            ecdf, np.linspace(model_data['price_return'].min(), model_data['price_return'].max(), 1000), vectorized=False
        )
        
        # Width levels
        self.width_levels = {
            'very_low': max(base_width * 0.6, 0.01),  # -40%
            'low': max(base_width * 0.8, 0.01),       # -20%
            'normal': base_width,
            'high': min(base_width * 1.2, 0.2),       # +20%
            'very_high': min(base_width * 1.4, 0.2)   # +40%
        }
        
        # Initial vol from history
        if len(model_data) >= 2:
            log_returns = np.diff(np.log(model_data['quotePrice']))
            initial_vol = np.std(log_returns) if len(log_returns) > 0 else 0.0
            self.dynamic_width = self.get_width_for_vol(initial_vol)
        else:
            self.dynamic_width = base_width  # Fallback

    def get_width_for_vol(self, vol):
        if vol < self.vol_threshold_very_low:
            return self.width_levels['very_low']
        elif vol < self.vol_threshold_low:
            return self.width_levels['low']
        elif vol < self.vol_threshold_high:
            return self.width_levels['normal']
        elif vol < self.vol_threshold_very_high:
            return self.width_levels['high']
        else:
            return self.width_levels['very_high']

    def set_liquidity_ranges(self, current_strat_obs, model_forecast=None):
        if current_strat_obs.strategy_info is None:
            strategy_info = {}
        else:
            strategy_info = copy.deepcopy(current_strat_obs.strategy_info)

        snapshot_price = current_strat_obs.price

        # Reset range with symmetry, centered on snapshot
        lower_prob_tau = (1 - self.tau_param) / 2
        upper_prob_tau = 1 - lower_prob_tau
        upper_tau = self.inverse_ecdf(upper_prob_tau)
        try:
            lower_tau = self.inverse_ecdf(lower_prob_tau)
        except ValueError as e:
            if "below" in str(e):
                lower_tau = -upper_tau
            else:
                raise
        strategy_info['reset_range_lower'] = snapshot_price * (1 + lower_tau)
        strategy_info['reset_range_upper'] = snapshot_price * (1 + upper_tau)

        # Base range using dynamic_width, centered on snapshot
        base_range_lower = snapshot_price * (1 - self.dynamic_width)
        base_range_upper = snapshot_price * (1 + self.dynamic_width)
        strategy_info['base_range_lower_price'] = base_range_lower
        strategy_info['base_range_upper_price'] = base_range_upper

        # Ticks and liquidity placement
        TICK_A_PRE = math.log(current_strat_obs.decimal_adjustment * base_range_lower, 1.0001)
        TICK_A = int(round(TICK_A_PRE / current_strat_obs.tickSpacing)) * current_strat_obs.tickSpacing
        TICK_B_PRE = math.log(current_strat_obs.decimal_adjustment * base_range_upper, 1.0001)
        TICK_B = int(round(TICK_B_PRE / current_strat_obs.tickSpacing)) * current_strat_obs.tickSpacing
        if TICK_A >= TICK_B:
            TICK_B = TICK_A + current_strat_obs.tickSpacing
        liquidity_placed = int(UNI_v3_funcs.get_liquidity(
            current_strat_obs.price_tick_current, TICK_A, TICK_B,
            current_strat_obs.liquidity_in_0, current_strat_obs.liquidity_in_1,
            current_strat_obs.decimals_0, current_strat_obs.decimals_1
        ))
        amount_0_placed, amount_1_placed = UNI_v3_funcs.get_amounts(
            current_strat_obs.price_tick_current, TICK_A, TICK_B,
            liquidity_placed, current_strat_obs.decimals_0, current_strat_obs.decimals_1
        )
        current_strat_obs.token_0_left_over = max([current_strat_obs.liquidity_in_0 - amount_0_placed, 0.0])
        current_strat_obs.token_1_left_over = max([current_strat_obs.liquidity_in_1 - amount_1_placed, 0.0])
        current_strat_obs.liquidity_in_0 = 0.0
        current_strat_obs.liquidity_in_1 = 0.0
        lower_bin_price = (1.0001 ** TICK_A) / current_strat_obs.decimal_adjustment
        upper_bin_price = (1.0001 ** TICK_B) / current_strat_obs.decimal_adjustment
        base_liq_range = {
            'price': current_strat_obs.price,
            'lower_bin_tick': TICK_A,
            'upper_bin_tick': TICK_B,
            'lower_bin_price': lower_bin_price,
            'upper_bin_price': upper_bin_price,
            'time': current_strat_obs.time,
            'token_0': amount_0_placed,
            'token_1': amount_1_placed,
            'position_liquidity': liquidity_placed,
            'reset_time': current_strat_obs.time
        }
        dummy_liq_range = {
            'price': current_strat_obs.price,
            'lower_bin_tick': current_strat_obs.price_tick_current,
            'upper_bin_tick': current_strat_obs.price_tick_current,
            'lower_bin_price': current_strat_obs.price,
            'upper_bin_price': current_strat_obs.price,
            'time': current_strat_obs.time,
            'token_0': 0.0,
            'token_1': 0.0,
            'position_liquidity': 0,
            'reset_time': current_strat_obs.time
        }

        # Reset EWMA and history
        self.ewma = current_strat_obs.price
        self.price_history.clear()
        self.price_history.append(current_strat_obs.price)

        return [base_liq_range, dummy_liq_range], strategy_info

    def check_strategy(self, current_strat_obs):
        # Update history and EWMA
        self.price_history.append(current_strat_obs.price)
        if self.ewma is None:
            self.ewma = current_strat_obs.price
        else:
            self.ewma = self.alpha_ewma * current_strat_obs.price + (1 - self.alpha_ewma) * self.ewma

        # Compute vol if enough history
        if len(self.price_history) >= 2:
            prices_arr = np.array(self.price_history)
            log_returns = np.diff(np.log(prices_arr))
            if len(log_returns) >= 1:
                current_vol = np.std(log_returns)
                self.dynamic_width = self.get_width_for_vol(current_vol)

        # Get ranges from strategy_info
        reset_lower = current_strat_obs.strategy_info.get('reset_range_lower', current_strat_obs.price)
        reset_upper = current_strat_obs.strategy_info.get('reset_range_upper', current_strat_obs.price)
        base_lower = current_strat_obs.strategy_info.get('base_range_lower_price', current_strat_obs.price)
        base_upper = current_strat_obs.strategy_info.get('base_range_upper_price', current_strat_obs.price)

        # Rebalance conditions
        ewma_out_base = (self.ewma < base_lower) or (self.ewma > base_upper)
        spot_out_reset = (current_strat_obs.price < reset_lower) or (current_strat_obs.price > reset_upper)

        if ewma_out_base or spot_out_reset:
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'ewma_out_base' if ewma_out_base else 'spot_out_reset'
            current_strat_obs.remove_liquidity()
            return self.set_liquidity_ranges(current_strat_obs)
        else:
            return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info

    def dict_components(self, strategy_observation):
        this_data = {
            'time': strategy_observation.time,
            'price': strategy_observation.price,
            'reset_point': strategy_observation.reset_point,
            'compound_point': False,
            'reset_reason': strategy_observation.reset_reason,
            'base_range_lower': strategy_observation.liquidity_ranges[0]['lower_bin_price'],
            'base_range_upper': strategy_observation.liquidity_ranges[0]['upper_bin_price'],
            'limit_range_lower': strategy_observation.liquidity_ranges[1]['lower_bin_price'],
            'limit_range_upper': strategy_observation.liquidity_ranges[1]['upper_bin_price'],
            'reset_range_lower': strategy_observation.strategy_info['reset_range_lower'],
            'reset_range_upper': strategy_observation.strategy_info['reset_range_upper'],
            'price_at_reset': strategy_observation.liquidity_ranges[0]['price'],
            'token_0_fees': strategy_observation.token_0_fees,
            'token_1_fees': strategy_observation.token_1_fees,
            'token_0_fees_uncollected': strategy_observation.token_0_fees_uncollected,
            'token_1_fees_uncollected': strategy_observation.token_1_fees_uncollected,
            'token_0_left_over': strategy_observation.token_0_left_over,
            'token_1_left_over': strategy_observation.token_1_left_over,
        }

        total_token_0 = sum([r['token_0'] for r in strategy_observation.liquidity_ranges])
        total_token_1 = sum([r['token_1'] for r in strategy_observation.liquidity_ranges])
        this_data['token_0_allocated'] = total_token_0
        this_data['token_1_allocated'] = total_token_1
        this_data['token_0_total'] = (total_token_0 + strategy_observation.token_0_left_over + 
                                      strategy_observation.token_0_fees_uncollected)
        this_data['token_1_total'] = (total_token_1 + strategy_observation.token_1_left_over + 
                                      strategy_observation.token_1_fees_uncollected)

        this_data['value_position_in_token_0'] = (this_data['token_0_total'] + 
                                                  this_data['token_1_total'] / this_data['price'])
        this_data['value_allocated_in_token_0'] = (this_data['token_0_allocated'] + 
                                                   this_data['token_1_allocated'] / this_data['price'])
        this_data['value_left_over_in_token_0'] = (this_data['token_0_left_over'] + 
                                                   this_data['token_1_left_over'] / this_data['price'])
        this_data['base_position_value_in_token_0'] = (strategy_observation.liquidity_ranges[0]['token_0'] + 
                                                       strategy_observation.liquidity_ranges[0]['token_1'] / this_data['price'])
        this_data['limit_position_value_in_token_0'] = (strategy_observation.liquidity_ranges[1]['token_0'] + 
                                                        strategy_observation.liquidity_ranges[1]['token_1'] / this_data['price'])

        return this_data