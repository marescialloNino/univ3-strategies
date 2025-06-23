import pandas as pd
import numpy as np
import math
import UNI_v3_funcs
import copy

class SimpleRangeStrategy:
    def __init__(self, width):
        """Initialize the strategy with a range width parameter.
        
        Args:
            width (float): The percentage width above and below the current price (e.g., 0.05 for 5%).
        """
        self.width = width

    def set_liquidity_ranges(self, current_strat_obs, model_forecast=None):
        """Set the liquidity range based on the current price and width, adjusting tokens to 50/50 USD value.
        
        Args:
            current_strat_obs: Current StrategyObservation object with price_0_usd.
            model_forecast: Not used in this strategy, included for compatibility.
        
        Returns:
            tuple: (liquidity_ranges, strategy_info)
        """
        # Calculate lower and upper prices
        lower_price = current_strat_obs.price / (1 + self.width)
        upper_price = current_strat_obs.price * (1 + self.width)

        # Calculate corresponding ticks
        TICK_A_PRE = math.log(current_strat_obs.decimal_adjustment * lower_price, 1.0001)
        TICK_A = int(round(TICK_A_PRE / current_strat_obs.tickSpacing)) * current_strat_obs.tickSpacing
        TICK_B_PRE = math.log(current_strat_obs.decimal_adjustment * upper_price, 1.0001)
        TICK_B = int(round(TICK_B_PRE / current_strat_obs.tickSpacing)) * current_strat_obs.tickSpacing

        # Ensure TICK_A < TICK_B
        if TICK_A >= TICK_B:
            TICK_A = TICK_B - current_strat_obs.tickSpacing

        # Adjust tokens to 50/50 USD value split if price_0_usd is available
        if hasattr(current_strat_obs, 'price_0_usd') and current_strat_obs.price_0_usd is not None:
            price_token0_usd = current_strat_obs.price_0_usd
            price_token1_usd = price_token0_usd / current_strat_obs.price  # price = token_1 / token_0
            value_0 = current_strat_obs.liquidity_in_0 * price_token0_usd
            value_1 = current_strat_obs.liquidity_in_1 * price_token1_usd
            total_value = value_0 + value_1
            target_value = total_value / 2
            delta_0 = (value_0 - value_1) / (2 * price_token0_usd)  # Amount of token_0 to swap
            adjusted_0 = current_strat_obs.liquidity_in_0 - delta_0
            adjusted_1 = current_strat_obs.liquidity_in_1 + (delta_0 * current_strat_obs.price)
            # Ensure non-negative amounts
            current_strat_obs.liquidity_in_0 = max(adjusted_0, 0)
            current_strat_obs.liquidity_in_1 = max(adjusted_1, 0)
        else:
            print("Warning: price_0_usd not available, skipping 50/50 adjustment")

        # Calculate liquidity with adjusted amounts
        liquidity_placed = int(UNI_v3_funcs.get_liquidity(
            current_strat_obs.price_tick_current, TICK_A, TICK_B,
            current_strat_obs.liquidity_in_0, current_strat_obs.liquidity_in_1,
            current_strat_obs.decimals_0, current_strat_obs.decimals_1
        ))

        # Calculate amounts placed
        amount_0_placed, amount_1_placed = UNI_v3_funcs.get_amounts(
            current_strat_obs.price_tick_current, TICK_A, TICK_B,
            liquidity_placed, current_strat_obs.decimals_0, current_strat_obs.decimals_1
        )

        # Update leftover tokens
        current_strat_obs.token_0_left_over = max([current_strat_obs.liquidity_in_0 - amount_0_placed, 0.0])
        current_strat_obs.token_1_left_over = max([current_strat_obs.liquidity_in_1 - amount_1_placed, 0.0])
        current_strat_obs.liquidity_in_0 = 0.0
        current_strat_obs.liquidity_in_1 = 0.0

        # Set actual prices for plotting
        lower_bin_price = (1.0001 ** TICK_A) / current_strat_obs.decimal_adjustment
        upper_bin_price = (1.0001 ** TICK_B) / current_strat_obs.decimal_adjustment

        # Define base position
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

        # Dummy limit position with zero liquidity
        dummy_liq_range = {
            'price': current_strat_obs.price,
            'lower_bin_tick': current_strat_obs.price_tick,
            'upper_bin_tick': current_strat_obs.price_tick,
            'lower_bin_price': current_strat_obs.price,
            'upper_bin_price': current_strat_obs.price,
            'time': current_strat_obs.time,
            'token_0': 0.0,
            'token_1': 0.0,
            'position_liquidity': 0,
            'reset_time': current_strat_obs.time
        }

        liquidity_ranges = [base_liq_range, dummy_liq_range]

        # Set strategy_info
        if current_strat_obs.strategy_info is None:
            strategy_info = {}
        else:
            strategy_info = copy.deepcopy(current_strat_obs.strategy_info)
        strategy_info['reset_range_lower'] = lower_bin_price
        strategy_info['reset_range_upper'] = upper_bin_price

        return liquidity_ranges, strategy_info

    def check_strategy(self, current_strat_obs):
        """Check if the current price is out of range and rebalance if necessary.
        
        Args:
            current_strat_obs: Current StrategyObservation object.
        
        Returns:
            tuple: (liquidity_ranges, strategy_info)
        """
        lower_tick = current_strat_obs.liquidity_ranges[0]['lower_bin_tick']
        upper_tick = current_strat_obs.liquidity_ranges[0]['upper_bin_tick']

        # Rebalance if price is out of range
        if (current_strat_obs.price_tick_current < lower_tick or 
            current_strat_obs.price_tick_current >= upper_tick):
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'out_of_range'
            current_strat_obs.remove_liquidity()
            liquidity_ranges, strategy_info = self.set_liquidity_ranges(current_strat_obs)
            return liquidity_ranges, strategy_info
        else:
            return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info

    def dict_components(self, strategy_observation):
        """Extract strategy data for analysis and plotting.
        
        Args:
            strategy_observation: StrategyObservation object.
        
        Returns:
            dict: Data components of the strategy.
        """
        this_data = {
            'time': strategy_observation.time,
            'price': strategy_observation.price,
            'reset_point': strategy_observation.reset_point,
            'compound_point': False,  # No compounding in this strategy
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

