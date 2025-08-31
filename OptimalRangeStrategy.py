import pandas as pd
import numpy as np
import math
import UNI_v3_funcs
import copy

class OptimalRangeStrategy:
    def __init__(self, volatility_data, rebalance_days=3, alpha_range=(1.02, 2.0), alpha_steps=50):
        """Initialize the optimal range strategy.
        
        Args:
            volatility_data: DataFrame with volatility and correlation data
            rebalance_days: Days between rebalancing (default: 3)
            alpha_range: Tuple of (min_alpha, max_alpha) for range width
            alpha_steps: Number of α values to test for optimization
        """
        self.volatility_data = volatility_data
        self.rebalance_days = rebalance_days
        self.alpha_range = alpha_range
        self.alpha_steps = alpha_steps
        
        # Generate α values to test
        self.alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_steps)
        
        # Track last rebalancing time
        self.last_rebalance_time = None
        
        print(f"Optimal Range Strategy initialized:")
        print(f"  Rebalancing frequency: Every {rebalance_days} days")
        print(f"  Alpha range: {alpha_range[0]} to {alpha_range[1]}")
        print(f"  Alpha steps: {alpha_steps}")

    def equation_54_objective(self, alpha, volatility, correlation, T_days):
        """
        Calculate Equation 54 objective function from thesis
        This is the function we want to maximize
        
        Args:
            alpha: Range width parameter
            volatility: Cross price volatility
            correlation: Correlation between tokens
            T_days: Time horizon in days
        
        Returns:
            Objective function value
        """
        # Convert days to hours for consistency with hourly data
        T_hours = T_days * 24
        
        # This is a placeholder for Equation 54
        # You'll need to replace this with the actual formula from your thesis
        
        # For now, let's use a reasonable approximation based on typical LP strategies:
        # Objective = Expected Return - Risk Penalty
        
        # Expected return component (increases with alpha)
        expected_return = np.log(alpha) / T_hours
        
        # Risk penalty component (decreases with alpha, increases with volatility)
        risk_penalty = (volatility ** 2) / (2 * alpha * T_hours)
        
        # Correlation adjustment
        correlation_penalty = (1 - abs(correlation)) * volatility / (alpha * T_hours)
        
        # Total objective function
        objective = expected_return - risk_penalty - correlation_penalty
        
        return objective

    def find_optimal_alpha(self, volatility, correlation, T_days):
        """
        Find the optimal α that maximizes Equation 54 objective
        
        Args:
            volatility: Current cross price volatility
            correlation: Current correlation
            T_days: Time horizon
            
        Returns:
            optimal_alpha: The α value that maximizes the objective
        """
        objectives = []
        
        for alpha in self.alphas:
            obj_value = self.equation_54_objective(alpha, volatility, correlation, T_days)
            objectives.append(obj_value)
        
        objectives = np.array(objectives)
        
        # Find optimal α
        max_idx = np.argmax(objectives)
        optimal_alpha = self.alphas[max_idx]
        
        return optimal_alpha

    def check_strategy(self, current_strat_obs):
        """Check if the current price is out of range or if rebalancing is due.
        
        Args:
            current_strat_obs: Current StrategyObservation object.
        
        Returns:
            tuple: (liquidity_ranges, strategy_info)
        """
        current_time = current_strat_obs.time
        
        # Check if rebalancing is due (every 3 days)
        if (self.last_rebalance_time is None or 
            (current_time - self.last_rebalance_time).days >= self.rebalance_days):
            print(f"Rebalancing due: {self.rebalance_days} days elapsed")
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'scheduled_rebalance'
            current_strat_obs.remove_liquidity()
            liquidity_ranges, strategy_info = self.set_liquidity_ranges(current_strat_obs)
            return liquidity_ranges, strategy_info
        
        # Check if price is out of range
        lower_tick = current_strat_obs.liquidity_ranges[0]['lower_bin_tick']
        upper_tick = current_strat_obs.liquidity_ranges[0]['upper_bin_tick']

        if (current_strat_obs.price_tick_current < lower_tick or 
            current_strat_obs.price_tick_current >= upper_tick):
            print(f"Price out of range, rebalancing")
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'out_of_range'
            current_strat_obs.remove_liquidity()
            liquidity_ranges, strategy_info = self.set_liquidity_ranges(current_strat_obs)
            return liquidity_ranges, strategy_info
        else:
            return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info

    def set_liquidity_ranges(self, current_strat_obs):
        """Set the liquidity range based on optimal α calculation using current market conditions.
        
        Args:
            current_strat_obs: Current StrategyObservation object with price_0_usd.
        
        Returns:
            tuple: (liquidity_ranges, strategy_info)
        """
        # Get current market conditions from volatility data
        current_time = current_strat_obs.time
        
        # Find closest timestamp in volatility data
        if self.volatility_data is not None and not self.volatility_data.empty:
            # Find the closest timestamp
            time_diff = abs(self.volatility_data.index - current_time)
            closest_idx = time_diff.argmin()
            current_market_data = self.volatility_data.iloc[closest_idx]
            
            # Get current volatility and correlation
            current_volatility = current_market_data['cross_price_volatility']
            current_correlation = current_market_data['correlation']
            
            # Calculate optimal α
            optimal_alpha = self.find_optimal_alpha(
                current_volatility, current_correlation, self.rebalance_days
            )
            
            print(f"Optimal α calculation:")
            print(f"  Time: {current_time}")
            print(f"  Volatility: {current_volatility:.2%}")
            print(f"  Correlation: {current_correlation:.3f}")
            print(f"  Optimal α: {optimal_alpha:.3f}")
            
            # Use optimal α as width
            self.width = optimal_alpha - 1  # Convert to percentage width
        else:
            # Fallback to default width if no volatility data
            self.width = 0.1  # 10% default
            print(f"Warning: No volatility data available, using default width: {self.width:.1%}")
        
        # Calculate lower and upper prices using optimal width
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
        strategy_info['optimal_alpha'] = self.width + 1  # Store the optimal α used
        strategy_info['volatility_at_reset'] = current_volatility if 'current_volatility' in locals() else None
        strategy_info['correlation_at_reset'] = current_correlation if 'current_correlation' in locals() else None

        # Update last rebalancing time
        self.last_rebalance_time = current_strat_obs.time

        return liquidity_ranges, strategy_info

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

        # Add optimal strategy information
        if 'optimal_alpha' in strategy_observation.strategy_info:
            this_data['optimal_alpha'] = strategy_observation.strategy_info['optimal_alpha']
            this_data['volatility_at_reset'] = strategy_observation.strategy_info['volatility_at_reset']
            this_data['correlation_at_reset'] = strategy_observation.strategy_info['correlation_at_reset']

        total_token_0 = sum([r['token_0'] for r in strategy_observation.liquidity_ranges])
        total_token_1 = sum([r['token_1'] for r in strategy_observation.liquidity_ranges])
        this_data['token_0_allocated'] = total_token_0
        this_data['token_1_allocated'] = total_token_1
        this_data['token_0_total'] = (total_token_0 + strategy_observation.token_0_left_over + 
                                      strategy_observation.token_0_fees_uncollected)
        this_data['token_1_total'] = (total_token_1 + strategy_observation.token_1_left_over + 
                                      strategy_observation.token_0_fees_uncollected)

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