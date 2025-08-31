import pandas as pd
import numpy as np
import math
import UNI_v3_funcs
import copy


class OptimalRangeStrategy:
    def __init__(self, volatility_data: pd.DataFrame, rebalance_days: int = 3,
                 alpha_range: tuple = (1.02, 2.0), alpha_steps: int = 50,
                 days_in_year: int = 365):
        """Strategy that selects the optimal price range at each rebalance.

        - Rebalances on schedule (every rebalance_days) or when price leaves range.
        - Chooses range width by maximizing Equation 54 objective (placeholder).

        Args:
            volatility_data: DataFrame indexed by time with columns
                'cross_price_volatility' and 'correlation'.
            rebalance_days: Days between scheduled rebalances.
            alpha_range: (min_alpha, max_alpha) search interval for range multiplier.
            alpha_steps: Grid resolution for alpha search.
        """

        self.volatility_data = volatility_data if volatility_data is not None else pd.DataFrame()
        self.rebalance_days = rebalance_days
        self.alpha_range = alpha_range
        self.alpha_steps = alpha_steps
        self.alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_steps)
        self.days_in_year = days_in_year

        self.last_rebalance_time = None
        self.width = 0.10  # default 10% if no vol data/alpha found

    #####################################
    # Rebalance logic: scheduled or out-of-range
    #####################################
    def check_strategy(self, current_strat_obs):
        """Check if a rebalance is due (scheduled or out-of-range).

        Returns (liquidity_ranges, strategy_info)
        """
        current_time = current_strat_obs.time

        # Scheduled rebalance
        if (self.last_rebalance_time is None or
            (current_time - self.last_rebalance_time).days >= self.rebalance_days):
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'scheduled_rebalance'
            current_strat_obs.remove_liquidity()
            liq_range, strategy_info = self.set_liquidity_ranges(current_strat_obs)
            return liq_range, strategy_info

        # Out-of-range rebalance
        lower_tick = current_strat_obs.liquidity_ranges[0]['lower_bin_tick']
        upper_tick = current_strat_obs.liquidity_ranges[0]['upper_bin_tick']
        if (current_strat_obs.price_tick_current < lower_tick or
            current_strat_obs.price_tick_current >= upper_tick):
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'out_of_range'
            current_strat_obs.remove_liquidity()
            liq_range, strategy_info = self.set_liquidity_ranges(current_strat_obs)
            return liq_range, strategy_info

        return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info

    #####################################
    # Range placement with optimal alpha
    #####################################
    def set_liquidity_ranges(self, current_strat_obs, model_forecast=None):
        """Set base (and dummy limit) ranges using optimal alpha from Equation 54.

        - Chooses alpha that maximizes objective with inputs from current vol/corr
        - Places a single base position like SimpleRangeStrategy
        - Adjusts deposit to 50/50 USD if price_0_usd is available
        """
        # Determine current volatility/correlation by nearest timestamp
        current_time = current_strat_obs.time
        current_volatility = None
        current_correlation = None
        if not self.volatility_data.empty:
            # Ensure DateTimeIndex
            vol_df = self.volatility_data.copy()
            if not isinstance(vol_df.index, pd.DatetimeIndex) and 'time' in vol_df.columns:
                vol_df['time'] = pd.to_datetime(vol_df['time'])
                vol_df = vol_df.set_index('time')
            # Find nearest index
            nearest_idx = (vol_df.index.get_indexer([current_time], method='nearest'))
            if len(nearest_idx) > 0 and nearest_idx[0] != -1:
                row = vol_df.iloc[nearest_idx[0]]
                current_volatility = float(row.get('cross_price_volatility', np.nan))
                current_correlation = float(row.get('correlation', np.nan))

        # Choose alpha -> width
        if current_volatility is not None and not np.isnan(current_volatility):
            alpha_opt = self._find_optimal_alpha(current_volatility,
                                                 0.0 if current_correlation is None else current_correlation,
                                                 self.rebalance_days)
            self.width = max(alpha_opt - 1.0, 0.001)
        else:
            # Fallback width if no volatility
            self.width = max(self.width, 0.10)

        # Convert width to price bounds
        lower_price = current_strat_obs.price / (1 + self.width)
        upper_price = current_strat_obs.price * (1 + self.width)

        # Price -> ticks
        TICK_A_PRE = math.log(current_strat_obs.decimal_adjustment * lower_price, 1.0001)
        TICK_A = int(round(TICK_A_PRE / current_strat_obs.tickSpacing)) * current_strat_obs.tickSpacing
        TICK_B_PRE = math.log(current_strat_obs.decimal_adjustment * upper_price, 1.0001)
        TICK_B = int(round(TICK_B_PRE / current_strat_obs.tickSpacing)) * current_strat_obs.tickSpacing
        if TICK_A >= TICK_B:
            TICK_A = TICK_B - current_strat_obs.tickSpacing

        # 50/50 USD adjustment if price feed available
        if hasattr(current_strat_obs, 'price_0_usd') and current_strat_obs.price_0_usd is not None:
            price_token0_usd = current_strat_obs.price_0_usd
            price_token1_usd = price_token0_usd / current_strat_obs.price
            value_0 = current_strat_obs.liquidity_in_0 * price_token0_usd
            value_1 = current_strat_obs.liquidity_in_1 * price_token1_usd
            delta_0 = (value_0 - value_1) / (2 * price_token0_usd)
            adjusted_0 = current_strat_obs.liquidity_in_0 - delta_0
            adjusted_1 = current_strat_obs.liquidity_in_1 + (delta_0 * current_strat_obs.price)
            current_strat_obs.liquidity_in_0 = max(adjusted_0, 0)
            current_strat_obs.liquidity_in_1 = max(adjusted_1, 0)

        # Compute liquidity and placed amounts
        liquidity_placed = int(UNI_v3_funcs.get_liquidity(
            current_strat_obs.price_tick_current, TICK_A, TICK_B,
            current_strat_obs.liquidity_in_0, current_strat_obs.liquidity_in_1,
            current_strat_obs.decimals_0, current_strat_obs.decimals_1
        ))

        amount_0_placed, amount_1_placed = UNI_v3_funcs.get_amounts(
            current_strat_obs.price_tick_current, TICK_A, TICK_B,
            liquidity_placed, current_strat_obs.decimals_0, current_strat_obs.decimals_1
        )

        # Update leftovers
        current_strat_obs.token_0_left_over = max([current_strat_obs.liquidity_in_0 - amount_0_placed, 0.0])
        current_strat_obs.token_1_left_over = max([current_strat_obs.liquidity_in_1 - amount_1_placed, 0.0])
        current_strat_obs.liquidity_in_0 = 0.0
        current_strat_obs.liquidity_in_1 = 0.0

        # Prices for reporting
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

        # Dummy limit position
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

        # strategy_info
        if current_strat_obs.strategy_info is None:
            strategy_info = {}
        else:
            strategy_info = copy.deepcopy(current_strat_obs.strategy_info)

        strategy_info['reset_range_lower'] = lower_bin_price
        strategy_info['reset_range_upper'] = upper_bin_price
        strategy_info['optimal_alpha'] = 1.0 + self.width
        strategy_info['volatility_at_reset'] = current_volatility
        strategy_info['correlation_at_reset'] = current_correlation

        self.last_rebalance_time = current_strat_obs.time
        return liquidity_ranges, strategy_info

    #####################################
    # Data export for analysis
    #####################################
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

    #####################################
    # Equation 54 objective and optimizer
    #####################################
    def _equation_54_objective(self, alpha: float, sigma_daily: float, correlation: float, T_days: int) -> float:
        """Equation 54: expected fraction of time in range over horizon T, divided by alpha.

        Assumptions:
        - Symmetric range around current price p0 with pmax = p0*r, pmin = p0/r, alpha = r^2.
        - sigma_daily is the cross-price volatility per sqrt-day.
        - T_days is the horizon in days.
        - correlation is ignored in this simplified equation (kept for signature compatibility).
        """
        # Enforce domain constraints
        alpha = max(float(alpha), 1.0000001)
        r = max(math.sqrt(alpha), 1.0000001)
        T = max(float(T_days), 1e-12)
        sigma = max(float(sigma_daily), 1e-12)

        ln_r = math.log(r)
        denom = (sigma * sigma) * T
        z = ln_r / (sigma * math.sqrt(2.0 * T))

        # Numerator of Equation 54 (expected time in range over T)
        term1 = (ln_r * ln_r + denom) * math.erf(z)
        term2 = ln_r * (math.sqrt(2.0 * T / math.pi) * sigma * math.exp(-(ln_r * ln_r) / (2.0 * sigma * sigma * T)) - ln_r)
        frac_time_in_range = (term1 + term2) / denom

        # Objective: fraction of time in range per unit alpha (narrower is better)
        return float(frac_time_in_range) / float(alpha)

    def _find_optimal_alpha(self, volatility: float, correlation: float, T_days: int) -> float:
        # volatility input is hourly-based annualized; convert to daily sigma
        sigma_daily = float(volatility) / math.sqrt(self.days_in_year)
        objectives = [self._equation_54_objective(a, sigma_daily, correlation, T_days) for a in self.alphas]
        best_idx = int(np.argmax(objectives))
        return float(self.alphas[best_idx])


