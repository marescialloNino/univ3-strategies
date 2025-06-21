import pandas as pd
import numpy as np
import math
import arch
import UNI_v3_funcs
import ActiveStrategyFramework
import scipy
import copy

import pandas as pd
import numpy as np
import statsmodels.tsa.ar_model as ar_model

class AutoRegressiveStrategy:
    def __init__(self, model_data, alpha_param, tau_param, volatility_reset_ratio, tokens_outside_reset=0.05, data_frequency='d', default_width=0.5, days_ar_model=180, return_forecast_cutoff=0.15, z_score_cutoff=5):
        if data_frequency == 'd':
            self.annualization_factor = 365**0.5
            self.resample_option = '1d'
            self.window_size = 15
        elif data_frequency == 'h':
            self.annualization_factor = (24*365)**0.5
            self.resample_option = '1h'
            self.window_size = 24 * 6  # 144 hours for 6-day dataset
        elif data_frequency == 'm':
            self.annualization_factor = (60*24*365)**0.5
            self.resample_option = '1min'
            self.window_size = 60
        
        self.alpha_param = alpha_param
        self.tau_param = tau_param
        self.volatility_reset_ratio = volatility_reset_ratio
        self.data_frequency = data_frequency
        self.tokens_outside_reset = tokens_outside_reset
        self.default_width = default_width
        self.return_forecast_cutoff = return_forecast_cutoff
        self.days_ar_model = days_ar_model
        self.z_score_cutoff = z_score_cutoff
        
        model_data = model_data.copy()
        if not isinstance(model_data.index, pd.DatetimeIndex):
            model_data.index = pd.to_datetime(model_data.index, utc=True)
        model_data.index = model_data.index.astype('datetime64[ns, UTC]')
        print(f"model_data index dtype before clean_data_for_garch: {model_data.index.dtype}")
        
        self.model_data = self.clean_data_for_garch(model_data)
        
        print(f"model_data index type after clean_data_for_garch: {type(self.model_data.index)}")
        print(f"model_data index dtype after clean_data_for_garch: {self.model_data.index.dtype}")
        print(f"model_data shape after clean_data_for_garch: {self.model_data.shape}")

    def clean_data_for_garch(self, data_in):
        print(f"data_in index dtype in clean_data_for_garch: {data_in.index.dtype}")
        data_filled = ActiveStrategyFramework.fill_time(data_in)
        
        print(f"data_filled index type in clean_data_for_garch: {type(data_filled.index)}")
        print(f"data_filled index dtype in clean_data_for_garch: {data_filled.index.dtype}")
        print(f"data_filled shape: {data_filled.shape}")
        
        # Adjust window_size based on data length
        data_length = len(data_filled)
        window_size = min(self.window_size, data_length // 2)
        print(f"Adjusted window_size in clean_data_for_garch: {window_size}")
        
        data_filled_rolling = data_filled.quotePrice.rolling(window=window_size)
        data_filled['roll_median'] = data_filled_rolling.median()
        roll_dev = np.abs(data_filled.quotePrice - data_filled.roll_median)
        data_filled['median_abs_dev'] = 1.4826 * roll_dev.rolling(window=window_size).median()
        outlier_indices = np.abs(data_filled.quotePrice - data_filled.roll_median) >= self.z_score_cutoff * data_filled['median_abs_dev']
        data_filled = data_filled[~outlier_indices]
        
        print(f"data_filled shape after outlier removal: {data_filled.shape}")
        return data_filled

    def generate_model_forecast(self, price_data, time_data):
        # Validate inputs
        if not isinstance(price_data, (pd.Series, np.ndarray)) or not isinstance(time_data, (pd.Index, np.ndarray)):
            raise TypeError("price_data must be a Series or array, time_data must be an Index or array")
        
        if len(price_data) == 0 or len(time_data) == 0:
            raise ValueError("Empty price_data or time_data provided to generate_model_forecast")
        
        # Convert to Series with DatetimeIndex
        if not isinstance(price_data, pd.Series):
            price_data = pd.Series(price_data, index=time_data)
        if not isinstance(time_data, pd.Index):
            time_data = pd.Index(time_data)
        
        if not isinstance(time_data, pd.DatetimeIndex):
            time_data = pd.to_datetime(time_data, utc=True)
        time_data = time_data.astype('datetime64[ns, UTC]')
        price_data.index = time_data
        
        print(f"generate_model_forecast price_data shape: {price_data.shape}")
        print(f"generate_model_forecast time_data range: {time_data.min()} to {time_data.max()}")
        
        # Resample price_data to match frequency
        data_resampled = pd.DataFrame(
            {'quotePrice': price_data.values},
            index=time_data
        ).resample(self.resample_option).last().ffill()
        
        print(f"data_resampled shape: {data_resampled.shape}")
        
        # Calculate returns
        data_resampled['price_return'] = data_resampled['quotePrice'].pct_change()
        returns = data_resampled['price_return'].dropna()
        
        # Adjust days_ar_model based on available data
        if self.data_frequency == 'd':
            samples_per_day = 1
        elif self.data_frequency == 'h':
            samples_per_day = 24
        elif self.data_frequency == 'm':
            samples_per_day = 60 * 24
        
        max_samples = len(returns)
        required_samples = self.days_ar_model * samples_per_day
        adjusted_samples = min(required_samples, max_samples - 1)
        adjusted_days = adjusted_samples // samples_per_day
        print(f"Adjusted days_ar_model: {adjusted_days}, samples: {adjusted_samples}")
        
        if adjusted_samples < 10:
            raise ValueError(f"Insufficient data for AR model: {adjusted_samples} samples available, need at least 10.")
        
        # Fit AR model
        returns_for_model = returns[-adjusted_samples:]
        ar_model_fit = ar_model.AutoReg(returns_for_model, lags=2).fit()
        
        # Generate forecast
        forecast = ar_model_fit.forecast(steps=1)
        forecast_mean = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
        
        # Fit GARCH model for volatility
        garch_model = arch.arch_model(returns_for_model * 100, vol='Garch', p=1, q=1, dist='normal')
        garch_fit = garch_model.fit(disp='off')
        garch_forecast = garch_fit.forecast(horizon=1)
        sd_forecast = np.sqrt(garch_forecast.variance.values[-1, -1]) / 100
        
        return {'return_forecast': forecast_mean, 'sd_forecast': sd_forecast}


        
    def check_compound_possible(self,current_strat_obs):
        baseLower  = current_strat_obs.liquidity_ranges[0]['lower_bin_tick']
        baseUpper  = current_strat_obs.liquidity_ranges[0]['upper_bin_tick']
        limitLower = current_strat_obs.liquidity_ranges[1]['lower_bin_tick']
        limitUpper = current_strat_obs.liquidity_ranges[1]['upper_bin_tick']
        
        base_assets_token_1  = current_strat_obs.liquidity_ranges[0]['token_0'] * current_strat_obs.price + current_strat_obs.liquidity_ranges[0]['token_1']
        limit_assets_token_1 = current_strat_obs.liquidity_ranges[0]['token_0'] * current_strat_obs.price + current_strat_obs.liquidity_ranges[0]['token_1']
        
        unused_token_0  = current_strat_obs.token_0_left_over + current_strat_obs.token_0_fees_uncollected
        unused_token_1  = current_strat_obs.token_1_left_over + current_strat_obs.token_1_fees_uncollected
        #####################################
        # Add all possible assets to base
        #####################################
        liquidity_placed_base   = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current,baseLower,baseUpper,unused_token_0, \
                                                                       unused_token_1,current_strat_obs.decimals_0,current_strat_obs.decimals_1))
        
        base_amount_0_placed,base_amount_1_placed   = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,baseLower,baseUpper,liquidity_placed_base\
                                                                 ,current_strat_obs.decimals_0,current_strat_obs.decimals_1)
        

        #####################################
        # Add remaining assets to limit
        #####################################
        
        limit_amount_0 = unused_token_0 - base_amount_0_placed
        limit_amount_1 = unused_token_1 - base_amount_1_placed
        
        token_0_limit  = limit_amount_0*current_strat_obs.price > limit_amount_1
        # Place single sided highest value
        if token_0_limit:        
            # Place Token 0
            limit_amount_1 = 0.0
        else:
            # Place Token 1
            limit_amount_0 = 0.0

        liquidity_placed_limit                      = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current,limitLower,limitUpper, \
                                                                       limit_amount_0,limit_amount_1,current_strat_obs.decimals_0,current_strat_obs.decimals_1))
        limit_amount_0_placed,limit_amount_1_placed =     UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,limitLower,limitUpper,\
                                                                     liquidity_placed_limit,current_strat_obs.decimals_0,current_strat_obs.decimals_1)  
                                                                     
        
        base_assets_after_compound_token_1  = base_amount_0_placed  * current_strat_obs.price + base_amount_1_placed + base_assets_token_1                            
        limit_assets_after_compound_token_1 = limit_amount_0_placed * current_strat_obs.price + limit_amount_1_placed + limit_assets_token_1                           

        # if assets changed more than 1%
        if ((base_assets_after_compound_token_1+limit_assets_after_compound_token_1)/(base_assets_token_1 + limit_assets_token_1) - 1) > .01:
            return True
        else:
            return False     
        
    #####################################
    # Check if a rebalance is necessary. 
    # If it is, remove the liquidity and set new ranges
    #####################################
        
    def check_strategy(self, current_strat_obs):
        LIMIT_ORDER_BALANCE = current_strat_obs.liquidity_ranges[1]['token_0'] * current_strat_obs.price + current_strat_obs.liquidity_ranges[1]['token_1']
        BASE_ORDER_BALANCE = current_strat_obs.liquidity_ranges[0]['token_0'] * current_strat_obs.price + current_strat_obs.liquidity_ranges[0]['token_1']
        
        if 'last_vol_check' not in current_strat_obs.strategy_info:
            current_strat_obs.strategy_info['last_vol_check'] = current_strat_obs.time
        
        # Check reset conditions
        LEFT_RANGE_LOW = current_strat_obs.price < current_strat_obs.strategy_info['reset_range_lower']
        LEFT_RANGE_HIGH = current_strat_obs.price > current_strat_obs.strategy_info['reset_range_upper']
        
        # Volatility rebalance check
        VOL_REBALANCE = False
        ar_check_frequency = 60  # Check every hour
        time_since_reset = (current_strat_obs.time - current_strat_obs.strategy_info['last_vol_check']).total_seconds() / 60
        
        if time_since_reset >= ar_check_frequency:
            current_strat_obs.strategy_info['last_vol_check'] = current_strat_obs.time
            # Use self.model_data up to current time
            price_data = self.model_data['quotePrice'][self.model_data.index <= current_strat_obs.time]
            time_data = self.model_data.index[self.model_data.index <= current_strat_obs.time]
            model_forecast = self.generate_model_forecast(price_data, time_data)
            
            if model_forecast['sd_forecast'] / current_strat_obs.liquidity_ranges[0]['volatility'] <= self.volatility_reset_ratio:
                VOL_REBALANCE = True
        
        # Initial reset check
        INITIAL_RESET = current_strat_obs.strategy_info.get('force_initial_reset', False)
        if INITIAL_RESET:
            current_strat_obs.strategy_info['force_initial_reset'] = False
        
        # Tokens outside check
        left_over_balance = (current_strat_obs.token_0_left_over + current_strat_obs.token_0_fees_uncollected) * current_strat_obs.price + \
                            (current_strat_obs.token_1_left_over + current_strat_obs.token_1_fees_uncollected)
        TOKENS_OUTSIDE_LARGE = left_over_balance > self.tokens_outside_reset * (LIMIT_ORDER_BALANCE + BASE_ORDER_BALANCE)
        
        # Reset if necessary
        if (LEFT_RANGE_LOW | LEFT_RANGE_HIGH) | VOL_REBALANCE | INITIAL_RESET:
            current_strat_obs.reset_point = True
            current_strat_obs.reset_reason = 'exited_range' if (LEFT_RANGE_LOW | LEFT_RANGE_HIGH) else \
                                            'vol_rebalance' if VOL_REBALANCE else 'initial_reset'
            current_strat_obs.remove_liquidity()
            liquidity_ranges, strategy_info = self.set_liquidity_ranges(current_strat_obs, model_forecast)
            return liquidity_ranges, strategy_info
        
        # Compound if necessary
        if TOKENS_OUTSIDE_LARGE:
            if self.check_compound_possible(current_strat_obs):
                current_strat_obs.compound_point = True
                current_strat_obs.reset_reason = 'compound'
                self.compound(current_strat_obs)
                return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info
            else:
                current_strat_obs.reset_point = True
                current_strat_obs.reset_reason = 'tokens_outside_large'
                current_strat_obs.remove_liquidity()
                liquidity_ranges, strategy_info = self.set_liquidity_ranges(current_strat_obs)
                return liquidity_ranges, strategy_info
        
        return current_strat_obs.liquidity_ranges, current_strat_obs.strategy_info

    ########################################################
    # Rebalance the position
    ########################################################
            
    def set_liquidity_ranges(self, current_strat_obs, model_forecast=None, price_data=None, time_data=None):
        if model_forecast is None:
            if price_data is None or time_data is None:
                # Fallback to self.model_data
                price_data = self.model_data['quotePrice'][self.model_data.index <= current_strat_obs.time]
                time_data = self.model_data.index[self.model_data.index <= current_strat_obs.time]
            else:
                # Use provided price_data up to current time
                mask = time_data <= current_strat_obs.time
                price_data = price_data[mask]
                time_data = time_data[mask]
            
            if len(price_data) < 10:
                print(f"Warning: Limited data for forecast at {current_strat_obs.time}, using recent data")
                price_data = self.model_data['quotePrice'].tail(720)  # Last 30 days
                time_data = self.model_data.index[-720:]
            
            model_forecast = self.generate_model_forecast(price_data, time_data)
            
        # Make sure strategy_info (dict with additional vars exists)    
        if current_strat_obs.strategy_info is None:
            strategy_info_here = dict()
        else:
            strategy_info_here = copy.deepcopy(current_strat_obs.strategy_info)
            
        # Limit return prediction to a return_forecast_cutoff % change
        if np.abs(model_forecast['return_forecast']) > self.return_forecast_cutoff:
                    model_forecast['return_forecast'] = np.sign(model_forecast['return_forecast']) * self.return_forecast_cutoff
                
        # If error in volatility computation use last or overall standard deviation of returns
        if np.isnan(model_forecast['sd_forecast']):
            if hasattr(current_strat_obs,'liquidity_ranges'):
                model_forecast['sd_forecast']  = current_strat_obs.liquidity_ranges[0]['volatility']
            else:
                model_forecast['sd_forecast'] = self.model_data.quotePrice.pct_change().std()

           
        target_price     = (1 + model_forecast['return_forecast']) * current_strat_obs.price

        # Set the base range
        base_range_lower           = current_strat_obs.price * (1 + model_forecast['return_forecast'] - self.alpha_param*model_forecast['sd_forecast'])
        base_range_upper           = current_strat_obs.price * (1 + model_forecast['return_forecast'] + self.alpha_param*model_forecast['sd_forecast'])
        
        # Set the reset range
        strategy_info_here['reset_range_lower'] = current_strat_obs.price * (1 + model_forecast['return_forecast'] - self.tau_param*self.alpha_param*model_forecast['sd_forecast'])
        strategy_info_here['reset_range_upper'] = current_strat_obs.price * (1 + model_forecast['return_forecast'] + self.tau_param*self.alpha_param*model_forecast['sd_forecast'])
        
        # If volatility is high enough reset range is less than zero, set at default_width of current price
        if strategy_info_here['reset_range_lower'] < 0.0:
            strategy_info_here['reset_range_lower'] = self.default_width * current_strat_obs.price
        
        liquidity_ranges                = []
        
        ########################################################### 
        # STEP 2: Set Base Liquidity
        ###########################################################
        
        # Store each token amount supplied to pool
        total_token_0_amount = current_strat_obs.liquidity_in_0
        total_token_1_amount = current_strat_obs.liquidity_in_1
                                    
        # Set baseLower
        if base_range_lower > 0.0:
            baseLowerPRE       = math.log(current_strat_obs.decimal_adjustment*base_range_lower,1.0001)
            baseLower          = int(math.floor(baseLowerPRE/current_strat_obs.tickSpacing)*current_strat_obs.tickSpacing)
        else:
            # If lower end of base range is negative, fix at 0.0
            base_range_lower   = 0.0
            baseLower          = math.ceil(math.log((2**-128),1.0001)/current_strat_obs.tickSpacing)*current_strat_obs.tickSpacing

        # Set baseUpper
        baseUpperPRE      = math.log(current_strat_obs.decimal_adjustment*base_range_upper,1.0001)
        baseUpper         = int(math.floor(baseUpperPRE/current_strat_obs.tickSpacing)*current_strat_obs.tickSpacing)

        ## Sanity Checks
        # Make sure baseLower < baseUpper. If not make two tick
        if baseLower >= baseUpper:
            baseLower = current_strat_obs.price_tick - current_strat_obs.tickSpacing
            baseUpper = current_strat_obs.price_tick + current_strat_obs.tickSpacing
        
        liquidity_placed_base   = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current,baseLower,baseUpper,current_strat_obs.liquidity_in_0, \
                                                                       current_strat_obs.liquidity_in_1,current_strat_obs.decimals_0,current_strat_obs.decimals_1))
        
        base_amount_0_placed,base_amount_1_placed   = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,baseLower,baseUpper,liquidity_placed_base\
                                                                 ,current_strat_obs.decimals_0,current_strat_obs.decimals_1)

        base_liq_range =       {'price'              : current_strat_obs.price,
                                'target_price'       : target_price,
                                'lower_bin_tick'     : baseLower,
                                'upper_bin_tick'     : baseUpper,
                                'lower_bin_price'    : base_range_lower,
                                'upper_bin_price'    : base_range_upper,
                                'time'               : current_strat_obs.time,
                                'token_0'            : base_amount_0_placed,
                                'token_1'            : base_amount_1_placed,
                                'position_liquidity' : liquidity_placed_base,
                                'volatility'         : model_forecast['sd_forecast'],
                                'reset_time'         : current_strat_obs.time,
                                'return_forecast'    : model_forecast['return_forecast']}

        liquidity_ranges.append(base_liq_range)

        ###########################
        # Step 3: Set Limit Position 
        ############################
        
        limit_amount_0 = total_token_0_amount - base_amount_0_placed
        limit_amount_1 = total_token_1_amount - base_amount_1_placed
        
        token_0_limit  = limit_amount_0*current_strat_obs.price > limit_amount_1
        # Place singe sided highest value
        if token_0_limit:        
            # Place Token 0
            limit_amount_1    = max([0.0,limit_amount_1])
            limit_range_lower = current_strat_obs.price
            limit_range_upper = base_range_upper                     
        else:
            # Place Token 1
            limit_amount_0    = max([0.0,limit_amount_0])
            limit_range_lower = base_range_lower
            limit_range_upper = current_strat_obs.price
        
        # Set limitLower
        if limit_range_lower > 0.0:
            limitLowerPRE      = math.log(current_strat_obs.decimal_adjustment*limit_range_lower,1.0001)
            limitLower         = int(math.floor(limitLowerPRE/current_strat_obs.tickSpacing)*current_strat_obs.tickSpacing)
        else:
            limit_range_lower  = 0.0
            limitLower         = math.ceil(math.log((2**-128),1.0001)/current_strat_obs.tickSpacing)*current_strat_obs.tickSpacing
                
        # Set limitUpper
        limitUpperPRE     = math.log(current_strat_obs.decimal_adjustment*limit_range_upper,1.0001)
        limitUpper        = int(math.floor(limitUpperPRE/current_strat_obs.tickSpacing)*current_strat_obs.tickSpacing)
        
        ## Sanity Checks
        if token_0_limit:        
            if limitLower <= current_strat_obs.price_tick_current:
                # If token 0 in limit, make sure lower tick is above active tick
                limitLower = limitLower + current_strat_obs.tickSpacing
            elif (limitLower / current_strat_obs.price_tick_current) < 1.25:
                # if bottom of limit tick is less than 125% of the current, add one tick space
                limitLower = limitLower + current_strat_obs.tickSpacing
        else:
            # In token 1 in limit, make sure upper tick is below active tick
            if limitUpper >= current_strat_obs.price_tick_current:
                limitUpper = limitUpper - current_strat_obs.tickSpacing
            elif (current_strat_obs.price_tick_current / limitUpper) < 1.25:
                # if current is less than 125% of top of limit tick, reduce one tick space
                limitUpper = limitUpper - current_strat_obs.tickSpacing    
        
        # Make sure limitLower < limitUpper. If not make one tick    
        if limitLower >= limitUpper:
            if token_0_limit:
                limitLower += current_strat_obs.tickSpacing
            else:
                limitUpper -= current_strat_obs.tickSpacing

        liquidity_placed_limit                      = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current,limitLower,limitUpper, \
                                                                       limit_amount_0,limit_amount_1,current_strat_obs.decimals_0,current_strat_obs.decimals_1))
        limit_amount_0_placed,limit_amount_1_placed =     UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,limitLower,limitUpper,\
                                                                     liquidity_placed_limit,current_strat_obs.decimals_0,current_strat_obs.decimals_1)  


        limit_liq_range =       {'price'              : current_strat_obs.price,
                                 'target_price'       : target_price,
                                 'lower_bin_tick'     : limitLower,
                                 'upper_bin_tick'     : limitUpper,
                                 'lower_bin_price'    : limit_range_lower,
                                 'upper_bin_price'    : limit_range_upper,                                 
                                 'time'               : current_strat_obs.time,
                                 'token_0'            : limit_amount_0_placed,
                                 'token_1'            : limit_amount_1_placed,
                                 'position_liquidity' : liquidity_placed_limit,
                                 'volatility'         : model_forecast['sd_forecast'],
                                 'reset_time'         : current_strat_obs.time,
                                 'return_forecast'    : model_forecast['return_forecast']}     

        liquidity_ranges.append(limit_liq_range)
        
        # How much liquidity is not allcated to ranges
        current_strat_obs.token_0_left_over = max([total_token_0_amount - base_amount_0_placed - limit_amount_0_placed,0.0])
        current_strat_obs.token_1_left_over = max([total_token_1_amount - base_amount_1_placed - limit_amount_1_placed,0.0])

        # Since liquidity was allocated, set to 0
        current_strat_obs.liquidity_in_0 = 0.0
        current_strat_obs.liquidity_in_1 = 0.0
        
        return liquidity_ranges,strategy_info_here

    
    ########################################################
    # Compound unused liquidity
    ########################################################
    
    def compound(self,current_strat_obs):
        
        unused_token_0 = current_strat_obs.token_0_left_over + current_strat_obs.token_0_fees_uncollected
        unused_token_1 = current_strat_obs.token_1_left_over + current_strat_obs.token_1_fees_uncollected
        
        baseLower  = current_strat_obs.liquidity_ranges[0]['lower_bin_tick']
        baseUpper  = current_strat_obs.liquidity_ranges[0]['upper_bin_tick']
        limitLower = current_strat_obs.liquidity_ranges[1]['lower_bin_tick']
        limitUpper = current_strat_obs.liquidity_ranges[1]['upper_bin_tick']
        
        #####################################
        # Add all possible assets to base
        #####################################
        liquidity_placed_base   = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current,baseLower,baseUpper,unused_token_0, \
                                                                       unused_token_1,current_strat_obs.decimals_0,current_strat_obs.decimals_1))
        
        base_amount_0_placed,base_amount_1_placed   = UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,baseLower,baseUpper,liquidity_placed_base\
                                                                 ,current_strat_obs.decimals_0,current_strat_obs.decimals_1)
        
        
        current_strat_obs.liquidity_ranges[0]['token_0'] += base_amount_0_placed
        current_strat_obs.liquidity_ranges[0]['token_1'] += base_amount_1_placed 

        #####################################
        # Add remaining assets to limit
        #####################################
        
        limit_amount_0 = unused_token_0 - base_amount_0_placed
        limit_amount_1 = unused_token_1 - base_amount_1_placed
        
        token_0_limit  = limit_amount_0*current_strat_obs.price > limit_amount_1
        # Place single sided highest value
        if token_0_limit:        
            # Place Token 0
            limit_amount_1 = 0.0
        else:
            # Place Token 1
            limit_amount_0 = 0.0

        liquidity_placed_limit                      = int(UNI_v3_funcs.get_liquidity(current_strat_obs.price_tick_current,limitLower,limitUpper, \
                                                                       limit_amount_0,limit_amount_1,current_strat_obs.decimals_0,current_strat_obs.decimals_1))
        limit_amount_0_placed,limit_amount_1_placed =     UNI_v3_funcs.get_amounts(current_strat_obs.price_tick_current,limitLower,limitUpper,\
                                                                     liquidity_placed_limit,current_strat_obs.decimals_0,current_strat_obs.decimals_1)  
        
        current_strat_obs.liquidity_ranges[1]['token_0'] += limit_amount_0_placed
        current_strat_obs.liquidity_ranges[1]['token_1'] += limit_amount_1_placed
        
        # Clean up prior accrued fees and tokens outside        
        current_strat_obs.token_0_fees_uncollected  = 0.0
        current_strat_obs.token_1_fees_uncollected  = 0.0
        
        # Due to price and asset deposit ratio sometimes can't deposit 100% of assets
        current_strat_obs.token_0_left_over         = max([0.0,unused_token_0 - base_amount_0_placed - limit_amount_0_placed])
        current_strat_obs.token_1_left_over         = max([0.0,unused_token_1 - base_amount_1_placed - limit_amount_1_placed])
        
    
    ########################################################
    # Extract strategy parameters
    ########################################################
    def dict_components(self,strategy_observation):
            this_data = dict()
            
            # General variables
            this_data['time']                   = strategy_observation.time
            this_data['price']                  = strategy_observation.price
            this_data['reset_point']            = strategy_observation.reset_point
            this_data['compound_point']         = strategy_observation.compound_point
            this_data['reset_reason']           = strategy_observation.reset_reason
            this_data['volatility']             = strategy_observation.liquidity_ranges[0]['volatility']
            this_data['return_forecast']        = strategy_observation.liquidity_ranges[0]['return_forecast']
            
            
            # Range Variables
            this_data['base_range_lower']       = strategy_observation.liquidity_ranges[0]['lower_bin_price']
            this_data['base_range_upper']       = strategy_observation.liquidity_ranges[0]['upper_bin_price']
            this_data['limit_range_lower']      = strategy_observation.liquidity_ranges[1]['lower_bin_price']
            this_data['limit_range_upper']      = strategy_observation.liquidity_ranges[1]['upper_bin_price']
            this_data['reset_range_lower']      = strategy_observation.strategy_info['reset_range_lower']
            this_data['reset_range_upper']      = strategy_observation.strategy_info['reset_range_upper']
            this_data['price_at_reset']         = strategy_observation.liquidity_ranges[0]['price']
            
            # Fee Varaibles
            this_data['token_0_fees']                 = strategy_observation.token_0_fees 
            this_data['token_1_fees']                 = strategy_observation.token_1_fees 
            this_data['token_0_fees_uncollected']     = strategy_observation.token_0_fees_uncollected
            this_data['token_1_fees_uncollected']     = strategy_observation.token_1_fees_uncollected
            
            # Asset Variables
            this_data['token_0_left_over']      = strategy_observation.token_0_left_over
            this_data['token_1_left_over']      = strategy_observation.token_1_left_over
            
            total_token_0 = 0.0
            total_token_1 = 0.0
            for i in range(len(strategy_observation.liquidity_ranges)):
                total_token_0 += strategy_observation.liquidity_ranges[i]['token_0']
                total_token_1 += strategy_observation.liquidity_ranges[i]['token_1']
                
            this_data['token_0_allocated']      = total_token_0
            this_data['token_1_allocated']      = total_token_1
            this_data['token_0_total']          = total_token_0 + strategy_observation.token_0_left_over + strategy_observation.token_0_fees_uncollected
            this_data['token_1_total']          = total_token_1 + strategy_observation.token_1_left_over + strategy_observation.token_1_fees_uncollected

            # Value Variables
            this_data['value_position_in_token_0']         = this_data['token_0_total']     + this_data['token_1_total']     / this_data['price']
            this_data['value_allocated_in_token_0']        = this_data['token_0_allocated'] + this_data['token_1_allocated'] / this_data['price']
            this_data['value_left_over_in_token_0']        = this_data['token_0_left_over'] + this_data['token_1_left_over'] / this_data['price']
            
            this_data['base_position_value_in_token_0']    = strategy_observation.liquidity_ranges[0]['token_0'] + strategy_observation.liquidity_ranges[0]['token_1'] / this_data['price']
            this_data['limit_position_value_in_token_0']   = strategy_observation.liquidity_ranges[1]['token_0'] + strategy_observation.liquidity_ranges[1]['token_1'] / this_data['price']
             
            return this_data