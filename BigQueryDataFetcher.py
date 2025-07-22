import os
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

class BigQueryDataFetcher:
    """
    A class for fetching Uniswap v3 pool data from Google BigQuery.
    
    This class handles querying swap data from BigQuery for Ethereum Mainnet or Polygon networks,
    preprocesses the data (e.g., decimal adjustments, tick calculations), and automatically saves
    the results in an organized directory structure for easy access and reproducibility.
    
    Directory structure for saved data:
    data/
    └── {network}/
        └── {contract_address}/
            └── pool_data_{date_begin}_{date_end}.csv  # or .pkl if format='pickle'
    
    Usage:
    fetcher = BigQueryDataFetcher()
    data = fetcher.fetch_pool_data(
        contract_address='0x...',
        date_begin='2021-05-05',
        date_end='2025-07-12',
        decimals_0=18,
        decimals_1=6,
        network='mainnet',
        block_start=0,
        save_format='csv'  # or 'pickle'
    )
    """

    def __init__(self, project_id=None, credentials_path=None):
        """
        Initialize the BigQuery client.
        
        Args:
            project_id (str, optional): Google Cloud project ID. If None, uses default credentials.
            credentials_path (str, optional): Path to service account JSON key file.
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = bigquery.Client(project=project_id)
        self.data_dir = 'data'  # Base directory for saving data

    def _build_query(self, contract_address, date_begin, date_end, block_start, network):
        """
        Build the BigQuery SQL query based on the network.
        
        Args:
            contract_address (str): Uniswap v3 pool contract address (lowercase).
            date_begin (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            block_start (int): Starting block number.
            network (str): 'mainnet' or 'polygon'.
        
        Returns:
            str: SQL query string.
        """
        if network == 'mainnet':
            query = f"""
                SELECT *
                FROM blockchain-etl.ethereum_uniswap.UniswapV3Pool_event_Swap
                WHERE contract_address = LOWER('{contract_address}')
                AND block_timestamp >= '{date_begin}'
                AND block_timestamp <= '{date_end}'
                AND block_number >= {block_start}
            """
        elif network == 'polygon':
            query = f"""
                SELECT
                  block_number,
                  transaction_index,
                  log_index,
                  block_hash,
                  transaction_hash,
                  address,
                  block_timestamp,
                  '0x' || RIGHT(topics[SAFE_OFFSET(1)],40) AS sender,
                  '0x' || RIGHT(topics[SAFE_OFFSET(1)],40) AS recipient,
                  '0x' || SUBSTR(DATA, 3, 64) AS amount0,
                  '0x' || SUBSTR(DATA, 67, 64) AS amount1,
                  '0x' || SUBSTR(DATA,131,64) AS sqrtPriceX96,
                  '0x' || SUBSTR(DATA,195,64) AS liquidity,
                  '0x' || SUBSTR(DATA,259,64) AS tick
                FROM public-data-finance.crypto_polygon.logs
                WHERE
                  topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
                  AND DATE(block_timestamp) >= DATE('{date_begin}')
                  AND DATE(block_timestamp) <= DATE('{date_end}')
                  AND block_number >= {block_start}
                  AND address = '{contract_address}'
            """
        else:
            raise ValueError(f"Unsupported network: {network}. Choose 'mainnet' or 'polygon'.")
        return query

    def _preprocess_data(self, df, decimals_0, decimals_1, network):
        """
        Preprocess the raw BigQuery data.
        
        Args:
            df (pd.DataFrame): Raw query results.
            decimals_0 (int): Decimals for token 0.
            decimals_1 (int): Decimals for token 1.
            network (str): 'mainnet' or 'polygon'.
        
        Returns:
            pd.DataFrame: Preprocessed data ready for simulations.
        """
        if network == 'polygon':
            # Convert hex to signed integers for Polygon data
            df['amount0'] = df['amount0'].apply(self._signed_int)
            df['amount1'] = df['amount1'].apply(self._signed_int)
            df['sqrtPriceX96'] = df['sqrtPriceX96'].apply(self._signed_int)
            df['liquidity'] = df['liquidity'].apply(self._signed_int)
            df['tick'] = df['tick'].apply(self._signed_int)

        decimal_adj = 10 ** (decimals_1 - decimals_0)
        df['sqrtPriceX96_float'] = df['sqrtPriceX96'].astype(float)
        df['quotePrice'] = (((df['sqrtPriceX96_float'] / 2 ** 96) ** 2) / decimal_adj).astype(float)
        df['block_date'] = pd.to_datetime(df['block_timestamp'])
        df = df.set_index('block_date', drop=False).sort_index()

        df['tick_swap'] = df['tick'].astype(int)
        df['amount0'] = df['amount0'].astype(float)
        df['amount1'] = df['amount1'].astype(float)
        df['amount0_adj'] = df['amount0'] / 10 ** decimals_0
        df['amount1_adj'] = df['amount1'] / 10 ** decimals_1
        df['virtual_liquidity'] = df['liquidity'].astype(float)
        df['virtual_liquidity_adj'] = df['liquidity'] / (10 ** ((decimals_0 + decimals_1) / 2))
        df['token_in'] = df.apply(lambda x: 'token0' if x['amount0_adj'] < 0 else 'token1', axis=1)
        df['traded_in'] = df.apply(lambda x: -x['amount0_adj'] if x['amount0_adj'] < 0 else -x['amount1_adj'], axis=1).astype(float)

        return df

    def _signed_int(self, h):
        """Convert hex values to signed integers."""
        s = bytes.fromhex(h[2:])
        i = int.from_bytes(s, 'big', signed=True)
        return i

    def _get_save_path(self, contract_address, date_begin, date_end, network, save_format):
        """
        Generate the save path in an organized directory structure.
        
        Args:
            contract_address (str): Pool contract address.
            date_begin (str): Start date.
            date_end (str): End date.
            network (str): 'mainnet' or 'polygon'.
            save_format (str): 'csv' or 'pickle'.
        
        Returns:
            str: Full file path.
        """
        dir_path = os.path.join(self.data_dir, network, contract_address.replace('0x', '').lower())
        os.makedirs(dir_path, exist_ok=True)
        filename = f"pool_data_{date_begin}_{date_end}.{save_format}"
        return os.path.join(dir_path, filename)

    def fetch_pool_data(self, contract_address, date_begin, date_end, decimals_0, decimals_1,
                        network='mainnet', block_start=0, save_format='csv', overwrite=False):
        """
        Fetch and preprocess pool data from BigQuery, then save it automatically.
        
        Args:
            contract_address (str): Uniswap v3 pool contract address.
            date_begin (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            decimals_0 (int): Decimals for token 0.
            decimals_1 (int): Decimals for token 1.
            network (str, optional): 'mainnet' or 'polygon'. Defaults to 'mainnet'.
            block_start (int, optional): Starting block number. Defaults to 0.
            save_format (str, optional): 'csv' or 'pickle'. Defaults to 'csv'.
            overwrite (bool, optional): Overwrite existing file if it exists. Defaults to False.
        
        Returns:
            pd.DataFrame: Preprocessed pool data.
        
        Raises:
            ValueError: If save_format is invalid or network unsupported.
        """
        if save_format not in ['csv', 'pickle']:
            raise ValueError("save_format must be 'csv' or 'pickle'.")

        save_path = self._get_save_path(contract_address, date_begin, date_end, network, save_format)

        # Check if file exists and overwrite flag
        if os.path.exists(save_path) and not overwrite:
            print(f"Loading existing data from {save_path}")
            if save_format == 'csv':
                return pd.read_csv(save_path, parse_dates=['block_timestamp', 'block_date'])
            else:
                return pd.read_pickle(save_path)

        # Build and execute query
        query = self._build_query(contract_address.lower(), date_begin, date_end, block_start, network)
        try:
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            if df.empty:
                print("No data found for the given parameters.")
                return df
        except NotFound:
            raise ValueError("BigQuery dataset or table not found. Ensure you have access to 'blockchain-etl' or 'public-data-finance' datasets.")
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

        # Preprocess
        processed_df = self._preprocess_data(df, decimals_0, decimals_1, network)

        # Save automatically
        if save_format == 'csv':
            processed_df.to_csv(save_path, index=False)
        else:
            processed_df.to_pickle(save_path)
        print(f"Data saved to {save_path}")

        return processed_df