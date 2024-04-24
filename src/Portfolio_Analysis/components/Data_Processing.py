import pandas as pd

def add_total_pv_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Adds a 'Tot_PV' column to the DataFrame which sums up all Balance except the first one.
    Args: df(pd.DataFrame): The DataFrame to process.
    Returns: pd.DataFrame: Updated DataFrame with the new 'Tot_PV' column.
    """
    df['Tot_PV'] = df.iloc[:, 1:].sum(axis=1)
    return df

def preprocess_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Converts date strings in the 'executedAt (UTC)' column to datetime objects and extracts the date component.
    Args: df (pd.DataFrame): DataFrame containing transaction data.
    Returns: pd.DataFrame: The processed DataFrame with a new 'Date' column.
    """
    df['executedAt (UTC)'] = pd.to_datetime(df['executedAt (UTC)'])
    df['Date'] = df['executedAt (UTC)'].dt.date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def merge_dataframes(crypto_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Merges two DataFrames on the 'Date' column and refines transaction data.

    Args:   crypto_df (pd.DataFrame): DataFrame with Cryptocurrency Positional History. 
            transaction_df (pd.DataFrame): DataFrame with transaction data like Inflow and Outflow
    Returns:
    pd.DataFrame: Merged DataFrame with essential transaction columns.
    """
    # Mege the datasets based on data for the required columns
    merged_df = pd.merge(crypto_df, transaction_df[['Date', 'inflow amount', 'outflow amount']], on="Date", how="left") 

    # Extract and rename the inflow and outflow amounts
    merged_df.rename(columns={'inflow amount': 'Inflow', 'outflow amount': 'Outflow'}, inplace=True) #Rename Columns
    
    # Drop the old amount columns if needed
    merged_df.drop(columns=['inflow amount', 'outflow amount'], errors='ignore', inplace=True)

    # Fill NaN values with zero for the new columns
    merged_df[['Inflow', 'Outflow']] = merged_df[['Inflow', 'Outflow']].fillna(0)

    return merged_df

def Combining_sheets(crypto_dataframe: pd.DataFrame, transaction_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Main function to process and merge cryptocurrency position and transaction data."""
    try:
        #crypto_df = add_total_pv_column(crypto_dataframe)
        transaction_df = preprocess_transaction_data(transaction_dataframe)
        
        # Merge dataframes
        merged_df = merge_dataframes(crypto_dataframe, transaction_dataframe)

        return merged_df

    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_returns(df1: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Calculate daily percentage change for 'Tot_PV' and adjust for inflows and outflows to provide adjusted returns.
    Args:  df(pd.DataFrame): DataFrame with total portfolio values and transactions.
    Returns: pd.DataFrame: DataFrame with additional columns for returns and adjusted returns.
    """
    df1['Tot_return'] = df1['Tot_PV'].pct_change() * 100
    df1['Opening'] = df1['Tot_PV'].shift(1)
    df1['Closing'] = df1['Tot_PV'] - df1['Inflow'] + df1['Outflow']
    df1['Adj_returns'] = ((df1['Closing'] / df1['Opening']) - 1) * 100
    df1 = df1.fillna(0)
    return df1

def calculate_column_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns for all columns except 'Date'.
    Args: df(pd.DataFrame): The DataFrame to process.
    Returns: pd.DataFrame: New DataFrame with percentage changes for each column.
    """
    df2 = pd.DataFrame(df['Date'])
    for column in df.columns:
        if column != 'Date' and column !='Tot_PV' and column != 'Dollars': 
            df2[column] = df[column].pct_change()
    df2.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df2 = df2.fillna(0)
    return df2

def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weights of asset columns by dividing each by 'Tot_PV'.
    Args: df (pd.DataFrame): DataFrame with assets and their values.
    Returns: pd.DataFrame: Adjusted DataFrame with normalized weights for assets.
    """
    cols_to_normalize = [col for col in df.columns if col not in ["Date", "Tot_PV","Dollars" ,"Inflow", "Outflow"]]
    for col in cols_to_normalize:
        df[col] = (df[col] / df["Tot_PV"])
    df.drop(['Tot_PV'], axis=1, inplace=True)
    return df

def combine_dataframes(df_weights: pd.DataFrame, df_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Objective:  Multiply normalized weights by returns and sum them to calculate the daily portfolio return.
    Args:  df_weights (pd.DataFrame): DataFrame with asset weights.
           df_returns (pd.DataFrame): DataFrame with asset returns.
    Returns:  pd.DataFrame: Resulting DataFrame with daily portfolio returns.
    """
    df_weights.set_index('Date', inplace=True)
    df_returns.set_index('Date', inplace=True)

    df_returns = df_returns.iloc[1:]
    
    result_df = df_weights.mul(df_returns, axis=1)

    result_df['Daily_Portfolio_Return'] = result_df.sum(axis=1)  # Convert to percentage    
    return result_df

def process_financial_data(merged_df: pd.DataFrame, crypto_df: pd.DataFrame) -> pd.DataFrame: 
    """
    Objective:Process financial data to calculate adjusted returns and normalize weights.
    Args:  merged_df (pd.DataFrame): DataFrame with asset and cash inflow and outflow.
           crypto_df (pd.DataFrame): DataFrame with asset sum(Balance)
    Returns:  pd.DataFrame: Resulting DataFrame with daily portfolio returns.
    """
    try:
        print("Method 1 to calculate Daily returns")
        tot_ret_df = calculate_returns(merged_df)
        
        print("Method 2 to calculate Daily returns")
        IR_df = calculate_column_returns(crypto_df.copy()).round(5)
        weights_df = normalize_weights(crypto_df.copy()).round(5)
    
        result_df = combine_dataframes(weights_df.copy(), IR_df.copy())
    
        return result_df 

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of errors