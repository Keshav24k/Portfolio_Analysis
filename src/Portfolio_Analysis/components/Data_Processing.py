import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
    # Merge the datasets based on data for the required columns
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
        if column != 'Date' and column !='Tot_PV': #and column != 'Dollars': 
            df2[column] = df[column].pct_change() * 100
    df2.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df2 = df2.fillna(0)
    return df2

def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weights of asset columns by dividing each by 'Tot_PV'.
    Args: df (pd.DataFrame): DataFrame with assets and their values.
    Returns: pd.DataFrame: Adjusted DataFrame with normalized weights for assets.
    """
    #df = df.abs()
    #df.iloc[:, 1:] = df.iloc[:, 1:].abs()
    #df.iloc[:,-1] = df.iloc[:, 1:-1].sum(axis=1)
    #st.dataframe(df.head())

    cols_to_normalize = [col for col in df.columns if col not in ["Date", "Tot_PV","Inflow", "Outflow"]] #,"Dollars" 
    for col in cols_to_normalize:
        df[col] = (df[col] / df["Tot_PV"])
    #st.dataframe(df.head())
    df.drop(['Tot_PV'], axis=1, inplace=True)
    #st.dataframe(df.head())
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

def Cumulative_returns_Plot(df: pd.DataFrame):
    """Plots the cumulative returns chart.
    Args: DataFrame with Daily Returns Data
    returns: None
    """
    st.write("Cumulative Returns Chart:")
    cumulative_returns = (1 + df['Daily_Portfolio_Return'] / 100).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1 Investment')
    plt.legend()
    plt.show()
    st.pyplot(plt)
    return

def Weights_plot(df: pd.DataFrame):
    """Plots the Portfolio weights chart.
    Args: DataFrame with  Daily Weights Data
    returns: None
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df*100
    st.write("Weights Plot:")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(ax=ax)  # Use DataFrame's plot method for correct axis handling
    ax.set_title('Asset Weights Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (%)')  # Assuming the data is in percentage form
    st.pyplot(fig)

def returns_plot(df: pd.DataFrame):
    """Plots the Portfolio total Returns chart.
    Args: DataFrame with  Daily Weights Data
    returns: None
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df.iloc[:,-1], label='Daily Returns', color='green')
    plt.title('Daily Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    return

def Individual_returns_plot(df: pd.DataFrame):
    """Plots the Portfolio total Returns chart.
    Args: DataFrame with  Daily Weights Data
    returns: None
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df.iloc[:,:-1], label=df.columns[:-1])
    plt.title(' Individual Asset Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    return

def data_visualization(tot_ret_df: pd.DataFrame,result_df: pd.DataFrame,Weight_df: pd.DataFrame):

    tot_ret_df.set_index('Date', inplace=True)
    st.subheader("Returns Plot:")
    st.write("Inflow Outflow Adjusted returns:")
    returns_plot(tot_ret_df)
    st.markdown(""" <style>
                    input[data-baseweb="input"] {
                    width: 90% !important;
                }
                </style>
                """, unsafe_allow_html=True)

    st.write("The plot depicts the daily returns of a financial portfolio that exhibits good volatility," 
             "Most returns fluctuate between -10% and +15%, indicating some risk control. \n A prominent feature is a sharp spike around the 60th data point," 
             "where returns shoot up to about 30%, suggesting a significant positive impact on the portfolio at that time.\n\n" 
             "The actual dates are not specified, making it challenging to link the performance with specific external events without further information.\n")

    st.write("\nWeight Adjusted returns:")
    returns_plot(result_df)
    st.write("The plot shows weight-adjusted daily portfolio returns, reflecting performance after accounting for the proportional impact of each asset.\n" 
             "The returns demonstrate volatility, with most values oscillating near zero, indicating days with minimal gain or loss.\n" 
             "However, there's a notable anomaly with returns spiking above 150% due to the inflow of cash, which could point to a significant event affecting a heavily weighted asset.\n" 
             "This extreme spike is an outlier compared to the general trend of the portfolio's performance.\n")
    
    st.write("\Individual Asset returns:")
    Individual_returns_plot(result_df)
    st.write("The graph illustrates the varied performance of different assets over the specified timeframe with most assets showing minimal returns" 
             "and a couple displaying short-term volatility with sharp increases in returns.\n"
             "From the visual, it's clear that most assets have returns fluctuating close to the 0% line, suggesting stable or nominal growth over the observed period.")

    Cumulative_returns_Plot(result_df)
    st.write("The plot represents the cumulative returns on an investment over time. It shows the growth of $1 invested in the portfolio." 
             "The investment value grows steadily at first, indicating a gradual increase in returns.\n" 
             "Around early February, there is a significant jump in the value of the investment, indicating a period of high returns.\n" 
             "The investment value continues to increase at a more moderate pace, with some fluctuation, until it levels off toward the end of the period shown.") 

    Weights_plot(Weight_df)
    st.write("The weights plot displays the changing daily weight composition of a portfolio.\n" 
             "The most notable movements are the significant increase in the weight of OCEAN/USDT in late February, where it rises sharply to over 60% of the portfolio," 
             "and a corresponding increase in RPL/USDT.\n"
             "BADGER/USDT initially holding the majority share, decreases slightly but remains a substantial portion of the portfolio.\n" 
             "Other assets like Frax Share, Maker, Render, Ribbon Finance, and Dollar have minimal changes, indicating stable but smaller positions within the portfolio.\n")

    return
    
def process_financial_data(merged_df: pd.DataFrame, crypto_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Objective:Process financial data to calculate adjusted returns and normalize weights.
    Args:  merged_df (pd.DataFrame): DataFrame with asset and cash inflow and outflow.
           crypto_df (pd.DataFrame): DataFrame with asset sum(Balance)
    Returns:  pd.DataFrame: Resulting DataFrame with daily portfolio returns, column wise daily return, 
              row-wise weights of portfolio, daily returns based on transactions """
    try:
        print("Method 1 to calculate Daily returns")
        tot_ret_df = calculate_returns(merged_df)
        
        print("Method 2 to calculate Daily returns")
        IR_df = calculate_column_returns(crypto_df.copy()).round(5)
        weights_df = normalize_weights(crypto_df.copy()).round(5)
    
        result_df = combine_dataframes(weights_df.copy(), IR_df.copy())
    
        return result_df, IR_df, weights_df, tot_ret_df  

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()  # Return an empty DataFrame in case of errors