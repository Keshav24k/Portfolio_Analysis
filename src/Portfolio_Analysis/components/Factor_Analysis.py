from IPython.display import display
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import timedelta

def Covariance_Analysis(df):  
    """
    Analyzes the covariance of financial data.
    Args: df(pd.DataFrame): DataFrame containing financial data with 'Date' and other numeric columns.
    Effects: Displays the covariance matrix using Streamlit.
    """
    df['Date'] = pd.to_datetime(df['Date'])

  # Setting Date as the index
    df.set_index('Date', inplace=True)

  # Calculate daily returns for BTC, ETH, and ^GSPC
  #df[['BTC_Returns', 'ETH_Returns', 'GSPC_Returns']] = df[['BTC-USD', 'ETH-USD', '^GSPC']].pct_change()

  # Compute covariance matrix between the returns of BTC, ETH, ^GSPC, and the portfolio returns
    covariance_matrix = df.cov()

    st.write("Covariance Matrix:")
    st.dataframe(covariance_matrix)

    #Print the covariance matrix
    #print(covariance_matrix,"\n\n")

def Permutation_Importance(df):
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]

  # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

  # Predict on the test set
    y_pred = model.predict(X_test)

  # Calculate R2 Score to evaluate
    r2 = r2_score(y_test, y_pred)
    print(f'R2 Score: {r2}')

  # Compute permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# Display the importance of each feature with additional context
#   for i in perm_importance.importances_mean.argsort()[::-1]:
#       significant = perm_importance.importances_mean[i] - 2 * perm_importance.importances_std[i] > 0
#       print(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.3f} +/- {perm_importance.importances_std[i]:.3f}"
#             f" - {'Significant' if significant else 'Not Significant'}")

    st.title('Feature Importance Analysis')
    st.write("Below is the feature importance calculated based on permutation importance method.")

# Display feature importances
    for i in perm_importance.importances_mean.argsort()[::-1]:
        significant = perm_importance.importances_mean[i] - 2 * perm_importance.importances_std[i] > 0
        st.text(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.3f} +/- {perm_importance.importances_std[i]:.3f} "
                f" - {'Significant' if significant else 'Not Significant'}")



def Shap_Importance(data):
    st.title('SHAP Visualizations')
    
    # Prepare the data
    X = data.iloc[:,:-1]  # Assuming the last column is the target
    y = data.iloc[:,-1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train an XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

   # Calculate SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # For the force plot, convert it to a static plot and increase figure size and DPI
    st.write("Force Plot")
    plt.figure(figsize=(20, 10), dpi=150)  # Adjust figure size and resolution
    shap.plots.force(explainer.expected_value, shap_values.values[0, :], feature_names=X_train.columns, matplotlib=True)
    plt.savefig('shap_force_plot.png', bbox_inches='tight')  # Save the plot
    plt.close()  # Close the plot explicitly to prevent display issues in Streamlit
    st.image('shap_force_plot.png', caption='SHAP Force Plot', use_column_width=True)  # Display in Streamlit

    # Summary Plot
    st.write("Summary Plot")
    plt.figure(figsize=(20, 10), dpi=150)  # Same adjustments for other plots
    shap.summary_plot(shap_values.values, X_train, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # Dependence Plot for a specific feature
    st.write("Dependence Plot")
    feature_index = st.slider(
        label="Select the feature index for SHAP Dependence Plot",
        min_value=0,
        max_value=len(X_train.columns) - 1,
        value=0  # default value to display
    )
    st.write(f"Showing SHAP Dependence Plot for: {X_train.columns[feature_index]}")
    plt.figure(figsize=(20, 10), dpi=150)  # Adjust figure size and resolution
    shap.dependence_plot(X_train.columns[feature_index], shap_values.values, X_train, show=False)
    st.pyplot(plt.gcf())  # Display the plot in Streamlit
    plt.clf() 



    
def fetch_adjusted_close(tickers, start_date, end_date):
    """
    Objective: Fetches adjusted close prices for specified tickers within a date range.
    Args:
        tickers (list of str): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices for given tickers.
    """
    start_date = pd.to_datetime(start_date) - timedelta(days=1)
    end_date = pd.to_datetime(end_date) + timedelta(days=1)

    data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if 'Adj Close' in data.columns.levels[0]:
        adj_close_prices = data['Adj Close']
    else:
        adj_close_prices = data
    adj_close_prices.reset_index(inplace=True)
    
    return adj_close_prices



def calculate_returns(df):
    """
    Calculates percentage changes in prices from one day to the next across specified columns.
    Parameters:
        df (pd.DataFrame): DataFrame with a 'Date' column and one or more price columns.
    Returns:
        pd.DataFrame: DataFrame containing the date and percentage changes.
    """
    df2 = pd.DataFrame(df['Date'])
    for column in df.columns:
        if column != 'Date':
            df2[column] = df[column].pct_change()*100
    df2.dropna(inplace=True)  # Removing the first row that always contains NaN due to pct_change
    return df2



def Factor_download(tickers, start_date, end_date):
  """
    Fetches adjusted close prices for specified tickers within a date range using Yahoo Finance.
    Args:
        tickers (list of str): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing the adjusted close prices for the given tickers.
    """
  factor_prices = fetch_adjusted_close(tickers, start_date, end_date)

  # Filling missing data
  factor_prices['^GSPC'].ffill(inplace=True)
  factor_prices['^GSPC'].bfill(inplace=True)

  factor_prices.reset_index(drop=True, inplace=True)
  
  # Calculating returns
  factor_returns = calculate_returns(factor_prices)

  return factor_returns



def Factor_Data(tickers, crypto_df, result_df):
  """
    Integrates factor data with cryptocurrency and portfolio results.
    Args:
        tickers (list of str): Ticker symbols for factors.
        crypto_df (pd.DataFrame): DataFrame containing cryptocurrency data.
        result_df (pd.DataFrame): DataFrame containing portfolio returns.
    Returns:
        pd.DataFrame: Merged DataFrame with factors and daily portfolio returns.
    """
  factor_DDF = Factor_download(tickers, crypto_df['Date'].min().strftime('%Y-%m-%d'), crypto_df['Date'].max().strftime('%Y-%m-%d'))
  # Merging DataFrames
  if 'Date' not in factor_DDF.columns:
      factor_DDF.reset_index(inplace=True)

  if 'Date' not in result_df.columns:
      result_df.reset_index(inplace=True)
  selected_columns = ['Date'] + tickers  

  factor_table = pd.merge(factor_DDF[selected_columns], result_df[['Date', 'Daily_Portfolio_Return']], on='Date', how='left')
  return factor_table



def Factor_Analysis(df):
    
    Covariance_Analysis(df)

    Permutation_Importance(df)

    Shap_Importance(df)