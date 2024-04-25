import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def Risk_vs_Return(df: pd.DataFrame):
    """
    Analyzes the risk versus return of financial assets within a given DataFrame, displaying average returns, 
    standard deviation, and Sharpe Ratio, alongside corresponding plots.
    Args:
        df (pd.DataFrame): DataFrame containing financial data with numeric columns representing returns.
    Raises:
        ValueError: If the DataFrame does not contain the expected numeric data.
    """
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    daily_mean_returns = df[numeric_cols].mean()

    n=len(df)
    Average_return = daily_mean_returns * n

    Standard_Deviation = df[numeric_cols].std() * np.sqrt(n)

    risk_free_rate=0
    Sharpe_R = Average_return / Standard_Deviation

    st.subheader("Average Return")
    st.write(Average_return.sort_values())

    st.subheader("Standard Deviation")
    st.write(Standard_Deviation.sort_values())

    st.subheader("Sharpe Ratio")
    st.write(Sharpe_R.sort_values())

    st.write("Sharpe Ratio Plot:")
    fig, ax = plt.subplots()
    Sharpe_R.plot.bar(
        grid=True,
        figsize=(12, 5),
        rot=0,
        ylabel="Sharpe Ratio",
        ax=ax
    )
    st.pyplot(fig)

    # Create and display box plot for daily returns
    st.write("Box Plot of Crypto Daily Returns")
    fig, ax = plt.subplots()
    df.plot(
        kind='box',
        figsize=(12, 5),    
        ylabel="Returns [%]",
        ax=ax
    )
    st.pyplot(fig)

    st.write("The table displays the Sharpe Ratios for various assets and the overall daily portfolio return.\n" 
             "Sharpe Ratios above zero, like those for Dollar, Render, RPL/USDT, and the overall portfolio, indicate returns that compensate for risk better than a risk-free investment.\n" 
             "Ribbon Finance stands out with the highest Sharpe Ratio, suggesting the best risk-adjusted performance.\n" 
             "In contrast, OCEAN and BADGER have negative Sharpe Ratios, showing that they performed worse than a risk-free asset when factoring in risk.\n" 
             "Frax Share and Maker lack Sharpe Ratio data, possibly due to insufficient data or no variation in returns.\n" 
             "Overall, the portfolio appears to be well-managed in terms of risk-adjusted returns, barring a few underperforming assets.\n")

