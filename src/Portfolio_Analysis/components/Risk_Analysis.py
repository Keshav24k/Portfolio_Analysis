import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def Risk_vs_Return(df):
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    daily_mean_returns = df[numeric_cols].mean()
    n=len(df)
    #print(df.mean())
    Average_return = daily_mean_returns * n
    #print("Average Return\n")
    #print(Average_return.sort_values())

    Standard_Deviation = df[numeric_cols].std() * np.sqrt(n)
    #print("Standard_Deviation\n")
    #print(Standard_Deviation.sort_values())

    risk_free_rate=0
    Sharpe_R = Average_return / Standard_Deviation
    #print("Sharpe_Ratio \n")
    #print(Sharpe_R.sort_values(),"\n")

    st.subheader("Average Return")
    st.write(Average_return.sort_values())

    st.subheader("Standard Deviation")
    st.write(Standard_Deviation.sort_values())

    st.subheader("Sharpe Ratio")
    st.write(Sharpe_R.sort_values())

    # Assuming 'Sharpe_R' is a DataFrame with the Sharpe Ratios
    # Sharpe_R.plot.bar(
    #     title="Sharpe Ratio",
    #     grid=True,
    #     figsize=(12,5),
    #     rot=0,
    #     ylabel="Sharpe Ratio"
    # )
    # plt.show()  # This ensures that the plot is shown and then the plotting area is cleared

    # print("\n\n")  # Just adding space between the plots for clarity

    # # Assuming 'df' is your DataFrame of daily returns
    # df.plot(
    #     kind='box',
    #     figsize=(12,5),
    #     title="Box Plot of Funds' Daily Returns",
    #     ylabel="Returns [%]"
    # )
    # plt.show()  # This will show the second plot separately

    # Create and display Sharpe Ratio bar plot
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

