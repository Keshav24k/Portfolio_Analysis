import os
import pandas as pd
import re
from typing import Dict,Tuple
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.Portfolio_Analysis.components.Risk_Analysis  import *
from src.Portfolio_Analysis.components.Factor_Analysis  import *
from src.Portfolio_Analysis.components.Data_Processing  import *
from src.Portfolio_Analysis.components.Data_Ingestion  import *


def main():
    # Define the path to the dataset
    file_path = 'Data/Quant Dev Home Assignment Dataset.xlsx'
    
    # Read data from multiple sheets of an Excel file into separate DataFrames
    dataframes = read_excel_sheets(file_path)

    # Check if the file exists
    if os.path.exists(file_path):
        print("File exists.")
    else:   
        print("File does not exist.")

    # Process position history data if available
    print("--------------------   Stage 1    --------------------")
    if 'position history' in dataframes:
      crypto_df = dataframes['position history'].copy()
      crypto_df = clean_column_names(crypto_df)
      columns = ', '.join(crypto_df.columns)      
      crypto_df = preprocess_dataframe(crypto_df)
      crypto_df_working = add_total_pv_column(crypto_df)
      print("__________FileIngested - Dataframe Created__________")

    st.title("Crypto Data Showcase")

    # Display the column names
    columns = ', '.join(crypto_df.columns)
    st.text(f"Cryptos used: {columns}")
    
    # Slider to select the number of rows to display
    print("--------------------   Stage 2    --------------------")
    num_rows = st.slider('Select number of rows to display:', min_value=1, max_value=len(crypto_df), value=3)
    st.write(f"Displaying {num_rows} rows of data:")
    # Display DataFrame using st.dataframe or st.write
    st.dataframe(crypto_df.head(num_rows))

    print("_________Merging - Portfolio & Transaction____________")
    
    if 'transaction history' in dataframes:
      transaction_df = dataframes['transaction history'].copy()
      complete_df = Combining_sheets(crypto_df_working, transaction_df)

    print("__________Stage:data __________")
    result_df, DR_DF, Weight_df,tot_ret_df = process_financial_data(complete_df, crypto_df)
    print("-------------------------------------------------********************-----------------------------------")
    print(DR_DF.head(),Weight_df.head())

    returns_plot(tot_ret_df)
    Cumulative_returns_Plot(result_df)
    Weights_plot(Weight_df)


    print("--------------------   Stage 3    --------------------")
    print("__________Stage:Factordata __________")
    factor_df = Factor_Data(['BTC-USD','^GSPC','ETH-USD'], crypto_df, result_df)



    print("--------------------   Stage 4    --------------------")
    print("__________Stage:Risk __________")
    Risk_vs_Return(result_df)

    print("__________Stage:Factor Analysis __________")
    Factor_Analysis(factor_df)

if __name__ == "__main__":
    main()