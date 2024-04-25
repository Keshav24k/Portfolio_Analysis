import pandas as pd
import re
from typing import Dict

def read_excel_sheets(excel_path: str) -> Dict[str, pd.DataFrame]:
    """ Read all sheets from an Excel file into a dictionary of DataFrames.
    Args: excel_path (str): Path to the Excel file to be read (Currently stored in github)
    Returns: Dict[str, pd.DataFrame]: A dictionary where the keys are sheet names and values are the respective DataFrames.
    """
    try:
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        dataframes = {sheet: excel_file.parse(sheet) for sheet in sheet_names}
        return dataframes
    except Exception as e:
        print(f"Failed to read excel file: {e}")
        return {}

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Clean column names by ensuring the 'Date' column is properly named and reordering columns.
    Args: df (pd.DataFrame): DataFrame with potentially incorrect column names.
    Returns: pd.DataFrame: DataFrame with cleaned column names.
    """
    if df.iloc[0, 0] == "Date":                                       #Replace the first columns with Date
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)      
    columns = [col for col in df.columns if "Unnamed" not in col]     #remove the columns with "Unnamed"
    columns.remove('Dollar')                                          #reorder the columns positions  
    columns.append('Dollar')                                           
    return df[columns]

def convert_monetary_values(x):
    """
    Objective: Convert monetary values from strings to floats with M and K multipliers, handles non-string types.
    Args: x: A potential string with monetary values indicated by 'M' or 'K'.
    Returns: float: Converted monetary value as a float.
    """
    if isinstance(x, str):
        if 'M' in x:
            return float(x.replace('M', '')) * 1e6
        elif 'K' in x:
            return float(x.replace('K', '')) * 1e3
        return float(x)
    return x

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Clean and convert data types in the dataframe for analysis.
    Args: df (pd.DataFrame): DataFrame with raw data.
    Returns: pd.DataFrame: DataFrame with cleaned and standardized data types.
    """
    for column in df.columns:
        if column == "Date":
            df[column] = df[column].apply(lambda x: re.sub(r'\-\> ', '', x) if isinstance(x, str) else x)       # Date inconsistency - "-> " 
            df[column] = pd.to_datetime(df[column], errors='coerce')                                            # change the date
        else:
            # Ensuring the column is treated as a string only if it's actually a string type
            if df[column].dtype == object:
                df[column] = df[column].apply(lambda x: re.sub(r'\$|\,', '', x) if isinstance(x, str) else x)   # reges to remove the characters "," and "$"
                df[column] = df[column].apply(convert_monetary_values)                                          # to remove the sumbols "M" and "K"
            df[column] = df[column].astype(float)                                                               # Convert to float for uniformity in numeric processing
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Objective: Preprocess the dataframe by dropping NA columns, filling missing values, and cleaning data.
    Args: df (pd.DataFrame): Initial DataFrame to be preprocessed.
    Returns: pd.DataFrame: Preprocessed DataFrame.
    """
    df = df.iloc[1:]                                                  # Remove the first row which may contain incorrect headers
    df = df.dropna(axis=1, how='all')                                 # Dropping empty Columns
    df = df.fillna(0)                                                 #
    df = clean_data(df)                                               #Call the function for cleaning
    df.sort_values(by=['Date'], inplace=True)                         #Sort based on date - Ascending
    return df

