import pandas as pd
import re
from typing import Dict
def read_excel_sheets(excel_path: str) -> Dict[str, pd.DataFrame]:
    """Read all sheets from an Excel file into a dictionary of DataFrames."""
    try:
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        dataframes = {sheet: excel_file.parse(sheet) for sheet in sheet_names}
        return dataframes
    except Exception as e:
        print(f"Failed to read excel file: {e}")
        return {}

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by ensuring the 'Date' column is properly named and reordering columns."""
    if df.iloc[0, 0] == "Date":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    columns = [col for col in df.columns if "Unnamed" not in col]
    columns.remove('Dollar')
    columns.append('Dollar')
    return df[columns]

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe by dropping NA columns, filling missing values, and cleaning data."""
    df = df.iloc[1:]  # Remove the first row which may contain incorrect headers
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)
    df = clean_data(df)
    df.sort_values(by=['Date'], inplace=True)
    return df

def convert_monetary_values(x):
    """Convert monetary values from strings to floats with M and K multipliers, handles non-string types."""
    if isinstance(x, str):
        if 'M' in x:
            return float(x.replace('M', '')) * 1e6
        elif 'K' in x:
            return float(x.replace('K', '')) * 1e3
        return float(x)
    return x

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert data types in the dataframe."""
    for column in df.columns:
        if column == "Date":
            df[column] = df[column].apply(lambda x: re.sub(r'\-\> ', '', x) if isinstance(x, str) else x)
            df[column] = pd.to_datetime(df[column], errors='coerce')
        else:
            # Ensuring the column is treated as a string only if it's actually a string type
            if df[column].dtype == object:
                df[column] = df[column].apply(lambda x: re.sub(r'\$|\,', '', x) if isinstance(x, str) else x)
                df[column] = df[column].apply(convert_monetary_values)
            df[column] = df[column].astype(float)  # Convert to float for uniformity in numeric processing
    return df