import pandas as pd

def load_data(file):
    """
    Loads data from a CSV file.

    Parameters:
    - file (str): Path to the CSV file containing the portfolio data.

    Returns:
    - DataFrame: Data with dates as index.
    """
    data = pd.read_csv(file, index_col=0)
    return data
