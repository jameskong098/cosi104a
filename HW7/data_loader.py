import pandas as pd

def load_portfolio_data(portfolio_file):
    """
    Loads portfolio data from a CSV file.

    Parameters:
    - portfolio_file (str): Path to the CSV file containing the portfolio data.

    Returns:
    - DataFrame: Portfolio data with dates as index.
    """
    portfolio_data = pd.read_csv(portfolio_file, index_col=0)
    return portfolio_data
