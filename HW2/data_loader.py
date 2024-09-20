import pandas as pd
from sklearn.calibration import LabelEncoder

def load_training_data(file_path):
    """
    Loads the training data from a CSV file.
    
    Args:
        file_path: str, path to the training data CSV.
    
    Returns:
        df: pd.DataFrame, loaded DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean "binnedInc" column as it is a string of ranged values
    df[['binnedInc1', 'binnedInc2']] = df['binnedInc'].str.extract(r'\((.*), (.*)\]')
    df['binnedInc1'] = df['binnedInc1'].astype(float)
    df['binnedInc2'] = df['binnedInc2'].astype(float)
    df = df.drop(columns=['binnedInc'])
    
    # Drop the 'Geography' column since it's categorical and not useful for regression
    df = df.drop(columns=['Geography'])
    
    # Check for missing values and handle them
    df = df.dropna()  
    
    return df

def load_test_data(file_path):
    """
    Loads the test data from a CSV file.
    
    Args:
        file_path: str, path to the test data CSV.
    
    Returns:
        df: pd.DataFrame, loaded DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean "binnedInc" column as it is a string of ranged values
    df[['binnedInc1', 'binnedInc2']] = df['binnedInc'].str.extract(r'\((.*), (.*)\]')
    df['binnedInc1'] = df['binnedInc1'].astype(float)
    df['binnedInc2'] = df['binnedInc2'].astype(float)
    df = df.drop(columns=['binnedInc'])
    
    # Drop the 'Geography' column since it's categorical and not useful for regression
    df = df.drop(columns=['Geography'])

    # Check for missing values and handle them
    df = df.dropna()  
    
    return df
