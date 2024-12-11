import pandas as pd

def load_data(train_file, test_file):
    """
    Load the training and test data from CSV files.

    Parameters:
    - train_file (str): Path to the training data CSV file.
    - test_file (str): Path to the test data CSV file.

    Returns:
    - X_train (DataFrame): Features of the training data.
    - y_train (Series): Target variable of the training data.
    - X_test (DataFrame): Features of the test data.
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Drop rows with missing target values
    train_data = train_data.dropna(subset=['sii'])

    # Extract features and target variable
    X_train = train_data.drop(columns=['id', 'sii'])
    y_train = train_data['sii']
    X_test = test_data.drop(columns=['id'])

    test_ids = test_data['id']

    return X_train, y_train, X_test, test_ids
