import pandas as pd

def load_data(train_file, test_file):
    """
    Loads training and testing data from CSV files.

    Parameters:
    - train_file (str): Path to the CSV file containing the training data, including labels.
    - test_file (str): Path to the CSV file containing the testing data.

    Returns:
    - tuple: A tuple containing:
        - X_train (DataFrame): Features for the training set.
        - y_train (Series): Labels for the training set.
        - X_test (DataFrame): Features for the testing set.
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    
    X_test = test_data
    
    return X_train, y_train, X_test
