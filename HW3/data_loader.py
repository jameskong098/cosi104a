import pandas as pd

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train = train_data.drop('Label', axis=1)
    y_train = train_data['Label']
    
    X_test = test_data
    
    return X_train, y_train, X_test
