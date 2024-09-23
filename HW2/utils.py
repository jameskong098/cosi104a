from sklearn.model_selection import train_test_split

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits data into training and validation sets.
    
    Args:
        df: pd.DataFrame, the dataset.
        target_col: str, the name of the target column.
        test_size: float, the proportion of the data to use as validation set.
    
    Returns:
        X_train, X_val, y_train, y_val: split features and target sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val
