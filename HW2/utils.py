from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

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


def save_model_params(target_col, coefficients, intercept):
    """
    Saves the model's coefficients and intercept to a CSV file.
    
    Args:
        target_col: str, the target column for which the model was trained.
        coefficients: list or array, the model's weights (coefficients).
        intercept: float, the model's bias (intercept).
    """
    params_df = pd.DataFrame({
        'Feature': ['Intercept'] + [f'Coefficient_{i+1}' for i in range(len(coefficients))],
        'Value': [intercept] + list(coefficients)
    })
    
    params_df.to_csv(f'trained_models/{target_col}_model_params.csv', index=False)
    print(f"Model parameters for {target_col} saved to {target_col}_model_params.csv.\n")


def save_trained_model(model, target_col):
    """
    Saves the trained model using pickle.
    
    Args:
        model: Trained sklearn model.
        target_col: str, the target column for which the model was trained.
    """

    model_filename = f'trained_models/{target_col}_linear_model.pkl'

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Trained model for {target_col} saved to {model_filename}.\n")
    