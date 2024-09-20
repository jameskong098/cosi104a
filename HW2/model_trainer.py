from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """
    Trains a linear regression model.
    
    Args:
        X_train: pd.DataFrame, training features.
        y_train: pd.Series, training target variable.
    
    Returns:
    - model: trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def get_model_params(model):
    """
    Returns the coefficients and intercept from a trained model.
    
    Args:
        model: trained linear regression model.
    
    Returns:
        coefficients, intercept: model parameters.
    """
    coefficients = model.coef_
    intercept = model.intercept_
    
    return coefficients, intercept
