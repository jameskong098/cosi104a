import pandas as pd
from sklearn.metrics import r2_score
from data_loader import load_training_data, load_test_data
from utils import split_data
from model_trainer import train_model, get_model_params

def main():
    train_df = load_training_data("HW1_training.csv")
    test_df = load_test_data("HW1_test_X.csv")

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(train_df, target_col="TARGET_deathRate")
    
    model = train_model(X_train, y_train)
    
    coefficients, intercept = get_model_params(model)
    print("\nCoefficients:", coefficients)
    print("Intercept:", intercept, "\n")

    # Predict on validation set
    y_val_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_val_pred)

    print("Validation R² Score (1 is Perfect):", r2, "\n")
    
    validation_predictions_df = pd.DataFrame({'TARGET_deathRate Predictions from Validation Set': y_val_pred})
    validation_predictions_df.to_csv('validation_predictions.csv', index=False)

    print("Validation predictions saved as validation_predictions.csv\n")

    # Prepare the test set by ensuring that the columns match the training data
    X_test = test_df[X_train.columns]  
    
    y_test_pred = model.predict(X_test)

    test_predictions_df = pd.DataFrame({'TARGET_deathRate Predictions from Test Set': y_test_pred})
    test_predictions_df.to_csv('test_predictions.csv', index=False)
    
    print("Test predictions saved to test_predictions.csv\n")

if __name__ == "__main__":
    main()
