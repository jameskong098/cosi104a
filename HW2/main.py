import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from data_loader import load_training_data, load_test_data
from utils import split_data
from model_trainer import train_model, get_model_params

def main():
    train_df = load_training_data("HW1_training.csv")
    test_df = load_test_data("HW1_test_X.csv")
    
    target_columns = ['avgAnnCount', 'avgDeathsPerYear', 'TARGET_deathRate', 'incidenceRate']
    
    # Get the common input features (excluding target columns)
    common_features = test_df.columns.tolist()

    validation_predictions = pd.DataFrame()
    test_predictions = pd.DataFrame()

    # Train a separate model for each target column and generate predictions
    for target_col in target_columns:
        print(f"\nTraining model for {target_col}...")

        # Use only common features for training to avoid column mismatch
        X_train, X_val, y_train, y_val = split_data(train_df[common_features + [target_col]], target_col=target_col)

        scaler = StandardScaler()

        # Standardize features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = train_model(X_train_scaled, y_train)
        
        coefficients, intercept = get_model_params(model)
        print(f"\nCoefficients for {target_col}:", coefficients)
        print(f"Intercept for {target_col}:", intercept, "\n")

        y_val_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_val_pred)
        print(f"Validation R² Score for {target_col} (1 is Perfect): {r2}\n")
        print(f"Data Fit: {r2*100:.2f}%\n")
        
        validation_predictions[target_col] = y_val_pred

        X_test = test_df[common_features]
        X_test_scaled = scaler.transform(X_test)  

        y_test_pred = model.predict(X_test_scaled)
        test_predictions[target_col] = y_test_pred

        print("===================\n")

    validation_predictions.to_csv('validation_predictions.csv', index=False)
    print("Validation predictions saved to validation_predictions.csv.\n")

    # Ensure correct order for target columns
    test_predictions = test_predictions[target_columns]

    test_predictions.to_csv('HW1_test_Y.csv', index=False)
    print("Test predictions saved to HW1_test_Y.csv.\n")

if __name__ == "__main__":
    main()
