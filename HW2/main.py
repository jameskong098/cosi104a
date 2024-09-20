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
    
    # Define the most relevant features for each target variable (adjust as needed based on R-squared score)
    feature_sets = {
        'avgAnnCount': ['medIncome', 'popEst2015', 'povertyPercent', 'MedianAge', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate', 'binnedInc1', 'binnedInc2'],
        'avgDeathsPerYear': ['popEst2015', 'povertyPercent', 'studyPerCap', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace'],
        'TARGET_deathRate': ['medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap', 'MedianAge', 'MedianAgeMale', 'MedianAgeFemale', 'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate', 'binnedInc1', 'binnedInc2'],
        'incidenceRate': ['studyPerCap', 'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack', 'PctAsian', 'PctOtherRace']
    }

    validation_predictions = pd.DataFrame()
    test_predictions = pd.DataFrame()

    # Train a separate model for each target column and generate predictions
    for target_col in target_columns:
        print(f"\nTraining model for {target_col}...")

        # Use only relevant features for the current target
        relevant_features = feature_sets[target_col]
        X_train, X_val, y_train, y_val = split_data(train_df[relevant_features + [target_col]], target_col=target_col)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = train_model(X_train_scaled, y_train)
        
        coefficients, intercept = get_model_params(model)
        print(f"\nCoefficients for {target_col}:", coefficients)
        print(f"Intercept for {target_col}:", intercept, "\n")

        y_val_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_val_pred)
        print(f"Validation R² Score for {target_col} (1 is Perfect): {r2:.4f}\n")
        print(f"Data Fit: {r2 * 100:.2f}%\n")
        
        validation_predictions[target_col] = y_val_pred

        X_test = test_df[relevant_features]
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
