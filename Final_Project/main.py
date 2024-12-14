from data_loader import load_data, prepare_test_data
from pre_processor import preprocess_data
from model import train_model, quadratic_weighted_kappa

def main():
    # Load and preprocess the data
    train_data, test_data, sample_submission = load_data()
    X, y, common_columns, scaler, pca = preprocess_data(train_data, test_data)

    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    print(f"Training set distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set distribution:\n{y_test.value_counts(normalize=True)}")

    # Train the model
    voting_classifier = train_model(X_train, y_train)

    # Evaluate the model
    ensemble_preds = voting_classifier.predict(X_test)
    qwk_score = quadratic_weighted_kappa(y_test.tolist(), ensemble_preds, num_ratings=4)
    print(f"Quadratic Weighted Kappa Score: {qwk_score:.4f}")

    # Prepare and save predictions for test data
    X_test_data = prepare_test_data(test_data, common_columns)
    X_test_scaled = scaler.transform(X_test_data)
    X_test_pca = pca.transform(X_test_scaled)

    ensemble_preds = voting_classifier.predict(X_test_pca)
    submission = sample_submission.copy()
    submission['sii'] = ensemble_preds
    submission.to_csv('submission.csv', index=False)
    print(submission.head())

    
if __name__ == "__main__":
    main()