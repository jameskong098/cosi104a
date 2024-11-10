import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

def train_ensemble(X_train, y_train, use_full_dataset):
    """
    Trains a Bagging ensemble model with either cross-validation or a single train-test split.
    
    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    - use_full_dataset: Boolean flag to use train with full dataset and validate using cross-validation or partial dataset validating with single train-test split.
    
    Returns:
    - Trained Bagging ensemble model.
    """
    print("\nTraining Bagging Ensemble Model...")
    
    # Define the base estimator
    base_estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Define the BaggingClassifier
    model = BaggingClassifier(estimator=base_estimator, n_estimators=50, random_state=42, n_jobs=-1)
    
    if use_full_dataset:
        # Option 1: Using Cross-validation for validation and the full training set for training
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
        mean_cv_score = np.mean(cv_scores)
        print(f"\nCross-validation F1 scores: {cv_scores}")
        print(f"\nMean cross-validation F1 Score: {mean_cv_score}\n")
        
        # Train the model on the full training set
        model.fit(X_train, y_train)
    else:
        # Option 2: Using a single train-test split for validation and training
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Train the model on the training subset
        model.fit(X_train_split, y_train_split)
        
        # Evaluate on the validation subset
        y_val_pred = model.predict(X_val_split)
        val_f1 = f1_score(y_val_split, y_val_pred, average='macro')
        print(f"\nValidation F1 on single split: {val_f1}\n")  
    
    return model
