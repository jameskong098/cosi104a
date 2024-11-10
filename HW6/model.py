import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score

def train_ensemble(X_train, y_train, use_full_dataset):
    """
    Trains an ensemble model with either cross-validation or a single train-test split.
    
    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    - use_full_dataset: Boolean flag to use train with full dataset and validate using cross-validation or partial dataset validating with single train-test split.
    
    Returns:
    - Trained ensemble model.
    """
    print("\nTraining Ensemble Model...")
    # Define individual models
    nn_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1))
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Combine models into an ensemble
    ensemble_model = VotingClassifier(estimators=[
        ('nn', nn_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ], voting='soft')
    
    if use_full_dataset:
        # Option 1: Using Cross-validation for validation and the full training set for training
        cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
        mean_cv_score = np.mean(cv_scores)
        print(f"\nCross-validation F1 scores: {cv_scores}")
        print(f"\nMean cross-validation F1: {mean_cv_score}\n")
        
        # Train the model on the full training set
        ensemble_model.fit(X_train, y_train)
    else:
        # Option 2: Using a single train-test split for validation and training
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Train the model on the training subset
        ensemble_model.fit(X_train_split, y_train_split)
        
        # Evaluate on the validation subset
        y_val_pred = ensemble_model.predict(X_val_split)
        val_f1 = f1_score(y_val_split, y_val_pred, average='macro')
        print(f"\nValidation F1 on single split: {val_f1}\n")  
    
    return ensemble_model
