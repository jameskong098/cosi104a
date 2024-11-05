import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def train_nn(X_train, y_train, use_full_dataset):
    """
    Trains a neural network model with either cross-validation or a single train-test split.
    
    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    - use_cross_val: Boolean flag to use cross-validation or single train-test split.
    
    Returns:
    - Trained neural network model.
    """
    nn_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1))
    print("\nTraining neural network model...")  
    if use_full_dataset:
        # Option 1: Using Cross-validation for validation and the full training set for training
        cv_scores = cross_val_score(nn_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        mean_cv_score = np.mean(cv_scores)
        print(f"\nCross-validation AUC scores: {cv_scores}")
        print(f"\nMean cross-validation AUC: {mean_cv_score}\n")
        
        # Train the model on the full training set
        nn_model.fit(X_train, y_train)
    else:
        # Option 2: Using a single train-test split for validation and training
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Train the model on the training subset
        nn_model.fit(X_train_split, y_train_split)
        
        # Evaluate on the validation subset
        y_val_pred = nn_model.predict_proba(X_val_split)[:, 1]
        val_auc = roc_auc_score(y_val_split, y_val_pred)
        print(f"\nValidation AUC on single split: {val_auc}\n")  
    
    return nn_model
