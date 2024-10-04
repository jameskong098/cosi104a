import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split

def train_svm(X_train, y_train, use_cross_val=True):
    """
    Trains the SVM model with either cross-validation or a single train-test split.
    
    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    - use_cross_val: Boolean flag to use cross-validation or single train-test split.
    
    Returns:
    - Trained SVM model.
    """
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    
    if use_cross_val:
        # Option 1: Using Cross-validation
        cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
        print(f"\nCross-validation accuracy scores: {cv_scores}")
        print(f"\nMean cross-validation accuracy: {np.mean(cv_scores)}\n")
        
        # Train the model on the full training set
        svm_model.fit(X_train, y_train)
    
    else:
        # Option 2: Using a single train-test split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Train the model on the training subset
        svm_model.fit(X_train_split, y_train_split)
        
        # Evaluate on the validation subset
        val_accuracy = svm_model.score(X_val_split, y_val_split)
        print(f"\nValidation accuracy on single split: {val_accuracy}\n")
    
    return svm_model
