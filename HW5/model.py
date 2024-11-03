import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_best_params_scores(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            best_params = eval(lines[0].strip())
            best_scores = eval(lines[1].strip())
            return best_params, best_scores
    except FileNotFoundError:
        return None, {'full_dataset': 0, 'partial_dataset': 0}

def save_best_params_scores(file_path, best_params, best_scores):
    with open(file_path, 'w') as file:
        file.write(f"{best_params}\n")
        file.write(f"{best_scores}\n")

def train_nn(X_train, y_train, use_full_dataset, force_retune):
    """
    Trains a neural network model with either cross-validation or a single train-test split.
    
    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    - use_cross_val: Boolean flag to use cross-validation or single train-test split.
    
    Returns:
    - Trained neural network model.
    """
    param_score_file = 'best_params_scores.txt'
    
    best_params, best_scores = load_best_params_scores(param_score_file)

    if best_params is None or force_retune:
        param_grid = {
            'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'mlpclassifier__activation': ['tanh', 'relu'],
            'mlpclassifier__solver': ['adam'],
            'mlpclassifier__alpha': [0.0001, 0.001],
            'mlpclassifier__learning_rate': ['constant'],
            'mlpclassifier__max_iter': [200, 400]
        }

        print("\nSearching for best parameters using GridSearchCV...")

        nn_model = make_pipeline(StandardScaler(), MLPClassifier())
        grid_search = GridSearchCV(nn_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        save_best_params_scores(param_score_file, best_params, best_scores)
    else:
        nn_model = make_pipeline(StandardScaler(), MLPClassifier(**{k.split('__')[1]: v for k, v in best_params.items()}))

    print("\nTraining neural network model...")  

    if use_full_dataset:
        # Option 1: Using Cross-validation for validation and the full training set for training
        cv_scores = cross_val_score(nn_model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_cv_score = np.mean(cv_scores)
        print(f"\nCross-validation AUC scores: {cv_scores}")
        print(f"\nMean cross-validation AUC: {mean_cv_score}\n")
        
        if mean_cv_score > best_scores['full_dataset']:
            print(f"New best full dataset score: {mean_cv_score} (previous: {best_scores['full_dataset']})")
            best_scores['full_dataset'] = mean_cv_score
            save_best_params_scores(param_score_file, best_params, best_scores)
        
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
        
        if val_auc > best_scores['partial_dataset']:
            print(f"New best partial dataset score: {val_auc} (previous: {best_scores['partial_dataset']})")
            best_scores['partial_dataset'] = val_auc
            save_best_params_scores(param_score_file, best_params, best_scores)

    print("\nValidation Best Scores:\n")
    print(f"Full dataset: {best_scores['full_dataset']}")
    print(f"Partial dataset: {best_scores['partial_dataset']}\n")
    
    return nn_model
