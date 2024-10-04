import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def train_svm(X_train, y_train):
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    
    print(f"\nCross-validation accuracy scores: {cv_scores}")
    print(f"\nMean cross-validation accuracy: {np.mean(cv_scores)}\n")
    
    svm_model.fit(X_train, y_train)
    
    return svm_model
