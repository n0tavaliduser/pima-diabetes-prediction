from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 21), output_path=None):
    """
    Mencari nilai k yang optimal untuk model k-NN berdasarkan akurasi.
    """
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        k_scores.append(accuracy_score(y_test, y_pred))

    optimal_k = k_range[np.argmax(k_scores)]
    
    if output_path:
        plt.figure(figsize=(12, 6))
        plt.plot(k_range, k_scores, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Akurasi vs. Nilai K untuk k-NN')
        plt.xlabel('Nilai K')
        plt.ylabel('Akurasi')
        plt.xticks(np.arange(min(k_range), max(k_range)+1, 1))
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    return optimal_k

def evaluate_model_split(X_train, X_test, y_train, y_test, model, model_name):
    """Mengevaluasi model dengan metode split."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

def evaluate_model_kfold(X, y, model, n_splits):
    """Mengevaluasi model dengan metode K-Fold."""
    scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
    return {"mean_accuracy": np.mean(scores)}

def run_experiments(X, y, config):
    """
    Menjalankan serangkaian eksperimen berdasarkan konfigurasi.
    """
    results = {"split_validation": {}, "k_fold_validation": {}}
    
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }

    # --- Split Validation Experiments ---
    for split_ratio in config['split_validation']['ratios']:
        test_size = split_ratio / 100
        split_name = f"{100-split_ratio}-{split_ratio}"
        results["split_validation"][split_name] = {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        for model_name, model in models.items():
            if model_name == "KNN":
                # Temukan k optimal untuk split saat ini
                optimal_k = find_optimal_k(X_train, y_train, X_test, y_test)
                model.set_params(n_neighbors=optimal_k)
            
            eval_metrics = evaluate_model_split(X_train, X_test, y_train, y_test, model, model_name)
            results["split_validation"][split_name][model_name] = eval_metrics

    # --- K-Fold Cross-Validation Experiments ---
    for k_fold in config['k_fold_validation']['folds']:
        fold_name = f"k={k_fold}"
        results["k_fold_validation"][fold_name] = {}
        
        for model_name, model in models.items():
            if model_name == "KNN":
                # Untuk K-Fold, kita bisa menggunakan k yang berbeda
                for k_val in config['k_fold_validation']['k_values_for_knn']:
                    model.set_params(n_neighbors=k_val)
                    k_fold_model_name = f"KNN (k={k_val})"
                    eval_metrics = evaluate_model_kfold(X, y, model, k_fold)
                    results["k_fold_validation"][fold_name][k_fold_model_name] = eval_metrics
            else:
                eval_metrics = evaluate_model_kfold(X, y, model, k_fold)
                results["k_fold_validation"][fold_name][model_name] = eval_metrics
                
    return results