from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_params):
    """
    Melatih dan mengevaluasi model KNN, Decision Tree, dan Naive Bayes.

    Parameters:
    X_train, X_test, y_train, y_test: Data pelatihan dan pengujian.
    model_params (dict): Kamus yang berisi parameter untuk setiap model.

    Returns:
    dict: Kamus yang berisi metrik evaluasi untuk setiap model.
    """
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(**model_params.get("K-Nearest Neighbors", {})),
        "Decision Tree": DecisionTreeClassifier(**model_params.get("Decision Tree", {})),
        "Naive Bayes": GaussianNB(**model_params.get("Naive Bayes", {}))
    }

    results = {}

    for name, model in models.items():
        # Pelatihan model
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }

        # Visualisasi Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'output/{name.lower().replace(" ", "_")}_cm.png')
        plt.close()

    return results