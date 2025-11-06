from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 21), output_path=None):
    """
    Mencari nilai k yang optimal untuk model k-NN berdasarkan akurasi.

    Parameters:
    X_train, y_train, X_test, y_test: Data pelatihan dan pengujian.
    k_range (range): Rentang nilai k yang akan diuji.
    output_path (str, optional): Path untuk menyimpan grafik. Jika None, grafik akan ditampilkan.

    Returns:
    int: Nilai k yang optimal.
    """
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        k_scores.append(accuracy_score(y_test, y_pred))

    optimal_k = k_range[np.argmax(k_scores)]
    
    # Plotting hasil
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, k_scores, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Akurasi vs. Nilai K untuk k-NN')
    plt.xlabel('Nilai K')
    plt.ylabel('Akurasi')
    plt.xticks(np.arange(min(k_range), max(k_range)+1, 1))
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Grafik pencarian K optimal disimpan di: {output_path}")
    else:
        plt.show()

    print(f"Nilai K optimal ditemukan: {optimal_k} dengan akurasi {max(k_scores):.4f}")
    return optimal_k

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