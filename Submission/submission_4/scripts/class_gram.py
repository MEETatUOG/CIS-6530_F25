# classify_opcodes.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
 
# =========================
# Step 1: Load Processed Dataset
# =========================
processed_path = Path(r'/home/adamd/Downloads/CIS-6530_F25/Submission/4/data/processed/opcodes_1gram.csv')
df = pd.read_csv(processed_path)
 
# =========================
# Step 2: Define Classifiers
# =========================
classifiers = {
    'SVM': LinearSVC(max_iter=50000, dual=False),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}
 
# =========================
# Step 3: LOO-CV Evaluation Function
# =========================
def evaluate_loocv(vectorizer, ngram_name):
    print(f"\n===== Evaluating {ngram_name} Features =====")
   
    X = vectorizer.fit_transform(df['opcodes'])
    y = df['label'].values
    loo = LeaveOneOut()
   
    for name, clf in classifiers.items():
        y_true_all = []
        y_pred_all = []
       
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
           
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
           
            y_true_all.append(y_test[0])
            y_pred_all.append(y_pred[0])
       
        # =========================
        # Step 4: Metrics
        # =========================
        acc = accuracy_score(y_true_all, y_pred_all)
        print(f"\n{name} Accuracy ({ngram_name}): {acc*100:.2f}%")
        print(classification_report(y_true_all, y_pred_all, zero_division=0))
       
        # Confusion Matrix
        labels_unique = sorted(df['label'].unique())
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_unique)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_unique, yticklabels=labels_unique, cmap="Blues")
        plt.title(f"{name} Confusion Matrix ({ngram_name})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
 
# =========================
# Step 5: Evaluate 1-Gram
# =========================
vectorizer_1gram = TfidfVectorizer(ngram_range=(1,1))
evaluate_loocv(vectorizer_1gram, "1-Gram")
 
# =========================
# Step 6: Evaluate 2-Gram
# =========================
vectorizer_2gram = TfidfVectorizer(ngram_range=(2,2))
evaluate_loocv(vectorizer_2gram, "2-Gram")