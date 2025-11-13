# classify.py
import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt

DATA_PATHS = {
    "1gram": "../data/processed/opcodes_1gram.csv",
    "2gram": "../data/processed/opcodes_2gram.csv",
}
OUTPUT_DIR = "../data/plots"

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_model(model, X_test, y_test, name, gram_label, class_labels):
    y_pred = model.predict(X_test)
    print(f"--- {name} ({gram_label}) ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', values_format='d')

    plt.title(f"{name} ({gram_label}) - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"{OUTPUT_DIR}/{name}_{gram_label}_cm.png"
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    models = {
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "DecisionTree": DecisionTreeClassifier()
    }

    for gram_label, data_path in DATA_PATHS.items():
        print(f"\n=== Processing {gram_label} data ===")
        df = pd.read_csv(data_path)
        X = df.drop('label', axis=1)
        y = df['label']

        class_labels = sorted(y.unique())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for name, model in models.items():
            model.fit(X_train, y_train)
            evaluate_model(model, X_test, y_test, name, gram_label, class_labels)

    print(f"\n✅ All confusion matrix plots saved to: {OUTPUT_DIR}")
