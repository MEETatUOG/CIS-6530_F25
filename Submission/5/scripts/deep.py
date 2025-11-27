import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_PATHS = {
    "1gram": "../data/processed/opcodes_1gram.csv",
    "2gram": "../data/processed/opcodes_2gram.csv",
}
OUTPUT_DIR = "../data/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_model(vocab_size, sequence_len, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=sequence_len))
    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    for gram_label, data_path in DATA_PATHS.items():
        print(f"\n=== Deep Model on {gram_label} ===")
        df = pd.read_csv(data_path)

        # Convert each row of n-gram columns into a single sequence of tokens
        opcode_sequences = df.drop("label", axis=1).astype(str).agg(" ".join, axis=1)
        labels = df["label"].astype("category").cat.codes
        class_mapping = dict(enumerate(df["label"].astype("category").cat.categories))

        # Tokenize
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(opcode_sequences)
        sequences = tokenizer.texts_to_sequences(opcode_sequences)

        max_len = 500  # fixed-length sequence
        X = pad_sequences(sequences, maxlen=max_len, padding="post")
        y = np.array(labels)

        vocab_size = len(tokenizer.word_index) + 1
        num_classes = len(class_mapping)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build model
        model = build_model(vocab_size, max_len, num_classes)

        # Train
        model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=5,
            batch_size=64,
            verbose=1
        )

        # Evaluation
        y_pred = np.argmax(model.predict(X_test), axis=1)

        labels = sorted(set(y_test))
        target_names = [class_mapping[l] for l in labels]

        # Classification report
        print(classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=target_names
        )

        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Deep Model ({gram_label}) - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/DeepModel_{gram_label}_cm.png")
        plt.close()

        print(f"Saved Deep Model confusion matrix for {gram_label}\n")
