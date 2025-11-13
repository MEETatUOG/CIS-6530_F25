# preprocess.py

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

RAW_DIR = "../data/raws"
PROCESSED_DIR = "../data/processed"

def infer_label_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    match = re.match(r'^(G\d{4})_([^_]+(?: [^_]+)*)_', base)
    if match:
        return match.group(2).strip()
    else:
        raise ValueError(f"Cannot infer label from filename: {filename}")

def extract_opcodes(filepath: str) -> str:
    try:
        df = pd.read_csv(filepath)
        if "opcode" not in df.columns:
            raise ValueError(f"Missing 'opcode' column in {filepath}")
        opcodes = df["opcode"].astype(str).tolist()
        return " ".join(opcodes)
    except Exception as e:
        print(f"[WARN] Skipping {filepath}: {e}")
        return ""

def load_dataset(raw_dir: str):
    texts, labels = [], []
    for file in os.listdir(raw_dir):
        if file.endswith(".opecode"):
            label = infer_label_from_filename(file)
            path = os.path.join(raw_dir, file)
            opcode_seq = extract_opcodes(path)
            if opcode_seq:
                texts.append(opcode_seq)
                labels.append(label)
    return texts, labels

def build_ngram_features(texts, n=1):
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(n, n))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer.get_feature_names_out()

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("[INFO] Loading .opecode files...")
    texts, labels = load_dataset(RAW_DIR)
    print(f"[INFO] Loaded {len(texts)} samples.")

    for n in [1, 2]:
        print(f"[INFO] Extracting {n}-gram features...")
        X, features = build_ngram_features(texts, n)
        df = pd.DataFrame(X.toarray(), columns=features)
        df["label"] = labels
        out_file = os.path.join(PROCESSED_DIR, f"opcodes_{n}gram.csv")
        df.to_csv(out_file, index=False)
        print(f"[INFO] Saved {out_file}")
