#!/usr/bin/env python
"""
Train all three CEFR classifiers and save them as .pkl files in models/.
Run from the project root: python models/train.py
"""
import pickle
import sys
import time
from pathlib import Path

import nltk
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    build_word_features,
    get_feature_sets,
    load_from_file,
    to_flat_lists,
    to_tagged_documents,
)

MODELS_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'cefr_leveled_texts.csv'

NLTK_PKGS = ['punkt_tab', 'stopwords', 'brown', 'reuters', 'movie_reviews']


def _download_nltk():
    for pkg in NLTK_PKGS:
        nltk.download(pkg, quiet=True)


def train_lr(texts: list[str], labels: list[str]) -> None:
    print("Training Logistic Regression...")
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    vec = CountVectorizer(min_df=2, max_features=8000)
    clf = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    clf.fit(vec.fit_transform(X_train), y_train)
    preds = clf.predict(vec.transform(X_test))
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"  accuracy: {acc:.3f}  F1: {f1:.3f}  ({time.time()-t0:.1f}s)")
    joblib.dump(vec, MODELS_DIR / 'lr_vectorizer.pkl')
    joblib.dump(clf, MODELS_DIR / 'lr_model.pkl')
    print("  -> lr_vectorizer.pkl, lr_model.pkl")


def train_mnb(texts: list[str], labels: list[str]) -> None:
    print("Training Multinomial NB (TF-IDF)...")
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    vec = TfidfVectorizer(min_df=2, max_features=8000, sublinear_tf=True)
    clf = MultinomialNB()
    clf.fit(vec.fit_transform(X_train), y_train)
    preds = clf.predict(vec.transform(X_test))
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"  accuracy: {acc:.3f}  F1: {f1:.3f}  ({time.time()-t0:.1f}s)")
    joblib.dump(vec, MODELS_DIR / 'mnb_vectorizer.pkl')
    joblib.dump(clf, MODELS_DIR / 'mnb_model.pkl')
    print("  -> mnb_vectorizer.pkl, mnb_model.pkl")


def train_nltk_nb(dataset: dict) -> None:
    print("Building NLTK word features from corpora (may take a minute)...")
    t0 = time.time()
    word_features = build_word_features()
    print(f"  {len(word_features)} word features built ({time.time()-t0:.1f}s)")

    tagged_docs = to_tagged_documents(dataset)
    feature_sets = get_feature_sets(tagged_docs, word_features)

    n = len(feature_sets)
    split = int(n * 0.75)
    train_set, test_set = feature_sets[:split], feature_sets[split:]

    print(f"Training NLTK NB on {len(train_set)} samples...")
    t1 = time.time()
    import nltk as _nltk
    classifier = _nltk.NaiveBayesClassifier.train(train_set)

    preds = [classifier.classify(fs) for fs, _ in test_set]
    actual = [label for _, label in test_set]
    acc = accuracy_score(actual, preds)
    f1 = f1_score(actual, preds, average='weighted')
    print(f"  accuracy: {acc:.3f}  F1: {f1:.3f}  ({time.time()-t1:.1f}s)")

    with open(MODELS_DIR / 'nltk_nb.pkl', 'wb') as fh:
        pickle.dump({'classifier': classifier, 'word_features': word_features}, fh)
    print("  -> nltk_nb.pkl")


if __name__ == '__main__':
    _download_nltk()

    print(f"Loading dataset from {DATA_PATH}...\n")
    dataset = load_from_file(str(DATA_PATH))
    texts, labels = to_flat_lists(dataset)
    level_counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    print(f"Dataset: {len(texts)} texts — {level_counts}\n")

    train_lr(texts, labels)
    print()
    train_mnb(texts, labels)
    print()
    train_nltk_nb(dataset)

    print("\nAll models saved to models/")
