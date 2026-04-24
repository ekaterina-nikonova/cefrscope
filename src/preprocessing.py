import csv
import random

import nltk
from nltk.corpus import stopwords

LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']


def ensure_nltk_data():
    for pkg in ['punkt_tab', 'stopwords']:
        nltk.download(pkg, quiet=True)


def remove_stopwords_from_text(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    return ' '.join(w for w in tokens if w.lower() not in stop_words)


def load_from_file(filepath: str) -> dict[str, list[str]]:
    dataset: dict[str, list[str]] = {label: [] for label in LEVELS}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile):
            if len(row) < 2:
                continue
            text, label = row[0], row[1].strip().upper()
            if label in LEVELS:
                dataset[label].append(remove_stopwords_from_text(text))
    min_count = min(len(texts) for texts in dataset.values())
    for label in dataset:
        dataset[label] = dataset[label][:min_count]
    return dataset


def to_flat_lists(dataset: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    pairs = [(t, lbl) for lbl, txts in dataset.items() for t in txts]
    random.shuffle(pairs)
    texts, labels = zip(*pairs)
    return list(texts), list(labels)


def to_tagged_documents(dataset: dict[str, list[str]]) -> list[tuple[list[str], str]]:
    docs = [
        ([w.lower() for w in nltk.word_tokenize(text)], label)
        for label, texts in dataset.items()
        for text in texts
    ]
    random.shuffle(docs)
    return docs


def build_word_features() -> list[str]:
    """Build the shared vocabulary used by the NLTK NB classifier.

    Requires brown, reuters, and movie_reviews corpora to be downloaded.
    Returns the frequency-ranked word list in positions 50–9000.
    """
    from nltk.corpus import brown, movie_reviews, reuters
    vocab = (
        [w.lower() for w in brown.words()]
        + [w.lower() for w in reuters.words()]
        + [w.lower() for w in movie_reviews.words()]
    )
    return list(nltk.FreqDist(vocab))[50:9000]


def make_feature_dict(doc: list[str], word_features: list[str]) -> dict[str, bool]:
    words = set(doc)
    return {f'contains({w})': (w in words) for w in word_features}


def get_feature_sets(
    documents: list[tuple[list[str], str]],
    word_features: list[str],
) -> list[tuple[dict[str, bool], str]]:
    return [(make_feature_dict(doc, word_features), label) for doc, label in documents]
