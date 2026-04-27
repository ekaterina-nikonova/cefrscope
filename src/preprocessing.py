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


def load_raw_from_file(filepath: str) -> dict[str, list[str]]:
    """Load texts without stopword removal, with the same balance as load_from_file."""
    dataset: dict[str, list[str]] = {label: [] for label in LEVELS}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile):
            if len(row) < 2:
                continue
            text, label = row[0], row[1].strip().upper()
            if label in LEVELS:
                dataset[label].append(text)
    min_count = min(len(texts) for texts in dataset.values())
    for label in dataset:
        dataset[label] = dataset[label][:min_count]
    return dataset


def to_flat_lists(dataset: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    pairs = [(t, lbl) for lbl, txts in dataset.items() for t in txts]
    random.shuffle(pairs)
    texts, labels = zip(*pairs)
    return list(texts), list(labels)


def to_flat_lists_paired(
    preprocessed: dict[str, list[str]],
    raw: dict[str, list[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle preprocessed and raw texts in tandem so their indices stay aligned."""
    pairs = [
        (prep_t, raw_t, lbl)
        for lbl in preprocessed
        for prep_t, raw_t in zip(preprocessed[lbl], raw[lbl])
    ]
    random.shuffle(pairs)
    prep_texts, raw_texts, labels = zip(*pairs)
    return list(prep_texts), list(raw_texts), list(labels)


def to_tagged_documents(dataset: dict[str, list[str]]) -> list[tuple[list[str], str]]:
    docs = [
        ([w.lower() for w in nltk.word_tokenize(text)], label)
        for label, texts in dataset.items()
        for text in texts
    ]
    random.shuffle(docs)
    return docs


def compute_readability(text: str) -> list[float]:
    """Return [avg_sentence_length, avg_word_length, type_token_ratio, long_word_ratio].

    Computed on raw (unprocessed) text using only nltk, which is already a dependency.
    """
    sentences = nltk.sent_tokenize(text)
    words = [w for w in nltk.word_tokenize(text) if w.isalpha()]
    if not words:
        return [0.0, 0.0, 0.0, 0.0]
    sent_lens = [
        len([w for w in nltk.word_tokenize(s) if w.isalpha()])
        for s in sentences
    ]
    avg_sent_len = sum(sent_lens) / len(sent_lens) if sent_lens else 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    ttr = len({w.lower() for w in words}) / len(words)
    long_ratio = sum(1 for w in words if len(w) >= 7) / len(words)
    return [avg_sent_len, avg_word_len, ttr, long_ratio]


def _corpus_bigrams(corpus) -> list[str]:
    stop_words = set(stopwords.words('english'))
    return [
        f"{a} {b}"
        for sent in corpus.sents()
        for a, b in nltk.bigrams(w.lower() for w in sent)
        if a.isalpha() and b.isalpha()
        and a not in stop_words and b not in stop_words
    ]


def build_word_features() -> list[str]:
    """Build the shared vocabulary used by the NLTK NB classifier.

    Requires brown, reuters, and movie_reviews corpora to be downloaded.
    Returns frequency-ranked unigrams (positions 50–9000) plus the top ~2000 bigrams
    (positions 50–2050 in their own FreqDist, skipping the most common stopword pairs).
    """
    from nltk.corpus import brown, movie_reviews, reuters

    unigram_vocab = (
        [w.lower() for w in brown.words() if w.isalpha()]
        + [w.lower() for w in reuters.words() if w.isalpha()]
        + [w.lower() for w in movie_reviews.words() if w.isalpha()]
    )
    unigram_freq = nltk.FreqDist(unigram_vocab)
    unigram_features = list(unigram_freq)[50:9000]

    bigram_vocab = (
        _corpus_bigrams(brown)
        + _corpus_bigrams(reuters)
        + _corpus_bigrams(movie_reviews)
    )
    bigram_freq = nltk.FreqDist(bigram_vocab)
    bigram_features = list(bigram_freq)[50:2050]

    return unigram_features + bigram_features


def make_feature_dict(doc: list[str], word_features: list[str]) -> dict[str, bool]:
    word_set = set(doc)
    bigram_set = {f"{a} {b}" for a, b in nltk.bigrams(doc)}
    all_tokens = word_set | bigram_set
    return {f'contains({w})': (w in all_tokens) for w in word_features}


def get_feature_sets(
    documents: list[tuple[list[str], str]],
    word_features: list[str],
) -> list[tuple[dict[str, bool], str]]:
    return [(make_feature_dict(doc, word_features), label) for doc, label in documents]
