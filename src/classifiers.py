import pickle
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import nltk
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack

from .preprocessing import compute_readability

LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


@dataclass
class Prediction:
    label: str
    probabilities: dict[str, float]
    top_features: list[tuple[str, float]] = field(default_factory=list)
    feature_description: str = ""


def _token_list(text: str) -> list[str]:
    """Return ordered, lowercase, alpha-only, non-stopword tokens."""
    stop_words = set(stopwords.words('english'))
    return [
        w.lower() for w in nltk.word_tokenize(text)
        if w.isalpha() and w.lower() not in stop_words
    ]


def _preprocess(text: str) -> str:
    return ' '.join(_token_list(text))


class LRClassifier:
    name = "Logistic Regression"
    description = (
        "Baseline model: bag-of-words + logistic regression. "
        "Coefficients show which words push the prediction toward the identified level."
    )
    accuracy = 0.541
    f1 = 0.534

    def __init__(self) -> None:
        self._vec = joblib.load(MODELS_DIR / 'lr_vectorizer.pkl')
        self._clf = joblib.load(MODELS_DIR / 'lr_model.pkl')
        self._classes: list[str] = [str(c) for c in self._clf.classes_]

    def predict(self, text: str) -> Prediction:
        processed = _preprocess(text)
        X_bow = self._vec.transform([processed])
        X_read = csr_matrix(np.array([compute_readability(text)]))
        vec = hstack([X_bow, X_read])
        label = str(self._clf.predict(vec)[0])
        proba = self._clf.predict_proba(vec)[0]
        probs = {str(c): float(p) for c, p in zip(self._clf.classes_, proba)}

        class_idx = self._classes.index(label)
        feat_names = self._vec.get_feature_names_out()
        n_vocab = len(feat_names)
        coefs = self._clf.coef_[class_idx]
        text_arr = X_bow.toarray()[0]

        present = text_arr > 0
        if present.any():
            masked = np.where(present, coefs[:n_vocab], -np.inf)
            top_idx = list(np.argsort(masked)[-10:][::-1])
            top_idx = [i for i in top_idx if present[i] and coefs[i] > 0]
        else:
            top_idx = list(np.argsort(coefs[:n_vocab])[-10:][::-1])

        top_features = [(feat_names[i], float(coefs[i])) for i in top_idx]

        return Prediction(
            label=label,
            probabilities=probs,
            top_features=top_features,
            feature_description="Logistic regression coefficient for the predicted level",
        )


class MNBClassifier:
    name = "Multinomial Naive Bayes"
    description = (
        "Best-performing model: TF-IDF vectorization + Multinomial NB. "
        "Feature scores reflect how much each word's TF-IDF weight contributed to the prediction."
    )
    accuracy = 0.551
    f1 = 0.548

    def __init__(self) -> None:
        self._vec = joblib.load(MODELS_DIR / 'mnb_vectorizer.pkl')
        self._clf = joblib.load(MODELS_DIR / 'mnb_model.pkl')
        self._classes: list[str] = [str(c) for c in self._clf.classes_]

    def predict(self, text: str) -> Prediction:
        processed = _preprocess(text)
        tfidf = self._vec.transform([processed])
        label = str(self._clf.predict(tfidf)[0])
        proba = self._clf.predict_proba(tfidf)[0]
        probs = {str(c): float(p) for c, p in zip(self._clf.classes_, proba)}

        class_idx = self._classes.index(label)
        feat_names = self._vec.get_feature_names_out()
        log_probs = self._clf.feature_log_prob_[class_idx]
        tfidf_arr = tfidf.toarray()[0]

        contribution = log_probs * tfidf_arr
        present = tfidf_arr > 0
        if present.any():
            masked = np.where(present, contribution, -np.inf)
            top_idx = list(np.argsort(masked)[-10:][::-1])
            top_idx = [i for i in top_idx if present[i]]
        else:
            top_idx = list(np.argsort(contribution)[-10:][::-1])

        top_features = [(feat_names[i], float(tfidf_arr[i])) for i in top_idx]


        return Prediction(
            label=label,
            probabilities=probs,
            top_features=top_features,
            feature_description="TF-IDF weight in input (term importance relative to corpus)",
        )


class NLTKNBClassifier:
    name = "NLTK Naive Bayes"
    description = (
        "Most explainable model: word-presence features from NLTK corpora. "
        "Shows which words in your text most strongly indicate the predicted level "
        "compared to the next-best class."
    )
    accuracy = 0.561
    f1 = 0.551

    def __init__(self) -> None:
        with open(MODELS_DIR / 'nltk_nb.pkl', 'rb') as fh:
            data = pickle.load(fh)
        self._clf = data['classifier']
        self._word_features: list[str] = data['word_features']

    def _make_features(self, text: str) -> dict[str, bool]:
        tokens = _token_list(text)
        word_set = set(tokens)
        bigram_set = {f"{a} {b}" for a, b in nltk.bigrams(tokens)}
        all_tokens = word_set | bigram_set
        return {f'contains({w})': (w in all_tokens) for w in self._word_features}

    def predict(self, text: str) -> Prediction:
        features = self._make_features(text)
        prob_dist = self._clf.prob_classify(features)
        label: str = prob_dist.max()

        all_labels = self._clf.labels()
        probs: dict[str, float] = {lbl: prob_dist.prob(lbl) for lbl in all_labels}
        for lbl in LEVELS:
            probs.setdefault(lbl, 0.0)

        other_labels = [lbl for lbl in all_labels if lbl != label]
        fd = self._clf._feature_probdist
        scored: list[tuple[str, float]] = []

        for w in self._word_features:
            fname = f'contains({w})'
            if not features.get(fname, False):
                continue
            key = (label, fname)
            if key not in fd:
                continue
            p_pred = fd[key].prob(True)
            others = [fd[(lbl, fname)].prob(True) for lbl in other_labels if (lbl, fname) in fd]
            if others and max(others) > 0 and p_pred > 0:
                ratio = p_pred / max(others)
                if ratio > 1.0:
                    scored.append((w, ratio))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            # No feature in the text strictly favours the predicted class over all others.
            # Fall back to showing the features that are present, ranked by P(feature|class).
            for w in self._word_features:
                fname = f'contains({w})'
                if not features.get(fname, False):
                    continue
                key = (label, fname)
                if key not in fd:
                    continue
                p_pred = fd[key].prob(True)
                if p_pred > 0:
                    scored.append((w, p_pred))
            scored.sort(key=lambda x: x[1], reverse=True)

        return Prediction(
            label=label,
            probabilities=probs,
            top_features=scored[:10],
            feature_description="Likelihood ratio vs. next-best class (higher = stronger signal)",
        )


def load_all() -> list:
    return [LRClassifier(), MNBClassifier(), NLTKNBClassifier()]
