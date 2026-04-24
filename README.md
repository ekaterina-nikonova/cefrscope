# CEFRscope

A web app that classifies English text by [CEFR level](https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions) (A1–C2) using three statistical NLP models. For each prediction, the words that most influenced the result are shown, making the decision transparent rather than a black box.

---

## What it does

Paste any English text — a reading passage, essay excerpt, news article, email — and the app applies three classifiers simultaneously. Each classifier outputs:

- the predicted CEFR level with confidence
- a probability bar chart across all six levels
- the words in your text that contributed most to the prediction

The models are trained on ~1,200 balanced samples (A1–C2) from British Council, Trinity College London, and Oxford University Press CEFR-graded materials. They work best on **full paragraphs** (40+ words); the UI warns you if your text is shorter.

---

## Showing the results

When the classification is complete, the app displays:

**Level badge** — predicted level, its plain-English description, and the model's confidence score.

**Confidence per level** — bar chart showing how the model distributes probability across all six levels. A flat distribution means the model is uncertain; a tall single bar means it is confident. Low overall confidence (all bars roughly equal) often signals adjacent-level ambiguity, which is a known property of CEFR classification.

**Top words driving this prediction** — horizontal bar chart of the words in your text that pushed the model toward the predicted level, with method-specific scores:

| Model | Score shown |
|---|---|
| Logistic Regression | LR coefficient for the predicted class |
| Multinomial NB | TF-IDF weight of the word in your text |
| NLTK Naive Bayes | Likelihood ratio vs. next-best class |

**About the models** (collapsible) — accuracy, F1, and a description of each model, plus a note on evaluation methodology and the adjacent-level confusion pattern.

---

## Models

| Model | Accuracy | F1 | Interpretable |
|---|---|---|---|
| Logistic Regression | 51.5% | 50.8% | Yes — coefficients |
| Multinomial NB + TF-IDF | 49.5% | 49.3% | Yes — TF-IDF × log-prob |
| NLTK Naive Bayes | 55.4% | 54.0% | Yes — likelihood ratios |

Evaluated on a stratified 25% held-out split. All models confuse adjacent levels most often, consistent with research showing expert disagreement on neighbouring CEFR levels. A feed-forward neural network was also trained but excluded from the app: it failed to beat the baseline and relies on a heavy TensorFlow dependency.

---

## Run locally

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd cefrscope

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Train the models (one-time, ~30 seconds)
.venv/Scripts/python models/train.py   # Windows
# or
.venv/bin/python models/train.py       # macOS / Linux

# Start the app
.venv/Scripts/streamlit run app.py     # Windows
# or
.venv/bin/streamlit run app.py         # macOS / Linux
```

The app will open at `http://localhost:8501`.

The training script downloads the required NLTK corpora (Brown, Reuters, Movie Reviews, punkt, stopwords) automatically on first run.

---

## Deploy to Streamlit Community Cloud

The app can be deployed for free with no server setup required.

1. Push this repository to GitHub (make sure the `models/*.pkl` files are committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs dependencies from `requirements.txt` and starts the app.

NLTK data (`punkt_tab`, `stopwords`) is downloaded automatically at startup via `nltk.download(..., quiet=True)` in `app.py`.

---

## Project structure

```
cefrscope/
├── app.py                    # Streamlit UI
├── src/
│   ├── preprocessing.py      # Data loading, stopword removal, feature extraction
│   └── classifiers.py        # Model wrappers: LRClassifier, MNBClassifier, NLTKNBClassifier
├── models/
│   ├── train.py              # Training script — produces the .pkl files
│   ├── lr_model.pkl          # Trained LogisticRegression
│   ├── lr_vectorizer.pkl     # Fitted CountVectorizer
│   ├── mnb_model.pkl         # Trained MultinomialNB
│   ├── mnb_vectorizer.pkl    # Fitted TfidfVectorizer
│   └── nltk_nb.pkl           # NLTK NaiveBayesClassifier + word features
├── data/
│   └── cefr_leveled_texts.csv
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## Dataset

[CEFR-Levelled Texts](https://www.kaggle.com/datasets/thedevastator/cefr-english-level-corpus) — 17,000+ texts labelled A1–C2, assembled from British Council (Learn English / Learn English Teens), Trinity College London (ISE), and Oxford University Press practice materials. The dataset is balanced to 202 samples per level (1,212 total) before training.

---

## Limitations

- **Short texts:** Models were trained on reading passages. Isolated sentences or very short texts (< 40 words) may produce unreliable results.
- **Accuracy:** ~50–55% on balanced 6-class classification (random baseline is 16.7%). Adjacent-level errors are most common.
- **Domain:** Optimised for general-purpose English prose. Highly specialised or non-standard texts (code, social media, dialect) may not classify well.
