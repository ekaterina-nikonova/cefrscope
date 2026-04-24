#### The task

Automatically assigning a CEFR level to an English text is useful for any language learner,
but it is a genuinely hard multi-class problem. Adjacent levels (e.g. B1 vs B2) share overlapping vocabulary
and grammar, and human experts frequently disagree on the boundary between them.
The best published results using deep-learning approaches reach around 69% F1;
bag-of-words statistical models typically land closer to 50–60%.

The random baseline for a balanced 6-class problem is **16.7%** — so 50–55%
represents a meaningful signal, not random noise.

---

#### Dataset

The [CEFR-Levelled Texts dataset](https://www.kaggle.com/datasets/thedevastator/cefr-english-level-corpus)
contains 1,494 English texts labelled A1–C2, assembled from British Council
(*Learn English*), Trinity College London (ISE), and Oxford University Press practice
materials. Some samples were automatically labelled using the Text Inspector lexical
profiler; this introduces label noise at the margins.

The dataset is **balanced to 202 samples per level** (1,212 total) before training,
so accuracy and F1-score are directly comparable across classes.

---

#### Preprocessing

English stopwords are removed from every text before vectorization. Stopwords
(*the*, *a*, *is*, …) carry no class signal and add noise to frequency-based
representations.

Three vectorization strategies were evaluated:

| Representation | Used for | Key property |
|---|---|---|
| Boolean word presence | NLTK Naive Bayes | 8,950 features from Brown + Reuters + Movie Reviews corpora |
| Count Vectorizer (BoW) | Logistic Regression baseline | Raw term frequency; fast, interpretable |
| TF-IDF Vectorizer | Multinomial NB | Down-weights common terms; rewards rare, discriminative words |

---

#### Model comparison

Five models were trained and evaluated on a stratified 75/25 train/test split:

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression (Count BoW) | 51.5% | 50.8% |
| NLTK Naive Bayes (boolean features) | 55.4% | 54.0% |
| Multinomial NB — Count Vectorizer | ~49% | ~48% |
| **Multinomial NB — TF-IDF** | **49.5%** | **49.3%** |
| Feed-forward neural network (Word2Vec) | ~44% | ~43% |

The **TF-IDF vectorizer outperformed Count Vectorizer** for Multinomial NB, consistent
with the expectation that rarer, level-specific vocabulary is more informative than
raw frequency. CEFR texts share many common words across levels; TF-IDF suppresses
those and amplifies the discriminative ones.

The **neural network fell below the logistic regression baseline**. The Word2Vec
embeddings were trained on the same small 1,212-sample dataset, producing poor
vector representations. A model that needs to learn both a good embedding space
*and* a classifier from 909 training examples (75% of 1,212) will overfit before it
generalises. Confusion matrices showed the same adjacent-level error pattern as the
statistical models — more complexity, no benefit. The FFNN was excluded from the app.

---

#### Error pattern

All models showed the same misclassification structure: **errors concentrate between
neighbouring levels** (A1↔A2, B1↔B2, C1↔C2). Direct cross-level errors (A1 predicted
as C2, for example) were rare. This mirrors what is observed in human expert
annotation, where adjacent levels share overlapping linguistic features and even
trained assessors regularly disagree on the exact boundary.

The NLTK Naive Bayes most informative features — the words most strongly associated
with a specific level — reflected increasing lexical sophistication: concrete everyday
vocabulary at A1/A2 and more abstract, domain-specific terms at C1/C2.

---

#### Why three models in the app

Each model offers a different interpretability lens, and all three are fast enough to
run interactively without a GPU:

- **Logistic Regression** — coefficient magnitudes show which words *pushed* the
  prediction toward a level
- **Multinomial NB + TF-IDF** — TF-IDF weight shows which words were *most distinctive
  in this specific text*
- **NLTK Naive Bayes** — likelihood ratio shows which words are the *strongest signal*
  for the predicted level relative to the next-best candidate

Showing three independent models side by side also lets you see when they agree
(high confidence) versus diverge (ambiguous text, likely on a level boundary).
