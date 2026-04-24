import matplotlib
import matplotlib.pyplot as plt
import nltk
import streamlit as st

matplotlib.use('Agg')

st.set_page_config(page_title="CEFRscope", page_icon="🔬", layout="wide")

LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

LEVEL_META = {
    'A1': ('Beginner',          '#43A047'),
    'A2': ('Elementary',        '#7CB342'),
    'B1': ('Intermediate',      '#F9A825'),
    'B2': ('Upper Intermediate','#FB8C00'),
    'C1': ('Advanced',          '#E53935'),
    'C2': ('Mastery',           '#8E24AA'),
}


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    for pkg in ['punkt_tab', 'stopwords']:
        nltk.download(pkg, quiet=True)
    from src.classifiers import load_all
    return load_all()


def level_badge(label: str, confidence: float) -> str:
    color = LEVEL_META[label][1]
    desc = LEVEL_META[label][0]
    return (
        f'<div style="background:{color};padding:18px 12px;border-radius:10px;'
        f'text-align:center;margin-bottom:12px;">'
        f'<span style="color:#fff;font-size:2.4em;font-weight:700;line-height:1;">{label}</span><br>'
        f'<span style="color:rgba(255,255,255,.85);font-size:.9em;">{desc}</span><br>'
        f'<span style="color:rgba(255,255,255,.7);font-size:.8em;">{confidence:.0%} confidence</span>'
        f'</div>'
    )


def prob_chart(probabilities: dict[str, float], highlight: str) -> plt.Figure:
    values = [probabilities.get(lv, 0.0) for lv in LEVELS]
    colors = [LEVEL_META[lv][1] if lv == highlight else '#D0D0D0' for lv in LEVELS]
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    bars = ax.bar(LEVELS, values, color=colors, edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, values):
        if val > 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=7, color='#444')
    ax.set_ylim(0, max(values) * 1.25 + 0.05)
    ax.set_ylabel("Probability", fontsize=8)
    ax.tick_params(labelsize=8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def feature_chart(
    top_features: list[tuple[str, float]],
    color: str,
    xlabel: str,
) -> plt.Figure | None:
    if not top_features:
        return None
    words = [w for w, _ in top_features]
    scores = [s for _, s in top_features]
    fig, ax = plt.subplots(figsize=(3.8, max(1.8, len(words) * 0.32)))
    ax.barh(range(len(words)), scores, color=color, alpha=0.82)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=4)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# ── Header ────────────────────────────────────────────────────────────────────

st.title("CEFRscope")
st.markdown(
    "Classify English text by **CEFR level** using three machine learning models "
    "from statistical NLP. Where possible, the words that most influenced the prediction are shown."
)

# CEFR scale legend
legend_cols = st.columns(6)
for col, lv in zip(legend_cols, LEVELS):
    desc, color = LEVEL_META[lv]
    col.markdown(
        f'<div style="background:{color};padding:7px 4px;border-radius:7px;'
        f'text-align:center;color:#fff;font-weight:600;font-size:.85em;">'
        f'{lv}<br><span style="font-weight:400;font-size:.8em;">{desc}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────

text_input = st.text_area(
    "Paste your English text (a full paragraph or more works best):",
    height=180,
    placeholder=(
        "Paste a reading passage, an essay excerpt, a news article, etc. "
        "At least 3–5 sentences give reliable results."
    ),
)
classify_btn = st.button("Classify ▶", type="primary")

# ── Results ───────────────────────────────────────────────────────────────────

if classify_btn:
    if not text_input.strip():
        st.warning("Please enter some text first.")
        st.stop()

    word_count = len(text_input.split())
    if word_count < 40:
        st.warning(
            f"Your text has only {word_count} words. "
            "These models were trained on reading passages; results are most reliable with 40+ words."
        )

    models = load_models()
    predictions = [m.predict(text_input) for m in models]

    st.markdown("---")
    st.subheader("Results")
    result_cols = st.columns(3)

    for col, model, pred in zip(result_cols, models, predictions):
        color = LEVEL_META[pred.label][1]
        confidence = pred.probabilities.get(pred.label, 0.0)

        with col:
            st.markdown(f"#### {model.name}")
            st.markdown(level_badge(pred.label, confidence), unsafe_allow_html=True)

            st.markdown("**Confidence per level**")
            fig = prob_chart(pred.probabilities, pred.label)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if pred.top_features:
                st.markdown("**Top words driving this prediction**")
                fig2 = feature_chart(pred.top_features, color, pred.feature_description)
                if fig2:
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
            else:
                st.caption("No vocabulary matches found in this text.")

# ── About ─────────────────────────────────────────────────────────────────────

with st.expander("About the models"):
    st.markdown(
        "These classifiers were trained on the "
        "[CEFR-Levelled Texts dataset](https://www.kaggle.com/datasets/thedevastator/cefr-english-level-corpus) "
        "(~1,200 balanced samples across A1–C2), originally assembled from "
        "British Council, Trinity College London, and Oxford University Press materials. "
        "Stopwords are removed before classification."
    )
    st.markdown("")

    models = load_models()
    cols = st.columns(3)
    for col, m in zip(cols, models):
        with col:
            st.markdown(f"**{m.name}**")
            st.markdown(
                f"Accuracy: **{m.accuracy:.0%}** &nbsp;|&nbsp; F1: **{m.f1:.0%}**",
                unsafe_allow_html=True,
            )
            st.caption(m.description)

    st.markdown("")
    st.markdown(
        "Models were evaluated on a held-out 25% split. "
        "All models were found to confuse adjacent CEFR levels most often — "
        "consistent with research showing expert disagreement on neighbouring levels."
    )

# ── Methodology ───────────────────────────────────────────────────────────────

with st.expander("How it was built"):
    st.markdown(
        """
#### The task

Automatically assigning a CEFR level to an English text is a genuinely hard
multi-class problem. Adjacent levels (e.g. B1 vs B2) share overlapping vocabulary
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
"""
    )
