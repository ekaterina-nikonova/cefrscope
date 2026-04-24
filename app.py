import matplotlib
import matplotlib.pyplot as plt
import nltk
import streamlit as st
from pathlib import Path


def _md(name: str) -> str:
    return (Path(__file__).parent / "content" / name).read_text(encoding="utf-8")

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

_heading_parts = list(zip(["CEFR", "s", "c", "o", "p", "e"], LEVELS))
_heading_html = "".join(
    f'<span style="color:{LEVEL_META[lv][1]}">{seg}</span>'
    for seg, lv in _heading_parts
)
st.markdown(f'<h1>{_heading_html}</h1>', unsafe_allow_html=True)
st.markdown(_md("intro.md"))

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
    st.markdown(_md("about_intro.md"))

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

    st.markdown(_md("about_footer.md"))

# ── Methodology ───────────────────────────────────────────────────────────────

with st.expander("How it was built"):
    st.markdown(_md("methodology.md"))
