"""
app.py
------
Bhagavad Gita Sentiment Analyser — Streamlit App
College Project

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import re

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geeta Sentiment Analyser",
    page_icon="🕉️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital@0;1&family=Nunito&display=swap');

    .main { background-color: #0f0800; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

    .om-header {
        text-align: center;
        font-size: 3rem;
        color: #E8841A;
        margin-bottom: 0;
    }
    .title {
        text-align: center;
        color: #F5A94A;
        font-size: 1.8rem;
        font-family: 'Playfair Display', serif;
        margin: 0;
    }
    .subtitle {
        text-align: center;
        color: #8B6B45;
        font-size: 0.85rem;
        letter-spacing: 2px;
        margin-bottom: 2rem;
    }
    .verse-card {
        background: linear-gradient(135deg, #1E1106, #261608);
        border: 1px solid rgba(232,132,26,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
    }
    .verse-id {
        color: #E8841A;
        font-size: 0.8rem;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .verse-sanskrit {
        color: #C4956A;
        font-size: 1.1rem;
        font-style: italic;
        margin-bottom: 0.5rem;
        line-height: 1.8;
    }
    .verse-translation {
        color: #FDF6EC;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .divider {
        border: none;
        border-top: 1px solid rgba(232,132,26,0.2);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_verses():
    try:
        df = pd.read_csv("gita_verses_labeled.csv")
        df = df.dropna(subset=["translation"])
        df = df[df["translation"].str.strip() != ""]
        return df
    except FileNotFoundError:
        return None

model, vectorizer = load_model()
verses_df = load_verses()

# ── Emotion config ───────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "sad": {
        "emoji": "😔",
        "label": "Sad / Grieving",
        "color": "#4A9ECC",
        "message": "The Gita acknowledges your pain. These verses speak to grief, loss, and finding strength within suffering.",
        "badge_style": "background:#1a3a4a;color:#4A9ECC;border:1px solid #4A9ECC"
    },
    "anxious": {
        "emoji": "😰",
        "label": "Anxious / Overwhelmed",
        "color": "#C880F0",
        "message": "You're not alone in your uncertainty. The Gita offers wisdom on calming the restless mind.",
        "badge_style": "background:#2a1a3a;color:#C880F0;border:1px solid #C880F0"
    },
    "angry": {
        "emoji": "😤",
        "label": "Angry / Frustrated",
        "color": "#CC4A4A",
        "message": "The Gita speaks directly about anger — its origins and how to transcend it.",
        "badge_style": "background:#3a1a1a;color:#CC4A4A;border:1px solid #CC4A4A"
    },
    "happy": {
        "emoji": "😊",
        "label": "Happy / Content",
        "color": "#5CB85C",
        "message": "Beautiful! The Gita celebrates inner joy and offers verses to deepen your peace.",
        "badge_style": "background:#1a3a1a;color:#5CB85C;border:1px solid #5CB85C"
    },
    "neutral": {
        "emoji": "🙏",
        "label": "Seeking / Reflective",
        "color": "#E8841A",
        "message": "A seeker's mind is a clear mind. The Gita offers wisdom for those walking the path.",
        "badge_style": "background:#3a2010;color:#E8841A;border:1px solid #E8841A"
    }
}

QUESTIONS = [
    "How are you feeling today? Describe your current state of mind in a few words.",
    "What is something that's been on your mind lately — a situation, a worry, or a thought you can't shake?",
    "How do you feel about the people around you right now — family, friends, colleagues?",
]

# ── App state ────────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0  # 0=intro, 1-3=questions, 4=results
if "answers" not in st.session_state:
    st.session_state.answers = []

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="om-header">ॐ</div>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Geeta Sentiment Analyser</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">DISCOVER VERSES THAT SPEAK TO YOUR SOUL</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Check if model/data loaded ───────────────────────────────────────────────
if model is None or verses_df is None:
    st.error("⚠️ Model or verse data not found. Please run these steps first:")
    st.code("""
# Step 1 — fetch verses (needs internet)
python fetch_verses.py

# Step 2 — train model
python train_model.py

# Step 3 — launch app
streamlit run app.py
    """)
    st.stop()

# ── INTRO ────────────────────────────────────────────────────────────────────
if st.session_state.step == 0:
    st.markdown("""
    > *"It is better to live your own destiny imperfectly than to live an imitation of somebody else's life with perfection."*
    > — Bhagavad Gita, 3.35

    This app asks you a few simple questions about how you're feeling.
    Based on your answers, it uses **NLP + Machine Learning** to detect your emotional state
    and recommends the most relevant Bhagavad Gita verses for you.
    """)
    st.write("")
    if st.button("🙏 Begin", use_container_width=True):
        st.session_state.step = 1
        st.rerun()

# ── QUESTIONS ────────────────────────────────────────────────────────────────
elif 1 <= st.session_state.step <= len(QUESTIONS):
    q_idx = st.session_state.step - 1
    current_q = QUESTIONS[q_idx]

    st.markdown(f"**Question {st.session_state.step} of {len(QUESTIONS)}**")
    progress = st.session_state.step / len(QUESTIONS)
    st.progress(progress)

    st.write("")
    answer = st.text_area(
        current_q,
        height=100,
        placeholder="Write freely, there are no wrong answers...",
        key=f"q_{q_idx}"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.step > 1:
            if st.button("← Back"):
                st.session_state.step -= 1
                if st.session_state.answers:
                    st.session_state.answers.pop()
                st.rerun()
    with col2:
        if st.button("Next →", use_container_width=True):
            if len(answer.strip()) < 5:
                st.warning("Please share a bit more — even a few words help!")
            else:
                st.session_state.answers.append(answer.strip())
                st.session_state.step += 1
                st.rerun()

# ── RESULTS ─────────────────────────────────────────────────────────────────
elif st.session_state.step == len(QUESTIONS) + 1:

    # Combine all answers into one text blob
    combined_text = " ".join(st.session_state.answers)

    # Predict emotion
    X = vectorizer.transform([combined_text])
    predicted_emotion = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    cfg = EMOTION_CONFIG[predicted_emotion]

    # Emotion result
    st.markdown(f"### Your Emotional State")
    st.markdown(
        f'<div class="emotion-badge" style="{cfg["badge_style"]}">'
        f'{cfg["emoji"]} {cfg["label"]}</div>',
        unsafe_allow_html=True
    )
    st.write(cfg["message"])

    # Confidence breakdown
    with st.expander("📊 Confidence Breakdown"):
        prob_df = pd.DataFrame({
            "Emotion": [EMOTION_CONFIG[c]["emoji"] + " " + EMOTION_CONFIG[c]["label"] for c in classes],
            "Confidence": [f"{p*100:.1f}%" for p in probabilities]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Fetch matching verses
    st.markdown(f"### 📖 Recommended Verses for You")

    matching = verses_df[verses_df["emotion"] == predicted_emotion].copy()

    if len(matching) == 0:
        matching = verses_df.sample(5)

    # Score verses by keyword overlap with user text
    user_words = set(re.findall(r'\w+', combined_text.lower()))

    def relevance_score(row):
        verse_words = set(re.findall(r'\w+', str(row["translation"]).lower()))
        return len(user_words & verse_words)

    matching["score"] = matching.apply(relevance_score, axis=1)
    top_verses = matching.sort_values("score", ascending=False).head(5)

    for _, row in top_verses.iterrows():
        verse_id = f"Chapter {row['chapter']}, Verse {row['verse']}"
        st.markdown(f"""
        <div class="verse-card">
            <div class="verse-id">🕉️ {verse_id}</div>
            <div class="verse-sanskrit">{str(row.get('transliteration', ''))[:120]}...</div>
            <div class="verse-translation">{row['translation']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Reset
    if st.button("🔄 Try Again", use_container_width=True):
        st.session_state.step = 0
        st.session_state.answers = []
        st.rerun()
