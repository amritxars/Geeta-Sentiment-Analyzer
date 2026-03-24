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

# ── Verse Summaries ───────────────────────────────────────────────────────────
# 3-5 line summaries for commonly recommended verses
# Key format: "chapter.verse"
VERSE_SUMMARIES = {
    # Chapter 1
    "1.1":  "This verse opens the Kurukshetra war scene, where King Dhritarashtra anxiously asks about the battle. It sets the stage for the entire Gita. The blind king's question reflects his attachment to his sons. His dependence on Sanjaya's divine sight foreshadows his moral blindness. The war symbolises the inner conflict between righteousness and desire.",
    "1.28": "Arjuna sees his relatives and teachers on the battlefield and is overwhelmed with grief. He loses his will to fight, overcome by compassion and sorrow. This emotional breakdown is the trigger for Krishna's divine teachings. It reflects how attachment clouds our judgment in moments of duty. Arjuna's confusion mirrors the confusion every human faces at moral crossroads.",
    "1.47": "Arjuna drops his bow and collapses in despair on the chariot. He refuses to fight, consumed by grief and confusion. This marks the lowest point of Arjuna's emotional state. It is from this place of complete surrender that the Gita's wisdom begins. The verse teaches that true guidance often begins when we admit we don't know what to do.",

    # Chapter 2
    "2.3":  "Krishna rebukes Arjuna's weakness and urges him to rise with courage. He calls this faint-heartedness unworthy of a great warrior. The verse is a call to overcome emotional paralysis with inner strength. Krishna reminds Arjuna of his identity and responsibility. It teaches that self-pity is not compassion — it is an obstacle to dharma.",
    "2.14": "Krishna explains that pleasure and pain, heat and cold are temporary sensations. The wise person endures them without being disturbed. This verse introduces the concept of equanimity — staying balanced in all circumstances. It teaches that attachment to comfort leads to suffering. True wisdom lies in accepting life's impermanence without losing inner peace.",
    "2.19": "Krishna declares that the soul neither kills nor is killed. It is eternal, beyond birth and death. This verse addresses Arjuna's grief over killing his relatives by revealing the immortal nature of the self. Physical death is not the end of existence. Understanding this truth removes the deepest fear — the fear of loss and mortality.",
    "2.20": "The soul is never born and never dies — it is eternal and ancient. It does not cease to exist when the body is destroyed. This is one of the most profound verses in the Gita on the nature of the Atman. It liberates us from the fear of death and the grief of losing loved ones. Recognising the soul's immortality brings permanent peace.",
    "2.47": "This is perhaps the most famous verse of the Gita — 'You have the right to perform your duties, but not to the fruits of your actions.' It teaches the principle of Nishkama Karma — acting without attachment to results. Focusing only on outcomes leads to anxiety and disappointment. True peace comes from doing your best and surrendering the result. This verse is the foundation of karma yoga.",
    "2.62": "When a person dwells on sense objects, attachment is born. From attachment comes desire, and from desire comes anger. This verse traces the psychological chain from thought to destruction. It warns against letting the mind dwell unchecked on cravings. Understanding this chain is the first step to breaking the cycle of suffering.",
    "2.63": "From anger arises delusion, and from delusion comes loss of memory. When memory is lost, the intellect is destroyed, and the person perishes. This verse continues the chain begun in 2.62, showing how unchecked emotion leads to total ruin. It emphasises the importance of mental discipline and self-awareness. The greatest danger lies not outside us, but in our own uncontrolled mind.",

    # Chapter 3
    "3.16": "Those who do not follow the cycle of duty and live for sense pleasure are living in vain. This verse stresses the importance of fulfilling one's responsibilities in the world. Selfish living disconnects us from the cosmic order. Contributing to the world through righteous action is the true purpose of life. Laziness and self-indulgence are obstacles to spiritual growth.",
    "3.21": "Whatever a great person does, others follow. Whatever standard they set, the world adopts. This verse highlights the responsibility of leaders and role models. Our actions influence those around us, whether we realise it or not. Living with integrity is not just personal — it shapes the culture and values of those we lead.",
    "3.35": "It is better to perform one's own duty imperfectly than to do another's duty perfectly. This verse teaches the importance of authenticity over imitation. Living someone else's life, however successfully, is spiritual dishonesty. Your own path, however difficult, is the only path that leads to your growth. Following your own dharma, even with mistakes, is more valuable than borrowed perfection.",

    # Chapter 4
    "4.7":  "Whenever righteousness declines and unrighteousness rises, I manifest myself. This is Krishna's promise of divine intervention in the world. It offers comfort that even in the darkest times, goodness is never fully extinguished. The divine takes form to restore balance and protect the righteous. It reminds us that history bends toward justice.",
    "4.8":  "I appear age after age to protect the virtuous, destroy the wicked, and re-establish dharma. This verse reassures us that evil never permanently wins. The universe has a self-correcting moral force. It encourages perseverance in the face of injustice. The righteous are never truly alone in their struggle.",
    "4.38": "Nothing in this world is as purifying as knowledge. One who is perfected in yoga finds this knowledge within themselves in time. This verse celebrates the supreme value of self-knowledge. External achievements cannot cleanse the soul — only inner wisdom can. The journey inward is the most important journey a person can take.",

    # Chapter 5
    "5.10": "One who acts without attachment, surrendering actions to the divine, is untouched by sin, like a lotus leaf untouched by water. The lotus is a perfect metaphor for spiritual living — fully engaged in the world but never contaminated by it. Attachment is what creates karma and suffering, not action itself. We can be fully active in life while remaining internally free. This is the art of spiritual living.",
    "5.18": "The wise see with equal vision a learned scholar, a cow, an elephant, a dog, and an outcaste. True wisdom dissolves all social and superficial distinctions. At the deepest level, the same divine consciousness exists in all beings. Practising this equal vision is the highest form of compassion. It is the natural result of genuine spiritual understanding.",
    "5.22": "The pleasures born of sense contact are sources of suffering. They have a beginning and an end. The wise person does not rejoice in them. This verse warns against the trap of chasing sensory pleasure. What feels like happiness in the moment often leads to craving, addiction, and pain. Lasting joy can only be found within, not in external objects.",

    # Chapter 6
    "6.5":  "One must elevate oneself by one's own mind, not degrade oneself. The mind is both the friend and the enemy of the self. This powerful verse places full responsibility for our life in our own hands. No one else can do our inner work for us. The same mind that traps us in suffering can, with discipline, be our greatest liberator.",
    "6.6":  "For one who has conquered the mind, it is the best of friends. For one who has failed to do so, the mind remains the greatest enemy. The mind is the battlefield of life. An undisciplined mind creates constant suffering through fear, desire, and doubt. But a trained mind becomes a steady guide toward peace, clarity, and wisdom.",
    "6.17": "Success in yoga comes to the person who is moderate in eating, recreation, work, sleep, and waking. This verse prescribes balance as the key to a healthy and spiritual life. Extremes in any direction — overindulgence or excessive austerity — disturb the mind. A regulated, moderate lifestyle is the foundation for inner peace. The middle path is the wisest path.",
    "6.35": "The mind is indeed restless and difficult to control, but it can be trained through practice and detachment. Krishna acknowledges that controlling the mind is genuinely hard — he does not dismiss Arjuna's struggle. But he assures that with consistent effort, it is absolutely possible. This gives hope to everyone who struggles with anxious or scattered thoughts. Spiritual progress is always available to those who persist.",

    # Chapter 7
    "7.19": "After many births of striving, the wise person surrenders to Me, knowing that all is the Divine. Such a great soul is very rare. This verse speaks of the long journey of spiritual evolution across lifetimes. The ultimate realisation is the oneness of everything — that all existence is divine. This knowledge is rare and precious, and leads to the highest liberation.",

    # Chapter 8
    "8.7":  "Therefore, remember Me at all times and fight. With mind and intellect fixed on Me, you will surely come to Me. This verse teaches the practice of constant remembrance of the divine. Whatever we hold in our mind at the time of death shapes our next state. By filling the mind with higher consciousness during life, we naturally transcend at the end. It is a call to integrate spirituality into all of life's activities.",

    # Chapter 9
    "9.22": "To those who worship Me with devotion, meditating on My transcendental form, I carry what they lack and preserve what they have. This is Krishna's personal promise to those who surrender with love. The divine personally takes care of the sincere devotee's needs. It is a verse of profound comfort for those who feel alone or helpless. Surrender is not weakness — it is the deepest form of trust.",

    # Chapter 10
    "10.20": "I am the self, seated in the heart of all beings. I am the beginning, middle, and end of all creatures. This verse reveals the omnipresence of the divine within all life. The search for God outside is unnecessary — the divine is the very core of our own existence. Realising this inner presence is the goal of all spiritual practice.",

    # Chapter 11
    "11.33": "Therefore, arise and attain glory. Conquer your enemies and enjoy a prosperous kingdom. These warriors are already slain by Me — you are merely the instrument. This verse addresses the burden of feeling solely responsible for outcomes. The divine orchestrates all events; we are instruments of a larger plan. It does not mean we are passive — it means we act without the crushing weight of ego. Freedom from ego-ownership is true spiritual maturity.",

    # Chapter 12
    "12.13": "One who has no hatred for any being, who is friendly and compassionate — such a devotee is dear to Me. This verse describes the qualities of a true devotee. Spirituality is not measured by rituals alone, but by genuine compassion for all beings. Hatred and cruelty are incompatible with real spiritual growth. The person who loves all beings lives closest to the divine.",
    "12.15": "One who does not disturb the world, and is not disturbed by it — who is free from elation, envy, fear, and anxiety — is dear to Me. This describes emotional maturity — neither seeking praise nor fearing criticism. Such a person has found an inner anchor that external events cannot shake. They live peacefully in the world without being controlled by it. This inner stability is the mark of genuine spiritual growth.",

    # Chapter 14
    "14.22": "One who does not hate the presence of enlightenment, activity, or delusion, nor desires them when they are absent, who remains steady and undisturbed — that person has transcended the three gunas. This verse describes the state of one who has gone beyond the natural qualities that bind all beings. Such a person is not pulled by moods or circumstances. They are a witness to life, not a victim of it. This is the state of complete inner freedom.",

    # Chapter 15
    "15.7": "The living entities in this world are My eternal fragments. Due to conditioned life, they struggle with the six senses, including the mind. This verse explains our fundamental spiritual nature — we are all sparks of the divine. Yet we forget this and suffer by identifying with the body and mind. The spiritual journey is essentially a journey of remembrance. We are not lost souls searching for God — we are divine souls temporarily forgetting our nature.",

    # Chapter 16
    "16.21": "Lust, anger, and greed are the three gates to hell — destructive to the soul. One must abandon all three. These three forces are the root causes of almost all human suffering and wrongdoing. They cloud judgment, destroy relationships, and sever us from our true self. Recognising them as enemies, not desires to be indulged, is the beginning of wisdom. Freedom from these three is the gateway to inner peace.",

    # Chapter 17
    "17.3":  "The faith of every individual corresponds to their nature. A person is made of their faith — as their faith is, so they are. This verse reveals that our beliefs shape our character and reality. Faith is not blind belief but the deep orientation of the heart. What we consistently invest our trust and energy in defines who we become. To change your life, examine and transform what you truly believe.",

    # Chapter 18
    "18.61": "The Supreme Lord is situated in the heart of all beings and directs their wandering, as if they are mounted on a machine. This verse reminds us that a deeper intelligence guides all of existence. We are not fully in control, nor are we entirely helpless. Aligning with this inner divine guidance, rather than fighting it, brings peace. Surrender to the inner guide is the highest form of self-leadership.",
    "18.63": "I have now given you this most secret knowledge. Reflect on it fully, and then do as you choose. This is a remarkable verse — after delivering the entire Gita, Krishna gives Arjuna complete freedom to choose. The divine does not coerce or manipulate. True wisdom is always offered freely, never forced. Genuine spiritual guidance respects human free will absolutely.",
    "18.66": "Abandon all varieties of dharma and simply surrender to Me. I shall deliver you from all sinful reactions. Do not fear. This is the final and most essential teaching of the Gita — total surrender. It does not mean abandoning responsibility, but surrendering the ego's need to control outcomes. When we let go of fear and trust the divine completely, we are freed from the heaviest burden. This verse is Krishna's ultimate promise of grace.",
}

def get_verse_summary(chapter, verse):
    """Return summary for a verse, or generate a contextual one from the translation."""
    key = f"{chapter}.{verse}"
    return VERSE_SUMMARIES.get(key, None)

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
    .verse-summary {
        color: #A89070;
        font-size: 0.85rem;
        line-height: 1.75;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(232,132,26,0.15);
    }
    .summary-label {
        color: #E8841A;
        font-size: 0.72rem;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
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
        transliteration = str(row.get('transliteration', ''))[:120]
        translation = str(row['translation'])

        # Get summary
        summary = get_verse_summary(int(row['chapter']), int(row['verse']))
        if not summary:
            sentences = re.split(r'(?<=[.!?]) +', translation)
            summary = " ".join(sentences[:3]) + " This verse encourages reflection on one's inner state and the path toward balance and wisdom as taught in the Bhagavad Gita."

        # Render card and summary as one clean HTML block
        html = (
            '<div class="verse-card">'
            f'<div class="verse-id">🕉️ {verse_id}</div>'
            f'<div class="verse-sanskrit">{transliteration}...</div>'
            f'<div class="verse-translation">{translation}</div>'
            '<div class="verse-summary">'
            '<div class="summary-label">📝 Summary</div>'
            f'{summary}'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Reset
    if st.button("🔄 Try Again", use_container_width=True):
        st.session_state.step = 0
        st.session_state.answers = []
        st.rerun()