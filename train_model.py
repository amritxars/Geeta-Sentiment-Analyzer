"""
train_model.py
--------------
Labels each Gita verse with a sentiment/theme using NLP,
then trains an ML classifier to predict user emotion → best matching verses.

Run AFTER fetch_verses.py:
    python train_model.py

Outputs:
    - gita_verses_labeled.csv   (verses with emotion labels)
    - model.pkl                 (trained classifier)
    - vectorizer.pkl            (TF-IDF vectorizer)
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. Load verses ──────────────────────────────────────────────────────────
try:
    df = pd.read_csv("gita_verses.csv")
    print(f"✅ Loaded {len(df)} verses from gita_verses.csv")
except FileNotFoundError:
    print("❌ gita_verses.csv not found. Run fetch_verses.py first.")
    exit()

# ── 2. Rule-based emotion labeller ──────────────────────────────────────────
# Maps keyword patterns in the verse translation → emotion label
# These are the 5 emotions we support in the app

EMOTION_KEYWORDS = {
    "sad": [
        "grief", "sorrow", "weep", "mourn", "lament", "pain", "suffer",
        "misery", "despair", "anguish", "tears", "distress", "lost"
    ],
    "anxious": [
        "fear", "afraid", "tremble", "dread", "worry", "doubt", "uncertain",
        "confused", "conflict", "overwhelm", "restless", "agitate"
    ],
    "angry": [
        "anger", "wrath", "fury", "rage", "violent", "destroy", "battle",
        "enemy", "conquer", "fight", "war", "desire", "lust", "greed"
    ],
    "happy": [
        "joy", "bliss", "delight", "peace", "content", "happy", "pleasure",
        "rejoice", "serene", "tranquil", "eternal", "liberate", "free"
    ],
    "neutral": [
        "duty", "action", "knowledge", "truth", "wisdom", "path",
        "perform", "practice", "discipline", "self", "soul", "mind"
    ]
}

def label_verse(text):
    """Assign an emotion label based on keyword presence."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "neutral"
    text_lower = text.lower()
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    # If no keywords matched at all, default to neutral
    if scores[best] == 0:
        return "neutral"
    return best

print("\n🏷️  Labelling verses with emotions...")
df["emotion"] = df["translation"].apply(label_verse)
print(df["emotion"].value_counts().to_string())

df.to_csv("gita_verses_labeled.csv", index=False, encoding="utf-8")
print("\n✅ Saved gita_verses_labeled.csv")

# ── 3. Build training data ───────────────────────────────────────────────────
# Training data: user-like phrases → emotion label
# The model learns to map user TEXT → emotion, then we recommend matching verses

TRAINING_DATA = [
    # sad
    ("I feel so sad and hopeless", "sad"),
    ("I am grieving the loss of someone close", "sad"),
    ("Everything feels meaningless", "sad"),
    ("I am devastated and broken inside", "sad"),
    ("I cry all the time and don't know why", "sad"),
    ("I feel empty and alone", "sad"),
    ("Nothing makes me happy anymore", "sad"),
    ("I lost my purpose in life", "sad"),
    ("I am deeply hurt by someone I trusted", "sad"),
    ("Life feels like a burden", "sad"),

    # anxious
    ("I am very anxious and stressed", "anxious"),
    ("I have too many responsibilities and I'm overwhelmed", "anxious"),
    ("I'm scared about my future", "anxious"),
    ("I overthink everything and can't sleep", "anxious"),
    ("There is so much uncertainty in my life", "anxious"),
    ("I feel like I'm failing at everything", "anxious"),
    ("I'm constantly worried about what others think", "anxious"),
    ("My exams are near and I am panicking", "anxious"),
    ("I keep doubting myself", "anxious"),
    ("I don't know what decision to make", "anxious"),

    # angry
    ("I am so angry at everything", "angry"),
    ("Someone betrayed me and I am furious", "angry"),
    ("I feel rage and frustration inside", "angry"),
    ("People keep taking me for granted", "angry"),
    ("I want to give up on everyone", "angry"),
    ("I feel disrespected and it makes me mad", "angry"),
    ("Why does everything go wrong for me", "angry"),
    ("I can't control my temper", "angry"),

    # happy
    ("I am feeling great and joyful today", "happy"),
    ("Life is beautiful and I am grateful", "happy"),
    ("I feel at peace with myself", "happy"),
    ("Everything is going well in my life", "happy"),
    ("I feel content and fulfilled", "happy"),
    ("I achieved my goal and feel amazing", "happy"),
    ("I am happy and want to stay positive", "happy"),
    ("I feel blessed and full of energy", "happy"),

    # neutral
    ("I want to know about my purpose in life", "neutral"),
    ("I am thinking about my goals and future", "neutral"),
    ("I want to improve myself", "neutral"),
    ("I am curious about spirituality and meaning", "neutral"),
    ("I want guidance on making the right choices", "neutral"),
    ("I feel okay, just looking for some wisdom", "neutral"),
    ("I want to understand the right way to live", "neutral"),
    ("I'm just reflecting on life", "neutral"),
]

train_df = pd.DataFrame(TRAINING_DATA, columns=["text", "emotion"])

# ── 4. Train classifier ──────────────────────────────────────────────────────
print("\n🤖 Training emotion classifier...")

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english")
X = vectorizer.fit_transform(train_df["text"])
y = train_df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=500, C=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print(classification_report(y_test, y_pred))

# ── 5. Save model & vectorizer ───────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl and vectorizer.pkl saved")
print("\n🎉 Training complete! Now run:  streamlit run app.py")
