# 🕉️ Bhagavad Gita Sentiment Analyser
**College Project | NLP + ML + Streamlit**

---

## What This Does

1. Asks the user 3 questions about how they're feeling
2. Uses **TF-IDF + Logistic Regression** to classify their emotion (sad / anxious / angry / happy / neutral)
3. Recommends the most relevant **Bhagavad Gita verses** based on the detected emotion

---

## Setup & Run (3 steps)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Fetch all 700 Gita verses
```bash
python fetch_verses.py
```
This calls the free VedicScriptures API and saves `gita_verses.csv`.
Takes ~3-5 minutes (700 API calls with small delay).

### Step 2 — Label verses & train the ML model
```bash
python train_model.py
```
This:
- Labels each verse with an emotion using keyword rules (NLP)
- Trains a Logistic Regression classifier on sample user phrases
- Saves `model.pkl`, `vectorizer.pkl`, and `gita_verses_labeled.csv`

### Step 3 — Launch the Streamlit app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## Files

| File | Purpose |
|------|---------|
| `fetch_verses.py` | Downloads all 700 Gita verses via API → CSV |
| `train_model.py` | Labels verses + trains emotion classifier |
| `app.py` | Streamlit web app |
| `requirements.txt` | Python dependencies |
| `gita_verses.csv` | Generated: raw verse data |
| `gita_verses_labeled.csv` | Generated: verses with emotion labels |
| `model.pkl` | Generated: trained classifier |
| `vectorizer.pkl` | Generated: TF-IDF vectorizer |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Source | VedicScriptures GitHub API (free) |
| NLP | TF-IDF Vectorization |
| ML Model | Logistic Regression (scikit-learn) |
| Sentiment Labelling | Rule-based keyword matching |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## How the ML Works

1. **NLP (Verse Labelling):** Each of the 700 verse translations is scanned for emotional keywords (e.g. "grief", "fear", "anger", "joy") and assigned one of 5 emotion labels.

2. **ML (User Classification):** A Logistic Regression model is trained on ~40 sample user phrases mapped to emotions. It uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numbers.

3. **Recommendation:** User's combined answers → predict emotion → filter verses by matching emotion → rank by keyword overlap → show top 5.

---

## Data Source

Verses fetched from **VedicScriptures API**:
- URL: `https://vedicscriptures.github.io/slok/{chapter}/{verse}/`
- Free, no API key needed
- Returns Sanskrit, transliteration, and English translation
- Covers all 18 chapters (700 verses total)
