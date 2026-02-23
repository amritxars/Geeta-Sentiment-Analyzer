"""
fetch_verses.py
---------------
Fetches all 700 Bhagavad Gita verses from the free VedicScriptures API
and saves them as gita_verses.csv

Run: python fetch_verses.py
"""

import requests
import pandas as pd
import time

# Each chapter and its verse count
CHAPTER_VERSES = {
    1: 47, 2: 72, 3: 43, 4: 42, 5: 29,
    6: 47, 7: 30, 8: 28, 9: 34, 10: 42,
    11: 55, 12: 20, 13: 35, 14: 27, 15: 20,
    16: 24, 17: 28, 18: 78
}

BASE_URL = "https://vedicscriptures.github.io/slok/{chapter}/{verse}/"

rows = []

print("Fetching Bhagavad Gita verses...\n")

for chapter, total_verses in CHAPTER_VERSES.items():
    for verse in range(1, total_verses + 1):
        url = BASE_URL.format(chapter=chapter, verse=verse)
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                # Pull English translation (Siva translation is clean)
                translation = ""
                if "siva" in data and data["siva"]:
                    translation = data["siva"].get("et", "")
                elif "purohit" in data and data["purohit"]:
                    translation = data["purohit"].get("et", "")
                elif "gambir" in data and data["gambir"]:
                    translation = data["gambir"].get("et", "")

                rows.append({
                    "chapter": chapter,
                    "verse": verse,
                    "verse_id": f"{chapter}.{verse}",
                    "sanskrit": data.get("slok", ""),
                    "transliteration": data.get("transliteration", ""),
                    "translation": translation.strip()
                })
                print(f"  ✓ Chapter {chapter}, Verse {verse}", end="\r")
            else:
                print(f"  ✗ Failed: {chapter}.{verse} — status {res.status_code}")
        except Exception as e:
            print(f"  ✗ Error at {chapter}.{verse}: {e}")

        time.sleep(0.1)  # Be polite to the API

    print(f"✅ Chapter {chapter} done ({total_verses} verses)")

df = pd.DataFrame(rows)
df.to_csv("gita_verses.csv", index=False, encoding="utf-8")
print(f"\n✅ Saved {len(df)} verses to gita_verses.csv")
print(df.head(3).to_string())
