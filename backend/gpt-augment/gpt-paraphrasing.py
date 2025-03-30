"""
Modul zur GPT-basierten Paraphrasierung von geopolitischen Tweets mit Augmentation in MongoDB.
"""

# GPT Augmentation mit MongoDB
import os
import time
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI

# 🔐 API & DB-Verbindung
load_dotenv()
logging.basicConfig(level=logging.INFO)
SLEEP_TIME = float(os.getenv("GPT_SLEEP_TIME", "1.5"))
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_mongo = MongoClient(os.getenv("MONGO_URI"))
collection_in = client_mongo["ukraineBiasDB"]["labelled_tweets_training"]
collection_out = client_mongo["ukraineBiasDB"]["labelled_augmentedCount_tweets_training"]


# 🔁 GPT-Funktion

def paraphrase_with_gpt(text, num_variants=2):
    prompt = (
        f"Paraphrasiere den folgenden Text {num_variants} mal. Behalte die geopolitische Bedeutung bei.\n"
        f"Text: \"{text}\"\n"
        f"Antwort als Liste:\n"
    )
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        content = response.choices[0].message.content.strip()
        return [line.strip("-• ") for line in content.split("\n") if line.strip()]
    except Exception as e:
        logging.info("GPT Fehler: %s", e)
        return []

# 📥 Daten aus Mongo laden
originals = list(collection_in.find({"label": {"$in": [0, 1, 2]}}))
logging.info("📦 Originale Trainingsbeispiele: %d", len(originals))

augmented = []

for i, doc in enumerate(originals):
    variants = paraphrase_with_gpt(doc["text"], num_variants=2)
    for variant in variants:
        augmented.append({"text": variant, "label": doc["label"]})
    time.sleep(SLEEP_TIME)
    if i % 10 == 0:
        logging.info("✅ %d verarbeitet", i + 1)

# Kombinieren & speichern
combined = originals + augmented
collection_out.insert_many(combined)
logging.info("🚀 Gesamt gespeichert: %d Dokumente in 'labelled_augmentedCount_tweets_training'", len(combined))
