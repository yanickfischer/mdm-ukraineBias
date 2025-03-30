import os, time, json
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
BATCH_SIZE = int(os.getenv("GPT_BATCH_SIZE", 10))
SLEEP_TIME = float(os.getenv("GPT_SLEEP_TIME", 1.5))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)
collection = mongo_client["ukraineBiasDB"]["tweets_balanced"]

# Aggregation laden
pipeline = [
    {"$group": {"_id": "$text", "doc": {"$first": "$$ROOT"}}},
    {"$replaceRoot": {"newRoot": "$doc"}},
    {"$project": {"_id": 1, "text": 1}}
]
cursor = collection.aggregate(pipeline)
df = pd.DataFrame(list(cursor))
df = df[df['text'].notnull()].reset_index(drop=True)
logging.info(f"📥 Texte geladen: {len(df)}")

def gpt_filter_relevant(texts):
    prompt = (
        "Du bekommst eine Liste mit Social Media Texten. Bitte gib nur diejenigen zurück, die sich thematisch mit dem Krieg zwischen Russland und der Ukraine, der Interaktion zwischen europäischen Staten mit dem Krieg in der Ukraine oder auch der US-Regierung im Kontext des Krieges in der Ukraine oder der Beziehung zu Russland beschäftigen \n\n"
        "Gib die Antwort als Liste von JSON-Objekten im Format:\n{\"text\": \"...\"}\n\n"
        "Texte:\n"
    )
    for i, t in enumerate(texts):
        prompt += f"{i+1}. {t}\n"

    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.info("❌ Fehler beim Filtern:", e)
        return None

def filter_relevant_texts(df, batch_size=BATCH_SIZE):
    relevant = []
    total_checked = 0

    for i in tqdm(range(0, len(df), batch_size), desc="🔍 Relevanz filtern"):
        texts = df.iloc[i:i+batch_size]["text"].tolist()
        result = gpt_filter_relevant(texts)
        total_checked += len(texts)

        if result:
            try:
                parsed = json.loads(result)
                filtered_batch = [p for p in parsed if isinstance(p, dict) and "text" in p]
                relevant.extend(filtered_batch)
            except Exception as e:
                logging.info("⚠️ Fehler beim Parsen:", e)
                logging.info(result)
        time.sleep(SLEEP_TIME)

    logging.info(f"✅ Geprüft: {total_checked}, Relevant: {len(relevant)}, Ausgeschieden: {total_checked - len(relevant)}")
    return pd.DataFrame(relevant)

# Anwenden
df_relevant = filter_relevant_texts(df)
df = df_relevant.copy().reset_index(drop=True)
df.to_json(os.path.join(OUTPUT_DIR, "filtered_relevant_texts.json"), orient="records", indent=2)

def gpt_label_batch(texts):
    prompt = (
        "Bitte klassifiziere die folgenden Texte auf Basis ihrer geopolitischen Haltung:\n"
        "0 = Pro-Russland\n1 = Neutral\n2 = Pro-Ukraine\n\n"
        "Gib die Antwort als Liste von JSON-Objekten im Format: {\"text\": \"…\", \"label\": 0}\n\n"
        "Texte:\n"
    )
    for i, t in enumerate(texts):
        prompt += f"{i+1}. {t}\n"

    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.info("❌ GPT API Fehler:", e)
        return None

# Sentiment Labeling starten
batch_size = BATCH_SIZE
results = []

for i in tqdm(range(0, len(df), batch_size), desc="🏷 Sentiment Labeling"):
    batch = df.iloc[i:i+batch_size]['text'].tolist()
    result = gpt_label_batch(batch)

    if result:
        try:
            parsed = json.loads(result)
            valid = [r for r in parsed if isinstance(r, dict) and "text" in r and "label" in r]
            results.extend(valid)
        except Exception as e:
            logging.info("⚠️ Parsing-Fehler bei GPT Antwort:", e)
            logging.info(result)
    time.sleep(SLEEP_TIME)

df_out = pd.DataFrame(results)
df_out.to_json(os.path.join(OUTPUT_DIR, "labeled_tweets.json"), orient="records", indent=2)
df_out.to_csv(os.path.join(OUTPUT_DIR, "labeled_tweets.csv"), index=False)

# In MongoDB speichern
target_collection = mongo_client["ukraineBiasDB"]["labelled_tweets_training"]
docs = df_out.to_dict(orient="records")
if docs:
    target_collection.insert_many(docs)
logging.info(f"✅ {len(docs)} gelabelte Dokumente gespeichert in 'labelled_tweets_training'.")
