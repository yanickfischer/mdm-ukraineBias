# scrape_balanced_sample.py

import logging
from ntscraper import Nitter
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
from datetime import datetime

# 🔌 Verbindung zu MongoDB herstellen
def connect_to_mongo():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("❌ MONGO_URI fehlt in .env-Datei!")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        logging.info("✅ Verbindung zu MongoDB erfolgreich")
        db = client["ukraineBiasDB"]
        return db["tweets_balanced"]
    except ConnectionFailure as e:
        logging.info(f"❌ Fehler bei Verbindung: {e}")
        return None
#Kernvariablen für die Suche
lang = "en"
since = datetime(2025, 3, 1)
until = datetime(2025, 3, 29)
# 🧠 Neutrale, balancierte Begriffe
terms = [
    "ukraine", "russia",
    "zelenskyy", 
    "putin",
    "kyiv", "moscow",
    "nato", "sanctions",
    "ukraine war", "ukraine conflict",
    "US peace talks", "weapons delivery",
    "russian occupation", "russian invasion",
    "#ZOV", "#SlavaUkraini"
]


# 🔍 Tweets sequentiell scrapen
def scrape_terms_sequential(terms, mode="term", lang="en", since=None, until=None, limit=500):
    scraper = Nitter(log_level=1)
    results = []
    for term in terms:
        try:
            logging.info(f"\n🔍 Scraping: {term}")
            result = scraper.get_tweets(term, mode=mode, number=limit)
            results.append(result)
        except Exception as e:
            logging.info(f"⚠️ Fehler bei '{term}': {e}")
    return results

# ▶️ Ausführung
if __name__ == "__main__":
    collection = connect_to_mongo()
    results = scrape_terms_sequential(terms, limit=500)

    total = 0

    for i, result in enumerate(results):
        tweets = result.get("tweets", [])
        valid_tweets = []

        term = terms[i]  # Suchbegriff für Zuordnung

        for tweet in tweets:
            tweet["search_term"] = term
            tweet.pop("_id", None)  # Falls vorhanden, entfernen
            valid_tweets.append(tweet)

        if valid_tweets:
            try:
                collection.insert_many(valid_tweets, ordered=False)
                total += len(valid_tweets)
                logging.info(f"✅ {len(valid_tweets)} gespeichert für '{term}'")
            except Exception as e:
                logging.info(f"⚠️ Fehler beim Speichern für '{term}': {e}")

    logging.info(f"\n🎉 Total gespeichert in 'tweets_balanced': {total} Tweets")