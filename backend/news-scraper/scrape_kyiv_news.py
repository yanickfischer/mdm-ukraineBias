import json
import logging
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# 🔌 Verbindung zu MongoDB herstellen
def connect_to_mongo():
    logging.basicConfig(level=logging.INFO)
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("❌ MONGO_URI nicht gesetzt – bitte via docker-compose.yaml oder Azure Konfiguration übergeben.")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        logging.info("✅ Verbindung zu MongoDB erfolgreich")
        db = client["ukraineBiasDB"]
        return db["ukraine-news"]
    except ConnectionFailure as e:
        logging.info(f"❌ Fehler bei Verbindung: {e}")
        return None

# ▶️ Ausführung
if __name__ == "__main__":
    collection = connect_to_mongo()
    if collection is None:
        exit(1)

    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, "..", "kyiv_news.json")
    with open(json_path, "r", encoding="utf-8") as file:
        articles = json.load(file)

    valid_articles = []
    for article in articles:
        article.pop("_id", None)
        article["source"] = "kyivindependent"
        valid_articles.append(article)

    if valid_articles:
        try:
            collection.insert_many(valid_articles, ordered=False)
            logging.info(f"✅ {len(valid_articles)} Artikel in 'ukraine-news' gespeichert.")
        except Exception as e:
            logging.info(f"⚠️ Fehler beim Speichern der Artikel: {e}")