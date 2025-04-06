from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
from pymongo import MongoClient
import os
import logging

logging.basicConfig(level=logging.INFO)
mongo_uri = os.environ.get("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("❌ MONGO_URI nicht gesetzt – bitte als Umgebungsvariable definieren.")
client = MongoClient(mongo_uri)

db = client["ukraineBiasDB"]
collection = db["labelled_augmentedCount_tweets_training"]

BASE_DIR = Path(__file__).resolve().parent.parent

model_path_raw = os.environ.get("MODEL_PATH", str(BASE_DIR / "app" / "model-final"))

MODEL_PATH = Path(model_path_raw).expanduser().resolve()

if not MODEL_PATH.exists():
    raise RuntimeError(f"❌ Modellpfad nicht gefunden: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

label_map = {0: "pro-Russland", 1: "neutral", 2: "pro-Ukraine"}

static_dir = BASE_DIR / "app" / "frontend"
app = Flask(__name__, static_folder=str(static_dir))

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", None)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = torch.max(probabilities).item()
    label = label_map[prediction]

    return jsonify({
        "prediction": label,
        "label": prediction,
        "confidence": confidence
    })

@app.route("/bias_score")
def get_bias_score():
    try:
        logging.info("Datenbank: %s", db.name)
        logging.info("Collection: %s", collection.name)
        logging.info("Anzahl Dokumente: %d", collection.count_documents({}))

        pipeline = [
            {"$group": {"_id": "$label", "count": {"$sum": 1}}}
        ]
        results = list(collection.aggregate(pipeline))

        logging.info("Aggregationsergebnis: %s", results)

        if not results:
            return jsonify({"bias_score": 0.0})

        total = sum(r["count"] for r in results)
        weighted_sum = sum((r["_id"] - 1) * r["count"] for r in results)

        score = round(weighted_sum / total, 2) if total > 0 else 0.0
        return jsonify({"bias_score": score})

    except Exception as e:
        logging.info("Fehler in /bias_score: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/label_distribution")
def label_distribution():
    try:
        pipeline = [
            {"$group": {"_id": "$label", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}  # Reihenfolge: 0, 1, 2
        ]
        results = list(collection.aggregate(pipeline))
        return jsonify(results)
    except Exception as e:
        logging.info("Fehler in /label_distribution: %s", e)
        return jsonify({"error": str(e)}), 500    

@app.route("/training_metadata")
def training_metadata():
    metadata = {
        "model_name": "ukraineBias-roBERTa-v1 (finetuned Russo-Ukraine War)",
        "trained_on": "2025-03-28",
        "original_tweets": 499,
        "augmented_tweets": 1497,
        "val_accuracy": 0.874,
        "model_size_mb": 124
    }
    return jsonify(metadata)   

@app.route("/random_training_example")
def random_training_example():
    try:
        sample = list(collection.aggregate([{"$sample": {"size": 1}}]))
        if not sample:
            return jsonify({"error": "Keine Beispiele gefunden"}), 404

        example = sample[0]
        return jsonify({
            "text": example.get("text", ""),
            "label": label_map.get(example.get("label", -1), "Unbekannt")
        })

    except Exception as e:
        logging.info("Fehler in /random_training_example: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/news")
def get_news():
    try:
        news_cursor = db["ukraine-news"].find({}, {"title": 1, "_id": 0}).limit(18)
        news_list = list(news_cursor)
        return jsonify(news_list)
    except Exception as e:
        logging.info("Fehler in /news: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
