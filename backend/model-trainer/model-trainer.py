import os
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import logging

# %%
# 🔌 MongoDB laden
logging.basicConfig(level=logging.INFO)
mongo_uri = os.environ.get("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("❌ MONGO_URI nicht gesetzt – bitte via Compose oder Azure bereitstellen.")
client = MongoClient(mongo_uri)
collection = client["ukraineBiasDB"]["labelled_augmentedCount_tweets_training"]
data = pd.DataFrame(list(collection.find({"label": {"$in": [0, 1, 2]}})))
data = data[["text", "label"]].dropna().reset_index(drop=True)
data['label'] = data['label'].astype(int)
logging.info(f"📦 Trainingsdaten geladen: {len(data)} Beispiele")

# %%
# 📊 Train/Val Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
)

# %%
# 🧠 Tokenizer & Encoding
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# %%
# Dataset-Klasse
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, list(train_labels))
val_dataset = TweetDataset(val_encodings, list(val_labels))

# %%
# Modell initialisieren
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# %%
# 📈 Evaluation
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
# Trainer konfigurieren
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# %%
# 🚀 Training starten
trainer.train()

# %%
# 📦 Modell & Tokenizer explizit speichern
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "app", "model-final")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

logging.info(f"✅ Modell gespeichert unter: {SAVE_DIR}")
