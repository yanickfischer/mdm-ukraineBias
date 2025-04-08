### ℹ️ Update nach Twitter Restriktion
Aufgrund einer Änderung bei Twitter funktioniert das Modul Twitter-Scraper leider aktuell nicht mehr. Dos Modell konnte aber mit realen Daten trainiert werden. 
Anstelle des Twitter-Scrapers wurde ein zusätzliches Scraping-Modul entwickelt, welches aktuelle News von https://kyivindependent.com/tag/ukraine scrapt und auf der Webseite bereitstellt.

# 🇺🇦 SentimentAnalyzer – Ukraine-Krieg auf Twitter verstehen

## 🎯 Motivation

Der Ukraine-Krieg ist nicht nur ein militärischer, sondern auch ein Informationskrieg. Propaganda wird von allen Seiten gezielt eingesetzt – sei es durch Russland, die Ukraine oder westliche Akteure.  
Das Einordnen von Meinungen, Emotionen und impliziten Botschaften in Tweets ist daher eine zentrale Herausforderung.

Dieses Tool soll als *Proof of Concept* zeigen, wie moderne NLP-Modelle und Automatisierungstechniken helfen können, politische Sentiments in sozialen Medien zu analysieren – und damit ein Stück weit Medienkompetenz algorithmisch zu unterstützen.

## 📌 Projektbeschreibung

In der heutigen Informationsflut rund um geopolitische Konflikte – insbesondere den Ukraine-Krieg – wird die objektive Einordnung von Meinungen immer schwieriger.  
Dieses Projekt zielt darauf ab, ein Werkzeug bereitzustellen, das hilft, die sentimentale Färbung von Twitter-Beiträgen rund um den Ukraine-Konflikt automatisch zu erkennen und einzuordnen.

## 🧠 Hauptfunktionen

Das Projekt automatisiert die gesamte ModelOps-Pipeline von der Datensammlung bis zur Online-Vorhersage und umfasst folgende Kernkomponenten:

1. **Scraping von Tweets (X)**  
   Mit dem Python-Tool `NTScraper` werden Twitter-Posts gesammelt, die thematisch mit dem Ukraine-Krieg zu tun haben. Es wird darauf geachtet, ein möglichst ausgewogenes Set an Begriffen zu verwenden, um Verzerrungen im Datensatz zu minimieren.

2. **Datenpersistenz mit MongoDB**  
   Die gescrapten Daten werden in einer MongoDB-Datenbank gespeichert – lokal oder in der Azure Cloud (CosmosDB-kompatibel).

3. **GPT-Filtering und Labeling für Sentiment-Kategorisierung**
   Gescrapte Tweets werden zuerst mithilfe einer GPT-API auf Relevanz zum Thema geprüft und nicht relevante Tweets werden aussortiert.
   Danach wird ein Teil der Tweets mithilfe der ChatGPT-API manuell (semi-supervised) gelabelt:  
   - `0 = pro Russland`  
   - `1 = neutral`  
   - `2 = pro Ukraine`

5. **Training eines eigenen Sentiment-Modells**  
   Basierend auf den gelabelten Daten wird ein `roBERTa`-basiertes Modell feingetuned, das auf den Ukraine-Kontext spezialisiert ist.

6. **Sentiment-Prediction für neue Tweets**  
   Das trainierte Modell kann nun neue Tweets automatisch klassifizieren – mit Fokus auf politische, propagandistische oder emotionale Tendenzen.

---

## 🛠️ Technologischer Stack

| Komponente           | Technologie / Tool                                           |
|----------------------|--------------------------------------------------------------|
| **Scraping**         | [NTScraper](https://github.com/JustAnotherArchivist/ntscraper), Scrapy |
| **Datenbank**        | MongoDB, Azure CosmosDB (Mongo API)                          |
| **Labeling**         | GPT-4 API                                                    |
| **Modelltraining**   | `transformers`, `scikit-learn`, `pandas`                     |
| **ML-Modell**        | RoBERTa (finetuned)                                          |
| **Serving Backend**  | Flask                                                        |
| **Frontend**         | Bootstrap, Vanilla JS                                        |
| **Containerisierung**| Docker                                                       |
| **Deployment**       | Azure App Service / Azure Container Instances (ACI)          |
| **CI/CD**            | GitHub Actions                                               |

---


---
