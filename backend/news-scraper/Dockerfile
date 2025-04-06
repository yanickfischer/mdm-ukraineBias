FROM python:3.10-slim

WORKDIR /app

# requirements installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Scrapy-Projekt und Scripts kopieren
COPY scrape_kyiv_news.py .
COPY kyiv_news.json .
COPY kyiv_scraper/ ./kyiv_scraper/

# Entry point
CMD ["python", "scrape_kyiv_news.py"]