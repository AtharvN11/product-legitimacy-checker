import subprocess
import pandas as pd
import os
import re
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

SCRAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'AMAZON-REVIEW-SCRAPER'))
CSV_FILE = os.path.join(SCRAPER_DIR, "amazon_reviews.csv")
BERT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bert_local'))

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR),
    tokenizer=AutoTokenizer.from_pretrained(BERT_MODEL_DIR),
    top_k=1,
    device=device,
    truncation=True,
    max_length=512,
    padding=True
)

def extract_asin_and_domain(url):
    match = re.search(r"amazon\.(\w+).*?/([dg]p/product|dp)/([A-Z0-9]{10})", url)
    if match:
        tld = match.group(1)
        asin = match.group(3)
        return asin, tld
    return None, None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/scrape", methods=["POST"])
def scrape_reviews():
    import json

    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    asin = extract_asin_and_domain(url)
    if not asin:
        return jsonify({"error": "Invalid Amazon URL"}), 400

    # Ensure old CSV is removed to avoid stale predictions
    try:
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
    except Exception as e:
        app.logger.warning(f"Could not remove old CSV: {e}")

    # === Step 1: Scrape reviews via Make ===
    try:
        subprocess.run(
            ["make", "scrape", f"ASIN_CODE={asin}"],
            cwd=SCRAPER_DIR,
            check=True
        )
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Scraper subprocess failed: {e}")
        return jsonify({"error": "Scraping process failed"}), 500

    # === Step 2: Read the generated CSV ===
    if not os.path.exists(CSV_FILE):
        return jsonify({"error": "No reviews found or scraper failed silently."}), 500

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        app.logger.exception("Failed to read amazon_reviews.csv")
        return jsonify({"error": f"Failed to load CSV: {str(e)}"}), 500

    if df.empty:
        return jsonify({"error": "CSV file is empty. No reviews extracted."}), 200

    # === Step 3: Prepare review list ===
    reviews = []
    for _, row in df.iterrows():
        reviews.append({
            "title": str(row.get("title", "No Title")).strip() or "No Title",
            "content": str(row.get("content", "")).strip(),
            "author": str(row.get("author", "")).strip() or "Unknown",
            "rating": float(row.get("rating", 0)) if pd.notna(row.get("rating")) else "N/A"
        })

    texts = [r["content"] for r in reviews if r["content"].strip()]
    if not texts:
        return jsonify({
            "asin": asin,
            "confidence": 0,
            "fake_review_percent": 0,
            "unworthy_score": 0,
            "reviews": reviews
        })

    # === Step 4: Run BERT classifier ===
    try:
        preds = classifier(texts, batch_size=8)
    except Exception as e:
        app.logger.exception("BERT classification failed")
        return jsonify({"error": f"Classifier failed: {str(e)}"}), 500

    malicious_probs, unworthy_probs = [], []
    for p in preds:
        scores = {x["label"].lower(): x["score"] for x in p}
        malicious_probs.append(scores.get("malicious", 0))
        unworthy_probs.append(scores.get("unworthy", 0))

    # === Step 5: Metrics ===
    n = len(malicious_probs)
    legitimacy_score = (sum([1 - x for x in malicious_probs]) / n) * 100
    fake_review_percent = 100 - legitimacy_score
    unworthy_score = (sum(unworthy_probs) / n) * 100

    return jsonify({
        "asin": asin,
        "confidence": round(legitimacy_score, 2),
        "fake_review_percent": round(fake_review_percent, 2),
        "unworthy_score": round(unworthy_score, 2),
        "reviews": reviews
    })

