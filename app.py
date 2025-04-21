from flask import Flask, render_template, request, jsonify
from scraper import scrape_product
import joblib
import re
import logging
from typing import List

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load ML components
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    logger.error(f"Error loading ML components: {e}")
    raise

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()

def analyze_reviews(reviews: List[str]) -> dict:
    """Analyze reviews and return scores"""
    try:
        cleaned_reviews = [clean_text(r) for r in reviews]
        X = vectorizer.transform(cleaned_reviews)
        probs = model.predict_proba(X)
        avg_prob = probs[:, 1].mean()

        legitimacy_score = int((1 - avg_prob) * 100)
        malicious_score = int(avg_prob * 100)
        unworthy_score = int((avg_prob * 0.7) * 100)

        return {
            "legitimacy_score": legitimacy_score,
            "malicious_score": malicious_score,
            "unworthy_score": unworthy_score
        }
    except Exception as e:
        logger.error(f"Error analyzing reviews: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            url = request.form.get("url", "").strip()

            if not url:
                return jsonify({"error": "Please provide a valid URL"})

            # Get product info and reviews
            result = scrape_product(url)

            if "error" in result:
                return jsonify({"error": result["error"]})

            # Analyze reviews
            reviews = result.get("reviews", [])
            if not reviews:
                return jsonify({
                    "error": "No reviews found",
                    "title": result.get("title", "N/A"),
                    "price": result.get("price", "N/A"),
                    "rating": result.get("rating", "N/A")
                })

            # Get scores
            scores = analyze_reviews(reviews)
            if not scores:
                return jsonify({"error": "Error analyzing reviews"})

            return jsonify({
                "title": result.get("title", "N/A"),
                "price": result.get("price", "N/A"),
                "rating": result.get("rating", "N/A"),
                "confidence": scores["legitimacy_score"],
                "malicious_score": scores["malicious_score"],
                "unworthy_score": scores["unworthy_score"],
                "review_count": len(reviews),
                "reason": (
                    f"Analyzed {len(reviews)} reviews. Product seems "
                    f"{'suspicious' if scores['malicious_score'] > 70 else 'reliable'} "
                    f"based on user feedback."
                )
            })

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)