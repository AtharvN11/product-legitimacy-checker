from flask import Flask, render_template, request, jsonify
import joblib
import re
from scraper import scrape_amazon_product
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load ML components with proper error handling
try:
    logger.info("Loading model and vectorizer...")
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    logger.info("ML components loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Missing required model files: {e}")
    raise Exception("Required model files are missing. Please ensure model.pkl and vectorizer.pkl exist.")
except Exception as e:
    logger.error(f"Error loading ML components: {e}")
    raise

def clean_text(text):
    """Basic text cleaning for reviews"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(text)).lower().strip()

def get_scraper(url):
    """Choose scraper based on domain"""
    if not url:
        return None
    if "amazon." in url.lower():
        return scrape_amazon_product
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            url = request.form.get("url", "").strip()
            
            if not url:
                logger.warning("Empty URL submitted")
                return jsonify({"error": "Please provide a valid URL."})
            
            logger.info(f"Processing URL: {url}")
            scraper = get_scraper(url)

            if not scraper:
                logger.warning(f"Unsupported platform for URL: {url}")
                return jsonify({"error": "Unsupported platform. Currently only Amazon is supported."})

            logger.info("Starting scraping...")
            product = scraper(url)
            
            if not product:
                logger.warning("No product data returned from scraper")
                return jsonify({"error": "Failed to fetch product data."})
            
            # Check if we have reviews to analyze
            if not product.get("reviews"):
                logger.warning("No reviews found for product")
                return jsonify({
                    "error": "No reviews found or product not accessible.",
                    "title": product.get("title", "N/A"),
                    "price": product.get("price", "N/A"),
                    "rating": product.get("rating", "N/A")
                })

            # Predict legitimacy
            logger.info(f"Processing {len(product['reviews'])} reviews")
            cleaned_reviews = [clean_text(r) for r in product["reviews"]]
            X = vectorizer.transform(cleaned_reviews)
            probs = model.predict_proba(X)
            avg_prob = probs[:, 1].mean()

            # Adjust scoring for well-known Amazon products
            if "amazon" in product["title"].lower() or "echo" in product["title"].lower():
                avg_prob *= 0.5  # Reduce probability of being a scam for Amazon products
            
            legitimacy_score = int((1 - avg_prob) * 100)
            malicious_score = int(avg_prob * 100)
            unworthy_score = int((avg_prob * 0.7) * 100)
            
            # Adjust threshold for scam detection
            label = "Scam" if avg_prob > 0.7 else "Legit"  # Increased threshold

            logger.info(f"Analysis complete. Label: {label}")
            
            return jsonify({
                "url": url,
                "title": product.get("title", "N/A"),
                "price": product.get("price", "N/A"),
                "rating": product.get("rating", "N/A"),
                "prediction": label,
                "confidence": legitimacy_score,
                "malicious_score": malicious_score,
                "unworthy_score": unworthy_score,
                "reason": f"Analyzed {len(product['reviews'])} reviews. Product seems {'suspicious' if label == 'Scam' else 'reliable'} based on user feedback."
            })

        return render_template("index.html")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
