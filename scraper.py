# universal_scraper.py

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import logging
import json
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        self.currency_symbols = r'[$€£¥]|EUR|USD|GBP|JPY'
        self.price_regex = fr'(?:{self.currency_symbols})\s*[0-9]+(?:[.,][0-9]{{2}})?|\b[0-9]+(?:[.,][0-9]{{2}})?\s*(?:{self.currency_symbols})'

    def make_request(self, url: str) -> Optional[str]:
        try:
            session = requests.Session()
            response = session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def extract_json_ld(self, soup) -> List[Dict]:
        """Extract and parse all JSON-LD data from the page"""
        json_ld_data = []
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    json_ld_data.extend(data)
                else:
                    json_ld_data.append(data)
            except:
                continue
        return json_ld_data

    def extract_meta_data(self, soup) -> Dict:
        """Extract metadata from meta tags"""
        meta_data = {}
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            property_name = tag.get('property', tag.get('name', ''))
            content = tag.get('content', '')
            if property_name and content:
                meta_data[property_name.lower()] = content
        return meta_data

    def clean_price(self, price_str: str) -> Optional[str]:
        """Clean and standardize price string"""
        if not price_str:
            return None
        # Remove whitespace and convert to lowercase
        price_str = price_str.strip().lower()
        # Extract numbers and currency
        match = re.search(self.price_regex, price_str)
        if match:
            price = match.group(0)
            # Standardize format
            price = price.replace(',', '.')
            # Ensure currency symbol is present
            if not any(sym in price for sym in ['$', '€', '£', '¥']):
                if 'eur' in price_str:
                    price = f'€{price}'
                elif 'usd' in price_str:
                    price = f'${price}'
                elif 'gbp' in price_str:
                    price = f'£{price}'
                elif 'jpy' in price_str:
                    price = f'¥{price}'
            return price
        return None

    def extract_price(self, soup) -> Optional[str]:
        # Method 1: JSON-LD data
        json_ld_data = self.extract_json_ld(soup)
        for data in json_ld_data:
            if isinstance(data, dict):
                # Check for various JSON-LD price patterns
                price = None
                if 'offers' in data:
                    offers = data['offers']
                    if isinstance(offers, dict):
                        price = offers.get('price')
                    elif isinstance(offers, list) and offers:
                        price = offers[0].get('price')
                elif 'price' in data:
                    price = data['price']
                
                if price:
                    cleaned_price = self.clean_price(str(price))
                    if cleaned_price:
                        return cleaned_price

        # Method 2: Meta tags
        meta_data = self.extract_meta_data(soup)
        for key, value in meta_data.items():
            if 'price' in key:
                cleaned_price = self.clean_price(value)
                if cleaned_price:
                    return cleaned_price

        # Method 3: Common price selectors with priority
        price_selectors = [
            '[data-price]',
            '[itemprop="price"]',
            '.product-price [content]',
            '.price-value',
            '.price--withoutTax',
            '.price--withTax',
            '.product__price',
            '[class*="price"]',
            '[id*="price"]',
            'span.amount',
            '.sale-price'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Try data attributes first
                price_candidates = [
                    element.get('data-price'),
                    element.get('content'),
                    element.get('value'),
                    element.get_text()
                ]
                
                for candidate in price_candidates:
                    if candidate:
                        cleaned_price = self.clean_price(candidate)
                        if cleaned_price:
                            return cleaned_price

        return None

    def clean_rating(self, rating_str: str) -> Optional[str]:
        """Clean and standardize rating string"""
        if not rating_str:
            return None
        # Extract number from string
        match = re.search(r'([0-9]{1,2}(?:\.[0-9]{1,2})?)', rating_str)
        if match:
            rating = float(match.group(1))
            # Normalize to 5-star scale if necessary
            if rating > 5:
                rating = (rating * 5) / 10
            return f"{rating:.1f}"
        return None

    def extract_rating(self, soup) -> Optional[str]:
        # Method 1: JSON-LD data
        json_ld_data = self.extract_json_ld(soup)
        for data in json_ld_data:
            if isinstance(data, dict):
                if 'aggregateRating' in data:
                    rating = data['aggregateRating'].get('ratingValue')
                    if rating:
                        return self.clean_rating(str(rating))

        # Method 2: Meta tags
        meta_data = self.extract_meta_data(soup)
        for key, value in meta_data.items():
            if 'rating' in key:
                cleaned_rating = self.clean_rating(value)
                if cleaned_rating:
                    return cleaned_rating

        # Method 3: Common rating selectors
        rating_selectors = [
            '[data-rating]',
            '[itemprop="ratingValue"]',
            '.rating-value',
            '.product-rating',
            '[class*="rating"]',
            '[class*="stars"]',
            '.average-rating'
        ]
        
        for selector in rating_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Try multiple attributes
                rating_candidates = [
                    element.get('data-rating'),
                    element.get('content'),
                    element.get('value'),
                    element.get_text()
                ]
                
                for candidate in rating_candidates:
                    if candidate:
                        cleaned_rating = self.clean_rating(candidate)
                        if cleaned_rating:
                            return cleaned_rating

        return None

    def clean_review(self, review_text: str) -> Optional[str]:
        """Clean and validate review text"""
        if not review_text:
            return None
        # Remove extra whitespace and normalize
        review = ' '.join(review_text.split())
        # Remove very short or invalid reviews
        if len(review.split()) < 3 or len(review) < 10:
            return None
        # Remove likely non-review content
        if any(x in review.lower() for x in ['404', 'error', 'javascript', 'cookie']):
            return None
        return review

    def extract_reviews(self, soup) -> List[str]:
        reviews = set()  # Use set to avoid duplicates

        # Method 1: JSON-LD data
        json_ld_data = self.extract_json_ld(soup)
        for data in json_ld_data:
            if isinstance(data, dict):
                # Handle different review formats in JSON-LD
                review_data = []
                if 'review' in data:
                    review_data.extend(data['review'] if isinstance(data['review'], list) else [data['review']])
                if 'reviews' in data:
                    review_data.extend(data['reviews'] if isinstance(data['reviews'], list) else [data['reviews']])
                
                for review in review_data:
                    if isinstance(review, dict):
                        review_text = review.get('reviewBody', review.get('description'))
                        if review_text:
                            cleaned_review = self.clean_review(review_text)
                            if cleaned_review:
                                reviews.add(cleaned_review)

        # Method 2: Common review selectors
        review_selectors = [
            '[itemprop="reviewBody"]',
            '.review-content',
            '.review-text',
            '.review-body',
            '[class*="review-"]',
            '[class*="comment-"]',
            '.user-review'
        ]
        
        for selector in review_selectors:
            elements = soup.select(selector)
            for element in elements:
                review_text = element.get_text()
                cleaned_review = self.clean_review(review_text)
                if cleaned_review:
                    reviews.add(cleaned_review)

        # Convert set to list and limit the number of reviews
        return list(reviews)[:30]  # Increased limit to 30 reviews

    def get_product_info(self, url: str) -> Dict:
        html = self.make_request(url)
        if not html:
            return {"error": "Failed to fetch product info"}

        soup = BeautifulSoup(html, 'html.parser')
        
        # Get basic product info
        title = self.extract_title(soup)
        price = self.extract_price(soup)
        rating = self.extract_rating(soup)
        reviews = self.extract_reviews(soup)

        return {
            "title": title or "Could not extract",
            "price": price or "Could not extract",
            "rating": rating or "Could not extract",
            "reviews": reviews if reviews else ["No reviews found"]
        }

    def extract_title(self, soup) -> Optional[str]:
        selectors = [
            'meta[property="og:title"]',
            'meta[name="title"]',
            'h1',
            'title'
        ]
        for sel in selectors:
            tag = soup.select_one(sel)
            if tag:
                return tag.get("content") if tag.has_attr("content") else tag.get_text(strip=True)
        return None

# Create singleton instance and expose scrape_product function
_scraper = UniversalScraper()

def scrape_product(url: str) -> Dict:
    return _scraper.get_product_info(url)
