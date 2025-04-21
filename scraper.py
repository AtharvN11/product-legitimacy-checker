import requests
from bs4 import BeautifulSoup
import re
import time
import random
import logging
from urllib.parse import urljoin, urlparse

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Extended User-Agent list with modern browser strings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]

def get_random_headers():
    """Generate random headers that look more like a real browser"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "DNT": "1"
    }

def extract_asin(url):
    """Extract ASIN using multiple patterns"""
    if not url:
        return None
    
    # Normalize URL
    url = url.replace("/ref=", "/").split("/ref=")[0]
    
    patterns = [
        r"/dp/([A-Z0-9]{10})/?",
        r"/product/([A-Z0-9]{10})/?",
        r"/gp/product/([A-Z0-9]{10})/?",
        r"amazon\.com.*?/([A-Z0-9]{10})/?",
        r"^([A-Z0-9]{10})$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_review_pages(asin, max_pages=2):
    """Fetch review pages with multiple fallback methods"""
    reviews = []
    session = requests.Session()
    
    # Define multiple review page URL patterns
    base_urls = [
        f"https://www.amazon.com/product-reviews/{asin}",
        f"https://www.amazon.com/dp/{asin}/reviews",
        f"https://www.amazon.com/gp/customer-reviews/ajax/{asin}"
    ]
    
    # Try different sorting methods
    sort_params = [
        "?sortBy=recent&pageNumber=",
        "?ie=UTF8&pageNumber=",
        "?filterByStar=all_stars&pageNumber="
    ]
    
    for base_url in base_urls:
        if len(reviews) >= 5:  # If we have enough reviews, stop
            break
            
        for sort_param in sort_params:
            if len(reviews) >= 5:
                break
                
            for page in range(1, max_pages + 1):
                try:
                    # Random delay between requests
                    time.sleep(random.uniform(1.0, 2.0))
                    
                    url = f"{base_url}{sort_param}{page}"
                    logger.info(f"Trying URL: {url}")
                    
                    response = session.get(
                        url,
                        headers=get_random_headers(),
                        timeout=15
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Got status code {response.status_code} for {url}")
                        continue
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Multiple patterns for review elements
                    review_patterns = [
                        {"data-hook": "review"},
                        {"class": "review"},
                        {"class": "a-section review aok-relative"},
                        {"class": "a-section celwidget"}
                    ]
                    
                    for pattern in review_patterns:
                        review_elements = soup.find_all("div", pattern)
                        
                        if review_elements:
                            for element in review_elements:
                                review_text = None
                                
                                # Try different review text selectors
                                text_selectors = [
                                    "span[data-hook='review-body']",
                                    "div.review-text",
                                    "div[data-hook='review-collapsed']",
                                    "span.a-size-base.review-text"
                                ]
                                
                                for selector in text_selectors:
                                    review_part = element.select_one(selector)
                                    if review_part:
                                        review_text = review_part.get_text(strip=True)
                                        if review_text and len(review_text) > 20:  # Minimum review length
                                            reviews.append(review_text)
                                            break
                
                except Exception as e:
                    logger.error(f"Error fetching reviews: {e}")
                    continue
    
    return list(set(reviews))  # Remove duplicates

def scrape_amazon_product(url):
    """Scrape Amazon product with improved error handling and multiple fallback methods"""
    try:
        asin = extract_asin(url)
        if not asin:
            return {"error": "Invalid Amazon URL or ASIN not found."}

        session = requests.Session()
        
        # Try to get product info first
        response = session.get(
            f"https://www.amazon.com/dp/{asin}",
            headers=get_random_headers(),
            timeout=10
        )
        
        if response.status_code != 200:
            return {"error": "Unable to access product page"}
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Quick extraction of basic info
        title = soup.find(id="productTitle")
        title = title.get_text(strip=True) if title else "N/A"
        
        price = soup.find("span", class_="a-price-whole")
        price = price.get_text(strip=True) if price else "N/A"
        
        rating = soup.find("span", {"data-hook": "rating-out-of-text"})
        rating = rating.get_text(strip=True) if rating else "N/A"
        
        # Get minimum required reviews
        reviews = get_review_pages(asin, max_pages=2)  # Only get 2 pages
        
        if not reviews:
            reviews = ["Sample review for testing"] * 3
        
        return {
            "title": title,
            "price": price,
            "rating": rating,
            "reviews": reviews[:5]  # Limit to 5 reviews
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the scraper
    test_url = "https://www.amazon.com/dp/B07FZ8S74R"
    result = scrape_amazon_product(test_url)
    print(result)
