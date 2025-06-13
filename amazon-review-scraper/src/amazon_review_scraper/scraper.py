import logging
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from amazon_review_scraper.conf import amazon_review_scraper_settings
from amazon_review_scraper.models import Review

logging.getLogger("WDM").setLevel(logging.ERROR)

class DriverInitializationError(BaseException):
    pass

class DriverGetReviewsError(BaseException):
    pass

class AmazonReviewScraper:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def _init_chrome_driver(self) -> webdriver.Chrome:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920x1080")
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def _parse_review_data(self, review: WebElement) -> Review:
        try:
            author = review.find_element(By.CLASS_NAME, "a-profile-name").text.strip()
        except:
            author = "Unknown"

        try:
            content = review.find_element(By.CLASS_NAME, "reviewText").text.strip()
        except:
            content = ""

        try:
            rating_text = review.find_element(By.CLASS_NAME, "a-icon-alt").get_attribute("innerHTML").strip()
            rating = float(rating_text.split(" out of")[0])
        except:
            rating = 0.0

        try:
            title = review.find_element(By.CSS_SELECTOR, ".review-title span").text.strip()
        except:
            title = "No Title"

        return Review(author=author, content=content, rating=rating, title=title)

    def _get_reviews_from_product_page(self, url: str, driver: webdriver.Chrome) -> List[Review]:
        driver.get(url)
        try:
            review_container = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "cm-cr-dp-review-list"))
            )
            review_elements = review_container.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
        except Exception:
            self._logger.warning("Timed out waiting for reviews to load.")
            return []

        parsed = []
        for elem in review_elements:
            try:
                parsed.append(self._parse_review_data(elem))
            except Exception:
                self._logger.exception("Error parsing a review, skipping.")
                continue
        return parsed

    def scrape_amazon_reviews(self, asin_code: str) -> List[Review]:
        self._logger.info(f"Scraping Amazon Reviews for ASIN {asin_code}...")
        try:
            driver = self._init_chrome_driver()
        except Exception as e:
            raise DriverInitializationError from e

        url = amazon_review_scraper_settings.get_amazon_product_url(asin_code)
        try:
            return self._get_reviews_from_product_page(url, driver)
        except Exception as e:
            raise DriverGetReviewsError from e
        finally:
            driver.quit()
