from pydantic_settings import BaseSettings

class AmazonReviewScraperSettings(BaseSettings):
    def get_amazon_product_url(self, asin_code: str) -> str:
        return f"https://www.amazon.com/dp/{asin_code}"

amazon_review_scraper_settings = AmazonReviewScraperSettings()
