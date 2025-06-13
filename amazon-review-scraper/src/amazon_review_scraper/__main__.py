"""
Main module for amazon_review_scraper.
"""

import logging
import click
from amazon_review_scraper.collector import AmazonReviewDataCollector

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option(
    "--asin-code",
    help="The ASIN code or product URL to scrape Amazon reviews for.",
    required=True,
)
def scrape_amazon_reviews(asin_code: str) -> None:
    collector = AmazonReviewDataCollector()

    if "amazon." in asin_code.lower():
        collector.collect_by_url(asin_code)
    else:
        collector.collect_by_asin(asin_code)

if __name__ == "__main__":
    scrape_amazon_reviews()
