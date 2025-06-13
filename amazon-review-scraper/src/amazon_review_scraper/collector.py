#!/usr/bin/env python3

import re
import logging
import os
from typing import List, Sequence
import pandas as pd
from amazon_review_scraper.models import Review
from amazon_review_scraper.scraper import AmazonReviewScraper

DEFAULT_OUTPUT_FILE = "amazon_reviews.csv"
_ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")

class AmazonReviewDataCollector:
    def __init__(self, *, output_file: str | None = None, logger: logging.Logger | None = None):
        self._scraper = AmazonReviewScraper()
        self._output_file = output_file or DEFAULT_OUTPUT_FILE
        self._logger = logger or logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def _save_to_csv(self, reviews: Sequence[Review]) -> None:
        self._logger.info(f"Writing {len(reviews)} reviews → {self._output_file}")
        df = pd.DataFrame([r.model_dump() for r in reviews])
        df.to_csv(self._output_file, index=False)

    @staticmethod
    def _asin_from_url(url: str) -> str | None:
        clean = url.split("?", 1)[0]
        match = _ASIN_RE.search(clean)
        return match.group(1) if match else None

    def collect_by_asin(self, asin: str) -> List[Review]:
        self._logger.info(f"Starting scrape for ASIN {asin}")
        try:
            reviews = self._scraper.scrape_amazon_reviews(asin)
        except Exception as exc:
            self._logger.exception("Scrape failure: %s", exc)
            return []

        if not reviews:
            self._logger.warning("No reviews found for ASIN %s", asin)
            if os.path.exists(self._output_file):
                os.remove(self._output_file)
            return []

        self._save_to_csv(reviews)
        return reviews

    def collect_by_url(self, url: str) -> List[Review]:
        asin = self._asin_from_url(url)
        if not asin:
            self._logger.error("Invalid Amazon URL, cannot parse ASIN: %s", url)
            return []
        return self.collect_by_asin(asin)

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Scrape Amazon reviews → CSV")
    parser.add_argument("input", help="Raw ASIN or Amazon product URL")
    parser.add_argument("--output", "-o", help="Custom CSV output path")
    args = parser.parse_args()

    collector = AmazonReviewDataCollector(output_file=args.output)
    if "amazon." in args.input.lower():
        reviews = collector.collect_by_url(args.input)
    else:
        reviews = collector.collect_by_asin(args.input)

    if not reviews:
        sys.exit("No reviews scraped.")
    print(f"Saved {len(reviews)} reviews to {collector._output_file}")
