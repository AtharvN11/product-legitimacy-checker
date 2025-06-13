# Makefile for running Amazon review scraper


.PHONY: install
install:
	pip install poetry==1.8.2
	poetry install


.PHONY: scrape
scrape:
ifndef ASIN_CODE
	$(error Error: An ASIN code is required. Use `make scrape ASIN_CODE="<asin_code>"`)
endif
	poetry run python -m src.amazon_review_scraper --asin-code="$(ASIN_CODE)"
