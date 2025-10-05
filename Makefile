# ========== Credit Card Fraud Detection ==========
# Basic automation commands for development

.PHONY: install lint format test clean

install:
	pip install -r requirements.txt
	pre-commit install

lint:
	flake8 src tests

format:
	isort .
	black .

test:
	pytest -v --maxfail=1 --disable-warnings

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
