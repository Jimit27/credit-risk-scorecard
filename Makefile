.PHONY: help install install-spark data pipeline train app test lint clean

help:
	@echo "install        Core dependencies (no Spark, no JVM needed)"
	@echo "install-spark  Adds PySpark, for rebuilding the medallion layers"
	@echo "data           Download the raw Give Me Some Credit extract"
	@echo "pipeline       Full run: bronze -> silver -> gold -> train -> monitor -> figures"
	@echo "train          Retrain and regenerate reports from the existing gold table"
	@echo "app            Launch the Streamlit scoring app"
	@echo "test           Run the test suite"
	@echo "lint           Ruff"
	@echo "clean          Remove built layers, models and reports"

install:
	pip install -r requirements-dev.txt
	# Editable install: without it `python -m creditrisk.pipeline` cannot find
	# the package, because the source lives under src/ rather than the repo root.
	pip install -e . --no-deps

install-spark:
	pip install "setuptools==75.8.0" wheel
	pip install -r requirements-spark.txt -r requirements-dev.txt
	pip install -e . --no-deps

data:
	python scripts/download_data.py

pipeline: data
	python -m creditrisk.pipeline all

train:
	python -m creditrisk.pipeline train monitor figures

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests -v

lint:
	ruff check src tests app notebooks

clean:
	rm -rf data/bronze/* data/silver/* data/gold/* models/*.joblib reports/*.json reports/*.csv reports/figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
