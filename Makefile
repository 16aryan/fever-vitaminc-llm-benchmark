.PHONY: setup test lint format clean scorecard report demo rag

## Install dependencies
setup:
	uv sync

## Run the complete test suite (162 tests, 100% offline, no Ollama needed)
test:
	python3 -m unittest discover -s tests -v

## Check formatting and lint
lint:
	python3 -m ruff format --check
	python3 -m ruff check

## Apply formatting and lint fixes
format:
	python3 -m ruff check --fix
	python3 -m ruff format

## Generate and display cross-model benchmark scorecard
scorecard:
	python3 scripts/compare_models.py

## Compile publication-quality PDF technical report via ReportLab
report:
	python3 src/generate_pdf_report.py

## Launch interactive Streamlit Fact-Checking & RAG Dashboard
demo:
	streamlit run app.py

## Run error taxonomy analysis
error-analysis:
	python3 src/analyse_errors.py

## Delete compiled Python files
clean:
	find . -type f -name "*.py[co]" -not -path "./.venv/*" -delete
	find . -type d -name "__pycache__" -not -path "./.venv/*" -delete
