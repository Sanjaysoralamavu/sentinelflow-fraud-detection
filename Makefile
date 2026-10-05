PYTHON ?= python
CONFIG ?= configs/full.yaml

.PHONY: help install format lint test show-config

help:
	@echo "SentinelFlow development commands"
	@echo "  make install      Install project and development dependencies"
	@echo "  make format       Apply Ruff fixes and formatting"
	@echo "  make lint         Check Ruff linting and formatting"
	@echo "  make test         Run the test suite"
	@echo "  make show-config  Validate and display configuration"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

show-config:
	$(PYTHON) -m sentinelflow.cli show-config --config $(CONFIG)