# Makefile for AthenaMyst:Divina
# This file provides shortcuts for common development tasks.

# Variables
PYTHON = python
PIP = pip
PYTEST = python -m pytest
BLACK = black .
RUFF = ruff check .
MYPY = mypy .
PRECOMMIT = pre-commit
COVERAGE = coverage

# Default target
.DEFAULT_GOAL := help

# Help target to show all available commands
.PHONY: help
help:
	@echo "AthenaMyst:Divina - Development Commands"
	@echo ""
	@echo "  make install      Install the package in development mode"
	@echo "  make dev          Install development dependencies"
	@echo "  make lint         Run all linters (black, ruff, mypy)"
	@echo "  make format       Format code with black and isort"
	@echo "  make test         Run tests with coverage"
	@echo "  make test-fast    Run tests without coverage"
	@echo "  make test-cov     Show test coverage report"
	@echo "  make test-html    Generate HTML coverage report"
	@echo "  make pre-commit   Install pre-commit hooks"
	@echo "  make clean        Remove build artifacts and cache"
	@echo ""

# Install the package in development mode
.PHONY: install
install:
	$(PIP) install -e .

# Install development dependencies
.PHONY: dev
dev:
	$(PIP) install -e ".[dev]"

# Run all linters
.PHONY: lint
lint: black ruff mypy

# Format code with black and isort
.PHONY: format
format:
	$(BLACK)
	python -m isort .

# Run black code formatter
.PHONY: black
black:
	$(BLACK) --check .

# Run ruff linter
.PHONY: ruff
ruff:
	$(RUFF) .

# Run mypy type checker
.PHONY: mypy
mypy:
	$(MYPY)

# Run tests with coverage
.PHONY: test
test:
	$(PYTEST) --cov=divina --cov-report=term-missing --cov-report=xml

# Run tests without coverage
.PHONY: test-fast
test-fast:
	$(PYTEST) -v

# Show test coverage report
.PHONY: test-cov
test-cov:
	$(COVERAGE) report -m

# Generate HTML coverage report
.PHONY: test-html
test-html:
	$(COVERAGE) html
	@echo "Open htmlcov/index.html in your browser to view the coverage report."

# Install pre-commit hooks
.PHONY: pre-commit
pre-commit:
	$(PRECOMMIT) install

# Clean build artifacts and cache
.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage coverage.xml
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

# Run all checks (lint, test, etc.)
.PHONY: check
check: lint test

# Run pre-commit on all files
.PHONY: pre-commit-all
pre-commit-all:
	$(PRECOMMIT) run --all-files

# Update dependencies
.PHONY: update-deps
update-deps:
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

# Run the application
.PHONY: run
run:
	python -m divina

# Run the application in development mode with auto-reload
.PHONY: dev-run
dev-run:
	uvicorn divina.main:app --reload

# Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t athenamyst/divina:latest .

# Run the Docker container
.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 athenamyst/divina:latest

# Clean Docker resources
.PHONY: docker-clean
docker-clean:
	docker system prune -f
	docker images -q athenamyst/divina | xargs -r docker rmi -f

# Generate requirements files
.PHONY: requirements
requirements:
	$(PIP) freeze > requirements.txt
	$(PIP) freeze | grep -v "athenamyst-divina" > requirements.txt
	$(PIP) freeze | grep -v "athenamyst-divina" | grep -v "pkg-resources" > requirements.txt

# Run security checks
.PHONY: security
security:
	safety check
	bandit -r divina

# Generate documentation
.PHONY: docs
docs:
	cd docs && make html

# Show help for Makefile
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install      - Install the package in development mode"
	@echo "  make dev          - Install development dependencies"
	@echo "  make lint         - Run all linters (black, ruff, mypy)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make test         - Run tests with coverage"
	@echo "  make test-fast    - Run tests without coverage"
	@echo "  make test-cov     - Show test coverage report"
	@echo "  make test-html    - Generate HTML coverage report"
	@echo "  make pre-commit   - Install pre-commit hooks"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo "  make check        - Run all checks (lint, test)"
	@echo "  make run          - Run the application"
	@echo "  make dev-run      - Run with auto-reload"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make requirements - Generate requirements.txt"
	@echo "  make security     - Run security checks"
	@echo "  make docs         - Generate documentation"

# .PHONY ensures that these targets are always run, even if files with those names exist
.PHONY: install dev lint format black ruff mypy test test-fast test-cov test-html pre-commit clean check run dev-run docker-build docker-run docker-clean requirements security docs help
