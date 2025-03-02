# Variables
VENV_DIR = .flask-venv
PYTHON = python3
PIP = pip3
APP = run.py
REQ_FILE = requirements.txt

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make venv    - Create and setup virtual environment"
	@echo "  make run     - Run the Flask app"
	@echo "  make setup   - Install dependencies from $(REQ_FILE)"
	@echo "  make clean   - Remove cache files and virtual environment"
	@echo "  make activate - Activate virtual environment"

# Create virtual environment and install requirements
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && $(PIP) install -r $(REQ_FILE)
	@echo "Virtual environment created. Activate it with: source $(VENV_DIR)/bin/activate.fish"

# Run the Flask app
.PHONY: run
run:
	$(PYTHON) $(APP)

# Setup: Install requirements (assumes venv is activated)
.PHONY: setup
setup:
	$(PIP) install -r $(REQ_FILE)


# Activate: Activating the venv
.PHONY: activate
activate:
	@echo "Activating virtual environment..."
	@exec fish -c "source $(VENV_DIR)/bin/activate.fish && exec fish"


# Clean: Remove Python cache
.PHONY: clean
clean:
	rm -rf __pycache__
	find . -type f -name '*.pyc' -delete


# Clean: Remove Python cache and virtual environment
.PHONY: clean-up
clean-up:
	rm -rf __pycache__
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete

.PHONY: routes
routes:
	exec flask routes


.PHONY: build run test clean

build:
	docker-compose build

run-docker:
	docker-compose up

test:
	docker-compose run flask_app pytest

clean-docker:
	docker-compose down
	docker system prune -f

lint:
	docker-compose run flask_app black .
	docker-compose run flask_app flake8 .
