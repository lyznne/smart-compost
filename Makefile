# Variables
VENV_DIR = .flask-venv
PYTHON = python3
PIP = pip3
APP = run.py
REQ_FILE = requirements.txt
GEN_SCRIPT = scripts/synthetic_dataset_generator.py


# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make venv        - Create and setup virtual environment"
	@echo "  make run         - Run the Flask app (standard)"
	@echo "  make run-socket  - Run the Flask app with Flask-SocketIO"
	@echo "  make setup       - Install dependencies from $(REQ_FILE)"
	@echo "  make clean       - Remove Python cache files"
	@echo "  make clean-full  - Remove cache files and virtual environment"
	@echo "  make activate    - Activate virtual environment"
	@echo "  make routes      - Show Flask routes"
	@echo "  make db-init     - Initialize Flask database migrations (run once)"
	@echo "  make db-migrate  - Generate a new migration (use MSG='your message')"
	@echo "  make db-upgrade  - Apply migrations to the database"
	@echo "  make build       - Build Docker Compose services"
	@echo "  make run-docker  - Run Docker Compose services"
	@echo "  make test        - Run tests with pytest in Docker"
	@echo "  make clean-docker- Clean Docker Compose services and prune"
	@echo "  make lint        - Lint code with black and flake8 in Docker"
	@echo "  make generate-dataset    - Generates a new sythentic dataset for training smart compost model"

# Generate sythentic dataset
.PHONY: generate-dataset
generate-dataset:
	$(PYTHON) $(GEN_SCRIPT)


# Create virtual environment and install requirements
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && $(PIP) install -r $(REQ_FILE)
	@echo "Virtual environment created. Activate it with: source $(VENV_DIR)/bin/activate.fish"

# Run the Flask app (standard)
.PHONY: run
run:
	$(PYTHON) $(APP)

# Run the Flask app with Flask-SocketIO
.PHONY: run-socket
run-socket:
	FLASK_APP=$(APP) flask run --with-threads
	# $(PYTHON) -m flask_socketio run
	# Alternative: python -m flask_socketio run (uncomment if preferred)
	# $(PYTHON) -m flask_socketio run

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
	rm -rf __pycache__ */__pycache__
	find . -type f -name '*.pyc' -delete

# Clean-full: Remove Python cache and virtual environment
.PHONY: clean-full
clean-full:
	rm -rf __pycache__
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete

# Show Flask routes
.PHONY: routes
routes:
	FLASK_APP=$(APP) flask routes

# Initialize Flask database migrations (run once)
.PHONY: db-init
db-init:
	FLASK_APP=$(APP) flask db init

# Generate a new migration (user can specify message with MSG='...')
.PHONY: db-migrate
db-migrate:
ifdef MSG
	FLASK_APP=$(APP) flask db migrate -m "$(MSG)"
else
	FLASK_APP=$(APP) flask db migrate -m "Unnamed migration"
endif

# Apply migrations to the database
.PHONY: db-upgrade
db-upgrade:
	FLASK_APP=$(APP) flask db upgrade

# Build Docker Compose services
.PHONY: build
build:
	docker-compose build

# Run Docker Compose services
.PHONY: run-docker
run-docker:
	docker-compose up

# Run tests with pytest in Docker
.PHONY: test
test:
	docker-compose run flask_app pytest

# Clean Docker Compose services and prune
.PHONY: clean-docker
clean-docker:
	docker-compose down
	docker system prune -f

# Lint code with black and flake8 in Docker
.PHONY: lint
lint:
	docker-compose run flask_app black .
	docker-compose run flask_app flake8 .
