"""
SMART COMPOST - MODEL PROJECT.

--- __init__.py file

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com



                        Copyright (c) 2024 - enos.vercel.app
"""

# Import modules/packages
from datetime import timedelta
import os
import secrets
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from .config  import DevelopmentConfig


from .util import gen_token, setup_logging

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Base directory of the Smart Compost application
basedir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Initialize Flask extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


def register_extensions(app):
    """
    Initialize Flask extensions.
    """
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)


def register_blueprints(app):
    """
    Dynamically register Flask blueprints.
    """
    from app.views import blueprint,  auth_blueprint
    app.register_blueprint(blueprint, url_prefix="/")
    app.register_blueprint(auth_blueprint, url_prefix="/auth/")
    # for module_name in ("views",):
    #     module = import_module(f"app.{module_name}")
    #     app.register_blueprint(module.blueprint)


def configure_database(app):
    """
    Configure the database for the Flask app.
    """

    with app.app_context():
        # Import sample data creation here to avoid circular import
        from app.models import create_sample_user

        """
        Automatically create tables on the application startup.
        """
        db.create_all()

        # Seed sample data
        create_sample_user()

    @app.teardown_request
    def shutdown_session(exception=None):
        """
        Clean up the session after each request.
        """
        db.session.remove()


# The user_loader function
@login_manager.user_loader
def load_user(user_id):
    from app.models import Users
    return Users.query.get(user_id)


def create_app(config_name=DevelopmentConfig):
    """
    Application factory for creating a Flask app instance.
    """

    app = Flask(__name__)

    # Load configuration from the provided config class
    app.config.from_object(config_name)

    # Set up custom logs
    setup_logging()

    # Generate and set a secret key
    gen_token()
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY_TOKEN", secrets.token_hex(32))
    app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME")
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Configure the database

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"sqlite:///{os.path.join(db_dir, os.getenv('SQLALCHEMY_DATABASE_NAME', 'smart-compost.db'))}"
    )

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)
    app.config["SESSION_PROTECTION"] = "strong"

    # Register extensions, blueprints, and database configuration
    register_extensions(app)
    register_blueprints(app)
    configure_database(app)

    # log info

    return app
