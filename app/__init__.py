"""
SMART COMPOST - MODEL PROJECT.

--- __init__.py file

* Author  -  Enos Muthiani
* Git     -  https://github.com/lyznne
* Date    -  22 Nov 2024
* Email   -  emuthiani26@gmail.com

                        Copyright (c) 2024 - enos.vercel.app
"""

# Import modules/packages
import os
import secrets
import click
from flask import Flask, render_template
from flask_migrate import Migrate
from flask_socketio import SocketIO
from dotenv import load_dotenv
from .config import DevelopmentConfig
from .util import gen_token, setup_logging, setup_network_permissions
from app.models import db, login_manager
from flask_session import Session
from flask_wtf.csrf import CSRFProtect
import logging
import sys


# Load environment variables
load_dotenv()

# Initialize Flask extensions
migrate = Migrate()
socketio = SocketIO(logger=True, engineio_logger=True)
csrf  =  CSRFProtect()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="smart_compost.log",
)
logger = logging.getLogger(__name__)


def register_extensions(app):
    """Initialize Flask extensions."""
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)


def register_blueprints(app):
    """Dynamically register Flask blueprints."""
    from app.views import blueprint, auth_blueprint
    from app.api import api_blueprint

    app.register_blueprint(blueprint, url_prefix="/")
    app.register_blueprint(auth_blueprint, url_prefix="/auth/")
    app.register_blueprint(api_blueprint, url_prefix="/api/")


def configure_database(app):
    """Configure the database for the Flask app."""
    with app.app_context():
        from app.models import (
            seed_environmental_data,
            create_sample_user,
        )

    @app.teardown_request
    def shutdown_session(exception=None):
        """Clean up the session after each request."""
        db.session.remove()


# The user_loader function
@login_manager.user_loader
def load_user(user_id):
    """Flask-Login user loader function."""
    from app.models import Users

    return Users.query.get(int(user_id))


def init_app(app):
    """Register CLI commands for Flask."""
    from app.models import seed_environmental_data, create_sample_user

    @app.cli.command("seed")
    def seed_db():
        """Seed the database with initial data."""
        seed_environmental_data()
        create_sample_user()
        click.echo("✅ Database seeded successfully.")

    @app.cli.command("key-generate")
    def key_generate():
        """Generate and save a secret key token to the .env file."""
        try:
            gen_token()
            click.echo("✅ Secret key generated successfully.")
        except Exception as e:
            click.echo(f"❌ Error generating secret key: {e}", err=True)


def register_error_handlers(app):
    """Registers error handlers for 404, 500, 403, and authentication errors."""

    @app.errorhandler(401)
    def unauthorized_error(error):
        return (
            render_template(
                "errors/error.html", error_code=401, message="Unauthorized Access"
            ),
            401,
        )

    @app.errorhandler(403)
    def forbidden_error(error):
        return (
            render_template(
                "errors/error.html", error_code=403, message="Access Forbidden"
            ),
            403,
        )

    @app.errorhandler(404)
    def not_found_error(error):
        return (
            render_template(
                "errors/error.html", error_code=404, message="Page Not Found"
            ),
            404,
        )

    @app.errorhandler(500)
    def internal_server_error(error):
        return (
            render_template(
                "errors/error.html",
                error_code=500,
                message="Internal Server Error",
            ),
            500,
        )


def create_app(config_name=DevelopmentConfig):
    """Application factory for creating a Flask app instance."""
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config_name)

    # Set up custom logs
    setup_logging()

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY_TOKEN", secrets.token_hex(32))
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    #  Enable WebSockets before other extensions
    socketio.init_app(app, cors_allowed_origins="*")

    # Register extensions, blueprints, and database configuration
    register_extensions(app)
    register_blueprints(app)
    register_error_handlers(app)

    # initialize th CSRF protection
    csrf.init_app(app)

    # Register CLI commands
    init_app(app)
    if not setup_network_permissions():
        print(
            "Failed to setup network permissions. Check network_permissions.log for details."
        )
        sys.exit(1)

    # Import models AFTER initializing `db` to prevent circular import issues
    with app.app_context():
        from app import models

    configure_database(app)

    # sesion
    Session(app)

    return app
