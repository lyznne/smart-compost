"""
SMART COMPOST - MODEL PROJECT.

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# import modules
from datetime import timedelta
import os
from typing import Any
from dotenv import load_dotenv

# ---
load_dotenv()

# base directory of the Smart compost Application
basedir: str | Any = os.path.abspath(os.path.dirname(__file__))


# Init config class
class Config:
    """Holds configuration variables."""

    SECRET_KEY = os.environ.get("SECRET_KEY_TOKEN", "default_secret_key")
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.example.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "true").lower() in [
        "true",
        "1",
        "yes",
    ]
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "your_email@example.com")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "your_password")
    SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME")

    # Correct SQLAlchemy Database URI
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "SQLALCHEMY_DATABASE_URI",
        f"sqlite:///{os.path.join(basedir, 'smart-compost.db')}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REMEMBER_COOKIE_DURATION = timedelta(days=7)
    SESSION_PROTECTION = "strong"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    # PostgreSQL database
    # SQLALCHEMY_DATABASE_URI = "{}://{}:{}@{}:{}/{}".format(
    #     config("DB_ENGINE", default="postgresql"),
    #     config("DB_USERNAME", default="appseed"),
    #     config("DB_PASS", default="pass"),
    #     config("DB_HOST", default="localhost"),
    #     config("DB_PORT", default=5432),
    #     config("DB_NAME", default="appseed-flask"),
    # )


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
