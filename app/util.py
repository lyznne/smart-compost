"""
SMART COMPOST - MODEL PROJECT.

---  Util file

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# import dependency

from ast import parse
import os
from datetime import datetime
import secrets
import logging
from flask import request
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from app.models import Device

API_KEY  =  os.getenv('IP_API_TOKEN')

# util function to gen secret key
def gen_token():
    """
    Generate a random secret key and save it to the .env file.
    If the variable `SECRET_KEY_TOKEN` does not exist, it is added.
    """
    token = secrets.token_hex(32)  # Generate a secure random token
    file_path = ".env"

    # Check if .env file exists
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write(f"SECRET_KEY_TOKEN={token}\n")
        return

    # Read the content of the .env file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Check if SECRET_KEY_TOKEN exists and update or append it
    key_found = False
    with open(file_path, "w") as file:
        for line in lines:
            if line.startswith("SECRET_KEY_TOKEN="):
                file.write(f"SECRET_KEY_TOKEN={token}\n")
                key_found = True
            else:
                file.write(line)

        if not key_found:
            file.write(f"SECRET_KEY_TOKEN={token}\n")


# my logger
def setup_logging():
    """
    Configure custom logs for the SMART COMPOST project.
    """

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        return

    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("smart-compost")

    #  log messages

    logger.error("")
    logger.info("")
    logger.info("  ->  . ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨")
    logger.info("  ->  . 🌿       SMART COMPOST - MODEL PROJECT       🌿")
    logger.info("  ->  . ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨")
    logger.info("  ->  . 🖊️  Author      : Enos Muthiani")
    logger.info("  ->  . 📧  Email       : emuthiani26@gmail.com")
    logger.info("  ->  . 🌐  Website     : https://enos.vercel.app")
    logger.info("  ->  . 💻  Project Git : https://github.com/lyznne")
    logger.info("  ->  . ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨")
    logger.info("")
    logger.error("")


def hash_pass(password):
    """Hash a password for storing."""
    # Flask's generate_password_hash automatically salts and hashes the password.
    return generate_password_hash(password)


def verify_pass(provided_password, stored_password):
    """Verify a stored password against the provided password."""
    # Flask's check_password_hash verifies the provided password against the stored hash.
    return check_password_hash(stored_password, provided_password)


def import_csv_to_model(csv_file, model_name):
    """
    Reads a CSV file and inserts the data into the specified SQLAlchemy model.

    :param csv_file: Path to the CSV file
    :param model_name: The SQLAlchemy model class (not a string)
    """
    from app import db

    try:
        # Read CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if model exists in the database
        if not hasattr(model_name, '__table__'):
            raise ValueError(f"Model {model_name} is not a valid SQLAlchemy model.")

        # Convert DataFrame to a list of dictionaries
        records = df.to_dict(orient="records")

        # Create model instances dynamically
        objects = [model_name(**record) for record in records]

        # Bulk insert records
        db.session.bulk_save_objects(objects)
        db.session.commit()

        print(f"✅ Successfully imported {len(records)} records into {model_name.__tablename__}")

    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"❌ Database error: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")



def get_client_ip()-> str:
    """Retrieve the real IP address of the user, handling proxy forwarding.

    Returns:
        str: ip address of user (user agent)
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        ip = forwarded_for.split(",")[0]  # Get first IP in the list
    else:
        ip = request.remote_addr
    return ip


def get_device_info()->dict:
    """Extract device details from the User-Agent header.

    Returns:
        dict: device info 1. device name 2. device_type 3. browser 4. os

    """
    user_agent = request.headers.get("User-Agent", "Unknown Device")
    ua = parse(user_agent)

    return {
        "device_name": ua.device.family if ua.device.family else "Unknown Device",
        "device_type": "Mobile" if ua.is_mobile else "Tablet" if ua.is_tablet else "PC",
        "browser": f"{ua.browser.family} {ua.browser.version_string}",
        "os": f"{ua.os.family} {ua.os.version_string}"
    }


def get_location_from_ip(ip: str) -> dict:
    """Retrieve geolocation details based on IP address using ipinfo.io API."""
    url = f"https://ipinfo.io/{ip}/json?token={API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return {
            "city": data.get("city", "Unknown"),
            "region": data.get("region", "Unknown"),
            "country": data.get("country", "Unknown")
        }
    except requests.RequestException:
        return {"city": "Unknown", "region": "Unknown", "country": "Unknown"}

def register_device(user_id: int, push_token: str = None):
    from app import db

    """Register a new device or update last seen for existing devices."""

    user_ip = get_client_ip()
    device_info = get_device_info()
    location_data = get_location_from_ip(user_ip)

    # Check if device already exists
    existing_device = Device.query.filter_by(user_id=user_id, device_ip=user_ip).first()

    if existing_device:
        existing_device.last_seen = datetime.utcnow()
        existing_device.push_token = push_token or existing_device.push_token  # Update push token if provided
    else:
        # Register new device
        new_device = Device(
            user_id=user_id,
            device_name=f"{device_info['device_name']} - {device_info['browser']} ({device_info['os']})",
            device_ip=user_ip,
            device_type=device_info["device_type"],
            browser=device_info["browser"],
            os=device_info["os"],
            location_city=location_data["city"],
            location_region=location_data["region"],
            location_country=location_data["country"],
            push_token=push_token
        )
        db.session.add(new_device)

    db.session.commit()
