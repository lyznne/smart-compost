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

import os
import hashlib
import binascii
import secrets
import logging
from werkzeug.security import generate_password_hash, check_password_hash


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


# def verify_pass(provided_password, stored_password):
#     """Verify a stored password against one provided by the user."""
#     # Ensure stored_password is in bytes
#     if isinstance(stored_password, str):
#         stored_password = stored_password.encode("utf-8")

#     try:
#         salt = stored_password[:64]  # Extract the salt
#         stored_password = stored_password[64:]  # Extract the hashed password

#         pwdhash = hashlib.pbkdf2_hmac(
#             "sha512", provided_password.encode("utf-8"), salt, 100000
#         )
#         pwdhash = binascii.hexlify(pwdhash)  # Convert to hex for comparison

#         return pwdhash == stored_password
#     except Exception as e:
#         # Log the error for debugging
#         print(f"Password verification failed: {e}")
#         return False

# def hash_pass(password):
#     """Hash a password for storing."""
#     password = str(password)  # Convert to string if it's not
#     salt = hashlib.sha256(os.urandom(60)).hexdigest().encode("ascii")
#     pwdhash = hashlib.pbkdf2_hmac("sha512", password.encode("utf-8"), salt, 100000)
#     pwdhash = binascii.hexlify(pwdhash)
#     return salt + pwdhash  # return bytes
