"""
SMART COMPOST - MODEL PROJECT.

---  views.py file

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# import modules
from flask import Response, flash, render_template, Blueprint, redirect, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app import login_manager
from app.models import Users
from app.util import verify_pass
from flask import request

# Define the blueprint for routes
blueprint = Blueprint("main", __name__)


# Home route
@blueprint.route("/", methods=["GET"])
@login_required
def home():
    """
    Home route (requires login).
    """

    return render_template("app/index.html", user=current_user)


# Login route
@blueprint.route("/signin", methods=["GET", "POST"])
def signin():
    """Handle user signin"""

    # if current_user.is_authenticated:
    #     print("User already authenticated:", current_user.email)
    #     return redirect(url_for(".home"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = request.form.get("remember") == "on"

        user = Users.query.filter_by(email=email).first()

        if user is None:
            flash("User does not exist", "danger")

            return render_template("auth/signin.html")

        print(user.verify_password(password))
        # Verify the password
        if user.verify_password(password):
            try:

                login_user(user, remember=remember)
                flash("Welcome back!", "success")
                return redirect(url_for(".home"))
            except Exception as e:
                print("Error in login_user:", e)
        else:

            flash("Invalid email or password", "error")
            return render_template("auth/signin.html")

    return render_template("auth/signin.html")


# Forgot password route
@blueprint.route("/forgot-password", methods=["GET"])
def forgot_password():
    """
    Render the forgot password page.
    """
    return render_template("auth/forgot-password.html")


# Logout user
@blueprint.route("/signout")
@login_required
def signout():
    logout_user()
    flash("You have been logged out", "info")
    return redirect(url_for(".signin"))


# Single unauthorized handler for Flask-Login
@login_manager.unauthorized_handler
def unauthorized_handler():
    """
    Redirect unauthorized users to the login page.
    """
    flash("You must be logged in to access this page.", "warning")
    return redirect(url_for("main.signin"))
