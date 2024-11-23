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


# home route
@blueprint.route("/", methods=["GET"])
@login_required
def home():
    """
    Home
    """
    print("current_user", current_user)
    return render_template("app/index.html", user=current_user)


# login route
@blueprint.route("/signin", methods=("GET", "POST"))
def signin() -> Response:
    """handle user signin"""

    if current_user.is_authenticated:
        return redirect(url_for(".home"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = request.form.get("remember") == "on"

        user = Users.query.filter_by(email=email).first()

        if user is None:
            flash("User does not exist", "danger")
            return render_template("auth/signin.html")
        # Verify the password
        if verify_pass(password, user.password):
            login_user(user, remember=remember)
            flash("Welcome back!", "success")
            return redirect(url_for(".home"))
        else:
            flash("Invalid email or password", "error")
            return render_template("auth/signin.html")

    return render_template("auth/signin.html")


@blueprint.route("/forgot-password", methods=["GET"])
def forgot_password():
    """
    Render the forgot password page.
    """
    return render_template("auth/forgot-password.html")


# logout user
@blueprint.route("/signout")
@login_required
def signout():
    logout_user()
    flash("You have been logged out", "info")
    return redirect(url_for(".signin"))


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template("errors/404.html"), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template("errors/500.html"), 500


# Single unauthorized_handler
@login_manager.unauthorized_handler
def unauthorized_handler():
    """
    Redirect unauthorized users to the login page or show a custom page.
    """
    flash("You must be logged in to access this page.", "warning")
    return redirect(url_for(".signin"))


# @login_manager.unauthorized_handler
# def unauthorized_handler():
#     return render_template("errors/403.html"), 403


# @blueprint.errorhandler(403)
# def access_forbidden(error):
#     return render_template("errors/403.html"), 403
