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
from flask import flash, render_template, Blueprint, redirect, url_for
from app.services.notification import NotificationManager
from flask_login import current_user, login_required, login_user, logout_user
from app import login_manager
from app.models import Users, ActivityLog , Device , Notification
from app.util import get_client_ip, register_device, verify_pass
from flask import request


# Define the blueprint for routes
blueprint = Blueprint("app", __name__)
auth_blueprint = Blueprint("auth", __name__)



# Home route
@blueprint.route("/", methods=["GET"])
@login_required
def home():
    """
    Home route (requires login).
    """

    return render_template("app/index.html", user=current_user, active_tab='home')


@blueprint.route('/devices', methods=['GET'])
@login_required
def manage_devices():
    # Get user's devices
    user_devices = Device.query.filter_by(user_id=current_user.id).all()

    # Get network status of IoT devices if applicable
    iot_devices = [d for d in user_devices if d.device_type == 'iot']

    return render_template(
        'app/index.html',
        devices=user_devices,
        iot_devices=iot_devices,
        user=current_user,
        active_tab='wifi'
    )

# Wifi
@blueprint.route("/wifi", methods=["GET", "POST"])
@login_required
def wifi():
    """
    WIFI Route ...
    """
    return render_template("app/index.html", user=current_user, active_tab='wifi')

# STATS
@blueprint.route("/stats", methods=["GET", "POST"])
@login_required
def stats():
    """
    STATS route ....
    """
    return render_template("app/index.html", user=current_user, active_tab='stats')


# PROFILE
@blueprint.route("/profile", methods=["GET", "POST"]) # url should be `profile/{username}`
@login_required
def profile():
    """
    User Profile  ...
    """
    return render_template("app/index.html", user=current_user, active_tab='profile')


# Login route
@auth_blueprint.route("/signin", methods=["GET", "POST"])
def signin():
    """Handle user signin"""

    if current_user.is_authenticated:
        return redirect(url_for("app.home"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = request.form.get("remember") == "on"

        user = Users.query.filter_by(email=email).first()

        if user is None:
            flash("User does not exist", "danger")

            return render_template("auth/signin.html")

        # Verify the password
        if user.verify_password(password):
            try:

                login_user(user, remember=remember)
                # Register user's device
                register_device(user.id, request)
                flash("Welcome back!", "success")
                
                return redirect(url_for("app.home"))
            except Exception as e:
                print("Error in login_user:", e)
        else:

            flash("Invalid email or password", "error")
            return render_template("auth/signin.html")

    return render_template("auth/signin.html")


# Signup
@auth_blueprint.route('/signup', methods=["GET", "POST"])
def signup():
    """User Registration with appropriate information"""
    if request.method == "POST":
        # Extract form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        phone = request.form.get('phone')  # Optional
        location = request.form.get('location')  # Optional

        # Validate required fields
        if not all([first_name, last_name, email, password]):
            flash('Required fields cannot be empty', 'error')
            return render_template("auth/signup.html")

        # Check for existing user
        existing_user = Users.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered', 'error')
            return render_template("auth/signup.html")

        # Create new user
        user = Users(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            phone=phone,
            location=location,
            ip_address=get_client_ip()
        )
        db.session.add(user)
        db.session.commit()

        # Log the registration activity
        activity = ActivityLog(
            user_id=user.id,
            activity_type="Registration",
            ip_address=get_client_ip(),
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(activity)

        # Register user's device
        register_device(user.id, request)

        # Log the user in
        login_user(user)
        user.update_last_login()
        NotificationManager.create_notification(user.id, "Welcome to the smart compost system")

        # Redirect to onboarding or dashboard
        return redirect(url_for('app.home'))

    return render_template("auth/signup.html")

@auth_blueprint.route('/complete_profile', methods=["GET", "POST"])
@login_required
def complete_profile():
    """Additional profile information specifically for compost monitoring"""
    user = current_user

    if request.method == "POST":
        # Get compost preferences and settings
        compost_experience = request.form.get('compost_experience')
        preferred_materials = request.form.get('preferred_materials')
        garden_size = request.form.get('garden_size')
        notification_preferences = request.form.get('notification_preferences', 'email')

        # Store this data in a user preferences table or extension
        # Since it's not in your current model, we'll need to add a new model

        # For now, log this activity
        activity = ActivityLog(
            user_id=user.id,
            activity_type="Profile Completion",
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(activity)
        db.session.commit()

        # Create a notification
        notification = Notification(
            user_id=user.id,
            message="Profile completed! You're all set to start monitoring your compost."
        )
        db.session.add(notification)
        db.session.commit()

        flash('Profile information saved successfully!', 'success')
        return redirect(url_for('dashboard.index'))

    return render_template("auth/complete_profile.html", user=user)


@auth_blueprint.route('/profile', methods=["GET", "POST"])
@login_required
def update_profile():
    """Update user profile settings"""
    user = current_user

    if request.method == "POST":
        # Update basic information
        user.first_name = request.form.get('first_name', user.first_name)
        user.last_name = request.form.get('last_name', user.last_name)
        user.email = request.form.get('email', user.email)
        user.phone = request.form.get('phone')
        user.location = request.form.get('location')

        # Handle password change
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')

        if current_password and new_password:
            if user.verify_password(current_password):
                user.password = new_password
            else:
                flash('Current password is incorrect', 'error')
                return render_template("auth/profile.html", user=user)

        # Handle avatar upload if provided
        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            if avatar_file and allowed_file(avatar_file.filename):
                filename = secure_filename(f"user_{user.id}_{int(time.time())}.png")
                avatar_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                user.avatar = filename

        # Log the update
        activity = ActivityLog(
            user_id=user.id,
            activity_type="Profile Update",
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(activity)

        db.session.commit()
        flash('Profile updated successfully!', 'success')

    # Get user's recent activities
    recent_activities = ActivityLog.query.filter_by(user_id=user.id).order_by(ActivityLog.timestamp.desc()).limit(5)

    # Get user's devices
    devices = Device.query.filter_by(user_id=user.id).all()

    return render_template(
        "auth/profile.html",
        user=user,
        activities=recent_activities,
        devices=devices
    )


# Forgot password route
@auth_blueprint.route("/forgot-password", methods=["GET"])
def forgot_password():
    """
    Render the forgot password page.
    """
    return render_template("auth/forgot-password.html")


# Logout user
@auth_blueprint.route("/signout")
@login_required
def signout():
    logout_user()
    flash("You have been logged out", "info")
    return redirect(url_for("auth.signin"))


# Single unauthorized handler for Flask-Login
@login_manager.unauthorized_handler
def unauthorized_handler():
    """
    Redirect unauthorized users to the login page.
    """
    flash("You must be logged in to access this page.", "warning")
    return redirect(url_for("auth.signin"))
