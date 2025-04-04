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
from flask import flash, jsonify, render_template, Blueprint, redirect, url_for
from app.services import network_manager
from app.services.data_service import DataService
from app.services.network_manager import NetworkManager
from app.services.notification import NotificationManager
from flask_login import current_user, login_required, login_user, logout_user
from app import login_manager
from app.models import Users, ActivityLog, Device, Notification
from app.util import get_client_ip, register_device
from flask import request
from app.models import db
from app import logger
import psutil
import os
import torch
import json
import threading
import paho.mqtt.client as mqtt
from datetime import datetime

# Define the blueprint for routes
blueprint = Blueprint("app", __name__)
auth_blueprint = Blueprint("auth", __name__)

# MQTT Configuration
MQTT_BROKER = "mqtt.broker.com"  # Update with your actual broker address
MQTT_PORT = 1883
MQTT_TOPICS = ["compost/sensors", "compost/status", "compost/alerts"]
mqtt_client = None
latest_sensor_data = {
    "temperature": None,
    "moisture": None,
    "pH": None,
    "timestamp": None
}

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    """Handle connection to MQTT broker"""
    if rc == 0:
        logger.info("Connected to MQTT broker")
        # Subscribe to all monitoring topics
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            logger.info(f"Subscribed to {topic}")
    else:
        logger.error(f"Failed to connect to MQTT broker with result code {rc}")

def on_message(client, userdata, message):
    """Process incoming MQTT messages"""
    topic = message.topic
    try:
        payload = json.loads(message.payload.decode())
        logger.info(f"Received message on topic {topic}: {payload}")

        # Process based on topic
        if topic == "compost/sensors":
            # Update latest sensor data
            global latest_sensor_data
            latest_sensor_data = {
                "temperature": payload.get("temperature"),
                "moisture": payload.get("moisture"),
                "pH": payload.get("pH", None),
                "timestamp": datetime.now()
            }

            # Store in database if user_id is provided
            if "user_id" in payload and payload["user_id"]:
                DataService.store_sensor_reading(
                    user_id=payload["user_id"],
                    temperature=payload.get("temperature"),
                    moisture=payload.get("moisture"),
                    pH=payload.get("pH", None)
                )

            # Check for alert conditions
            check_alert_conditions(payload)

        elif topic == "compost/status":
            # Process status updates
            if "user_id" in payload and "status" in payload:
                DataService.update_compost_status(
                    user_id=payload["user_id"],
                    status=payload["status"]
                )

        elif topic == "compost/alerts":
            # Process explicit alerts
            if "user_id" in payload and "message" in payload:
                NotificationManager.create_notification(
                    user_id=payload["user_id"],
                    message=payload["message"],
                    alert_level=payload.get("level", "info")
                )

    except json.JSONDecodeError:
        logger.error(f"Failed to parse MQTT payload: {message.payload}")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {str(e)}")

def check_alert_conditions(sensor_data):
    """Check if sensor data requires alerts"""
    if "user_id" not in sensor_data:
        return

    user_id = sensor_data["user_id"]

    # Example alert conditions - customize based on your requirements
    if "temperature" in sensor_data:
        temp = sensor_data["temperature"]
        if temp > 65:  # Too hot
            NotificationManager.create_notification(
                user_id=user_id,
                message=f"High temperature alert: {temp}°C",
                alert_level="warning"
            )
        elif temp < 20:  # Too cold
            NotificationManager.create_notification(
                user_id=user_id,
                message=f"Low temperature alert: {temp}°C",
                alert_level="warning"
            )

    if "moisture" in sensor_data:
        moisture = sensor_data["moisture"]
        if moisture < 30:  # Too dry
            NotificationManager.create_notification(
                user_id=user_id,
                message=f"Low moisture alert: {moisture}%",
                alert_level="warning"
            )
        elif moisture > 70:  # Too wet
            NotificationManager.create_notification(
                user_id=user_id,
                message=f"High moisture alert: {moisture}%",
                alert_level="warning"
            )

def start_mqtt_client():
    """Initialize and start MQTT client in a separate thread"""
    global mqtt_client

    try:
        # Create new MQTT client
        mqtt_client = mqtt.Client()

        # Set callbacks
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message

        # Connect to broker
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

        # Start MQTT loop in a separate thread
        mqtt_thread = threading.Thread(target=mqtt_client.loop_forever)
        mqtt_thread.daemon = True  # Thread will close when main application stops
        mqtt_thread.start()

        logger.info("MQTT client started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start MQTT client: {str(e)}")
        return False

# Initialize MQTT client when application starts
def init_mqtt():
    """Initialize MQTT client when Flask app starts"""
    success = start_mqtt_client()
    if not success:
        logger.warning("MQTT client initialization failed. Sensor data may not be available.")

# Call init_mqtt when this module is imported
init_mqtt()


# Home route
@blueprint.route("/", methods=["GET"])
@login_required
def home():
    """
    Home route (requires login).
    """
    compost_data = DataService.get_compost_status(current_user.id)
    recent_activity = DataService.get_recent_activity(current_user.id)
    device = DataService.get_device_info(current_user.id)
    notifications = DataService.get_notifications(current_user.id)
    connected_devices = DataService.get_connected_devices(current_user.id)
    wifi_connected = DataService.get_wifi_status(current_user.id)
    current_time = DataService.get_current_time()

    # Include latest sensor data if available
    if latest_sensor_data["temperature"] is not None:
        compost_data["temperature"] = latest_sensor_data["temperature"]
    if latest_sensor_data["moisture"] is not None:
        compost_data["moisture"] = latest_sensor_data["moisture"]

    return render_template(
        "app/index.html",
        user=current_user,
        active_tab="home",
        compost_status=compost_data["status"],
        temperature=compost_data["temperature"],
        moisture=compost_data["moisture"],
        recent_activity=recent_activity,
        device=device,
        notifications=notifications,
        connected_devices=connected_devices,
        wifi_connected=wifi_connected,
        current_time=current_time,
    )


@blueprint.route("/devices", methods=["GET"])
@login_required
def manage_devices():
    # Get user's devices
    user_devices = Device.query.filter_by(user_id=current_user.id).all()

    # Get network status of IoT devices if applicable
    iot_devices = [d for d in user_devices if d.device_type == "iot"]

    return render_template(
        "app/index.html",
        devices=user_devices,
        iot_devices=iot_devices,
        user=current_user,
        active_tab="wifi",
    )


@blueprint.route("/wifi", methods=["GET", "POST"])
@login_required
def wifi():
    """
    WiFi management route for scanning and connecting to networks.

    GET: Scan and display available networks
    POST: Attempt to connect to a specific network
    """
    # Initialize network manager
    try:
        network_manager = NetworkManager()
    except Exception as init_error:
        logger.error(f"Failed to initialize NetworkManager: {init_error}")
        return render_template(
            "app/index.html",
            active_tab="wifi",
            error="Network interface initialization failed",
            user=current_user,
        )

    # Handle POST request (network connection)
    if request.method == "POST":
        # Validate request data
        data = request.get_json()
        if not data:
            logger.warning("POST request without JSON data")
            return jsonify({"error": "Invalid request"}), 400

        ssid = data.get("ssid")
        password = data.get("password", None)

        # Validate SSID
        if not ssid:
            logger.warning("Connection attempt without SSID")
            return jsonify({"error": "SSID is required"}), 400

        # Attempt network connection
        try:
            # Log connection attempt
            logger.info(f"Attempting to connect to network: {ssid}")

            # Attempt connection
            connection_result = network_manager.connect_to_network(ssid, password)

            if connection_result:
                # Successful connection
                connection_details = network_manager.get_current_connection()
                logger.info(f"Successfully connected to {ssid}")
                return (
                    jsonify(
                        {
                            "message": "Connected successfully",
                            "connection": connection_details,
                        }
                    ),
                    200,
                )
            else:
                # Connection failed
                logger.warning(f"Failed to connect to network: {ssid}")
                return (
                    jsonify(
                        {
                            "error": "Connection failed. Check credentials and network availability"
                        }
                    ),
                    400,
                )

        except Exception as connection_error:
            # Log and handle unexpected errors during connection
            logger.error(f"Unexpected error connecting to network: {connection_error}")
            return (
                jsonify(
                    {"error": f"Unexpected connection error: {str(connection_error)}"}
                ),
                500,
            )

    # Handle GET request (network scanning)
    try:
        # Scan for available networks
        logger.info("Scanning for available WiFi networks")
        networks = network_manager.scan_networks()

        # Sort networks by signal strength
        sorted_networks = sorted(
            networks, key=lambda x: x.get("signal_strength", 0), reverse=True
        )

        # Render template with networks
        return render_template(
            "app/index.html",
            networks=sorted_networks,
            active_tab="wifi",
            user=current_user,
        )

    except Exception as scan_error:
        # Log scanning error
        logger.error(f"Network scanning failed: {scan_error}")

        # Render template with error message
        return render_template(
            "app/index.html",
            active_tab="wifi",
            error="Failed to scan networks. Check system permissions.",
            user=current_user,
        )


# STATS
@blueprint.route("/stats", methods=["GET", "POST"])
@login_required
def stats():
    """
    STATS route ....
    """
    user_id = current_user.id
    temperature_history = DataService.get_temperature_history(user_id)
    moisture_history = DataService.get_moisture_history(user_id)
    compost_maturity = DataService.get_compost_maturity(user_id)
    environmental_impact = DataService.get_environmental_impact(user_id)

    return render_template(
        "app/index.html",
        user=current_user,
        active_tab="stats",
        temperature_history=temperature_history,
        moisture_history=moisture_history,
        compost_maturity=compost_maturity,
        environmental_impact=environmental_impact,
    )


@blueprint.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint for the Caddy server and container orchestration.

    Returns:
        JSON response with status 'ok' if the service is running.
    """
    from flask import Blueprint, jsonify, current_app

    # Get basic system information
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Check GPU availability if using PyTorch
    gpu_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
    gpu_info = {}
    if gpu_available:
        gpu_info = {
            'device_name': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
            'memory_reserved': f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        }

    # Check if we can access the database
    db_status = 'unknown'
    try:
        from app.models import db
        db.engine.execute('SELECT 1')
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'

    # Check socketio status
    socketio_status = 'active' if hasattr(current_app, 'socketio') else 'inactive'

    # Check MQTT client status
    mqtt_status = 'connected' if mqtt_client and mqtt_client.is_connected() else 'disconnected'

    return jsonify({
        'status': 'ok',
        'service': 'smart_compost',
        'version': '1.0.0',
        'env': os.environ.get('FLASK_ENV', 'production'),
        'system': {
            'memory_percent': f"{memory.percent}%",
            'cpu_percent': f"{psutil.cpu_percent()}%",
            'disk_percent': f"{disk.percent}%"
        },
        'database': db_status,
        'socketio': socketio_status,
        'mqtt': mqtt_status,
        'gpu': gpu_info if gpu_available else 'not available'
    }), 200


@blueprint.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """
    User Profile Route
    """
    # Fetch the current user's data
    user = Users.query.get(current_user.id)

    # Fetch the user's primary device (most recently used)
    primary_device = (
        Device.query.filter_by(user_id=current_user.id)
        .order_by(Device.last_seen.desc())
        .first()
    )
    notifications = DataService.get_notifications(current_user.id)
    connected_devices = DataService.get_connected_devices(current_user.id)
    wifi_connected = DataService.get_wifi_status(current_user.id)
    current_time = DataService.get_current_time()

    # Fetch notification preferences (example: stored in the Users model)
    notification_prefs = {
        "temp_alerts": True,  # Example: Fetch from database
        "moisture_alerts": True,
        "weekly_reports": True,
        "compost_ready": True,
    }

    # Handle form submission for updating profile
    if request.method == "POST":
        # Update user profile data
        user.first_name = request.form.get("first_name", user.first_name)
        user.last_name = request.form.get("last_name", user.last_name)
        user.email = request.form.get("email", user.email)
        user.phone = request.form.get("phone", user.phone)
        user.location = request.form.get("location", user.location)

        # Update notification preferences
        notification_prefs["temp_alerts"] = "temp_alerts" in request.form
        notification_prefs["moisture_alerts"] = "moisture_alerts" in request.form
        notification_prefs["weekly_reports"] = "weekly_reports" in request.form
        notification_prefs["compost_ready"] = "compost_ready" in request.form

        # Save changes to the database
        db.session.commit()

        return redirect(url_for("app.profile"))

    return render_template(
        "app/index.html",
        user=user,
        primary_device=primary_device,
        devices=Device.query.filter_by(user_id=current_user.id).all(),
        notification_prefs=notification_prefs,
        notifications=notifications,
        connected_devices=connected_devices,
        wifi_connected=wifi_connected,
        current_time=current_time,
        active_tab="profile",
        device=primary_device,
    )

@blueprint.route("/control", methods=["GET", "POST"])
@login_required
def control_system():
    """
    Route for manual control of compost system components
    """
    if request.method == "POST":
        action = request.form.get("action")
        if not action:
            flash("No action specified", "error")
            return redirect(url_for("app.control_system"))

        # Validate MQTT connection
        if not mqtt_client or not mqtt_client.is_connected():
            flash("MQTT connection unavailable, cannot send commands", "error")
            return redirect(url_for("app.control_system"))

        try:
            # Prepare command payload
            command = {
                "user_id": current_user.id,
                "action": action,
                "timestamp": datetime.now().isoformat()
            }

            # Add additional parameters if present
            if "duration" in request.form:
                command["duration"] = int(request.form.get("duration", 0))

            if "target" in request.form:
                command["target"] = request.form.get("target")

            # Send command via MQTT
            mqtt_client.publish("compost/commands", json.dumps(command))

            # Log the action
            activity = ActivityLog(
                user_id=current_user.id,
                activity_type="System Control",
                details=f"Manual action: {action}",
                ip_address=get_client_ip(request),
                user_agent=request.headers.get("User-Agent")
            )
            db.session.add(activity)
            db.session.commit()

            flash(f"Command '{action}' sent successfully", "success")

        except Exception as e:
            logger.error(f"Error sending command: {str(e)}")
            flash(f"Error sending command: {str(e)}", "error")

    # Get current system status
    system_status = DataService.get_system_status(current_user.id)

    # Get available control actions
    control_actions = [
        {"name": "mix", "label": "Mix Compost", "icon": "mdi:rotate-3d"},
        {"name": "water", "label": "Add Water", "icon": "mdi:water"},
        {"name": "aerate", "label": "Aerate", "icon": "mdi:fan"},
        {"name": "harvest", "label": "Harvest", "icon": "mdi:shovel"}
    ]

    return render_template(
        "app/control.html",  # You'll need to create this template
        user=current_user,
        active_tab="control",
        system_status=system_status,
        control_actions=control_actions
    )

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
                register_device(user.id)
                flash("Welcome back!", "success")
                return redirect(url_for("app.home"))
            except Exception as e:
                print("Error in login_user:", e)
        else:
            flash("Invalid email or password", "error")
            return render_template("auth/signin.html")

    return render_template("auth/signin.html")


# Signup
@auth_blueprint.route("/signup", methods=["GET", "POST"])
def signup():
    """User Registration with appropriate information"""
    if request.method == "POST":
        from app.models import db

        # Extract form data
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        password = request.form.get("password")
        phone = request.form.get("phone")  # Optional
        location = request.form.get("location")  # Optional
        agree = request.form.get("agree")

        # IF USER HAS NOT AGREE TO THE TERMS AND POLICY
        if not agree:
            flash("First agree to policy and conditions to proceed!", "error")
            return render_template(
                "auth/signup.html",
                first_name=first_name,
                last_name=last_name,
                email=email,
                password=password,
                agree=agree,
            )

        # Validate required fields
        if not all([first_name, last_name, email, password]):
            flash("Required fields cannot be empty", "error")
            return render_template("auth/signup.html")

        # Check for existing user
        existing_user = Users.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered", "error")
            return render_template(
                "auth/signup.html",
                first_name=first_name,
                last_name=last_name,
                email=email,
                password=password,
                agree=agree,
            )

        # Create new user
        user = Users(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            phone=phone,
            location=location,
            ip_address=get_client_ip(request),
        )
        db.session.add(user)
        db.session.commit()

        # Log the registration activity
        activity = ActivityLog(
            user_id=user.id,
            activity_type="Registration",
            ip_address=get_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
        )
        db.session.add(activity)

        # Register user's device
        register_device(user.id)

        # Log the user in
        login_user(user)
        user.update_last_login()
        NotificationManager.create_notification(
            user.id, "Welcome to the smart compost system"
        )

        # Redirect to onboarding or dashboard
        return redirect(url_for("app.home"))

    return render_template("auth/signup.html")


@auth_blueprint.route("/complete_profile", methods=["GET", "POST"])
@login_required
def complete_profile():
    """Additional profile information specifically for compost monitoring"""
    user = current_user

    if request.method == "POST":
        # Get compost preferences and settings
        compost_experience = request.form.get("compost_experience")
        preferred_materials = request.form.get("preferred_materials")
        garden_size = request.form.get("garden_size")
        notification_preferences = request.form.get("notification_preferences", "email")

        # Store this data in a user preferences table or extension
        # Since it's not in your current model, we'll need to add a new model

        # For now, log this activity
        activity = ActivityLog(
            user_id=user.id,
            activity_type="Profile Completion",
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
        )
        db.session.add(activity)
        db.session.commit()

        # Create a notification
        notification = Notification(
            user_id=user.id,
            message="Profile completed! You're all set to start monitoring your compost.",
        )
        db.session.add(notification)
        db.session.commit()

        flash("Profile information saved successfully!", "success")
        return redirect(url_for("dashboard.index"))

    return render_template("auth/complete_profile.html", user=user)


@auth_blueprint.route("/profile", methods=["GET", "POST"])
@login_required
def update_profile():
    """Update user profile settings"""
    user = current_user

    if request.method == "POST":
        # Update basic information
        user.first_name = request.form.get("first_name", user.first_name)
        user.last_name = request.form.get("last_name", user.last_name)
        user.email = request.form.get("email", user.email)
        user.phone = request.form.get("phone")
        user.location = request.form.get("location")

        # Handle password change
        current_password = request.form.get("current_password")
        new_password = request.form.get("new_password")

        if current_password and new_password:
            if user.verify_password(current_password):
                user.password = new_password
            else:
                flash("Current password is incorrect", "error")
                return render_template("auth/profile.html", user=user)

        # Handle avatar upload if provided
        if "avatar" in request.files:
            avatar_file = request.files["avatar"]
            if avatar_file and allowed_file(avatar_file.filename):
                filename = secure_filename(f"user_{user.id}_{int(time.time())}.png")
                avatar_file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                user.avatar = filename

        # Log the update
        activity = ActivityLog(
            user_id=user.id,
            activity_type="Profile Update",
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
        )
        db.session.add(activity)

        db.session.commit()
        flash("Profile updated successfully!", "success")

    # Get user's recent activities
    recent_activities = (
        ActivityLog.query.filter_by(user_id=user.id)
        .order_by(ActivityLog.timestamp.desc())
        .limit(5)
    )

    # Get user's devices
    devices = Device.query.filter_by(user_id=user.id).all()

    return render_template(
        "auth/profile.html", user=user, activities=recent_activities, devices=devices
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
