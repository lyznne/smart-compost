"""
SMART COMPOST - MODEL PROJECT.

---  Models file

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# import dependencies
from datetime import datetime
from email.policy import default
from flask_login import UserMixin
from werkzeug.security import generate_password_hash,check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()
login_manager = LoginManager()


class Users(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(64), index=True)
    last_name = db.Column(db.String(64), index=True)
    email = db.Column(db.String(120), unique=True, index=True)
    password_hash = db.Column(db.String(255))
    avatar = db.Column(
        db.String(255),
        default="'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=150&q=80",
    )
    location = db.Column(db.String(255))  # User's location
    ip_address = db.Column(db.String(45))  # Last known IP address
    phone = db.Column(db.String(20), nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = db.Column(db.Boolean, default=False) # soft delete

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            if hasattr(value, "__iter__") and not isinstance(value, str):
                value = value[0]

            if property == "password":
                self.password = value
                # self.password = value

            setattr(self, property, value)

    def __repr__(self):
        return f"<User {self.first_name} {self.last_name}>"

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self,password):
        self.password_hash  = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def update_last_login(self):
        """Update the last login time."""
        self.last_login = datetime.utcnow()
        db.session.commit()


class ActivityLog(db.Model):
    __tablename__ = "activity_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    activity_type = db.Column(db.String(128), nullable=False)  # e.g., "Login", "Settings Change"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))  # IP address of activity
    user_agent = db.Column(db.String(255))  # Browser/Device info

    user = db.relationship("Users", backref=db.backref("activities", lazy="dynamic"))

    def __repr__(self):
        return f"<Activity {self.activity_type} by User {self.user_id}>"

class Device(db.Model):
    """Represents user devices for tracking logins & notifications."""

    __tablename__ = "devices"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    device_name = db.Column(db.String(128), nullable=False)  # Parsed from User-Agent
    device_ip = db.Column(db.String(45), nullable=False)  # IPv4/IPv6 support
    last_seen = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    push_token = db.Column(db.String(255), nullable=True)  # FCM push token
    device_type = db.Column(db.String(20), nullable=True)  # Mobile, Tablet, PC
    browser = db.Column(db.String(100), nullable=True)
    os = db.Column(db.String(100), nullable=True)
    location_city = db.Column(db.String(100), nullable=True)
    location_region = db.Column(db.String(100), nullable=True)
    location_country = db.Column(db.String(50), nullable=True)
    wifi_connected = db.Column(db.Boolean, default=False)
    is_online = db.Column(db.Boolean, default=True)  # True if online, False if offline
    # is_deleted = db.Column(db.Boolean, default=False)  # Soft delete
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("Users", backref=db.backref("devices", lazy="dynamic"))

    def __repr__(self):
        return f"<Device {self.device_name} - {self.device_ip}>"


class Notification(db.Model):
    __tablename__ = "notifications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    channel = db.Column(db.String(20), default="web")  # web, email, push, sms
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    read_status = db.Column(db.Boolean, default=False)
    delivery_status = db.Column(db.String(20), default="pending")   # False = unread, True = read
    alert_level = db.Column(db.String(20), default="info")  # info, warning, error
    user = db.relationship("Users", backref=db.backref("notifications", lazy="dynamic"))

    def __repr__(self):
        return f"<Notification {self.message} for User {self.user_id}>"

class TrainingRun(db.Model):
    __tablename__ = "training_runs"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(128), nullable=False)
    experiment_id = db.Column(db.String(128), unique=True, nullable=False)
    parameters = db.Column(db.JSON)  # Hyperparameters as JSON
    metrics = db.Column(db.JSON)  # Loss, accuracy, etc.
    status = db.Column(db.String(20), default="running")  # ["running", "completed", "failed"]
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f"<TrainingRun {self.model_name} - {self.experiment_id}>"


class EnvironmentalVariable(db.Model):
    __tablename__ = "environmental_variables"

    id = db.Column(db.Integer, primary_key=True)  # Unique identifier
    variable = db.Column(db.String(100), unique=True, nullable=False)  # Name of variable
    variable_type = db.Column(db.String(50), nullable=False)  # Numerical/Categorical
    measurement_unit = db.Column(db.String(50), nullable=False)  # Unit of measurement
    optimal_range = db.Column(db.String(50), nullable=False)  # Optimal range value
    dependencies = db.Column(db.Text, nullable=True)  # Dependencies (comma-separated)
    introduction_stage = db.Column(db.String(50), nullable=False)  # Initial, Continuous, etc.
    frequency = db.Column(db.String(50), nullable=False)  # Frequency of measurement
    notes = db.Column(db.Text, nullable=True)  # Additional details
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<EnvironmentalVariable {self.variable}>"


class CompostUserPreferences(db.Model):
    __tablename__ = "compost_user_preferences"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    compost_experience = db.Column(db.String(20))  # novice, intermediate, expert
    preferred_materials = db.Column(db.String(255))  # comma-separated list
    garden_size = db.Column(db.String(20))  # small, medium, large
    notification_frequency = db.Column(db.String(20), default="daily")  # daily, weekly, real-time
    data_sharing_consent = db.Column(db.Boolean, default=False)
    prediction_alerts = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship("Users", backref=db.backref("compost_preferences", uselist=False))

    def __repr__(self):
        return f"<CompostPreferences for User {self.user_id}>"

class CompostData(db.Model):
    """
    Model for storing real-time compost data.
    """
    __tablename__ = "compost_data"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    status = db.Column(db.String(50), nullable=False)  # e.g., "Optimal", "Suboptimal"
    temperature = db.Column(db.Float, nullable=False)  # Temperature in Celsius
    moisture = db.Column(db.Float, nullable=False)  # Moisture level in percentage
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp of the data
    ph = db.Column(db.Float, nullable=False)  # pH level
    oxygen_level = db.Column(db.Float, nullable=False)  # Oxygen level in percentage
    carbon_nitrogen_ratio = db.Column(db.Float, nullable=False)  # C:N ratio
    nitrogen_content = db.Column(db.Float, nullable=False)  # Nitrogen content in percentage
    potassium_content = db.Column(db.Float, nullable=False)  # Potassium content in percentage
    phosphorus_content = db.Column(db.Float, nullable=False)  # Phosphorus content in percentage
    user = db.relationship("Users", backref=db.backref("compost_data", lazy="dynamic"))

    def __repr__(self):
        return f"<CompostData {self.status} - {self.timestamp}>"


def seed_environmental_data():
    """Seed the database with initial environmental variables."""

    data = [
        ("Temperature", "Numerical", "Celsius", "45-65", "Moisture_Content|Oxygen_Level|Materials_Ratio", "Initial", "Continuous", "Critical for decomposition, affects microbial activity"),
        ("Moisture_Content", "Numerical", "Percentage", "40-60", "Temperature|Material_Type|Particle_Size", "Initial", "Daily", "Higher for green materials, lower for browns"),
        ("pH_Level", "Numerical", "pH Scale", "6.5-8.0", "Nitrogen_Content|Temperature|Moisture_Content", "Initial", "Weekly", "Affects microbial growth, decomposition rate"),
        ("Oxygen_Level", "Numerical", "Percentage", "5-15", "Temperature|Moisture_Content|Bulk_Density", "Initial", "Continuous", "Must be maintained through turning or aeration"),
        ("Carbon_Nitrogen_Ratio", "Numerical", "Ratio", "25-30:1", "Material_Type|Temperature", "Initial", "At_Addition", "Crucial for proper decomposition, affects smell"),
        ("Nitrogen_Content", "Numerical", "Percentage", "1.5-2.0", "Material_Type|Temperature", "Initial", "At_Addition", "Higher in green materials"),
        ("Potassium_Content", "Numerical", "Percentage", "0.5-1.5", "Material_Type|Moisture_Content", "Initial", "Monthly", "Affects final fertilizer quality"),
        ("Phosphorus_Content", "Numerical", "Percentage", "0.3-1.0", "Material_Type|pH_Level", "Initial", "Monthly", "Affects final fertilizer quality"),
        ("Ambient_Humidity", "Numerical", "Percentage", "40-70", "Moisture_Content", "Continuous", "Continuous", "External factor affecting moisture"),
        ("Ambient_Temperature", "Numerical", "Celsius", "15-35", "Temperature|Moisture_Content", "Continuous", "Continuous", "External factor affecting process"),
        ("Waste_Type", "Numerical", "Millimeters", "5-50", "Material_Type|Moisture_Content", "Initial", "At_Addition", "Smaller particles decompose faster but need more aeration"),
        ("Waste_Height", "Numerical", "Centimeters", "90-150", "Bulk_Density|Moisture_Content", "Initial", "Weekly", "Affects heat retention and oxygen flow"),
        ("Particle_Size", "Numerical", "Millimeters", "5-50", "Material_Type|Moisture_Content", "Initial", "At_Addition", "Smaller particles decompose faster but need more aeration"),
        ("Bulk_Density", "Numerical", "kg/m3", "300-650", "Particle_Size|Moisture_Content|Material_Type", "Initial", "Weekly", "Affects oxygen flow and decomposition rate"),
        ("Final_Volume_Reduction", "Numerical", "Percentage", "40-60", "All_Above_Variables", "Final_Stage", "Final", "Indicates process efficiency, completion"),
        ("Odor_Level", "Numerical", "Scale 1-5", "1-2", "Oxygen_Level|Moisture_Content|Carbon_Nitrogen_Ratio", "Continuous", "Daily", "Indicator of anaerobic conditions"),
        ("Decomposition_Rate", "Numerical", "Percentage/Day", "1-3", "Temperature|Moisture_Content|Oxygen_Level", "After_Day_7", "Weekly", "Rate of mass reduction"),
        ("Time_Elapsed", "Numerical", "Days", "30-90", "All_Above_Variables", "Initial", "Continuous", "Total duration until maturity"),
        ("Maturity_Index", "Numerical", "Scale 1-8", ">7", "Temperature|Moisture_Content|Time", "Final_Stage", "Weekly", "Indicates completion of composting"),
        ("Turning_Frequency", "Numerical", "Days", "3-7", "Temperature|Oxygen_Level|Moisture_Content", "After_Day_3", "Weekly", "Based on temperature and oxygen readings"),
    ]

    for item in data:
        variable = EnvironmentalVariable(
            variable=item[0],
            variable_type=item[1],
            measurement_unit=item[2],
            optimal_range=item[3],
            dependencies=item[4],
            introduction_stage=item[5],
            frequency=item[6],
            notes=item[7],
            created_at=datetime.now(),
        )
        print(f"✅ Adding: {variable}")
        db.session.add(variable)

    try:
        db.session.commit()
        print("✅ Environmental variables added successfully.")
    except Exception as e:
        db.session.rollback()  # Prevent database corruption
        print(f"❌ Error committing data: {e}")


def create_sample_user():
    """
    Create sample user data.
    """
    existing_user = Users.query.filter_by(email="emuthiani26@gmail.com").first()
    if existing_user:
        print(" User already exists.")
        return existing_user

    sample_user = Users(
        first_name="Enos",
        last_name="Muthiani",
        email="emuthiani26@gmail.com",
        password="enos",
    )
    db.session.add(sample_user)
    db.session.commit()
    print(f" successfully created { sample_user } and saved!")
    return sample_user


@login_manager.user_loader
def user_loader(user_id):
    """
    Load user by ID for Flask-Login.
    """

    return Users.query.get(int(user_id))


@login_manager.request_loader
def request_loader(request):
    """
    Load user by request for Flask-Login.
    """
    email = request.form.get("email")
    return Users.query.filter_by(email=email).first()
