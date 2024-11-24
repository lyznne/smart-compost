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

# Import dependencies
from flask_login import UserMixin
from datetime import datetime
from app import db, login_manager
from app.util import hash_pass
from werkzeug.security import generate_password_hash,check_password_hash


class Users(db.Model, UserMixin):
    __tablename__ = "Users"

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(64), unique=True, index=True)
    last_name = db.Column(db.String(64), index=True)
    email = db.Column(db.String(120), unique=True, index=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

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
def user_loader(id):
    """
    Load user by ID for Flask-Login.
    """
    return Users.query.get(int(id))


@login_manager.request_loader
def request_loader(request):
    """
    Load user by request for Flask-Login.
    """
    email = request.form.get("email")
    return Users.query.filter_by(email=email).first()
