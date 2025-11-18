"""
SMART COMPOST - MODEL PROJECT.

--- channels.py file

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 2 Mar 2025
* email   - emuthiani26@gmail.com

                        Copyright (c) 2025 - enos.vercel.app
"""

from flask_socketio import emit, join_room
from flask_login import current_user
from app import socketio

@socketio.on("connect", namespace="/notification")
def handle_connect():
    """Handles a new WebSocket connection."""
    if current_user.is_authenticated:
        user_room = f"user_{current_user.id}"
        join_room(user_room)
        emit("connected", {"message": "Connected to notifications!"}, room=user_room)

@socketio.on("disconnect", namespace="/notification")
def handle_disconnect():
    """Handles WebSocket disconnection."""
    print(f"User {current_user.id} disconnected from notifications.")
