from typing import Optional, List
from app.models import Notification, Users, Device
from flask import current_app as app
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import emit
from app import socketio

db = SQLAlchemy()

class NotificationManager:
    """Manages notifications for users across multiple channels, including real-time WebSockets."""

    @staticmethod
    def create_notification(user_id: int, message: str, channel: str = "web") -> Notification:
        """Create and store a notification for a user, then send it via WebSockets if applicable.

        Args:
            user_id (int): The ID of the user receiving the notification.
            message (str): The notification message content.
            channel (str, optional): Notification delivery method. Defaults to "web".

        Returns:
            Notification: The created notification object.
        """
        notification = Notification(
            user_id=user_id,
            message=message,
            channel=channel
        )

        db.session.add(notification)
        db.session.commit()

        # ✅ Send WebSocket notification if it's a "web" notification
        if channel == "web":
            NotificationManager.send_web_notification(user_id, message)

        # ✅ Send via other channels if applicable
        elif channel in ["email", "push", "sms"]:
            NotificationManager.send_notification(notification)

        return notification

    @staticmethod
    def send_notification(notification: Notification) -> bool:
        """Send a notification based on its channel (email, push, SMS).

        Args:
            notification (Notification): The notification object to be sent.

        Returns:
            bool: True if the notification was sent successfully, False otherwise.
        """
        try:
            if notification.channel == "email":
                NotificationManager._send_email(notification)
            elif notification.channel == "push":
                NotificationManager._send_push(notification)
            elif notification.channel == "sms":
                NotificationManager._send_sms(notification)

            notification.delivery_status = "sent"
            db.session.commit()
            return True
        except Exception as e:
            notification.delivery_status = "failed"
            db.session.commit()
            app.logger.error(f"Failed to send notification: {str(e)}")
            return False

    @staticmethod
    def send_web_notification(user_id: int, message: str) -> None:
        """Send a real-time WebSocket notification to the user.

        Args:
            user_id (int): The ID of the user receiving the notification.
            message (str): The notification message.
        """
        user_room = f"user_{user_id}"  # Each user has a unique room
        socketio.emit(
            "new_notification",
            {"user_id": user_id, "message": message, "channel": "web"},
            room=user_room,
            namespace="/notifications"
        )

    @staticmethod
    def _send_email(notification: Notification) -> None:
        """Send an email notification to the user.

        Args:
            notification (Notification): The notification to send via email.
        """
        user: Optional[Users] = Users.query.get(notification.user_id)
        if user and user.email:
            # Use Flask-Mail or similar service to send the email
            pass

    @staticmethod
    def _send_push(notification: Notification) -> None:
        """Send a push notification to the user's devices.

        Args:
            notification (Notification): The notification to send via push notification.
        """
        devices: List[Device] = Device.query.filter_by(user_id=notification.user_id).all()
        for device in devices:
            # Integrate with Firebase Cloud Messaging (FCM) or other push notification services
            pass

    @staticmethod
    def _send_sms(notification: Notification) -> None:
        """Send an SMS notification to the user's phone.

        Args:
            notification (Notification): The notification to send via SMS.
        """
        user: Optional[Users] = Users.query.get(notification.user_id)
        if user and user.phone:
            # Integrate with Twilio or similar SMS gateway
            pass
