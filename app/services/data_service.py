from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.models import CompostData, ActivityLog
from app.models import Device, Notification, db


class DataService:
    @staticmethod
    def get_compost_status(user_id: int) -> Dict:
        """Fetch the latest compost status for a specific user."""
        latest_data = (
            CompostData.query.filter_by(user_id=user_id)
            .order_by(CompostData.timestamp.desc())
            .first()
        )
        if latest_data:
            return {
                "status": latest_data.status,
                "temperature": latest_data.temperature,
                "moisture": latest_data.moisture,
            }
        return {
            "status": "Unknown",
            "temperature": 0.0,
            "moisture": 0.0,
        }

    @staticmethod
    def get_recent_activity(user_id: int, limit: int = 5) -> List:
        """Fetch recent activity logs for a specific user."""
        activities = (
            ActivityLog.query.filter_by(user_id=user_id)
            .order_by(ActivityLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "text": activity.activity_type,
                "time": activity.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for activity in activities
        ]

    @staticmethod
    def get_device_info(user_id: int) -> Dict:
        """Fetch the user's primary device information."""
        device = (
            Device.query.filter_by(user_id=user_id)
            .order_by(Device.last_seen.desc())
            .first()
        )
        if device:
            return {
                "device_name": device.device_name,
                "device_type": device.device_type,
                "last_seen": device.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
            }
        return {
            "device_name": "Unknown",
            "device_type": "Unknown",
            "last_seen": "Never",
        }

    @staticmethod
    def get_notifications(user_id: int, limit: int = 5) -> list:
        """Fetch recent notifications for a specific user."""
        notifications = (
            Notification.query.filter_by(user_id=user_id)
            .order_by(Notification.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "message": notification.message,
                "timestamp": notification.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "read_status": notification.read_status,
            }
            for notification in notifications
        ]

    @staticmethod
    def get_connected_devices(user_id: int) -> list:
        """Fetch all connected devices for a specific user."""
        devices = Device.query.filter_by(user_id=user_id).all()
        return [
            {
                "device_name": device.device_name,
                "device_type": device.device_type,
                "last_seen": device.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for device in devices
        ]

    @staticmethod
    def get_wifi_status(user_id: int) -> bool:
        """Check if the user's primary device is connected to Wi-Fi."""
        device = (
            Device.query.filter_by(user_id=user_id)
            .order_by(Device.last_seen.desc())
            .first()
        )
        if device:
            return (
                device.wifi_connected
            )
        return False

    @staticmethod
    def get_current_time() -> str:
        """Get the current time."""
        return datetime.now()

    @staticmethod
    def get_temperature_history(user_id: int, days: int = 7) -> List[Dict]:
        """Fetch temperature history for the last `days` days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        temperature_data = (
            CompostData.query.filter(
                CompostData.user_id == user_id,
                CompostData.timestamp >= start_date,
                CompostData.timestamp <= end_date,
            )
            .order_by(CompostData.timestamp.asc())
            .all()
        )
        return [
            {
                "day": data.timestamp.strftime("%Y-%m-%d"),
                "temperature": data.temperature,
            }
            for data in temperature_data
        ]

    @staticmethod
    def get_moisture_history(user_id: int, days: int = 7) -> List[Dict]:
        """Fetch moisture history for the last `days` days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        moisture_data = (
            CompostData.query.filter(
                CompostData.user_id == user_id,
                CompostData.timestamp >= start_date,
                CompostData.timestamp <= end_date,
            )
            .order_by(CompostData.timestamp.asc())
            .all()
        )
        return [
            {
                "day": data.timestamp.strftime("%Y-%m-%d"),
                "moisture": data.moisture,
            }
            for data in moisture_data
        ]

    @staticmethod
    def get_compost_maturity(user_id: int) -> Dict:
        """Fetch compost maturity data."""
        # This is a placeholder. You would need to implement the logic to calculate maturity.
        return {
            "percent_complete": 78,
            "estimated_completion": "14 days",
            "stages": [
                "Fresh Material",
                "Active Decomposition",
                "Curing",
                "Ready to Use",
            ],
        }

    @staticmethod
    def get_environmental_impact(user_id: int) -> Dict:
        """Fetch environmental impact data."""
        # This is a placeholder. You would need to implement the logic to calculate impact.
        return {
            "co2_reduction": 24.5,  # in kg
            "waste_processed": 78,  # in kg
            "compost_produced": 32,  # in kg
        }
