"""
SMART COMPOST - MODEL PROJECT.

--- api.py file

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 2 Mar 2025
* email   - emuthiani26@gmail.com

                        Copyright (c) 2025 - enos.vercel.app
"""

from flask import  jsonify, Blueprint, Response
from flask_login import current_user, login_required
from flask import request




api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/device/setup', methods=['POST'])
@login_required
def setup_device():
    from app import db
    from app.models import Device
    """API to setup a new IoT device"""
    device_id = request.form.get('device_id')
    device_name = request.form.get('device_name')

    # Check if device exists (or create it)
    device = Device.query.filter_by(
        user_id=current_user.id,
        device_id=device_id
    ).first()

    if not device:
        device = Device(
            user_id=current_user.id,
            device_id=device_id,
            device_name=device_name,
            device_ip=request.remote_addr,
            device_type='iot'
        )
        db.session.add(device)

    db.session.commit()

    return jsonify({'status': 'success', 'device_id': device.id})

@api_blueprint.route('/device/wifi', methods=['POST'])
@login_required
def configure_wifi():
    from app import db
    from app.models import Device, ActivityLog
    """API to configure WiFi on a device"""
    device_id = request.form.get('device_id')
    ssid = request.form.get('ssid')
    password = request.form.get('password')

    # Ensure device belongs to user
    device = Device.query.filter_by(id=device_id, user_id=current_user.id).first()
    if not device:
        return jsonify({'status': 'error', 'message': 'Device not found'}), 404

    # In real implementation, you would:
    # 1. Send this configuration to the device
    # 2. Wait for confirmation
    # 3. Update the device status

    # For now, we'll simulate success
    activity = ActivityLog(
        user_id=current_user.id,
        activity_type="Device WiFi Configuration",
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )
    db.session.add(activity)
    db.session.commit()

    return jsonify({'status': 'success', 'message': 'WiFi configuration sent'})
