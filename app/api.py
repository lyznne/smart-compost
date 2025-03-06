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
from Models.data_setup import CompostTimeSeriesDataset
from flask_login import current_user, login_required
from flask import request
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from Models.data_setup import CompostTimeSeriesDataset
from Models.engine import train_compost_model
from flask import Blueprint, request, jsonify
import torch
from datetime import datetime
from Models.model import CompostLSTM
from app.models import db, CompostData

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


@api_blueprint.route("/train", methods=["POST"])
def train():
    """
    Train the compost model.
    """
    try:
        # Path to the dataset
        dataset_path = "data/smart_compost_dataset101.csv"

        # Create dataset and dataloaders
        dataset = CompostTimeSeriesDataset(dataset_path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        # Train the model
        train_compost_model(train_loader, val_loader)

        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_blueprint.route('/predict', methods=["GET"])
def predict():
    """
    Make predictions using the trained model and save the results to the database.
    """
    try:
        # Load input data from request
        input_data = request.json.get("input_data")
        user_id = request.json.get("user_id")  # Get user_id from the request
        if not input_data or not user_id:
            return jsonify({"error": "Input data and user_id are required"}), 400

        # Convert input data to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        # Load the trained model
        model = CompostLSTM(input_size=input_tensor.shape[-1])
        model.load_state_dict(torch.load("best_compost_model.pth"))
        model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = model(input_tensor)

        # Extract temperature and moisture predictions
        temperature_pred = predictions[0][0].item()  # First output (temperature)
        moisture_pred = predictions[0][1].item()  # Second output (moisture)

        # Determine status based on predictions
        status = (
            "Optimal"
            if (50 <= temperature_pred <= 70) and (50 <= moisture_pred <= 70)
            else "Suboptimal"
        )

        # Save prediction details to the database
        compost_data = CompostData(
            user_id=user_id,
            status=status,
            temperature=temperature_pred,
            moisture=moisture_pred,
            timestamp=datetime.utcnow(),
        )
        db.session.add(compost_data)
        db.session.commit()

        # Return predictions and status
        return (
            jsonify(
                {
                    "predictions": {
                        "temperature": temperature_pred,
                        "moisture": moisture_pred,
                    },
                    "status": status,
                }
            ),
            200,
        )
    except Exception as e:
        db.session.rollback()  # Rollback in case of error
        return jsonify({"error": str(e)}), 500
