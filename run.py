'''
SMART COMPOST - MODEL PROJECT.

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2025      - enos.vercel.app
'''


# Import required modules
from random import shuffle
from Models.data_setup import CompostTimeSeriesDataset
from Models.engine import train_compost_model
from app import create_app, socketio
from app.config import  DevelopmentConfig
from torch.utils.data import DataLoader
import torch


# ---
# Path to the dataset
# Path to the dataset
dataset_path = "data/smart_compost_dataset101.csv"

def create_loaders():
    """Create data loaders for training and validation"""
    dataset = CompostTimeSeriesDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        DataLoader(val_dataset, batch_size=32, shuffle=True)
    )

# Create app instance based on environment
app = create_app(DevelopmentConfig if os.getenv('FLASK_ENV') == 'development' else ProductionConfig)

# Driver code
if __name__ == "__main__":
    # Only train model if explicitly requested
    if os.getenv('TRAIN_MODEL', 'false').lower() == 'true':
        print("Starting model training...")
        train_loader, val_loader = create_loaders()
        train_compost_model(train_loader, val_loader)
        print("Model training completed!")

    # Serve the app with websockets
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=app.config['DEBUG'],
        allow_unsafe_werkzeug=True,
    )
