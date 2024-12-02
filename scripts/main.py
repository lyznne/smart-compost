"""
SMART COMPOST - MODEL PROJECT.

---  D R I V E R S _ C O D E
---   main.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# where all the processing and running of the smart-compost models is run and all its connectins

# import dependencies



what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}
# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)



#     It's train time!
# do the forward pass,
# calculate the loss,
# optimizer zero grad,
# losssss backwards!

# Optimizer step step step

# Let's test now!
# with torch no grad:
# do the forward pass,
# calculate the loss,
# watch it go down down down!

# initiate class for Composting Prediction
class SmartCompostModel():

























# Required installations:
# pip install flask pandas numpy scikit-learn pytorch lightgbm xgboost
# pip install plotly dash python-dotenv joblib mlflow
# pip install comet_ml python-socketio eventlet

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from comet_ml import Experiment
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import mlflow
import plotly.express as px
from datetime import datetime
import joblib





















# Custom Neural Network for Composting Prediction
class CompostingNN(nn.Module):
    def __init__(self, input_size):
        super(CompostingNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output(x)


# Data preprocessing function
def preprocess_data(df):
    # Convert categorical variables
    df["Material_Type"] = df["Material_Type"].map({"Green": 0, "Brown": 1})

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = [
        col for col in df.columns if df[col].dtype in ["int64", "float64"]
    ]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler


# Model training wrapper
def train_models(X_train, y_train, X_test, y_test, experiment):
    models = {}

    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=100
    )
    lgb_model.fit(X_train, y_train)
    models["lightgbm"] = lgb_model

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=6, learning_rate=0.05, n_estimators=100
    )
    xgb_model.fit(X_train, y_train)
    models["xgboost"] = xgb_model

    # Train PyTorch Neural Network
    input_size = X_train.shape[1]
    nn_model = CompostingNN(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)

    # Train NN
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = nn_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            experiment.log_metric("nn_loss", loss.item(), epoch)

    models["neural_network"] = nn_model

    return models


# Flask application setup
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Process input data and make predictions using loaded models
    # Return predictions as JSON
    return jsonify({"predictions": predictions})


# MLflow experiment tracking
def track_experiment(models, X_test, y_test):
    mlflow.set_experiment("composting_optimization")

    with mlflow.start_run():
        # Log parameters and metrics for each model
        for model_name, model in models.items():
            mlflow.log_param(f"{model_name}_type", type(model).__name__)

            if model_name != "neural_network":
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                mlflow.log_metric(f"{model_name}_mse", mse)


# Main execution
if __name__ == "__main__":
    # Initialize Comet.ml experiment
    experiment = Experiment(
        api_key="YOUR_API_KEY", project_name="composting-optimization"
    )

    # Load and preprocess data
    df = pd.read_csv("composting_data.csv")
    df_processed, scaler = preprocess_data(df)

    # Split features and target
    X = df_processed.drop(["Maturity_Index"], axis=1)
    y = df_processed["Maturity_Index"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    models = train_models(X_train, y_train, X_test, y_test, experiment)

    # Track experiments
    track_experiment(models, X_test, y_test)

    # Save models
    joblib.dump(models["lightgbm"], "models/lightgbm_model.joblib")
    joblib.dump(models["xgboost"], "models/xgboost_model.joblib")
    torch.save(models["neural_network"].state_dict(), "models/nn_model.pth")

    # Start Flask application
    app.run(debug=True)
