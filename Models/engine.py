"""
SMART COMPOST - MODEL PROJECT.

---  the training of the model
---   Models/engine.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 3 Dec 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app


"""

from comet_ml.integration.pytorch import log_model
from comet_ml import Experiment
from json import dumps
from Models.model import CompostLSTM
from Models.train import CompostModelTrainer
from app.models import TrainingRun, db
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader
import torch
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd

# Visualization imports
from .visualization import (
    plot_training_history,
    plot_predictions_vs_actual,
    plot_residuals,
    save_model_architecture,
    plot_correlation_matrix,
    plot_hidden_state_tsne,
    plot_prediction_time_series,
    plot_feature_distributions,
    plot_feature_importance
)
import shutil

# Load environment variables
load_dotenv()


def train_compost_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyper_params: dict = None,
    feature_names: list = None
) -> CompostLSTM:
    """
    Train the compost model and log metrics to Comet ML.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        hyper_params (dict): Dictionary of hyperparameters. Defaults to None.
        feature_names (list): List of feature names for visualization. Defaults to None.

    Returns:
        CompostLSTM: Trained model.
    """
    # Default hyperparameters
    default_hyper_params = {
        "input_size": 25,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.2,
        "learning_rate": 0.0001,
        "batch_size": 64,
        "epochs": 100,
        "early_stopping_patience": 15,
        "gradient_clip": 1.0,
    }

    # Use provided hyperparameters or defaults
    if hyper_params is not None:
        default_hyper_params.update(hyper_params)
    hyper_params = default_hyper_params

    # Create results directories
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Initialize Comet ML experiment
        experiment = Experiment(
            api_key=os.getenv("SMART_COMPOST_COMET_API_KEY"),
            project_name=os.getenv("SMART_COMPOST_PROJECT_NAME"),
            workspace=os.getenv("SMART_COMPOST_WORKSPACE"),
        )
        experiment.log_parameters(hyper_params)
    except Exception as e:
        print(f"Failed to initialize Comet ML experiment: {e}")
        experiment = None

    try:
        # Calculate input size from data and analyze features
        sample_features, sample_targets = next(iter(train_loader))
        input_size = sample_features.shape[-1]  # number of features

        # Extract batch of data for feature analysis
        features_batch = sample_features.numpy().reshape(-1, input_size)
        targets_batch = sample_targets.numpy()

        # Default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(input_size)]
            target_names = [f"Target_{i+1}" for i in range(sample_targets.shape[-1])]
        else:
            # Separate feature names and target names if provided together
            if len(feature_names) > input_size:
                target_names = feature_names[input_size:]
                feature_names = feature_names[:input_size]
            else:
                target_names = [f"Target_{i+1}" for i in range(sample_targets.shape[-1])]

        # Visualize feature distributions
        plot_feature_distributions(features_batch, feature_names)

        # Visualize correlation matrix for features
        plot_correlation_matrix(features_batch, feature_names)

        # Log visualizations to Comet ML
        if experiment:
            experiment.log_image("results/plots/correlation_matrix.png", name="feature_correlation")
            experiment.log_image("results/plots/features_boxplot.png", name="feature_boxplot")
    except Exception as e:
        print(f"Failed to analyze features: {e}")
        # Fall back to default input size
        input_size = hyper_params["input_size"]

    # Initialize model and trainer
    model = CompostLSTM(
        input_size=input_size,
        hidden_size=hyper_params["hidden_size"],
        num_layers=hyper_params["num_layers"],
        dropout=hyper_params["dropout"],
    )
    trainer = CompostModelTrainer(model)

    # Save model architecture
    save_model_architecture(model, os.path.join(results_dir, "model_architecture.txt"))

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs

    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hyper_params["epochs"],
        early_stopping_patience=hyper_params["early_stopping_patience"],
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip=hyper_params["gradient_clip"],
    )

    # Add learning rates to history if not already there
    if 'learning_rates' not in history:
        history['learning_rates'] = [
            hyper_params["learning_rate"] * (0.1 ** (epoch // 10))
            for epoch in range(len(history['train_losses']))
        ]

    # Generate and save training history plot
    plot_training_history(history)

    # Log training and validation losses
    if experiment:
        for epoch, (train_loss, val_loss) in enumerate(
            zip(history["train_losses"], history["val_losses"])
        ):
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)

            # Log learning rate if available
            if 'learning_rates' in history and epoch < len(history['learning_rates']):
                experiment.log_metric("learning_rate", history['learning_rates'][epoch], step=epoch)

    # Save the best model
    model_path = "best_compost_model.pth"
    torch.save(model.state_dict(), model_path)

    # Log the model to Comet ML
    if experiment:
        log_model(experiment, model=model, model_name="CompostLSTM")

    # Log a table of training metrics
    table_data = []
    for epoch, (train_loss, val_loss) in enumerate(
        zip(history["train_losses"], history["val_losses"])
    ):
        # Calculate learning rate for the current epoch
        lr = hyper_params["learning_rate"] * (
            0.1 ** (epoch // 10)
        )  # Match the scheduler logic

        # Safely access gradient norms if they exist in history
        gradient_norms = history.get("gradient_norms", [])
        gradient_norm = gradient_norms[epoch] if epoch < len(gradient_norms) else 0.0

        # Add metrics to table data
        table_data.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
                "gradient_norm": gradient_norm,
            }
        )

    # Log the table
    if experiment:
        experiment.log_table(
            filename="training_metrics.csv",
            tabular_data=table_data,
            headers=[
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "gradient_norm",
            ],
        )

    # Generate predictions and visualizations on validation set
    try:
        model.eval()
        all_val_preds = []
        all_val_targets = []
        all_hidden_states = []

        # Collect predictions and targets from validation set
        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_preds = model(val_batch_x)
                all_val_preds.append(val_preds.cpu().numpy())
                all_val_targets.append(val_batch_y.cpu().numpy())

                # Collect hidden states for t-SNE visualization
                hidden = model.get_hidden_states(val_batch_x)
                all_hidden_states.append(hidden.cpu().numpy())

        # Concatenate batches
        all_val_preds = np.vstack(all_val_preds)
        all_val_targets = np.vstack(all_val_targets)
        all_hidden_states = np.vstack(all_hidden_states)

        # Generate prediction vs actual and residual plots
        plot_predictions_vs_actual(all_val_targets, all_val_preds, feature_names=target_names)
        plot_residuals(all_val_targets, all_val_preds, feature_names=target_names)

        # Generate time series prediction plots (for a subset of data)
        max_samples = min(200, len(all_val_targets))
        plot_prediction_time_series(
            all_val_targets[:max_samples],
            all_val_preds[:max_samples],
            feature_names=target_names
        )

        # t-SNE visualization of hidden states
        # Use prediction error as color labels
        error_values = np.mean(np.abs(all_val_preds - all_val_targets), axis=1)
        plot_hidden_state_tsne(all_hidden_states, labels=error_values)

        # Log visualizations to Comet ML
        if experiment:
            # Upload all plots from the results/plots directory
            plots_dir = "results/plots"
            if os.path.exists(plots_dir):
                for plot_file in os.listdir(plots_dir):
                    if plot_file.endswith('.png'):
                        experiment.log_image(os.path.join(plots_dir, plot_file), name=plot_file[:-4])

            # Log model architecture file
            experiment.log_asset("results/model_architecture.txt")
    except Exception as e:
        print(f"Failed to generate visualization: {e}")

    # Log 3D points using real model data
    if experiment and model:
        try:
            # Get a batch of data for visualization
            batch_x, batch_y = next(iter(train_loader))

            # Get model outputs for coloring by prediction error
            model.eval()
            with torch.no_grad():
                predictions = model(batch_x).detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # Extract hidden states for 3D coordinates
                hidden_states = model.get_hidden_states(batch_x)

                # Use t-SNE or PCA to reduce dimensionality if needed
                from sklearn.decomposition import PCA

                num_samples = min(50, len(hidden_states))
                hidden_sample = hidden_states[:num_samples].reshape(num_samples, -1)

                # Reduce to 3 dimensions for visualization
                pca = PCA(n_components=3)
                coords_3d = pca.fit_transform(hidden_sample)

                # Calculate error for color mapping (if applicable)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # For multi-output models
                    error = np.mean(
                        np.abs(predictions[:num_samples] - batch_y[:num_samples]),
                        axis=1,
                    )
                else:
                    # For single output models
                    error = np.abs(
                        predictions[:num_samples].flatten()
                        - batch_y[:num_samples].flatten()
                    )

                # Normalize error for color scaling
                max_error = np.max(error) if np.max(error) > 0 else 1.0
                normalized_error = error / max_error

                # Create RGB colors (red for high error, blue for low)
                colors = np.zeros((num_samples, 3))
                colors[:, 0] = normalized_error  # Red channel
                colors[:, 2] = 1 - normalized_error  # Blue channel

                # Combine coordinates and colors into points format
                points_data = np.hstack((coords_3d, colors))

                # Log 3D points to Comet
                experiment.log_points_3d(
                    scene_name="LSTM_Hidden_States",
                    points=points_data.tolist(),
                    step=len(history["train_losses"]) - 1,  # Last epoch
                    metadata={
                        "description": "LSTM hidden states visualized in 3D with prediction error coloring",
                        "feature_names": [
                            "PCA1",
                            "PCA2",
                            "PCA3",
                            "Red",
                            "Green",
                            "Blue",
                        ],
                        "max_error": float(max_error),
                        "mean_error": float(np.mean(error)),
                    },
                )
        except Exception as e:
            print(f"Failed to log 3D points from model data: {e}")
            # Fall back to logging
