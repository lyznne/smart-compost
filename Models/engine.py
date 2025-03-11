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

# Load environment variables
load_dotenv()


def train_compost_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyper_params: dict = None,
) -> CompostLSTM:
    """
    Train the compost model and log metrics to Comet ML.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        hyper_params (dict): Dictionary of hyperparameters. Defaults to None.

    Returns:
        CompostLSTM: Trained model.
    """
    # Default hyperparameters
    default_hyper_params = {
        "input_size": 10,
        "hidden_size": 64,
        "num_layers": 3,
        "dropout": 0.3,
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

    try:
        # Initialize Comet ML experiment
        experiment = Experiment(
            api_key=os.getenv("SMART_COMPOST_COMET_API_KEY"),
            project_name=os.getenv("SMART_COMPOST_PROJECT_NAME"),
            workspace=os.getenv("SMART_COMPOST_WORKSPACE"),
        )
    except Exception as e:
        print(f"Failed to initialize Comet ML experiment: {e}")
        experiment = None

    # Log hyperparameters
    if experiment:
        experiment.log_parameters(hyper_params)

    try:
        # Calculate input size from data
        sample_features, sample_targets = next(iter(train_loader))
        input_size = sample_features.shape[-1]  # number of features
    except Exception as e:
        print(f"Failed to calculate input size: {e}")
        return None

    # Initialize model and trainer
    model = CompostLSTM(
        input_size=input_size,
        hidden_size=hyper_params["hidden_size"],
        num_layers=hyper_params["num_layers"],
        dropout=hyper_params["dropout"],
    )
    trainer = CompostModelTrainer(model)

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

    # Log training and validation losses
    if experiment:
        for epoch, (train_loss, val_loss) in enumerate(
            zip(history["train_losses"], history["val_losses"])
        ):
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)

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
            # Fall back to logging basic 3D visualization if the advanced one fails
            try:
                # Generate points from model weights (first 3 layers)
                params = list(model.parameters())
                if len(params) >= 3:
                    weights = [
                        p.detach().cpu().numpy().flatten()[:100] for p in params[:3]
                    ]
                    weights = [
                        w / np.linalg.norm(w) if np.linalg.norm(w) > 0 else w
                        for w in weights
                    ]
                    weights_array = np.array(weights).T

                    # Ensure we have at least 10 points
                    if weights_array.shape[0] < 10:
                        weights_array = np.tile(
                            weights_array, (10 // weights_array.shape[0] + 1, 1)
                        )[:10]

                    # Add simple coloring
                    colors = np.zeros((weights_array.shape[0], 3))
                    colors[:, 0] = np.linspace(
                        0, 1, colors.shape[0]
                    )  # Gradient coloring
                    colors[:, 2] = np.linspace(1, 0, colors.shape[0])

                    points = np.hstack((weights_array, colors))

                    experiment.log_points_3d(
                        scene_name="Model_Weights",
                        points=points.tolist(),
                        step=len(history["train_losses"]) - 1,
                        metadata={
                            "description": "Model weights visualized in 3D space"
                        },
                    )
            except Exception as nested_e:
                print(f"Failed to log fallback 3D visualization: {nested_e}")

    # Save training metadata to the database
    try:
        training_run = TrainingRun(
            model_name="CompostLSTM",
            experiment_id=experiment.get_key() if experiment else None,
            parameters=dumps(hyper_params),
            metrics=dumps(
                {
                    "final_train_loss": history["train_losses"][-1],
                    "final_val_loss": history["val_losses"][-1],
                }
            ),
            status="completed",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )
        db.session.add(training_run)
        db.session.commit()
    except Exception as e:
        print(f"Failed to save training metadata to the database: {e}")

    # End the experiment
    if experiment:
        experiment.end()

    return model


# def train_compost_model(
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     hyper_params: dict = None,
# ) -> CompostLSTM:
#     """
#     Train the compost model and log metrics to Comet ML.

#     Args:
#         train_loader (DataLoader): DataLoader for training data.
#         val_loader (DataLoader): DataLoader for validation data.
#         hyper_params (dict): Dictionary of hyperparameters. Defaults to None.

#     Returns:
#         CompostLSTM: Trained model.
#     """
#     # Default hyperparameters
#     default_hyper_params = {
#         "input_size": 10,
#         "hidden_size": 64,
#         "num_layers": 2,
#         "dropout": 0.3,
#         "learning_rate": 0.0001,
#         "batch_size": 64,
#         "epochs": 100,
#         "early_stopping_patience": 15,
#         "gradient_clip": 1.0,
#     }

#     # Use provided hyperparameters or defaults
#     if hyper_params is not None:
#         default_hyper_params.update(hyper_params)
#     hyper_params = default_hyper_params

#     # Initialize Comet ML experiment
#     experiment = Experiment(
#         api_key=os.getenv("SMART_COMPOST_COMET_API_KEY"),
#         project_name=os.getenv("SMART_COMPOST_PROJECT_NAME"),
#         workspace=os.getenv("SMART_COMPOST_WORKSPACE"),
#     )

#     # Log hyperparameters
#     experiment.log_parameters(hyper_params)

#     # Calculate input size from data
#     sample_features = next(iter(train_loader))[0]
#     input_size = sample_features.shape[-1]  # number of features

#     # Initialize model and trainer
#     model = CompostLSTM(
#         input_size=input_size,
#         hidden_size=hyper_params["hidden_size"],
#         num_layers=hyper_params["num_layers"],
#         dropout=hyper_params["dropout"],
#     )
#     trainer = CompostModelTrainer(model)

#     # Define optimizer and scheduler
#     optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])
#     scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs

#     # Train the model
#     history = trainer.train(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         epochs=hyper_params["epochs"],
#         early_stopping_patience=hyper_params["early_stopping_patience"],
#         optimizer=optimizer,
#         scheduler=scheduler,
#         gradient_clip=hyper_params["gradient_clip"],
#     )

#     # Log training and validation losses
#     for epoch, (train_loss, val_loss) in enumerate(
#         zip(history["train_losses"], history["val_losses"])
#     ):
#         experiment.log_metric("train_loss", train_loss, step=epoch)
#         experiment.log_metric("val_loss", val_loss, step=epoch)

#     # Save the best model
#     model_path = "best_compost_model.pth"
#     torch.save(model.state_dict(), model_path)

#     # Log the model to Comet ML
#     log_model(experiment, model=model, model_name="CompostLSTM")

#     # Log a 3D table of training metrics
#     table_data = []
#     for epoch, (train_loss, val_loss) in enumerate(
#         zip(history["train_losses"], history["val_losses"])
#     ):
#         # Calculate learning rate for the current epoch
#         lr = optimizer.param_groups[0]["lr"]

#         # Add additional metrics (e.g., gradient norms, learning rate)
#         table_data.append(
#             {
#                 "epoch": epoch + 1,
#                 "train_loss": train_loss,
#                 "val_loss": val_loss,
#                 "learning_rate": lr,
#                 "gradient_norm": history.get(
#                     "gradient_norms", [0] * len(history["train_losses"])
#                 )[
#                     epoch
#                 ],  # Log gradient norm (if available)
#             }
#         )

#     # Log the 3D table
#     experiment.log_table(
#         filename="training_metrics_3d.csv",
#         tabular_data=table_data,
#         headers=["epoch", "train_loss", "val_loss", "learning_rate", "gradient_norm"],
#     )

#     # Save training metadata to the database
#     training_run = TrainingRun(
#         model_name="CompostLSTM",
#         experiment_id=experiment.get_key(),
#         parameters=dumps(hyper_params),
#         metrics=dumps(
#             {
#                 "final_train_loss": history["train_losses"][-1],
#                 "final_val_loss": history["val_losses"][-1],
#             }
#         ),
#         status="completed",
#         start_time=datetime.utcnow(),
#         end_time=datetime.utcnow(),
#     )
#     db.session.add(training_run)
#     db.session.commit()

#     # End the experiment
#     experiment.end()

#     return model
