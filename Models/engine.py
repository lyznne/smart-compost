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
        sample_features = next(iter(train_loader))[0]
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

    # Log a 3D table of training metrics
    table_data = []
    for epoch, (train_loss, val_loss) in enumerate(
        zip(history["train_losses"], history["val_losses"])
    ):
        # Calculate learning rate for the current epoch
        lr = optimizer.param_groups[0]["lr"]

        # Add additional metrics (e.g., gradient norms, learning rate)
        table_data.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr,
                "gradient_norm": history.get(
                    "gradient_norms", [0] * len(history["train_losses"])
                )[
                    epoch
                ],  # Log gradient norm (if available)
            }
        )

    # Log the 3D table
    if experiment:
        experiment.log_table(
            filename="training_metrics_3d.csv",
            tabular_data=table_data,
            headers=[
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate",
                "gradient_norm",
            ],
        )

    # Log 3D points (example)
    if experiment:
        try:
            # Generate synthetic 3D points for demonstration
            import numpy as np

            np.random.seed(42)
            points = np.random.rand(10, 6)  # 10 points, with XYZ and RGB values
            points[:, 3:] = points[:, 3:] / np.max(points[:, 3:])  # Normalize colors

            # Log 3D points to Comet
            experiment.log_points_3d(
                scene_name="Training3DPoints",
                points=points.tolist(),  # Convert numpy array to list
                step=0,  # Associate with step 0
                metadata={"description": "Example 3D points logged during training"},
            )
        except Exception as e:
            print(f"Failed to log 3D points: {e}")

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
