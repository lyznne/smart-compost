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


# Load environment variables
load_dotenv()

# Load environment variables
load_dotenv()


def train_compost_model(
    train_loader: DataLoader, val_loader: DataLoader
) -> CompostLSTM:


    """
    Train the compost model and log metrics to Comet ML.
    """

    # Initialize Comet ML experiment
    experiment = Experiment(
        api_key=os.getenv("SMART_COMPOST_COMET_API_KEY"),
        project_name=os.getenv("SMART_COMPOST_PROJECT_NAME"),
        workspace=os.getenv("SMART_COMPOST_WORKSPACE"),
    )

    # Log hyperparameters
    hyper_params = {
        "input_size": 10,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
    }
    experiment.log_parameters(hyper_params)

    # Calculate input size from data
    sample_features = next(iter(train_loader))[0]
    input_size = sample_features.shape[-1]  # number of features

    # Initialize model and trainer
    model = CompostLSTM(input_size=input_size)
    trainer = CompostModelTrainer(model)

    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=10,
    )

    # Log training and validation losses
    for epoch, (train_loss, val_loss) in enumerate(
        zip(history["train_losses"], history["val_losses"])
    ):
        experiment.log_metric("train_loss", train_loss, step=epoch)
        experiment.log_metric("val_loss", val_loss, step=epoch)

    # Save the trained model
    model_path = "best_compost_model.pth"
    torch.save(model.state_dict(), model_path)

    # Log the model to Comet ML
    log_model(experiment, model=model, model_name="CompostLSTM")

    # Save training metadata to the database
    training_run = TrainingRun(
        model_name="CompostLSTM",
        experiment_id=experiment.get_key(),
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

    # End the experiment
    experiment.end()

    return model
