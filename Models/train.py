"""
SMART COMPOST - MODEL PROJECT.

---  the training of the model
---   Models/train.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 3 Dec 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# imports
from typing import Dict
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CompostModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_features)
            loss = self.criterion(predictions, batch_targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ) -> Dict:

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_compost_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.show()


# def train_model(model_name, hyperparameters):
#     """Train a model and log results to Comet ML and MySQL."""

#     # Create a new Comet ML experiment
#     experiment = Experiment(
#         api_key=COMET_API_KEY,
#         project_name=COMET_PROJECT_NAME,
#         workspace=COMET_WORKSPACE
#     )

#     # Start tracking in MySQL
#     training_run = TrainingRun(
#         model_name=model_name,
#         experiment_id=experiment.get_key(),
#         parameters=json.dumps(hyperparameters),
#         status="running"
#     )
#     db.session.add(training_run)
#     db.session.commit()

#     try:
#         # Simulate training (Replace with actual ML model training)
#         for epoch in range(1, 6):
#             loss = 0.05 * (6 - epoch)  # Simulated loss
#             accuracy = 0.80 + (0.05 * epoch)  # Simulated accuracy

#             # Log metrics
#             experiment.log_metric("loss", loss, step=epoch)
#             experiment.log_metric("accuracy", accuracy, step=epoch)

#             print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
#             time.sleep(1)  # Simulate training time

#         # Mark training as complete
#         training_run.status = "completed"
#         training_run.metrics = json.dumps({"loss": loss, "accuracy": accuracy})
#         training_run.end_time = datetime.utcnow()
#         db.session.commit()

#     except Exception as e:
#         # Log failure
#         training_run.status = "failed"
#         db.session.commit()
#         print(f"Training failed: {e}")

#     finally:
#         # End Comet experiment
#         experiment.end()

#     return training_run
