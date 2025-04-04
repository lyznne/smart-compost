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


from typing import Dict
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class CompostModelTrainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        gradient_clip: float = None,
    ) -> dict:
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs to train.
            early_stopping_patience (int): Patience for early stopping.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            gradient_clip (float): Gradient clipping value.

        Returns:
            dict: Training history containing losses.
        """
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_losses": [], "val_losses": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip
                    )

                optimizer.step()

                train_loss += loss.item()

            # Step the scheduler
            if scheduler is not None:
                scheduler.step()

            # Calculate average training loss
            train_loss /= len(train_loader)
            history["train_losses"].append(train_loss)

            # Validation
            val_loss = self.validate(val_loader, criterion)
            history["val_losses"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), "best_compost_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        return history

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Validate the model.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.

        Returns:
            float: Validation loss.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)


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
