"""
SMART COMPOST - MODEL PROJECT.

---  PyTorch Model 🌿
---   Models/model.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 4 Dec 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# imports
import torch
import torch.nn as nn


class CompostLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(CompostLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

        # Output layers for temperature and moisture prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),  # temperature and moisture as the outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: [batch, seq_len, hidden_size]

        # Apply attention
        attention_weights = self.attention(lstm_out)  # Shape: [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # Shape: [batch, hidden_size]

        # Generate predictions
        predictions = self.regression_head(context)  # Shape: [batch, 2]

        return predictions
