"""
SMART COMPOST - MODEL PROJECT.

---  PyTorch Model 🌿
---   Models/model.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 4 Dec 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2025      - enos.vercel.app
"""

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),  # hyperbolic tangent
            nn.Linear(hidden_size, 1),
        )

        # Dropout layer for LSTM outputs
        self.dropout = nn.Dropout(dropout)

        # Output layers for temperature and moisture prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),  # Experiment with LeakyReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),  # temperature and moisture as the outputs
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for LSTM and attention layers."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.regression_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: [batch, seq_len, hidden_size]

        # Apply dropout to LSTM outputs
        lstm_out = self.dropout(lstm_out)

        # Apply attention
        attention_weights = self.attention(lstm_out)  # Shape: [batch, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # Shape: [batch, hidden_size]

        # Generate predictions
        predictions = self.regression_head(context)  # Shape: [batch, 2]

        # Apply output activation (optional)
        # predictions = torch.sigmoid(predictions)  # Use if outputs are normalized

        return predictions

    def get_hidden_states(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))

        # Return hidden states (last layer)
        return hn[-1]
