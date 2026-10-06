import numpy as np
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Stacked LSTM encoder -> linear head producing the next `horizon` days in one shot."""

    def __init__(self, n_features: int, horizon: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])                    # (batch, horizon)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = [model(torch.from_numpy(X[i : i + batch_size])).numpy() for i in range(0, len(X), batch_size)]
    return np.concatenate(preds)
