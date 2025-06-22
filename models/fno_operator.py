import torch
from .base_operator import BaseOperator

class FNOOperator(BaseOperator):
    """Thin adapter around Neural Operator FNO with adaptive grid size."""

    def __init__(self, device: torch.device, grid_size: int = 16, 
                 hidden_channels: int = 32, n_layers: int = 3):
        super().__init__(device, grid_size)
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        # Adaptive modes based on grid size - use approximately half the grid size
        # but ensure it's reasonable for the spectral method
        max_modes = min(grid_size // 2, 16)  # Cap at 16 for computational efficiency
        self.n_modes = (max_modes, max_modes)

    def setup(self, data_info):
        from neuralop.models import FNO

        self.model = FNO(
            n_modes=self.n_modes,
            hidden_channels=self.hidden_channels,
            in_channels=1,
            out_channels=1,
            lifting_channels=32,
            projection_channels=32,
            n_layers=self.n_layers,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()

    def train_epoch(self, loader):
        self.model.train()
        running = 0.0
        for batch in loader:
            self.opt.zero_grad()
            pred = self.model(batch["x"].to(self.device))
            loss = self.loss(pred, batch["y"].to(self.device))
            loss.backward()
            self.opt.step()
            running += loss.item()
        return running / len(loader)

    def predict(self, batch):
        return self.model(batch["x"].to(self.device))

    def get_model_info(self):
        return {
            "name": "Fourier Neural Operator (FNO)",
            "architecture": {
                "type": "Spectral convolution in Fourier space",
                "grid_size": f"{self.grid_size}×{self.grid_size}",
                "n_modes": self.n_modes,
                "hidden_channels": self.hidden_channels,
                "n_layers": self.n_layers,
                "lifting_channels": 32,
                "projection_channels": 32
            },
            "parameters": self.count_parameters(),
            "optimizer": "Adam",
            "learning_rate": 1e-3
        }