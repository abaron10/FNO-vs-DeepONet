import torch
import numpy as np
from .base_operator import BaseOperator

class DeepONet(torch.nn.Module):
    """A minimal DeepONet suitable for variable grid sizes."""

    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 hidden_size: int = 128, num_layers: int = 4):
        super().__init__()
        self.branch_input_size = branch_input_size
        self.trunk_input_size = trunk_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ----- branch -----
        branch = [torch.nn.Linear(branch_input_size, hidden_size), torch.nn.ReLU()]
        for _ in range(num_layers - 1):
            branch.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()])
        branch.append(torch.nn.Linear(hidden_size, hidden_size))  # no activation
        self.branch_net = torch.nn.Sequential(*branch)
        # ----- trunk -----
        trunk = [torch.nn.Linear(trunk_input_size, hidden_size), torch.nn.ReLU()]
        for _ in range(num_layers - 1):
            trunk.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()])
        trunk.append(torch.nn.Linear(hidden_size, hidden_size))
        self.trunk_net = torch.nn.Sequential(*trunk)
        # bias term
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        """branch_in: [B, n_sensors]  trunk_in: [N, 2] returns [B, N]"""
        b = self.branch_net(branch_in)          # [B, H]
        t = self.trunk_net(trunk_in)            # [N, H]
        out = torch.sum(b.unsqueeze(1) * t.unsqueeze(0), dim=2)  # tensor product
        return out + self.bias


class DeepONetOperator(BaseOperator):
    """Adapter for our PyTorch DeepONet with adaptive grid size."""

    def __init__(self, device: torch.device, grid_size: int = 16, 
                 n_sensors: int = 1000, hidden_size: int = 128, num_layers: int = 4):
        super().__init__(device, grid_size)
        # Adaptive sensor count based on grid size
        max_sensors = grid_size ** 2
        self.n_sensors = min(n_sensors, max_sensors)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def setup(self, data_info):
        self.sensor_idx = data_info["sensor_idx"]
        self.trunk = torch.FloatTensor(data_info["coords"]).to(self.device)
        in_branch = len(self.sensor_idx)
        
        self.model = DeepONet(
            in_branch, 
            trunk_input_size=2, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()
        # Store sensor indices as list for JSON serialization
        self.sensor_idx_list = self.sensor_idx.tolist() if hasattr(self.sensor_idx, 'tolist') else list(self.sensor_idx)

    # helpers -------------------------------------------
    def _take_sensors(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W] -> [B, H*W] -> select
        B = x.shape[0]
        flat = x.view(B, -1)
        return flat[:, self.sensor_idx]

    # main interface ------------------------------------
    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        
        for batch in train_loader:
            self.opt.zero_grad()

            branch = self._take_sensors(batch["x"].to(self.device))
            pred = self.model(branch, self.trunk)  # [B, grid_size^2]

            tgt = batch["y"].to(self.device)
            B = tgt.shape[0]
            tgt = tgt.view(B, -1)

            loss = self.loss(pred, tgt)
            loss.backward()
            
            # Calcular accuracy para el batch actual
            batch_accuracy = self._calculate_accuracy(pred, tgt)
            running_accuracy += batch_accuracy
            
            self.opt.step()
            running_loss += loss.item()
        
        # Calcular métricas promedio
        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)
        
        # Calcular accuracy de validación si se proporciona val_loader
        val_accuracy = 0.0
        if val_loader is not None:
            val_accuracy = self._evaluate_accuracy(val_loader)
        
        return {
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'val_accuracy': val_accuracy
        }

    def _calculate_accuracy(self, pred, target, threshold=0.1):
        """Calcular accuracy como porcentaje de predicciones dentro del threshold"""
        rel_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
        accuracy = (rel_error < threshold).float().mean().item() * 100
        return accuracy

    @torch.no_grad()
    def _evaluate_accuracy(self, val_loader):
        """Evaluar accuracy en el conjunto de validación"""
        self.model.eval()
        total_accuracy = 0.0
        
        for batch in val_loader:
            branch = self._take_sensors(batch["x"].to(self.device))
            pred = self.model(branch, self.trunk)
            
            tgt = batch["y"].to(self.device)
            B = tgt.shape[0]
            tgt = tgt.view(B, -1)
            
            batch_accuracy = self._calculate_accuracy(pred, tgt)
            total_accuracy += batch_accuracy
        
        self.model.train()  # Volver a modo entrenamiento
        return total_accuracy / len(val_loader)

    def predict(self, batch):
        branch = self._take_sensors(batch["x"].to(self.device))
        pred = self.model(branch, self.trunk)
        # reshape back to [B,1,H,W] - use actual grid size
        B = branch.shape[0]
        return pred.view(B, 1, self.grid_size, self.grid_size)

    def get_model_info(self):
        return {
            "name": "Deep Operator Network (DeepONet)",
            "architecture": {
                "type": "Branch-Trunk neural network",
                "grid_size": f"{self.grid_size}×{self.grid_size}",
                "branch_input_size": self.model.branch_input_size,
                "trunk_input_size": self.model.trunk_input_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "n_sensors": self.n_sensors,
                "actual_sensors": len(self.sensor_idx) if hasattr(self, 'sensor_idx') else self.n_sensors
            },
            "parameters": self.count_parameters(),
            "optimizer": "Adam",
            "learning_rate": 1e-3
        }