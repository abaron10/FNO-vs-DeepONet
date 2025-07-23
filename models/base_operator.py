import torch
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from typing import Dict, Any

class BaseOperator(ABC):

    def __init__(self, device: torch.device, grid_size: int = 16):
        self.device = device
        self.grid_size = grid_size

    @abstractmethod
    def setup(self, data_info: Dict[str, Any]):
        ...

    @abstractmethod
    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        ...

    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model architecture information"""
        ...

    @abstractmethod
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                    val_loader: torch.utils.data.DataLoader = None) -> Union[float, Dict[str, float]]:
        ...

    # ----------------------------------------------------
    def eval(self, loader, metrics: Dict[str, callable]):
        self.model.eval()
        agg = {k: 0.0 for k in metrics}
        with torch.no_grad():
            for batch in loader:
                pred = self.predict(batch)
                true = batch["y"].to(self.device)
                for k, f in metrics.items():
                    agg[k] += f(pred, true)
        return {k: v / len(loader) for k, v in agg.items()}

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)