import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from neuralop.data.datasets import load_darcy_flow_small

class ResolutionAdaptiveDataLoader:
    """Wrapper to adapt data resolution on-the-fly"""
    
    def __init__(self, base_loader, target_grid_size):
        self.base_loader = base_loader
        self.target_grid_size = target_grid_size
        self.original_grid_size = 16  # neuralop's load_darcy_flow_small uses 16x16
    
    def __iter__(self):
        for batch in self.base_loader:
            # Interpolate to target resolution if needed
            if self.target_grid_size != self.original_grid_size:
                batch = self._resize_batch(batch)
            yield batch
    
    def __len__(self):
        return len(self.base_loader)
    
    def _resize_batch(self, batch):
        """Resize batch tensors to target grid size using bilinear interpolation"""
        x = batch["x"]  # [B, 1, H, W]
        y = batch["y"]  # [B, 1, H, W]
        
        # Resize using bilinear interpolation
        x_resized = F.interpolate(
            x, 
            size=(self.target_grid_size, self.target_grid_size), 
            mode='bilinear', 
            align_corners=False
        )
        y_resized = F.interpolate(
            y, 
            size=(self.target_grid_size, self.target_grid_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return {"x": x_resized, "y": y_resized}

@dataclass
class DataModule:
    grid: int = 16
    n_train: int = 100
    batch: int = 4
    n_test: int = 50
    n_sensors: int = 100  # Number of sensors for DeepONet

    def setup(self):
        # Always load base data at 16x16 and our target resolution for testing
        test_resolutions = [16]  # Base resolution
        if self.grid != 16:
            test_resolutions.append(self.grid)  # Add target resolution
        
        tr_loader, te_loaders, _ = load_darcy_flow_small(
            n_train=self.n_train,
            batch_size=self.batch,
            test_resolutions=test_resolutions,
            n_tests=[self.n_test] * len(test_resolutions),
            test_batch_sizes=[self.batch] * len(test_resolutions),
        )
        
        # Wrap loaders with resolution adaptation
        self.train = ResolutionAdaptiveDataLoader(tr_loader, self.grid)
        
        # Use the appropriate test loader
        if self.grid in te_loaders:
            self.test = te_loaders[self.grid]  # Use native resolution if available
        else:
            self.test = ResolutionAdaptiveDataLoader(te_loaders[16], self.grid)  # Resize base resolution
        
        # Adaptive sensor subset based on grid size
        rng = np.random.default_rng(42)
        N = self.grid ** 2
        # Ensure we don't try to sample more sensors than grid points
        actual_n_sensors = min(self.n_sensors, N)
        sensor_idx = rng.choice(N, size=actual_n_sensors, replace=False)
        
        # Generate coordinates for the specific grid size
        x = np.linspace(0, 1, self.grid)
        X, Y = np.meshgrid(x, x)
        coords = np.column_stack([X.flatten(), Y.flatten()])
        
        self.info = dict(sensor_idx=sensor_idx, coords=coords)

    def get_data_info(self):
        """Return dataset characteristics"""
        return {
            "name": "Darcy Flow",
            "description": "2D steady-state Darcy flow equation with variable permeability",
            "equation": "∇ · (κ(x)∇p(x)) = f(x)",
            "input": "Permeability field κ(x)",
            "output": "Pressure field p(x)",
            "grid_resolution": f"{self.grid}×{self.grid}",
            "n_train_samples": self.n_train,
            "n_test_samples": self.n_test,
            "batch_size": self.batch,
            "spatial_domain": "[0, 1]²",
            "n_grid_points": self.grid ** 2,
            "n_sensors": min(self.n_sensors, self.grid ** 2)
        }