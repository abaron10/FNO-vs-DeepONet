import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from neuralop.data.datasets import load_darcy_flow_small

class ResolutionAdaptiveDataLoader:
    """Wrapper to adapt data resolution on-the-fly."""
    
    def __init__(self, base_loader, target_grid_size):
        self.base_loader = base_loader
        self.target_grid_size = target_grid_size
        # infer original from first batch
        first_batch = next(iter(base_loader))
        self.original_grid_size = first_batch["x"].shape[-1]
    
    def __iter__(self):
        for batch in self.base_loader:
            if self.target_grid_size != self.original_grid_size:
                batch = self._resize_batch(batch)
            yield batch
    
    def __len__(self):
        return len(self.base_loader)
    
    def _resize_batch(self, batch):
        x, y = batch["x"], batch["y"]  # [B,1,H,W]
        x_resized = F.interpolate(
            x,
            size=(self.target_grid_size, self.target_grid_size),
            mode="bilinear",
            align_corners=False
        )
        y_resized = F.interpolate(
            y,
            size=(self.target_grid_size, self.target_grid_size),
            mode="bilinear",
            align_corners=False
        )
        return {"x": x_resized, "y": y_resized}


@dataclass
class DataModule:
    grid: int = 16
    n_train: int = 100
    batch: int = 4
    n_test: int = 50
    n_sensors: int = 100  # for DeepONet

    def setup(self):
        # decide test resolutions
        test_res = [16]
        if self.grid != 16:
            test_res.append(self.grid)

        tr_loader, te_loaders, _ = load_darcy_flow_small(
            n_train=self.n_train,
            batch_size=self.batch,
            test_resolutions=test_res,
            n_tests=[self.n_test] * len(test_res),
            test_batch_sizes=[self.batch] * len(test_res),
        )

        # training loader at desired grid
        self.train = ResolutionAdaptiveDataLoader(tr_loader, self.grid)

        # test loader: native if available, else adapt 16→grid
        if self.grid in te_loaders:
            self.test = te_loaders[self.grid]
        else:
            self.test = ResolutionAdaptiveDataLoader(te_loaders[16], self.grid)

        # compute κ(x) normalization stats on training set
        k_list = []
        for batch in self.train:
            k_list.append(batch["x"])
        k_all = torch.cat(k_list, dim=0)
        k_mean = k_all.mean().item()
        k_std  = k_all.std().item()

        # sensors & coords for DeepONet
        rng = np.random.default_rng(42)
        N = self.grid ** 2
        actual_n = min(self.n_sensors, N)
        sensor_idx = rng.choice(N, size=actual_n, replace=False)

        x = np.linspace(0, 1, self.grid)
        X, Y = np.meshgrid(x, x)
        coords = np.column_stack([X.flatten(), Y.flatten()])

        self.info = {
            "sensor_idx": sensor_idx,
            "coords": coords,
            "k_mean": k_mean,
            "k_std": k_std
        }

    def get_data_info(self):
        return {
            "name": "Darcy Flow",
            "description": "2D steady-state Darcy flow with variable permeability",
            "equation": "∇·(κ(x)∇p(x)) = f(x)",
            "input": "Permeability κ(x)",
            "output": "Pressure p(x)",
            "grid_resolution": f"{self.grid}×{self.grid}",
            "n_train_samples": self.n_train,
            "n_test_samples": self.n_test,
            "batch_size": self.batch,
            "n_grid_points": self.grid ** 2,
            "n_sensors": min(self.n_sensors, self.grid ** 2),
            "k_mean": self.info["k_mean"],
            "k_std": self.info["k_std"],
        }
