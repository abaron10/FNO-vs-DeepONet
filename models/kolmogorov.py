                      
import numpy as np
import torch
from kan import KAN
from typing import Dict, Any
from models.base_operator import BaseOperator                                    


class PyKANOperator(BaseOperator):

    def __init__(
        self,
        device: torch.device,
        hidden_neurons: int = 80,                                            
        lr: float = 1e-3,
        log_every: int = 5,                                   
    ):
        super().__init__(device)
        self.hidden_neurons = hidden_neurons
        self.lr = lr
        self.log_every = log_every

                                                                          
    def setup(self, data_info: Dict[str, Any]):
        N = data_info["coords"].shape[0]                 
        self.grid_size = int(np.sqrt(N))

        self.model = KAN(
            width=[N, self.hidden_neurons, N],
            grid=5,                           
            k=3
        ).to(self.device)

                                                                 
        self.model.ckpt_path = None
        self.model.save_every = 0
        self.model.save = lambda *_, **__: None                        

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

                                                                          
    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        running = 0.0

        for i, batch in enumerate(loader, start=1):
            self.opt.zero_grad()

            B = batch["x"].shape[0]
            x_flat = batch["x"].view(B, -1).to(self.device)
            y_flat = batch["y"].view(B, -1).to(self.device)

            pred = self.model(x_flat)
            loss = self.loss_fn(pred, y_flat)
            loss.backward()
            self.opt.step()

            running += loss.item()

            if self.log_every and i % self.log_every == 0:
                print(f"    • batch {i}/{len(loader)}  loss={loss.item():.3e}")

        return running / len(loader)

                                                                          
    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()

        B = batch["x"].shape[0]
        x_flat = batch["x"].view(B, -1).to(self.device)
        out = self.model(x_flat)                           

        G = self.grid_size
        return out.view(B, 1, G, G)

                                                                          
    def get_model_info(self):
        n_params = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        return {
            "name": f"PyKAN_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "width": self.model.width,
                "hidden_neurons": self.hidden_neurons,
            },
            "parameters": n_params,
            "optimizer": "Adam",
            "learning_rate": self.lr,
        }
