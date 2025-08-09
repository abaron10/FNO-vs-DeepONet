import math
import numpy as np
import torch
from .base_operator import BaseOperator

# ------------------------ Fourier features ------------------------
class FourierFeatures(torch.nn.Module):
    def __init__(self, in_dim=2, m=32, scale=2.0 * math.pi):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        self.register_buffer("B", B)

    def forward(self, x):  # x: [N,2]
        xb = x @ self.B
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # [N,2m]

# ------------------------ Residual MLP block ------------------------
class ResBlock(torch.nn.Module):
    def __init__(self, d, p=0.0, act="gelu"):
        super().__init__()
        act_fn = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "silu": torch.nn.SiLU}.get(act, torch.nn.GELU)
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(d),
            torch.nn.Linear(d, d * 4),
            act_fn(),
            torch.nn.Dropout(p),
            torch.nn.Linear(d * 4, d),
        )

    def forward(self, x):
        return x + self.net(x)

# ------------------------ DeepONet (Fourier + FiLM + dot) ------------------------
class DeepONet(torch.nn.Module):
    """
    Branch-Trunk con:
      - Fourier features en el trunk
      - Bloques residuales + LayerNorm
      - FiLM (branch produce gammas/betas que modulan el trunk)
      - Ruta clásica de dot-product + pequeño bias dependiente de x
    Mantiene la misma interfaz de forward(branch_in, trunk_in) -> [B, N]
    """
    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 hidden_size: int = 256, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 fourier_m: int = 32):
        super().__init__()
        self.hidden = hidden_size
        self.activation = activation

        # Fourier features para el trunk
        self.ff = FourierFeatures(trunk_input_size, m=fourier_m)

        # ----- Branch: embedding + parámetros FiLM -----
        b_layers = [torch.nn.Linear(branch_input_size, hidden_size)]
        for _ in range(num_layers - 1):
            b_layers.append(ResBlock(hidden_size, p=dropout, act=activation))
        self.branch_net = torch.nn.Sequential(*b_layers)
        self.to_gamma = torch.nn.Linear(hidden_size, num_layers * hidden_size)
        self.to_beta = torch.nn.Linear(hidden_size, num_layers * hidden_size)

        # ----- Trunk -----
        t_in = 2 * fourier_m
        self.trunk_in = torch.nn.Linear(t_in, hidden_size)
        self.trunk_blocks = torch.nn.ModuleList([ResBlock(hidden_size, p=dropout, act=activation)
                                                 for _ in range(num_layers)])

        # Proyecciones para ruta de producto punto
        self.branch_out = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.trunk_out = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Bias dependiente de x y bias global
        self.bias_trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 4, 1)
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        # Branch
        B = self.branch_net(branch_in)                      # [B,H]
        gammas = self.to_gamma(B).view(B.size(0), -1, self.hidden)  # [B,L,H]
        betas = self.to_beta(B).view(B.size(0), -1, self.hidden)    # [B,L,H]

        # Trunk con Fourier
        T = self.trunk_in(self.ff(trunk_in))               # [N,H]
        for l, blk in enumerate(self.trunk_blocks):
            T = blk(T)                                     # [N,H]
            # FiLM broadcast a [B,N,H]
            g = gammas[:, l, :].unsqueeze(1)               # [B,1,H]
            b = betas[:, l, :].unsqueeze(1)                # [B,1,H]
            T = g * T.unsqueeze(0) + b + T.unsqueeze(0)    # [B,N,H]

        # Ruta dot-product clásica
        b_proj = self.branch_out(B).unsqueeze(1)           # [B,1,H]
        t_proj = self.trunk_out(T)                         # [B,N,H]
        dot = (b_proj * t_proj).sum(-1)                    # [B,N]

        # Bias dependiente de x
        bias_x = self.bias_trunk(T).squeeze(-1)            # [B,N]
        return dot + bias_x + self.bias


# ------------------------ Operator ------------------------
class DeepONetOperator(BaseOperator):
    """
    Operador compatible con BaseOperator.
    - Sensores: random/uniform/chebyshev/adaptive (como antes)
    - Normaliza x (sensores) e y (campo)
    - CosineAnnealingWarmRestarts + AMP + EMA de pesos
    """
    def __init__(self, device: torch.device, name: str = "", grid_size: int = 64,
                 n_sensors: int = 256, hidden_size: int = 256, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 lr: float = 3e-4, epochs: int = 600, weight_decay: float = 1e-4,
                 step_size: int = 100, gamma: float = 0.5,   # mantienen firma usada en main aunque no se usen
                 sensor_strategy: str = 'chebyshev', normalize_sensors: bool = True):
        super().__init__(device, grid_size)
        self.name = name
        self.n_sensors = n_sensors
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.sensor_strategy = sensor_strategy
        self.normalize_sensors = normalize_sensors

        self.best_val_loss = float('inf')
        self._ema = None
        self.ema_decay = 0.995
        self.use_amp = (device.type == "cuda")

    # ---------- setup ----------
    def setup(self, data_info):
        self._setup_sensors_and_coords(data_info)

        in_branch = len(self.sensor_idx)
        self.model = DeepONet(
            in_branch,
            trunk_input_size=2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            activation=self.activation,
            dropout=self.dropout,
            fourier_m=32
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        # Cosine con warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.loss_fn = torch.nn.MSELoss()

        # stats de normalización (se van actualizando on-the-fly)
        self._x_mu = torch.zeros(1, in_branch, device=self.device)
        self._x_std = torch.ones(1, in_branch, device=self.device)
        self._y_mu = 0.0
        self._y_std = 1.0

    # ---------- sensores/coords ----------
    def _setup_sensors_and_coords(self, data_info):
        max_sensors = self.grid_size ** 2
        k = min(self.n_sensors, max_sensors)

        if self.sensor_strategy == 'random':
            rng = np.random.default_rng(42)
            self.sensor_idx = rng.choice(max_sensors, size=k, replace=False)
        elif self.sensor_strategy == 'uniform':
            step = max(1, max_sensors // k)
            self.sensor_idx = np.arange(0, max_sensors, step)[:k]
        elif self.sensor_strategy == 'chebyshev':
            n = int(np.sqrt(k))
            cheb = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
            cheb = (cheb + 1) / 2
            idx = []
            for i in range(n):
                for j in range(n):
                    x = int(cheb[i] * (self.grid_size - 1))
                    y = int(cheb[j] * (self.grid_size - 1))
                    idx.append(x * self.grid_size + y)
            self.sensor_idx = np.array(idx[:k])
        elif self.sensor_strategy == 'adaptive':
            rng = np.random.default_rng(42)
            base = int(k * 0.6)
            step = max(1, int(np.sqrt((self.grid_size ** 2) / base)))
            uniform = []
            for i in range(0, self.grid_size, step):
                for j in range(0, self.grid_size, step):
                    if len(uniform) < base:
                        uniform.append(i * self.grid_size + j)
            all_sel = set(uniform)
            edges = set()
            G = self.grid_size
            for i in range(G):
                edges.update({i, (G - 1) * G + i, i * G, i * G + (G - 1)})
            edges = list(edges)
            all_sel.update(edges[:int(k * 0.2)])
            remain = k - len(all_sel)
            if remain > 0:
                pool = list(set(range(max_sensors)) - all_sel)
                all_sel.update(rng.choice(pool, size=remain, replace=False))
            self.sensor_idx = np.array(sorted(all_sel))[:k]
        else:
            rng = np.random.default_rng(42)
            self.sensor_idx = rng.choice(max_sensors, size=k, replace=False)

        self.sensor_idx = np.sort(self.sensor_idx)

        x = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, x, indexing='ij')
        coords = np.column_stack([X.flatten(), Y.flatten()])
        if self.normalize_sensors:
            coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
        self.trunk = torch.tensor(coords, dtype=torch.float32, device=self.device)
        self.sensor_idx_list = self.sensor_idx.tolist()
        print(f"✓ Set up {len(self.sensor_idx)} sensors using '{self.sensor_strategy}' strategy")

    # ---------- utils ----------
    def _take_sensors(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x.view(B, -1)[:, self.sensor_idx]

    def _update_stats(self, xb, yb, momentum=0.01):
        with torch.no_grad():
            mu = xb.mean(0, keepdim=True); sd = xb.std(0, keepdim=True).clamp_min(1e-6)
            self._x_mu = (1 - momentum) * self._x_mu + momentum * mu
            self._x_std = (1 - momentum) * self._x_std + momentum * sd
            mu_y = yb.mean(); sd_y = yb.std().clamp_min(1e-6)
            self._y_mu = float((1 - momentum) * self._y_mu + momentum * mu_y)
            self._y_std = float((1 - momentum) * self._y_std + momentum * sd_y)

    def _norm_x(self, xb): return (xb - self._x_mu) / (self._x_std + 1e-8)
    def _norm_y(self, y):  return (y - self._y_mu) / (self._y_std + 1e-8)
    def _denorm_y(self, y): return y * self._y_std + self._y_mu

    def _rel_l2_batch(self, pred, target):
        d = (pred - target).view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        return (d.norm(dim=1) / (t.norm(dim=1) + 1e-8))

    # ---------- train ----------
    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0

        for step, batch in enumerate(train_loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            xb = self._take_sensors(x)
            self._update_stats(xb, y)
            xb = self._norm_x(xb)
            yt = self._norm_y(y.view(y.size(0), -1))

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(xb, self.trunk)
                # combinación MSE + término de relative L2 (suaviza optimización)
                rel = self._rel_l2_batch(pred, yt).mean()
                loss = self.loss_fn(pred, yt) + 0.2 * rel

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(step)

            with torch.no_grad():
                total_loss += loss.item() * x.size(0)
                total_acc += (1.0 - rel.item()) * 100 * x.size(0)
                total_n += x.size(0)

            # EMA de pesos
            with torch.no_grad():
                if self._ema is None:
                    self._ema = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
                else:
                    for ema, p in zip(self._ema, [q for q in self.model.parameters() if q.requires_grad]):
                        ema.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

        val_loss, val_acc = (float("inf"), 0.0)
        if val_loader is not None:
            val_loss, val_acc = self.evaluate(val_loader)

        return {
            'train_loss': total_loss / total_n,
            'train_accuracy': total_acc / total_n,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'lr': self.optimizer.param_groups[0]['lr'],
            'should_stop': False,
            'info': 'Accuracy = 100*(1 - relative_L2)'
        }

    @torch.no_grad()
    def _with_ema(self, fn):
        if self._ema is None:
            return fn()
        params = [p for p in self.model.parameters() if p.requires_grad]
        backup = [p.detach().clone() for p in params]
        for p, e in zip(params, self._ema):
            p.data.copy_(e)
        out = fn()
        for p, b in zip(params, backup):
            p.data.copy_(b)
        return out

    @torch.no_grad()
    def evaluate(self, data_loader):
        def _eval():
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total_n = 0
            for batch in data_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                xb = self._norm_x(self._take_sensors(x))
                yt = self._norm_y(y.view(y.size(0), -1))
                pred = self.model(xb, self.trunk)
                loss = self.loss_fn(pred, yt)
                rel = self._rel_l2_batch(pred, yt).mean().item()
                total_loss += loss.item() * x.size(0)
                total_acc += (1.0 - rel) * 100 * x.size(0)
                total_n += x.size(0)
            return total_loss / total_n, total_acc / total_n

        return self._with_ema(_eval)

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        x = batch["x"].to(self.device)
        xb = self._norm_x(self._take_sensors(x))
        pred = self.model(xb, self.trunk)                      # normalizado
        B = xb.size(0)
        pred = self._denorm_y(pred).view(B, 1, self.grid_size, self.grid_size)
        return pred

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "name": f"{self.name}_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "type": "DeepONet Fourier + FiLM",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "sensor_strategy": self.sensor_strategy
            },
            "parameters": trainable_params,
            "total_parameters": total_params,
            "optimizer": f"AdamW(lr={self.lr})",
            "accuracy_method": "100*(1-relative_L2)"
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
