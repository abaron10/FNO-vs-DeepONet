import math
import numpy as np
import torch
import torch.nn.functional as F
from .base_operator import BaseOperator

# ------------------------ Fourier features ------------------------
class FourierFeatures(torch.nn.Module):
    def __init__(self, in_dim=2, m=96, scale=2.0 * math.pi):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        self.register_buffer("B", B)

    def forward(self, x):  # [N,2]
        xb = x @ self.B
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # [N,2m]

# ------------------------ Residual block ------------------------
class ResBlock(torch.nn.Module):
    def __init__(self, d, p=0.0, act="gelu"):
        super().__init__()
        act_fn = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "silu": torch.nn.SiLU}.get(act, torch.nn.GELU)
        self.ln = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, d * 4)
        self.act = act_fn()
        self.drop = torch.nn.Dropout(p)
        self.fc2 = torch.nn.Linear(d * 4, d)

    def forward(self, x):  # [..., d]
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y

# ------------------------ DeepONet (Fourier + FiLM estable) ------------------------
class DeepONet(torch.nn.Module):
    """
    Forward: (branch_in[B,S], trunk_in[N,2]) -> [B,N]
    """
    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 hidden_size: int = 256, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 fourier_m: int = 96):
        super().__init__()
        self.hidden = hidden_size
        self.num_layers = num_layers

        self.ff = FourierFeatures(trunk_input_size, m=fourier_m)

        # Branch -> embedding y parámetros FiLM (γ, β)
        b_layers = [torch.nn.Linear(branch_input_size, hidden_size)]
        for _ in range(num_layers - 1):
            b_layers.append(ResBlock(hidden_size, p=dropout, act=activation))
        self.branch_net = torch.nn.Sequential(*b_layers)
        self.to_gamma = torch.nn.Linear(hidden_size, num_layers * hidden_size)
        self.to_beta  = torch.nn.Linear(hidden_size, num_layers * hidden_size)

        # Trunk
        t_in = 2 * fourier_m
        self.trunk_in = torch.nn.Linear(t_in, hidden_size)
        self.trunk_blocks = torch.nn.ModuleList(
            [ResBlock(hidden_size, p=dropout, act=activation) for _ in range(num_layers)]
        )

        # Heads
        self.branch_out = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.trunk_out  = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 4, 1)
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))

        # escalas para estabilizar
        self.dot_scale = 1.0 / math.sqrt(hidden_size)
        self.beta_scale = 0.1
        self.gamma_scale = 0.5

        # init
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        # Branch
        B = self.branch_net(branch_in)  # [B,H]
        gammas = self.to_gamma(B).view(B.size(0), self.num_layers, self.hidden)  # [B,L,H]
        betas  = self.to_beta(B).view(B.size(0), self.num_layers, self.hidden)   # [B,L,H]

        # Estabilizar γ, β
        gammas = 1.0 + self.gamma_scale * torch.tanh(gammas)  # ~ [0.5, 1.5]
        betas  = self.beta_scale * torch.tanh(betas)           # ~ [-0.1, 0.1]

        # Trunk
        T = self.trunk_in(self.ff(trunk_in))  # [N,H]

        # Capas + FiLM (sin residual extra)
        for l, blk in enumerate(self.trunk_blocks):
            T = blk(T)                                      # [N,H]
            g = gammas[:, l, :].unsqueeze(1)               # [B,1,H]
            b = betas[:, l, :].unsqueeze(1)                # [B,1,H]
            T = g * T.unsqueeze(0) + b                     # [B,N,H]

        # Proyecciones y combinación
        b_proj = self.branch_out(B).unsqueeze(1)           # [B,1,H]
        t_proj = self.trunk_out(T)                         # [B,N,H]
        dot = ((b_proj * t_proj).sum(-1)) * self.dot_scale # [B,N]
        bias_x = 0.1 * self.bias_trunk(T).squeeze(-1)      # [B,N]
        out = dot + bias_x + self.bias
        return out.contiguous()                            # [B,N]

# ------------------------ Operator ------------------------
class DeepONetOperator(BaseOperator):
    """
    - Normalización online (para loss), accuracy en escala real: 100*(1 - relative_L2_error).
    - AMP torch.amp + CosineAnnealingWarmRestarts + EMA.
    - Sensor Dropout con annealing + Pérdida Sobolev (gradientes espaciales).
    - **SWA** (Stochastic Weight Averaging) desde 60% de training para bajar error de generalización.
    - **Loss schedule** 2 fases (caliente → afinado).
    """
    def __init__(self, device: torch.device, name: str = "", grid_size: int = 64,
                 n_sensors: int = 256, hidden_size: int = 256, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 lr: float = 3e-4, epochs: int = 600, weight_decay: float = 1e-4,
                 step_size: int = 100, gamma: float = 0.5,
                 sensor_strategy: str = 'chebyshev', normalize_sensors: bool = True,
                 fourier_m: int = 96,
                 sensor_dropout_p: float = 0.10,
                 grad_loss_weight: float = 0.10,
                 swa_start_frac: float = 0.6):
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
        self.fourier_m = fourier_m

        self.sensor_dropout_p0 = sensor_dropout_p
        self.grad_loss_weight0 = grad_loss_weight

        self._ema = None
        self.ema_decay = 0.995
        self.use_amp = (device.type == "cuda")
        self._epoch_float = 0.0
        self._stats_initialized = False
        self._global_epoch = 0

        # SWA
        self.swa_start_frac = swa_start_frac
        self._swa_params = None
        self._swa_n = 0

        # kernels Sobel como tensores normales (no buffers de nn.Module)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(self.device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(self.device)

    # ---------- setup ----------
    def setup(self, data_info):
        self._setup_sensors_and_coords(data_info)

        in_branch = len(self.sensor_idx)
        self.model = DeepONet(
            in_branch, trunk_input_size=2,
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            activation=self.activation, dropout=self.dropout, fourier_m=self.fourier_m
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.loss_fn = torch.nn.MSELoss()

        # stats normalización
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
            cheb = np.cos((2*np.arange(n)+1)*np.pi/(2*n))
            cheb = (cheb + 1) / 2
            idx=[]
            for i in range(n):
                for j in range(n):
                    x = int(cheb[i]*(self.grid_size-1))
                    y = int(cheb[j]*(self.grid_size-1))
                    idx.append(x*self.grid_size + y)
            self.sensor_idx = np.array(idx[:k])
        elif self.sensor_strategy == 'adaptive':
            rng = np.random.default_rng(42)
            base = int(k*0.6)
            step = max(1, int(np.sqrt((self.grid_size**2)/base)))
            uniform=[]
            for i in range(0, self.grid_size, step):
                for j in range(0, self.grid_size, step):
                    if len(uniform) < base:
                        uniform.append(i*self.grid_size + j)
            all_sel=set(uniform)
            G=self.grid_size
            edges=set()
            for i in range(G):
                edges.update({i,(G-1)*G+i, i*G, i*G+(G-1)})
            all_sel.update(list(edges)[:int(k*0.2)])
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
            mu_y = yb.mean(); sd_y = yb.std().clamp_min(1e-6)
            if not self._stats_initialized:
                self._x_mu, self._x_std = mu, sd
                self._y_mu, self._y_std = float(mu_y), float(sd_y)
                self._stats_initialized = True
            else:
                self._x_mu = (1-momentum)*self._x_mu + momentum*mu
                self._x_std= (1-momentum)*self._x_std+ momentum*sd
                self._y_mu = float((1-momentum)*self._y_mu + momentum*mu_y)
                self._y_std= float((1-momentum)*self._y_std+ momentum*sd_y)

    def _norm_x(self, xb): return (xb - self._x_mu) / (self._x_std + 1e-8)
    def _norm_y(self, y):  return (y - self._y_mu) / (self._y_std + 1e-8)
    def _denorm_y(self, y): return y * self._y_std + self._y_mu

    def _rel_l2_per_sample(self, pred, target):  # -> [B]
        d = (pred - target).view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        return (d.norm(dim=1) / (t.norm(dim=1) + 1e-8))

    def _gradients(self, field_flat_norm: torch.Tensor):
        """Gradientes aproximados (Sobel) sobre campo [B,N] en escala normalizada."""
        B = field_flat_norm.size(0)
        g = self.grid_size
        f = field_flat_norm.view(B, 1, g, g)
        gx = F.conv2d(f, self.sobel_x, padding=1)
        gy = F.conv2d(f, self.sobel_y, padding=1)
        return gx.view(B, -1), gy.view(B, -1)

    def _loss_weights(self):
        """
        Devuelve: (w_rel, w_grad, p_drop)
        - Annealing lineal desde epoch 0 → epochs:
          * p_drop: decae de p0 → 0
          * w_grad: decae de w0 → 0.05*w0
          * w_rel : decae de 0.05 → 0.02 (suave)
        """
        t = min(max(self._global_epoch / max(self.epochs, 1), 0.0), 1.0)
        p_drop = self.sensor_dropout_p0 * (1.0 - t)
        w_grad = self.grad_loss_weight0 * (0.05 + 0.95 * (1.0 - t))
        w_rel  = 0.05 - 0.03 * t
        return w_rel, w_grad, p_drop

    def _maybe_update_swa(self):
        start = int(self.swa_start_frac * self.epochs)
        if self._global_epoch < start:
            return
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self._swa_params is None:
            self._swa_params = [p.detach().clone() for p in params]
            self._swa_n = 1
        else:
            self._swa_n += 1
            with torch.no_grad():
                for swa, p in zip(self._swa_params, params):
                    swa.add_(p.detach().sub(swa), alpha=1.0 / self._swa_n)

    # ---------- train ----------
    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0

        steps_per_epoch = max(1, len(train_loader))

        w_rel, w_grad, p_drop = self._loss_weights()

        for batch in train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            xb = self._take_sensors(x)

            # --- Sensor dropout (annealed)
            if p_drop > 0.0:
                mask = (torch.rand_like(xb) > p_drop)
                xb = xb * mask

            self._update_stats(xb, y)
            xb = self._norm_x(xb)
            yt_n = self._norm_y(y.view(y.size(0), -1))  # [B,N]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred_n = self.model(xb, self.trunk).reshape(yt_n.shape[0], yt_n.shape[1])

                # --- Grad loss (Sobolev) en escala normalizada
                px, py = self._gradients(pred_n)
                tx, ty = self._gradients(yt_n)
                grad_loss = self.loss_fn(px, tx) + self.loss_fn(py, ty)

                rel_n  = self._rel_l2_per_sample(pred_n, yt_n).mean()
                loss   = self.loss_fn(pred_n, yt_n) + w_rel * rel_n + w_grad * grad_loss

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._epoch_float += 1.0 / steps_per_epoch
            self.scheduler.step(self._epoch_float)

            # accuracy en escala real (como tu implementación)
            with torch.no_grad():
                pred_r = self._denorm_y(pred_n)
                tgt_r  = y.view(y.size(0), -1)
                acc_b  = (1.0 - self._rel_l2_per_sample(pred_r, tgt_r)) * 100.0
                total_acc += acc_b.clamp(0.0, 100.0).sum().item()  # clamp solo para logging
                total_loss += loss.item() * x.size(0)
                total_n    += x.size(0)

        # EMA
        with torch.no_grad():
            if self._ema is None:
                self._ema = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
            else:
                for ema, p in zip(self._ema, [q for q in self.model.parameters() if q.requires_grad]):
                    ema.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

        # SWA update al final de la época
        self._maybe_update_swa()
        self._global_epoch += 1

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
            'info': 'Loss=MSE+w_rel*RelL2+w_grad*GradLoss (annealed); Accuracy=100*(1-RelL2) real scale; EMA+SWA eval'
        }

    @torch.no_grad()
    def _with_weights(self, weight_list, fn):
        params = [p for p in self.model.parameters() if p.requires_grad]
        backup = [p.detach().clone() for p in params]
        for p, w in zip(params, weight_list):
            p.data.copy_(w)
        out = fn()
        for p, b in zip(params, backup):
            p.data.copy_(b)
        return out

    @torch.no_grad()
    def evaluate(self, data_loader):
        # preferimos SWA si existe; si no, EMA; si no, pesos actuales
        def _eval_core():
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total_n = 0
            w_rel, w_grad, _ = self._loss_weights()  # para logging consistente
            for batch in data_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                xb = self._norm_x(self._take_sensors(x))
                yt_n = self._norm_y(y.view(y.size(0), -1))
                pred_n = self.model(xb, self.trunk).reshape(yt_n.shape[0], yt_n.shape[1])

                px, py = self._gradients(pred_n)
                tx, ty = self._gradients(yt_n)
                grad_loss = self.loss_fn(px, tx) + self.loss_fn(py, ty)

                loss = self.loss_fn(pred_n, yt_n) + w_rel * self._rel_l2_per_sample(pred_n, yt_n).mean() \
                       + w_grad * grad_loss

                pred_r = self._denorm_y(pred_n)
                tgt_r  = y.view(y.size(0), -1)
                acc_b  = (1.0 - self._rel_l2_per_sample(pred_r, tgt_r)) * 100.0

                total_loss += loss.item() * x.size(0)
                total_acc  += acc_b.clamp(0.0, 100.0).sum().item()
                total_n    += x.size(0)
            return total_loss / total_n, total_acc / total_n

        # 1) SWA si está lista
        if self._swa_params is not None and self._swa_n > 0:
            return self._with_weights(self._swa_params, _eval_core)
        # 2) EMA si existe
        if self._ema is not None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            return self._with_weights(self._ema, _eval_core)
        # 3) pesos actuales
        return _eval_core()

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        x = batch["x"].to(self.device)
        xb = self._norm_x(self._take_sensors(x))
        pred_n = self.model(xb, self.trunk)
        B = xb.size(0)
        pred_r = self._denorm_y(pred_n).reshape(B, 1, self.grid_size, self.grid_size)
        return pred_r

    def get_model_info(self):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "name": f"{self.name}_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "type": "DeepONet Fourier + FiLM (stable) + Sobolev + SWA",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "sensor_strategy": self.sensor_strategy,
                "fourier_m": self.fourier_m
            },
            "parameters": params,
            "optimizer": f"AdamW(lr={self.lr})",
            "accuracy_method": "100*(1-relative_L2_error) on real scale"
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
