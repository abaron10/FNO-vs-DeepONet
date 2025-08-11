import math
import numpy as np
import torch
import torch.nn.functional as F
from .base_operator import BaseOperator

# ============================================================
# Multi-Scale Random Fourier Features + Fourier-Dropout
# ============================================================
class MultiScaleFourierFeatures(torch.nn.Module):
    """
    Genera [sin(xB_i), cos(xB_i)] para varias escalas y concatena.
    - in_dim: 2 (x,y)
    - m_per_scale: nº de bases por escala (por banco)
    - scales: lista de factores de frecuencia, p.ej. (pi, 4pi, 16pi)
    - dropout_p: probabilidad de apagar componentes (sin/cos) por batch (annealed)
    """
    def __init__(self, in_dim=2, m_per_scale=64, scales=(math.pi, 4*math.pi, 16*math.pi)):
        super().__init__()
        self.in_dim = in_dim
        self.m_per_scale = m_per_scale
        self.scales = tuple(scales)
        self.dropout_p = 0.0  # se ajusta externamente (annealing)
        # Matrices B por escala (amplificadas por 'scale')
        self.B_list = torch.nn.ParameterList()
        for s in self.scales:
            B = torch.randn(in_dim, m_per_scale) * s
            param = torch.nn.Parameter(B, requires_grad=False)  # fijo (random features)
            self.B_list.append(param)
        # Dimensión de salida total
        self.out_dim = 2 * m_per_scale * len(self.scales)

    def set_dropout_p(self, p: float):
        self.dropout_p = float(max(0.0, min(1.0, p)))

    def forward(self, x):  # x: [N, 2]
        feats = []
        for B in self.B_list:
            xb = x @ B  # [N, m]
            feats.append(torch.sin(xb))
            feats.append(torch.cos(xb))
        Fcat = torch.cat(feats, dim=-1)  # [N, 2*m_per_scale*#scales]
        if self.training and self.dropout_p > 0.0:
            # Apagamos canales completos (sin+cos comparten máscara a nivel de columna)
            N, D = Fcat.shape
            with torch.no_grad():
                mask = (torch.rand(1, D, device=Fcat.device) > self.dropout_p).float()
            Fcat = Fcat * mask
        return Fcat

# ============================================================
# Bloque residual MLP
# ============================================================
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

# ============================================================
# DeepONet (Branch-Trunk con FiLM estable y multi-escala)
# ============================================================
class DeepONet(torch.nn.Module):
    """
    Forward: (branch_in[B,S], trunk_in[N,2]) -> [B,N]
    """
    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 hidden_size: int = 320, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 m_per_scale: int = 64, ff_scales=(math.pi, 4*math.pi, 16*math.pi)):
        super().__init__()
        self.hidden = hidden_size
        self.num_layers = num_layers

        # Multi-escala en el tronco
        self.ff = MultiScaleFourierFeatures(trunk_input_size, m_per_scale, ff_scales)

        # Branch: embedding y parámetros FiLM (γ, β)
        b_layers = [torch.nn.Linear(branch_input_size, hidden_size)]
        for _ in range(num_layers - 1):
            b_layers.append(ResBlock(hidden_size, p=dropout, act=activation))
        self.branch_net = torch.nn.Sequential(*b_layers)
        self.to_gamma = torch.nn.Linear(hidden_size, num_layers * hidden_size)
        self.to_beta  = torch.nn.Linear(hidden_size, num_layers * hidden_size)

        # Trunk MLP
        t_in = self.ff.out_dim
        self.trunk_in = torch.nn.Linear(t_in, hidden_size)
        self.trunk_blocks = torch.nn.ModuleList(
            [ResBlock(hidden_size, p=dropout, act=activation) for _ in range(num_layers)]
        )

        # Proyecciones
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

    def set_fourier_dropout(self, p: float):
        self.ff.set_dropout_p(p)

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        # Branch
        B = self.branch_net(branch_in)  # [B,H]
        gammas = self.to_gamma(B).view(B.size(0), self.num_layers, self.hidden)  # [B,L,H]
        betas  = self.to_beta(B).view(B.size(0), self.num_layers, self.hidden)   # [B,L,H]
        gammas = 1.0 + self.gamma_scale * torch.tanh(gammas)  # [0.5, 1.5]
        betas  = self.beta_scale * torch.tanh(betas)          # [-0.1, 0.1]

        # Trunk
        T = self.trunk_in(self.ff(trunk_in))  # [N,H]

        # Capas + FiLM (sin residual extra para estabilidad)
        for l, blk in enumerate(self.trunk_blocks):
            T = blk(T)                             # [N,H]
            g = gammas[:, l, :].unsqueeze(1)      # [B,1,H]
            b = betas[:, l, :].unsqueeze(1)       # [B,1,H]
            T = g * T.unsqueeze(0) + b            # [B,N,H]

        # Proyecciones y combinación
        b_proj = self.branch_out(B).unsqueeze(1)            # [B,1,H]
        t_proj = self.trunk_out(T)                          # [B,N,H]
        dot = ((b_proj * t_proj).sum(-1)) * self.dot_scale  # [B,N]
        bias_x = 0.1 * self.bias_trunk(T).squeeze(-1)       # [B,N]
        out = dot + bias_x + self.bias
        return out.contiguous()                             # [B,N]

# ============================================================
# Operador
# ============================================================
class DeepONetOperator(BaseOperator):
    """
    - Normalización online (loss), accuracy en escala real: 100*(1 - relative_L2_error).
    - AMP torch.amp + CosineAnnealingWarmRestarts + EMA + SWA.
    - Sensor-Dropout (annealed) + Fourier-Dropout (annealed).
    - Pérdida Sobolev (gradientes) + Annealing de pesos.
    - Ponderación espacial de borde (más peso en una franja).
    """
    def __init__(self, device: torch.device, name: str = "", grid_size: int = 64,
                 n_sensors: int = 576, hidden_size: int = 320, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 lr: float = 2.5e-4, epochs: int = 1400, weight_decay: float = 1.5e-4,
                 step_size: int = 100, gamma: float = 0.5,
                 sensor_strategy: str = 'chebyshev', normalize_sensors: bool = True,
                 m_per_scale: int = 64, ff_scales=(math.pi, 4*math.pi, 16*math.pi),
                 # annealing/training knobs
                 sensor_dropout_p0: float = 0.10,
                 fourier_dropout_p0: float = 0.30,
                 w_grad0: float = 0.12, w_grad_end: float = 0.02,
                 w_rel0: float = 0.05,  w_rel_end: float = 0.01,
                 swa_start_frac: float = 0.5,
                 boundary_band: int = 3, boundary_boost: float = 1.5):
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

        self.m_per_scale = m_per_scale
        self.ff_scales = tuple(ff_scales)

        self.sensor_dropout_p0 = sensor_dropout_p0
        self.fourier_dropout_p0 = fourier_dropout_p0
        self.w_grad0 = w_grad0
        self.w_grad_end = w_grad_end
        self.w_rel0 = w_rel0
        self.w_rel_end = w_rel_end

        self.swa_start_frac = swa_start_frac
        self.boundary_band = boundary_band
        self.boundary_boost = boundary_boost

        # estado entrenamiento
        self._ema = None
        self.ema_decay = 0.995
        self.use_amp = (device.type == "cuda")
        self._epoch_float = 0.0
        self._stats_initialized = False
        self._global_epoch = 0
        self._swa_params = None
        self._swa_n = 0

        # Kernels Sobel (tensores atados al device)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(self.device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(self.device)

    # -------------------- setup --------------------
    def setup(self, data_info):
        self._setup_sensors_and_coords(data_info)

        in_branch = len(self.sensor_idx)
        self.model = DeepONet(
            in_branch, trunk_input_size=2,
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            activation=self.activation, dropout=self.dropout,
            m_per_scale=self.m_per_scale, ff_scales=self.ff_scales
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.loss_fn = torch.nn.MSELoss(reduction='none')  # para ponderación espacial

        # stats normalización
        self._x_mu = torch.zeros(1, in_branch, device=self.device)
        self._x_std = torch.ones(1, in_branch, device=self.device)
        self._y_mu = 0.0
        self._y_std = 1.0

        # Máscara espacial de borde (franja 'boundary_band')
        g = self.grid_size
        W2d = torch.ones(1, 1, g, g, device=self.device)
        if self.boundary_band > 0:
            k = self.boundary_band
            W2d[:, :, k:-k, k:-k] = 1.0  # interior
            # borde reforzado:
            W2d *= 1.0
            W2d[:, :, :k, :] *= self.boundary_boost
            W2d[:, :, -k:, :] *= self.boundary_boost
            W2d[:, :, :, :k] *= self.boundary_boost
            W2d[:, :, :, -k:] *= self.boundary_boost
        self.boundary_w2d = W2d              # [1,1,H,W]
        self.boundary_wflat = W2d.view(1, -1)  # [1,N]

    # -------------------- sensores/coords --------------------
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

    # -------------------- utilidades --------------------
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
        """Gradientes aprox (Sobel) sobre campo [B,N] en escala normalizada."""
        B = field_flat_norm.size(0)
        g = self.grid_size
        f = field_flat_norm.view(B, 1, g, g)
        gx = F.conv2d(f, self.sobel_x, padding=1)
        gy = F.conv2d(f, self.sobel_y, padding=1)
        return gx.view(B, -1), gy.view(B, -1)

    def _weighted_mse(self, pred, target, w_flat):
        # pred/target: [B,N], w_flat: [1,N]
        diff2 = (pred - target) ** 2
        w = w_flat.to(pred.dtype)
        return (diff2 * w).mean()

    def _loss_weights(self):
        """
        Annealing lineal 0→1 sobre épocas:
          - p_sensor: p0 → 0
          - p_fourier: p0 → 0
          - w_grad: w0 → w_end
          - w_rel:  w0 → w_end
        """
        t = min(max(self._global_epoch / max(self.epochs, 1), 0.0), 1.0)
        p_sensor = self.sensor_dropout_p0 * (1.0 - t)
        p_fourier = self.fourier_dropout_p0 * (1.0 - t)
        w_grad = self.w_grad0 + (self.w_grad_end - self.w_grad0) * t
        w_rel  = self.w_rel0 + (self.w_rel_end  - self.w_rel0)  * t
        return w_rel, w_grad, p_sensor, p_fourier

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

    # -------------------- training --------------------
    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0

        steps_per_epoch = max(1, len(train_loader))
        w_rel, w_grad, p_sensor, p_fourier = self._loss_weights()
        # Actualizamos dropout del módulo de Fourier
        self.model.set_fourier_dropout(p_fourier)

        for batch in train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            xb = self._take_sensors(x)
            # Sensor-dropout (annealed)
            if p_sensor > 0.0:
                mask = (torch.rand_like(xb) > p_sensor)
                xb = xb * mask

            self._update_stats(xb, y)
            xb = self._norm_x(xb)
            yt_n = self._norm_y(y.view(y.size(0), -1))  # [B,N]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred_n = self.model(xb, self.trunk).reshape(yt_n.shape[0], yt_n.shape[1])

                # Sobolev (grad) loss ponderada espacialmente
                px, py = self._gradients(pred_n)
                tx, ty = self._gradients(yt_n)
                # Reescalar máscara a [B,N]
                wflat = self.boundary_wflat  # [1,N]
                grad_loss = self._weighted_mse(px, tx, wflat) + self._weighted_mse(py, ty, wflat)

                # MSE ponderada espacialmente
                mse_w = self._weighted_mse(pred_n, yt_n, wflat)

                rel_n  = self._rel_l2_per_sample(pred_n, yt_n).mean()
                loss   = mse_w + w_rel * rel_n + w_grad * grad_loss

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._epoch_float += 1.0 / steps_per_epoch
            self.scheduler.step(self._epoch_float)

            # accuracy en escala REAL
            with torch.no_grad():
                pred_r = self._denorm_y(pred_n)
                tgt_r  = y.view(y.size(0), -1)
                acc_b  = (1.0 - self._rel_l2_per_sample(pred_r, tgt_r)) * 100.0
                total_acc += acc_b.clamp(0.0, 100.0).sum().item()
                total_loss += loss.item() * x.size(0)
                total_n    += x.size(0)

        # EMA
        with torch.no_grad():
            if self._ema is None:
                self._ema = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
            else:
                for ema, p in zip(self._ema, [q for q in self.model.parameters() if q.requires_grad]):
                    ema.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

        # SWA al final de época
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
            'info': 'MultiScale RFF + Fourier/Sensor Dropout (annealed), Sobolev+Boundary weights, EMA+SWA'
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
        def _eval_core():
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total_n = 0
            w_rel, w_grad, _, _ = self._loss_weights()
            self.model.set_fourier_dropout(0.0)  # eval sin dropout de Fourier
            for batch in data_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                xb = self._norm_x(self._take_sensors(x))
                yt_n = self._norm_y(y.view(y.size(0), -1))
                pred_n = self.model(xb, self.trunk).reshape(yt_n.shape[0], yt_n.shape[1])

                px, py = self._gradients(pred_n)
                tx, ty = self._gradients(yt_n)
                wflat = self.boundary_wflat
                grad_loss = self._weighted_mse(px, tx, wflat) + self._weighted_mse(py, ty, wflat)

                mse_w = self._weighted_mse(pred_n, yt_n, wflat)
                loss = mse_w + w_rel * self._rel_l2_per_sample(pred_n, yt_n).mean() + w_grad * grad_loss

                pred_r = self._denorm_y(pred_n)
                tgt_r  = y.view(y.size(0), -1)
                acc_b  = (1.0 - self._rel_l2_per_sample(pred_r, tgt_r)) * 100.0

                total_loss += loss.item() * x.size(0)
                total_acc  += acc_b.clamp(0.0, 100.0).sum().item()
                total_n    += x.size(0)
            return total_loss / total_n, total_acc / total_n

        # Prioridad: SWA > EMA > pesos actuales
        if self._swa_params is not None and self._swa_n > 0:
            return self._with_weights(self._swa_params, _eval_core)
        if self._ema is not None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            return self._with_weights(self._ema, _eval_core)
        return _eval_core()

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        x = batch["x"].to(self.device)
        xb = self._norm_x(self._take_sensors(x))
        self.model.set_fourier_dropout(0.0)
        pred_n = self.model(xb, self.trunk)
        B = xb.size(0)
        pred_r = self._denorm_y(pred_n).reshape(B, 1, self.grid_size, self.grid_size)
        return pred_r

    def get_model_info(self):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "name": f"{self.name}_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "type": "DeepONet (FiLM) + MultiScale RFF + Sobolev + SWA",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "sensor_strategy": self.sensor_strategy,
                "m_per_scale": self.m_per_scale,
                "ff_scales": self.ff_scales
            },
            "parameters": params,
            "optimizer": f"AdamW(lr={self.lr})",
            "accuracy_method": "100*(1-relative_L2_error) on real scale"
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
