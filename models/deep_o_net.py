import math
import numpy as np
import torch
import torch.nn.functional as F
from .base_operator import BaseOperator

# ============================================================
# Multi-Scale Random Fourier Features (+ opcional dropout)
# ============================================================
class MultiScaleFourierFeatures(torch.nn.Module):
    """
    Genera [sin(xB), cos(xB)] para varias escalas y concatena.
    - in_dim: dimensión de entrada (2 para [x,y])
    - m_per_scale: nº de bases por escala
    - scales: lista de factores de frecuencia, p.ej. (pi, 4pi, 16pi)
    """
    def __init__(self, in_dim=2, m_per_scale=64, scales=(math.pi, 4*math.pi, 16*math.pi)):
        super().__init__()
        self.in_dim = in_dim
        self.m_per_scale = m_per_scale
        self.scales = tuple(scales)
        # Pesos fijos (random features)
        self.B_list = torch.nn.ParameterList()
        for s in self.scales:
            B = torch.randn(in_dim, m_per_scale) * s
            self.B_list.append(torch.nn.Parameter(B, requires_grad=False))
        self.out_dim = 2 * m_per_scale * len(self.scales)
        self.dropout_p = 0.0  # se controla desde fuera (annealing)

    def set_dropout_p(self, p: float):
        self.dropout_p = float(max(0.0, min(1.0, p)))

    def forward(self, x):  # [N, in_dim]
        feats = []
        for B in self.B_list:
            xb = x @ B
            feats.append(torch.sin(xb))
            feats.append(torch.cos(xb))
        Fcat = torch.cat(feats, dim=-1)  # [N, out_dim]
        if self.training and self.dropout_p > 0.0:
            mask = (torch.rand(1, Fcat.shape[1], device=Fcat.device) > self.dropout_p).float()
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

    def forward(self, x):
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
    Forward: (branch_in[B,D_branch], trunk_in[N,2]) -> [B,N]
    """
    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 hidden_size: int = 384, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 m_per_scale: int = 64, ff_scales=(math.pi, 4*math.pi, 16*math.pi)):
        super().__init__()
        self.hidden = hidden_size
        self.num_layers = num_layers

        # Tronco multi-escala
        self.ff = MultiScaleFourierFeatures(trunk_input_size, m_per_scale, ff_scales)

        # Branch: integra características ya pre-agregadas (RFF-integral)
        b_layers = [torch.nn.Linear(branch_input_size, hidden_size)]
        for _ in range(num_layers - 1):
            b_layers.append(ResBlock(hidden_size, p=dropout, act=activation))
        self.branch_net = torch.nn.Sequential(*b_layers)
        self.to_gamma = torch.nn.Linear(hidden_size, num_layers * hidden_size)
        self.to_beta  = torch.nn.Linear(hidden_size, num_layers * hidden_size)

        # Trunk MLP
        self.trunk_in = torch.nn.Linear(self.ff.out_dim, hidden_size)
        self.trunk_blocks = torch.nn.ModuleList(
            [ResBlock(hidden_size, p=dropout, act=activation) for _ in range(num_layers)]
        )

        # Proyecciones y combinación
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
        gammas = 1.0 + self.gamma_scale * torch.tanh(gammas)
        betas  = self.beta_scale * torch.tanh(betas)

        # Trunk
        T = self.trunk_in(self.ff(trunk_in))  # [N,H]

        for l, blk in enumerate(self.trunk_blocks):
            T = blk(T)                             # [N,H]
            g = gammas[:, l, :].unsqueeze(1)      # [B,1,H]
            b = betas[:, l, :].unsqueeze(1)       # [B,1,H]
            T = g * T.unsqueeze(0) + b            # [B,N,H]

        b_proj = self.branch_out(B).unsqueeze(1)            # [B,1,H]
        t_proj = self.trunk_out(T)                          # [B,N,H]
        dot = ((b_proj * t_proj).sum(-1)) * self.dot_scale  # [B,N]
        bias_x = 0.1 * self.bias_trunk(T).squeeze(-1)       # [B,N]
        return (dot + bias_x + self.bias).contiguous()      # [B,N]

# ============================================================
# Operador DeepONet
# ============================================================
class DeepONetOperator(BaseOperator):
    """
    - Branch-integral (RFF sobre coords de sensores) + Tronco MS-RFF + FiLM.
    - Sensor/Fourier Dropout con annealing.
    - Sobolev loss con pesos edge-aware (|∇κ|) + refuerzo de borde.
    - Multi-res loss (32, 16) con annealing.
    - EMA + SWA.  Accuracy: 100*(1-RelL2) en escala real.
    - MC Dropout/TTA en evaluación (K pasadas con dropout activo).
    """
    def __init__(self, device: torch.device, name: str = "", grid_size: int = 64,
                 n_sensors: int = 576, hidden_size: int = 384, num_layers: int = 4,
                 activation: str = 'gelu', dropout: float = 0.05,
                 lr: float = 2.0e-4, epochs: int = 1600, weight_decay: float = 1.5e-4,
                 sensor_strategy: str = 'random', normalize_sensors: bool = True,
                 # RFF (trunk & branch)
                 m_per_scale: int = 64, ff_scales=(math.pi, 4*math.pi, 16*math.pi),
                 branch_m_per_scale: int = 64, branch_scales=(math.pi, 4*math.pi, 16*math.pi),
                 # annealing
                 sensor_dropout_p0: float = 0.10,
                 fourier_dropout_p0: float = 0.30,
                 w_grad0: float = 0.12, w_grad_end: float = 0.02,
                 w_rel0: float = 0.05,  w_rel_end: float = 0.01,
                 # edge/border weights
                 boundary_band: int = 3, boundary_boost: float = 1.6,
                 edge_boost_alpha: float = 1.0,
                 # SWA/EMA
                 swa_start_frac: float = 0.5,
                 # MC-TTA
                 mc_samples_eval: int = 16, mc_fourier_p: float = 0.2, mc_sensor_p: float = 0.05):
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
        self.sensor_strategy = sensor_strategy
        self.normalize_sensors = normalize_sensors

        self.m_per_scale = m_per_scale
        self.ff_scales = tuple(ff_scales)
        self.branch_m_per_scale = branch_m_per_scale
        self.branch_scales = tuple(branch_scales)

        self.sensor_dropout_p0 = sensor_dropout_p0
        self.fourier_dropout_p0 = fourier_dropout_p0
        self.w_grad0, self.w_grad_end = w_grad0, w_grad_end
        self.w_rel0, self.w_rel_end   = w_rel0, w_rel_end

        self.boundary_band = boundary_band
        self.boundary_boost = boundary_boost
        self.edge_boost_alpha = edge_boost_alpha

        self.swa_start_frac = swa_start_frac
        self.mc_samples_eval = mc_samples_eval
        self.mc_fourier_p = mc_fourier_p
        self.mc_sensor_p = mc_sensor_p

        # estado entrenamiento
        self._ema = None
        self.ema_decay = 0.995
        self.use_amp = (device.type == "cuda")
        self._epoch_float = 0.0
        self._stats_initialized = False
        self._global_epoch = 0
        self._swa_params = None
        self._swa_n = 0

        # Kernels Sobel en device
        sobel_x = torch.tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(self.device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(self.device)

    # -------------------- setup --------------------
    def setup(self, data_info):
        self._setup_sensors_and_coords(data_info)

        # Base de RFF en coordenadas de SENSORES (para el branch-integral)
        sensor_coords = self.trunk[self.sensor_idx]  # [S,2]
        self.branch_ff = MultiScaleFourierFeatures(2, self.branch_m_per_scale, self.branch_scales).to(self.device)
        self.branch_ff.eval()
        with torch.no_grad():
            self.branch_basis = self.branch_ff(sensor_coords)  # [S, D_branch]
        self.branch_dim = self.branch_basis.shape[1]

        # Modelo
        self.model = DeepONet(
            branch_input_size=self.branch_dim, trunk_input_size=2,
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            activation=self.activation, dropout=self.dropout,
            m_per_scale=self.m_per_scale, ff_scales=self.ff_scales
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.loss_fn = torch.nn.MSELoss(reduction='none')  # para pesos espaciales

        # stats normalización (branch agregado)
        self._x_mu = torch.zeros(1, self.branch_dim, device=self.device)
        self._x_std = torch.ones(1, self.branch_dim, device=self.device)
        self._y_mu = 0.0
        self._y_std = 1.0

        # Máscara de borde fija
        g = self.grid_size
        W2d = torch.ones(1, 1, g, g, device=self.device)
        if self.boundary_band > 0:
            k = self.boundary_band
            W2d[:, :, :k, :] *= self.boundary_boost
            W2d[:, :, -k:, :] *= self.boundary_boost
            W2d[:, :, :, :k] *= self.boundary_boost
            W2d[:, :, :, -k:] *= self.boundary_boost
        self.boundary_w2d = W2d
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
            cheb = np.cos((2*np.arange(n)+1)*np.pi/(2*n)); cheb = (cheb + 1) / 2
            idx=[]
            for i in range(n):
                for j in range(n):
                    x = int(cheb[i]*(self.grid_size-1)); y = int(cheb[j]*(self.grid_size-1))
                    idx.append(x*self.grid_size + y)
            self.sensor_idx = np.array(idx[:k])
        elif self.sensor_strategy == 'adaptive':
            rng = np.random.default_rng(42)
            base = int(k*0.6)
            step = max(1, int(np.sqrt((self.grid_size**2)/base)))
            uniform=[]
            for i in range(0, self.grid_size, step):
                for j in range(0, self.grid_size, step):
                    if len(uniform) < base: uniform.append(i*self.grid_size + j)
            all_sel=set(uniform); G=self.grid_size; edges=set()
            for i in range(G): edges.update({i,(G-1)*G+i, i*G, i*G+(G-1)})
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

        # Coordenadas de toda la malla para el tronco
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
        return x.view(B, -1)[:, self.sensor_idx]  # [B,S]

    def _aggregate_branch(self, x_sensors: torch.Tensor) -> torch.Tensor:
        # x_sensors: [B,S], self.branch_basis: [S,D]  -> [B,D]
        return x_sensors @ self.branch_basis

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

    def _rel_l2_per_sample(self, pred, target):
        d = (pred - target).view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        return (d.norm(dim=1) / (t.norm(dim=1) + 1e-8))

    def _gradients(self, field_flat_norm: torch.Tensor):
        B = field_flat_norm.size(0); g = self.grid_size
        f = field_flat_norm.view(B, 1, g, g)
        gx = F.conv2d(f, self.sobel_x, padding=1)
        gy = F.conv2d(f, self.sobel_y, padding=1)
        return gx.view(B, -1), gy.view(B, -1)

    def _weighted_mse(self, pred, target, w_flat):
        diff2 = (pred - target) ** 2
        return (diff2 * w_flat.to(pred.dtype)).mean()

    def _edge_weights(self, x_field: torch.Tensor):
        """x_field: [B,1,H,W] -> edge map |∇κ| normalizado -> [B,N]"""
        gx = F.conv2d(x_field, self.sobel_x, padding=1)
        gy = F.conv2d(x_field, self.sobel_y, padding=1)
        mag = (gx.abs() + gy.abs())  # [B,1,H,W]
        mag = mag / (mag.amax(dim=(-1, -2, -3), keepdim=True) + 1e-8)
        return (1.0 + self.edge_boost_alpha * mag).view(x_field.size(0), -1)  # [B,N]

    def _loss_weights(self):
        t = min(max(self._global_epoch / max(self.epochs, 1), 0.0), 1.0)
        p_sensor = self.sensor_dropout_p0 * (1.0 - t)
        p_fourier = self.fourier_dropout_p0 * (1.0 - t)
        w_grad = self.w_grad0 + (self.w_grad_end - self.w_grad0) * t
        w_rel  = self.w_rel0  + (self.w_rel_end  - self.w_rel0)  * t
        # multi-res annealing (desaparece al final)
        lam32 = 0.3 * (1.0 - t)
        lam16 = 0.1 * (1.0 - t)
        return w_rel, w_grad, p_sensor, p_fourier, lam32, lam16

    def _maybe_update_swa(self):
        start = int(self.swa_start_frac * self.epochs)
        if self._global_epoch < start: return
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self._swa_params is None:
            self._swa_params = [p.detach().clone() for p in params]; self._swa_n = 1
        else:
            self._swa_n += 1
            with torch.no_grad():
                for swa, p in zip(self._swa_params, params):
                    swa.add_(p.detach().sub(swa), alpha=1.0 / self._swa_n)

    # -------------------- train --------------------
    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = total_acc = total_n = 0.0

        steps_per_epoch = max(1, len(train_loader))
        w_rel, w_grad, p_sensor, p_fourier, lam32, lam16 = self._loss_weights()
        self.model.set_fourier_dropout(p_fourier)

        for batch in train_loader:
            x = batch["x"].to(self.device)  # [B,1,H,W]
            y = batch["y"].to(self.device)

            x_sensors = self._take_sensors(x)  # [B,S]
            if p_sensor > 0.0:
                x_sensors = x_sensors * (torch.rand_like(x_sensors) > p_sensor)
            xb = self._aggregate_branch(x_sensors)  # [B,D_branch]

            self._update_stats(xb, y)
            xb_n = self._norm_x(xb)
            yt_n = self._norm_y(y.view(y.size(0), -1))

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred_n = self.model(xb_n, self.trunk).reshape_as(yt_n)

                # pesos espaciales: borde * edge(|∇κ|)
                edge_w = self._edge_weights(x)  # [B,N]
                w_flat = (self.boundary_wflat * edge_w).clamp_min(1.0)

                # Sobolev
                px, py = self._gradients(pred_n)
                tx, ty = self._gradients(yt_n)
                grad_loss = self._weighted_mse(px, tx, w_flat) + self._weighted_mse(py, ty, w_flat)

                # MSE ponderada
                mse_w = self._weighted_mse(pred_n, yt_n, w_flat)

                # Multi-res (no ponderada, estabiliza)
                B, g = pred_n.size(0), self.grid_size
                p2d = pred_n.view(B, 1, g, g); t2d = yt_n.view(B, 1, g, g)
                p32 = F.avg_pool2d(p2d, 2); t32 = F.avg_pool2d(t2d, 2)
                p16 = F.avg_pool2d(p2d, 4); t16 = F.avg_pool2d(t2d, 4)
                mr_loss = lam32 * F.mse_loss(p32, t32) + lam16 * F.mse_loss(p16, t16)

                rel_n = self._rel_l2_per_sample(pred_n, yt_n).mean()
                loss  = mse_w + w_rel * rel_n + w_grad * grad_loss + mr_loss

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._epoch_float += 1.0 / steps_per_epoch
            self.scheduler.step(self._epoch_float)

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
            'info': 'Integral-Branch RFF + MS-Trunk + Sobolev/Boundary/Edge + MultiRes + EMA+SWA'
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
    def _eval_core_pass(self, data_loader, use_dropout=False):
        """Una pasada de evaluación: si use_dropout=True, activa MC-dropout/TTA."""
        if use_dropout:
            self.model.train()
            self.model.set_fourier_dropout(self.mc_fourier_p)
        else:
            self.model.eval()
            self.model.set_fourier_dropout(0.0)

        total_loss = total_acc = total_n = 0.0
        w_rel, w_grad, _, _, lam32, lam16 = self._loss_weights()

        for batch in data_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            x_sensors = self._take_sensors(x)
            if use_dropout and self.mc_sensor_p > 0.0:
                x_sensors = x_sensors * (torch.rand_like(x_sensors) > self.mc_sensor_p)
            xb = self._aggregate_branch(x_sensors)
            xb_n = self._norm_x(xb)
            yt_n = self._norm_y(y.view(y.size(0), -1))

            pred_n = self.model(xb_n, self.trunk).reshape_as(yt_n)

            # pérdidas (mismas que train para coherencia)
            edge_w = self._edge_weights(x)
            wflat  = (self.boundary_wflat * edge_w).clamp_min(1.0)
            px, py = self._gradients(pred_n); tx, ty = self._gradients(yt_n)
            grad_loss = self._weighted_mse(px, tx, wflat) + self._weighted_mse(py, ty, wflat)
            mse_w = self._weighted_mse(pred_n, yt_n, wflat)

            B, g = pred_n.size(0), self.grid_size
            p2d = pred_n.view(B, 1, g, g); t2d = yt_n.view(B, 1, g, g)
            p32 = F.avg_pool2d(p2d, 2); t32 = F.avg_pool2d(t2d, 2)
            p16 = F.avg_pool2d(p2d, 4); t16 = F.avg_pool2d(t2d, 4)
            mr_loss = lam32 * F.mse_loss(p32, t32) + lam16 * F.mse_loss(p16, t16)

            loss = mse_w + w_rel * self._rel_l2_per_sample(pred_n, yt_n).mean() + w_grad * grad_loss + mr_loss

            pred_r = self._denorm_y(pred_n)
            tgt_r  = y.view(y.size(0), -1)
            acc_b  = (1.0 - self._rel_l2_per_sample(pred_r, tgt_r)) * 100.0

            total_loss += loss.item() * x.size(0)
            total_acc  += acc_b.clamp(0.0, 100.0).sum().item()
            total_n    += x.size(0)

        return total_loss / total_n, total_acc / total_n

    @torch.no_grad()
    def evaluate(self, data_loader):
        # Preferimos SWA; si no, EMA; si no, pesos actuales
        def eval_with_weights(weights=None):
            if weights is not None:
                return self._with_weights(weights, lambda: self._eval_core_pass(data_loader, use_dropout=False))
            return self._eval_core_pass(data_loader, use_dropout=False)

        # 1) SWA
        if self._swa_params is not None and self._swa_n > 0:
            base_loss, base_acc = eval_with_weights(self._swa_params)
        # 2) EMA
        elif self._ema is not None:
            base_loss, base_acc = eval_with_weights(self._ema)
        else:
            base_loss, base_acc = eval_with_weights(None)

        # MC-TTA: promedio de K pasadas con dropout activo
        if self.mc_samples_eval and self.mc_samples_eval > 1:
            mc_losses, mc_accs = [], []
            for _ in range(self.mc_samples_eval):
                l, a = self._eval_core_pass(data_loader, use_dropout=True)
                mc_losses.append(l); mc_accs.append(a)
            # Promediamos con la pasada determinista también
            mc_losses.append(base_loss); mc_accs.append(base_acc)
            return float(np.mean(mc_losses)), float(np.mean(mc_accs))

        return base_loss, base_acc

    @torch.no_grad()
    def predict(self, batch):
        """Predicción con MC-TTA (promedio)"""
        K = max(1, self.mc_samples_eval)
        preds = []
        for k in range(K):
            use_dropout = (k < K - 1)  # última pasada determinista
            if use_dropout:
                self.model.train(); self.model.set_fourier_dropout(self.mc_fourier_p)
            else:
                self.model.eval();  self.model.set_fourier_dropout(0.0)

            x = batch["x"].to(self.device)
            x_sensors = self._take_sensors(x)
            if use_dropout and self.mc_sensor_p > 0.0:
                x_sensors = x_sensors * (torch.rand_like(x_sensors) > self.mc_sensor_p)
            xb = self._norm_x(self._aggregate_branch(x_sensors))
            pred_n = self.model(xb, self.trunk)
            preds.append(pred_n)

        pred_n_mean = torch.stack(preds, dim=0).mean(0)
        B = pred_n_mean.size(0)
        pred_r = self._denorm_y(pred_n_mean).reshape(B, 1, self.grid_size, self.grid_size)
        return pred_r

    def get_model_info(self):
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "name": f"{self.name}_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "type": "DeepONet (FiLM) + Integral-Branch RFF + MS Trunk + Sobolev + SWA + MC-TTA",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "sensor_strategy": self.sensor_strategy,
                "m_per_scale": self.m_per_scale,
                "ff_scales": self.ff_scales,
                "branch_m_per_scale": self.branch_m_per_scale,
                "branch_scales": self.branch_scales
            },
            "parameters": params,
            "optimizer": f"AdamW(lr={self.lr})",
            "accuracy_method": "100*(1-relative_L2_error) on real scale"
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
