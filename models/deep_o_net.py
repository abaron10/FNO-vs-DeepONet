import torch
import numpy as np
from .base_operator import BaseOperator

class DeepONet(torch.nn.Module):
    """Enhanced DeepONet with configurable architecture."""

    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 hidden_size: int = 128, num_layers: int = 4, 
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        self.branch_input_size = branch_input_size
        self.trunk_input_size = trunk_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.act_fn = torch.nn.ReLU()
        elif activation == 'gelu':
            self.act_fn = torch.nn.GELU()
        elif activation == 'mish':
            self.act_fn = torch.nn.Mish()
        else:
            self.act_fn = torch.nn.ReLU()
        
        # IMPROVEMENT 1: Asymmetric architecture - deeper branch, simpler trunk
        branch_depth = max(num_layers, 8)  # At least 8 layers for branch
        trunk_depth = max(4, num_layers // 2)  # Simpler trunk
        
        # ----- Enhanced Branch Network -----
        branch_layers = []
        
        # Initial layer with expansion
        branch_layers.extend([
            torch.nn.Linear(branch_input_size, hidden_size * 2),
            self.act_fn,
            torch.nn.BatchNorm1d(hidden_size * 2),
            torch.nn.Dropout(dropout)
        ])
        
        # Deep branch layers with residual connections
        current_size = hidden_size * 2
        for i in range(branch_depth - 2):
            # Gradual size reduction
            next_size = hidden_size * 2 if i < branch_depth // 2 else hidden_size
            branch_layers.extend([
                torch.nn.Linear(current_size, next_size),
                self.act_fn,
                torch.nn.BatchNorm1d(next_size)
            ])
            if dropout > 0 and i % 2 == 0:
                branch_layers.append(torch.nn.Dropout(dropout))
            current_size = next_size
                
        # Final branch layer
        branch_layers.append(torch.nn.Linear(current_size, hidden_size))
        self.branch_net = torch.nn.Sequential(*branch_layers)
        
        # ----- Enhanced Trunk Network with Fourier Features -----
        # IMPROVEMENT 2: Add Fourier features
        self.n_fourier = 16
        self.fourier_freqs = torch.nn.Parameter(
            torch.randn(self.n_fourier, trunk_input_size) * 10
        )
        
        trunk_input_enhanced = trunk_input_size + 2 * self.n_fourier  # sin + cos features
        
        trunk_layers = []
        trunk_layers.extend([
            torch.nn.Linear(trunk_input_enhanced, hidden_size),
            self.act_fn,
            torch.nn.LayerNorm(hidden_size)  # LayerNorm works better for trunk
        ])
        
        for _ in range(trunk_depth - 2):
            trunk_layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                self.act_fn,
                torch.nn.LayerNorm(hidden_size)
            ])
                
        trunk_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.trunk_net = torch.nn.Sequential(*trunk_layers)
        
        # IMPROVEMENT 3: Nonlinear decoder instead of just dot product
        self.use_nonlinear_decoder = True
        if self.use_nonlinear_decoder:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                self.act_fn,
                torch.nn.Linear(hidden_size, hidden_size // 2),
                self.act_fn,
                torch.nn.Linear(hidden_size // 2, 1)
            )
        
        # Bias term
        self.bias = torch.nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """IMPROVEMENT 4: Better initialization following recent papers"""
        for name, m in self.named_modules():
            if isinstance(m, torch.nn.Linear):
                # Special initialization for branch network (sensor processing)
                if 'branch' in name:
                    # Scale by sensor count for better gradient flow
                    scale = np.sqrt(2.0 / (m.in_features + m.out_features))
                    if self.branch_input_size > 1000:  # Many sensors
                        scale *= np.sqrt(1000 / self.branch_input_size)
                    torch.nn.init.normal_(m.weight, 0, scale)
                # Trunk network initialization
                elif 'trunk' in name:
                    torch.nn.init.xavier_normal_(m.weight, gain=0.5)
                # Decoder initialization
                elif 'decoder' in name:
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
                
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
        # Initialize Fourier frequencies with specific scale
        with torch.no_grad():
            self.fourier_freqs.data = torch.randn_like(self.fourier_freqs) * np.pi

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        """branch_in: [B, n_sensors]  trunk_in: [N, 2] returns [B, N]"""
        # Branch network
        b = self.branch_net(branch_in)  # [B, H]
        
        # Trunk network with Fourier features
        fourier_features = []
        x_proj = trunk_in @ self.fourier_freqs.T  # [N, n_fourier]
        fourier_features.append(torch.sin(x_proj))
        fourier_features.append(torch.cos(x_proj))
        trunk_in_enhanced = torch.cat([trunk_in] + fourier_features, dim=1)
        
        t = self.trunk_net(trunk_in_enhanced)  # [N, H]
        
        # IMPROVEMENT 5: Nonlinear decoder
        if self.use_nonlinear_decoder:
            # Efficient batched computation
            B, H = b.shape
            N, _ = t.shape
            
            # Expand and concatenate
            b_exp = b.unsqueeze(1).expand(B, N, H)  # [B, N, H]
            t_exp = t.unsqueeze(0).expand(B, N, H)  # [B, N, H]
            combined = torch.cat([b_exp, t_exp], dim=2)  # [B, N, 2H]
            
            # Apply decoder
            out = self.decoder(combined).squeeze(-1)  # [B, N]
        else:
            # Original dot product
            out = torch.sum(b.unsqueeze(1) * t.unsqueeze(0), dim=2)
        
        return out + self.bias


class DeepONetOperator(BaseOperator):
    """Enhanced DeepONet with FNO-style configuration options"""

    def __init__(self, device: torch.device, name: str = "", grid_size: int = 64, 
                 n_sensors: int = 1000, hidden_size: int = 128, num_layers: int = 4,
                 activation: str = 'relu', dropout: float = 0.0,
                 
                 # Training settings (FNO-style)
                 lr: float = 1e-3, step_size: int = 100, gamma: float = 0.5,
                 weight_decay: float = 1e-4, epochs: int = 500,
                 
                 # DeepONet specific
                 sensor_strategy: str = 'random',  # 'random', 'uniform', 'chebyshev', 'adaptive'
                 normalize_sensors: bool = True):
        
        super().__init__(device, grid_size)
        
        # Model configuration
        self.name = name
        self.n_sensors = n_sensors
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        # Training configuration
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        # DeepONet specific
        self.sensor_strategy = sensor_strategy
        self.normalize_sensors = normalize_sensors
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 75
        self.patience_counter = 0
        
        # INTERNAL: Enhanced data augmentation
        self._use_augmentation = True
        self._augmentation_level = 0.01
        
        # INTERNAL: Learning rate warmup
        self._use_warmup = True
        self._warmup_epochs = 10

    def setup(self, data_info):
        # Generate sensor locations and coordinates
        self._setup_sensors_and_coords(data_info)
        
        # Build model
        in_branch = len(self.sensor_idx)
        self.model = DeepONet(
            in_branch, 
            trunk_input_size=2, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            activation=self.activation,
            dropout=self.dropout
        ).to(self.device)
        
        # Optimizer with improved settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler instead of StepLR for smoother decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.step_size,
            T_mult=2,
            eta_min=self.lr * 0.01
        )
        
        self.loss_fn = torch.nn.MSELoss()

    def _setup_sensors_and_coords(self, data_info):
        """IMPROVEMENT 6: Enhanced adaptive sensor placement"""
        max_sensors = self.grid_size ** 2
        actual_n_sensors = min(self.n_sensors, max_sensors)
        
        if self.sensor_strategy == 'random':
            rng = np.random.default_rng(42)
            self.sensor_idx = rng.choice(max_sensors, size=actual_n_sensors, replace=False)
            
        elif self.sensor_strategy == 'uniform':
            step = max(1, max_sensors // actual_n_sensors)
            self.sensor_idx = np.arange(0, max_sensors, step)[:actual_n_sensors]
            
        elif self.sensor_strategy == 'chebyshev':
            # Chebyshev nodes in 2D
            n_per_dim = int(np.sqrt(actual_n_sensors))
            cheb_1d = np.cos((2*np.arange(n_per_dim) + 1) * np.pi / (2*n_per_dim))
            cheb_1d = (cheb_1d + 1) / 2  # Map to [0, 1]
            
            indices = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    x_idx = int(cheb_1d[i] * (self.grid_size - 1))
                    y_idx = int(cheb_1d[j] * (self.grid_size - 1))
                    indices.append(x_idx * self.grid_size + y_idx)
            self.sensor_idx = np.array(indices[:actual_n_sensors])
            
        elif self.sensor_strategy == 'adaptive':
            # ENHANCED adaptive strategy with physics-aware placement
            rng = np.random.default_rng(42)
            
            # 1. Start with coarse uniform grid (40% of sensors)
            uniform_count = int(actual_n_sensors * 0.4)
            step = max(2, int(np.sqrt(max_sensors / uniform_count)))
            uniform_indices = []
            for i in range(0, self.grid_size, step):
                for j in range(0, self.grid_size, step):
                    if len(uniform_indices) < uniform_count:
                        uniform_indices.append(i * self.grid_size + j)
            
            # 2. Add boundary sensors (20% of sensors)
            boundary_count = int(actual_n_sensors * 0.2)
            boundary_indices = []
            
            # Corners first (important for boundary conditions)
            corners = [0, self.grid_size-1, 
                      (self.grid_size-1)*self.grid_size, 
                      self.grid_size*self.grid_size-1]
            boundary_indices.extend(corners)
            
            # Then edges with higher density near corners
            for i in range(1, self.grid_size-1):
                # Weight function for edge sampling (higher near corners)
                weight = 1 / (1 + 0.1 * min(i, self.grid_size-1-i))
                if rng.random() < weight:
                    # Top and bottom
                    boundary_indices.extend([i, (self.grid_size-1)*self.grid_size + i])
                    # Left and right
                    boundary_indices.extend([i*self.grid_size, i*self.grid_size + (self.grid_size-1)])
            
            boundary_indices = list(set(boundary_indices))[:boundary_count]
            
            # 3. Add Chebyshev-like points in interior (20% of sensors)
            cheb_count = int(actual_n_sensors * 0.2)
            n_cheb = int(np.sqrt(cheb_count))
            cheb_indices = []
            for i in range(n_cheb):
                for j in range(n_cheb):
                    # Chebyshev points mapped to interior
                    x = 0.5 + 0.4 * np.cos((2*i+1)*np.pi/(2*n_cheb))
                    y = 0.5 + 0.4 * np.cos((2*j+1)*np.pi/(2*n_cheb))
                    idx = int(x*(self.grid_size-1))*self.grid_size + int(y*(self.grid_size-1))
                    cheb_indices.append(idx)
            
            # 4. Fill remaining with strategic random sampling
            all_selected = set(uniform_indices + boundary_indices + cheb_indices[:cheb_count])
            remaining = actual_n_sensors - len(all_selected)
            
            if remaining > 0:
                # Create importance map (higher importance near center and boundaries)
                importance_map = np.zeros(max_sensors)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        # Distance to center
                        dc = np.sqrt((i-self.grid_size/2)**2 + (j-self.grid_size/2)**2)
                        # Distance to nearest boundary
                        db = min(i, j, self.grid_size-1-i, self.grid_size-1-j)
                        # Combined importance
                        importance = np.exp(-dc/(self.grid_size/4)) + 0.5*np.exp(-db/5)
                        importance_map[i*self.grid_size + j] = importance
                
                # Sample based on importance
                available = list(set(range(max_sensors)) - all_selected)
                probs = importance_map[available]
                probs = probs / probs.sum()
                additional = rng.choice(available, size=remaining, replace=False, p=probs)
                all_selected.update(additional)
            
            self.sensor_idx = np.array(list(all_selected)[:actual_n_sensors])
        
        self.sensor_idx = np.sort(self.sensor_idx)
        
        # Create coordinate grid with enhanced normalization
        x = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, x, indexing='ij')
        coords = np.column_stack([X.flatten(), Y.flatten()])
        
        if self.normalize_sensors:
            # Enhanced normalization: zero mean, unit variance
            coords = (coords - 0.5) * 2  # Map to [-1, 1]
        
        self.trunk = torch.FloatTensor(coords).to(self.device)
        
        # Store for JSON serialization
        self.sensor_idx_list = self.sensor_idx.tolist()
        
        print(f"✓ Set up {len(self.sensor_idx)} sensors using '{self.sensor_strategy}' strategy")

    def _take_sensors(self, x: torch.Tensor) -> torch.Tensor:
        """Extract sensor values from input field"""
        B = x.shape[0]
        flat = x.view(B, -1)
        return flat[:, self.sensor_idx]

    def train_epoch(self, train_loader, val_loader=None):
        """Training epoch with enhanced techniques"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        
        # Track epoch number
        if not hasattr(self, 'epoch_num'):
            self.epoch_num = 0
        self.epoch_num += 1
        
        # Learning rate warmup
        if self._use_warmup and self.epoch_num <= self._warmup_epochs:
            warmup_factor = self.epoch_num / self._warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr * warmup_factor
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            # Enhanced data augmentation
            if self._use_augmentation and self.model.training:
                # Progressive augmentation
                aug_factor = min(1.0, self.epoch_num / 100)
                noise_scale = self._augmentation_level * aug_factor
                
                # Input augmentation
                if torch.rand(1).item() > 0.2:
                    noise = torch.randn_like(x) * noise_scale
                    x = x + noise
                
                # Slight output perturbation for regularization
                if torch.rand(1).item() > 0.5:
                    y_noise = torch.randn_like(y) * (noise_scale * 0.2)
                    y = y + y_noise
            
            # Extract sensor data
            branch_input = self._take_sensors(x)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(branch_input, self.trunk)
            
            # Reshape target
            B = y.shape[0]
            target = y.view(B, -1)
            
            # Compute loss with gradient penalty for stability
            loss = self.loss_fn(pred, target)
            
            # Optional: Add gradient penalty for stability
            if self.epoch_num > 50 and self.epoch_num % 10 == 0:
                grad_penalty = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_penalty += param.grad.norm()
                loss = loss + 1e-5 * grad_penalty
            
            # Calculate accuracy
            with torch.no_grad():
                diff = (pred - target).view(pred.size(0), -1)
                true = target.view(target.size(0), -1)
                rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8))
                sample_accuracy = (1 - rel_l2) * 100
                batch_accuracy = sample_accuracy.mean().item()
                total_accuracy += batch_accuracy * x.size(0)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping with adaptive threshold
            max_grad_norm = 1.0 if self.epoch_num < 50 else 0.5
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        avg_train_loss = total_loss / total_samples
        avg_train_accuracy = total_accuracy / total_samples
        
        # Validation
        val_loss = float('inf')
        val_accuracy = 0
        if val_loader is not None:
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': self.optimizer.param_groups[0]['lr'],
            'should_stop': self.patience_counter >= self.patience,
            'info': 'Accuracy computed as 100*(1-relative_L2_error) following Li et al.'
        }

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate model with FNO-style accuracy calculation"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        for batch in data_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            branch_input = self._take_sensors(x)
            pred = self.model(branch_input, self.trunk)
            
            B = y.shape[0]
            target = y.view(B, -1)
            
            loss = self.loss_fn(pred, target)
            
            # Calculate accuracy using Li et al. method
            diff = (pred - target).view(pred.size(0), -1)
            true = target.view(target.size(0), -1)
            rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8))
            sample_accuracy = (1 - rel_l2) * 100
            
            total_loss += loss.item() * pred.size(0)
            total_accuracy += sample_accuracy.sum().item()
            total_samples += pred.size(0)
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        return avg_loss, avg_accuracy

    @torch.no_grad()
    def predict(self, batch):
        """Make predictions"""
        self.model.eval()
        x = batch["x"].to(self.device)
        
        branch_input = self._take_sensors(x)
        pred = self.model(branch_input, self.trunk)
        
        B = branch_input.shape[0]
        return pred.view(B, 1, self.grid_size, self.grid_size)

    def get_model_info(self):
        """Get model information in FNO-style format"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "name": f"{self.name}_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "type": "Enhanced Branch-Trunk Neural Network",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "sensor_strategy": self.sensor_strategy,
                "features": "Nonlinear decoder, Fourier trunk, Asymmetric architecture"
            },
            "parameters": trainable_params,
            "total_parameters": total_params,
            "optimizer": f"AdamW(lr={self.lr})",
            "scheduler": "CosineAnnealingWarmRestarts",
            "accuracy_method": "Li et al. (100*(1-relative_L2_error))"
        }

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class DeepONetEnsembleOperator(DeepONetOperator):
    """Ensemble of DeepONet models for better accuracy"""
    
    def __init__(self, device, name="", grid_size=64, n_models=2, **kwargs):
        super().__init__(device, name, grid_size, **kwargs)
        self.n_models = n_models
        
    def setup(self, data_info):
        self._setup_sensors_and_coords(data_info)
        
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        for i in range(self.n_models):
            # Set different seed for each model
            torch.manual_seed(42 + i)
            
            # Build model with slight variations
            in_branch = len(self.sensor_idx)
            model = DeepONet(
                in_branch, 
                trunk_input_size=2, 
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                activation=self.activation,
                dropout=self.dropout * (1 + i * 0.1)  # Vary dropout
            ).to(self.device)
            
            self.models.append(model)
            
            # Create optimizer with slight LR variation
            lr_factor = 1 + (i - self.n_models/2) * 0.1
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.lr * lr_factor, 
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            self.optimizers.append(optimizer)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.step_size,
                T_mult=2,
                eta_min=self.lr * 0.01
            )
            self.schedulers.append(scheduler)
        
        self.loss_fn = torch.nn.MSELoss()
        
        torch.manual_seed(42)
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train all models in ensemble"""
        results = []
        
        for i, (model, optimizer, scheduler) in enumerate(zip(self.models, self.optimizers, self.schedulers)):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            
            # Vary augmentation per model
            if hasattr(self, '_use_augmentation'):
                self._augmentation_level = 0.01 * (1 + i * 0.2)
            
            result = super().train_epoch(train_loader, val_loader)
            results.append(result)
        
        avg_result = {
            'train_loss': sum(r['train_loss'] for r in results) / len(results),
            'train_accuracy': sum(r['train_accuracy'] for r in results) / len(results),
            'val_loss': sum(r['val_loss'] for r in results) / len(results),
            'val_accuracy': sum(r['val_accuracy'] for r in results) / len(results),
            'lr': results[0]['lr'],
            'should_stop': any(r['should_stop'] for r in results),
            'info': results[0].get('info', '')
        }
        
        return avg_result
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate ensemble on validation/test set"""
        all_losses = []
        all_accuracies = []
        
        for model in self.models:
            self.model = model
            loss, accuracy = super().evaluate(data_loader)
            all_losses.append(loss)
            all_accuracies.append(accuracy)
        
        return sum(all_losses) / len(all_losses), sum(all_accuracies) / len(all_accuracies)
    
    @torch.no_grad()
    def predict(self, batch):
        """Ensemble prediction by averaging"""
        predictions = []
        
        for model in self.models:
            self.model = model
            pred = super().predict(batch)
            predictions.append(pred)
        
        return torch.stack(predictions).mean(dim=0)
    
    def get_model_info(self):
        """Get ensemble model info"""
        total_params = sum(p.numel() for p in self.models[0].parameters())
        trainable_params = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        
        return {
            "name": self.name,
            "architecture": {
                "type": "Enhanced Branch-Trunk Ensemble",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "n_models": self.n_models,
                "sensor_strategy": self.sensor_strategy,
                "features": "Nonlinear decoder, Fourier trunk, Asymmetric architecture"
            },
            "parameters": trainable_params * self.n_models,
            "parameters_per_model": trainable_params,
            "optimizer": f"AdamW(lr={self.lr})",
            "scheduler": "CosineAnnealingWarmRestarts",
            "accuracy_method": "Li et al. (100*(1-relative_L2_error))"
        }
    
    def count_parameters(self):
        """Count total trainable parameters in ensemble"""
        params_per_model = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        return params_per_model * self.n_models