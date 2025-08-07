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
        
        # ----- Branch Network -----
        branch_layers = []
        
        # First layer with optional expansion
        branch_layers.append(torch.nn.Linear(branch_input_size, hidden_size))
        branch_layers.append(self.act_fn)
        if dropout > 0:
            branch_layers.append(torch.nn.Dropout(dropout))
            
        # Hidden layers
        for i in range(num_layers - 1):
            branch_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            branch_layers.append(self.act_fn)
            if dropout > 0 and i < num_layers - 2:  # No dropout before last layer
                branch_layers.append(torch.nn.Dropout(dropout))
                
        # Output layer (no activation)
        branch_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.branch_net = torch.nn.Sequential(*branch_layers)
        
        # ----- Trunk Network -----
        trunk_layers = []
        trunk_layers.append(torch.nn.Linear(trunk_input_size, hidden_size))
        trunk_layers.append(self.act_fn)
        if dropout > 0:
            trunk_layers.append(torch.nn.Dropout(dropout))
            
        for i in range(num_layers - 1):
            trunk_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(self.act_fn)
            if dropout > 0 and i < num_layers - 2:
                trunk_layers.append(torch.nn.Dropout(dropout))
                
        trunk_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.trunk_net = torch.nn.Sequential(*trunk_layers)
        
        # Bias term
        self.bias = torch.nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with improved scheme"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Use Kaiming initialization for ReLU/GELU
                if self.activation in ['relu', 'gelu']:
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    torch.nn.init.xavier_normal_(m.weight)
                
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
        # Special initialization for bias
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        """branch_in: [B, n_sensors]  trunk_in: [N, 2] returns [B, N]"""
        b = self.branch_net(branch_in)  # [B, H]
        t = self.trunk_net(trunk_in)    # [N, H]
        
        # Standard dot product (proven to work well)
        out = torch.sum(b.unsqueeze(1) * t.unsqueeze(0), dim=2)  # [B, N]
        
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
        self.patience = 100  # Increased patience
        self.patience_counter = 0
        
        # INTERNAL: Conservative data augmentation
        self._use_augmentation = True
        self._augmentation_level = 0.005  # Reduced from 0.01

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
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use StepLR as in original
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.step_size, 
            gamma=self.gamma
        )
        
        self.loss_fn = torch.nn.MSELoss()

    def _setup_sensors_and_coords(self, data_info):
        """Setup sensor locations and coordinate grid"""
        # Adaptive sensor count based on grid size and config
        max_sensors = self.grid_size ** 2
        actual_n_sensors = min(self.n_sensors, max_sensors)
        
        # Generate sensor indices based on strategy
        if self.sensor_strategy == 'random':
            rng = np.random.default_rng(42)  # Reproducible
            self.sensor_idx = rng.choice(max_sensors, size=actual_n_sensors, replace=False)
        elif self.sensor_strategy == 'uniform':
            # Uniform grid sampling
            step = max(1, max_sensors // actual_n_sensors)
            self.sensor_idx = np.arange(0, max_sensors, step)[:actual_n_sensors]
        elif self.sensor_strategy == 'chebyshev':
            # Chebyshev nodes (better for interpolation)
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
            # Adaptive strategy: combination of uniform and boundary points
            rng = np.random.default_rng(42)
            
            # Start with coarse uniform grid (60% of sensors)
            uniform_count = int(actual_n_sensors * 0.6)
            step = max(1, int(np.sqrt(max_sensors / uniform_count)))
            uniform_indices = []
            for i in range(0, self.grid_size, step):
                for j in range(0, self.grid_size, step):
                    if len(uniform_indices) < uniform_count:
                        idx = i * self.grid_size + j
                        uniform_indices.append(idx)
            
            # Add boundary points (20% of sensors)
            boundary_count = int(actual_n_sensors * 0.2)
            boundary_indices = []
            # Corners
            corners = [0, self.grid_size-1, 
                      (self.grid_size-1)*self.grid_size, 
                      self.grid_size*self.grid_size-1]
            boundary_indices.extend(corners)
            
            # Edges
            for i in range(1, self.grid_size-1):
                # Top and bottom
                boundary_indices.extend([i, (self.grid_size-1)*self.grid_size + i])
                # Left and right
                boundary_indices.extend([i*self.grid_size, i*self.grid_size + (self.grid_size-1)])
            
            boundary_indices = list(set(boundary_indices))[:boundary_count]
            
            # Fill remaining with random sampling
            all_selected = set(uniform_indices + boundary_indices)
            remaining = actual_n_sensors - len(all_selected)
            
            if remaining > 0:
                available = list(set(range(max_sensors)) - all_selected)
                if len(available) > 0:
                    additional = rng.choice(available, size=min(remaining, len(available)), replace=False)
                    all_selected.update(additional)
            
            self.sensor_idx = np.array(list(all_selected)[:actual_n_sensors])
        else:
            # Default fallback to random
            print(f"Warning: Unknown sensor strategy '{self.sensor_strategy}', falling back to 'random'")
            rng = np.random.default_rng(42)
            self.sensor_idx = rng.choice(max_sensors, size=actual_n_sensors, replace=False)
        
        # Ensure sensor_idx is sorted
        self.sensor_idx = np.sort(self.sensor_idx)
        
        # Create coordinate grid
        x = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, x, indexing='ij')
        coords = np.column_stack([X.flatten(), Y.flatten()])
        
        # Normalize coordinates if requested
        if self.normalize_sensors:
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        
        self.trunk = torch.FloatTensor(coords).to(self.device)
        
        # Store for JSON serialization
        self.sensor_idx_list = self.sensor_idx.tolist()
        
        print(f"✓ Set up {len(self.sensor_idx)} sensors using '{self.sensor_strategy}' strategy")

    def _take_sensors(self, x: torch.Tensor) -> torch.Tensor:
        """Extract sensor values from input field"""
        # x: [B, 1, H, W] -> [B, H*W] -> select sensors
        B = x.shape[0]
        flat = x.view(B, -1)
        return flat[:, self.sensor_idx]

    def train_epoch(self, train_loader, val_loader=None):
        """Training epoch with FNO-style metrics tracking"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        
        # Track epoch number
        if not hasattr(self, 'epoch_num'):
            self.epoch_num = 0
        self.epoch_num += 1
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            # Conservative data augmentation
            if self._use_augmentation and self.model.training and self.epoch_num > 10:
                # Only augment after initial training
                if torch.rand(1).item() > 0.5:  # 50% of the time
                    noise_scale = self._augmentation_level * min(1.0, self.epoch_num / 200)
                    noise = torch.randn_like(x) * noise_scale
                    x = x + noise
            
            # Extract sensor data
            branch_input = self._take_sensors(x)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(branch_input, self.trunk)  # [B, grid_size^2]
            
            # Reshape target to match prediction
            B = y.shape[0]
            target = y.view(B, -1)
            
            loss = self.loss_fn(pred, target)
            
            # Calculate accuracy (Li et al. style)
            with torch.no_grad():
                diff = (pred - target).view(pred.size(0), -1)
                true = target.view(target.size(0), -1)
                rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8))
                sample_accuracy = (1 - rel_l2) * 100
                batch_accuracy = sample_accuracy.mean().item()
                total_accuracy += batch_accuracy * x.size(0)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
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
                "type": "Branch-Trunk Neural Network",
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
            "optimizer": f"Adam(lr={self.lr})",
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
            
            # Build model
            in_branch = len(self.sensor_idx)
            model = DeepONet(
                in_branch, 
                trunk_input_size=2, 
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                activation=self.activation,
                dropout=self.dropout
            ).to(self.device)
            
            self.models.append(model)
            
            # Create optimizer and scheduler
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
            self.optimizers.append(optimizer)
            
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
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
            
            if hasattr(self, '_use_augmentation'):
                # Slightly different augmentation per model
                self._augmentation_level = 0.005 * (1 + i * 0.2)
            
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
                "type": "Branch-Trunk Ensemble",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "n_models": self.n_models,
                "sensor_strategy": self.sensor_strategy
            },
            "parameters": trainable_params * self.n_models,
            "parameters_per_model": trainable_params,
            "optimizer": f"Adam(lr={self.lr})",
            "accuracy_method": "Li et al. (100*(1-relative_L2_error))"
        }
    
    def count_parameters(self):
        """Count total trainable parameters in ensemble"""
        params_per_model = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        return params_per_model * self.n_models