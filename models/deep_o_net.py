import torch
import numpy as np
from .base_operator import BaseOperator

class ImprovedDeepONet(torch.nn.Module):
    """Enhanced DeepONet following Lu et al. (2019) best practices."""

    def __init__(self, branch_input_size: int, trunk_input_size: int = 2,
                 p: int = 100,  # Number of basis functions (outputs)
                 branch_layers: list = None, trunk_layers: list = None,
                 activation: str = 'relu'):
        super().__init__()
        
        self.p = p
        
        # Default architectures from paper
        if branch_layers is None:
            branch_layers = [branch_input_size, 40, 40, p]
        if trunk_layers is None:
            trunk_layers = [trunk_input_size, 40, 40, p]
            
        # Activation function
        if activation == 'relu':
            self.act = torch.nn.ReLU()
        elif activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'gelu':
            self.act = torch.nn.GELU()
        else:
            self.act = torch.nn.ReLU()
        
        # Build Branch Network
        branch_net = []
        for i in range(len(branch_layers) - 1):
            branch_net.append(torch.nn.Linear(branch_layers[i], branch_layers[i+1]))
            if i < len(branch_layers) - 2:  # No activation on last layer
                branch_net.append(self.act)
        self.branch_net = torch.nn.Sequential(*branch_net)
        
        # Build Trunk Network  
        trunk_net = []
        for i in range(len(trunk_layers) - 1):
            trunk_net.append(torch.nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            # Apply activation on ALL layers including last (as per paper)
            trunk_net.append(self.act)
        self.trunk_net = torch.nn.Sequential(*trunk_net)
        
        # Bias term (critical for performance as per paper)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Use gain=1.0 for standard initialization
                torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    
    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_in: [B, n_sensors] - input function at sensor locations
            trunk_in: [N, 2] - evaluation points (x,y coordinates)
        Returns:
            [B, N] - output values
        """
        b = self.branch_net(branch_in)  # [B, p]
        t = self.trunk_net(trunk_in)    # [N, p]
        
        # Proper dot product as per paper equation
        # out[i,j] = sum_k b[i,k] * t[j,k] + bias
        out = torch.matmul(b, t.T) + self.bias  # [B, N]
        
        return out


class DeepONetOperator(BaseOperator):
    """DeepONet operator following paper recommendations"""

    def __init__(self, device: torch.device, name: str = "", grid_size: int = 64,
                 # Sensor configuration
                 n_sensors: int = 100,  # Paper shows 100 is often sufficient
                 sensor_strategy: str = 'chebyshev',  # Better for interpolation
                 
                 # Architecture (from paper)
                 p: int = 100,  # Number of basis functions
                 branch_layers: list = None,
                 trunk_layers: list = None,
                 activation: str = 'tanh',  # Paper uses tanh
                 
                 # Training settings
                 lr: float = 1e-3,  # Paper default
                 epochs: int = 50000,  # Paper uses 50k-100k iterations
                 batch_size: int = 100,  # Reasonable batch size
                 
                 # Advanced options
                 sensor_noise: float = 0.0,  # Add noise to sensors for robustness
                 normalize_inputs: bool = True,
                 normalize_outputs: bool = False):
        
        super().__init__(device, grid_size)
        
        self.name = name
        self.n_sensors = min(n_sensors, grid_size ** 2)
        self.sensor_strategy = sensor_strategy
        self.p = p
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.sensor_noise = sensor_noise
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        
        # For input/output normalization
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def setup(self, data_info):
        # Generate sensor locations
        self._setup_sensors()
        
        # Create coordinate grid
        self._setup_coordinates()
        
        # Build model
        self.model = ImprovedDeepONet(
            branch_input_size=len(self.sensor_idx),
            trunk_input_size=2,
            p=self.p,
            branch_layers=self.branch_layers,
            trunk_layers=self.trunk_layers,
            activation=self.activation
        ).to(self.device)
        
        # Optimizer - Adam as per paper
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # For early stopping
        self.best_val_loss = float('inf')
        self.patience = 100
        self.patience_counter = 0

    def _setup_sensors(self):
        """Setup sensor locations using various strategies"""
        max_sensors = self.grid_size ** 2
        n_sensors = min(self.n_sensors, max_sensors)
        
        if self.sensor_strategy == 'random':
            rng = np.random.default_rng(42)
            self.sensor_idx = rng.choice(max_sensors, size=n_sensors, replace=False)
            
        elif self.sensor_strategy == 'uniform':
            # Uniform grid
            sqrt_n = int(np.sqrt(n_sensors))
            x_idx = np.linspace(0, self.grid_size-1, sqrt_n, dtype=int)
            y_idx = np.linspace(0, self.grid_size-1, sqrt_n, dtype=int)
            xx, yy = np.meshgrid(x_idx, y_idx)
            flat_idx = xx.flatten() * self.grid_size + yy.flatten()
            self.sensor_idx = flat_idx[:n_sensors]
            
        elif self.sensor_strategy == 'chebyshev':
            # Chebyshev nodes for better interpolation
            n = int(np.sqrt(n_sensors))
            i = np.arange(n)
            # Chebyshev points in [-1, 1]
            cheb = np.cos((2*i + 1) * np.pi / (2 * n))
            # Map to [0, grid_size-1]
            cheb_grid = ((cheb + 1) / 2 * (self.grid_size - 1)).astype(int)
            
            xx, yy = np.meshgrid(cheb_grid, cheb_grid)
            flat_idx = xx.flatten() * self.grid_size + yy.flatten()
            self.sensor_idx = np.unique(flat_idx)[:n_sensors]
            
        elif self.sensor_strategy == 'lhs':
            # Latin Hypercube Sampling for better coverage
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=2, seed=42)
            sample = sampler.random(n=n_sensors)
            coords = (sample * self.grid_size).astype(int)
            flat_idx = coords[:, 0] * self.grid_size + coords[:, 1]
            self.sensor_idx = np.unique(flat_idx)[:n_sensors]
            
        # Store for JSON serialization
        self.sensor_idx_list = self.sensor_idx.tolist()

    def _setup_coordinates(self):
        """Create normalized coordinate grid"""
        # Create grid in [0, 1]
        x = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, x)
        coords = np.column_stack([X.flatten(), Y.flatten()])
        
        # Convert to tensor
        self.trunk = torch.FloatTensor(coords).to(self.device)

    def _extract_sensors(self, x: torch.Tensor) -> torch.Tensor:
        """Extract sensor values with optional noise"""
        B = x.shape[0]
        flat = x.view(B, -1)
        sensor_vals = flat[:, self.sensor_idx]
        
        # Add noise during training for robustness
        if self.training and self.sensor_noise > 0:
            noise = torch.randn_like(sensor_vals) * self.sensor_noise
            sensor_vals = sensor_vals + noise
            
        return sensor_vals

    def _normalize_data(self, train_loader):
        """Compute normalization statistics from training data"""
        if not self.normalize_inputs and not self.normalize_outputs:
            return
            
        all_inputs = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                
                if self.normalize_inputs:
                    inputs = self._extract_sensors(x)
                    all_inputs.append(inputs)
                    
                if self.normalize_outputs:
                    all_outputs.append(y.view(y.size(0), -1))
        
        if self.normalize_inputs:
            all_inputs = torch.cat(all_inputs, dim=0)
            self.input_mean = all_inputs.mean(dim=0, keepdim=True)
            self.input_std = all_inputs.std(dim=0, keepdim=True) + 1e-8
            
        if self.normalize_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            self.output_mean = all_outputs.mean()
            self.output_std = all_outputs.std() + 1e-8

    def train_epoch(self, train_loader, val_loader=None):
        """Training epoch with proper normalization and metrics"""
        # Compute normalization stats on first epoch
        if not hasattr(self, 'epoch_num'):
            self.epoch_num = 0
            self._normalize_data(train_loader)
        self.epoch_num += 1
        
        self.model.train()
        self.training = True
        
        total_loss = 0
        total_samples = 0
        total_rel_error = 0
        
        for batch in train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            # Extract and normalize sensor data
            branch_input = self._extract_sensors(x)
            if self.normalize_inputs:
                branch_input = (branch_input - self.input_mean) / self.input_std
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(branch_input, self.trunk)  # [B, grid_size^2]
            
            # Reshape target
            B = y.shape[0]
            target = y.view(B, -1)
            
            # Normalize target if needed
            if self.normalize_outputs:
                target_norm = (target - self.output_mean) / self.output_std
                loss = self.loss_fn(pred, target_norm)
                # Denormalize prediction for metrics
                pred_denorm = pred * self.output_std + self.output_mean
            else:
                loss = self.loss_fn(pred, target)
                pred_denorm = pred
            
            # Calculate relative L2 error
            with torch.no_grad():
                diff = (pred_denorm - target).view(B, -1)
                true = target.view(B, -1)
                rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8)).mean()
                total_rel_error += rel_l2.item() * B
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * B
            total_samples += B
        
        avg_train_loss = total_loss / total_samples
        avg_rel_error = total_rel_error / total_samples
        avg_train_accuracy = (1 - avg_rel_error) * 100
        
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
        
        self.training = False
        
        return {
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': self.lr,
            'should_stop': self.patience_counter >= self.patience,
            'epoch': self.epoch_num
        }

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        self.training = False
        
        total_loss = 0
        total_rel_error = 0
        total_samples = 0
        
        for batch in data_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            # Extract and normalize sensor data
            branch_input = self._extract_sensors(x)
            if self.normalize_inputs:
                branch_input = (branch_input - self.input_mean) / self.input_std
                
            pred = self.model(branch_input, self.trunk)
            
            B = y.shape[0]
            target = y.view(B, -1)
            
            # Handle normalization
            if self.normalize_outputs:
                target_norm = (target - self.output_mean) / self.output_std
                loss = self.loss_fn(pred, target_norm)
                pred_denorm = pred * self.output_std + self.output_mean
            else:
                loss = self.loss_fn(pred, target)
                pred_denorm = pred
            
            # Calculate relative L2 error
            diff = (pred_denorm - target).view(B, -1)
            true = target.view(B, -1)
            rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8)).mean()
            
            total_loss += loss.item() * B
            total_rel_error += rel_l2.item() * B
            total_samples += B
        
        avg_loss = total_loss / total_samples
        avg_accuracy = (1 - total_rel_error / total_samples) * 100
        
        return avg_loss, avg_accuracy

    @torch.no_grad()
    def predict(self, batch):
        """Make predictions"""
        self.model.eval()
        self.training = False
        
        x = batch["x"].to(self.device)
        
        branch_input = self._extract_sensors(x)
        if self.normalize_inputs:
            branch_input = (branch_input - self.input_mean) / self.input_std
            
        pred = self.model(branch_input, self.trunk)
        
        if self.normalize_outputs:
            pred = pred * self.output_std + self.output_mean
        
        # Reshape back
        B = branch_input.shape[0]
        return pred.view(B, 1, self.grid_size, self.grid_size)

    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "name": f"{self.name}_{self.grid_size}x{self.grid_size}",
            "architecture": {
                "type": "DeepONet (Lu et al. 2019)",
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_sensors": len(self.sensor_idx),
                "sensor_coverage": f"{len(self.sensor_idx)/(self.grid_size**2)*100:.1f}%",
                "p_basis": self.p,
                "branch_arch": self.branch_layers if self.branch_layers else "default",
                "trunk_arch": self.trunk_layers if self.trunk_layers else "default",
                "activation": self.activation,
                "sensor_strategy": self.sensor_strategy
            },
            "parameters": trainable_params,
            "total_parameters": total_params,
            "optimizer": f"Adam(lr={self.lr})",
            "accuracy_method": "100*(1-relative_L2_error)"
        }

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)