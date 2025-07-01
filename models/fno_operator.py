import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
from .base_operator import BaseOperator

class FNOOperator(BaseOperator):
    """FNO optimized for small datasets based on Zongyi Li's research insights."""

    def __init__(self, device, grid_size=32,
                 # ULTRA-SMALL MODEL for 500 samples
                 hidden_channels=16,     # Even smaller
                 n_layers=2,            # Minimal layers
                 lifting_channels=24,   # Slightly larger lifting for feature extraction
                 projection_channels=8, # Very small projection
                 
                 # AGGRESSIVE LEARNING 
                 lr=1e-2,               # Very high initial LR
                 weight_decay=0,        # No weight decay for small models
                 epochs=1000,           # More epochs with early stopping
                 dropout=0.0,           # No dropout
                 
                 # FNO-specific settings
                 n_modes_ratio=0.25,    # Use 25% of grid size for modes
                 pad_ratio=0.0,         
                 clip=5.0,              # Less restrictive clipping
                 
                 # Training tricks
                 use_residual=True,     # Add residual connections
                 use_spectral_reg=True, # Spectral regularization
                 spectral_reg_weight=1e-3,
                 warmup_epochs=50):

        super().__init__(device, grid_size)
        
        # Calculate modes based on grid size and data amount
        # For small datasets, use fewer modes to avoid overfitting
        self.n_modes = (int(grid_size * n_modes_ratio), 
                       int(grid_size * n_modes_ratio))  # 8x8 for 32x32 grid
        
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.pad_ratio = pad_ratio
        self.clip = clip
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_spectral_reg = use_spectral_reg
        self.spectral_reg_weight = spectral_reg_weight
        self.warmup_epochs = warmup_epochs
        
        # Early stopping with aggressive settings
        self.best_val_loss = float('inf')
        self.patience = 150  # Very patient for small datasets
        self.patience_counter = 0
        self.min_delta = 1e-7
        
        # Track training statistics
        self.loss_history = []

    def setup(self, data_info):
        from neuralop.models import FNO
        
        # Robust normalization
        self.k_mean = torch.tensor(data_info["k_mean"], device=self.device)
        self.k_std = torch.tensor(data_info["k_std"], device=self.device)
        
        # Add small epsilon to avoid division by zero
        self.k_std = torch.clamp(self.k_std, min=1e-6)
        
        # Initialize base FNO
        self.model = FNO(
            n_modes=self.n_modes,
            hidden_channels=self.hidden_channels,
            in_channels=1,
            out_channels=1,
            lifting_channels=self.lifting_channels,
            projection_channels=self.projection_channels,
            n_layers=self.n_layers,
            pad_ratio=self.pad_ratio,
            domain_padding=None,
            # Additional FNO settings from neuraloperator library
            mlp_ratio=2,  # Smaller MLP expansion
            non_linearity=F.gelu,  # GELU often works better than ReLU
            norm="instance_norm",  # Instance norm for small batches
            preactivation=False,
            skip="soft-gating",  # Soft gating skip connections
            separable=False,
            factorization=None  # No factorization for small model
        ).to(self.device)
        
        # Add residual connection wrapper if enabled
        if self.use_residual:
            self.model = ResidualWrapper(self.model).to(self.device)
        
        # Custom initialization for small datasets
        self._initialize_weights()
        
        # Adam with momentum scheduling
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),  # Slightly lower beta2 for faster adaptation
            eps=1e-9
        )
        
        # Cosine annealing with warm restarts
        self.sched = CosineAnnealingWarmRestarts(
            self.opt,
            T_0=100,  # Restart every 100 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Loss function with potential for multiple components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def _initialize_weights(self):
        """Special initialization for small datasets"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Smaller initialization for small datasets
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)

    def _norm(self, x):
        """Robust normalization"""
        # Z-score normalization with clipping
        normalized = (x - self.k_mean) / self.k_std
        # Clip extreme values to prevent instability
        return torch.clamp(normalized, -5, 5)
    
    def _denorm(self, x):
        """Denormalize predictions"""
        return x * self.k_std + self.k_mean

    def _spectral_regularization(self):
        """Add spectral regularization to prevent overfitting on high frequencies"""
        if not self.use_spectral_reg:
            return 0
        
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if 'spectral' in name or 'fourier' in name:
                # Penalize high frequency components
                reg_loss += torch.norm(param, p=2)
        
        return self.spectral_reg_weight * reg_loss

    def _compute_loss(self, pred, target):
        """Multi-component loss function"""
        # Main loss
        mse = self.mse_loss(pred, target)
        
        # Add L1 for robustness with small weight
        l1 = self.l1_loss(pred, target)
        
        # Spectral regularization
        spec_reg = self._spectral_regularization()
        
        # Combine losses
        total_loss = mse + 0.1 * l1 + spec_reg
        
        return total_loss, {'mse': mse, 'l1': l1, 'spec_reg': spec_reg}

    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        loss_components = {'mse': 0, 'l1': 0, 'spec_reg': 0}
        
        # Warmup phase
        epoch_num = len(self.loss_history)
        if epoch_num < self.warmup_epochs:
            warmup_lr = self.lr * (epoch_num + 1) / self.warmup_epochs
            for param_group in self.opt.param_groups:
                param_group['lr'] = warmup_lr
        
        for batch_idx, batch in enumerate(train_loader):
            x = self._norm(batch["x"].to(self.device))
            y = batch["y"].to(self.device)
            
            # Forward pass
            self.opt.zero_grad()
            pred = self.model(x)
            
            # Compute loss
            loss, components = self._compute_loss(pred, y)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}, skipping")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping only for real gradients
            real_params = []
            grad_norm = 0
            for p in self.model.parameters():
                if p.grad is not None and not torch.is_complex(p.grad):
                    real_params.append(p)
                    grad_norm += p.grad.data.norm(2).item() ** 2
            
            grad_norm = grad_norm ** 0.5
            
            if real_params and grad_norm > self.clip:
                clip_grad_norm_(real_params, max_norm=self.clip)
            
            self.opt.step()
            
            # Track metrics
            accuracy = self._calculate_accuracy(pred, y, threshold=0.15)
            running_accuracy += accuracy
            running_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Update scheduler after warmup
        if epoch_num >= self.warmup_epochs:
            self.sched.step()
        
        # Average metrics
        n_batches = len(train_loader)
        avg_train_loss = running_loss / n_batches
        avg_train_accuracy = running_accuracy / n_batches
        for k in loss_components:
            loss_components[k] /= n_batches
        
        # Validation
        val_loss = float('inf')
        val_accuracy = 0.0
        should_stop = False
        
        if val_loader is not None:
            val_loss, val_accuracy = self._evaluate_validation(val_loader)
            
            # Early stopping
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'epoch': epoch_num,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, 'best_fno_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    should_stop = True
        
        # Store history
        self.loss_history.append(avg_train_loss)
        
        # Debug output every 100 epochs
        if epoch_num % 100 == 0:
            print(f"\n[Epoch {epoch_num}] Loss: {avg_train_loss:.6f}, "
                  f"Acc: {avg_train_accuracy:.1f}%, "
                  f"Val Acc: {val_accuracy:.1f}%, "
                  f"LR: {self.opt.param_groups[0]['lr']:.2e}")
            print(f"  Components - MSE: {loss_components['mse']:.6f}, "
                  f"L1: {loss_components['l1']:.6f}, "
                  f"Spec: {loss_components['spec_reg']:.6f}")
        
        return {
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'lr': self.opt.param_groups[0]['lr'],
            'should_stop': should_stop,
            'components': loss_components
        }

    def _calculate_accuracy(self, pred, target, threshold=0.15):
        """Calculate relative error accuracy"""
        with torch.no_grad():
            rel_error = torch.abs(pred - target) / (torch.abs(target).mean() + 1e-8)
            accuracy = (rel_error < threshold).float().mean().item() * 100
            return accuracy

    @torch.no_grad()
    def _evaluate_validation(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        
        for batch in val_loader:
            x = self._norm(batch["x"].to(self.device))
            y = batch["y"].to(self.device)
            
            pred = self.model(x)
            loss, _ = self._compute_loss(pred, y)
            
            accuracy = self._calculate_accuracy(pred, y, threshold=0.15)
            total_accuracy += accuracy
            total_loss += loss.item()
        
        avg_val_loss = total_loss / len(val_loader)
        avg_val_accuracy = total_accuracy / len(val_loader)
        
        self.model.train()
        return avg_val_loss, avg_val_accuracy

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        x = self._norm(batch["x"].to(self.device))
        pred = self.model(x)
        return pred

    def get_model_info(self):
        param_count = self.count_parameters()
        return {
            "name": "FNO_SmallData_Optimized",
            "architecture": {
                "grid": f"{self.grid_size}×{self.grid_size}",
                "n_modes": self.n_modes,
                "hidden": self.hidden_channels,
                "layers": self.n_layers,
                "lifting": self.lifting_channels,
                "projection": self.projection_channels,
                "residual": self.use_residual,
            },
            "parameters": param_count,
            "optimizer": f"Adam(lr={self.lr})",
            "scheduler": "CosineAnnealingWarmRestarts",
            "regularization": f"Spectral({self.spectral_reg_weight})"
        }


class ResidualWrapper(nn.Module):
    """Wrapper to add residual connections to FNO"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Simple residual connection
        return self.model(x) + x