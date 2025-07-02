"""
FNO implementation for small datasets based on Li et al. (2020)
"Fourier Neural Operator for Parametric Partial Differential Equations"

Key adaptations for small datasets (500 samples):
- Very few Fourier modes (4-6) to prevent overfitting
- Narrow architecture (width 16-20)
- Data augmentation through flips
- Proper positional encoding with grid
- Step learning rate decay

Accuracy calculation follows Li et al.:
- Accuracy = 100 * (1 - relative_L2_error)
- Where relative_L2_error = ||u_pred - u_true||_2 / ||u_true||_2

Note: FFT operations create complex gradients that require special handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from .base_operator import BaseOperator


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Initialize weights with smaller scale for small datasets
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    """FNO block optimized for small datasets"""
    def __init__(self, in_channels, out_channels, modes1, modes2, activation='gelu'):
        super().__init__()
        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.w = nn.Conv2d(in_channels, out_channels, 1)
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = lambda x: x
            
    def forward(self, x):
        return self.activation(self.conv(x) + self.w(x))


class FNOOperator(BaseOperator):
    """FNO optimized for very small datasets following Li et al. recommendations"""

    def __init__(self, device, grid_size=32,
                 modes=6,               
                 width=20,              
                 n_layers=4,            
                 in_channels=1,         
                 
                 # Training settings
                 lr=1e-3,              
                 step_size=100,         
                 gamma=0.5,            
                 weight_decay=1e-4,    
                 epochs=500,
                 
                 # Data augmentation
                 use_augmentation=True,
                 
                 # Architecture choices
                 share_weights=False,    
                 activation='gelu'):     # GELU as in paper

        super().__init__(device, grid_size)
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels  
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.use_augmentation = use_augmentation
        self.share_weights = share_weights
        self.activation = activation
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 50
        self.patience_counter = 0

    def setup(self, data_info):
        self.input_mean = torch.tensor(data_info["k_mean"], device=self.device)
        self.input_std = torch.tensor(data_info["k_std"], device=self.device)
        
        # Ensure numerical stability
        self.input_std = torch.clamp(self.input_std, min=1e-8)
        
        # Grid has x,y coordinates
        self.grid_channels = 2
        
        # Build model architecture following Li et al.
        self.build_model()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer, 
            step_size=self.step_size, 
            gamma=self.gamma
        )

    def build_model(self):
        """Build FNO model following Li et al. architecture"""
        layers = []
        
        # Total input channels = data channels + grid channels
        total_in_channels = self.in_channels + self.grid_channels
        
        # Lifting layer: lift from total input channels to width
        layers.append(nn.Conv2d(total_in_channels, self.width, 1))
        
        # FNO blocks
        if self.share_weights:
            # Share weights across layers (reduces parameters)
            fno_block = FNOBlock(self.width, self.width, self.modes, self.modes, self.activation)
            for _ in range(self.n_layers):
                layers.append(fno_block)
        else:
            # Independent weights for each layer
            for _ in range(self.n_layers):
                layers.append(FNOBlock(self.width, self.width, self.modes, self.modes, self.activation))
        
        # Projection layers
        layers.append(nn.Conv2d(self.width, 128, 1))
        layers.append(self._get_activation())
        layers.append(nn.Conv2d(128, 1, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        # Initialize weights carefully
        self._initialize_weights()

    def _get_activation(self):
        """Get activation function"""
        if self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'relu':
            return nn.ReLU()
        else:
            return nn.Identity()

    def _initialize_weights(self):
        """Initialize weights following best practices for small datasets"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _create_grid(self, shape, device):
        """Create mesh grid for positional encoding"""
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, device=device)
        gridy = torch.linspace(0, 1, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        grid = torch.stack([gridx, gridy], dim=-1)
        grid = grid.reshape(1, size_x, size_y, 2).permute(0, 3, 1, 2)
        return grid.repeat(batchsize, 1, 1, 1)

    def _augment_data(self, x, y):
        """Simple data augmentation for PDEs"""
        if not self.use_augmentation or not self.model.training:
            return x, y
        
        # Random horizontal/vertical flips (if PDE is symmetric)
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[3])
            
        return x, y

    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        
        # Track epoch number for debug output
        if not hasattr(self, 'epoch_num'):
            self.epoch_num = 0
        epoch_num = self.epoch_num
        self.epoch_num += 1
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            # Auto-detect input channels on first batch
            if batch_idx == 0 and not hasattr(self, '_input_channels_verified'):
                actual_channels = x.shape[1]
                expected_channels = self.in_channels
                if actual_channels != expected_channels:
                    print(f"\n⚠️  Channel mismatch detected!")
                    print(f"   Expected: {expected_channels} channels")
                    print(f"   Received: {actual_channels} channels") 
                    print(f"   Input shape: {x.shape}")
                    print(f"   Please set in_channels={actual_channels} when creating the model\n")
                    raise ValueError(f"Input has {actual_channels} channels but model expects {expected_channels}")
                self._input_channels_verified = True
            
            # Data augmentation
            x, y = self._augment_data(x, y)
            
            # Normalize input
            x = (x - self.input_mean) / self.input_std
            
            # Add grid
            grid = self._create_grid(x.shape, x.device)
            x = torch.cat([x, grid], dim=1)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = F.mse_loss(out, y)
            
            # Calculate accuracy for training (Li et al. style)
            with torch.no_grad():
                # Compute per-sample relative L2 error
                diff = (out - y).view(out.size(0), -1)
                true = y.view(y.size(0), -1)
                rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8))  # shape (batch,)
                
                # Convert to accuracy = 100 * (1 - error)
                sample_accuracy = (1 - rel_l2) * 100  # shape (batch,)
                
                # Average over batch
                batch_accuracy = sample_accuracy.mean().item()
                total_accuracy += batch_accuracy * x.size(0)
                
                # Debug: print first batch stats
                if batch_idx == 0 and epoch_num == 0:
                    avg_rel_l2 = rel_l2.mean().item()
                    print(f"\n📊 Accuracy calculation (Li et al. method):")
                    print(f"   Relative L2 error: {avg_rel_l2:.4f}")
                    print(f"   Accuracy = 100*(1-L2_err) = {batch_accuracy:.1f}%")
                    print(f"   Note: Negative accuracy means L2 error > 1.0")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping - handle complex gradients from FFT
            # FFT operations in SpectralConv2d create complex gradients that
            # cannot be clipped with standard norm operations
            real_params = []
            for p in self.model.parameters():
                if p.grad is not None and not torch.is_complex(p.grad):
                    real_params.append(p)
            
            if real_params:
                torch.nn.utils.clip_grad_norm_(real_params, 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        avg_train_loss = total_loss / total_samples
        avg_train_accuracy = total_accuracy / total_samples
        
        # Validation
        val_loss = float('inf')
        val_accuracy = 0
        val_rel_l2 = 1.0  # Default relative L2 error
        if val_loader is not None:
            val_loss, val_accuracy = self.evaluate(val_loader)
            # Calculate relative L2 from accuracy: acc = 100*(1-L2) => L2 = 1-acc/100
            val_rel_l2 = 1 - val_accuracy/100
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_fno_model.pth')
            else:
                self.patience_counter += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_rel_l2': val_rel_l2,
            'lr': self.optimizer.param_groups[0]['lr'],
            'should_stop': self.patience_counter >= self.patience,
            'info': 'Accuracy computed as 100*(1-relative_L2_error) following Li et al.'
        }

    def _calculate_relative_l2_error(self, pred, target):
        """Calculate relative L2 error as in Li et al."""
        with torch.no_grad():
            # Flatten to compute norms
            diff = (pred - target).view(pred.size(0), -1)
            true = target.view(target.size(0), -1)
            
            # Compute per-sample relative L2 error
            rel_l2 = diff.norm(dim=1) / (true.norm(dim=1) + 1e-8)
            
            return rel_l2.mean().item()

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        for batch in data_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            # Normalize
            x = (x - self.input_mean) / self.input_std
            
            # Add grid
            grid = self._create_grid(x.shape, x.device)
            x = torch.cat([x, grid], dim=1)
            
            # Forward pass
            out = self.model(x)
            loss = F.mse_loss(out, y)
            
            # Calculate accuracy using Li et al. method (same as training)
            # Compute per-sample relative L2 error
            diff = (out - y).view(out.size(0), -1)
            true = y.view(y.size(0), -1)
            rel_l2 = (diff.norm(dim=1) / (true.norm(dim=1) + 1e-8))  # shape (batch,)
            
            # Convert to accuracy = 100 * (1 - error)
            sample_accuracy = (1 - rel_l2) * 100  # shape (batch,)
            
            # Sum for averaging later
            total_loss += loss.item() * out.size(0)
            total_accuracy += sample_accuracy.sum().item()
            total_samples += out.size(0)
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        return avg_loss, avg_accuracy

    @torch.no_grad()
    def predict(self, batch):
        """Make predictions"""
        self.model.eval()
        x = batch["x"].to(self.device)
        
        # Normalize
        x = (x - self.input_mean) / self.input_std
        
        # Add grid
        grid = self._create_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=1)
        
        return self.model(x)

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "name": "FNO_LiEtAl_AccuracyMethod",
            "architecture": {
                "grid": f"{self.grid_size}×{self.grid_size}",
                "modes": self.modes,
                "width": self.width,
                "layers": self.n_layers,
                "activation": self.activation,
                "share_weights": self.share_weights
            },
            "parameters": trainable_params,
            "total_parameters": total_params,
            "optimizer": f"Adam(lr={self.lr})",
            "accuracy_method": "Li et al. (100*(1-relative_L2_error))"
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class FNOEnsembleOperator(FNOOperator):
    """Ensemble of FNO models for better accuracy (simplified SpecBoost)"""
    
    def __init__(self, device, grid_size=32, n_models=2, **kwargs):
        super().__init__(device, grid_size, **kwargs)
        self.n_models = n_models
        
    def setup(self, data_info):
        super().setup(data_info)
        
        # Create ensemble of models with different initializations
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        for i in range(self.n_models):
            # Set different seed for each model
            torch.manual_seed(42 + i)
            
            # Build model
            self.build_model()
            self.models.append(self.model)
            
            # Create optimizer
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
            self.optimizers.append(optimizer)
            
            # Create scheduler
            scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            self.schedulers.append(scheduler)
        
        # Reset seed
        torch.manual_seed(42)
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train all models in ensemble"""
        results = []
        
        for i, (model, optimizer, scheduler) in enumerate(zip(self.models, self.optimizers, self.schedulers)):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            
            result = super().train_epoch(train_loader, val_loader)
            results.append(result)
        
        # Average results
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
        
        # Return average performance
        return sum(all_losses) / len(all_losses), sum(all_accuracies) / len(all_accuracies)
    
    @torch.no_grad()
    def predict(self, batch):
        """Ensemble prediction"""
        predictions = []
        
        for model in self.models:
            self.model = model
            pred = super().predict(batch)
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def get_model_info(self):
        """Get ensemble model info"""
        # Count parameters from first model (all should be same size)
        total_params = sum(p.numel() for p in self.models[0].parameters())
        trainable_params = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        
        return {
            "name": f"FNO_Ensemble_{self.n_models}models_LiAccuracy",
            "architecture": {
                "grid": f"{self.grid_size}×{self.grid_size}",
                "modes": self.modes,
                "width": self.width,
                "layers": self.n_layers,
                "activation": self.activation,
                "n_models": self.n_models,
                "share_weights": self.share_weights
            },
            "parameters": trainable_params * self.n_models,  # Total params across all models
            "parameters_per_model": trainable_params,
            "optimizer": f"Adam(lr={self.lr})",
            "accuracy_method": "Li et al. (100*(1-relative_L2_error))"
        }
    
    def count_parameters(self):
        """Count total trainable parameters in ensemble"""
        params_per_model = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        return params_per_model * self.n_models