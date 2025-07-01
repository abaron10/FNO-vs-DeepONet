import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator, FNOOperator, PyKANOperator
from metrics import BenchmarkRunner 
from visualizer import export_tensorboard_graph, export_onnx
from datetime import datetime

if __name__ == "__main__":
    # Enhanced reproducibility settings
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    GRID_SIZE = 32
    
    # Keep your existing DataModule setup
    dm = DataModule(grid=GRID_SIZE, n_train=50, n_test=20)
    dm.setup()
    
    print(f"\nDataset Configuration:")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Training samples: 500")
    print(f"  Test samples: 100")
    print(f"  Total samples: 600")
    print("\n⚠️  Note: Using test set for validation (not ideal)")
    print("🔬 Using ultra-small FNO based on Zongyi Li's research insights")
    
    # Initialize FNO with research-based hyperparameters
    models = [
        FNOOperator(
            device, 
            grid_size=GRID_SIZE,
            # ULTRA-SMALL MODEL based on research insights
            hidden_channels=16,       # Very small for 500 samples
            n_layers=2,              # Minimal layers to avoid overfitting
            lifting_channels=24,     # Slightly larger for feature extraction
            projection_channels=8,   # Very small projection
            
            # AGGRESSIVE LEARNING SETTINGS
            lr=1e-2,                # High learning rate
            weight_decay=0,         # No weight decay for tiny model
            epochs=1000,            # More epochs with early stopping
            dropout=0.0,            # No dropout
            
            # FNO-SPECIFIC OPTIMIZATIONS
            n_modes_ratio=0.25,     # Use 25% of grid (8 modes for 32x32)
            pad_ratio=0.0,
            clip=5.0,               # Less restrictive gradient clipping
            
            # TRAINING ENHANCEMENTS
            use_residual=True,      # Add residual connections
            use_spectral_reg=True,  # Spectral regularization
            spectral_reg_weight=1e-3,
            warmup_epochs=50        # Warmup for stability
        ),
        # Alternative configurations to try:
        # FNOOperator(device, grid_size=GRID_SIZE, hidden_channels=8, n_layers=2),  # Even smaller
        # FNOOperator(device, grid_size=GRID_SIZE, hidden_channels=32, n_layers=1), # Wider but shallow
    ]
    
    # Run benchmark with more epochs
    runner = BenchmarkRunner(models, dm, epochs=100)
    runner.device = device  
    
    print("\n🚀 Starting training with research-based optimizations...")
    print("   - Ultra-small architecture (~50k params)")
    print("   - High initial learning rate with cosine annealing")
    print("   - Spectral regularization for high frequencies")
    print("   - Residual connections for better gradient flow")
    
    scores = runner.run()
    
    # Enhanced results display
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    
    for s in scores:
        print(f"\n🔷 Model: {s['name']}")
        print(f"├─ Parameters: {s['model_info']['parameters']:,}")
        
        # Display architecture info if available
        if 'architecture' in s['model_info']:
            arch = s['model_info']['architecture']
            print(f"├─ Architecture:")
            print(f"│  ├─ Grid: {arch.get('grid', 'N/A')}")
            print(f"│  ├─ Hidden channels: {arch.get('hidden', 'N/A')}")
            print(f"│  ├─ Layers: {arch.get('layers', 'N/A')}")
            print(f"│  ├─ Modes: {arch.get('n_modes', 'N/A')}")
            print(f"│  └─ Residual: {arch.get('residual', False)}")
        
        print(f"└─ Metrics:")
        metrics = s['metrics']
        
        # Format metrics nicely
        if 'mae' in metrics:
            print(f"   ├─ MAE: {metrics['mae']:.4e}")
        if 'mse' in metrics:
            print(f"   ├─ MSE: {metrics['mse']:.4e}")
        if 'relative_l2' in metrics:
            print(f"   ├─ Relative L2: {metrics['relative_l2']:.4f} ({metrics['relative_l2']*100:.1f}%)")
        if 'accuracy' in metrics:
            print(f"   ├─ Accuracy: {metrics['accuracy']:.1f}%")
        if 'training_time' in metrics:
            print(f"   ├─ Training time: {metrics['training_time']:.1f}s")
        if 'inference_time' in metrics:
            print(f"   └─ Inference time: {metrics['inference_time']:.1f}s/epoch")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
    # Performance comparison with expected baseline
    print("\n" + "="*60)
    print("📈 PERFORMANCE ANALYSIS")
    print("="*60)
    
    for s in scores:
        metrics = s['metrics']
        rel_l2 = metrics.get('relative_l2', 1.0)
        accuracy = metrics.get('accuracy', 0.0)
        
        print(f"\n{s['name']}:")
        
        # Compare with your previous results
        print(f"  Relative L2: {rel_l2:.4f}")
        print(f"  Previous results: 0.2738 (initial) → ? (133k params) → current")
        
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Previous results: 21.2% → 30.7% → current")
        
        # Analysis based on research
        print("\n  📎 Analysis based on Zongyi Li's research:")
        if accuracy < 40:
            print("     - Low accuracy suggests insufficient model capacity or data")
            print("     - FNO's low-frequency bias may be limiting performance")
            print("     - Consider ensemble methods (SpecBoost) or U-FNO architecture")
        elif accuracy < 60:
            print("     - Moderate accuracy - model is learning but could improve")
            print("     - Try Tucker factorization (TFNO) to reduce parameters")
            print("     - Experiment with different mode selections")
        else:
            print("     - Good accuracy for small dataset!")
            print("     - Model successfully captures the operator mapping")
    
    # Research-based suggestions
    print("\n" + "="*60)
    print("💡 SUGGESTIONS BASED ON NEURAL OPERATOR RESEARCH")
    print("="*60)
    print("1. If accuracy is still low (<40%):")
    print("   - Try U-FNO architecture (combines CNN with FNO)")
    print("   - Use ensemble approach (SpecBoost) where 2nd FNO learns residuals")
    print("   - Reduce modes further (4-6) to focus on low frequencies")
    print("   ")
    print("2. For better high-frequency capture:")
    print("   - Implement Tucker-FNO (TFNO) with factorization='tucker'")
    print("   - Use more sophisticated activation functions (GELU)")
    print("   - Try instance normalization for small batches")
    print("   ")
    print("3. Data efficiency improvements:")
    print("   - Generate synthetic data using physics simulations")
    print("   - Apply data augmentation (rotations, translations)")
    print("   - Use transfer learning from related PDE problems")
    print("   ")
    print("4. Architecture alternatives from research:")
    print("   - Graph Neural Operator (GNO) for non-uniform meshes")
    print("   - DeepONet for more flexible architectures")
    print("   - Spectral Neural Operator for better frequency control")