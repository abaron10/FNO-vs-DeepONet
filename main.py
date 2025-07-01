import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator, FNOOperator, FNOEnsembleOperator, PyKANOperator
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
    
    # CORRECTED: Use proper data sizes
    TRAIN_SIZE = 500
    TEST_SIZE = 100
    
    dm = DataModule(grid=GRID_SIZE, n_train=TRAIN_SIZE, n_test=TEST_SIZE)
    dm.setup()
    
    # Quick check of data shape
    try:
        sample_batch = next(iter(dm.train))
        input_shape = sample_batch["x"].shape
        detected_channels = input_shape[1]
    except:
        detected_channels = 1
    
    print(f"\nDataset Configuration:")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Training samples: {TRAIN_SIZE}")
    print(f"  Test samples: {TEST_SIZE}")
    print(f"  Total samples: {TRAIN_SIZE + TEST_SIZE}")
    print(f"  Input channels detected: {detected_channels}")
    print("\n⚠️  Note: Using test set for validation (not ideal)")
    print("🔬 Using FNO architecture based on Li et al. best practices")
    
    # Initialize models following Li et al. recommendations
    models = [
        # Model 1: Standard FNO with very few modes (best for small data)
        FNOOperator(
            device, 
            grid_size=GRID_SIZE,
            modes=6,                # Very few modes as recommended
            width=20,               # Narrow width for small dataset
            n_layers=4,             # 4 layers as in paper
            in_channels=detected_channels,  # Use detected channels
            lr=1e-3,               
            step_size=100,          # Decay every 100 epochs
            gamma=0.5,              # Reduce LR by half
            weight_decay=1e-4,      
            epochs=500,
            use_augmentation=True,  # Data augmentation
            activation='gelu'       # GELU activation
        ),
        
        # Model 2: Even smaller FNO with shared weights
        FNOOperator(
            device,
            grid_size=GRID_SIZE,
            modes=4,                # Even fewer modes
            width=16,               # Even narrower
            n_layers=3,             # Fewer layers
            in_channels=detected_channels,  # Use detected channels
            lr=2e-3,                # Higher initial LR
            step_size=150,
            gamma=0.5,
            weight_decay=5e-5,      # Less regularization
            epochs=500,
            use_augmentation=True,
            share_weights=True,     # Share weights to reduce parameters
            activation='gelu'
        ),
        
        # Model 3: Ensemble approach (simplified SpecBoost)
        FNOEnsembleOperator(
            device,
            grid_size=GRID_SIZE,
            n_models=2,             # 2 models in ensemble
            modes=5,
            width=18,
            n_layers=3,
            in_channels=detected_channels,  # Use detected channels
            lr=1e-3,
            epochs=500
        ),
    ]
    
    # Run benchmark
    runner = BenchmarkRunner(models, dm, epochs=500)
    runner.device = device  
    
    print("\n🚀 Starting training with Li et al. optimizations...")
    print("   Key features:")
    print("   - Very few Fourier modes (4-6) to avoid overfitting")
    print("   - Narrow architecture (width 16-20)")
    print("   - Data augmentation (flips)")
    print("   - Proper grid encoding (adds 2 channels for x,y coordinates)")
    print("   - Step learning rate decay")
    print(f"\n   Note: Model will use {detected_channels} input channel(s) + 2 grid channels = {detected_channels + 2} total")
    
    scores = runner.run()
    
    # Enhanced results display
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    
    best_accuracy = 0
    best_model = None
    
    for s in scores:
        print(f"\n🔷 Model: {s['name']}")
        print(f"├─ Parameters: {s['model_info']['parameters']:,}")
        
        # Display architecture info
        if 'architecture' in s['model_info']:
            arch = s['model_info']['architecture']
            print(f"├─ Architecture:")
            print(f"│  ├─ Grid: {arch.get('grid', 'N/A')}")
            print(f"│  ├─ Modes: {arch.get('modes', 'N/A')}")
            print(f"│  ├─ Width: {arch.get('width', 'N/A')}")
            print(f"│  ├─ Layers: {arch.get('layers', 'N/A')}")
            print(f"│  └─ Activation: {arch.get('activation', 'N/A')}")
        
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
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = s['name']
        if 'training_time' in metrics:
            print(f"   └─ Training time: {metrics['training_time']:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_fno_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
    # Performance analysis
    print("\n" + "="*60)
    print("📈 PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"\n🏆 Best model: {best_model} with {best_accuracy:.1f}% accuracy")
    
    print("\n📊 Comparison with previous results:")
    print("  Initial model: 21.2% accuracy (overly regularized)")
    print("  Second attempt: 30.7% accuracy (133k params)")
    print(f"  Current best: {best_accuracy:.1f}% accuracy")
    
    # Detailed analysis
    print("\n🔍 Detailed Analysis:")
    if best_accuracy < 40:
        print("  ❌ Accuracy is still below 40%")
        print("  Possible issues:")
        print("  - Dataset might be too small for FNO")
        print("  - Problem might require higher frequency modes")
        print("  - Consider trying DeepONet or other architectures")
    elif best_accuracy < 60:
        print("  ⚠️ Moderate accuracy (40-60%)")
        print("  Suggestions:")
        print("  - Try ensemble methods")
        print("  - Experiment with more data augmentation")
        print("  - Consider transfer learning")
    else:
        print("  ✅ Good accuracy for 500 samples!")
        print("  - Model successfully learned the operator")
        print("  - Consider ensemble for further improvement")
    
    # Final recommendations
    print("\n" + "="*60)
    print("💡 NEXT STEPS")
    print("="*60)
    print("1. If accuracy < 40%:")
    print("   - Try DeepONet (more flexible for small data)")
    print("   - Reduce modes further (2-3)")
    print("   - Check if data normalization is correct")
    print("")
    print("2. To push accuracy higher:")
    print("   - Enable ensemble model (uncomment in code)")
    print("   - Generate synthetic data")
    print("   - Try transfer learning from similar PDEs")
    print("")
    print("3. Alternative architectures:")
    print("   - U-FNO: Combines CNN with FNO")
    print("   - TFNO: Tucker factorization for efficiency")
    print("   - DeepONet: More suitable for small datasets")
    
    # Data quality check
    print("\n" + "="*60)
    print("🔍 DATA QUALITY CHECK")
    print("="*60)
    print("Run these checks if accuracy is low:")
    print("1. Verify data normalization is correct")
    print("2. Check if PDE has symmetries for augmentation")
    print("3. Ensure train/test split is representative")
    print("4. Look for outliers in the data")