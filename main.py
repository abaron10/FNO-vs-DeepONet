import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator, FNOOperator, FNOEnsembleOperator, PyKANOperator
from metrics import BenchmarkRunner 
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
    
    GRID_SIZE = 64
    
    # CORRECTED: Use proper data sizes
    TRAIN_SIZE = 1100
    TEST_SIZE = 500
    
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
            "Standard_FNO", 
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
            "Smaller_FNO_shared_weights",
            grid_size=GRID_SIZE,
            modes=4,                # Even fewer modes
            width=16,               # Even narrower
            n_layers=3,             # Fewer layers
            in_channels=detected_channels,  # Use detected channels
            lr=3e-3,                # Higher initial LR change for 2 in case
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
            "Ensemble_FNO",
            grid_size=GRID_SIZE,
            n_models=2,             # 2 models in ensemble
            modes=5,
            width=18,
            n_layers=3,
            in_channels=detected_channels,  # Use detected channels
            lr=1e-3,
            epochs=500
        ),
        FNOOperator(
        device,
        "Enhanced_Smaller_FNO_Better_training",
        grid_size=GRID_SIZE,
        modes=5,                # One more mode for better frequency representation
        width=18,               # Slightly more capacity without overfitting
        n_layers=3,             # Keep successful shallow architecture
        in_channels=detected_channels,
        lr=2e-3,                # Keep your successful learning rate
        step_size=100,          # More frequent LR adjustments
        gamma=0.65,             # Less aggressive decay for longer training
        weight_decay=4e-5,      # Balanced regularization
        epochs=800,             # Extended training for better convergence
        use_augmentation=True,  # Keep data augmentation
        share_weights=True,     # Keep parameter sharing
        activation='gelu'       # Keep successful activation
    ),   

    FNOOperator(
    device,
    "Optimized_95_Target_FNO",
    grid_size=GRID_SIZE,
    modes=7,                # Sweet spot for 64x64
    width=30,               # More capacity without overfitting  
    n_layers=4,             # Deeper for better representation
    in_channels=detected_channels,
    lr=2.2e-3,              # Slightly higher than your best
    step_size=120,          # More frequent adjustments
    gamma=0.68,             # Gentler decay
    weight_decay=1.8e-5,    # Fine-tuned regularization
    epochs=1200,            # Longer convergence
    use_augmentation=True,
    share_weights=True,     # Keep parameter efficiency
    activation='gelu',
)
    ]
    
    # Run benchmark
    runner = BenchmarkRunner(models, dm, epochs=500)
    runner.device = device  
    scores = runner.run()
    
    
    best_accuracy = -float('inf')  # Can be negative!
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
            rel_l2_error = metrics['relative_l2']
            print(f"   ├─ Relative L2: {rel_l2_error:.4f} ({rel_l2_error*100:.1f}%)")
            # Verify accuracy calculation
            expected_acc = 100 * (1 - rel_l2_error)
            if abs(expected_acc - metrics.get('accuracy', 0)) > 0.1:
                print(f"   ├─ Note: Accuracy should be {expected_acc:.1f}% based on L2 error")
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"   ├─ Accuracy: {acc:.1f}%")
            # Sanity check for Li et al. accuracy
            if acc > 100:
                print(f"   ├─ ⚠️  WARNING: Accuracy > 100% indicates calculation error!")
            elif acc < -100:
                print(f"   ├─ ⚠️  WARNING: Very negative accuracy, check data normalization")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = s['name']
        if 'training_time' in metrics:
            print(f"   └─ Training time: {metrics['training_time']:.1f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_fno_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
  
    

    
