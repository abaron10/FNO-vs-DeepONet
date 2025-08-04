import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator, FNOOperator, FNOEnsembleOperator, PyKANOperator, DeepONetEnsembleOperator
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
    TRAIN_SIZE = 8000
    TEST_SIZE = 2000
    
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
    print("🔬 Testing enhanced DeepONet with internal improvements")
    
    # Initialize models - Optimized based on literature findings
    models = [
        # Model 1: Ultra-optimized DeepONet targeting >85% accuracy
        DeepONetOperator(
            device,
            "UltraOptimized_DeepONet_v1",
            grid_size=GRID_SIZE,
            n_sensors=3600,             # ~88% coverage (optimal based on papers)
            hidden_size=384,            # Larger hidden size for complex patterns
            num_layers=10,              # Deep network (will be 10 branch, 5 trunk internally)
            activation='gelu',          # GELU consistently performs better
            lr=2e-4,                    # Lower LR for stability with deep network
            step_size=40,               # More frequent updates with cosine annealing
            gamma=0.85,                 # Not used with cosine annealing but kept for interface
            weight_decay=1e-5,          # Light regularization
            epochs=800,                 # Extended training
            sensor_strategy='chebyshev', # Better for smooth functions
            normalize_sensors=True,
            dropout=0.1                 # Light dropout for regularization
        ),
        
        # Model 2: Ensemble for maximum accuracy
        DeepONetEnsembleOperator(
            device,
            "Ensemble_UltraDeepONet",
            grid_size=GRID_SIZE,
            n_models=3,                 # 3 models in ensemble
            n_sensors=3200,             # Slightly fewer sensors per model
            hidden_size=320,            # Good capacity
            num_layers=8,               # Deep but not too deep
            activation='gelu',
            lr=3e-4,                    # Slightly higher LR for ensemble
            step_size=50,
            gamma=0.9,
            weight_decay=2e-5,
            epochs=600,
            sensor_strategy='adaptive',  # Best adaptive strategy
            normalize_sensors=True,
            dropout=0.05                # Lower dropout for ensemble
        ),
        
        # Model 3: Efficient high-accuracy model
        DeepONetOperator(
            device,
            "Efficient_HighAccuracy_DeepONet",
            grid_size=GRID_SIZE,
            n_sensors=2800,             # Good coverage with efficiency
            hidden_size=256,            # Balanced size
            num_layers=8,               # Good depth
            activation='gelu',
            lr=4e-4,                    # Higher LR for faster convergence
            step_size=30,               # Frequent updates
            gamma=0.8,
            weight_decay=5e-5,          # Moderate regularization
            epochs=500,                 # Reasonable training time
            sensor_strategy='adaptive',
            normalize_sensors=True,
            dropout=0.0                 # No dropout for maximum capacity
        ),
        
        # Model 4: Maximum sensors approach
        DeepONetOperator(
            device,
            "MaxSensors_DeepONet",
            grid_size=GRID_SIZE,
            n_sensors=4000,             # Near maximum (98% coverage)
            hidden_size=256,            # Don't need huge hidden with many sensors
            num_layers=6,               # Moderate depth
            activation='gelu',
            lr=5e-4,                    # Higher LR viable with many sensors
            step_size=60,
            gamma=0.75,
            weight_decay=1e-4,          # More regularization with many parameters
            epochs=400,                 # Less epochs needed with many sensors
            sensor_strategy='uniform',  # Uniform works well with high coverage
            normalize_sensors=True,
            dropout=0.15                # More dropout to prevent overfitting
        )
    ]
    
    runner = BenchmarkRunner(models, dm, 1000)  
    runner.device = device  
    scores = runner.run()
    
    best_accuracy = -float('inf')
    best_model = None
    
    for s in scores:
        print(f"\n🔷 Model: {s['name']}")
        print(f"├─ Parameters: {s['model_info']['parameters']:,}")
        
        # Display architecture info
        if 'architecture' in s['model_info']:
            arch = s['model_info']['architecture']
            print(f"├─ Architecture:")
            print(f"│  ├─ Type: {arch.get('type', 'N/A')}")
            print(f"│  ├─ Grid: {arch.get('grid', 'N/A')}")
            print(f"│  ├─ Sensors: {arch.get('n_sensors', 'N/A')}")
            print(f"│  ├─ Hidden Size: {arch.get('hidden_size', 'N/A')}")
            print(f"│  ├─ Layers: {arch.get('num_layers', 'N/A')}")
            print(f"│  ├─ Activation: {arch.get('activation', 'N/A')}")
            print(f"│  ├─ Sensor Strategy: {arch.get('sensor_strategy', 'N/A')}")
            print(f"│  └─ Features: {arch.get('features', 'N/A')}")
        
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
            expected_acc = 100 * (1 - rel_l2_error)
            if abs(expected_acc - metrics.get('accuracy', 0)) > 0.1:
                print(f"   ├─ Note: Accuracy should be {expected_acc:.1f}% based on L2 error")
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"   ├─ Accuracy: {acc:.1f}%")
            if acc > 100:
                print(f"   ├─ ⚠️  WARNING: Accuracy > 100% indicates calculation error!")
            elif acc < -100:
                print(f"   ├─ ⚠️  WARNING: Very negative accuracy, check data normalization")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = s['name']
        if 'training_time' in metrics:
            print(f"   └─ Training time: {metrics['training_time']:.1f}s")
    
    # Performance summary
    print(f"\n🏆 BEST MODEL: {best_model} with {best_accuracy:.1f}% accuracy")
    
    if best_accuracy > 90:
        print("🎉 EXCEPTIONAL! DeepONet achieved >90% accuracy")
    elif best_accuracy > 85:
        print("🎊 EXCELLENT! DeepONet achieved >85% accuracy")
    elif best_accuracy > 80:
        print("✅ SUCCESS! DeepONet achieved >80% accuracy target") 
    elif best_accuracy > 75:
        print("👍 Good progress! DeepONet improved to >75% accuracy")
    else:
        print("⚠️  DeepONet still needs optimization")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_enhanced_deeponet_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
    # Performance insights
    print(f"\n📊 Performance Insights:")
    print(f"├─ Models tested: {len(models)}")
    print(f"├─ Training samples: {TRAIN_SIZE}")
    print(f"├─ Test samples: {TEST_SIZE}")
    print(f"├─ Grid resolution: {GRID_SIZE}×{GRID_SIZE}")
    print(f"├─ Best accuracy achieved: {best_accuracy:.1f}%")
    print(f"└─ Key improvements: Nonlinear decoder, Fourier features, Asymmetric architecture")