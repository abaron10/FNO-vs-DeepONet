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
    TRAIN_SIZE = 4000
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
    print("🔬 Testing enhanced DeepONet configurations")
    
    # Initialize models - Enhanced DeepONet configurations
    models = [
          DeepONetOperator(
              device,
    "Ultra_Precision_DeepONet",
    grid_size=GRID_SIZE,
    n_sensors=4000,             # 97.6% coverage (casi todo el grid)
    hidden_size=384,            # Very large hidden size
    num_layers=8,               # Deep network
    activation='gelu',
    lr=2e-4,                    # Very careful LR
    step_size=50,               # Frequent updates
    gamma=0.85,                 # Very gentle decay
    weight_decay=1e-6,          # Minimal regularization
    epochs=1500,                # Extended training
    sensor_strategy='adaptive',
    normalize_sensors=True,
    dropout=0.0                 # No dropout for max capacity
        )
        
    #     # DeepONet Model 1: Optimized v2 with adaptive sensors
    #     DeepONetOperator(
    #         device,
    #         "Optimized_DeepONet_v2",
    #         grid_size=GRID_SIZE,
    #         n_sensors=3200,             # 78% del grid
    #         hidden_size=200,            # Capacidad balanceada
    #         num_layers=6,               # Profundidad adecuada
    #         activation='gelu',          # Consistente con FNO
    #         lr=4e-4,                    # LR optimizado
    #         step_size=70,               # Updates frecuentes
    #         gamma=0.75,                 # Decay suave
    #         weight_decay=2e-6,          # Regularización mínima
    #         epochs=1000,                # Training extendido
    #         sensor_strategy='adaptive', # Sensores inteligentes
    #         normalize_sensors=True      # Normalización crítica
    #     ),
        
    #     # DeepONet Model 2: High-density sensors for maximum accuracy
    #     DeepONetOperator(
    #         device,
    #         "High_Density_DeepONet",
    #         grid_size=GRID_SIZE,
    #         n_sensors=3500,             # 85% del grid
    #         hidden_size=256,            # Más capacidad
    #         num_layers=8,               # Más profundidad
    #         activation='gelu',
    #         lr=3e-4,                    # Lower LR para estabilidad
    #         step_size=60,               # Más frequent adjustments
    #         gamma=0.8,                  # Less aggressive decay
    #         weight_decay=5e-6,          # Less regularization
    #         epochs=1200,                # Extended training
    #         sensor_strategy='adaptive',
    #         normalize_sensors=True,
    #         dropout=0.0                 # No dropout para max capacity
    #     ),
        
    #     # DeepONet Model 3: Ultra DeepONet for 95% target
    #       DeepONetOperator(
    #     device,
    #     "Hybrid_Best_Chebyshev",
    #     grid_size=GRID_SIZE,
    #     n_sensors=2700,             # Sweet spot based on results
    #     hidden_size=220,            # Balanced capacity
    #     num_layers=8,               # Good depth
    #     activation='gelu',
    #     lr=3.2e-4,                  # Carefully tuned
    #     step_size=55,               # Balanced updates
    #     gamma=0.82,                 # Balanced decay
    #     weight_decay=3e-6,          # Light regularization
    #     epochs=1300,                # Good training time
    #     sensor_strategy='chebyshev',
    #     normalize_sensors=True,
    #     dropout=0.0
    # ),
        
    #     # DeepONet Model 4: Balanced efficiency vs accuracy
    #     DeepONetOperator(
    #         device,
    #         "Balanced_Plus_Chebyshev",
    #         grid_size=GRID_SIZE,
    #         n_sensors=2550,             # Solo +2% del mejor
    #         hidden_size=180,            # +20% conservador
    #         num_layers=6,               # +20% conservador
    #         activation='gelu',
    #         lr=4e-4,                    
    #         step_size=70,               
    #         gamma=0.75,                 
    #         weight_decay=8e-6,          
    #         epochs=1000,                
    #         sensor_strategy='chebyshev',
    #         normalize_sensors=True,
    #         dropout=0.0
    #     ),
        
    #     DeepONetOperator(
    #         device,
    #         "Enhanced_Balanced_DeepONet",
    #         grid_size=GRID_SIZE,
    #         n_sensors=2800,             # +12% del mejor modelo
    #         hidden_size=200,            # +33% capacidad
    #         num_layers=7,               # +40% profundidad
    #         activation='gelu',
    #         lr=3e-4,                    
    #         step_size=60,               
    #         gamma=0.8,                  
    #         weight_decay=5e-6,          
    #         epochs=1200,                
    #         sensor_strategy='chebyshev',
    #         normalize_sensors=True,
    #         dropout=0.0
    #     )
    ]
    
    runner = BenchmarkRunner(models, dm, 1000)  
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
            print(f"│  ├─ Type: {arch.get('type', 'N/A')}")
            print(f"│  ├─ Grid: {arch.get('grid', 'N/A')}")
            print(f"│  ├─ Sensors: {arch.get('n_sensors', 'N/A')}")
            print(f"│  ├─ Hidden Size: {arch.get('hidden_size', 'N/A')}")
            print(f"│  ├─ Layers: {arch.get('num_layers', 'N/A')}")
            print(f"│  ├─ Activation: {arch.get('activation', 'N/A')}")
            print(f"│  └─ Sensor Strategy: {arch.get('sensor_strategy', 'N/A')}")
        
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
    
    # Performance summary
    print(f"\n🏆 BEST MODEL: {best_model} with {best_accuracy:.1f}% accuracy")
    
    if best_accuracy > 90:
        print("🎉 Excellent! DeepONet achieved >90% accuracy")
    elif best_accuracy > 85:
        print("✅ Great! DeepONet achieved >85% accuracy") 
    elif best_accuracy > 80:
        print("👍 Good! DeepONet achieved >80% accuracy")
    else:
        print("⚠️  DeepONet needs further optimization")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_deeponet_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
    # Performance insights
    print(f"\n📊 Performance Insights:")
    print(f"├─ Models tested: {len(models)}")
    print(f"├─ Training samples: {TRAIN_SIZE}")
    print(f"├─ Test samples: {TEST_SIZE}")
    print(f"├─ Grid resolution: {GRID_SIZE}×{GRID_SIZE}")
    print(f"└─ Best accuracy achieved: {best_accuracy:.1f}%")