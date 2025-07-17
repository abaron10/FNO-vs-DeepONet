import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import OptimizedDeepONetOperator, DeepONetOperator, FNOOperator
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
    TRAIN_SIZE = 1100
    TEST_SIZE = 500
    
    dm = DataModule(grid=GRID_SIZE, n_train=TRAIN_SIZE, n_test=TEST_SIZE)
    dm.setup()
    
    print(f"\nDataset Configuration:")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Training samples: {TRAIN_SIZE}")
    print(f"  Test samples: {TEST_SIZE}")
    print(f"  Total grid points: {GRID_SIZE**2} = {GRID_SIZE**2}")
    print("\n🔬 Testing DeepONet following Lu et al. (2019) best practices")
    
    # Initialize models based on paper recommendations
    models = [
        # Model 1: Paper default configuration
        OptimizedDeepONetOperator(
            device,
            "Paper_Default_Config",
            grid_size=GRID_SIZE,
            n_sensors=100,              # Paper shows 100 is often sufficient
            sensor_strategy='chebyshev', # Better interpolation
            p=100,                      # Number of basis functions
            activation='tanh',          # Paper uses tanh
            lr=1e-3,                    # Paper default
            epochs=50000,               # Paper uses 50k iterations
            normalize_inputs=True,      # Normalize sensor inputs
            normalize_outputs=False     # Don't normalize outputs
        ),
        
        # Model 2: Optimized for accuracy
        OptimizedDeepONetOperator(
            device,
            "High_Accuracy_Config",
            grid_size=GRID_SIZE,
            n_sensors=150,              # Slightly more sensors
            sensor_strategy='chebyshev',
            p=150,                      # More basis functions
            branch_layers=[150, 60, 60, 150],  # Wider networks
            trunk_layers=[2, 60, 60, 150],
            activation='tanh',
            lr=1e-3,
            epochs=80000,               # More training
            sensor_noise=0.01,          # Small noise for robustness
            normalize_inputs=True,
            normalize_outputs=True      # Full normalization
        ),
        
        # Model 3: LHS sampling for better coverage
        OptimizedDeepONetOperator(
            device,
            "LHS_Sampling",
            grid_size=GRID_SIZE,
            n_sensors=120,
            sensor_strategy='lhs',      # Latin Hypercube Sampling
            p=120,
            branch_layers=[120, 50, 50, 120],
            trunk_layers=[2, 50, 50, 120],
            activation='tanh',
            lr=1e-3,
            epochs=60000,
            normalize_inputs=True,
            normalize_outputs=False
        ),
        
        # Model 4: Different activation (ReLU)
        OptimizedDeepONetOperator(
            device,
            "ReLU_Activation",
            grid_size=GRID_SIZE,
            n_sensors=100,
            sensor_strategy='chebyshev',
            p=100,
            activation='relu',          # Test ReLU
            lr=1e-3,
            epochs=50000,
            normalize_inputs=True,
            normalize_outputs=False
        ),
        
        # Model 5: Minimal sensors (efficiency test)
        OptimizedDeepONetOperator(
            device,
            "Minimal_Sensors",
            grid_size=GRID_SIZE,
            n_sensors=50,               # Only 50 sensors (~1.2% of grid)
            sensor_strategy='chebyshev',
            p=80,
            branch_layers=[50, 40, 40, 80],
            trunk_layers=[2, 40, 40, 80],
            activation='tanh',
            lr=1.5e-3,                  # Slightly higher LR
            epochs=40000,
            normalize_inputs=True,
            normalize_outputs=False
        ),
        
        # Model 6: Deeper architecture test
        OptimizedDeepONetOperator(
            device,
            "Deeper_Architecture",
            grid_size=GRID_SIZE,
            n_sensors=100,
            sensor_strategy='chebyshev',
            p=100,
            branch_layers=[100, 50, 50, 50, 100],  # 4 hidden layers
            trunk_layers=[2, 50, 50, 50, 100],
            activation='tanh',
            lr=8e-4,                    # Lower LR for deeper network
            epochs=60000,
            normalize_inputs=True,
            normalize_outputs=False
        ),
        
        # Model 7: Large basis with uniform sampling
        OptimizedDeepONetOperator(
            device,
            "Large_Basis_Uniform",
            grid_size=GRID_SIZE,
            n_sensors=144,              # 12x12 uniform grid
            sensor_strategy='uniform',
            p=200,                      # Large number of basis functions
            branch_layers=[144, 80, 80, 200],
            trunk_layers=[2, 80, 80, 200],
            activation='tanh',
            lr=8e-4,
            epochs=70000,
            normalize_inputs=True,
            normalize_outputs=True
        ),
        
        # Model 8: Original implementation for comparison
        DeepONetOperator(
            device,
            "Original_Best_Config",
            grid_size=GRID_SIZE,
            n_sensors=400,              # More reasonable sensor count
            hidden_size=100,            # Moderate size
            num_layers=4,               # Moderate depth
            activation='gelu',
            lr=1e-3,                    # Standard LR
            step_size=100,
            gamma=0.5,
            weight_decay=1e-5,
            epochs=50000,               # Match paper iterations
            sensor_strategy='chebyshev',
            normalize_sensors=True
        )
    ]
    
    # Run benchmark
    runner = BenchmarkRunner(models, dm)
    runner.device = device
    
    # Override epochs to use model-specific epochs
    for model in models:
        model.max_epochs = model.epochs
    
    scores = runner.run()
    
    # Analysis and reporting
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    best_accuracy = -float('inf')
    best_model = None
    results_summary = []
    
    for s in scores:
        print(f"\n🔷 Model: {s['name']}")
        print(f"├─ Parameters: {s['model_info']['parameters']:,}")
        
        # Display architecture info
        if 'architecture' in s['model_info']:
            arch = s['model_info']['architecture']
            print(f"├─ Architecture:")
            print(f"│  ├─ Type: {arch.get('type', 'N/A')}")
            print(f"│  ├─ Grid: {arch.get('grid', 'N/A')}")
            print(f"│  ├─ Sensors: {arch.get('n_sensors', 'N/A')} ({arch.get('sensor_coverage', 'N/A')})")
            print(f"│  ├─ Basis functions (p): {arch.get('p_basis', arch.get('hidden_size', 'N/A'))}")
            print(f"│  ├─ Activation: {arch.get('activation', 'N/A')}")
            print(f"│  └─ Sensor Strategy: {arch.get('sensor_strategy', 'N/A')}")
        
        print(f"└─ Metrics:")
        metrics = s['metrics']
        
        # Format metrics
        if 'relative_l2' in metrics:
            rel_l2_error = metrics['relative_l2']
            print(f"   ├─ Relative L2 Error: {rel_l2_error:.4f} ({rel_l2_error*100:.1f}%)")
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"   ├─ Accuracy: {acc:.1f}%")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = s['name']
        if 'mae' in metrics:
            print(f"   ├─ MAE: {metrics['mae']:.4e}")
        if 'mse' in metrics:
            print(f"   ├─ MSE: {metrics['mse']:.4e}")
        if 'training_time' in metrics:
            print(f"   └─ Training time: {metrics['training_time']:.1f}s")
            
        # Store for summary
        results_summary.append({
            'name': s['name'],
            'accuracy': metrics.get('accuracy', 0),
            'params': s['model_info']['parameters'],
            'sensors': s['model_info']['architecture'].get('n_sensors', 0),
            'time': metrics.get('training_time', 0)
        })
    
    # Performance summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"\n🏆 BEST MODEL: {best_model} with {best_accuracy:.1f}% accuracy")
    
    # Sort by accuracy
    results_summary.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\nTop 5 Models by Accuracy:")
    print(f"{'Rank':<5} {'Model':<30} {'Accuracy':<10} {'Sensors':<10} {'Parameters':<15} {'Time (s)':<10}")
    print("-" * 90)
    for i, r in enumerate(results_summary[:5]):
        print(f"{i+1:<5} {r['name']:<30} {r['accuracy']:<10.1f} {r['sensors']:<10} {r['params']:<15,} {r['time']:<10.1f}")
    
    # Efficiency analysis
    print("\nEfficiency Analysis (Accuracy per Parameter):")
    efficiency = [(r['name'], r['accuracy'] / (r['params'] / 1000), r['accuracy'], r['params']) 
                  for r in results_summary if r['params'] > 0]
    efficiency.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<30} {'Acc/1K params':<15} {'Accuracy':<10} {'Parameters':<15}")
    print("-" * 70)
    for name, eff, acc, params in efficiency[:3]:
        print(f"{name:<30} {eff:<15.2f} {acc:<10.1f} {params:<15,}")
    
    # Performance insights
    print(f"\n📊 Performance Insights:")
    if best_accuracy > 90:
        print("🎉 Excellent! DeepONet achieved >90% accuracy")
    elif best_accuracy > 85:
        print("✅ Great! DeepONet achieved >85% accuracy") 
    elif best_accuracy > 80:
        print("👍 Good! DeepONet achieved >80% accuracy")
    else:
        print("⚠️  Performance below expectations. Consider:")
        print("   - Increasing training iterations")
        print("   - Adjusting sensor placement strategy")
        print("   - Tuning normalization settings")
        print("   - Checking data quality")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_deeponet_optimized_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
    # Paper comparison
    print(f"\n📚 Comparison with Lu et al. (2019):")
    print("├─ Paper reports exponential convergence for small datasets")
    print("├─ 100 sensors typically sufficient for smooth functions")
    print("├─ Tanh activation often outperforms ReLU")
    print("└─ Normalization significantly improves stability")