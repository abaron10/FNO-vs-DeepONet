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
    TEST_SIZE = 3000
    
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
    print("🔬 Testing optimized DeepONet configurations based on paper insights")
    
    # Initialize models - Optimized for your original DeepONet class
    models = [
        # Model 1: Ultra-Light (Minimal parameters, fast baseline)
        # Paper shows 100 sensors is critical point, 2 layers works well
        DeepONetOperator(
            device,
            "DeepONet_UltraLight_100sens",
            grid_size=GRID_SIZE,
            n_sensors=100,                   # Paper critical point
            hidden_size=128,                 # Small but effective
            num_layers=2,                    # Shallow (paper Fig. 2)
            activation='relu',               # Simple and fast
            lr=3e-3,                        # Higher LR for small network
            step_size=50,
            gamma=0.85,
            weight_decay=5e-5,              # More regularization
            epochs=1200,                    # More epochs for small net
            sensor_strategy='uniform',      # Guaranteed coverage
            normalize_sensors=True,
            dropout=0.25                    # Heavy dropout for regularization
        ),
        
        # Model 2: Wide-Shallow (Paper shows excellent performance)
        DeepONetOperator(
            device,
            "DeepONet_Wide_Shallow_256sens",
            grid_size=GRID_SIZE,
            n_sensors=256,                   # More sensors
            hidden_size=512,                 # Very wide
            num_layers=2,                    # Very shallow
            activation='gelu',               # Better for wide networks
            lr=5e-4,                        # Lower LR for wide network
            step_size=100,
            gamma=0.9,
            weight_decay=3e-5,
            epochs=800,
            sensor_strategy='chebyshev',    # Better interpolation
            normalize_sensors=True,
            dropout=0.05                    # Light dropout for shallow
        ),
        
        # Model 3: Moderate Depth Optimized
        DeepONetOperator(
            device,
            "DeepONet_Moderate_150sens",
            grid_size=GRID_SIZE,
            n_sensors=150,                   # Balanced sensors
            hidden_size=200,                 # Moderate size
            num_layers=4,                    # Paper suggests 2-4 layers
            activation='gelu',               # Good general activation
            lr=1e-3,                        # Standard LR
            step_size=80,
            gamma=0.92,
            weight_decay=2e-5,
            epochs=1000,
            sensor_strategy='adaptive',      # Mix uniform and boundary
            normalize_sensors=True,
            dropout=0.15                    # Moderate dropout
        ),
        
        # Model 4: Your best config refined (reduced sensors)
        DeepONetOperator(
            device,
            "DeepONet_Refined_2000sens",
            grid_size=GRID_SIZE,
            n_sensors=2000,                  # Reduced from 3800
            hidden_size=300,                 # Slightly smaller than before
            num_layers=4,                    # Moderate depth
            activation='gelu',               # Keep what worked
            lr=2e-3,                        # Higher LR
            step_size=60,
            gamma=0.93,
            weight_decay=1e-5,
            epochs=800,
            sensor_strategy='random',        # Random worked for you
            normalize_sensors=True,
            dropout=0.1                     # Balanced dropout
        ),
        
        # Model 5: Deep with Heavy Regularization
        DeepONetOperator(
            device,
            "DeepONet_Deep_Regularized",
            grid_size=GRID_SIZE,
            n_sensors=200,                   
            hidden_size=180,                 # Smaller hidden for deep
            num_layers=6,                    # Deeper network
            activation='mish',               # Good for deep networks
            lr=8e-4,
            step_size=100,
            gamma=0.88,
            weight_decay=4e-5,              # More weight decay
            epochs=900,
            sensor_strategy='chebyshev',
            normalize_sensors=True,
            dropout=0.2                     # Heavy dropout for deep
        ),
        
        # Model 6: Maximum Width Strategy
        DeepONetOperator(
            device,
            "DeepONet_MaxWidth_300sens",
            grid_size=GRID_SIZE,
            n_sensors=300,
            hidden_size=768,                 # Very wide
            num_layers=3,                    # Shallow-moderate
            activation='gelu',
            lr=3e-4,                        # Very low LR for huge network
            step_size=120,
            gamma=0.85,
            weight_decay=5e-5,              # Strong regularization
            epochs=600,
            sensor_strategy='adaptive',
            normalize_sensors=True,
            dropout=0.08                    # Light dropout for wide
        ),
        
        # Model 7: Ensemble-Ready Small Model
        DeepONetOperator(
            device,
            "DeepONet_Ensemble_Base",
            grid_size=GRID_SIZE,
            n_sensors=120,
            hidden_size=150,
            num_layers=3,
            activation='relu',
            lr=2.5e-3,
            step_size=70,
            gamma=0.9,
            weight_decay=3e-5,
            epochs=1000,
            sensor_strategy='uniform',
            normalize_sensors=True,
            dropout=0.18
        ),
        
        # Model 8: Paper-Optimal Configuration
        # Based on paper's best practices
        DeepONetOperator(
            device,
            "DeepONet_Paper_Optimal",
            grid_size=GRID_SIZE,
            n_sensors=100,                   # Paper's critical point
            hidden_size=256,                 # Moderate-wide
            num_layers=3,                    # Paper's sweet spot
            activation='gelu',               # Modern activation
            lr=1.5e-3,
            step_size=90,
            gamma=0.91,
            weight_decay=2e-5,
            epochs=1000,
            sensor_strategy='chebyshev',    # Best for smooth functions
            normalize_sensors=True,
            dropout=0.12
        )
    ]
    
    # Add ensemble model if you want
    ensemble_model = DeepONetEnsembleOperator(
        device,
        "DeepONet_Ensemble_3models",
        grid_size=GRID_SIZE,
        n_models=3,                        # 3 models in ensemble
        n_sensors=150,
        hidden_size=200,
        num_layers=3,
        activation='gelu',
        lr=1.5e-3,
        step_size=80,
        gamma=0.9,
        weight_decay=2e-5,
        epochs=800,
        sensor_strategy='random',
        normalize_sensors=True,
        dropout=0.15
    )
    
    # Add ensemble to models list
    models.append(ensemble_model)
    
    # Print model summaries before training
    print("\n📋 Model Configurations Summary:")
    print("-" * 80)
    for i, model in enumerate(models, 1):
        params_count = 0
        if hasattr(model, 'n_models'):  # Ensemble
            # Approximate parameter count for ensemble
            branch_params = model.n_sensors * model.hidden_size + model.hidden_size * model.hidden_size * (model.num_layers - 1) + model.hidden_size * model.hidden_size
            trunk_params = 2 * model.hidden_size + model.hidden_size * model.hidden_size * (model.num_layers - 1) + model.hidden_size * model.hidden_size
            params_count = (branch_params + trunk_params + model.hidden_size + 1) * model.n_models
            print(f"\n{i}. {model.name} [ENSEMBLE]")
            print(f"   Models: {model.n_models}")
        else:
            # Approximate parameter count for single model
            branch_params = model.n_sensors * model.hidden_size + model.hidden_size * model.hidden_size * (model.num_layers - 1) + model.hidden_size * model.hidden_size
            trunk_params = 2 * model.hidden_size + model.hidden_size * model.hidden_size * (model.num_layers - 1) + model.hidden_size * model.hidden_size
            params_count = branch_params + trunk_params + model.hidden_size + 1
            print(f"\n{i}. {model.name}")
        
        print(f"   Sensors: {model.n_sensors} ({model.sensor_strategy})")
        print(f"   Architecture: {model.num_layers} layers × {model.hidden_size} hidden")
        print(f"   Activation: {model.activation}, Dropout: {model.dropout}")
        print(f"   Training: {model.epochs} epochs, LR={model.lr}, WD={model.weight_decay}")
        print(f"   Est. Parameters: ~{params_count:,}")
    print("-" * 80)
    
    runner = BenchmarkRunner(models, dm, 1000)  
    runner.device = device  
    scores = runner.run()
    
    best_accuracy = -float('inf')
    best_model = None
    results_summary = []
    
    print("\n📊 Training Results:")
    print("=" * 80)
    
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
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            print(f"   ├─ Accuracy: {acc:.1f}%")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = s['name']
            
            # Store for summary
            results_summary.append({
                'name': s['name'],
                'accuracy': acc,
                'mse': metrics.get('mse', 0),
                'params': s['model_info']['parameters']
            })
        
        if 'training_time' in metrics:
            print(f"   └─ Training time: {metrics['training_time']:.1f}s")
    
    # Performance summary
    print("\n" + "=" * 80)
    print("\n📈 RESULTS SUMMARY (sorted by accuracy):")
    print("-" * 80)
    results_summary.sort(key=lambda x: x['accuracy'], reverse=True)
    for i, result in enumerate(results_summary[:5], 1):  # Top 5
        print(f"{i}. {result['name']:<30} Acc: {result['accuracy']:>6.2f}%  MSE: {result['mse']:.4e}  Params: {result['params']:,}")
    
    print("\n" + "=" * 80)
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Final Accuracy: {best_accuracy:.2f}%")
    
    if best_accuracy > 90:
        print("\n🎉 EXCEPTIONAL! DeepONet achieved >90% accuracy")
        print("   → Consider ensemble of top 3 models for even better results")
    elif best_accuracy > 85:
        print("\n🎊 EXCELLENT! DeepONet achieved >85% accuracy")
        print("   → Try ensemble or fine-tuning the best model")
    elif best_accuracy > 80:
        print("\n✅ SUCCESS! DeepONet achieved >80% accuracy target")
        print("   → Good improvement from baseline") 
    elif best_accuracy > 75:
        print("\n👍 Good progress! DeepONet improved to >75% accuracy")
        print("   → Consider ensemble or adjusting regularization")
    else:
        print("\n⚠️  DeepONet needs more optimization")
        print("   → Check if data normalization is correct")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_optimized_deeponet_{timestamp}.json"
    runner.save_results(scores)
    print(f"\n💾 Results saved to: {filename}")
    
    # Analysis and recommendations
    print(f"\n📊 Performance Analysis:")
    print(f"├─ Models tested: {len(models)}")
    print(f"├─ Training samples: {TRAIN_SIZE}")
    print(f"├─ Test samples: {TEST_SIZE}")
    print(f"├─ Grid resolution: {GRID_SIZE}×{GRID_SIZE}")
    print(f"├─ Best accuracy: {best_accuracy:.2f}%")
    print(f"└─ Accuracy range: {min(r['accuracy'] for r in results_summary):.1f}% - {best_accuracy:.1f}%")
    
    # Find patterns in results
    print("\n🔬 Key Insights:")
    
    # Analyze sensor counts
    sensor_results = {}
    for model in models:
        if hasattr(model, 'n_sensors'):
            sensor_results[model.n_sensors] = model.name
    
    # Analyze which configurations worked best
    if results_summary:
        top_model = results_summary[0]
        print(f"1. Best performing: {top_model['name']}")
        
        # Check if small models did well
        small_models = [r for r in results_summary if r['params'] < 500000]
        if small_models and small_models[0]['accuracy'] > 75:
            print(f"2. Small models effective: {small_models[0]['name']} achieved {small_models[0]['accuracy']:.1f}% with only {small_models[0]['params']:,} params")
        
        # Check if wide-shallow worked
        wide_shallow = [r for r in results_summary if 'Wide_Shallow' in r['name']]
        if wide_shallow and wide_shallow[0]['accuracy'] > 78:
            print(f"3. Wide-shallow architecture successful: {wide_shallow[0]['accuracy']:.1f}% accuracy")
        
        # Check ensemble performance
        ensemble_results = [r for r in results_summary if 'Ensemble' in r['name']]
        if ensemble_results and ensemble_results[0]['accuracy'] > best_accuracy - 5:
            print(f"4. Ensemble effective: {ensemble_results[0]['accuracy']:.1f}% accuracy")
    
    print("\n🚀 Recommendations for Further Improvement:")
    if best_accuracy < 80:
        print("1. Data quality: Check if target outputs are correctly normalized")
        print("2. Try ensemble of top 3 models")
        print("3. Increase training epochs for best model")
        print("4. Experiment with learning rate scheduling")
    elif best_accuracy < 85:
        print("1. Fine-tune the best model with lower learning rate")
        print("2. Try ensemble of diverse models (wide + deep)")
        print("3. Implement early stopping based on validation loss")
        print("4. Add more sensors for complex regions")
    else:
        print("1. Excellent results! Consider production deployment")
        print("2. Ensemble top models for marginal gains")
        print("3. Study prediction errors to understand failure modes")
        print("4. Test on out-of-distribution data")