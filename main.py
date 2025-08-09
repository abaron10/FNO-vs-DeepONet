import torch
import random
import numpy as np
import set_up_libs  # si ya lo usas
from data import DataModule
from metrics import BenchmarkRunner
from datetime import datetime

# Importa el operador del archivo único
from deep_o_net import DeepONetOperator

if __name__ == "__main__":
    # Seeds
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    GRID_SIZE = 64
    TRAIN_SIZE = 9000
    TEST_SIZE = 3000

    dm = DataModule(grid=GRID_SIZE, n_train=TRAIN_SIZE, n_test=TEST_SIZE)
    dm.setup()

    # Info dataset
    try:
        sample_batch = next(iter(dm.train))
        detected_channels = sample_batch["x"].shape[1]
    except Exception:
        detected_channels = 1

    print(f"\nDataset Configuration:")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Training samples: {TRAIN_SIZE}")
    print(f"  Test samples: {TEST_SIZE}")
    print(f"  Total samples: {TRAIN_SIZE + TEST_SIZE}")
    print(f"  Input channels detected: {detected_channels}")
    print("\n⚠️  Using test set for validation (temporary)")
    print("🔬 DeepONet Fourier+FiLM configs")

    # Modelos (solo cambia estrategia de sensores / regularización)
    models = [
        DeepONetOperator(
            device,
            name="DON_FourierFiLM_256_chebyshev",
            grid_size=GRID_SIZE,
            n_sensors=256,
            hidden_size=256,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=3e-4,
            epochs=600,
            weight_decay=1e-4,
            sensor_strategy='chebyshev',
            normalize_sensors=True
        ),
        DeepONetOperator(
            device,
            name="DON_FourierFiLM_256_random",
            grid_size=GRID_SIZE,
            n_sensors=256,
            hidden_size=256,
            num_layers=4,
            activation='gelu',
            dropout=0.10,        # más regularización para random
            lr=2.5e-4,
            epochs=600,
            weight_decay=1.5e-4,
            sensor_strategy='random',
            normalize_sensors=True
        ),
        DeepONetOperator(
            device,
            name="DON_FourierFiLM_256_uniform",
            grid_size=GRID_SIZE,
            n_sensors=256,
            hidden_size=320,     # un poco más ancho para uniform
            num_layers=4,
            activation='gelu',
            dropout=0.08,
            lr=2e-4,
            epochs=600,
            weight_decay=2e-4,
            sensor_strategy='uniform',
            normalize_sensors=True
        ),
    ]

    print("\n📋 Model Configurations Summary:")
    print("-" * 80)
    for i, model in enumerate(models, 1):
        # estimación rápida de parámetros (opcional)
        branch = model.n_sensors * model.hidden_size + (model.num_layers - 1) * (model.hidden_size ** 2) + model.hidden_size ** 2
        trunk = (2 * 32) * model.hidden_size + (model.num_layers - 1) * (model.hidden_size ** 2) + model.hidden_size ** 2
        params_est = branch + trunk + model.hidden_size + 1
        print(f"\n{i}. {model.name}")
        print(f"   Sensors: {model.n_sensors} ({model.sensor_strategy})")
        print(f"   Architecture: {model.num_layers} layers × {model.hidden_size} hidden")
        print(f"   Activation: {model.activation}, Dropout: {model.dropout}")
        print(f"   Training: {model.epochs} epochs, LR={model.lr}, WD={model.weight_decay}")
        print(f"   Est. Parameters: ~{params_est:,}")
    print("-" * 80)

    runner = BenchmarkRunner(models, dm, 2000)
    runner.device = device
    scores = runner.run()

    best_accuracy = -float('inf'); best_model = None; results_summary = []

    print("\n📊 Training Results:")
    print("=" * 80)
    for s in scores:
        print(f"\n🔷 Model: {s['name']}")
        print(f"├─ Parameters: {s['model_info']['parameters']:,}")
        if 'architecture' in s['model_info']:
            arch = s['model_info']['architecture']
            print(f"├─ Architecture:")
            print(f"│  ├─ Type: {arch.get('type')}")
            print(f"│  ├─ Grid: {arch.get('grid')}")
            print(f"│  ├─ Sensors: {arch.get('n_sensors')}")
            print(f"│  ├─ Hidden Size: {arch.get('hidden_size')}")
            print(f"│  ├─ Layers: {arch.get('num_layers')}")
            print(f"│  ├─ Activation: {arch.get('activation')}")
            print(f"│  └─ Sensor Strategy: {arch.get('sensor_strategy')}")
        metrics = s['metrics']
        if 'mae' in metrics: print(f"   ├─ MAE: {metrics['mae']:.4e}")
        if 'mse' in metrics: print(f"   ├─ MSE: {metrics['mse']:.4e}")
        if 'relative_l2' in metrics:
            r = metrics['relative_l2']; print(f"   ├─ Relative L2: {r:.4f} ({r*100:.1f}%)")
        if 'accuracy' in metrics:
            acc = metrics['accuracy']; print(f"   ├─ Accuracy: {acc:.1f}%")
            if acc > best_accuracy: best_accuracy, best_model = acc, s['name']
        if 'training_time' in metrics: print(f"   └─ Training time: {metrics['training_time']:.1f}s")

    print("\n" + "=" * 80)
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Final Accuracy: {best_accuracy:.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(scores)
    print(f"\n💾 Results saved: results_optimized_deeponet_{timestamp}.json")
