import torch
import random
import numpy as np
import set_up_libs
from data import DataModule
from metrics import BenchmarkRunner
from datetime import datetime

from models.deep_o_net import DeepONetOperator

if __name__ == "__main__":
    # Reproducibilidad
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

    print("\nDataset Configuration:")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Training samples: {TRAIN_SIZE}")
    print(f"  Test samples: {TEST_SIZE}")
    print(f"  Total samples: {TRAIN_SIZE + TEST_SIZE}")
    print("\n🔬 DeepONet Fourier+FiLM (m=96) + Sobolev + SensorDropout")

    models = [
        DeepONetOperator(
            device,
            name="DON_FourierFiLM_Sob_400_chebyshev",
            grid_size=GRID_SIZE,
            n_sensors=400,            # 20x20
            hidden_size=256,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=2.5e-4,
            epochs=1100,
            weight_decay=2e-4,
            sensor_strategy='chebyshev',
            normalize_sensors=True,
            fourier_m=96,
            sensor_dropout_p=0.10,
            grad_loss_weight=0.10
        ),
        DeepONetOperator(
            device,
            name="DON_FourierFiLM_Sob_484_chebyshev",
            grid_size=GRID_SIZE,
            n_sensors=484,            # 22x22
            hidden_size=256,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=2.5e-4,
            epochs=1100,
            weight_decay=2e-4,
            sensor_strategy='chebyshev',
            normalize_sensors=True,
            fourier_m=96,
            sensor_dropout_p=0.10,
            grad_loss_weight=0.10
        ),
        DeepONetOperator(
            device,
            name="DON_FourierFiLM_Sob_400_random",
            grid_size=GRID_SIZE,
            n_sensors=400,
            hidden_size=256,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=2.5e-4,
            epochs=1100,
            weight_decay=2e-4,
            sensor_strategy='random',
            normalize_sensors=True,
            fourier_m=96,
            sensor_dropout_p=0.10,
            grad_loss_weight=0.10
        ),
    ]

    print("\n📋 Model Configurations Summary:")
    print("-" * 80)
    for i, m in enumerate(models, 1):
        print(f"{i}. {m.name} — sensors={m.n_sensors} ({m.sensor_strategy}), "
              f"H={m.hidden_size}, L={m.num_layers}, fourier_m={m.fourier_m}, "
              f"epochs={m.epochs}, wd={m.weight_decay}, p_drop={m.sensor_dropout_p}, "
              f"grad_w={m.grad_loss_weight}")
    print("-" * 80)

    runner = BenchmarkRunner(models, dm, 2000)
    runner.device = device
    scores = runner.run()

    best = max(scores, key=lambda s: s['metrics'].get('accuracy', -1))
    print(f"\n🏆 BEST MODEL: {best['name']}  Acc: {best['metrics'].get('accuracy', 0):.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(scores)
    print(f"💾 Results saved: results_optimized_deeponet_{timestamp}.json")
