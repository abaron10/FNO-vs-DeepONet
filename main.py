import torch
import random
import math
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
    print("\n🔬 DeepONet Multi-Scale RFF + Sobolev + Boundary Weight + SWA")

    # Config base (best-bet) para superar 80%
    models = [
        DeepONetOperator(
            device,
            name="DON_MSFF_SWA_576_chebyshev",
            grid_size=GRID_SIZE,
            n_sensors=576,          # 24x24
            hidden_size=320,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=2.5e-4,
            epochs=1500,
            weight_decay=1.5e-4,
            sensor_strategy='chebyshev',
            normalize_sensors=True,
            m_per_scale=64,                     # 3 escalas → out_dim = 3*2*64 = 384
            ff_scales=(math.pi, 4*math.pi, 16*math.pi),
            sensor_dropout_p0=0.10,             # annealed → 0
            fourier_dropout_p0=0.30,            # annealed → 0
            w_grad0=0.12, w_grad_end=0.02,
            w_rel0=0.05,  w_rel_end=0.01,
            swa_start_frac=0.5,
            boundary_band=3, boundary_boost=1.6
        ),
        DeepONetOperator(
            device,
            name="DON_MSFF_SWA_484_chebyshev",
            grid_size=GRID_SIZE,
            n_sensors=484,          # 22x22 (por si memoria)
            hidden_size=320,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=2.5e-4,
            epochs=1500,
            weight_decay=1.5e-4,
            sensor_strategy='chebyshev',
            normalize_sensors=True,
            m_per_scale=64,
            ff_scales=(math.pi, 4*math.pi, 16*math.pi),
            sensor_dropout_p0=0.10,
            fourier_dropout_p0=0.30,
            w_grad0=0.12, w_grad_end=0.02,
            w_rel0=0.05,  w_rel_end=0.01,
            swa_start_frac=0.5,
            boundary_band=3, boundary_boost=1.6
        ),
        DeepONetOperator(
            device,
            name="DON_MSFF_SWA_576_random",
            grid_size=GRID_SIZE,
            n_sensors=576,
            hidden_size=320,
            num_layers=4,
            activation='gelu',
            dropout=0.05,
            lr=2.5e-4,
            epochs=1500,
            weight_decay=1.5e-4,
            sensor_strategy='random',
            normalize_sensors=True,
            m_per_scale=64,
            ff_scales=(math.pi, 4*math.pi, 16*math.pi),
            sensor_dropout_p0=0.10,
            fourier_dropout_p0=0.30,
            w_grad0=0.12, w_grad_end=0.02,
            w_rel0=0.05,  w_rel_end=0.01,
            swa_start_frac=0.5,
            boundary_band=3, boundary_boost=1.6
        ),
    ]

    print("\n📋 Model Configurations Summary:")
    print("-" * 80)
    for i, m in enumerate(models, 1):
        print(f"{i}. {m.name} — sensors={m.n_sensors} ({m.sensor_strategy}), "
              f"H={m.hidden_size}, L={m.num_layers}, m_per_scale={m.m_per_scale}, "
              f"epochs={m.epochs}, wd={m.weight_decay}, swa_start={m.swa_start_frac}")
    print("-" * 80)

    runner = BenchmarkRunner(models, dm, 500)
    runner.device = device
    scores = runner.run()

    best = max(scores, key=lambda s: s['metrics'].get('accuracy', -1))
    print(f"\n🏆 BEST MODEL: {best['name']}  Acc: {best['metrics'].get('accuracy', 0):.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(scores)
    print(f"💾 Results saved: results_optimized_deeponet_{timestamp}.json")
