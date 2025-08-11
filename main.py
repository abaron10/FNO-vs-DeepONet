import math
import random
import numpy as np
import torch
import set_up_libs

from data import DataModule
from metrics import BenchmarkRunner
from models.deep_o_net import DeepONetOperator
from datetime import datetime

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
    print("\n🔬 Single Model — DeepONet Integral-Branch + MS-RFF + Sobolev + SWA + MC-TTA")

    model = DeepONetOperator(
        device,
        name="DON_IntegralRFF_MSFF_SWA_576_random",
        grid_size=GRID_SIZE,
        n_sensors=576,                 # 24×24 sensores
        hidden_size=384,
        num_layers=4,
        activation='gelu',
        dropout=0.05,
        lr=2.0e-4,
        epochs=1600,
        weight_decay=1.5e-4,
        sensor_strategy='random',      # usa 'random' (te fue mejor que chebyshev)
        normalize_sensors=True,
        # Trunk/Branch Fourier features
        m_per_scale=64,
        ff_scales=(math.pi, 4*math.pi, 16*math.pi),
        branch_m_per_scale=64,
        branch_scales=(math.pi, 4*math.pi, 16*math.pi),
        # Annealing de regularizaciones
        sensor_dropout_p0=0.10,
        fourier_dropout_p0=0.30,
        w_grad0=0.12, w_grad_end=0.02,
        w_rel0=0.05,  w_rel_end=0.01,
        boundary_band=3, boundary_boost=1.6,
        edge_boost_alpha=1.0,
        swa_start_frac=0.5,
        # MC-TTA en evaluación
        mc_samples_eval=16, mc_fourier_p=0.20, mc_sensor_p=0.05
    )

    models = [model]
    print("\n📋 Model Configuration:")
    print("-" * 80)
    print(f"{model.name} — sensors={model.n_sensors} ({model.sensor_strategy}), "
          f"H={model.hidden_size}, L={model.num_layers}, m_per_scale={model.m_per_scale}, "
          f"branch_m={model.branch_m_per_scale}, epochs={model.epochs}, wd={model.weight_decay}, "
          f"swa_start={model.swa_start_frac}, mcK={model.mc_samples_eval}")
    print("-" * 80)

    runner = BenchmarkRunner(models, dm, 2000)
    runner.device = device
    scores = runner.run()

    best = scores[0]
    print(f"\n🏆 FINAL MODEL: {best['name']}  Acc: {best['metrics'].get('accuracy', 0):.2f}%")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(scores)
    print(f"💾 Results saved: results_deeponet_single_{timestamp}.json")
