import torch
import random
import numpy as np
import set_up_libs
from data import DataModule
from metrics import BenchmarkRunner
from datetime import datetime

from models.deep_o_net import DeepONetOperator

if __name__ == "__main__":
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

    models = [
        DeepONetOperator(device, name="DON_FourierFiLM_256_chebyshev",
                         grid_size=GRID_SIZE, n_sensors=256,
                         hidden_size=256, num_layers=4, activation='gelu', dropout=0.05,
                         lr=3e-4, epochs=600, weight_decay=1e-4,
                         sensor_strategy='chebyshev', normalize_sensors=True),
        DeepONetOperator(device, name="DON_FourierFiLM_256_random",
                         grid_size=GRID_SIZE, n_sensors=256,
                         hidden_size=256, num_layers=4, activation='gelu', dropout=0.10,
                         lr=2.5e-4, epochs=600, weight_decay=1.5e-4,
                         sensor_strategy='random', normalize_sensors=True),
        DeepONetOperator(device, name="DON_FourierFiLM_256_uniform",
                         grid_size=GRID_SIZE, n_sensors=256,
                         hidden_size=320, num_layers=4, activation='gelu', dropout=0.08,
                         lr=2e-4, epochs=600, weight_decay=2e-4,
                         sensor_strategy='uniform', normalize_sensors=True),
    ]

    print("\n📋 Model Configurations Summary:")
    print("-" * 80)
    for i, m in enumerate(models, 1):
        print(f"{i}. {m.name} — sensors={m.n_sensors} ({m.sensor_strategy}), H={m.hidden_size}, L={m.num_layers}, drop={m.dropout}")

    runner = BenchmarkRunner(models, dm, 500)
    runner.device = device
    scores = runner.run()

    best = max(scores, key=lambda s: s['metrics'].get('accuracy', -1))
    print(f"\n🏆 BEST MODEL: {best['name']}  Acc: {best['metrics'].get('accuracy', 0):.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(scores)
    print(f"💾 Results saved: results_optimized_deeponet_{timestamp}.json")
