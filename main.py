import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator, FNOOperator, PyKANOperator
from metrics import BenchmarkRunner 

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    GRID_SIZE = 32 
    
    dm = DataModule(grid=GRID_SIZE, n_train=100, n_test=50)
    dm.setup()
    
    models = [
        FNOOperator(device, grid_size=GRID_SIZE),
        DeepONetOperator(device, grid_size=GRID_SIZE),
        PyKANOperator(device, hidden_neurons=10, lr=1e-3)   # ← nuevo
    ]
    
    runner = BenchmarkRunner(models, dm, 1)
    runner.device = device  
    
    scores = runner.run()
    
    print("\n===== RESULTS =====")
    for s in scores:
        print(f"\n{s['name']}:")
        print(f"  Parameters: {s['model_info']['parameters']:,}")
        print(f"  Metrics: {s['metrics']}")
    
    runner.save_results(scores)