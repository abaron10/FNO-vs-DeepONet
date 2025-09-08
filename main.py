import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator 
from metrics import BenchmarkRunner 
from datetime import datetime

if __name__ == "__main__":
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
    
    TRAIN_SIZE = 8000
    TEST_SIZE = 3000
    
    dm = DataModule(grid=GRID_SIZE, n_train=TRAIN_SIZE, n_test=TEST_SIZE)
    dm.setup()
    
    try:
        sample_batch = next(iter(dm.train))
        input_shape = sample_batch["x"].shape
        detected_channels = input_shape[1]
    except:
        detected_channels = 1
    
    models = [
        DeepONetOperator(
            device,
            "DeepONet_Model1_random",
            grid_size=GRID_SIZE,
            n_sensors=3500,             
            hidden_size=280,            
            num_layers=6,               
            activation='gelu',
            lr=3e-4,                    
            step_size=50,
            gamma=0.85,
            weight_decay=5e-5,
            epochs=800,
            sensor_strategy='random', 
            normalize_sensors=True,
            dropout=0.15
        ),
    
        DeepONetOperator(
            device,
            "DeepONet_Model1_chebyshev",
            grid_size=GRID_SIZE,
            n_sensors=3500,             
            hidden_size=280,            
            num_layers=6,               
            activation='gelu',
            lr=3e-4,                    
            step_size=50,
            gamma=0.85,
            weight_decay=5e-5,
            epochs=800,
            sensor_strategy='chebyshev',  
            normalize_sensors=True,
            dropout=0.15
        ),
    
        DeepONetOperator(
            device,
            "DeepONet_Model1_adaptive",
            grid_size=GRID_SIZE,
            n_sensors=200,            
            hidden_size=280,            
            num_layers=6,               
            activation='gelu',
            lr=3e-4,                    
            step_size=50,
            gamma=0.85,
            weight_decay=5e-5,
            epochs=800,
            sensor_strategy='adaptive',  
            normalize_sensors=True,
            dropout=0.15
        ),
        DeepONetOperator(
            device,
            "DeepONet_Model2_uniform",
            grid_size=GRID_SIZE,
            n_sensors=200,            
            hidden_size=256,            
            num_layers=6,               
            activation='gelu',
            lr=3e-4,                    
            step_size=50,               
            gamma=0.9,                  
            weight_decay=5e-6,          
            epochs=600,                 
            sensor_strategy='uniform',
            normalize_sensors=True,
            dropout=0.03                
        )
    ]
    
    runner = BenchmarkRunner(models, dm, 1000)  
    runner.device = device  
    scores = runner.run()
    
    best_accuracy = -float('inf')
    best_model = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_enhanced_deeponet_{timestamp}.json"
    runner.save_results(scores)