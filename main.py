import torch
import random
import set_up_libs
import numpy as np
from data import DataModule
from models import DeepONetOperator, FNOOperator, FNOEnsembleOperator, PyKANOperator
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
    
                                      
    TRAIN_SIZE = 1100
    TEST_SIZE = 500
    
    dm = DataModule(grid=GRID_SIZE, n_train=TRAIN_SIZE, n_test=TEST_SIZE)
    dm.setup()
    
                               
    try:
        sample_batch = next(iter(dm.train))
        input_shape = sample_batch["x"].shape
        detected_channels = input_shape[1]
    except:
        detected_channels = 1
    
    models = [
                                                                         
        FNOOperator(
            device,
            "Standard_FNO", 
            grid_size=GRID_SIZE,
            modes=6,                                               
            width=20,                                               
            n_layers=4,                                   
            in_channels=detected_channels,                         
            lr=1e-3,               
            step_size=100,                                  
            gamma=0.5,                                 
            weight_decay=1e-4,      
            epochs=500,
            use_augmentation=True,                     
            activation='gelu'                        
        ),
        
                                                       
        FNOOperator(
            device,
            "Smaller_FNO_shared_weights",
            grid_size=GRID_SIZE,
            modes=4,                                  
            width=16,                              
            n_layers=3,                           
            in_channels=detected_channels,                         
            lr=3e-3,                                                        
            step_size=150,
            gamma=0.5,
            weight_decay=5e-5,                           
            epochs=500,
            use_augmentation=True,
            share_weights=True,                                         
            activation='gelu'
        ),
        
                                                           
        FNOEnsembleOperator(
            device,
            "Ensemble_FNO",
            grid_size=GRID_SIZE,
            n_models=2,                                   
            modes=5,
            width=18,
            n_layers=3,
            in_channels=detected_channels,                         
            lr=1e-3,
            epochs=500
        ),
        FNOOperator(
        device,
        "Enhanced_Smaller_FNO_Better_training",
        grid_size=GRID_SIZE,
        modes=5,                                                                   
        width=18,                                                           
        n_layers=3,                                                   
        in_channels=detected_channels,
        lr=2e-3,                                                    
        step_size=100,                                        
        gamma=0.65,                                                        
        weight_decay=4e-5,                               
        epochs=800,                                                       
        use_augmentation=True,                          
        share_weights=True,                             
        activation='gelu'                                   
    ),   

    FNOOperator(
    device,
    "Optimized_95_Target_FNO",
    grid_size=GRID_SIZE,
    modes=7,                                      
    width=30,                                                    
    n_layers=4,                                               
    in_channels=detected_channels,
    lr=2.2e-3,                                              
    step_size=120,                                     
    gamma=0.68,                            
    weight_decay=1.8e-5,                               
    epochs=1200,                                
    use_augmentation=True,
    share_weights=True,                                
    activation='gelu',
)
    ]
    
    runner = BenchmarkRunner(models, dm, epochs=500)
    runner.device = device  
    scores = runner.run()
    
    best_accuracy = -float('inf')                    
    best_model = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_fno_{timestamp}.json"
    runner.save_results(scores)

  
    

    
