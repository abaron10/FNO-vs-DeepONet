import time
import json
import torch
from datetime import datetime
from typing import Dict, Any, Sequence
from models import BaseOperator
from data import DataModule
from visualizer import Visualizer

def mse(p, y):
    return torch.mean((p - y) ** 2).item()

def mae(p, y):
    return torch.mean(torch.abs(p - y)).item()

def rel_l2(p, y):
    return (torch.linalg.vector_norm(p - y) / torch.linalg.vector_norm(y)).item()

def accuracy(p, y, threshold=0.1):
                                           
    relative_l2_error = rel_l2(p, y)
    
                                
    accuracy_value = 100 * (1 - relative_l2_error)
    
    return accuracy_value

def accuracy_pointwise(p, y, threshold=0.15):
    rel_error = torch.abs(p - y) / (torch.abs(y) + 1e-8)
    return (rel_error < threshold).float().mean().item() * 100

METRICS = dict(
    mse=mse, 
    mae=mae, 
    relative_l2=rel_l2,                                         
    accuracy=accuracy,
    accuracy_pointwise=accuracy_pointwise                                     
)

class BenchmarkRunner:
    def __init__(self, models: Sequence[BaseOperator], dm: DataModule, epochs: int = 25):
        self.models = models
        self.dm = dm
        self.epochs = epochs
        self.vis = Visualizer(dm)
        self.train_history = {}
        self.accuracy_history = {}                                                  

    def run(self):
        results = []
        
        for m in self.models:
            model_name = f"{m.name}_{m.grid_size}x{m.grid_size}"
            print(f"\n===== {model_name} =====")
            m.setup(self.dm.info)
            
            train_losses = []
            train_accuracies = []                                                          
            val_accuracies = []                                                         
            
            t0 = time.time()
            for ep in range(1, self.epochs + 1):
                                                                
                epoch_metrics = m.train_epoch(self.dm.train, self.dm.test)
                
                                                                                         
                if isinstance(epoch_metrics, dict):
                    train_losses.append(epoch_metrics['train_loss'])
                    train_accuracies.append(epoch_metrics['train_accuracy'])
                    val_accuracies.append(epoch_metrics.get('val_accuracy', 0))
                    tr_loss = epoch_metrics['train_loss']
                else:
                                                
                    tr_loss = epoch_metrics
                    train_losses.append(tr_loss)
                    train_accuracies.append(0)                         
                    val_accuracies.append(0)
                
                if ep % 5 == 0:
                    if isinstance(epoch_metrics, dict):
                        print(f"  epoch {ep:02d}/{self.epochs}  train_loss={tr_loss:.4e}  " + 
                              f"train_acc={epoch_metrics['train_accuracy']:.1f}%  " +
                              f"val_acc={epoch_metrics.get('val_accuracy', 0):.1f}%")
                    else:
                        print(f"  epoch {ep:02d}/{self.epochs}  train_loss={tr_loss:.4e}")
                        
            wall = time.time() - t0
            
            self.train_history[model_name] = train_losses
                                                       
            self.accuracy_history[model_name] = {
                'train': train_accuracies,
                'validation': val_accuracies
            }
            
                                          
            metrics = m.eval(self.dm.test, METRICS)
            metrics["wall_sec"] = wall
            metrics["avg_time_per_epoch"] = wall / self.epochs
            
                                                       
            if 'relative_l2' in metrics and 'accuracy' in metrics:
                expected_accuracy = 100 * (1 - metrics['relative_l2'])
                if abs(expected_accuracy - metrics['accuracy']) > 0.1:
                    print(f"\n⚠️  WARNING: Inconsistent accuracy calculation!")
                    print(f"   Relative L2: {metrics['relative_l2']:.4f}")
                    print(f"   Expected accuracy: {expected_accuracy:.1f}%")
                    print(f"   Reported accuracy: {metrics['accuracy']:.1f}%")
                            
                    metrics['accuracy'] = expected_accuracy
            
                                         
            print(f"\n  Final metrics (Li et al. method):")
            print(f"  Relative L2 error: {metrics.get('relative_l2', 'N/A'):.4f}")
            print(f"  Accuracy (100*(1-L2)): {metrics.get('accuracy', 'N/A'):.1f}%")
            if 'accuracy_pointwise' in metrics:
                print(f"  Pointwise accuracy (legacy): {metrics['accuracy_pointwise']:.1f}%")
            
            model_info = m.get_model_info()
            
            plot_path = self.vis.sample_and_plot(m)
            
            results.append({
                "name": model_name,
                "model_info": model_info,
                "metrics": metrics,
                "plot_path": f"../{plot_path}"
            })
            
        return results

    def save_results(self, results, output_path="/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/visualizer/benchmark_results.json"):
        clean_results = []
        for r in results:
            clean_model_info = r["model_info"].copy()
            
            if "architecture" in clean_model_info:
                arch = clean_model_info["architecture"]
                for k, v in arch.items():
                    if hasattr(v, 'tolist'):
                        arch[k] = v.tolist()
                    elif isinstance(v, (list, tuple)) and len(v) > 0 and hasattr(v[0], 'item'):
                        arch[k] = [x.item() if hasattr(x, 'item') else x for x in v]
            
                                                        
            metrics_with_note = r["metrics"].copy()
            metrics_with_note["accuracy_method"] = "Li et al. (100*(1-relative_L2_error))"
            
            clean_results.append({
                "name": r["name"],
                "model_info": clean_model_info,
                "metrics": {k: float(v) if isinstance(v, (int, float)) else v 
                          for k, v in metrics_with_note.items()},
                "plot_path": str(r["plot_path"])
            })
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device) if hasattr(self, 'device') else "unknown",
            "epochs": self.epochs,
            "dataset_info": self.dm.get_data_info(),
            "models": clean_results,
            "training_history": {
                model: [float(loss) for loss in losses] 
                for model, losses in self.train_history.items()
            },
                                                                
            "accuracy_history": {
                model: {
                    split: [float(acc) for acc in accs]
                    for split, accs in history.items()
                }
                for model, history in self.accuracy_history.items()
            },
            "accuracy_calculation_note": "Accuracy computed using Li et al. method: 100*(1-relative_L2_error)"
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[BenchmarkRunner] Results saved to {output_path}")