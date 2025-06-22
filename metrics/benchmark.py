
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

METRICS = dict(mse=mse, mae=mae, rel_l2=rel_l2)

class BenchmarkRunner:
    def __init__(self, models: Sequence[BaseOperator], dm: DataModule, epochs: int = 25):
        self.models = models
        self.dm = dm
        self.epochs = epochs
        self.vis = Visualizer(dm)
        self.train_history = {}

    def run(self):
        results = []
        
        for m in self.models:
            model_name = f"{m.__class__.__name__}_{m.grid_size}x{m.grid_size}"
            print(f"\n===== {model_name} =====")
            m.setup(self.dm.info)
            
            train_losses = []
            
            t0 = time.time()
            for ep in range(1, self.epochs + 1):
                tr_loss = m.train_epoch(self.dm.train)
                train_losses.append(tr_loss)
                if ep % 5 == 0:
                    print(f"  epoch {ep:02d}/{self.epochs}  train_loss={tr_loss:.4e}")
            wall = time.time() - t0
            
            self.train_history[model_name] = train_losses
            
            metrics = m.eval(self.dm.test, METRICS)
            metrics["wall_sec"] = wall
            metrics["avg_time_per_epoch"] = wall / self.epochs
            
            model_info = m.get_model_info()
            
            plot_path = self.vis.sample_and_plot(m)
            
            results.append({
                "name": model_name,
                "model_info": model_info,
                "metrics": metrics,
                "plot_path": plot_path
            })
            
        return results

    def save_results(self, results, output_path="/Users/andres.baron/Documents/Computer-Science/Tesis/Laboratory/visualizer/benchmark_results.json"):
        """Save all results to JSON"""
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
            
            clean_results.append({
                "name": r["name"],
                "model_info": clean_model_info,
                "metrics": {k: float(v) for k, v in r["metrics"].items()},
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
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[BenchmarkRunner] Results saved to {output_path}")