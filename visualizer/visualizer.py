import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Visualizer:
    def __init__(self, dm, out_dir="figs"):
        self.dm = dm
        self.out = Path(out_dir)
        self.out.mkdir(exist_ok=True)

    @torch.no_grad()
    def sample_and_plot(self, operator, n: int = 2, fname: str | None = None):
        batch = next(iter(self.dm.test))
        preds = operator.predict(batch).cpu()
        x = batch["x"].cpu().numpy()
        y = batch["y"].cpu().numpy()
        p = preds.numpy()
        
        errors = np.abs(p - y)
        
        fig, ax = plt.subplots(n, 4, figsize=(14, 4 * n))
        if n == 1:
            ax = ax.reshape(1, -1)  # Ensure 2D array for indexing
            
        for i in range(n):
            im0 = ax[i, 0].imshow(x[i, 0], cmap="viridis")
            ax[i, 0].set_title(f"Input κ(x) [{operator.grid_size}×{operator.grid_size}]")
            ax[i, 0].axis("off")
            plt.colorbar(im0, ax=ax[i, 0], fraction=0.046)
            
            im1 = ax[i, 1].imshow(y[i, 0], cmap="viridis")
            ax[i, 1].set_title("True p(x)")
            ax[i, 1].axis("off")
            plt.colorbar(im1, ax=ax[i, 1], fraction=0.046)
            
            im2 = ax[i, 2].imshow(p[i, 0], cmap="viridis")
            ax[i, 2].set_title("Predicted p̂(x)")
            ax[i, 2].axis("off")
            plt.colorbar(im2, ax=ax[i, 2], fraction=0.046)
            
            im3 = ax[i, 3].imshow(errors[i, 0], cmap="hot")
            ax[i, 3].set_title(f"Absolute Error (max: {errors[i, 0].max():.3f})")
            ax[i, 3].axis("off")
            plt.colorbar(im3, ax=ax[i, 3], fraction=0.046)
            
        fig.tight_layout()
        grid_info = f"_{operator.grid_size}x{operator.grid_size}"
        fpath = self.out / (fname or f"{operator.__class__.__name__}{grid_info}.png")
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] saved {fpath}")
        return str(fpath)