# visualizer/architecture.py
import torch
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from pathlib import Path

def export_tensorboard_graph(model: torch.nn.Module,
                             sample_input: torch.Tensor,
                             logdir: str = "runs/graph"):
    """
    Guarda el grafo de `model` en TensorBoard bajo `logdir`.
    Luego: `tensorboard --logdir runs`.
    """
    writer = SummaryWriter(logdir)
    writer.add_graph(model, sample_input)
    writer.close()
    print(f"[archviz] Graph written to TensorBoard logdir='{logdir}'")


def export_onnx(model: torch.nn.Module,
                sample_input: torch.Tensor,
                onnx_path: str = "model.onnx"):
    """
    Exporta `model` + `sample_input` a un .onnx, que puedes abrir en Netron.
    """
    Path(onnx_path).parent.mkdir(exist_ok=True)
    torch.onnx.export(model,
                      sample_input,
                      onnx_path,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=16)
    print(f"[archviz] ONNX model written to '{onnx_path}'")


def export_torchviz(model: torch.nn.Module,
                    sample_input: torch.Tensor,
                    out_path: str = "graph.pdf"):
    """
    Genera un PDF con el flujo de tensores/operaciones usando torchviz.
    """
    model.eval()
    y = model(sample_input)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = out_path.split(".")[-1]
    dot.render(out_path.replace(f".{dot.format}", ""))
    print(f"[archviz] Torchviz graph written to '{out_path}'")
