# Darcy Flow FNO (Fourier Neural Operator)

This project implements a Fourier Neural Operator (FNO) model for solving Darcy flow problems. The implementation uses PyTorch and the neuraloperator library.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python darcy_flow_fno.py
```

The script will:
1. Load the Darcy flow dataset
2. Create and initialize the FNO model
3. Train the model for 11 epochs
4. Save the training progress plot as 'loss_plot.png'
5. Save the trained model as 'darcy_flow_fno_model.pth'

## Model Configuration

The FNO model is configured with the following parameters:
- Input channels: 1
- Output channels: 1
- Hidden channels: 32
- Lifting channels: 32
- Projection channels: 32
- Number of modes: (16, 16)
- Number of layers: 3

## Datasets

The script uses the `load_darcy_flow_small` dataset with:
- 100 training samples
- 50 test samples
- Batch size: 4
- Test resolution: 16

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- neuraloperator 0.4.0+
- NumPy 1.21.0+
- Matplotlib 3.4.0+ 