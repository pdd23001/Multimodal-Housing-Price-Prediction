# Visualization Workflow

This document explains how to save and visualize training results without retraining.

## How It Works

When you run `main.py`, all training artifacts are automatically saved to a timestamped directory:

```
results/
├── tabular_20251129_213000/
│   ├── model.pth           # Model weights
│   ├── history.pkl         # Training/validation loss curves
│   ├── predictions.pkl     # Predictions and ground truth
│   ├── metrics.txt         # Final metrics
│   ├── config.pkl          # Run configuration
│   ├── history.png         # Loss curve plot
│   └── predictions.png     # Predictions scatter plot
└── hybrid_20251129_214500/
    └── ...
```

## Usage Examples

### 1. Train a Model (Artifacts Auto-Saved)

```bash
python main.py --model_type tabular --epochs 50 --lr 0.01 --seed 42
```

This creates: `results/tabular_YYYYMMDD_HHMMSS/` with all artifacts.

### 2. Regenerate Plots for a Single Run

```bash
python visualize_results.py --run_dir results/tabular_20251129_213000
```

**Output:**
- `visualizations/tabular_history.png`
- `visualizations/tabular_predictions.png`
- `visualizations/tabular_residuals.png`

### 3. Compare Multiple Models

```bash
python visualize_results.py \
    --run_dir results/hybrid_20251129_214500 \
    --compare results/tabular_20251129_213000 results/image_20251129_220000
```

**Output:**
- All individual plots for hybrid model
- `visualizations/model_comparison.png` (side-by-side comparison)

### 4. Customize Output Directory

```bash
python visualize_results.py \
    --run_dir results/hybrid_20251129_214500 \
    --output_dir paper_figures
```

## Benefits

1. **No Retraining**: Iterate on plot styles, colors, labels without waiting for training
2. **Reproducibility**: Every run is saved with exact configuration and seed
3. **Comparison**: Easily compare different hyperparameters or architectures
4. **Publication Ready**: Generate high-resolution plots (300 DPI) for papers

## Tips

- Use `--seed 42` for reproducible results
- Keep the timestamped directories to track experiments
- Delete old runs to save disk space: `rm -rf results/tabular_20251129_*`
