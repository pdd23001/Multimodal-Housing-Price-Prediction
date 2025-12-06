"""
Visualize results from saved training runs without retraining.

Usage:
    python visualize_results.py --run_dir results/tabular_20251129_213000
    python visualize_results.py --run_dir results/hybrid_20251129_214500 --compare results/tabular_20251129_213000
"""

import argparse
import pickle
import os
import json
import matplotlib.pyplot as plt
import numpy as np


def load_run_data(run_dir):
    """Load all saved data from a training run."""
    data = {}

    # Load history
    with open(os.path.join(run_dir, "history.pkl"), "rb") as f:
        data["history"] = pickle.load(f)

    # Load predictions
    with open(os.path.join(run_dir, "predictions.pkl"), "rb") as f:
        pred_data = pickle.load(f)
        data["y_true"] = pred_data["y_true"]
        data["y_pred"] = pred_data["y_pred"]
        data["log_transform"] = pred_data.get("log_transform", True)

    # Load metrics from JSON
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        data["metrics"] = json.load(f)

    # Load config
    with open(os.path.join(run_dir, "config.pkl"), "rb") as f:
        data["config"] = pickle.load(f)

    return data


def calculate_price_range_metrics(y_true_log, y_pred_log):
    """Calculate R² and RMSE (log space) for different price ranges (<= $2M only)."""
    y_true_log = np.asarray(y_true_log).reshape(-1)
    y_pred_log = np.asarray(y_pred_log).reshape(-1)

    # Convert back to actual prices for bin masks
    y_true_actual = np.expm1(y_true_log)

    # Only keep bins up to $2M (dataset filtered)
    ranges = [
        (0, 500000, "$0-500k"),
        (500000, 1000000, "$500k-1M"),
        (1000000, 2000000, "$1M-2M"),
    ]

    results = []
    for min_price, max_price, label in ranges:
        mask = (y_true_actual >= min_price) & (y_true_actual < max_price)
        if mask.sum() == 0:
            continue

        y_true_range = y_true_log[mask]
        y_pred_range = y_pred_log[mask]

        # R² and RMSE computed in log space
        ss_res = np.sum((y_true_range - y_pred_range) ** 2)
        ss_tot = np.sum((y_true_range - np.mean(y_true_range)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((y_true_range - y_pred_range) ** 2)))

        results.append({"range": label, "count": int(mask.sum()), "r2": float(r2), "rmse": rmse})

    return results


def calculate_accuracy_bands(y_true_log, y_pred_log):
    """Calculate percentage of predictions within different relative error bands (<= $2M only)."""
    y_true_log = np.asarray(y_true_log).reshape(-1)
    y_pred_log = np.asarray(y_pred_log).reshape(-1)

    y_true_actual = np.expm1(y_true_log)
    y_pred_actual = np.expm1(y_pred_log)

    # Guard against divide-by-zero (shouldn't happen with prices, but safe)
    eps = 1e-9
    pct_error = np.abs((y_pred_actual - y_true_actual) / np.maximum(y_true_actual, eps)) * 100.0

    bands = [5, 10, 15, 20]
    results = {}
    for band in bands:
        within_band = (pct_error <= band).sum()
        results[f"±{band}%"] = float(within_band / len(pct_error) * 100.0)

    return results


def plot_training_history(data, save_path=None):
    """Plot training and validation loss curves."""
    history = data["history"]
    config = data["config"]

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE in log scale)", fontsize=12)
    plt.title(f"{config['model_type'].capitalize()} Model - Training History", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0,1)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_predictions_scatter(data, save_path=None):
    """Plot predicted vs actual prices in log space."""
    y_true = np.asarray(data["y_true"]).reshape(-1)
    y_pred = np.asarray(data["y_pred"]).reshape(-1)
    config = data["config"]

    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)

    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    plt.xlabel("Actual Price (log transformed)", fontsize=12)
    plt.ylabel("Predicted Price (log transformed)", fontsize=12)
    plt.title(f"{config['model_type'].capitalize()} Model - Predictions", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    r2_log = data["metrics"].get("R2 (log)", 0)
    rmse_log = data["metrics"].get("RMSE (log)", 0)
    info_text = f"R² (log) = {r2_log:.3f}\nRMSE (log) = {rmse_log:.3f}"
    plt.text(
        0.05,
        0.95,
        info_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved predictions scatter plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_residuals(data, save_path=None):
    """Plot residuals (errors) distribution."""
    y_true = np.asarray(data["y_true"]).reshape(-1)
    y_pred = np.asarray(data["y_pred"]).reshape(-1)
    config = data["config"]

    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax1.set_xlabel("Predicted Price (log scale)", fontsize=12)
    ax1.set_ylabel("Residuals", fontsize=12)
    ax1.set_title("Residual Plot", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Residuals", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=14)
    ax2.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{config['model_type'].capitalize()} Model - Residual Analysis", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved residual plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_price_range_performance(data, save_path=None):
    """Plot model performance across different price ranges (<= $2M only)."""
    y_true = np.asarray(data["y_true"]).reshape(-1)
    y_pred = np.asarray(data["y_pred"]).reshape(-1)
    config = data["config"]

    range_metrics = calculate_price_range_metrics(y_true, y_pred)
    if not range_metrics:
        print("No price range data to plot")
        return

    ranges = [r["range"] for r in range_metrics]
    r2_values = [r["r2"] for r in range_metrics]
    rmse_values = [r["rmse"] for r in range_metrics]
    counts = [r["count"] for r in range_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    bars1 = ax1.bar(ranges, r2_values, alpha=0.8)
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.set_xlabel("Price Range", fontsize=12)
    ax1.set_title("R² by Price Range (log space)", fontsize=14)
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.4)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars1, counts):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h, f"n={count}", ha="center", va=va, fontsize=10)

    bars2 = ax2.bar(ranges, rmse_values, alpha=0.8)
    ax2.set_ylabel("RMSE (log)", fontsize=12)
    ax2.set_xlabel("Price Range", fontsize=12)
    ax2.set_title("RMSE by Price Range (log space)", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars2, counts):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h, f"n={count}", ha="center", va="bottom", fontsize=10)

    plt.suptitle(f"{config['model_type'].capitalize()} Model - Performance by Price Range", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved price range performance plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_accuracy_bands(data, save_path=None):
    """Plot percentage of predictions within different accuracy bands."""
    y_true = np.asarray(data["y_true"]).reshape(-1)
    y_pred = np.asarray(data["y_pred"]).reshape(-1)
    config = data["config"]

    accuracy_bands = calculate_accuracy_bands(y_true, y_pred)

    bands = list(accuracy_bands.keys())
    percentages = list(accuracy_bands.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bands, percentages, alpha=0.8)
    plt.ylabel("Percentage of Predictions (%)", fontsize=12)
    plt.xlabel("Error Band", fontsize=12)
    plt.title(f"{config['model_type'].capitalize()} Model - Prediction Accuracy Bands", fontsize=14)
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3, axis="y")

    for bar, pct in zip(bars, percentages):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved accuracy bands plot to {save_path}")
    else:
        plt.show()
    plt.close()


def compare_models(run_dirs, output_dir="."):
    """Compare multiple model runs - saves 3 separate plots."""
    all_data = []
    labels = []

    for run_dir in run_dirs:
        data = load_run_data(run_dir)
        all_data.append(data)
        labels.append(data["config"]["model_type"].capitalize())

    # Color scheme
    color_map = {"Hybrid": "#1E90FF", "Tabular": "#DC143C", "Image": "#228B22"}
    colors = [color_map.get(label, "#808080") for label in labels]

    models_str = "_vs_".join([label.lower() for label in labels])

    # 1) Validation loss
    plt.figure(figsize=(10, 6))
    for data, label, color in zip(all_data, labels, colors):
        plt.plot(data["history"]["val_loss"], label=label, linewidth=2, marker="o", markersize=4, color=color)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title("Validation Loss Comparison (MSE on log-transformed prices)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    loss_path = os.path.join(output_dir, f"{models_str}_validation_loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    print(f"Saved validation loss comparison to {loss_path}")
    plt.close()

    # 2) R² bars
    fig, ax = plt.subplots(figsize=(10, 6))
    r2_metrics = ["R2 (log)", "R2"]
    x = np.arange(len(r2_metrics))
    width = 0.8 / len(all_data)

    for i, (data, label, color) in enumerate(zip(all_data, labels, colors)):
        values = [data["metrics"].get(m, 0) for m in r2_metrics]
        ax.bar(x + i * width, values, width, label=label, alpha=0.8, color=color)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("R² Score Comparison", fontsize=14)
    ax.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax.set_xticklabels(r2_metrics)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    r2_path = os.path.join(output_dir, f"{models_str}_r2_comparison.png")
    plt.savefig(r2_path, dpi=300, bbox_inches="tight")
    print(f"Saved R² comparison to {r2_path}")
    plt.close()

    # 3) RMSE(log)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (data, label, color) in enumerate(zip(all_data, labels, colors)):
        rmse_log_value = data["metrics"].get("RMSE (log)", 0)
        ax.bar(i, rmse_log_value, width=0.6, label=label, alpha=0.8, color=color)

    ax.set_ylabel("RMSE (log)", fontsize=12)
    ax.set_title("RMSE (log) Comparison", fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    rmse_path = os.path.join(output_dir, f"{models_str}_rmse_comparison.png")
    plt.savefig(rmse_path, dpi=300, bbox_inches="tight")
    print(f"Saved RMSE comparison to {rmse_path}")
    plt.close()


def compare_price_range_performance(all_data, labels, colors, output_dir="."):
    """Compare model performance across price ranges (<= $2M only)."""
    models_str = "_vs_".join([label.lower() for label in labels])

    all_range_metrics = []
    for data, label in zip(all_data, labels):
        range_metrics = calculate_price_range_metrics(data["y_true"], data["y_pred"])
        all_range_metrics.append(range_metrics)

        print(f"\n{label} model - Price ranges found:")
        for r in range_metrics:
            print(f"  {r['range']}: n={r['count']}, R²={r['r2']:.3f}")

    if not all_range_metrics or not all_range_metrics[0]:
        print("No price range data to plot")
        return

    # Collect all ranges present
    all_ranges = []
    for metrics in all_range_metrics:
        for r in metrics:
            if r["range"] not in all_ranges:
                all_ranges.append(r["range"])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_ranges))
    width = 0.8 / len(all_data)

    for i, (metrics, label, color) in enumerate(zip(all_range_metrics, labels, colors)):
        r2_values, counts = [], []
        for range_label in all_ranges:
            found = next((m for m in metrics if m["range"] == range_label), None)
            if found:
                r2_values.append(found["r2"])
                counts.append(found["count"])
            else:
                r2_values.append(0.0)
                counts.append(0)

        bars = ax.bar(x + i * width, r2_values, width, label=label, alpha=0.8, color=color)

        for bar, count, r2 in zip(bars, counts, r2_values):
            if count > 0:
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"n={count}", ha="center", va=va, fontsize=8)

    ax.set_xlabel("Price Range", fontsize=12)
    ax.set_ylabel("R² Score (log space)", fontsize=12)
    ax.set_title("R² Score Comparison by Price Range", fontsize=14)
    ax.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax.set_xticklabels(all_ranges)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    r2_path = os.path.join(output_dir, f"{models_str}_price_range_r2.png")
    plt.savefig(r2_path, dpi=300, bbox_inches="tight")
    print(f"Saved price range R² comparison to {r2_path}")
    plt.close()


def compare_accuracy_bands(all_data, labels, colors, output_dir="."):
    """Compare model accuracy bands."""
    models_str = "_vs_".join([label.lower() for label in labels])

    all_bands = []
    for data in all_data:
        all_bands.append(calculate_accuracy_bands(data["y_true"], data["y_pred"]))

    band_labels = list(all_bands[0].keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(band_labels))
    width = 0.8 / len(all_data)

    for i, (bands, label, color) in enumerate(zip(all_bands, labels, colors)):
        percentages = list(bands.values())
        bars = ax.bar(x + i * width, percentages, width, label=label, alpha=0.8, color=color)

        for bar, pct in zip(bars, percentages):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Error Band", fontsize=12)
    ax.set_ylabel("Percentage of Predictions (%)", fontsize=12)
    ax.set_title("Prediction Accuracy Bands Comparison", fontsize=14)
    ax.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax.set_xticklabels(band_labels)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    bands_path = os.path.join(output_dir, f"{models_str}_accuracy_bands.png")
    plt.savefig(bands_path, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy bands comparison to {bands_path}")
    plt.close()


def print_summary(data):
    """Print summary of the run."""
    config = data["config"]
    metrics = data["metrics"]

    print("\n" + "=" * 60)
    print(f"MODEL: {config['model_type'].upper()}")
    print("=" * 60)
    print(f"Timestamp: {config.get('timestamp', 'N/A')}")
    print(f"Learning Rate: {config.get('lr', 'N/A')}")
    print(f"Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"Epochs: {len(data['history']['train_loss'])}")
    print(f"Seed: {config.get('seed', 'N/A')}")
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize saved training results")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to saved run directory")
    parser.add_argument("--compare", type=str, nargs="+", help="Additional run directories to compare")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and visualize main run
    print(f"Loading data from {args.run_dir}...")
    data = load_run_data(args.run_dir)
    print_summary(data)

    model_name = data["config"]["model_type"]
    plot_training_history(data, save_path=os.path.join(args.output_dir, f"{model_name}_history.png"))
    plot_predictions_scatter(data, save_path=os.path.join(args.output_dir, f"{model_name}_predictions.png"))
    plot_residuals(data, save_path=os.path.join(args.output_dir, f"{model_name}_residuals.png"))
    plot_price_range_performance(data, save_path=os.path.join(args.output_dir, f"{model_name}_price_range.png"))
    plot_accuracy_bands(data, save_path=os.path.join(args.output_dir, f"{model_name}_accuracy_bands.png"))

    # Compare with other runs
    if args.compare:
        print("\nComparing with other runs...")
        all_runs = [args.run_dir] + args.compare

        all_data = [data]
        labels = [data["config"]["model_type"].capitalize()]
        for run_dir in args.compare:
            other_data = load_run_data(run_dir)
            all_data.append(other_data)
            labels.append(other_data["config"]["model_type"].capitalize())

        color_map = {"Hybrid": "#1E90FF", "Tabular": "#DC143C", "Image": "#228B22"}
        colors = [color_map.get(label, "#808080") for label in labels]

        compare_models(all_runs, output_dir=args.output_dir)
        compare_price_range_performance(all_data, labels, colors, output_dir=args.output_dir)
        compare_accuracy_bands(all_data, labels, colors, output_dir=args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
