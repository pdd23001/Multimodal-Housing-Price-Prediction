import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

def calculate_metrics(y_true, y_pred, log_transform=True):
    """
    Calculate metrics. If log_transform=True, also calculate metrics in original scale.
    """
    # Metrics in log scale
    mse_log = mean_squared_error(y_true, y_pred)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE (log)': mse_log,
        'RMSE (log)': rmse_log,
        'R2 (log)': r2_log
    }
    
    # If log-transformed, also calculate metrics in original scale
    if log_transform:
        y_true_orig = np.expm1(y_true)  # inverse of log1p
        y_pred_orig = np.expm1(y_pred)
        
        mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
        rmse_orig = np.sqrt(mse_orig)
        r2_orig = r2_score(y_true_orig, y_pred_orig)
        
        metrics.update({
            'MSE': mse_orig,
            'RMSE': rmse_orig,
            'R2': r2_orig
        })
    
    return metrics

def plot_history(history, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true, y_pred, title='Predictions vs Actual', save_path=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(title)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    if save_path:
        plt.savefig(save_path)
    plt.close()
