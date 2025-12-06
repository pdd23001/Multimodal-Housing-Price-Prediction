import argparse
import os
import torch
from torchvision import transforms
from src.preprocess import load_and_preprocess_data
from src.data_loader import get_data_loaders
from src.models import TabularModel, ImageModel, HybridModel
from src.train import train_model, evaluate_model
from src.utils import calculate_metrics, plot_history, plot_predictions
import random
import numpy as np
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description='House Price Prediction')
    parser.add_argument('--csv_path', type=str, default='data/socal2.csv', help='Path to CSV file')
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['tabular', 'image', 'hybrid'], help='Model type')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load and Preprocess Data
    print("Loading data...")
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        # Create dummy data for demonstration if file missing
        print("Creating dummy data for demonstration...")
        os.makedirs('data', exist_ok=True)
        dummy_df = pd.DataFrame({
            'Price': [100000, 200000, 150000, 300000] * 25,
            'Area': [1000, 2000, 1500, 2500] * 25,
            'Bedrooms': [2, 3, 2, 4] * 25,
            'image_path': ['data/dummy.jpg'] * 100
        })
        dummy_df.to_csv(args.csv_path, index=False)
        # Create dummy image
        from PIL import Image
        Image.new('RGB', (224, 224)).save('data/dummy.jpg')
    
    dataset_dict = load_and_preprocess_data(args.csv_path, random_state=args.seed)
    
    # 2. Create DataLoaders
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader, test_loader = get_data_loaders(dataset_dict, batch_size=args.batch_size, image_transform=image_transform)
    
    # 3. Initialize Model
    print(f"Initializing {args.model_type} model...")
    X_train_processed = dataset_dict['train'][0]
    tabular_input_dim = X_train_processed.shape[1]
    
    if args.model_type == 'tabular':
        model = TabularModel(input_dim=tabular_input_dim)
    elif args.model_type == 'image':
        model = ImageModel()
    elif args.model_type == 'hybrid':
        model = HybridModel(tabular_input_dim=tabular_input_dim)
        
    # 4. Train
    print("Starting training...")
    model, history = train_model(model, train_loader, test_loader, device, num_epochs=args.epochs, lr=args.lr, patience=args.patience, model_type=args.model_type)
    
    # 5. Evaluate
    print("Evaluating...")
    y_pred, y_true = evaluate_model(model, test_loader, device, model_type=args.model_type)
    log_transform = dataset_dict.get('log_transform', True)
    metrics = calculate_metrics(y_true, y_pred, log_transform=log_transform)
    print(f"Test Metrics: {metrics}")
    
    # 6. Save Results
    import pickle
    
    # Create a unique run directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f'{args.model_type}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(run_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(run_dir, 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    # Save predictions and ground truth
    predictions_path = os.path.join(run_dir, 'predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred, 'log_transform': log_transform}, f)
    print(f"Predictions saved to {predictions_path}")
    
    # Save metrics as JSON (human-readable)
    import json
    metrics_path = os.path.join(run_dir, 'metrics.json')
    # Convert numpy types to native Python types for JSON serialization
    metrics_json = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save run configuration
    config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'seed': args.seed,
        'timestamp': timestamp,
        'tabular_input_dim': tabular_input_dim
    }
    config_path = os.path.join(run_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    # Generate and save plots
    plot_history(history, save_path=os.path.join(run_dir, 'history.png'))
    plot_predictions(y_true, y_pred, title=f'{args.model_type} Predictions', save_path=os.path.join(run_dir, 'predictions.png'))
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete! All results saved to:")
    print(f"  {run_dir}")
    print(f"{'='*60}")
    print("Done!")

if __name__ == '__main__':
    main()
