import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3, patience=5, model_type='hybrid'):
    """
    Trains the model and returns the best model based on validation loss.
    
    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.
        num_epochs: Maximum number of epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        model_type: 'tabular', 'image', or 'hybrid'.
        
    Returns:
        best_model, history
    """
    criterion = nn.MSELoss()
    
    if model_type == 'hybrid':
        # Differential learning rates
        # Image backbone needs small LR (finetuning)
        # Tabular and Fusion head need higher LR (training from scratch)
        optimizer = optim.Adam([
            {'params': model.image_model.parameters(), 'lr': lr},          # Base LR (e.g., 0.001)
            {'params': model.tabular_model.parameters(), 'lr': lr * 100},   # 10x LR (e.g., 0.01)
            {'params': model.fusion_head.parameters(), 'lr': lr * 100}      # 10x LR
        ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': []}
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            
            # Iterate over data
            for images, tabular, targets in tqdm(dataloader, desc=phase):
                images = images.to(device)
                tabular = tabular.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if model_type == 'hybrid':
                        outputs = model(images, tabular)
                    elif model_type == 'image':
                        outputs = model(images)
                    elif model_type == 'tabular':
                        outputs = model(tabular)
                    
                    loss = criterion(outputs, targets)
                    
                    if phase == 'train':
                        loss.backward()
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            history[f'{phase}_loss'].append(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
            
    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, device, model_type='hybrid'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for images, tabular, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            tabular = tabular.to(device)
            
            if model_type == 'hybrid':
                outputs = model(images, tabular)
            elif model_type == 'image':
                outputs = model(images)
            elif model_type == 'tabular':
                outputs = model(tabular)
                
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
            
    return np.array(predictions), np.array(actuals)
