import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class RealEstateDataset(Dataset):
    def __init__(self, tabular_data, targets, image_paths, transform=None):
        """
        Args:
            tabular_data (np.array): Preprocessed tabular features.
            targets (np.array): Target values (prices).
            image_paths (list or pd.Series): Paths to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.tabular_data = torch.FloatTensor(tabular_data)
        self.targets = torch.FloatTensor(targets).view(-1, 1)
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # Handle missing or corrupt images by creating a black image
            # print(f"Warning: Could not load image at {img_path}. Using black image.")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        tabular = self.tabular_data[idx]
        target = self.targets[idx]

        return image, tabular, target

def get_data_loaders(dataset_dict, batch_size=32, num_workers=4, image_transform=None):
    """
    Creates DataLoaders for train and test sets.
    
    Args:
        dataset_dict (dict): Dictionary from preprocess.py containing 'train' and 'test' data.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        image_transform (callable): Transformations for images.
        
    Returns:
        train_loader, test_loader
    """
    
    # Unpack train data
    X_train, y_train, df_train = dataset_dict['train']
    X_test, y_test, df_test = dataset_dict['test']
    
    # Identify image path column
    # We look for common names or assume a specific one. 
    # For now, let's assume the user will ensure a column named 'image_path' or similar exists,
    # or we try to find it.
    possible_img_cols = ['image_path', 'file_path', 'path', 'Image']
    img_col = next((c for c in df_train.columns if c in possible_img_cols), None)
    
    if not img_col:
        # If not found, maybe construct it from ID? 
        # For now, raise error or handle gracefully.
        # We'll assume a dummy list if not found for testing purposes, but warn.
        print("Warning: Image path column not found. Using dummy paths.")
        train_img_paths = ["dummy.jpg"] * len(df_train)
        test_img_paths = ["dummy.jpg"] * len(df_test)
    else:
        train_img_paths = df_train[img_col].values
        test_img_paths = df_test[img_col].values

    train_dataset = RealEstateDataset(X_train, y_train, train_img_paths, transform=image_transform)
    test_dataset = RealEstateDataset(X_test, y_test, test_img_paths, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
