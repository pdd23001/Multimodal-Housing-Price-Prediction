import torch
import torch.nn as nn
import torchvision.models as models

class TabularModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.1):
        super(TabularModel, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.regressor = nn.Linear(in_dim, 1)
        
        # Initialize regressor bias to 0 (will learn the mean during training)
        nn.init.constant_(self.regressor.bias, 0.0)
        nn.init.xavier_uniform_(self.regressor.weight)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.regressor(features)
        return output

    def get_embedding(self, x):
        return self.feature_extractor(x)

class ImageModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageModel, self).__init__()
        # Load ResNet18
        # weights='DEFAULT' is equivalent to pretrained=True in newer versions
        self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Remove the last fully connected layer to use as feature extractor
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Replace fc with identity to get embeddings
        
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.embedding_dim = num_ftrs

    def forward(self, x):
        # x shape: [batch, 3, 224, 224]
        embedding = self.backbone(x)
        output = self.regressor(embedding)
        return output

    def get_embedding(self, x):
        return self.backbone(x)

class HybridModel(nn.Module):
    def __init__(self, tabular_input_dim, tabular_hidden_dims=[128, 64], image_pretrained=True):
        super(HybridModel, self).__init__()
        
        # Image Branch
        self.image_model = ImageModel(pretrained=image_pretrained)
        # Remove the regressor from image model, we only want the backbone
        self.image_backbone = self.image_model.backbone
        image_out_dim = self.image_model.embedding_dim
        
        # Tabular Branch
        self.tabular_model = TabularModel(input_dim=tabular_input_dim, hidden_dims=tabular_hidden_dims)
        # Remove regressor, keep feature extractor
        self.tabular_backbone = self.tabular_model.feature_extractor
        tabular_out_dim = tabular_hidden_dims[-1]
        
        # Fusion
        fusion_input_dim = image_out_dim + tabular_out_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular):
        img_embed = self.image_backbone(image)
        tab_embed = self.tabular_backbone(tabular)
        
        # Concatenate
        combined = torch.cat((img_embed, tab_embed), dim=1)
        
        output = self.fusion_head(combined)
        return output
