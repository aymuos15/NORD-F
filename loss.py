import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define VGGPerceptualLoss class
class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 8, 15, 22], use_gpu=True):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.feature_layers = feature_layers
        self.use_gpu = use_gpu
        if use_gpu:
            self.vgg = self.vgg.to(device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)
        loss = 0
        for xf, yf in zip(x_features, y_features):
            # loss += F.mse_loss(xf, yf)
            # L1
            loss += F.l1_loss(xf, yf)
        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features
    
class VGGPerceptualLossV2(nn.Module):
    def __init__(self, model='vgg16', feature_layers=[3, 8, 15, 22], layer_weights=None, use_l1=False):
        super(VGGPerceptualLossV2, self).__init__()
        self.model = self._get_model(model)
        self.feature_layers = feature_layers
        self.layer_weights = layer_weights or [1.0] * len(feature_layers)
        self.use_l1 = use_l1
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Create a sequential model for feature extraction
        self.features = nn.Sequential(*list(self.model.children())[:max(feature_layers)+1])
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def _get_model(self, model_name):
        if model_name == 'vgg16':
            return models.vgg16(pretrained=True).features
        elif model_name == 'resnet50':
            return models.resnet50(pretrained=True)
        elif model_name == 'efficientnet_b0':
            return models.efficientnet_b0(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)
        loss = 0
        for xf, yf, weight in zip(x_features, y_features, self.layer_weights):
            if self.use_l1:
                loss += weight * F.l1_loss(xf, yf)
            else:
                loss += weight * F.mse_loss(xf, yf)
        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features