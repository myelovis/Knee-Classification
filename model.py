import torchvision.models as models
import torch.nn as nn


def build_model(backbone_name='ResNet50', num_classes=5):
    if backbone_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
