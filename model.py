import torch
import torchvision.models as models
import torch.nn as nn


def VGG16_pretrained_model(numClasses, featureExtract=True, usePretrained=True):
    model = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model, featureExtract)
    numFtrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(numFtrs, numClasses)
    return model


def set_parameter_requires_grad(model, featureExtracting):
    if featureExtracting:
        for param in model.parameters():
            param.requires_grad = False