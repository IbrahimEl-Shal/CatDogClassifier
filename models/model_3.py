import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim


class CNN_3():
    def __init__(self):
        self.model = models.densenet121(pretrained=True)

    def forward(self):
        '''Freeze parameters so we don't backprop through them'''
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                              nn.ReLU(),
                                              nn.Dropout(0.2),
                                              nn.Linear(512, 256),
                                              nn.ReLU(),
                                              nn.Dropout(0.1),
                                              nn.Linear(256, 2),
                                              nn.LogSoftmax(dim=1))
        return self.model
    
    def build(self):
        model = self.forward()
        return model
