import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, input_dim,num_classes):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.lin1=nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self,x):
        x = self.lin1(x)
        return x
