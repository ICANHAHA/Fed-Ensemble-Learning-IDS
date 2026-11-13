import torch.nn as nn
import torch
class RNN(nn.Module):
    def __init__(self, input_size=74, hidden_size=128, num_classes=15):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            nonlinearity='tanh'
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        _, hidden = self.rnn(x)

        features = hidden[-1]

        logits = self.classifier(features)

        return logits
