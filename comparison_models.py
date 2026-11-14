"""CNN 与 RNN 对照模型；输入均为 [批次, 特征数]，输出为 [批次, 类别数]。"""

import torch
from torch import nn


class CNN1D(nn.Module):
    """将表格特征按固定列顺序看成一维输入，做卷积分类。"""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"CNN 需要 [B, {self.input_size}]，实际为 {tuple(x.shape)}")
        x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]
        return self.classifier(self.features(x))  # 原始分类分数 logits


class RNNClassifier(nn.Module):
    """将每条表格记录的特征按固定顺序作为序列，使用普通 RNN 分类。"""

    def __init__(self, input_size, num_classes, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.rnn = nn.RNN(
            input_size=1, hidden_size=hidden_size, num_layers=2,
            batch_first=True, dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"RNN 需要 [B, {self.input_size}]，实际为 {tuple(x.shape)}")
        x = x.unsqueeze(-1)  # [B, F] -> [B, F, 1]
        _, hidden = self.rnn(x)
        return self.classifier(hidden[-1])  # [B, hidden] -> [B, classes]
