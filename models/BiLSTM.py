import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self,num_classes:int=15,hidden_size:int=64,):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM=nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size*2,
                out_features=64
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),  
        )
    def forward(self,x):
        # 外部传入 [B,F]；只在这里增加一次维度，得到 [B,F,1]。
        x=x.unsqueeze(-1)
        out,(hidden,_)=self.LSTM(x)
        features=torch.cat((hidden[-2],hidden[-1]),dim=1)
        return self.classifier(features)
