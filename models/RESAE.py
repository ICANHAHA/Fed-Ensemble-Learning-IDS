import torch
import torch.nn as nn


#----------------------------------------残差块--------------------------------
class ResidualBlock(nn.Module):
    def __init__(self,hidden_size):
          super().__init__()
          self.branch = nn.Sequential(
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
          )
          self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.branch(x)
        out = out + residual
        out = self.relu(out)
        return out
#-----------------------------------------编码器--------------------------------
class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Encoder,self).__init__()
        self.res1=ResidualBlock(hidden_size)
        self.encoder=nn.Sequential(
            nn.Linear(input_size,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
        )
    def forward(self,x):
        mid=self.encoder(x)
        mid=self.res1(mid)
        return mid
#-------------------------------------解码器--------------------------------------------
class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Decoder,self).__init__()
        self.decoder=nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )
    def forward(self,x):
       out=self.decoder(x)
       return out
#-------------------------------------分类器-----------------------------------------
class classifier(nn.Module):
    def __init__(self,hidden_size,num_classes):
        super(classifier,self).__init__()
        self.classifier=nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,num_classes),
        )
    def forward(self,x):
        logits=self.classifier(x)
        return logits
#----------------------------------AE编码器--------------------------------------------
class ResAutoEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(ResAutoEncoder,self).__init__()
        self.encoder=Encoder(input_size,hidden_size)
        self.decoder=Decoder(input_size,hidden_size)
        self.classifier=classifier(hidden_size,num_classes)
    def forward(self,x):
        mid=self.encoder(x)
        reconstructed=self.decoder(mid)
        logits=self.classifier(mid)
        return mid,reconstructed,logits
#----------------------------损失函数--------------------------------------
# 沿用当前 CE + alpha*MSE 实现，未确认与论文损失完全一致。
class ResAEloss(nn.Module):
    def __init__(self,alpha:float=0.2):
        super(ResAEloss,self).__init__()
        self.loss1=nn.CrossEntropyLoss()
        self.loss2=nn.MSELoss()
        self.alpha=alpha
    def forward(self,inputs,targets,logits,reconstructed):
        loss1=self.loss1(logits,targets)
        loss2=self.loss2(inputs,reconstructed)
        loss=loss1+self.alpha*loss2
        return loss,loss1,loss2



