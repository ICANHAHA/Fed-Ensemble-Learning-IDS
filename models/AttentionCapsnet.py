import torch.nn.functional as F
import torch
import torch.nn as nn


#现在写初始卷积模块
class ConvBlock(nn.Module):
    def __init__(self,num_features:int=74,out_channels:int=32):
        super(ConvBlock, self).__init__()
        self.num_features = num_features
        if not 1 <= num_features <= 100:
            raise ValueError("当前卷积模块要求特征数在 1 到 100 之间")
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            )
        )
    def forward(self, x):
        if x.ndim != 2 or x.size(1) != self.num_features:
            raise ValueError(f"CapsNet 需要 [B,{self.num_features}]，实际是 {tuple(x.shape)}")
        # 按实际 F 个特征补零到 100，组成 10×10；不改变预处理数据。
        x=F.pad(x,(0,100-self.num_features),'constant',0)
        x=x.reshape(x.size(0),1,10,10)
        out=self.conv1(x)
        return out
#现在写注意力通道
class AttentionChanel(nn.Module):
    def __init__(self):
        super(AttentionChanel, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        B, C, H, W = x.shape
        residual=x
        x=x.flatten(start_dim=2)
        x_T=torch.transpose(x,1,2)
        attention_x=torch.matmul(x,x_T)
        #归一化分数,每行进行归一化
        attention_x=attention_x.softmax(dim=-1)
        out=torch.matmul(attention_x,x)
        out=out.reshape(B,C,H,W)
        #残差连接
        out=self.gamma*out+residual
        return out
#实现压缩函数
class squash(nn.Module):
    def __init__(self):
        super(squash, self).__init__()
    def forward(self,x):
        x_module=torch.norm(x,p=2,dim=-1,keepdim=True)
        #先计算单位向量
        x_unit=x/(x_module+ 1e-8)
        #
        x_sqush=(x_module.pow(2)/(1+x_module.pow(2)))*x_unit
        return x_sqush

#--------------------现在写初级胶囊:B,800,8--------------
class PrimaryCaps(nn.Module):
    def __init__(self,in_channels:int,out_channels:int=256):
        super(PrimaryCaps, self).__init__()
        self.conv2d=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.squash = squash()
    def forward(self,x):
        out=self.conv2d(x)
        #重塑为 128*32*8*5*5
        out=out.reshape(-1,32,8,5,5)
        #调整顺序
        out=out.permute(0,3,4,1,2)
        out=out.reshape(out.shape[0],out.shape[1]*out.shape[2]*out.shape[3],out.shape[4])
        out = self.squash(out)
        return out
#----------------现在写动态路由----------------
#----------------现在写权重矩阵W和votes----------------
class capsulW(nn.Module):
    def __init__(self,num_primary_caps:int=800,num_classes:int=15,num_in_features:int=8,num_out_features:int=16):
        super(capsulW, self).__init__()
        self.num_primary_caps = num_primary_caps
        self.num_classes = num_classes
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        # 创建权重矩阵
        self.W = nn.Parameter(torch.empty(
            self.num_primary_caps,
            self.num_classes,
            self.num_in_features,
            self.num_out_features,
        ))
        # 初始化张量,正态分布
        nn.init.normal_(self.W, mean=0.0, std=0.01)
    def forward(self,x):
        votes=torch.einsum('bid,ijdh->bijh',x,self.W)
        return votes

# -----------------实现动态路由---------------------------------
class dynamicRouting(nn.Module):
    def __init__(self, num_iterations=3):
        super(dynamicRouting, self).__init__()
        self.num_iterations = num_iterations
        self.squash = squash()

    def forward(self, votes):
        # 获取预测向量的维度
        B, num_primary, num_classes, digit_dim = votes.shape
        # 初始化路由logits
        b = votes.new_zeros(B, num_primary, num_classes)
        # 开始动态路由
        for r in range(self.num_iterations):
            # 计算耦合系数
            c = torch.softmax(b, dim=-1)
            # 加权求和
            s = torch.sum(c.unsqueeze(-1) * votes, dim=1)
            # Squash压缩
            v = self.squash(s)
            # 更新路由logits
            if r < self.num_iterations - 1:
                agreement = torch.sum(
                    votes * v.unsqueeze(1),
                    dim=-1
                )
                b = b + agreement
        return v
#最终分类
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

    def forward(self, x):
        # 计算每个高级胶囊的L2范数
        out = torch.norm(x, p=2, dim=-1)
        return out
#注意力胶囊
class AttentionCapsNet(nn.Module):
    def __init__(self,num_features:int=74,num_classes:int=15):
        super(AttentionCapsNet, self).__init__()
        self.conv1=ConvBlock(num_features=num_features)
        self.attention=AttentionChanel()
        self.capsuleW=capsulW(num_classes=num_classes)
        self.primaryCaps=PrimaryCaps(in_channels=32)
        self.dynamicRouting=dynamicRouting()
        self.classification=Classification()
    def forward(self,x):
        x=self.conv1(x)
        x=self.attention(x)
        x=self.primaryCaps(x)
        votes= self.capsuleW(x)
        x=self.dynamicRouting(votes)
        scores=self.classification(x)
        # 返回原始胶囊长度，训练时交给 CrossEntropyLoss，不在这里做 Softmax。
        return scores





