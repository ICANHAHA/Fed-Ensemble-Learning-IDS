import torch
import torch.nn as nn
from models.AttentionCapsnet import AttentionCapsNet
from models.BiLSTM import BiLSTM
from models.RESAE import ResAutoEncoder, ResAEloss
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import csv
import json


def build_dataloaders(data_path, batch_size=128, seed=42):
    # 1. 加载预处理数据
    data = torch.load(
        data_path,
        map_location="cpu",
        weights_only=True
    )
    # 2. 创建三个 TensorDataset
    train_dataset = TensorDataset(
        data["X_train"],
        data["Y_train"]
    )
    val_dataset = TensorDataset(
        data["X_val"],
        data["Y_val"]
    )
    test_dataset = TensorDataset(
        data["X_test"],
        data["Y_test"]
    )
    train_dataset.class_names = data["class_names"]
    train_dataset.feature_names = data["feature_names"]
    # 3. 获取验证集标签和索引
    val_labels = data["Y_val"]
    indices = list(range(len(val_dataset)))
    # 4. 第一次划分：Base Val 和剩余数据
    base_val_indices, meta_indices = train_test_split(
        indices,
        test_size=0.5,
        stratify=val_labels.numpy(),
        random_state=seed
    )
    # 5. 第二次划分：Meta Train 和 Meta Val
    meta_train_indices, meta_val_indices = train_test_split(
        meta_indices,
        test_size=0.5,
        stratify=val_labels[meta_indices].numpy(),
        random_state=seed
    )
    # 创建三个子数据集
    base_val_dataset = Subset(
        val_dataset,
        base_val_indices
    )
    meta_train_dataset = Subset(
        val_dataset,
        meta_train_indices
    )
    meta_val_dataset = Subset(
        val_dataset,
        meta_val_indices
    )
    # 6. 创建五个 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    base_val_loader = DataLoader(
        base_val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    meta_train_loader = DataLoader(
        meta_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    meta_val_loader = DataLoader(
        meta_val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    # 7. 返回五个 DataLoader
    return (
        train_loader,
        base_val_loader,
        meta_train_loader,
        meta_val_loader,
        test_loader
    )
#-------------------------------保存训练记录---------------------------------------

def save_history(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)



#-------------------------------基础模型训练---------------------------------------

def train_base_model(
    model,
    model_type,
    train_loader,
    val_loader,
    device,
    epochs=100,
    lr=1e-3,
    save_path="result/ML/base_model/best_model.pt",
    patience=15
):

    # 1. 初始化模型、损失函数和优化器
    model = model.to(device)
    if model_type == "resae":
        # 沿用你当前的 CE + 0.2*MSE 实现，未确认与论文原始损失完全一致。
        criterion = ResAEloss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    wait = 0
    num_classes = 15
    history = []
    # 2. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        train_correct = 0
        for x, y in train_loader:
            # RESAE 使用 BatchNorm，跳过单样本 Batch
            if model_type == "resae" and x.size(0) == 1:
                continue
            x = x.to(device)
            y = y.to(device)
            # 前向传播
            if model_type == "resae":
                mid, reconstructed, logits = model(x)
                loss, cls_loss, rec_loss = criterion(
                    x, y, logits, reconstructed
                )
            elif model_type == "capsnet":
                scores = model(x)
                logits = torch.log(scores + 1e-8)
                loss = criterion(logits, y)
            elif model_type == "bilstm":
                logits = model(x)
                loss = criterion(logits, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_samples += y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_loss /= train_samples
        train_accuracy = train_correct / train_samples
        # 3. 验证模型
        model.eval()
        val_loss = 0.0
        val_samples = 0
        confusion = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
            device=device
        )
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                if model_type == "resae":
                    mid, reconstructed, logits = model(x)
                    loss, cls_loss, rec_loss = criterion(
                        x, y, logits, reconstructed
                    )
                elif model_type == "capsnet":
                    scores = model(x)
                    logits = torch.log(scores + 1e-8)
                    loss = criterion(logits, y)
                elif model_type == "bilstm":
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                val_samples += y.size(0)
                preds = logits.argmax(dim=1)
                indices = y * num_classes + preds
                confusion += torch.bincount(
                    indices,
                    minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
        val_loss /= val_samples
        val_accuracy = confusion.diag().sum().item() / val_samples
        # 4. 计算 Macro-F1
        tp = confusion.diag().float()
        fp = confusion.sum(dim=0).float() - tp
        fn = confusion.sum(dim=1).float() - tp
        f1 = 2 * tp / (
            2 * tp + fp + fn
        ).clamp_min(1)
        val_f1 = f1.mean().item()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1
        })
        save_history(history, save_path.parent)
        # 5. 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_macro_f1": best_f1,
                "input_size": x.size(1),
                "num_classes": num_classes
            }, save_path)
        else:
            wait += 1

        print(
            f"{model_type} Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f} "
            f"Val Macro-F1: {val_f1:.4f} "
            f"Best F1: {best_f1:.4f}"
        )
        if patience > 0 and wait >= patience:
            print(f"验证集连续 {patience} 轮没有提升，停止训练。")
            break
    # 6. 加载最优模型
    checkpoint = torch.load(
        save_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
class MetaClassifier(nn.Module):
    def __init__(self,input_size:int=45,hidden_size:int=64,num_classes:int=15):
        super(MetaClassifier,self).__init__()
        self.lin1=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size,num_classes)
        )
    def forward(self,x):
        x = self.lin1(x)
        return x
class EnsembleModel(nn.Module):
    def __init__(self,num_features:int=74):
        super(EnsembleModel,self).__init__()
        self.num_features = num_features
        self.attention_capsnet=AttentionCapsNet(
            num_features=num_features,
            num_classes=15
        )
        self.resae=ResAutoEncoder(
            input_size=num_features,
            hidden_size=32,
            num_classes=15
        )
        self.BiLstm=BiLSTM(
            num_classes=15,
            hidden_size=64
        )
        self.classifier = MetaClassifier(
            input_size = 45,
            hidden_size = 64,
            num_classes = 15
        )
    @torch.no_grad()
    def extract_features(self,x):
        atten_scores=self.attention_capsnet(x)
        # 胶囊长度按行归一化，作为融合特征。
        atten_probs=atten_scores+1e-8
        atten_probs=atten_probs/atten_probs.sum(dim=1,keepdim=True)

        _,_,resae_scores=self.resae(x)
        resae_probs=torch.softmax(resae_scores,dim=1)

        lstm_scores=self.BiLstm(x)
        lstm_probs=torch.softmax(lstm_scores,dim=1)

        features=torch.cat((atten_probs,resae_probs,lstm_probs),dim=1)
        return features

    def forward(self,x):
        features=self.extract_features(x)
        logits=self.classifier(features)
        return logits

    # 加载预训练模型参数
    def load_pretrained(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        resae_ckpt = torch.load(
            checkpoint_dir / "resae" / "best_model.pt",
            map_location="cpu",
            weights_only=True
        )
        attention_ckpt = torch.load(
            checkpoint_dir / "attentioncapsnet" / "best_model.pt",
            map_location="cpu",
            weights_only=True
        )
        bilstm_ckpt = torch.load(
            checkpoint_dir / "bilstm" / "best_model.pt",
            map_location="cpu",
            weights_only=True
        )
        self.resae.load_state_dict(resae_ckpt["model_state_dict"])
        self.attention_capsnet.load_state_dict(attention_ckpt["model_state_dict"])
        self.BiLstm.load_state_dict(bilstm_ckpt["model_state_dict"])
    #冻结参数
    def freeze_base_models(self):
        self.attention_capsnet.requires_grad_(False)
        self.resae.requires_grad_(False)
        self.BiLstm.requires_grad_(False)

        self.attention_capsnet.eval()
        self.resae.eval()
        self.BiLstm.eval()

#-------------------------------生成融合特征---------------------------------------

def make_meta_data(model, loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            meta_features = model.extract_features(x)
            features.append(meta_features.cpu())
            labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return TensorDataset(features, labels)



#-------------------------------客户端本地训练融合层---------------------------------------

def train_classifier(model, train_loader, device, epochs=1, lr=1e-3):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            epoch_samples += y.size(0)

        history.append({
            "epoch": epoch,
            "loss": epoch_loss / epoch_samples,
            "accuracy": epoch_correct / epoch_samples,
            "num_examples": epoch_samples
        })
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_examples": len(train_loader.dataset)
    }
    return metrics, history


#-------------------------------评估融合层---------------------------------------

def evaluate_classifier(model, loader, device):
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    labels = []
    predictions = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            labels.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "loss": total_loss / len(labels),
        "accuracy": float(np.mean(np.array(labels) == np.array(predictions))),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(15)),
                                   average="macro", zero_division=0)),
        "samples": len(labels)
    }
    return metrics, labels, predictions
