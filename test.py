"""从真实训练集每类取两条样本，检查 CUDA 上的一步训练和融合特征。"""

import torch
import torch.nn as nn

from ensemble import EnsembleModel, ResAEloss


def check_tensor(name, tensor, shape):
    assert tuple(tensor.shape) == shape, f"{name} 形状错误：{tuple(tensor.shape)}"
    assert torch.isfinite(tensor).all().item(), f"{name} 含有 NaN 或 Inf"


def check_gradients(name, model):
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name}.{parameter_name} 没有梯度"
            assert torch.isfinite(parameter.grad).all().item(), (
                f"{name}.{parameter_name} 梯度含有 NaN 或 Inf"
            )


def check_parameters(name, model):
    for parameter_name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all().item(), (
            f"{name}.{parameter_name} 更新后含有 NaN 或 Inf"
        )


def main():
    import train_federated

    torch.backends.cudnn.enabled = train_federated.USE_CUDNN
    torch.manual_seed(train_federated.SEED)
    device = train_federated.BASE_DEVICE
    if device.type != "cuda":
        raise RuntimeError("本检查需要 CUDA；当前 BASE_DEVICE 不是 CUDA")

    # 1. 只使用训练集；每个类别取两条真实样本，不生成随机输入。
    data = torch.load(
        train_federated.DATA_PATH, map_location="cpu", weights_only=True
    )
    x_train, y_train = data["X_train"], data["Y_train"]
    classes = torch.unique(y_train, sorted=True)
    assert torch.equal(classes, torch.arange(15)), "训练集必须包含编号 0 到 14 的15类"
    sample_indices = []
    for class_id in classes:
        indices = torch.nonzero(y_train == class_id, as_tuple=True)[0][:2]
        assert len(indices) == 2, f"类别 {class_id.item()} 不足两条训练样本"
        sample_indices.append(indices)
    sample_indices = torch.cat(sample_indices)
    num_features = x_train.shape[1]
    x = x_train[sample_indices].to(device=device, dtype=torch.float32)
    y = y_train[sample_indices].to(device=device, dtype=torch.long)
    check_tensor("训练输入", x, (30, num_features))
    check_tensor("训练标签", y, (30,))
    print(
        f"设备：{device}；cuDNN：{torch.backends.cudnn.enabled}；"
        f"真实训练样本：30；特征数：{num_features}", flush=True
    )

    # 2. 三个基础模型分别执行一次当前损失的反向传播和 Adam 更新。
    model = EnsembleModel(num_features=num_features).to(device)
    ce = nn.CrossEntropyLoss()
    resae_loss = ResAEloss()
    base_models = [
        ("RESAE", model.resae),
        ("AttentionCapsNet", model.attention_capsnet),
        ("BiLSTM", model.BiLstm),
    ]
    for name, base_model in base_models:
        base_model.train()
        optimizer = torch.optim.Adam(base_model.parameters(), lr=train_federated.LR)
        optimizer.zero_grad()
        if name == "RESAE":
            mid, reconstructed, logits = base_model(x)
            check_tensor("RESAE 潜在特征", mid, (30, 32))
            check_tensor("RESAE 重建", reconstructed, (30, num_features))
            loss, cls_loss, rec_loss = resae_loss(x, y, logits, reconstructed)
            check_tensor("RESAE 分类损失", cls_loss, ())
            check_tensor("RESAE 重建损失", rec_loss, ())
        elif name == "AttentionCapsNet":
            scores = base_model(x)
            check_tensor("胶囊长度", scores, (30, 15))
            logits = torch.log(scores + 1e-8)
            loss = ce(logits, y)
        else:
            logits = base_model(x)
            loss = ce(logits, y)
        check_tensor(f"{name} 输出", logits, (30, 15))
        check_tensor(f"{name} 损失", loss, ())
        loss.backward()
        check_gradients(name, base_model)
        optimizer.step()
        check_parameters(name, base_model)
        print(
            f"{name}：输出 (30, 15)，loss={loss.item():.6f}；"
            "梯度与参数有限，Adam 一步通过", flush=True
        )

    # 3. 冻结基模型，检查45维融合特征，再单独更新一次元分类器。
    model.freeze_base_models()
    features = model.extract_features(x)
    check_tensor("融合特征", features, (30, 45))
    assert not features.requires_grad, "冻结后的融合特征不应保留梯度"
    for name, base_model in base_models:
        assert not any(parameter.requires_grad for parameter in base_model.parameters()), (
            f"{name} 未完全冻结"
        )
    model.classifier.train()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=train_federated.LR)
    optimizer.zero_grad()
    logits = model.classifier(features)
    check_tensor("元分类器输出", logits, (30, 15))
    loss = ce(logits, y)
    check_tensor("元分类器损失", loss, ())
    loss.backward()
    check_gradients("元分类器", model.classifier)
    optimizer.step()
    check_parameters("元分类器", model.classifier)
    torch.cuda.synchronize(device)
    print(
        f"融合特征：(30, 45)；元分类器输出：(30, 15)，loss={loss.item():.6f}；"
        "梯度与参数有限，Adam 一步通过", flush=True
    )
    print("CUDA 一步检查全部通过；未保存检查点，未使用测试集。", flush=True)


if __name__ == "__main__":
    main()

