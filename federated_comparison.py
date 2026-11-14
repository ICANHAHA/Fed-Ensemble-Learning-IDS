"""CNN、RNN 与集成方法的联邦对照实验（单进程模拟）。

默认使用完整数据、3 个客户端、基模型最多 100 epoch、50 个联邦轮次。
默认使用 Flower 的 FedAvgM 策略；--backend manual 便于学习与离线检查。
客户端按顺序运行在同一进程中，模拟参数下发、训练与聚合，不建立网络连接。
"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader, Subset

from comparison_models import CNN1D, RNNClassifier
from ensemble import (
    EnsembleModel, MetaClassifier, build_dataloaders,
    evaluate_classifier, make_meta_data, train_base_model, train_classifier,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "preprocessed_ml_70_10_20" / "processed_data.pt"
OUTPUT_ROOT = ROOT / "result" / "federated_comparison"
METHODS = ("Proposed", "CNN", "RNN")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--backend", choices=("flower", "manual"), default="flower")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--base-epochs", type=int, default=100)
    parser.add_argument("--base-patience", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--server-momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample-limit", type=int, default=0,
                        help="每个客户端每份数据的样本上限；0 使用完整数据")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.clients < 2 or args.base_epochs < 1 or args.rounds < 1 or args.local_epochs < 1:
        parser.error("客户端至少 2 个，基模型 epoch、通信轮数和本地 epoch 均至少为 1")
    if args.batch_size < 2 or args.sample_limit < 0 or args.sample_limit == 1:
        parser.error("batch-size 至少为 2；sample-limit 为 0（完整数据）或至少为 2")
    if args.base_patience < 0 or args.lr <= 0 or args.server_lr <= 0:
        parser.error("早停等待次数不能为负，学习率必须大于 0")
    if not 0 <= args.server_momentum < 1:
        parser.error("server-momentum 应在 [0, 1) 范围内")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods 中不能重复选择同一方法")
    return args


def make_loader(dataset, args, shuffle=False, seed=None):
    generator = torch.Generator().manual_seed(args.seed if seed is None else seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator)


def split_clients(dataset, num_clients, seed, limit=0):
    """随机分配互不重复的样本；这是近似 IID，不按类别分层。"""
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    sizes = [len(dataset) // num_clients] * num_clients
    sizes[-1] += len(dataset) % num_clients
    parts, offset = [], 0
    for size in sizes:
        selected = indices[offset:offset + size]
        offset += size
        if limit:
            selected = selected[:limit]
        if len(selected) < 2:
            raise ValueError("每个客户端每份数据至少需要 2 条样本")
        parts.append(Subset(dataset, selected))
    return parts


def get_parameters(model):
    """与 Flower 一致：按 state_dict 顺序转成 NumPy 数组列表。"""
    arrays = [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]
    if not all(np.issubdtype(a.dtype, np.floating) for a in arrays):
        raise TypeError("本示例聚合对象只支持浮点模型参数；请单独处理整数缓冲区")
    return arrays


def set_parameters(model, arrays):
    state = model.state_dict()
    if len(state) != len(arrays):
        raise ValueError("参数个数与模型结构不一致")
    updated = {}
    for (name, value), array in zip(state.items(), arrays):
        tensor = torch.from_numpy(array.copy())
        if tensor.shape != value.shape:
            raise ValueError(f"参数形状不一致：{name}")
        updated[name] = tensor.to(dtype=value.dtype)
    model.load_state_dict(updated)


def new_model(method, input_size, num_classes):
    if method == "Proposed":
        return MetaClassifier(input_size=3 * num_classes, num_classes=num_classes)
    if method == "CNN":
        return CNN1D(input_size=input_size, num_classes=num_classes)
    if method == "RNN":
        return RNNClassifier(input_size=input_size, num_classes=num_classes)
    raise ValueError(f"未知方法：{method}")


class FedAvgMAggregator:
    """只负责服务器聚合；不保存客户端数据。"""

    def __init__(self, initial_arrays, args):
        self.args = args
        self.current = [a.copy() for a in initial_arrays]
        self.velocity = [np.zeros_like(a) for a in initial_arrays]
        self.strategy = None
        if args.backend == "flower":
            try:
                import flwr as fl
            except ImportError as exc:
                raise RuntimeError("Flower 或 gRPC 导入失败，请检查 torch_cuda 环境") from exc
            self.fl = fl
            self.strategy = fl.server.strategy.FedAvgM(
                initial_parameters=fl.common.ndarrays_to_parameters(self.current),
                server_learning_rate=args.server_lr,
                server_momentum=args.server_momentum,
            )

    def aggregate(self, round_id, client_results):
        """每项为（本地参数、训练样本数）；按样本数加权。"""
        if not client_results or any(n <= 0 for _, n in client_results):
            raise ValueError("至少需要一个包含样本的客户端")
        if self.strategy is not None:
            # 直接调用 Flower 策略；None 是本机模拟中无需使用的客户端代理。
            status = self.fl.common.Status(code=self.fl.common.Code.OK, message="")
            results = [
                (None, self.fl.common.FitRes(
                    status=status,
                    parameters=self.fl.common.ndarrays_to_parameters(arrays),
                    num_examples=count,
                    metrics={},
                ))
                for arrays, count in client_results
            ]
            parameters, _ = self.strategy.aggregate_fit(round_id, results, [])
            if parameters is None:
                raise RuntimeError("Flower 没有返回聚合参数")
            self.current = self.fl.common.parameters_to_ndarrays(parameters)
        else:
            # 同一方向约定：delta=客户端平均参数-旧全局参数。
            total = sum(n for _, n in client_results)
            average = [sum(arrays[i] * n for arrays, n in client_results) / total
                       for i in range(len(self.current))]
            self.velocity = [self.args.server_momentum * v + (avg - old)
                             for v, avg, old in zip(self.velocity, average, self.current)]
            self.current = [old + self.args.server_lr * v
                            for old, v in zip(self.current, self.velocity)]
        return [a.copy() for a in self.current]


def source_indices(dataset):
    """把多层 Subset 的索引转换回其最初数据集的行号。"""
    indices = np.arange(len(dataset), dtype=np.int64)
    while isinstance(dataset, Subset):
        indices = np.asarray(dataset.indices, dtype=np.int64)[indices]
        dataset = dataset.dataset
    return indices


def prepare_splits(args, run_dir):
    loaders = build_dataloaders(args.data_path, batch_size=args.batch_size, seed=args.seed)
    names = ("base_train", "base_val", "meta_train", "meta_val", "test")
    datasets = {name: loader.dataset for name, loader in zip(names, loaders)}
    class_names = list(datasets["base_train"].class_names)
    feature_names = list(datasets["base_train"].feature_names)
    if len(class_names) != 15:
        raise ValueError("现有 EnsembleModel 和指标函数固定为 15 类")
    parts = {
        name: split_clients(dataset, args.clients, args.seed + i, args.sample_limit)
        for i, (name, dataset) in enumerate(datasets.items())
    }
    # 分别记录客户端索引：base_train/test 相对于各自数据集，其余相对于原验证集。
    indexes = {f"{name}_client_{i}": source_indices(part)
               for name, client_parts in parts.items() for i, part in enumerate(client_parts)}
    np.savez(run_dir / "client_split_indices.npz", **indexes)
    print("数据特征数:", len(feature_names), "类别数:", len(class_names))
    for name, client_parts in parts.items():
        print(f"  {name}: {[len(p) for p in client_parts]}")
    return parts, class_names, feature_names


def prepare_proposed(parts, input_size, args, run_dir, device):
    """每个客户端训练三个基模型，冻结后生成融合训练和验证特征。"""
    client_data = []
    for client_id in range(args.clients):
        print(f"\n[Proposed] 客户端 {client_id}: 开始本地基模型训练", flush=True)
        torch.manual_seed(args.seed + client_id)
        ensemble = EnsembleModel(num_features=input_size).to(device)
        train_loader = make_loader(parts["base_train"][client_id], args, True, args.seed + client_id)
        val_loader = make_loader(parts["base_val"][client_id], args)
        client_dir = run_dir / "Proposed" / "clients" / f"client_{client_id}"
        for model, model_type, folder in (
            (ensemble.resae, "resae", "resae"),
            (ensemble.attention_capsnet, "capsnet", "attentioncapsnet"),
            (ensemble.BiLstm, "bilstm", "bilstm"),
        ):
            train_base_model(
                model=model, model_type=model_type, train_loader=train_loader,
                val_loader=val_loader, device=device, epochs=args.base_epochs,
                patience=args.base_patience, lr=args.lr,
                save_path=client_dir / folder / "best_model.pt",
            )
        ensemble.freeze_base_models()
        meta_train = make_meta_data(
            ensemble, make_loader(parts["meta_train"][client_id], args), device)
        meta_val = make_meta_data(
            ensemble, make_loader(parts["meta_val"][client_id], args), device)
        client_data.append({"train": meta_train, "val": meta_val, "base_model": ensemble.cpu()})
    return client_data


def prepare_baseline(parts, args):
    """CNN/RNN 使用基模型训练集和元训练集的原始数据并集。"""
    return [
        {
            "train": ConcatDataset([parts["base_train"][i], parts["meta_train"][i]]),
            "val": parts["meta_val"][i],
        }
        for i in range(args.clients)
    ]


def evaluate_clients(model, datasets, args, device, num_classes):
    """返回合并预测后的全局指标；不平均各客户端 Macro-F1。"""
    labels, predictions, loss_sum = [], [], 0.0
    for dataset in datasets:
        metrics, y_true, y_pred = evaluate_classifier(model, make_loader(dataset, args), device)
        labels.extend(y_true)
        predictions.extend(y_pred)
        loss_sum += metrics["loss"] * metrics["samples"]
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    return {"loss": loss_sum / len(labels), "accuracy": float(tp.sum() / len(labels)),
            "macro_f1": float(f1.mean())}


def run_federated(method, client_data, input_size, num_classes, args, run_dir, device):
    """一轮：下发相同参数 -> 本地训练 -> FedAvgM 聚合 -> 本地验证。"""
    output_dir = run_dir / method
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    global_model = new_model(method, input_size, num_classes)
    aggregator = FedAvgMAggregator(get_parameters(global_model), args)
    best_f1, best_round, best_state = -1.0, 0, None
    history = []
    for round_id in range(1, args.rounds + 1):
        local_results = []
        train_samples, train_loss, train_correct = 0, 0.0, 0.0
        for client_id, data in enumerate(client_data):
            # 所有客户端从这一轮相同的全局参数开始。
            torch.manual_seed(args.seed + round_id * 1000 + client_id)
            model = new_model(method, input_size, num_classes).to(device)
            set_parameters(model, aggregator.current)
            loader = make_loader(data["train"], args, True, args.seed + round_id * 1000 + client_id)
            metrics, _ = train_classifier(model, loader, device, epochs=args.local_epochs, lr=args.lr)
            count = metrics["num_examples"]
            local_results.append((get_parameters(model), count))
            train_samples += count
            train_loss += metrics["loss"] * count
            train_correct += metrics["accuracy"] * count
        # 不把 CNN、RNN、融合头的参数混合；每个方法各有自己的聚合器。
        aggregated = aggregator.aggregate(round_id, local_results)
        set_parameters(global_model, aggregated)
        val = evaluate_clients(global_model, [d["val"] for d in client_data], args, device, num_classes)
        row = {"round": round_id, "train_loss": train_loss / train_samples,
               "train_accuracy": train_correct / train_samples, "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]}
        history.append(row)
        state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        torch.save({"model_state_dict": state, "round": round_id}, output_dir / "latest_model.pt")
        if val["macro_f1"] > best_f1:
            best_f1, best_round, best_state = val["macro_f1"], round_id, state
            torch.save({"model_state_dict": state, "round": round_id,
                        "val_macro_f1": best_f1}, output_dir / "best_model.pt")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"[{method}] Round {round_id}/{args.rounds} "
              f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
              f"val loss={val['loss']:.4f} acc={val['accuracy']:.4f} "
              f"macro-F1={val['macro_f1']:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"{method} 未产生可用的全局模型")
    global_model.load_state_dict(best_state)
    return global_model, best_round, best_f1, history
