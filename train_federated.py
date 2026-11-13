from pathlib import Path
import csv
import json
import random
import threading

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Subset

from ensemble import (
    EnsembleModel,
    MetaClassifier,
    build_dataloaders,
    evaluate_classifier,
    make_meta_data,
    train_base_model,
    train_classifier,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "preprocessed_ml_70_10_20" / "processed_data.pt"
RESULT_DIR = ROOT / "result" / "ML"
NUM_CLIENTS = 3
BASE_EPOCHS = 100
BASE_PATIENCE = 15  # 验证集连续15轮没有提升就停止；设为0则训练满BASE_EPOCHS。
NUM_ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LR = 1e-3
SERVER_LR = 1.0
SERVER_MOMENTUM = 0.9
USE_CUDNN = True  # 当前Windows环境下先关闭cuDNN加速，基础模型仍使用CUDA。
SEED = 42
SERVER_ADDRESS = "127.0.0.1:8080"
NUM_CLASSES = 15
BASE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
META_DEVICE = torch.device("cpu")


#-------------------------------客户端数据划分---------------------------------------

def split_client_indices(dataset, seed):
    # 1. 各类别分别打乱，再平均分给客户端，索引相对于当前数据集。
    if isinstance(dataset, Subset):
        labels = dataset.dataset.tensors[1][dataset.indices].numpy()
    else:
        labels = dataset.tensors[1].numpy()
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(NUM_CLIENTS)]
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        for client_id, part in enumerate(np.array_split(indices, NUM_CLIENTS)):
            client_indices[client_id].extend(part.tolist())
    for indices in client_indices:
        rng.shuffle(indices)
    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


def make_loader(dataset, shuffle=False, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def save_config_and_splits(datasets, client_indices, class_names, feature_names):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    saved_indices = {}
    for split_name, dataset in datasets.items():
        if isinstance(dataset, Subset):
            source_indices = np.asarray(dataset.indices, dtype=np.int64)
        else:
            source_indices = np.arange(len(dataset), dtype=np.int64)
        for client_id in range(NUM_CLIENTS):
            saved_indices[f"client_{client_id}_{split_name}"] = (
                source_indices[client_indices[split_name][client_id]]
            )
    np.savez(RESULT_DIR / "client_split_indices.npz", **saved_indices)

    config = {
        "data_path": str(DATA_PATH),
        "dataset_name": "ML-EdgeIIoT-dataset.csv",
        "input_size": len(feature_names),
        "num_classes": len(class_names),
        "num_clients": NUM_CLIENTS,
        "base_epochs": BASE_EPOCHS,
        "base_patience": BASE_PATIENCE,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "server_learning_rate": SERVER_LR,
        "server_momentum": SERVER_MOMENTUM,
        "server_address": SERVER_ADDRESS,
        "seed": SEED,
        "base_device": str(BASE_DEVICE),
        "use_cudnn": USE_CUDNN,
        "meta_device": str(META_DEVICE),
        "flower_version": fl.__version__,
        "torch_version": str(torch.__version__),
        "class_names": class_names,
        "feature_names": feature_names,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "client_split_sizes": {
            f"client_{client_id}": {
                name: len(indices[client_id])
                for name, indices in client_indices.items()
            }
            for client_id in range(NUM_CLIENTS)
        },
        "index_spaces": {
            "base_train": "X_train / Y_train 原始行号",
            "base_val": "X_val / Y_val 原始行号",
            "meta_train": "X_val / Y_val 原始行号",
            "meta_val": "X_val / Y_val 原始行号",
            "test": "X_test / Y_test 原始行号",
        },
        "paper_protocol": {
            "base_models": "每客户端独立预训练 RESAE、AttentionCapsNet、BiLSTM，选择本地最优后冻结；基模型不参与聚合",
            "meta_features": "拼接本地三个基模型的15类预测，共45维；顺序 AttentionCapsNet、RESAE、BiLSTM",
            "aggregation": "仅 MetaClassifier 通过 Flower FedAvgM 按本地元训练样本数聚合，每轮所有客户端参加",
            "local_optimizer": "Adam，每个联邦轮次重新创建优化器",
            "data_split": "保留当前预处理70%训练/10%验证/20%测试；10%验证再分为5%基础验证、2.5%元训练、2.5%元验证；区别于论文80%/20%",
            "client_partition": "每份数据分别按类别打乱后 np.array_split，固定互斥IID划分，未引入non-IID",
            "evaluation": "单机模拟中，服务器仅为记录与选择模型，读取拼接的客户端缓存元训练/元验证特征；独立测试只在联邦训练结束后使用",
            "runs": "本次仅运行一次模拟，并非论文的10次重复实验",
            "comparison": "同一测试样本上比较三个客户端本地基模型与共享全局融合头；未生成论文CNN/DNN/RNN/CNN_BiLSTM基线结果",
        },
        "metric_definitions": {
            "accuracy": "全局多分类准确率 = confusion_matrix.trace() / N",
            "per_class_accuracy": "one-vs-rest: (TP + TN) / N",
            "specificity": "TN / (TN + FP)",
            "macro_metrics": "固定全部15类取算术平均，分母为0时该类指标取0",
        },
        "loss_notes": {
            "resae_training": "沿用当前 ResAEloss: CE + 0.2*MSE，未认定为论文原始损失",
            "capsnet_training": "CrossEntropyLoss(log(capsule_length + 1e-8), y)",
            "bilstm_meta_training": "CrossEntropyLoss",
            "base_test": "三个基模型仅导出分类指标，不把归一化概率NLL写成其训练损失",
        },
    }
    (RESULT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


#-------------------------------本地预训练与特征缓存---------------------------------

def prepare_clients(datasets, client_indices):
    num_features = datasets["base_train"].tensors[0].shape[1]
    client_data = []
    for client_id in range(NUM_CLIENTS):
        print(f"\nClient {client_id}: 预训练本地基础模型，device={BASE_DEVICE}")
        torch.manual_seed(SEED + client_id)
        client_dir = RESULT_DIR / "clients" / f"client_{client_id}"
        subsets = {
            name: Subset(dataset, client_indices[name][client_id].tolist())
            for name, dataset in datasets.items()
        }
        train_loader = make_loader(
            subsets["base_train"], shuffle=True, seed=SEED + client_id
        )
        val_loader = make_loader(subsets["base_val"])
        model = EnsembleModel(num_features=num_features).to(BASE_DEVICE)
        for base_model, model_type, directory in [
            (model.resae, "resae", "resae"),
            (model.attention_capsnet, "capsnet", "attentioncapsnet"),
            (model.BiLstm, "bilstm", "bilstm"),
        ]:
            train_base_model(
                base_model, model_type, train_loader, val_loader, BASE_DEVICE,
                epochs=BASE_EPOCHS, lr=LR,
                save_path=client_dir / directory / "best_model.pt",
                patience=BASE_PATIENCE,
            )

        # 2. 返回的基础模型已经加载本地best；冻结并缓存训练、验证特征。
        model.freeze_base_models()
        meta_train_data = make_meta_data(
            model, make_loader(subsets["meta_train"]), BASE_DEVICE
        )
        meta_val_data = make_meta_data(
            model, make_loader(subsets["meta_val"]), BASE_DEVICE
        )
        model.cpu()
        client_data.append({
            "client_id": client_id,
            "base_model": model,
            "meta_train_data": meta_train_data,
            "meta_val_data": meta_val_data,
            "test_dataset": subsets["test"],
            "test_indices": client_indices["test"][client_id],
        })
    return client_data


#-------------------------------Flower客户端----------------------------------------

def get_parameters(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_parameters(model, parameters):
    state_dict = {
        name: torch.from_numpy(value.copy())
        for name, value in zip(model.state_dict().keys(), parameters)
    }
    model.load_state_dict(state_dict)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_data, val_data, initial_parameters):
        self.client_id = client_id
        self.model = MetaClassifier().to(META_DEVICE)
        set_parameters(self.model, initial_parameters)
        self.train_loader = make_loader(
            train_data, shuffle=True, seed=SEED + client_id
        )
        self.val_loader = make_loader(val_data)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # 3. 只接收和训练融合头，不加载或覆盖本地基础模型。
        set_parameters(self.model, parameters)
        metrics, epoch_history = train_classifier(
            self.model, self.train_loader, META_DEVICE,
            epochs=config["local_epochs"], lr=config["lr"],
        )
        return get_parameters(self.model), metrics["num_examples"], {
            "client_id": self.client_id,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "epoch_history": json.dumps(epoch_history),
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics, _, _ = evaluate_classifier(
            self.model, self.val_loader, META_DEVICE
        )
        return metrics["loss"], metrics["samples"], {
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]
        }


class LoggedFedAvgM(fl.server.strategy.FedAvgM):
    def __init__(self, client_history_path, **kwargs):
        super().__init__(**kwargs)
        self.client_history_path = client_history_path

    def aggregate_fit(self, server_round, results, failures):
        # 4. 权重与服务器动量完全交给Flower，只额外记录本地训练历史。
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        with self.client_history_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            for _, fit_res in sorted(
                results, key=lambda result: int(result[1].metrics["client_id"])
            ):
                client_id = int(fit_res.metrics["client_id"])
                for row in json.loads(fit_res.metrics["epoch_history"]):
                    writer.writerow([
                        server_round, client_id, row["epoch"],
                        row["loss"], row["accuracy"], row["num_examples"],
                    ])
        return parameters, metrics


def fit_config(server_round):
    return {"round": server_round, "local_epochs": LOCAL_EPOCHS, "lr": LR}


#-------------------------------全局训练记录与最优模型-------------------------------

def run_federated_training(client_data):
    proposed_dir = RESULT_DIR / "Proposed"
    proposed_dir.mkdir(parents=True, exist_ok=True)
    history_path = proposed_dir / "history.csv"
    client_history_path = proposed_dir / "client_history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow([
            "round", "train_loss", "train_accuracy", "val_loss",
            "val_accuracy", "val_macro_f1",
        ])
    with client_history_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow([
            "round", "client_id", "local_epoch", "train_loss",
            "train_accuracy", "num_examples",
        ])

    train_loader = make_loader(ConcatDataset([
        client["meta_train_data"] for client in client_data
    ]))
    val_loader = make_loader(ConcatDataset([
        client["meta_val_data"] for client in client_data
    ]))
    torch.manual_seed(SEED)
    global_model = MetaClassifier().to(META_DEVICE)
    initial_parameters = get_parameters(global_model)
    best_f1 = -1.0
    best_round = 0

    def evaluate_global(server_round, parameters, config):
        nonlocal best_f1, best_round
        # 5. 不把初始化round=0计入训练曲线或最优模型选择。
        if server_round == 0:
            return None
        set_parameters(global_model, parameters)
        train_metrics, _, _ = evaluate_classifier(global_model, train_loader, META_DEVICE)
        val_metrics, _, _ = evaluate_classifier(global_model, val_loader, META_DEVICE)
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                server_round, train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"],
            ])

        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]
            best_round = server_round
        checkpoint = {
            "model_state_dict": global_model.state_dict(),
            "dataset_name": "ML-EdgeIIoT-dataset.csv",
            "input_size": client_data[0]["base_model"].num_features,
            "num_classes": NUM_CLASSES,
            "round": server_round,
            "best_round": best_round,
            "best_macro_f1": best_f1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "server_learning_rate": SERVER_LR,
            "server_momentum": SERVER_MOMENTUM,
        }
        if strategy.momentum_vector is not None:
            checkpoint["momentum_vector"] = [
                torch.from_numpy(value.copy()) for value in strategy.momentum_vector
            ]
        torch.save(checkpoint, proposed_dir / "latest_model.pt")
        if is_best:
            torch.save(checkpoint, proposed_dir / "best_model.pt")

        print(
            f"Round [{server_round}/{NUM_ROUNDS}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} "
            f"Best F1: {best_f1:.4f}"
        )
        return val_metrics["loss"], {
            "accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"]
        }

    strategy = LoggedFedAvgM(
        client_history_path=client_history_path,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_global,
        fit_metrics_aggregation_fn=lambda metrics: {},
        accept_failures=False,
        server_learning_rate=SERVER_LR,
        server_momentum=SERVER_MOMENTUM,
    )
    clients = [
        FlowerClient(
            client["client_id"], client["meta_train_data"],
            client["meta_val_data"], initial_parameters,
        )
        for client in client_data
    ]
    threads = [
        threading.Thread(
            target=fl.client.start_client,
            kwargs={
                "server_address": SERVER_ADDRESS,
                "client": client.to_client(),
                "insecure": True,
            },
            daemon=True,
        )
        for client in clients
    ]
    # 6. 客户端使用Flower连接重试，服务器在主线程运行。
    for thread in threads:
        thread.start()
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    for thread in threads:
        thread.join()

    checkpoint = torch.load(
        proposed_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    global_model.load_state_dict(checkpoint["model_state_dict"])
    global_model.eval()
    return global_model, checkpoint


#-------------------------------真实测试结果保存-------------------------------------

def save_test_results(
    model_name, labels, predictions, sample_indices, client_ids,
    class_names, extra_metrics,
):
    output_dir = RESULT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    class_ids = list(range(NUM_CLASSES))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    samples = int(matrix.sum())
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = samples - tp - fp - fn
    support = tp + fn
    numerators = np.stack([tp + tn, tp, tp, tn, 2 * tp])
    denominators = np.stack([
        np.full(NUM_CLASSES, samples), tp + fp, tp + fn, tn + fp,
        2 * tp + fp + fn,
    ])
    values = np.divide(
        numerators, denominators, out=np.zeros_like(numerators, dtype=float),
        where=denominators != 0,
    )
    accuracy, precision, recall, specificity, f1_score = values

    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "client_id", "y_true", "y_pred"])
        writer.writerows(zip(sample_indices, client_ids, labels, predictions))
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true / predicted"] + class_names)
        for name, row in zip(class_names, matrix.tolist()):
            writer.writerow([name] + row)
    report = classification_report(
        labels, predictions, labels=class_ids,
        target_names=class_names, digits=4, zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "class_name", "support", "accuracy", "precision", "recall",
            "specificity", "f1_score", "tp", "fp", "fn", "tn",
        ])
        for index, class_name in enumerate(class_names):
            writer.writerow([
                class_name, int(support[index]), accuracy[index], precision[index],
                recall[index], specificity[index], f1_score[index], int(tp[index]),
                int(fp[index]), int(fn[index]), int(tn[index]),
            ])

    metrics = {
        "model": model_name,
        "samples": samples,
        "accuracy": float(tp.sum() / samples),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_specificity": float(specificity.mean()),
        "macro_f1": float(f1_score.mean()),
        "per_class_accuracy_definition": "one-vs-rest: (TP + TN) / N",
    }
    metrics.update(extra_metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{model_name} Test Acc: {metrics['accuracy']:.4f} Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics


def evaluate_all_models(global_model, checkpoint, client_data, class_names):
    sample_indices = []
    client_ids = []
    labels = []
    predictions = {
        "RESAE": [], "AttentionCapsNet": [], "BiLSTM": [], "Proposed": []
    }
    total_loss = 0.0
    for client in client_data:
        # 7. 测试特征到此时才提取，仍由该客户端自己的冻结基模型生成。
        base_model = client["base_model"].to(BASE_DEVICE)
        meta_test_data = make_meta_data(
            base_model, make_loader(client["test_dataset"]), BASE_DEVICE
        )
        base_model.cpu()
        metrics, client_labels, client_predictions = evaluate_classifier(
            global_model, make_loader(meta_test_data), META_DEVICE
        )
        sample_indices.extend(client["test_indices"].tolist())
        client_ids.extend([client["client_id"]] * len(client_labels))
        labels.extend(client_labels)
        predictions["Proposed"].extend(client_predictions)
        total_loss += metrics["loss"] * metrics["samples"]

        # 8. 三个概率块的argmax就是各本地基模型预测，不增加测试前向。
        features = meta_test_data.tensors[0]
        for block, model_name in enumerate(["AttentionCapsNet", "RESAE", "BiLSTM"]):
            predictions[model_name].extend(
                features[:, block * NUM_CLASSES:(block + 1) * NUM_CLASSES]
                .argmax(dim=1).tolist()
            )

    # 9. 四个模型使用同一组原始测试行号、标签、客户端归属与排序。
    order = np.argsort(sample_indices)
    sample_indices = np.asarray(sample_indices)[order]
    client_ids = np.asarray(client_ids)[order]
    labels = np.asarray(labels)[order]
    for model_name, model_predictions in predictions.items():
        extra_metrics = {}
        if model_name == "Proposed":
            extra_metrics = {
                "loss": total_loss / len(labels),
                "best_round": checkpoint["round"],
                "best_val_macro_f1": checkpoint["best_macro_f1"],
            }
        save_test_results(
            model_name, labels, np.asarray(model_predictions)[order],
            sample_indices, client_ids, class_names, extra_metrics,
        )


#-------------------------------主流程-----------------------------------------------

def main():
    torch.backends.cudnn.enabled = USE_CUDNN
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"基础模型训练与特征提取: {BASE_DEVICE}; 联邦融合头: {META_DEVICE}")

    loaders = build_dataloaders(DATA_PATH, BATCH_SIZE, SEED)
    split_names = ["base_train", "base_val", "meta_train", "meta_val", "test"]
    datasets = {name: loader.dataset for name, loader in zip(split_names, loaders)}
    class_names = datasets["base_train"].class_names
    feature_names = datasets["base_train"].feature_names
    print(f"ML 数据：{len(datasets['base_train']):,} 条训练样本，"
          f"{len(feature_names)} 个特征，{len(class_names)} 类")
    client_indices = {
        name: split_client_indices(dataset, SEED + index)
        for index, (name, dataset) in enumerate(datasets.items())
    }
    save_config_and_splits(datasets, client_indices, class_names, feature_names)
    client_data = prepare_clients(datasets, client_indices)
    global_model, checkpoint = run_federated_training(client_data)
    evaluate_all_models(global_model, checkpoint, client_data, class_names)
    print(f"训练记录、模型和四个模型的真实测试结果已保存到：{RESULT_DIR}")


if __name__ == "__main__":
    main()
