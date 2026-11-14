
import torch
import flwr as fl
from ensemble import (
    EnsembleModel,
    MetaClassifier,
    build_dataloaders,
    evaluate_classifier,
    make_meta_data,
    train_base_model,
    train_classifier,
)
from torch.utils.data import ConcatDataset, DataLoader, Subset
from pathlib import Path
from torch.utils.data import random_split
from datetime import datetime

data_path = (
    Path(__file__).resolve().parent
    / "preprocessed_ml_70_10_20"
    / "processed_data.pt"
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED=42
BATCH_SIZE=128
BASE_EPOCHS = 100
BASE_PATIENCE = 15# 验证集连续15轮没有提升就停止；设为0则训练满BASE_EPOCHS。

(
    base_train_loader,
    base_val_loader,
    meta_train_loader,
    meta_val_loader,
    test_loader,
) = build_dataloaders(
    data_path,
    batch_size=BATCH_SIZE,
    seed=SEED,
)



def make_loader(dataset, shuffle=False, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )
#----------------------------------------客户端---------------------------------------------
#上传参数
def get_parameters(model):
    return[
        value.detach().cpu().numpy().copy() for value in model.state_dict().values()
    ]
#接收参数
def set_parameters(model, parameters):
    state_dict = {
        name: torch.from_numpy(value.copy())
        for name, value in zip(model.state_dict().keys(), parameters)
    }
    model.load_state_dict(state_dict)
class FlowerClient(fl.client.NumPyClient):
    def __init__ (self,client_id, train_data, val_data, initial_parameters):
        self.client_id=client_id
        self.model=MetaClassifier().to(device)
        set_parameters(self.model, initial_parameters)
        self.train_loader=make_loader(
            train_data,
            shuffle=True,
            seed=SEED+client_id
        )
        self.val_loader=make_loader(val_data)
    def fit(self,parameters, config):
        #客户端接收参数
        set_parameters(self.model,parameters)
        #客户端本地训练
        metrics,history=train_classifier(
            self.model,
            self.train_loader,
            device,
            epochs=1,
            lr=1e-3
        )
        return(
            get_parameters(self.model),
            metrics["num_examples"],
            {
                "client_id": self.client_id,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
            }
        )
    def evaluate(self, parameters, config):
        set_parameters(self.model,parameters)
        metrics, labels, predictions=evaluate_classifier(
            self.model,
            self.val_loader,
            device
        )
        return(
            metrics["loss"],
            metrics["samples"],
            {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            },

        )
#--------------------------接下来写服务器----------------------------------
# ---------------------- 服务器配置 ----------------------
NUM_CLIENTS = 3
# 服务器初始化全局融合分类器
torch.manual_seed(SEED)
global_models = MetaClassifier()

# 客户端使用 NumPy 数组列表
initial_parameters = get_parameters(global_models)

# 服务器策略使用 Flower 参数格式
flower_parameters = fl.common.ndarrays_to_parameters(initial_parameters)


def aggregate_accuracy(client_results):
    """
    按客户端验证样本数计算整体验证准确率。

    client_results 的每一项是：
    (客户端验证样本数, 客户端返回的指标字典)
    """
    total_samples = sum(
        num_samples
        for num_samples, metrics in client_results
    )

    if total_samples == 0:
        return {}

    weighted_correct = sum(
        num_samples * metrics["accuracy"]
        for num_samples, metrics in client_results
    )

    return {
        "accuracy": weighted_correct / total_samples
    }


strategy = fl.server.strategy.FedAvgM(
    initial_parameters=flower_parameters,

    # 每轮训练和验证都选择全部可用客户端
    fraction_fit=1.0,
    fraction_evaluate=1.0,

    # 等待三个客户端连接
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,

    # 一个客户端失败时，不使用不完整的结果聚合
    accept_failures=False,

    # 汇总客户端返回的验证准确率
    evaluate_metrics_aggregation_fn=aggregate_accuracy,

    server_learning_rate=1.0,
    server_momentum=0.9,
)
def split_for_clients(dataset, num_clients, seed):
    # 计算每个客户端的样本数量
    sizes=[len(dataset)//num_clients]*num_clients
    #如果除不尽，就把剩下的都给最后一个客户端
    sizes[-1]+=len(dataset)%num_clients
    #随机化种子
    Seed=torch.Generator().manual_seed(seed)
    return random_split(dataset,sizes,Seed)
#基础模型训练集
base_train_parts = split_for_clients(
    base_train_loader.dataset, NUM_CLIENTS, SEED
)
#基础模型验证集
base_val_parts = split_for_clients(
    base_val_loader.dataset, NUM_CLIENTS, SEED + 1
)
#集成学习分类器训练集
meta_train_parts = split_for_clients(
    meta_train_loader.dataset, NUM_CLIENTS, SEED + 2
)
#集成学习验证集
meta_val_parts = split_for_clients(
    meta_val_loader.dataset, NUM_CLIENTS, SEED + 3
)
#-------------------------------------客户端训练----------------------------------

# 1. 一次实验只生成一个时间目录
# 三个客户端的结果都保存在这个目录下面
run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

run_dir = (
    Path(__file__).resolve().parent
    / "result"
    / "learn_local_models"
    / run_name
)

# 2. 保存每个客户端训练好的模型
# 后面生成融合特征时，需要使用对应客户端的基模型
client_models = []


# 3. 外层循环：依次处理每个客户端
for client_id in range(NUM_CLIENTS):

    print(f"\n========== 开始处理客户端 {client_id} ==========")
    client_train_data = base_train_parts[client_id]
    client_val_data = base_val_parts[client_id]

    print(f"基模型训练样本数：{len(client_train_data)}")
    print(f"基模型验证样本数：{len(client_val_data)}")
    # 3.2 创建当前客户端的 DataLoader
    client_train_loader = make_loader(
        client_train_data,
        shuffle=True,
        seed=SEED + client_id,
    )

    client_val_loader = make_loader(
        client_val_data,
        shuffle=False,
    )
    # 3.3 根据一条样本读取输入特征数
    sample_features, sample_label = client_train_data[0]
    num_features = sample_features.numel()

    print(f"输入特征数：{num_features}")
    print(f"训练设备：{device}")
    # 3.4 创建这个客户端自己的集成模型
    torch.manual_seed(SEED + client_id)

    client_model = EnsembleModel(
        num_features=num_features,
    ).to(device)

    # 当前客户端的模型保存目录
    client_dir = run_dir / f"client_{client_id}"

    # 3.5 准备三个基模型
    # 每一项依次是：
    # 模型对象、训练时的模型类型、保存目录名
    models = [
        (client_model.resae, "resae", "resae"),
        (
            client_model.attention_capsnet,
            "capsnet",
            "attentioncapsnet",
        ),
        (client_model.BiLstm, "bilstm", "bilstm"),
    ]
    # 3.6 内层循环：依次训练当前客户端的三个基模型
    for base_model, model_type, folder_name in models:

        print(f"\n客户端 {client_id}：开始训练 {model_type}")

        train_base_model(
            model=base_model,
            model_type=model_type,
            train_loader=client_train_loader,
            val_loader=client_val_loader,
            device=device,
            epochs=BASE_EPOCHS,
            patience=BASE_PATIENCE,
            lr=1e-3,
            save_path=client_dir / folder_name / "best_model.pt",
        )


    # 3.7 三个基模型都训练完成后，再冻结它们
    client_model.freeze_base_models()
    # 将模型移回 CPU 并保存到列表。
    # 后面需要用 GPU 生成融合特征时，再移到 device。
    client_models.append(client_model.cpu())

    print(f"\n客户端 {client_id} 的三个基模型训练完成并已冻结")
    print(f"模型保存目录：{client_dir}")


print("\n========== 所有客户端的基模型训练完成 ==========")
print(f"本次实验目录：{run_dir}")
print(f"已保存的客户端模型数量：{len(client_models)}")
# ============================================================
# 为每个客户端生成融合数据，并创建 Flower 客户端
# ============================================================

clients = []

for client_id in range(NUM_CLIENTS):

    print(f"\n开始准备客户端 {client_id} 的融合数据")

    # 1. 取出该客户端自己的三个基模型
    # 前面保存时移到了 CPU；提取特征前移到训练设备。
    local_model = client_models[client_id].to(device)

    # 2. 取出该客户端自己的元训练数据和元验证数据
    local_meta_train_loader = make_loader(
        meta_train_parts[client_id],
        shuffle=False,
    )

    local_meta_val_loader = make_loader(
        meta_val_parts[client_id],
        shuffle=False,
    )

    # 3. 用三个基模型生成 45 维融合特征
    client_meta_train = make_meta_data(
        local_model,
        local_meta_train_loader,
        device,
    )

    client_meta_val = make_meta_data(
        local_model,
        local_meta_val_loader,
        device,
    )

    print(f"融合训练样本数：{len(client_meta_train)}")
    print(f"融合验证样本数：{len(client_meta_val)}")

    sample_features, sample_label = client_meta_train[0]
    print(f"一条融合特征的形状：{sample_features.shape}")

    # 4. 特征已生成，基模型移回 CPU
    local_model.cpu()

    # 5. 创建 Flower 客户端
    client = FlowerClient(
        client_id=client_id,
        train_data=client_meta_train,
        val_data=client_meta_val,
        initial_parameters=initial_parameters,
    )

    clients.append(client)

print(f"\nFlower 客户端创建完成，数量：{len(clients)}")
# ============================================================
# 启动 Flower 客户端和服务器
# ============================================================

import threading

SERVER_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 50


# 1. 创建 Flower 服务器
# 服务器内部使用前面定义好的 FedAvgM 策略。
server = fl.server.Server(
    client_manager=fl.server.SimpleClientManager(),
    strategy=strategy,
)


# 2. 为每个客户端创建一个线程
client_threads = []

for client in clients:
    thread = threading.Thread(
        target=fl.client.start_client,
        kwargs={
            "server_address": SERVER_ADDRESS,
            "client": client.to_client(),
            "insecure": True,
        },
        daemon=True,
    )

    client_threads.append(thread)


# 3. 启动客户端线程
for thread in client_threads:
    thread.start()


# 4. 启动服务器，进行指定轮数的联邦训练
history = fl.server.start_server(
    server_address=SERVER_ADDRESS,
    server=server,
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS,
    ),
)


# 5. 等待所有客户端线程结束
for thread in client_threads:
    thread.join()

print("\n联邦训练结束")
print(history)


# ============================================================
# 保存最后一轮可用的全局融合分类器
# ============================================================

# Flower 参数格式 → NumPy 数组列表
final_parameters = fl.common.parameters_to_ndarrays(
    server.parameters
)

# 创建一个融合分类器，并加载最终全局参数
final_global_model = MetaClassifier().to("cpu")
set_parameters(final_global_model, final_parameters)

global_model_path = run_dir / "global_meta_final.pt"

torch.save(
    {
        "model_state_dict": final_global_model.state_dict(),
        "num_rounds": NUM_ROUNDS,
    },
    global_model_path,
)

print(f"最终全局融合分类器已保存：{global_model_path}")

