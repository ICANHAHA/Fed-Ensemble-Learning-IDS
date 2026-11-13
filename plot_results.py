from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result" / "ML"
MODEL_NAMES = ["RESAE", "AttentionCapsNet", "BiLSTM", "Proposed"]
# MODEL_NAMES = ["CNN", "DNN", "RNN", "CNN_BiLSTM", "Proposed"]
DISPLAY_NAMES = {
    "RESAE": "RESAE",
    "AttentionCapsNet": "CapsNet",
    "BiLSTM": "BiLSTM",
    "CNN": "CNN",
    "DNN": "DNN",
    "RNN": "RNN",
    "CNN_BiLSTM": "CNN_BiLSTM",
    "Proposed": "Ensemble"
}
CLASS_NAMES = ["Normal", "DDoS_UDP", "Fingerprinting", "MITM", "XSS"]
METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("f1_score", "F1 Score")
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
DPI = 200


#-------------------------------读取实验结果---------------------------------------

def read_history(history_path):
    # 1. 读取真实训练记录，绘图时选取 epoch 或 round 列
    with history_path.open("r", newline="", encoding="utf-8-sig") as file:
        history = list(csv.DictReader(file))
    return history


def read_per_class_metrics(result_dir):
    # 2. 只读取 MODEL_NAMES 指定模型的 CSV，不补充其他模型的数据
    model_metrics = {}
    for model_name in MODEL_NAMES:
        metrics_path = result_dir / model_name / "per_class_metrics.csv"
        with metrics_path.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        model_metrics[model_name] = {
            row["class_name"]: {
                metric: float(row[metric]) for metric, _ in METRICS
            }
            for row in rows
        }
    return model_metrics


#-------------------------------训练曲线-------------------------------------------

def plot_learning_curves(history, save_dir, x_key, x_label, model_title):
    # 3. 保存训练集和验证集的准确率曲线
    save_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(row[x_key]) for row in history]
    train_accuracy = [float(row["train_accuracy"]) for row in history]
    val_accuracy = [float(row["val_accuracy"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_accuracy, label="Training", color=COLORS[0])
    ax.plot(steps, val_accuracy, label="Validation", color=COLORS[1])
    ax.set_xlabel(x_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_title} Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # 4. 损失可能大于 1，纵轴按真实数值显示
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, label="Training", color=COLORS[0])
    ax.plot(steps, val_loss, label="Validation", color=COLORS[1])
    ax.set_xlabel(x_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_title} Loss")
    ax.set_ylim(0, max(train_loss + val_loss) * 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_global_curves(result_dir):
    # 5. 全局融合头按联邦通信轮数绘图
    history = read_history(result_dir / "Proposed" / "history.csv")
    plot_learning_curves(
        history,
        result_dir / "figures",
        "round",
        "Federated Round",
        DISPLAY_NAMES["Proposed"]
    )


def plot_local_curves(result_dir):
    # 6. 各客户端基础模型按本地 epoch 绘图
    for history_path in sorted((result_dir / "clients").glob("client_*/*/history.csv")):
        client_name = history_path.parent.parent.name
        model_name = history_path.parent.name
        history = read_history(history_path)
        plot_learning_curves(
            history,
            result_dir / "figures" / client_name / model_name,
            "epoch",
            "Epoch",
            f"{client_name} / {model_name}"
        )


#-------------------------------分类指标对比---------------------------------------

def draw_metric_bars(ax, model_metrics, metric, metric_title):
    # 7. 每个类别一组，每组包含 MODEL_NAMES 中各模型的实际测试指标
    # 各模型应使用同一测试集；每类 accuracy = (TP + TN) / N。
    # recall = TP / (TP + FN)，表示该类样本的检出率，与每类 accuracy 含义不同。
    positions = np.arange(len(CLASS_NAMES))
    bar_width = 0.8 / len(MODEL_NAMES)
    for index, model_name in enumerate(MODEL_NAMES):
        values = [
            model_metrics[model_name][class_name][metric]
            for class_name in CLASS_NAMES
        ]
        offset = (index - (len(MODEL_NAMES) - 1) / 2) * bar_width
        ax.bar(
            positions + offset,
            values,
            width=bar_width,
            label=DISPLAY_NAMES[model_name],
            color=COLORS[index]
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel(metric_title)
    ax.set_title(f"Per-Class {metric_title}")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)


def plot_model_comparisons(result_dir):
    # 8. 分别保存五张指标柱状图，图例放在绘图区外
    save_dir = result_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = read_per_class_metrics(result_dir)
    for metric, metric_title in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        draw_metric_bars(ax, model_metrics, metric, metric_title)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(MODEL_NAMES)
        )
        fig.tight_layout()
        fig.savefig(save_dir / f"{metric}_comparison.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # 9. 五指标合图：前两行各两张，第三行居中一张
    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(3, 4)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 1:3])
    ]
    for ax, (metric, metric_title) in zip(axes, METRICS):
        draw_metric_bars(ax, model_metrics, metric, metric_title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Performance Comparison", y=0.985, fontsize=18)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(MODEL_NAMES)
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.88,
        hspace=0.7,
        wspace=1.0
    )
    fig.savefig(save_dir / "metrics_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    # 10. 依次绘制全局曲线、本地曲线和模型对比图
    plot_global_curves(RESULT_DIR)
    plot_local_curves(RESULT_DIR)
    plot_model_comparisons(RESULT_DIR)


if __name__ == "__main__":
    main()
