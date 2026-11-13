
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET_NAME = "ML"
DATA_PATH = ROOT / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
OUTPUT_DIR = ROOT / "preprocessed_ml_70_10_20"
USE_LOG1P = True

RANDOM_STATE = 42

DROP_COLUMNS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]

DUMMY_COLUMNS = [
    "http.request.method", "http.referer", "http.request.version",
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
]


def main():
    # 1. 读取数据、删除无关特征和无效样本
    df = pd.read_csv(DATA_PATH, low_memory=False)
    original_rows = len(df)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    df = df.replace([np.inf, -np.inf, "", "nan", "NaN", "null", "None"], np.nan)
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # 2. 多分类标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Attack_type"])
    X = df.drop(columns=["Attack_type", "Attack_label"], errors="ignore")

    # 3. 第一次划分：80%训练+验证，20%测试
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 4. 第二次划分：70%训练，10%验证
    # 10% / 80% = 0.125
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # 5. 对非二值数值特征做有符号 log1p
    log_cols = [
        col for col in X_train.select_dtypes(include="number").columns
        if col not in DUMMY_COLUMNS and X_train[col].nunique() > 2
    ]

    if USE_LOG1P:
        for part in (X_train, X_val, X_test):
            values = part[log_cols].to_numpy(dtype=np.float64)
            part[log_cols] = (
                np.sign(values) * np.log1p(np.abs(values))
            ).astype(np.float32)

    # 6. 独热编码：验证集和测试集与训练集列对齐
    cat_cols = [
        col for col in X_train.columns
        if col in DUMMY_COLUMNS or not pd.api.types.is_numeric_dtype(X_train[col])
    ]

    X_train = pd.get_dummies(X_train, columns=cat_cols, dtype=np.float32)
    X_val = pd.get_dummies(X_val, columns=cat_cols, dtype=np.float32)
    X_test = pd.get_dummies(X_test, columns=cat_cols, dtype=np.float32)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    feature_names = X_train.columns.tolist()
    input_size = len(feature_names)
    num_classes = len(label_encoder.classes_)

    # 7. 标准化：只使用训练集拟合
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 8. 转换为 PyTorch 张量
    data = {
        "dataset_name": DATASET_NAME,
        "data_path": str(DATA_PATH),
        "input_size": input_size,
        "num_classes": num_classes,
        "X_train": torch.from_numpy(X_train),
        "Y_train": torch.from_numpy(y_train.astype(np.int64)),

        "X_val": torch.from_numpy(X_val),
        "Y_val": torch.from_numpy(y_val.astype(np.int64)),

        "X_test": torch.from_numpy(X_test),
        "Y_test": torch.from_numpy(y_test.astype(np.int64)),

        "class_names": label_encoder.classes_.tolist(),
        "feature_names": feature_names,
    }

    # 9. 保存预处理结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, OUTPUT_DIR / "processed_data.pt")

    report = {
        "dataset_name": DATASET_NAME,
        "data_path": str(DATA_PATH),
        "input_size": input_size,
        "num_classes": num_classes,
        "class_names": label_encoder.classes_.tolist(),
        "original_rows": original_rows,
        "clean_rows": len(df),

        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test),

        "train_ratio": len(y_train) / len(df),
        "val_ratio": len(y_val) / len(df),
        "test_ratio": len(y_test) / len(df),

        "feature_count": len(feature_names),
        "class_mapping": {
            name: i for i, name in enumerate(label_encoder.classes_)
        },

        "log1p_enabled": USE_LOG1P,
        "log1p_columns": log_cols if USE_LOG1P else [],
        "categorical_columns": cat_cols,
        "feature_names": feature_names,

        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),

        "random_state": RANDOM_STATE,
        "split": "70:10:20",
        "stratified": True,
    }

    (OUTPUT_DIR / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    pd.DataFrame({"feature_name": feature_names}).to_csv(
        OUTPUT_DIR / "feature_names.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 10. 输出结果
    print(f"数据集：{DATASET_NAME} ({DATA_PATH.name})")
    print(f"原始样本：{original_rows:,}")
    print(f"清洗后样本：{len(df):,}")

    print(f"训练集：{len(y_train):,} ({len(y_train)/len(df):.2%})")
    print(f"验证集：{len(y_val):,} ({len(y_val)/len(df):.2%})")
    print(f"测试集：{len(y_test):,} ({len(y_test)/len(df):.2%})")

    print(f"训练集形状：{data['X_train'].shape}")
    print(f"验证集形状：{data['X_val'].shape}")
    print(f"测试集形状：{data['X_test'].shape}")

    print(f"特征数：{len(feature_names)}")
    print(f"类别数：{len(label_encoder.classes_)}")
    print(f"对数变换：{'开启' if USE_LOG1P else '关闭'}")

    print(f"数据已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
