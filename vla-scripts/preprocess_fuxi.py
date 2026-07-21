"""
伏羲挖掘机数据集预处理脚本
功能: 读取 H5/HDF5 文件，转换为 OpenVLA 可用的 RLDS 格式
路径: vla-scripts/preprocess_fuxi.py

输入: FuXiData/excavator-motion/data/data/{75,306,490}/*.h5
输出: FuXiData/processed/  (RLDS TFRecord + 归一化统计)
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

INSTRUCTION = "挖掘装车"
DATASET_NAME = "excavator_motion"
IMG_SIZE = (224, 224)
FPS = 10


def build_rlds_dataset(h5_dir: str, output_dir: str):
    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.rglob("*.h5")) + sorted(h5_dir.rglob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"未找到 H5 文件: {h5_dir}")

    # --- 第一遍: 收集所有帧，计算统计量 ---
    all_actions = []
    total_frames = 0
    file_frames = []

    for fpath in h5_files:
        with h5py.File(fpath, "r") as f:
            qpos = np.array(f["observations/qpos"], dtype=np.float32)
            if "action" in f:
                action_next = np.array(f["action"], dtype=np.float32)
            else:
                action_next = np.concatenate([qpos[1:], qpos[-1:]], axis=0)

            n = min(len(qpos) - 1, len(action_next) - 1)
            deltas = action_next[:n] - qpos[:n]
            all_actions.append(deltas)
            total_frames += n
            file_frames.append((fpath, n))

    all_actions = np.concatenate(all_actions, axis=0)  # [total_frames, 4]
    q01 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)

    stats = {
        "action": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mean": np.mean(all_actions, axis=0).tolist(),
            "std": np.std(all_actions, axis=0).tolist(),
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
        },
        "num_frames": int(total_frames),
    }

    with open(output_dir / "dataset_statistics.json", "w") as f:
        json.dump({DATASET_NAME: stats}, f, indent=2, ensure_ascii=False)

    print(f"统计量已保存: {output_dir / 'dataset_statistics.json'}")
    print(f"  总帧数: {total_frames}")
    print(f"  动作 Q01: {q01}")
    print(f"  动作 Q99: {q99}")

    # --- 第二遍: 写入 RLDS TFRecord ---
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "action": tf.io.FixedLenFeature([4], tf.float32),
    }

    def _float_list_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    tfrecord_path = str(output_dir / "excavator_motion.tfrecord")
    frame_idx = 0

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for fpath, n_frames in tqdm(file_frames, desc="写入 TFRecord"):
            with h5py.File(fpath, "r") as f:
                images = np.array(f["observations/images/main"])
                qpos = np.array(f["observations/qpos"], dtype=np.float32)
                if "action" in f:
                    action_next = np.array(f["action"], dtype=np.float32)
                else:
                    action_next = np.concatenate([qpos[1:], qpos[-1:]], axis=0)

            for t in range(min(n_frames, min(len(images), len(qpos) - 1, len(action_next) - 1))):
                img_bgr = images[t]
                img_rgb = img_bgr[..., ::-1]  # BGR → RGB
                pil_img = Image.fromarray(img_rgb).resize(IMG_SIZE, Image.BILINEAR)
                img_bytes = np.array(pil_img).tobytes()

                delta = action_next[t] - qpos[t]

                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": _bytes_feature(img_bytes),
                    "action": _float_list_feature(delta.astype(np.float32)),
                }))
                writer.write(example.SerializeToString())
                frame_idx += 1

    print(f"TFRecord 已保存: {tfrecord_path}")
    print(f"  帧数: {frame_idx}")

    # --- 保存配置 ---
    config = {
        "dataset_name": DATASET_NAME,
        "instruction": INSTRUCTION,
        "action_dim": 4,
        "img_size": list(IMG_SIZE),
        "fps": FPS,
        "tfrecord": str(tfrecord_path),
        "num_frames": frame_idx,
        "action_Q01": q01.tolist(),
        "action_Q99": q99.tolist(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"配置已保存: {output_dir / 'config.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="FuXiData/excavator-motion/data/data",
                        help="H5 文件目录")
    parser.add_argument("--output_dir", type=str,
                        default="FuXiData/processed",
                        help="输出目录")
    args = parser.parse_args()

    build_rlds_dataset(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
