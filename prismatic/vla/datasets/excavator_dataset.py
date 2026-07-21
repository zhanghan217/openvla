"""
excavator_dataset.py

伏羲挖掘机数据集加载器 (RTX 5090 Blackwell 适配)
从预处理后的 TFRecord 读取数据，适配 OpenVLA 训练管道。

数据路径: FuXiData/processed/excavator_motion.tfrecord
数据格式: 每帧 {image bytes (RGB 224x224), action float[4]}
"""

import json
from pathlib import Path
from typing import Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

import tensorflow as tf

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

IGNORE_INDEX = -100


class ExcavatorDataset(Dataset):
    """从 TFRecord 读取挖掘机数据，按 DummyDataset 接口返回训练样本"""

    def __init__(
        self,
        tfrecord_path: str,
        config_path: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # 读取配置
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.instruction = self.config["instruction"]

        # 读取 TFRecord
        tfrecord_path = Path(tfrecord_path)
        self.tfrecord_path = tfrecord_path
        self.tfrecord = str(tfrecord_path)
        self.image_size = tuple(self.config["img_size"])

        # 一次性全量加载所有帧到内存 (TFRecord 不支持随机访问，
        # 逐帧 skip().take(1) 会随 idx 线性变慢，训练几千步后单步耗时会从几秒涨到几十秒)
        feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "action": tf.io.FixedLenFeature([4], tf.float32),
        }
        self._records: list[dict] = []
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord)
        for raw_record in raw_dataset:
            example = tf.io.parse_single_example(raw_record, feature_desc)
            self._records.append(
                {
                    "image_bytes": example["image"].numpy(),
                    "action": example["action"].numpy().astype(np.float32),
                }
            )
        self._len = len(self._records)

        # 加载归一化统计
        stats_path = tfrecord_path.parent / "dataset_statistics.json"
        with open(stats_path, "r") as f:
            self.dataset_statistics = json.load(f)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # 循环索引
        idx = idx % self._len

        record = self._records[idx]
        # 从 raw bytes 重建图片 (按需解码，避免常驻内存里存 PIL 对象)
        image = Image.frombytes("RGB", self.image_size, record["image_bytes"])
        action = record["action"]

        # 构建 Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {self.instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
