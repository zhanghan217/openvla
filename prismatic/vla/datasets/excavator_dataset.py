"""
excavator_dataset.py

伏羲挖掘机数据集加载器 (RTX 5090 Blackwell 适配)
从预处理后的 TFRecord 读取数据，适配 OpenVLA 训练管道。

数据路径: FuXiData/processed/excavator_motion.tfrecord
数据格式: 每帧 {image bytes (RGB 224x224), action float[4]}
"""

import json
import struct
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

        # 统计帧数 (快速扫描)
        dataset = tf.data.TFRecordDataset(self.tfrecord)
        count = 0
        for _ in dataset:
            count += 1
        self._len = count

        # 加载归一化统计
        stats_path = tfrecord_path.parent / "dataset_statistics.json"
        with open(stats_path, "r") as f:
            self.dataset_statistics = json.load(f)

        # LRU 缓存: 按需加载
        self._frames_cache = {}
        self._cache_size = 5000

    def __len__(self):
        return self._len

    def _load_frame(self, idx: int) -> dict:
        """从 TFRecord 加载单帧"""
        dataset = tf.data.TFRecordDataset(self.tfrecord)
        dataset = dataset.skip(idx).take(1)

        feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "action": tf.io.FixedLenFeature([4], tf.float32),
        }

        for raw_record in dataset:
            example = tf.io.parse_single_example(raw_record, feature_desc)
            img_bytes = example["image"].numpy()
            action = example["action"].numpy()

            # 从 raw bytes 重建图片
            img = Image.frombytes("RGB", self.image_size, img_bytes)
            return {"image": img, "action": action}

        raise IndexError(f"Frame {idx} not found in TFRecord")

    def __getitem__(self, idx):
        # 循环索引
        idx = idx % self._len

        frame = self._load_frame(idx)
        image = frame["image"]
        action = frame["action"].astype(np.float32)

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
