# src/data/collate.py
"""
图数据collate_fn的顶层可序列化实现
用于解决spawn模式下的pickle序列化问题
"""

import torch


class GraphCollator:
    """
    顶层可序列化的图数据collator类
    将graph_data作为CPU张量存储，避免worker进程中的CUDA初始化
    """

    def __init__(self, graph_data):
        """
        初始化collator

        Args:
            graph_data: 图数据字典，必须是CPU张量
        """
        self.graph_data = graph_data  # 仅CPU张量，不在这里调用.to(device)

    def __call__(self, batch):
        """
        批量处理函数

        Args:
            batch: 来自Dataset的样本列表

        Returns:
            dict: 包含批量数据的字典
        """
        return {
            "x": torch.stack([item["x"] for item in batch]),
            "y": torch.stack([item["y"] for item in batch]),
            "x_time_features": torch.stack([item["x_time_features"] for item in batch]),
            "graph_data": self.graph_data,  # CPU张量，主进程会负责.to(device)
        }
