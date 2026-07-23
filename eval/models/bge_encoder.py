"""BGE (BAAI General Embedding) 编码器

BGE-large-en-v1.5 使用 SentenceTransformer 加载，采用 CLS pooling + L2 归一化。
查询需要前置指令: "Represent this sentence for searching relevant passages: "
文档无需指令前缀。
"""

import logging
import os
from typing import List, Optional

import torch

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/home/luwa/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class BGEEncoder(BaseEncoder):
    """BGE 编码器 - 基于 SentenceTransformer，正确处理查询指令前缀

    BGE-large-en-v1.5:
    - 模型规模: 335M 参数
    - 嵌入维度: 1024
    - Pooling: CLS token
    - 归一化: L2
    - 查询指令: "Represent this sentence for searching relevant passages: "
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        logger.info("📥 加载 BGE 模型: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model = self.model.half()

        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        else:
            self.model.max_seq_length = 512

        logger.info(
            "✅ BGE 模型加载完成 (float16, dim=%d, max_seq=%d)",
            self.model.get_sentence_embedding_dimension(),
            self.model.max_seq_length,
        )

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询 - 前置 BGE 查询指令"""
        batch_size = batch_size or self.batch_size
        prefixed = [self.QUERY_INSTRUCTION + t for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档 - 无指令前缀"""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
