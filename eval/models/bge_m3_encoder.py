"""BGE-M3 编码器

BAAI/bge-m3 多语言嵌入模型，与 BGE-large 不同的是：
- 查询无需前置指令前缀
- 支持多语言和多粒度嵌入
- 嵌入维度: 1024
"""

import logging
import os
from typing import List, Optional

import torch

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/home/luwa/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class BGEM3Encoder(BaseEncoder):
    """BGE-M3 编码器 - 无查询指令前缀"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
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

        logger.info("Loading BGE-M3 model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model = self.model.half()

        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        else:
            self.model.max_seq_length = 8192

        logger.info(
            "BGE-M3 model loaded (float16, dim=%d, max_seq=%d)",
            self.model.get_sentence_embedding_dimension(),
            self.model.max_seq_length,
        )

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询 - BGE-M3 无需指令前缀"""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
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
