"""
INF-X-Retriever 编码器
基于 inf-retriever-v1-pro 模型，使用 last_token_pool 和 instruction prefix
参考: https://github.com/yaoyichen/INF-X-Retriever
"""

import os
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last token pooling - 与 INF-X-Retriever 一致"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class INFXRetrieverEncoder(BaseEncoder):
    """INF-X-Retriever 编码器 - 使用 inf-retriever-v1-pro"""

    def __init__(
        self,
        model_name: str = "inf-retriever-v1-pro",
        device: str = "cuda",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length or 8192

        # 解析本地路径
        local_path = self._resolve_model_path(model_name)
        logger.info(f"Loading INF-X-Retriever encoder...")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Resolved path: {local_path}")

        # 加载 tokenizer 和模型（不用 trust_remote_code，直接用标准 Qwen2）
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = AutoModel.from_pretrained(local_path)

        # 设备和精度
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(device)
        self.model = self.model.half()
        self.model.eval()

        logger.info(f"INF-X-Retriever encoder loaded successfully (max_seq_length={self.max_seq_length})")

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """解析模型名称到本地路径"""
        path_map = {
            "inf-retriever-v1-pro": "/home/luwa/Documents/models/inf-retriever-v1-pro",
            "infly/inf-retriever-v1-pro": "/home/luwa/Documents/models/inf-retriever-v1-pro",
        }
        if model_name in path_map:
            return path_map[model_name]
        if os.path.isdir(model_name):
            return model_name
        local_base = "/home/luwa/Documents/models"
        local_path = os.path.join(local_base, model_name)
        if os.path.isdir(local_path):
            return local_path
        return model_name

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询 - 添加 instruction prefix"""
        task = "Given a web search query, retrieve relevant passages that answer the query"
        formatted = [f"Instruct: {task}\nQuery: {text.strip()}" for text in texts]
        return self._encode(formatted, batch_size, **kwargs)

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档 - 不添加 instruction"""
        return self._encode(texts, batch_size, **kwargs)

    @torch.no_grad()
    def _encode(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文本"""
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            batch_dict = self.tokenizer(
                batch_texts,
                max_length=self.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # 移到设备
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

            with torch.amp.autocast('cuda'):
                outputs = self.model(**batch_dict)

            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

            if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(texts):
                logger.info(f"  编码进度: {min(i + batch_size, len(texts))}/{len(texts)}")

        return torch.cat(all_embeddings, dim=0)
