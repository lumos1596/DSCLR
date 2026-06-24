"""
LLaMA 3.1 8B Instruct 编码器（原始模型，未检索微调）
使用 mean pooling + L2 归一化，不加 query/passage 前缀
注意：此编码器用于原始 LLaMA3 模型，检索微调版请使用 RepLLaMAEncoder
"""

import logging
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)


class LLaMA31Encoder(BaseEncoder):
    """LLaMA 3.1 8B Instruct 编码器（原始模型）- 使用 mean pooling"""

    def __init__(
        self,
        model_name: str = "Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda",
        batch_size: int = 4,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length or 2048

        # 解析本地路径
        local_path = self._resolve_model_path(model_name)
        logger.info(f"Loading LLaMA 3.1 8B Instruct encoder (raw, mean pooling)...")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Resolved path: {local_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # float16 加载，使用 device_map="auto" 分布到所有可用 GPU
        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        # 记录模型第一层所在设备
        self.input_device = self.model.device
        logger.info(f"Model device map: {self.model.hf_device_map}")
        logger.info(f"Input device: {self.input_device}")

        logger.info("LLaMA 3.1 8B Instruct encoder loaded successfully (float16, mean pooling, multi-GPU)")

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """解析模型名称到本地路径"""
        import os
        path_map = {
            "Meta-Llama-3.1-8B-Instruct": "/home/luwa/Documents/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "/home/luwa/Documents/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
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

    def _encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = True) -> torch.Tensor:
        """编码文本列表，使用 mean pooling + L2 归一化"""
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        num_batches = (len(texts) + batch_size - 1) // batch_size
        pbar = tqdm(total=num_batches, desc="编码中", unit="batch", disable=not show_progress)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_seq_length,
                return_token_type_ids=False,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.input_device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                last_hidden_state = outputs.last_hidden_state
                # Mean pooling: 对 attention_mask 覆盖的 token 取平均
                attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
                reps = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                if self.normalize_embeddings:
                    reps = F.normalize(reps, p=2, dim=-1)
                all_embeddings.append(reps.cpu())

            pbar.update(1)

        pbar.close()
        return torch.cat(all_embeddings, dim=0)

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询 - 直接编码原始文本（不加前缀）"""
        return self._encode(texts, batch_size, **kwargs)

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档 - 直接编码原始文本（不加前缀）"""
        return self._encode(texts, batch_size, **kwargs)

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return 4096
