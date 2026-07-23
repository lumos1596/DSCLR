"""
E5-Mistral-7B-Instruct 编码器模块（官方规范实现）

严格遵循官方实现要求：
1. 使用 BFloat16 精度（RTX 3090 24GB 显存优化）
2. Last Token Pooling（严禁 Mean Pooling）
3. 强制追加 EOS Token
4. 特殊 Prompt 前缀格式

参考: https://huggingface.co/intfloat/e5-mistral-7b-instruct
"""

import logging
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    提取最后一个有效 Token 的隐藏状态作为 Embedding
    
    这是 Mistral Decoder 模型的标准池化方式，严禁使用 Mean Pooling！
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class E5MistralEncoder:
    """
    E5-Mistral-7B-Instruct 编码器（满血版）
    
    硬件要求: RTX 3090 (24GB VRAM)
    精度: BFloat16 (约 14GB 显存占用)
    """
    
    # FollowIR 通用检索指令
    DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-mistral-7b-instruct",
        device: str = "cuda",
        batch_size: int = 4,  # RTX 3090 建议初始值，可尝试提升到 8
        max_length: int = 4095,  # 留一个位置给 EOS
        max_seq_length: int = None,  # 兼容参数，会被 max_length 覆盖
        normalize_embeddings: bool = True,
        local_files_only: bool = False,  # 是否只使用本地文件
        device_map: str = "auto",  # 多卡并行：auto 自动分配层到多 GPU
        **kwargs  # 吸收其他未使用的参数
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # 检查本地缓存路径
        local_path = f"/home/luwa/.cache/huggingface/e5-mistral-7b-instruct"
        import os
        if os.path.exists(local_path):
            logger.info(f"📥 从本地加载 E5-Mistral 模型: {local_path}")
            model_path = local_path
            local_files_only = True
        else:
            logger.info(f"📥 从 HuggingFace 加载 E5-Mistral 模型: {model_name}")
            model_path = model_name

        logger.info(f"   设备: {device}, 精度: BFloat16, Batch: {batch_size}, device_map: {device_map}")

        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only
        )

        # 加载模型 - 使用 BFloat16 防止 FP16 溢出
        # device_map="auto" 时 accelerate 自动将模型层分配到多 GPU
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=local_files_only
        )
        self.model.eval()

        # 记录模型实际所在的设备（多卡时为第一个参数的设备）
        first_device = next(self.model.parameters()).device
        self._input_device = str(first_device)

        logger.info(f"✅ E5-Mistral 模型加载完成 (BFloat16), input_device={self._input_device}")
    
    def _format_query(self, query: str, task: Optional[str] = None) -> str:
        """
        格式化查询文本（添加指令模板）
        
        格式: Instruction: {task}\nQuery: {query}
        """
        task = task or self.DEFAULT_TASK
        return f"Instruction: {task}\nQuery: {query}"
    
    def encode_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs
    ) -> Tensor:
        """
        编码查询文本（Q+ 或 Q-）
        
        Args:
            texts: 查询文本列表
            batch_size: 批处理大小（默认 4，3090 可尝试 8）
            task: 任务描述（用于指令模板）
        
        Returns:
            归一化后的特征张量 [num_texts, 4096]
        """
        batch_size = batch_size or self.batch_size
        
        # 格式化查询（添加指令前缀）
        formatted_texts = [self._format_query(t, task) for t in texts]
        
        return self._encode_batch(formatted_texts, batch_size)
    
    def encode_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Tensor:
        """
        编码文档文本
        
        文档不需要指令前缀，原样输入
        """
        batch_size = batch_size or self.batch_size
        
        # 文档不添加前缀
        return self._encode_batch(texts, batch_size)
    
    def _encode_batch(self, texts: List[str], batch_size: int) -> Tensor:
        """
        核心编码逻辑
        
        严格遵循官方规范：
        1. 强制追加 EOS Token
        2. Last Token Pooling
        3. L2 归一化
        4. 立即移回 CPU 防止显存泄漏
        """
        all_embeddings = []
        
        with torch.no_grad():  # 绝对禁止计算梯度
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 第一步：Tokenize（不 padding，先截断）
                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    return_attention_mask=False,
                    padding=False,
                    truncation=True
                )
                
                # 第二步：强制追加 EOS Token（关键！）
                batch_dict['input_ids'] = [
                    input_ids + [self.tokenizer.eos_token_id]
                    for input_ids in batch_dict['input_ids']
                ]
                
                # 第三步：Padding 并转为 Tensor
                batch_dict = self.tokenizer.pad(
                    batch_dict,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                ).to(self._input_device)
                
                # 第四步：模型前向传播
                outputs = self.model(**batch_dict)
                
                # 第五步：Last Token Pooling（严禁 Mean Pooling！）
                # 多卡模式下 last_hidden_state 和 attention_mask 可能在不同设备
                hidden = outputs.last_hidden_state
                attn_mask = batch_dict['attention_mask'].to(hidden.device)
                embeddings = last_token_pool(hidden, attn_mask)
                
                # 第六步：L2 归一化（必须做！）
                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # 第七步：立即移回 CPU，防止显存泄漏
                all_embeddings.append(embeddings.cpu())
                
                # 进度日志
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"  已编码 {min(i + batch_size, len(texts))}/{len(texts)}")
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return 4096  # Mistral-7B 的 hidden size


# 别名兼容
E5Mistral7BEncoder = E5MistralEncoder