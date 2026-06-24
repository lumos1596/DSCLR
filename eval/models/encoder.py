"""
模型加载模块
支持多种类型的稠密检索模型
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BaseEncoder(ABC):
    """编码器基类"""
    
    @abstractmethod
    def encode_queries(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码查询文本"""
        pass
    
    @abstractmethod
    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码文档文本"""
        pass


class SentenceTransformerEncoder(BaseEncoder):
    """基于 SentenceTransformer 的编码器"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"📥 加载模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        self.model = self.model.half()
        
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
            
        logger.info(f"✅ 模型加载完成: {model_name} (float16)")
    
    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询"""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            **kwargs
        )
        return embeddings
    
    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档"""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            **kwargs
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.model.get_sentence_embedding_dimension()


class ModelFactory:
    """模型工厂"""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """注册模型类"""
        cls._models[name] = model_class
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseEncoder:
        """创建模型实例
        
        自动识别模型类型：
        - repllama: 使用 RepLLaMAEncoder
        - e5-mistral: 使用 E5MistralEncoder
        - 其他: 使用 SentenceTransformerEncoder
        """
        # 检查是否已注册
        if model_name in cls._models:
            return cls._models[model_name](model_name=model_name, **kwargs)
        
        model_name_lower = model_name.lower()

        # 自动识别 RepLLaMA / Promptriever PEFT 模型
        if "repllama" in model_name_lower or "promptriever" in model_name_lower:
            from .repllama_encoder import RepLLaMAEncoder
            logger.info(f"🔍 自动识别为 RepLLaMA/Promptriever PEFT 模型: {model_name}")
            return RepLLaMAEncoder(model_name=model_name, **kwargs)
        
        # 自动识别 E5-Mistral 模型
        if "e5-mistral" in model_name_lower:
            # 延迟导入避免循环依赖
            from .e5_mistral_encoder import E5MistralEncoder
            logger.info(f"🔍 自动识别为 E5-Mistral 模型: {model_name}")
            return E5MistralEncoder(model_name=model_name, **kwargs)
        
        # 自动识别 LLaMA 3.1 8B Instruct 模型
        if "meta-llama" in model_name_lower or "llama-3.1" in model_name_lower or "llama3.1" in model_name_lower:
            from .llama3_encoder import LLaMA31Encoder
            logger.info(f"🔍 自动识别为 LLaMA 3.1 模型: {model_name}")
            return LLaMA31Encoder(model_name=model_name, **kwargs)
        
        # 自动识别 INF-X-Retriever 模型
        if "inf-retriever" in model_name_lower or "infx" in model_name_lower:
            from .infx_encoder import INFXRetrieverEncoder
            logger.info(f"🔍 自动识别为 INF-X-Retriever 模型: {model_name}")
            return INFXRetrieverEncoder(model_name=model_name, **kwargs)
        
        # 默认使用 SentenceTransformer
        return SentenceTransformerEncoder(
            model_name=model_name,
            **kwargs
        )
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用模型列表"""
        return list(cls._models.keys())


ModelFactory.register("sentence_transformer", SentenceTransformerEncoder)


class DenseRetriever:
    """稠密检索器"""
    
    def __init__(self, encoder: BaseEncoder):
        self.encoder = encoder
        self.doc_embeddings: Dict[str, torch.Tensor] = {}
        self.doc_ids: List[str] = []
    
    def index_documents(self, doc_ids: List[str], doc_texts: List[str], batch_size: int = 64, 
                        checkpoint_dir: Optional[str] = None, checkpoint_interval: int = 1000) -> None:
        """构建文档索引（支持分块保存）
        
        Args:
            doc_ids: 文档ID列表
            doc_texts: 文档文本列表
            batch_size: 批处理大小
            checkpoint_dir: 检查点保存目录，如果提供则定期保存
            checkpoint_interval: 每处理多少文档保存一次检查点
        """
        logger.info(f"📚 索引 {len(doc_ids)} 个文档...")
        
        # 如果提供了检查点目录，尝试加载已有的检查点
        start_idx = 0
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            start_idx = self._load_checkpoint(checkpoint_dir, doc_ids)
            if start_idx > 0:
                logger.info(f"🔄 从检查点恢复，已处理 {start_idx}/{len(doc_ids)} 个文档")
        
        # 增量编码
        if start_idx < len(doc_ids):
            for i in range(start_idx, len(doc_ids), batch_size):
                end_idx = min(i + batch_size, len(doc_ids))
                batch_ids = doc_ids[i:end_idx]
                batch_texts = doc_texts[i:end_idx]
                
                # 编码当前批次
                batch_embeddings = self.encoder.encode_documents(batch_texts, batch_size=batch_size)
                
                # 存储到内存
                for idx, doc_id in enumerate(batch_ids):
                    self.doc_embeddings[doc_id] = batch_embeddings[idx]
                self.doc_ids.extend(batch_ids)
                
                # 定期保存检查点
                if checkpoint_dir is not None and end_idx % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_dir, doc_ids, end_idx)
                    logger.info(f"💾 检查点已保存: {end_idx}/{len(doc_ids)} 个文档")
                
                # 进度日志
                if (end_idx - start_idx) % (batch_size * 10) == 0 or end_idx == len(doc_ids):
                    logger.info(f"  已编码 {end_idx}/{len(doc_ids)}")
        
        self.doc_ids = doc_ids  # 确保顺序一致
        logger.info(f"✅ 文档索引构建完成")
        
        # 最终保存检查点
        if checkpoint_dir is not None:
            self._save_checkpoint(checkpoint_dir, doc_ids, len(doc_ids))
            logger.info(f"💾 最终检查点已保存")
    
    def _save_checkpoint(self, checkpoint_dir: str, doc_ids: List[str], processed_count: int) -> None:
        """保存检查点"""
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
        # 只保存已处理的文档
        processed_ids = doc_ids[:processed_count]
        embeddings_list = [self.doc_embeddings[did] for did in processed_ids]
        
        checkpoint = {
            'processed_count': processed_count,
            'doc_ids': processed_ids,
            'embeddings': torch.stack(embeddings_list) if embeddings_list else torch.empty(0),
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_dir: str, doc_ids: List[str]) -> int:
        """加载检查点，返回已处理的文档数量"""
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            return 0
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            processed_ids = checkpoint['doc_ids']
            embeddings = checkpoint['embeddings']
            
            # 验证检查点是否匹配当前文档集合
            if set(processed_ids).issubset(set(doc_ids)):
                # 恢复已处理的文档
                for idx, doc_id in enumerate(processed_ids):
                    self.doc_embeddings[doc_id] = embeddings[idx]
                self.doc_ids = list(processed_ids)
                
                logger.info(f"✅ 检查点加载成功: {len(processed_ids)} 个文档")
                return len(processed_ids)
            else:
                logger.warning("⚠️ 检查点文档ID不匹配，重新编码")
                return 0
        except Exception as e:
            logger.warning(f"⚠️ 检查点加载失败: {e}，重新编码")
            return 0
    
    def set_embeddings(self, embeddings: torch.Tensor, doc_ids: List[str]) -> None:
        """直接设置已有文档向量"""
        self.doc_embeddings = {doc_id: embeddings[idx] for idx, doc_id in enumerate(doc_ids)}
        self.doc_ids = doc_ids
        logger.info(f"✅ 文档向量已设置 (共 {len(doc_ids)} 个)")
    
    def search(
        self,
        query_embeddings: torch.Tensor,
        doc_ids: List[str],
        top_k: int = 100
    ) -> List[List[Dict[str, Any]]]:
        """执行检索"""
        doc_emb = torch.stack([self.doc_embeddings[did] for did in doc_ids]).to(query_embeddings.device)
        
        scores = torch.matmul(query_embeddings, doc_emb.T)
        
        results = []
        for query_scores in scores:
            _, top_indices = torch.topk(query_scores, min(top_k, len(doc_ids)))
            
            query_results = []
            for idx in top_indices:
                doc_id = doc_ids[idx.item()]
                score = query_scores[idx.item()].item()
                query_results.append({
                    "doc_id": doc_id,
                    "score": score
                })
            results.append(query_results)
        
        return results
    
    def compute_scores(
        self,
        query_embedding: torch.Tensor,
        doc_ids: List[str]
    ) -> Dict[str, float]:
        """计算单个查询与文档的得分"""
        doc_emb = torch.stack([self.doc_embeddings[did] for did in doc_ids]).to(query_embedding.device)
        
        scores = torch.matmul(query_embedding, doc_emb.T).squeeze(0)
        
        return {doc_ids[idx]: scores[idx].item() for idx in range(len(doc_ids))}
