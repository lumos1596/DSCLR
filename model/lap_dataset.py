"""
LAP (Lightweight Asymmetric Projection) 数据集加载器

核心功能：
1. 加载 FollowIR 训练数据
2. 构建 (query_text, doc_pos_text, doc_neg_text) 三元组
3. 支持 Hard Negative Mining（使用 label=False 的踩雷文档）
4. 利用预计算的文档嵌入缓存，仅对 Q_minus 实时编码

数据来源：
- train_data_dsclr.jsonl: 包含 query, instruction, pos, neg
- distilled_queries_v4.jsonl: 包含 Q_plus, Q_minus
- embeddings/{encoder}/dsclr_train_embeddings_{encoder}.pt: 缓存的嵌入向量
"""

import json
import os
import random
import torch
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


class LAPDataset(Dataset):
    """
    LAP 训练数据集
    
    每个样本返回一个三元组：(q_minus_text, doc_pos_text, doc_neg_text)
    
    Args:
        data_dir: 训练数据目录
        train_file: 训练数据文件名
        query_file: 双流查询文件名
        num_neg_per_query: 每个 query 使用的负样本数量
        max_neg_samples: 最大负样本池大小（用于随机采样）
    """
    
    def __init__(
        self,
        data_dir: str = "dataset/FollowIR_train",
        train_file: str = "train_data_dsclr.jsonl",
        query_file: str = "distilled_queries_v4.jsonl",
        num_neg_per_query: int = 1,
        max_neg_samples: int = 15
    ):
        self.data_dir = data_dir
        self.num_neg_per_query = num_neg_per_query
        self.max_neg_samples = max_neg_samples
        
        # 加载训练数据
        train_path = os.path.join(data_dir, train_file)
        self.train_data = self._load_jsonl(train_path)
        print(f"加载训练数据: {len(self.train_data)} 条")
        
        # 加载双流查询数据
        query_path = os.path.join(data_dir, query_file)
        self.query_data = self._load_jsonl(query_path)
        print(f"加载双流查询: {len(self.query_data)} 条")
        
        # 构建 idx -> Q_minus 的映射
        self.idx_to_q_minus = {}
        for item in self.query_data:
            idx = item['idx']
            output = json.loads(item['output'])
            q_minus = output.get('Q_minus', '[NONE]')
            self.idx_to_q_minus[idx] = q_minus
        
        # 构建样本列表（展开多负样本）
        self.samples = self._build_samples()
        print(f"构建训练样本: {len(self.samples)} 条")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        """加载 JSONL 文件"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _build_samples(self) -> List[Tuple]:
        """
        构建训练样本列表
        
        每个样本是一个元组: (q_minus_text, doc_pos_text, doc_neg_text, idx)
        """
        samples = []
        
        for item in self.train_data:
            idx = item['idx']
            
            # 获取 Q_minus 文本
            q_minus_text = self.idx_to_q_minus.get(idx, '[NONE]')
            
            # 获取正样本文档列表
            pos_docs = item.get('pos', [])
            if not pos_docs:
                continue
            
            # 获取负样本文档列表（Hard Negatives）
            neg_docs = item.get('neg', [])
            if not neg_docs:
                continue
            
            # 为每个正样本构建训练样本
            for pos_doc in pos_docs:
                # 选择负样本
                # 策略1: 随机采样
                # 策略2: 使用所有负样本（展开）
                selected_negs = neg_docs[:self.num_neg_per_query]
                
                for neg_doc in selected_negs:
                    samples.append((q_minus_text, pos_doc, neg_doc, idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, str]:
        """
        获取单个样本
        
        Returns:
            Dict with keys:
                - q_minus_text: 负向查询文本
                - doc_pos_text: 正样本文档文本
                - doc_neg_text: 负样本文档文本
                - idx: 样本索引
        """
        q_minus_text, doc_pos_text, doc_neg_text, idx = self.samples[index]
        
        return {
            'q_minus_text': q_minus_text,
            'doc_pos_text': doc_pos_text,
            'doc_neg_text': doc_neg_text,
            'idx': idx
        }


class LAPDatasetWithFullQuery(Dataset):
    """
    LAP 训练数据集（带完整查询信息）
    
    额外返回 instruction + query，用于调试和分析
    
    Args:
        data_dir: 训练数据目录
        train_file: 训练数据文件名
        query_file: 双流查询文件名
        num_neg_per_query: 每个 query 使用的负样本数量
    """
    
    def __init__(
        self,
        data_dir: str = "dataset/FollowIR_train",
        train_file: str = "train_data_dsclr.jsonl",
        query_file: str = "distilled_queries_v4.jsonl",
        num_neg_per_query: int = 1
    ):
        self.data_dir = data_dir
        self.num_neg_per_query = num_neg_per_query
        
        # 加载数据
        train_path = os.path.join(data_dir, train_file)
        self.train_data = self._load_jsonl(train_path)
        
        query_path = os.path.join(data_dir, query_file)
        self.query_data = self._load_jsonl(query_path)
        
        # 构建 idx -> Q_minus 映射
        self.idx_to_q_minus = {}
        for item in self.query_data:
            idx = item['idx']
            output = json.loads(item['output'])
            q_minus = output.get('Q_minus', '[NONE]')
            self.idx_to_q_minus[idx] = q_minus
        
        self.samples = self._build_samples()
        print(f"LAP数据集: {len(self.samples)} 个训练样本")
    
    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _build_samples(self) -> List[Tuple]:
        samples = []
        
        for item in self.train_data:
            idx = item['idx']
            instruction = item.get('instruction', '')
            query = item.get('query', '')
            full_query = f"{instruction} {query}".strip()
            
            q_minus_text = self.idx_to_q_minus.get(idx, '[NONE]')
            
            pos_docs = item.get('pos', [])
            neg_docs = item.get('neg', [])
            
            if not pos_docs or not neg_docs:
                continue
            
            for pos_doc in pos_docs:
                for neg_doc in neg_docs[:self.num_neg_per_query]:
                    samples.append((
                        q_minus_text,
                        pos_doc,
                        neg_doc,
                        full_query,
                        instruction,
                        query,
                        idx
                    ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, str]:
        (q_minus_text, doc_pos_text, doc_neg_text, 
         full_query, instruction, query, idx) = self.samples[index]
        
        return {
            'q_minus_text': q_minus_text,
            'doc_pos_text': doc_pos_text,
            'doc_neg_text': doc_neg_text,
            'full_query': full_query,
            'instruction': instruction,
            'query': query,
            'idx': idx
        }


def lap_collate_fn(batch: List[Dict]) -> Dict[str, List[str]]:
    """
    DataLoader 的 collate 函数
    
    将批次样本整理为列表格式
    """
    return {
        'q_minus_text': [item['q_minus_text'] for item in batch],
        'doc_pos_text': [item['doc_pos_text'] for item in batch],
        'doc_neg_text': [item['doc_neg_text'] for item in batch],
        'idx': [item['idx'] for item in batch]
    }


def lap_collate_fn_full(batch: List[Dict]) -> Dict[str, List[str]]:
    """
    DataLoader 的 collate 函数（带完整查询信息）
    """
    return {
        'q_minus_text': [item['q_minus_text'] for item in batch],
        'doc_pos_text': [item['doc_pos_text'] for item in batch],
        'doc_neg_text': [item['doc_neg_text'] for item in batch],
        'full_query': [item['full_query'] for item in batch],
        'instruction': [item['instruction'] for item in batch],
        'query': [item['query'] for item in batch],
        'idx': [item['idx'] for item in batch]
    }


class LAPDatasetWithCache(Dataset):
    """
    LAP 数据集（利用缓存的文档嵌入）

    策略：
    - Q_minus: 实时编码（因为需要被 LAP 投影）
    - D_pos / D_neg: 从缓存加载（已预计算）

    数据来源：
    - train_data_dsclr.jsonl: 包含 query, instruction, pos, neg
    - distilled_queries_v4.jsonl: 包含 Q_plus, Q_minus
    - embeddings/{encoder}/dsclr_train_embeddings_{encoder}.pt: 缓存的嵌入向量

    Args:
        data_dir: 训练数据目录
        cache_path: 缓存的嵌入向量路径
        encoder_type: 编码器类型，用于加载对应的缓存
        num_neg_per_query: 每个 query 使用的负样本数量
    """

    def __init__(
        self,
        data_dir: str = "dataset/FollowIR_train",
        cache_path: str = None,
        encoder_type: str = "repllama",
        num_neg_per_query: int = 15,
        train_file: str = None
    ):
        self.data_dir = data_dir
        self.num_neg_per_query = num_neg_per_query

        # 自动推断训练文件和缓存路径
        if train_file is None:
            # 根据 cache_path 推断训练文件
            if cache_path and "hard_negatives" in cache_path:
                train_file = "train_data_hard_negatives.jsonl"
                print(f"检测到难负样本缓存，使用训练文件: {train_file}")
            else:
                train_file = "train_data_dsclr.jsonl"
        self.train_file = train_file

        if cache_path is None:
            encoder_dir = {
                "repllama": "repllama-v1-7b",
                "mistral": "e5-mistral-7b",
                "bge": "bge-large-en"
            }.get(encoder_type, encoder_type)
            
            # 根据训练文件推断缓存文件名
            if "hard_negatives" in train_file:
                cache_name = f"dsclr_train_hard_negatives_embeddings.pt"
                print(f"使用难负样本缓存: {cache_name}")
            else:
                cache_name = f"dsclr_train_embeddings_{encoder_type}.pt"
                if encoder_type == "repllama":
                    cache_name = f"dsclr_train_embeddings_{encoder_dir}.pt"
            
            cache_path = os.path.join(
                data_dir, "embeddings", encoder_dir, cache_name
            )
        self.cache_path = cache_path

        train_path = os.path.join(data_dir, self.train_file)
        query_path = os.path.join(data_dir, "distilled_queries_v4.jsonl")

        self.train_data = self._load_jsonl(train_path)
        self.query_data = self._load_jsonl(query_path)

        self.idx_to_q_minus = {}
        for item in self.query_data:
            idx = item['idx']
            output = json.loads(item['output'])
            q_minus = output.get('Q_minus', '[NONE]')
            self.idx_to_q_minus[idx] = q_minus

        self.samples = self._build_samples()
        print(f"加载缓存文档嵌入: {cache_path}")
        self._load_cache()
        print(f"LAP数据集(缓存模式): {len(self.samples)} 个训练样本")

    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _load_cache(self):
        """加载缓存的嵌入向量"""
        raw_data = torch.load(self.cache_path, weights_only=False)

        if 'q_plus_embeddings' in raw_data:
            self.q_plus_emb = raw_data['q_plus_embeddings'].float()
            self.q_minus_emb = raw_data['q_minus_embeddings'].float()
            self.pos_emb = raw_data['pos_embeddings'].float()
            self.neg_emb = raw_data['neg_embeddings'].float()
        else:
            self.q_plus_emb = raw_data['q_plus'].float()
            self.q_minus_emb = raw_data['q_minus'].float()
            self.pos_emb = raw_data['pos'].float()
            self.neg_emb = raw_data['neg'].float()

        self.num_neg_cache = raw_data.get('num_neg_per_query', self.num_neg_per_query)
        print(f"  缓存形状: q_plus={self.q_plus_emb.shape}, pos={self.pos_emb.shape}, neg={self.neg_emb.shape}")

    def _build_samples(self) -> List[Tuple]:
        samples = []

        for item in self.train_data:
            idx = item['idx']
            q_minus_text = self.idx_to_q_minus.get(idx, '[NONE]')

            pos_docs = item.get('pos', [])
            neg_docs = item.get('neg', [])

            if not pos_docs or not neg_docs:
                continue

            for pos_idx, pos_doc in enumerate(pos_docs):
                for neg_idx, neg_doc in enumerate(neg_docs[:self.num_neg_per_query]):
                    samples.append((
                        q_minus_text,
                        idx,
                        pos_idx,
                        neg_idx
                    ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        q_minus_text, idx, pos_idx, neg_idx = self.samples[index]

        q_plus_emb = self.q_plus_emb[idx].float()
        q_minus_emb = self.q_minus_emb[idx].float()
        pos_emb = self.pos_emb[idx].float()
        neg_start = idx * self.num_neg_cache
        neg_emb = self.neg_emb[neg_start:neg_start + self.num_neg_cache].float()

        return {
            'q_minus_text': q_minus_text,
            'q_minus_emb': q_minus_emb,
            'q_plus_emb': q_plus_emb,
            'doc_pos_emb': pos_emb,
            'doc_neg_emb': neg_emb,
            'idx': idx
        }


def lap_cache_collate_fn(batch: List[Dict]) -> Dict:
    """
    DataLoader 的 collate 函数（缓存模式）

    Returns:
        q_minus_text: Q- 文本列表（用于调试）
        q_minus_emb: Q- 嵌入 [batch, hidden_dim]（预缓存）
        q_plus_emb: Q+ 嵌入 [batch, hidden_dim]
        doc_pos_emb: 正文档嵌入 [batch, hidden_dim]
        doc_neg_emb: 负文档嵌入 [batch, hidden_dim]
        idx: 样本索引
    """
    q_minus_text = [item['q_minus_text'] for item in batch]
    q_minus_emb = torch.stack([item['q_minus_emb'] for item in batch])
    q_plus_emb = torch.stack([item['q_plus_emb'] for item in batch])
    doc_pos_emb = torch.stack([item['doc_pos_emb'] for item in batch])
    doc_neg_emb = torch.stack([item['doc_neg_emb'] for item in batch])
    idx = [item['idx'] for item in batch]

    return {
        'q_minus_text': q_minus_text,
        'q_minus_emb': q_minus_emb,
        'q_plus_emb': q_plus_emb,
        'doc_pos_emb': doc_pos_emb,
        'doc_neg_emb': doc_neg_emb,
        'idx': idx
    }


if __name__ == "__main__":
    # 测试数据集加载
    print("="*60)
    print("测试 LAPDataset")
    print("="*60)
    
    dataset = LAPDataset(
        data_dir="dataset/FollowIR_train",
        num_neg_per_query=1
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"\n样本示例:")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, str):
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
