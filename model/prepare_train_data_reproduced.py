#!/usr/bin/env python
"""
训练数据准备脚本 - 使用 RepLLaMAEncoder
支持 samaya-ai/RepLLaMA-reproduced 模型
确保 max_seq_length=2600，无截断
"""

import json
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.models.repllama_encoder import RepLLaMAEncoder


def load_jsonl(path):
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_training_data(
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    device: str = "cuda:0",
    batch_size: int = 8,
    output_dir: str = "dataset/FollowIR_train/embeddings",
    data_dir: str = "dataset/FollowIR_train",
    max_seq_length: int = 2600
):
    """
    准备训练数据缓存
    
    Args:
        model_name: 模型名称
        device: GPU 设备
        batch_size: 批处理大小
        output_dir: 输出目录
        data_dir: 训练数据目录
        max_seq_length: 最大序列长度（确保无截断）
    """
    print("="*60)
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"批次: {batch_size}")
    print(f"最大序列长度: {max_seq_length}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_short = "repllama-reproduced"
    cache_filename = f"dsclr_train_embeddings_{model_short}.pt"
    cache_path = os.path.join(output_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"⚠️ 缓存已存在: {cache_path}")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("使用已有缓存，退出")
            return cache_path
    
    print("\n" + "="*60)
    print("Step 1: 加载数据...")
    print("="*60)
    
    train_data = load_jsonl(os.path.join(data_dir, "train_data_dsclr.jsonl"))
    print(f"训练数据: {len(train_data)} 条")
    
    distilled_queries = load_jsonl(os.path.join(data_dir, "distilled_queries_v4.jsonl"))
    print(f"双流查询: {len(distilled_queries)} 条")
    
    q_plus_list = []
    q_minus_list = []
    for item in tqdm(distilled_queries, desc="解析双流查询"):
        output = json.loads(item['output'])
        q_plus_list.append(output['Q_plus'])
        q_minus_list.append(output['Q_minus'])
    
    print("\n" + "="*60)
    print("Step 2: 加载编码器...")
    print("="*60)
    
    encoder = RepLLaMAEncoder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
        max_seq_length=max_seq_length
    )
    
    print(f"✅ 编码器加载完成，max_seq_length={encoder.max_seq_length}")
    
    print("\n" + "="*60)
    print("Step 3: 编码 Q+ 和 Q- ...")
    print("="*60)
    
    print("\n编码 Q+ (正向查询)...")
    q_plus_embeddings = encoder.encode_queries(q_plus_list, batch_size=batch_size)
    if isinstance(q_plus_embeddings, np.ndarray):
        q_plus_embeddings = torch.from_numpy(q_plus_embeddings)
    print(f"Q+ embeddings shape: {q_plus_embeddings.shape}")
    
    print("\n编码 Q- (负向查询)...")
    q_minus_embeddings = encoder.encode_queries(q_minus_list, batch_size=batch_size)
    if isinstance(q_minus_embeddings, np.ndarray):
        q_minus_embeddings = torch.from_numpy(q_minus_embeddings)
    print(f"Q- embeddings shape: {q_minus_embeddings.shape}")
    
    print("\n处理 Q- = [NONE] 的样本...")
    none_count = 0
    for i, q_minus_text in enumerate(q_minus_list):
        if q_minus_text == '[NONE]' or not q_minus_text.strip():
            q_minus_embeddings[i] = torch.zeros_like(q_minus_embeddings[i])
            none_count += 1
    print(f"已将 {none_count} 个 [NONE] 样本的嵌入设置为零向量")
    
    print("\n" + "="*60)
    print("Step 4: 编码文档 (Pos + Neg) ...")
    print("="*60)
    
    pos_docs = []
    neg_docs = []
    for item in train_data:
        pos_docs.extend(item['pos'])
        neg_docs.extend(item['neg'])
    
    print(f"正样本文档: {len(pos_docs)} 个")
    print(f"负样本文档: {len(neg_docs)} 个")
    
    print("\n编码正样本文档...")
    pos_embeddings = encoder.encode_documents(pos_docs, batch_size=batch_size)
    if isinstance(pos_embeddings, np.ndarray):
        pos_embeddings = torch.from_numpy(pos_embeddings)
    print(f"Pos embeddings shape: {pos_embeddings.shape}")
    
    print("\n编码负样本文档...")
    neg_embeddings = encoder.encode_documents(neg_docs, batch_size=batch_size)
    if isinstance(neg_embeddings, np.ndarray):
        neg_embeddings = torch.from_numpy(neg_embeddings)
    print(f"Neg embeddings shape: {neg_embeddings.shape}")
    
    print("\n" + "="*60)
    print("Step 5: 保存缓存...")
    print("="*60)
    
    torch.save({
        'q_plus_embeddings': q_plus_embeddings,
        'q_minus_embeddings': q_minus_embeddings,
        'pos_embeddings': pos_embeddings,
        'neg_embeddings': neg_embeddings,
        'model_name': model_name,
        'embed_dim': q_plus_embeddings.shape[1],
        'max_seq_length': max_seq_length
    }, cache_path)
    
    print(f"✅ 缓存已保存: {cache_path}")
    print(f"文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")
    
    return cache_path


def main():
    parser = argparse.ArgumentParser(description='准备 RepLLaMA-reproduced 训练数据缓存')
    parser.add_argument('--model_name', type=str, 
                        default='samaya-ai/RepLLaMA-reproduced',
                        help='模型名称')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--output_dir', type=str, default='dataset/FollowIR_train/embeddings/repllama-reproduced',
                        help='输出目录')
    parser.add_argument('--data_dir', type=str, default='dataset/FollowIR_train',
                        help='训练数据目录')
    parser.add_argument('--max_seq_length', type=int, default=2600,
                        help='最大序列长度（确保无截断）')
    
    args = parser.parse_args()
    
    prepare_training_data(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length
    )


if __name__ == "__main__":
    main()
