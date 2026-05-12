#!/usr/bin/env python
"""
DSCLR-DEO Manual Evaluation Script
使用手动改写的 DEO 风格查询进行评估

DEO 核心逻辑:
1. 正向得分: 取 3 个正向锚点的平均分 (Mean-Pooling)
2. 负向得分: 取 5 个负向约束中的最大值 (Max-Pooling)
3. 最终得分: S_final = Mean(S_pos) - α * Softplus(max(S_neg) - τ)
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

sys.path.insert(0, '/home/luwa/Documents/DSCLR')

from model.reformulator import QueryReformulator
from eval.models.repllama_encoder import RepLLaMAEncoder
from eval.metrics.evaluator import FollowIREvaluator
from eval.engine import FollowIRDataLoader
from eval.metrics.evaluator import DataLoader
from model.dsclr_scoring import dsclr_softplus_score


def load_deo_queries(file_path: str) -> Dict[str, Dict]:
    """加载 DEO 风格的查询数据"""
    queries = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            qid = data['qid']
            queries[qid] = data
    return queries


def encode_anchors(encoder, anchors: List[str], device: str) -> torch.Tensor:
    """编码正向锚点集合"""
    embeddings = []
    for anchor in anchors:
        emb = encoder.encode_queries([anchor]).to(device)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)  # [num_anchors, dim]


def encode_negatives(encoder, negatives: List[str], device: str) -> torch.Tensor:
    """编码负向约束集合"""
    if not negatives or negatives == ["[NONE]"]:
        return None

    embeddings = []
    for neg in negatives:
        if neg and neg != "[NONE]":
            emb = encoder.encode_queries([neg]).to(device)
            embeddings.append(emb)

    if not embeddings:
        return None

    return torch.cat(embeddings, dim=0)  # [num_negatives, dim]


def compute_deo_score(
    anchor_embeddings: torch.Tensor,
    negative_embeddings: Optional[torch.Tensor],
    corpus_embeddings: torch.Tensor,
    alpha: float,
    delta: float,
    beta: float = 20.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 DEO 风格的最终得分
    
    Args:
        anchor_embeddings: [num_anchors, dim] 正向锚点嵌入
        negative_embeddings: [num_negatives, dim] 或 None 负向约束嵌入
        corpus_embeddings: [num_docs, dim] 文档嵌入
        alpha: 惩罚强度
        delta: tau 偏移量
        beta: Softplus 温度参数
    
    Returns:
        S_final: 最终得分
        S_pos_mean: 正向平均得分 (用于调试)
    """
    # 1. 计算正向得分: Mean-Pooling over anchors
    S_pos = torch.mm(anchor_embeddings, corpus_embeddings.t())  # [num_anchors, num_docs]
    S_pos_mean = S_pos.mean(dim=0, keepdim=True)  # [1, num_docs]
    
    # 2. 计算 tau (使用原始查询和锚点的最小余弦相似度 + delta)
    # 简化: 使用锚点之间的相似度计算 tau
    tau = S_pos.mean() + delta
    
    # 3. 计算负向得分: Max-Pooling over negatives
    if negative_embeddings is not None:
        S_neg = torch.mm(negative_embeddings, corpus_embeddings.t())  # [num_negatives, num_docs]
        S_neg_max = S_neg.max(dim=0, keepdim=True)[0]  # [1, num_docs]
        
        # 4. 计算惩罚
        overflow = S_neg_max - tau
        penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
        penalty = alpha * penalty
        
        S_final = S_pos_mean - penalty
    else:
        S_final = S_pos_mean
        penalty = torch.zeros_like(S_pos_mean)
    
    return S_final, S_pos_mean


def evaluate_deo(
    model_name: str = 'samaya-ai/RepLLaMA-reproduced',
    device: str = 'cuda:0',
    alpha: float = 1.0,
    delta: float = -0.15,
    beta: float = 20.0
):
    """评估 DEO 风格改写的效果"""
    
    print("=" * 80)
    print("DSCLR-DEO Manual Evaluation")
    print("=" * 80)
    print(f"Parameters: alpha={alpha}, delta={delta}, beta={beta}")
    print("=" * 80)
    
    # 1. 加载数据
    task_name = 'Core17InstructionRetrieval'
    data_loader = FollowIRDataLoader(task_name)
    corpus, q_og, q_changed, candidates = data_loader.load()
    
    dl = DataLoader(task_name)
    qrels = dl.load_qrels()
    qrel_diff = dl.load_qrel_diff()
    
    # 2. 加载 DEO 查询
    deo_file = '/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v5/deo_manual_core17_changed.jsonl'
    deo_queries = load_deo_queries(deo_file)
    print(f"\n✅ Loaded {len(deo_queries)} DEO-style queries")

    # 3. 加载编码器和文档嵌入
    encoder = RepLLaMAEncoder(model_name=model_name, device=device)

    cache_dir = '/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/RepLLaMA_reproduced'
    corpus_emb_path = os.path.join(cache_dir, f'{task_name}_RepLLaMA_reproduced_corpus_embeddings.npy')
    corpus_ids_path = os.path.join(cache_dir, f'{task_name}_RepLLaMA_reproduced_corpus_ids.json')

    # 加载嵌入 - 缓存是字典格式 {doc_id: tensor}
    embeddings_dict = np.load(corpus_emb_path, allow_pickle=True).item()
    with open(corpus_ids_path, 'r') as f:
        doc_ids = json.load(f)

    # 将字典转换为 tensor 矩阵
    corpus_embeddings = torch.stack([embeddings_dict[did] for did in doc_ids]).to(device)
    corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
    print(f"✅ Loaded corpus: {corpus_embeddings.shape[0]} documents")
    
    # 4. 评估每个查询
    results_og = {}
    results_changed = {}
    
    for qid in qrel_diff.keys():
        changed_qid = f'{qid}-changed'
        og_qid = f'{qid}-og'
        
        # 获取原始查询
        og_text = q_og.get(og_qid, "")
        
        # 编码原始查询 (baseline)
        og_emb = encoder.encode_queries([og_text])
        og_emb = torch.nn.functional.normalize(og_emb, p=2, dim=1).to(device)
        
        with torch.no_grad():
            S_og = torch.mm(og_emb, corpus_embeddings.t())
        
        # 提取原始查询结果
        top_indices = torch.argsort(S_og[0], descending=True)[:1000]
        results_og[og_qid] = {doc_ids[idx]: float(S_og[0, idx]) for idx in top_indices.cpu().numpy()}
        
        # 处理 changed 查询
        if changed_qid in deo_queries:
            deo_data = deo_queries[changed_qid]

            # 编码正向锚点
            anchors = deo_data.get('positive_anchors', [])
            if anchors:
                anchor_embeddings = encode_anchors(encoder, anchors, device)
                anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1).to(device)
            else:
                # 回退到原始查询
                anchor_embeddings = og_emb

            # 编码负向约束
            negatives = deo_data.get('negative_constraints', [])
            negative_embeddings = encode_negatives(encoder, negatives, device)
            if negative_embeddings is not None:
                negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=1).to(device)
            
            # 计算 DEO 得分
            with torch.no_grad():
                S_final, S_pos_mean = compute_deo_score(
                    anchor_embeddings,
                    negative_embeddings,
                    corpus_embeddings,
                    alpha=alpha,
                    delta=delta,
                    beta=beta
                )
            
            # 提取结果
            top_indices = torch.argsort(S_final[0], descending=True)[:1000]
            results_changed[changed_qid] = {doc_ids[idx]: float(S_final[0, idx]) for idx in top_indices.cpu().numpy()}
        else:
            # 如果没有 DEO 数据，使用原始查询
            results_changed[changed_qid] = results_og[og_qid].copy()
    
    # 5. 评估指标
    evaluator = FollowIREvaluator(task_name)
    metrics = evaluator.evaluate(results_og, results_changed)
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Original MAP@1000:  {metrics['original']['map_at_1000']:.4f}")
    print(f"Changed MAP@1000:   {metrics['changed']['map_at_1000']:.4f}")
    print(f"Changed NDCG@5:     {metrics['changed']['ndcg_at_5']:.4f}")
    print(f"Changed NDCG@10:    {metrics['changed']['ndcg_at_10']:.4f}")
    print(f"Changed NDCG@100:   {metrics['changed'].get('ndcg_at_100', 0):.4f}")
    print(f"p-MRR:              {metrics['p-MRR']:.4f}")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    # 使用之前最稳的参数进行测试
    metrics = evaluate_deo(
        alpha=1.0,
        delta=-0.15,
        beta=20.0
    )
