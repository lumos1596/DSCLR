"""
DeIR 混合检索器训练脚本

支持四种配置：
1. baseline: 传统双塔 Baseline（原始 DSCLR 损失）
2. mlp_only: 纯 MLP 动态路由（DSCLR + MLP 预测）
3. lap_only: 纯 LAP 矩阵投影（LAP 对比损失）
4. hybrid: 终极完全体 Hybrid（DSCLR + MLP + LAP 组合）

梯度隔离说明：
- baseline: 无需梯度（全是静态参数）
- mlp_only: 仅训练 MLP.parameters()
- lap_only: 仅训练 lap.parameters()
- hybrid: 训练 lap.parameters() + mlp.parameters()
"""

import os
import sys
import argparse
import json
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.deir_hybrid_retriever import DeIR_HybridRetriever, DeIR_HybridRetrieverConfig
from model.dsclr_mlp import DSCLR_MLP
from model.lap_module import LAPProjection
from model.lap_loss import LAPContrastiveLoss
from model.lap_dataset import LAPDataset, LAPDatasetWithCache, lap_collate_fn, lap_cache_collate_fn


class DSCLRDataset(Dataset):
    """DSCLR 数据集加载器"""
    def __init__(self, cache_path, num_neg=15):
        raw_data = torch.load(cache_path, weights_only=False)

        if 'q_plus_embeddings' in raw_data:
            self.data = {
                'q_plus': raw_data['q_plus_embeddings'].float(),
                'q_minus': raw_data['q_minus_embeddings'].float(),
                'pos': raw_data['pos_embeddings'].float(),
                'neg': raw_data['neg_embeddings'].float()
            }
            self.num_neg = raw_data.get('num_neg_per_query', num_neg)
        else:
            self.data = {
                'q_plus': raw_data['q_plus'].float(),
                'q_minus': raw_data['q_minus'].float(),
                'pos': raw_data['pos'].float(),
                'neg': raw_data['neg'].float()
            }
            self.num_neg = num_neg

        self.num_queries = len(self.data['q_plus'])

    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):
        q_plus = self.data['q_plus'][idx]
        q_minus = self.data['q_minus'][idx]

        pos_start = idx
        pos = self.data['pos'][pos_start]

        neg_start = idx * self.num_neg
        neg = self.data['neg'][neg_start:neg_start + self.num_neg]

        return {
            'q_plus': q_plus,
            'q_minus': q_minus,
            'pos': pos,
            'neg': neg
        }


def collate_fn(batch):
    q_plus = torch.stack([item['q_plus'] for item in batch])
    q_minus = torch.stack([item['q_minus'] for item in batch])
    pos = torch.stack([item['pos'] for item in batch])
    neg = torch.stack([item['neg'] for item in batch])

    return {
        'q_plus': q_plus,
        'q_minus': q_minus,
        'pos': pos,
        'neg': neg,
        'doc_pos_emb': pos,
        'doc_neg_emb': neg
    }


def compute_scores_dsclr(q_plus, q_minus, pos, neg, alpha, tau):
    """
    计算 DSCLR 分数
    """
    batch_size = q_plus.size(0)
    num_neg = neg.size(1)

    q_plus_expanded = q_plus.unsqueeze(1)
    pos_expanded = pos.unsqueeze(2)
    neg_transposed = neg.transpose(1, 2)

    S_base_pos = torch.bmm(q_plus_expanded, pos_expanded).squeeze(-1)
    S_base_neg = torch.bmm(q_plus_expanded, neg_transposed).squeeze(1)

    tau_expanded = tau.unsqueeze(1)
    alpha_expanded = alpha.unsqueeze(1)

    penalty_pos = torch.zeros_like(S_base_pos)
    penalty_neg = torch.relu(S_base_neg - tau_expanded)

    S_final_pos = S_base_pos - alpha_expanded * penalty_pos
    S_final_neg = S_base_neg - alpha_expanded * penalty_neg

    S_final = torch.cat([S_final_pos, S_final_neg], dim=1)

    return S_final


def dsclr_loss(S_final, temperature=0.1):
    """DSCLR InfoNCE 损失"""
    logits = S_final / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def pairwise_margin_loss(S_final_pos, S_final_neg, margin=0.2):
    """Pairwise Margin Ranking Loss

    强制要求好文章的分数大于每个烂文章的分数至少 margin

    Args:
        S_final_pos: 好文章最终得分 [batch_size, 1] 或 [batch_size]
        S_final_neg: 烂文章最终得分 [batch_size, num_neg]
        margin: 安全边界，默认 0.2
    """
    batch_size = S_final_pos.size(0)
    num_neg = S_final_neg.size(1)

    S_pos_expanded = S_final_pos.view(batch_size, 1).expand(-1, num_neg)

    target = torch.ones_like(S_final_neg)

    criterion = nn.MarginRankingLoss(margin=margin)
    loss = criterion(S_pos_expanded, S_final_neg, target)

    return loss


class DeIRFriendlyFireLoss(nn.Module):
    """DeIR MLP 阶段损失函数：排名损失 + Friendly Fire Regularization (Shield Loss)
    
    引入 Shield Loss 强制提升护盾水位 tau，解决正样本被误伤问题。
    """
    def __init__(self, margin=0.2, shield_epsilon=0.05, shield_weight=2.0):
        super().__init__()
        self.margin = margin
        self.shield_epsilon = shield_epsilon  # 安全缓冲区
        self.shield_weight = shield_weight    # 护盾损失权重
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        
    def forward(self, s_final_pos, s_final_neg, tau, s_neg_proj_pos, target=None):
        """
        Args:
            s_final_pos: 最终的正样本得分 (S_pos - alpha * max(0, S_neg_proj_pos - tau))
            s_final_neg: 最终的负样本得分
            tau: MLP 预测的当前 Query 的护盾值 [batch_size]
            s_neg_proj_pos: 正样本在负向空间的投影得分 (S_neg_proj) [batch_size]
            target: 标签 (通常为 1)，可选
        Returns:
            total_loss, loss_rank, loss_shield
        """
        batch_size = s_final_pos.size(0)
        num_neg = s_final_neg.size(1)
        
        # 扩展正样本得分以匹配负样本维度
        s_pos_expanded = s_final_pos.view(batch_size, 1).expand(-1, num_neg)
        
        if target is None:
            target = torch.ones_like(s_final_neg)
        
        # 1. 基础排名损失：确保正样本排在负样本前面
        loss_rank = self.ranking_loss(s_pos_expanded, s_final_neg, target)
        
        # 2. 护盾损失：强制要求 tau > s_neg_proj_pos + epsilon
        # 当正样本得分击穿护盾时产生梯度，逼迫 MLP 调高 tau
        loss_shield = torch.mean(torch.relu(s_neg_proj_pos - tau + self.shield_epsilon))
        
        # 3. 联合优化：给予护盾损失高权重，感知"误杀代价"
        total_loss = loss_rank + self.shield_weight * loss_shield
        
        return total_loss, loss_rank, loss_shield


def plot_realtime_loss(history, output_dir, is_mlp_phase=False):
    """实时绘制并保存损失曲线和相似度曲线
    
    Args:
        history: 训练历史记录
        output_dir: 输出目录
        is_mlp_phase: 是否是MLP阶段（不绘制LAP相似度曲线）
    """
    try:
        import matplotlib.pyplot as plt

        if not history['train_loss']:
            return

        steps = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：总损失对比
        axes[0].plot(steps, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
        axes[0].plot(steps, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('LAP Training - Total Loss', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 右图：LAP相似度曲线（关键监控指标！）
        if is_mlp_phase:
            # MLP阶段：显示说明文字
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'MLP Phase\n(LAP Frozen)\n\nNo LAP Similarity', 
                        ha='center', va='center', fontsize=14, 
                        transform=axes[1].transAxes)
            axes[1].set_title('MLP Phase (LAP Frozen)', fontsize=14)
        elif 'sim_good' in history and len(history['sim_good']) > 0:
            # LAP阶段：显示相似度曲线（这才是应该看的！）
            axes[1].plot(steps, history['sim_good'], 'g-', label='Sim_Good (q_neg vs d_pos)', linewidth=2, marker='v')
            axes[1].plot(steps, history['sim_bad'], 'orange', label='Sim_Bad (q_neg vs d_neg)', linewidth=2, marker='^')
            # 💥 完全自适应：显示动态目标线
            if 'adaptive_target' in history and len(history['adaptive_target']) > 0:
                axes[1].plot(steps, history['adaptive_target'], 'g--', alpha=0.7, label='Dynamic Target (adaptive)', linewidth=1.5)
            axes[1].axhline(y=0.80, color='gray', linestyle=':', alpha=0.3, label='Ref: sim_bad baseline')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Cosine Similarity', fontsize=12)
            axes[1].set_title('LAP - Similarity Tracking (Fully Adaptive)', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
            axes[1].set_title('No LAP Similarity Data', fontsize=14)

        plt.tight_layout()
        loss_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        json_path = os.path.join(output_dir, "training_history.json")
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


def lap_loss(q_minus_proj, q_minus_raw, d_pos, d_neg, criterion_lap, dynamic_tau=0.80):
    """LAP 逆向对比损失 - 自适应动态锚点版
    
    Args:
        q_minus_proj: LAP 投影后的负向查询向量
        q_minus_raw: 原始负向查询向量（未投影）
        d_pos: 正样本文档向量
        d_neg: 负样本文档向量
        criterion_lap: LAP 损失函数
        dynamic_tau: 动态底噪水平
    
    Returns:
        total_loss: 总损失
        sim_good: 好文相似度均值
        sim_bad: 烂文相似度均值
        base_sim_bad: 原始烂文相似度均值
    """
    return criterion_lap(q_minus_proj, q_minus_raw, d_pos, d_neg, dynamic_tau)


def encode_texts(encoder, texts, batch_size, mode="document"):
    """
    编码文本为向量

    Args:
        encoder: 编码器实例
        texts: 文本列表
        batch_size: 批处理大小
        mode: "query" 或 "document"

    Returns:
        embeddings: [N, hidden_dim] 已 L2 归一化
    """
    import numpy as np
    from eval.models.e5_mistral_encoder import E5MistralEncoder
    from eval.models.repllama_encoder import RepLLaMAEncoder

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        if isinstance(encoder, E5MistralEncoder):
            if mode == "query":
                batch_emb = encoder.encode_queries(batch_texts, batch_size=batch_size)
            else:
                batch_emb = encoder.encode_documents(batch_texts, batch_size=batch_size)
        elif isinstance(encoder, RepLLaMAEncoder):
            if mode == "query":
                batch_emb = encoder.encode_queries(batch_texts, batch_size=batch_size)
            else:
                batch_emb = encoder.encode_documents(batch_texts, batch_size=batch_size)
        else:
            # SentenceTransformer
            batch_emb = encoder.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        if isinstance(batch_emb, np.ndarray):
            batch_emb = torch.from_numpy(batch_emb)
        elif hasattr(batch_emb, 'cpu'):
            batch_emb = batch_emb.cpu()

        all_embeddings.append(batch_emb)

    return torch.cat(all_embeddings, dim=0)


def load_encoder(model_type, device, batch_size):
    """
    加载编码器

    Args:
        model_type: 编码器类型 ('bge', 'mistral', 'repllama')
        device: GPU 设备
        batch_size: 批处理大小

    Returns:
        encoder, hidden_dim
    """
    if model_type == "mistral":
        from eval.models.e5_mistral_encoder import E5MistralEncoder
        encoder = E5MistralEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            device=device,
            batch_size=min(batch_size, 28),
            normalize_embeddings=True
        )
        hidden_dim = 4096
    elif model_type == "repllama":
        from eval.models.repllama_encoder import RepLLaMAEncoder
        encoder = RepLLaMAEncoder(
            model_name="samaya-ai/RepLLaMA-reproduced",
            device=device,
            batch_size=min(batch_size, 28),
            normalize_embeddings=True
        )
        hidden_dim = 4096
    else:  # bge
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
        encoder = encoder.half()
        hidden_dim = 1024

    return encoder


def train_epoch_baseline(dataloader, model, optimizer, device):
    """Baseline 模式训练（无可训练参数）"""
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Baseline Training"):
        q_plus = batch['q_plus'].to(device)
        q_minus = batch['q_minus'].to(device)
        pos = batch['pos'].to(device)
        neg = batch['neg'].to(device)

        alpha = torch.full((q_plus.size(0),), model.static_alpha, device=device)
        tau = torch.full((q_plus.size(0),), model.static_tau, device=device)

        S_final = compute_scores_dsclr(q_plus, q_minus, pos, neg, alpha, tau)
        loss = dsclr_loss(S_final)

        total_loss += loss.item()
        num_batches += 1

    return {'loss': total_loss / num_batches}


def train_epoch_mlp(dataloader, model, optimizer, device, encoder_type):
    """MLP Only 模式训练"""
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="MLP Training"):
        q_plus = batch['q_plus'].to(device)
        q_minus = batch['q_minus'].to(device)
        pos = batch['pos'].to(device)
        neg = batch['neg'].to(device)

        alpha, tau = model.mlp(q_plus, q_minus)

        S_final = compute_scores_dsclr(q_plus, q_minus, pos, neg, alpha, tau)
        loss = dsclr_loss(S_final)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {'loss': total_loss / num_batches}


def train_epoch_mlp_with_cache(dataloader, model, optimizer, device, encoder, encode_batch_size):
    """MLP 阶段训练（缓存模式，LAP 已冻结，使用 q_plus_emb 和 q_minus_proj）

    MLP 接收：
    - q_plus_emb: 来自缓存（正向意图）
    - q_minus_proj: LAP(q_minus_emb)，需要实时编码

    使用 Pairwise Margin Ranking Loss 替代 InfoNCE
    """
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="MLP Training"):
        q_minus_texts = batch['q_minus_text']
        q_plus_emb = batch['q_plus_emb'].to(device)
        doc_pos_emb = batch['doc_pos_emb'].to(device)
        doc_neg_emb = batch['doc_neg_emb'].to(device)

        with torch.no_grad():
            q_minus_emb = encode_texts(encoder, q_minus_texts, encode_batch_size, mode="query")
        q_minus_emb = q_minus_emb.to(device).float()

        with torch.no_grad():
            q_minus_proj = model.lap(q_minus_emb)
            q_minus_proj = F.normalize(q_minus_proj, p=2, dim=-1)

        alpha, tau = model.mlp(q_plus_emb, q_minus_proj)

        batch_size = q_plus_emb.size(0)
        num_neg = doc_neg_emb.size(1)

        S_base_pos = torch.sum(q_minus_proj * doc_pos_emb, dim=-1)
        S_base_neg = torch.bmm(q_minus_proj.unsqueeze(1), doc_neg_emb.transpose(1, 2)).squeeze(1)

        penalty_pos = torch.zeros(batch_size, device=q_plus_emb.device)
        penalty_neg = torch.relu(S_base_neg - tau.unsqueeze(1))

        S_final_pos = S_base_pos - alpha * penalty_pos
        S_final_neg = S_base_neg - alpha.unsqueeze(1) * penalty_neg

        loss = pairwise_margin_loss(S_final_pos, S_final_neg, margin=0.2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {'loss': total_loss / num_batches}


def train_epoch_mlp_with_cache_with_mid_val(dataloader, model, optimizer, device, encoder, encode_batch_size, val_loader, history, output_dir=None):
    """MLP 阶段训练（支持中途每10%验证一次）

    使用 DeIRFriendlyFireLoss: 排名损失 + Shield Loss (Friendly Fire Regularization)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # 初始化 Friendly Fire Loss
    criterion = DeIRFriendlyFireLoss(margin=0.2, shield_epsilon=0.05, shield_weight=2.0)
    
    total_loss = 0
    total_rank_loss = 0
    total_shield_loss = 0
    num_batches = 0
    # 用于记录当前区间的损失
    interval_loss = 0
    interval_rank_loss = 0
    interval_shield_loss = 0
    interval_batches = 0
    total_batches = len(dataloader)
    checkpoint_interval = max(1, total_batches // 10)
    
    MONITOR_INTERVAL = 100  # 每100 step打印监控信息
    
    # 监控统计
    running_alpha = []
    running_tau = []
    running_activation_rate = []
    running_loss_shield = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="MLP Training")):
        q_minus_texts = batch['q_minus_text']
        q_plus_emb = batch['q_plus_emb'].to(device)
        doc_pos_emb = batch['doc_pos_emb'].to(device)
        doc_neg_emb = batch['doc_neg_emb'].to(device)

        with torch.no_grad():
            q_minus_emb = encode_texts(encoder, q_minus_texts, encode_batch_size, mode="query")
        q_minus_emb = q_minus_emb.to(device).float()

        with torch.no_grad():
            q_minus_proj = model.lap(q_minus_emb)
            q_minus_proj = F.normalize(q_minus_proj, p=2, dim=-1)

        alpha, tau = model.mlp(q_plus_emb, q_minus_proj)

        batch_size = q_plus_emb.size(0)
        num_neg = doc_neg_emb.size(1)

        # 计算 S_base_pos: 正样本在负向空间的投影得分 (S_neg_proj for positive docs)
        S_base_pos = torch.sum(q_minus_proj * doc_pos_emb, dim=-1)
        S_base_neg = torch.bmm(q_minus_proj.unsqueeze(1), doc_neg_emb.transpose(1, 2)).squeeze(1)

        penalty_pos = torch.zeros(batch_size, device=q_plus_emb.device)
        penalty_neg = torch.relu(S_base_neg - tau.unsqueeze(1))

        S_final_pos = S_base_pos - alpha * penalty_pos
        S_final_neg = S_base_neg - alpha.unsqueeze(1) * penalty_neg

        # 使用 DeIRFriendlyFireLoss 计算总损失
        loss, loss_rank, loss_shield = criterion(S_final_pos, S_final_neg, tau, S_base_pos)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mlp.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rank_loss += loss_rank.item()
        total_shield_loss += loss_shield.item()
        num_batches += 1
        
        # 累积当前区间的损失
        interval_loss += loss.item()
        interval_rank_loss += loss_rank.item()
        interval_shield_loss += loss_shield.item()
        interval_batches += 1
        
        # 收集监控统计
        running_alpha.append(alpha.mean().item())
        running_tau.append(tau.mean().item())
        running_loss_shield.append(loss_shield.item())
        # Activation rate: 正样本分数高于 tau 的比例（误伤率）
        activation = (S_base_pos > tau).float().mean().item()
        running_activation_rate.append(activation)
        
        # 每100 step打印监控信息
        if (batch_idx + 1) % MONITOR_INTERVAL == 0:
            mean_alpha = np.mean(running_alpha)
            mean_tau = np.mean(running_tau)
            mean_loss_shield = np.mean(running_loss_shield)
            mean_activation = np.mean(running_activation_rate)
            shield_ratio = mean_loss_shield * 2.0 / (mean_loss_shield * 2.0 + np.mean([total_rank_loss / max(num_batches, 1)])) * 100
            
            print(f"\n[Step {batch_idx+1}] Shield Monitor:")
            print(f"  mean_alpha: {mean_alpha:.4f}, mean_tau: {mean_tau:.4f}")
            print(f"  loss_shield: {mean_loss_shield:.4f} (护盾损失)")
            print(f"  activation_rate: {mean_activation*100:.2f}% (正样本 S_neg_proj > tau, 即误伤率)")
            
            # 关键诊断信息
            if mean_loss_shield < 0.001:
                print(f"  ⚠️  WARNING: loss_shield 接近0，说明 tau 已经足够高或 epsilon 太小")
            elif mean_loss_shield > 0.5:
                print(f"  ⚠️  WARNING: loss_shield 过高，正样本被大量误伤，tau 需要大幅提升")
            
            # 重置 running 统计
            running_alpha = []
            running_tau = []
            running_loss_shield = []
            running_activation_rate = []

        if (batch_idx + 1) % checkpoint_interval == 0 or batch_idx == total_batches - 1:
            val_metrics = validate(model, val_loader, device, None, 'mlp_only', None, encoder, encode_batch_size)
            progress = (batch_idx + 1) / total_batches * 100
            avg_rank_loss = total_rank_loss / num_batches
            avg_shield_loss = total_shield_loss / num_batches
            avg_total_loss = total_loss / num_batches
            
            print(f"\n  [{progress:.0f}%][{batch_idx+1}/{total_batches}] Train Loss: {interval_loss/interval_batches:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Rank Loss: {avg_rank_loss:.4f}, Shield Loss: {avg_shield_loss:.4f}, Total: {avg_total_loss:.4f}")

            history['train_loss'].append(interval_loss / interval_batches)
            # MLP 阶段不计算 LAP 分解损失，保持为 None 或前值
            # 不修改 train_loss_pos 和 train_loss_neg，保留 LAP 阶段的最后值
            history['val_loss'].append(val_metrics['loss'])
            plot_realtime_loss(history, output_dir, is_mlp_phase=True)
            
            # 重置区间统计
            interval_loss = 0
            interval_rank_loss = 0
            interval_shield_loss = 0
            interval_batches = 0

    return {'loss': total_loss / num_batches, 'loss_rank': total_rank_loss / num_batches, 'loss_shield': total_shield_loss / num_batches}


def train_epoch_lap(dataloader, model, optimizer, device, criterion_lap, encoder, encode_batch_size, wandb_logger=None):
    """LAP Only 模式训练（使用缓存的 Q_minus 嵌入，无需实时编码）

    Args:
        wandb_logger: Wandb logger instance, if None then skip wandb logging
    """
    total_loss = 0
    total_loss_pos = 0
    total_loss_neg = 0
    total_adaptive_target = 0
    num_batches = 0
    global_step = 0

    for batch in tqdm(dataloader, desc="LAP Training"):
        q_minus_emb = batch['q_minus_emb'].to(device)
        doc_pos_emb = batch['doc_pos_emb'].to(device)
        doc_neg_emb = batch['doc_neg_emb'].to(device)

        batch_size = doc_pos_emb.size(0)
        num_neg = doc_neg_emb.size(1)

        raw_q_minus_proj = model.lap(q_minus_emb, return_raw=True)
        q_minus_proj = F.normalize(raw_q_minus_proj, p=2, dim=-1)

        doc_neg_emb_2d = doc_neg_emb.reshape(batch_size * num_neg, -1)

        sim_pos = torch.matmul(q_minus_proj, doc_pos_emb.T)
        sim_neg = torch.matmul(q_minus_proj, doc_neg_emb_2d.T)
        diag_sim_pos = torch.diag(sim_pos)
        diag_sim_neg = torch.diag(sim_neg)

        # 💥 完全自适应动态锚点：传递原始向量 q_minus_emb 作为锚点
        # 返回的动态目标 = 原始烂文分数 - margin_push - margin_drop
        loss, adaptive_target = lap_loss(q_minus_proj, q_minus_emb, doc_pos_emb, doc_neg_emb_2d, criterion_lap)
        
        # 计算当前相似度用于监控
        with torch.no_grad():
            sim_good = torch.mean(torch.sum(q_minus_proj * doc_pos_emb, dim=-1)).item()
            sim_bad = torch.mean(torch.sum(q_minus_proj.unsqueeze(1) * doc_neg_emb, dim=-1)).item()
        
        # 为了兼容旧代码，将相似度作为"损失"记录
        loss_pos = sim_good  # sim_good 作为 "push loss"
        loss_neg = sim_bad   # sim_bad 作为 "pull loss"
        
        # 保存动态目标用于后续日志
        batch_adaptive_target = adaptive_target

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lap.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_pos += loss_pos  # sim_good 已经是 float
        total_loss_neg += loss_neg  # sim_bad 已经是 float
        total_adaptive_target += batch_adaptive_target  # 💥 累加动态目标
        num_batches += 1

        if wandb_logger is not None:
            # 计算矩阵健康指标
            w_diff = torch.norm(model.lap.proj.weight - torch.eye(model.hidden_dim, device=device), p='fro')
            mean_norm = torch.mean(torch.norm(raw_q_minus_proj, p=2, dim=-1))

            # 安全记录到 wandb（detach + cpu）
            wandb_logger.log({
                "Loss/Total": loss.detach().cpu().item(),
                "Loss/Push_Pos": loss_pos,
                "Loss/Pull_Neg": loss_neg,
                "Matrix/Frobenius_Dist": w_diff.detach().cpu().item(),
                "Matrix/Mean_Norm": mean_norm.detach().cpu().item(),
                "Distribution/Sim_to_Good_Docs": wandb.Histogram(diag_sim_pos.detach().cpu().numpy()),
                "Distribution/Sim_to_Bad_Docs": wandb.Histogram(diag_sim_neg.detach().cpu().numpy()),
                "Step": global_step
            })
            global_step += 1

    # 计算平均动态目标值
    avg_adaptive_target = total_adaptive_target / num_batches if num_batches > 0 else 0
    
    return {
        'loss': total_loss / num_batches,
        'loss_pos': total_loss_pos / num_batches,
        'loss_neg': total_loss_neg / num_batches,
        'sim_good_mean': total_loss_pos / num_batches,  # 实际上是 sim_good 的平均
        'sim_bad_mean': total_loss_neg / num_batches,   # 实际上是 sim_bad 的平均
        'adaptive_target': avg_adaptive_target          # 💥 动态目标值
    }


def train_epoch_lap_with_mid_val(dataloader, model, optimizer, device, criterion_lap, encoder, encode_batch_size, val_loader, history, wandb_logger=None, output_dir=None):
    """LAP 训练（支持中途每10%验证一次）

    Args:
        wandb_logger: Wandb logger instance, if None then skip wandb logging
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    total_loss = 0
    total_loss_pos = 0
    total_loss_neg = 0
    num_batches = 0
    # 用于记录当前区间的损失
    interval_loss = 0
    interval_loss_pos = 0
    interval_loss_neg = 0
    interval_batches = 0
    total_batches = len(dataloader)
    checkpoint_interval = max(1, total_batches // 10)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="LAP Training")):
        q_minus_emb = batch['q_minus_emb'].to(device)
        doc_pos_emb = batch['doc_pos_emb'].to(device)
        doc_neg_emb = batch['doc_neg_emb'].to(device)

        batch_size = doc_pos_emb.size(0)
        num_neg = doc_neg_emb.size(1)

        raw_q_minus_proj = model.lap(q_minus_emb, return_raw=True)
        q_minus_proj = F.normalize(raw_q_minus_proj, p=2, dim=-1)

        # 硬负样本平滑：添加微小高斯噪声防止过拟合
        if model.training:
            noise = torch.randn_like(q_minus_proj) * 0.01
            q_minus_proj = q_minus_proj + noise
            q_minus_proj = F.normalize(q_minus_proj, p=2, dim=-1)

        doc_neg_emb_2d = doc_neg_emb.reshape(batch_size * num_neg, -1)

        loss, loss_pos, loss_neg, loss_metrics = lap_loss(q_minus_proj, doc_pos_emb, doc_neg_emb_2d, criterion_lap)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lap.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_pos += loss_pos.item()
        total_loss_neg += loss_neg.item()
        num_batches += 1
        
        # 累积当前区间的损失
        interval_loss += loss.item()
        interval_loss_pos += loss_pos.item()
        interval_loss_neg += loss_neg.item()
        interval_batches += 1

        if wandb_logger is not None:
            w_diff = torch.norm(model.lap.proj.weight - torch.eye(model.hidden_dim, device=device), p='fro')
            mean_norm = torch.mean(torch.norm(raw_q_minus_proj, p=2, dim=-1))
            wandb_logger.log({
                "Loss/Total": loss.detach().cpu().item(),
                "Loss/Push_Pos": loss_pos.detach().cpu().item(),
                "Loss/Pull_Neg": loss_neg.detach().cpu().item(),
                "Matrix/Frobenius_Dist": w_diff.detach().cpu().item(),
                "Matrix/Mean_Norm": mean_norm.detach().cpu().item(),
                "Step": batch_idx
            })

        if (batch_idx + 1) % checkpoint_interval == 0 or batch_idx == total_batches - 1:
            val_metrics = validate(model, val_loader, device, None, 'lap_only', criterion_lap, encoder, encode_batch_size)
            progress = (batch_idx + 1) / total_batches * 100
            
            pull_push_ratio = (interval_loss_neg / interval_batches) / (interval_loss_pos / interval_batches + 1e-8)
            top1_score_gap = val_metrics.get('top1_score_gap', 0.0)
            avg_push_loss = interval_loss_pos / interval_batches
            
            circuit_breaker_warning = ""
            if avg_push_loss > 0.4:
                circuit_breaker_warning = " ⚠️ 熔断警告: Push Loss过高!"
            if top1_score_gap < 0:
                circuit_breaker_warning = " ⚠️ 熔断警告: Score Gap为负(完全失效)!"
            
            print(f"  [{progress:.0f}%][{batch_idx+1}/{total_batches}] Train Loss: {interval_loss/interval_batches:.4f}, Val Loss: {val_metrics['loss']:.4f} | Push: {avg_push_loss:.4f}, Pull: {interval_loss_neg/interval_batches:.4f} | Score Gap: {top1_score_gap:+.4f}{circuit_breaker_warning}")

            history['train_loss'].append(interval_loss / interval_batches)
            history['train_loss_pos'].append(avg_push_loss)
            history['train_loss_neg'].append(interval_loss_neg / interval_batches)
            history['val_loss'].append(val_metrics['loss'])
            history['score_gap'] = history.get('score_gap', [])
            history['score_gap'].append(top1_score_gap)
            # 记录相似度（关键监控指标！）
            sim_good = loss_metrics.get('sim_good_mean', 0)
            sim_bad = loss_metrics.get('sim_bad_mean', 0)
            history['sim_good'] = history.get('sim_good', [])
            history['sim_bad'] = history.get('sim_bad', [])
            history['sim_good'].append(sim_good)
            history['sim_bad'].append(sim_bad)
            plot_realtime_loss(history, output_dir, is_mlp_phase=False)
            
            # 重置区间统计
            interval_loss = 0
            interval_loss_pos = 0
            interval_loss_neg = 0
            interval_batches = 0

    return {
        'loss': total_loss / num_batches,
        'loss_pos': total_loss_pos / num_batches,
        'loss_neg': total_loss_neg / num_batches
    }


def train_epoch_hybrid(dataloader, model, optimizer, device, encoder_type, criterion_lap, encoder, encode_batch_size, alpha_lap=0.5, wandb_logger=None):
    """Hybrid 模式训练（MLP + LAP 联合，Q_minus 实时编码，文档从缓存加载）

    Args:
        wandb_logger: Wandb logger instance, if None then skip wandb logging
    """
    total_loss = 0
    total_loss_dsclr = 0
    total_loss_lap = 0
    num_batches = 0
    global_step = 0

    for batch in tqdm(dataloader, desc="Hybrid Training"):
        q_minus_texts = batch['q_minus_text']
        q_plus_emb = batch['q_plus_emb'].to(device)
        doc_pos_emb = batch['doc_pos_emb'].to(device)
        doc_neg_emb = batch['doc_neg_emb'].to(device)

        batch_size = doc_pos_emb.size(0)
        num_neg = doc_neg_emb.size(1)

        with torch.no_grad():
            q_minus_emb = encode_texts(encoder, q_minus_texts, encode_batch_size, mode="query")
        q_minus_emb = q_minus_emb.to(device).float()

        # ========== LAP 链路 ==========
        raw_q_minus_proj = model.lap(q_minus_emb, return_raw=True)
        q_minus_proj = F.normalize(raw_q_minus_proj, p=2, dim=-1)

        # ========== MLP 链路 ==========
        alpha, tau = model.mlp(q_plus_emb, q_minus_proj)

        # ========== 损失计算 ==========
        S_final = compute_scores_dsclr_with_emb(q_minus_proj, doc_pos_emb, doc_neg_emb, alpha, tau)
        loss_dsclr = dsclr_loss(S_final)

        doc_neg_emb_2d = doc_neg_emb.reshape(batch_size * num_neg, -1)
        loss_lap, loss_lap_pos, loss_lap_neg, _ = lap_loss(q_minus_proj, doc_pos_emb, doc_neg_emb_2d, criterion_lap)

        loss = loss_dsclr + alpha_lap * loss_lap

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_dsclr += loss_dsclr.item()
        total_loss_lap += loss_lap.item()
        num_batches += 1

        if wandb_logger is not None:
            # 计算矩阵健康指标
            w_diff = torch.norm(model.lap.proj.weight - torch.eye(model.hidden_dim, device=device), p='fro')
            mean_norm = torch.mean(torch.norm(raw_q_minus_proj, p=2, dim=-1))

            # 相似度分布
            sim_pos = torch.matmul(q_minus_proj, doc_pos_emb.T)
            sim_neg = torch.matmul(q_minus_proj, doc_neg_emb.T)
            diag_sim_pos = torch.diag(sim_pos)
            diag_sim_neg = torch.diag(sim_neg)

            wandb_logger.log({
                "Loss/Total": loss.detach().cpu().item(),
                "Loss/DSCLR": loss_dsclr.detach().cpu().item(),
                "Loss/LAP": loss_lap.detach().cpu().item(),
                "Loss/LAP_Push_Pos": loss_lap_pos.detach().cpu().item(),
                "Loss/LAP_Pull_Neg": loss_lap_neg.detach().cpu().item(),
                "Matrix/Frobenius_Dist": w_diff.detach().cpu().item(),
                "Matrix/Mean_Norm": mean_norm.detach().cpu().item(),
                "Distribution/Sim_to_Good_Docs": wandb.Histogram(diag_sim_pos.detach().cpu().numpy()),
                "Distribution/Sim_to_Bad_Docs": wandb.Histogram(diag_sim_neg.detach().cpu().numpy()),
                "Step": global_step
            })
            global_step += 1

    return {
        'loss': total_loss / num_batches,
        'loss_dsclr': total_loss_dsclr / num_batches,
        'loss_lap': total_loss_lap / num_batches
    }


def compute_scores_dsclr_with_emb(q_plus, d_pos, d_neg, alpha, tau):
    """
    计算 DSCLR 分数（使用嵌入向量）

    Args:
        q_plus: Q+ 嵌入 [batch_size, hidden_dim]
        d_pos: 正文档嵌入 [batch_size, hidden_dim]
        d_neg: 负文档嵌入 [batch_size, num_neg, hidden_dim]
        alpha: 惩罚系数
        tau: 容忍阈值
    """
    batch_size = q_plus.size(0)

    S_base_pos = torch.sum(q_plus * d_pos, dim=-1)
    S_base_neg = torch.bmm(q_plus.unsqueeze(1), d_neg.transpose(1, 2)).squeeze(1)

    penalty_pos = torch.zeros(batch_size, device=q_plus.device)
    penalty_neg = torch.relu(S_base_neg - tau.unsqueeze(1))

    S_final_pos = S_base_pos - alpha * penalty_pos
    S_final_neg = S_base_neg - alpha.unsqueeze(1) * penalty_neg

    S_final = torch.cat([S_final_pos.unsqueeze(1), S_final_neg], dim=1)

    return S_final


@torch.no_grad()
def validate(model, dataloader, device, encoder_type, mode, criterion_lap=None, encoder=None, encode_batch_size=32):
    """验证"""
    total_loss = 0
    num_batches = 0
    top1_score_gap = 0.0

    for batch in tqdm(dataloader, desc="Validation"):
        if mode == 'lap_only' or mode == 'hybrid' or mode == 'mlp_only':
            q_minus_emb = batch['q_minus_emb'].to(device)
            doc_pos_emb = batch['doc_pos_emb'].to(device)
            doc_neg_emb = batch['doc_neg_emb'].to(device)
        else:
            q_minus_emb = batch['q_minus'].to(device)
            doc_pos_emb = batch['pos'].to(device)
            doc_neg_emb = batch['neg'].to(device)

        if mode == 'baseline':
            alpha = torch.full((q_minus_emb.size(0),), model.static_alpha, device=device)
            tau = torch.full((q_minus_emb.size(0),), model.static_tau, device=device)
            S_final = compute_scores_dsclr_with_emb(q_minus_emb, doc_pos_emb, doc_neg_emb, alpha, tau)
            loss = dsclr_loss(S_final)

        elif mode == 'mlp_only':
            if 'q_plus_emb' in batch:
                q_plus_emb = batch['q_plus_emb'].to(device)
            else:
                q_plus_emb = batch['q_plus'].to(device)
            if 'doc_neg_emb' in batch:
                doc_neg_emb = batch['doc_neg_emb'].to(device)
            else:
                doc_neg_emb = batch['neg'].to(device)
            alpha, tau = model.mlp(q_plus_emb, q_minus_emb)
            S_final = compute_scores_dsclr_with_emb(q_plus_emb, doc_pos_emb, doc_neg_emb, alpha, tau)
            loss = dsclr_loss(S_final)

        elif mode == 'lap_only':
            q_minus_proj = model.lap(q_minus_emb)
            q_minus_proj_norm = F.normalize(q_minus_proj, p=2, dim=-1)
            batch_size = doc_pos_emb.size(0)
            num_neg = doc_neg_emb.size(1)
            doc_neg_emb_2d = doc_neg_emb.reshape(batch_size * num_neg, -1)
            doc_neg_emb_2d_norm = F.normalize(doc_neg_emb_2d, p=2, dim=-1)
            # 💥 完全自适应动态锚点：传递原始向量 q_minus_emb
            loss, adaptive_target = lap_loss(q_minus_proj_norm, q_minus_emb, doc_pos_emb, doc_neg_emb_2d_norm, criterion_lap)
            # 计算当前相似度用于监控
            with torch.no_grad():
                sim_good = torch.mean(torch.sum(q_minus_proj_norm * doc_pos_emb, dim=-1)).item()
                sim_bad = torch.mean(torch.sum(q_minus_proj_norm.unsqueeze(1) * doc_neg_emb, dim=-1)).item()
            top1_score_gap = sim_bad - sim_good  # 简单的分数差距

        elif mode == 'hybrid':
            q_plus_emb = batch['q_plus_emb'].to(device)
            q_minus_proj = model.lap(q_minus_emb)
            alpha, tau = model.mlp(q_plus_emb, q_minus_proj)
            S_final = compute_scores_dsclr_with_emb(q_minus_proj, doc_pos_emb, doc_neg_emb, alpha, tau)
            loss_dsclr = dsclr_loss(S_final)
            batch_size = doc_pos_emb.size(0)
            num_neg = doc_neg_emb.size(1)
            doc_neg_emb_2d = doc_neg_emb.reshape(batch_size * num_neg, -1)
            # 💥 自适应动态锚点：传递原始向量 q_minus_emb
            loss_lap, _ = lap_loss(q_minus_proj, q_minus_emb, doc_pos_emb, doc_neg_emb_2d, criterion_lap)
            loss = loss_dsclr + 0.5 * loss_lap

        total_loss += loss.item()
        num_batches += 1

    return {'loss': total_loss / num_batches, 'top1_score_gap': top1_score_gap}


def main():
    parser = argparse.ArgumentParser(description='DeIR Hybrid Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--encode_batch_size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--output_dir', type=str, default='train/output/Hybrid', help='Output directory')
    parser.add_argument('--output_dir_is_final', type=lambda x: x.lower() == 'true', default=False, help='If True, use output_dir as final path without appending subdirectories')
    parser.add_argument('--data_dir', type=str, default='dataset/FollowIR_train', help='Training data directory (for LAP)')
    parser.add_argument('--cache_path', type=str, default=None, help='Training data cache path (for MLP/baseline)')
    parser.add_argument('--mlp_cache_path', type=str, default=None, help='Cache path for MLP phase in lap_then_mlp mode (if different from cache_path)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lap_lr', type=float, default=5e-5, help='Learning rate for LAP stage (half of normal lr)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_neg', type=int, default=15, help='Number of negatives')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val split')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--model_type', type=str, default='bge', choices=['bge', 'mistral', 'repllama'])
    parser.add_argument('--use_lap', type=lambda x: x.lower() == 'true', default=False, help='Enable LAP module')
    parser.add_argument('--use_mlp', type=lambda x: x.lower() == 'true', default=False, help='Enable MLP module')
    parser.add_argument('--static_alpha', type=float, default=1.0, help='Static alpha')
    parser.add_argument('--static_tau', type=float, default=0.5, help='Static tau')
    parser.add_argument('--lap_margin_pos', type=float, default=0.1, help='LAP margin for push-away')
    parser.add_argument('--lap_margin_neg', type=float, default=0.6, help='LAP margin for pull-closer')
    parser.add_argument('--lap_loss_weight', type=float, default=0.5, help='LAP loss weight in hybrid mode')
    parser.add_argument('--use_wandb', type=lambda x: x.lower() == 'true', default=False, help='Enable wandb logging')
    parser.add_argument('--lap_then_mlp', type=lambda x: x.lower() == 'true', default=False, help='Enable decoupled training: LAP first, then MLP')
    parser.add_argument('--lap_epochs', type=int, default=8, help='Number of epochs for LAP training in lap_then_mlp mode (50% more than original 5)')
    parser.add_argument('--lap_checkpoint', type=str, default=None, help='Path to LAP checkpoint to load for MLP-only training (skips LAP phase)')
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"

    # 确定训练模式
    if args.lap_then_mlp:
        TRAIN_MODE = 'lap_then_mlp'
    elif not args.use_lap and not args.use_mlp:
        TRAIN_MODE = 'baseline'
    elif not args.use_lap and args.use_mlp:
        TRAIN_MODE = 'mlp_only'
    elif args.use_lap and not args.use_mlp:
        TRAIN_MODE = 'lap_only'
    else:
        TRAIN_MODE = 'hybrid'

    # 输出目录
    if args.output_dir_is_final:
        # 使用传入的目录作为最终路径
        OUTPUT_DIR = args.output_dir
    else:
        # 默认：拼接子目录
        timestamp = datetime.now().strftime("%m.%d-%H%M")
        OUTPUT_DIR = os.path.join(args.output_dir, TRAIN_MODE, f"{timestamp}-{args.model_type}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("DeIR 混合检索器训练")
    print("="*60)
    print(f"训练模式: {TRAIN_MODE}")
    if TRAIN_MODE == 'lap_then_mlp':
        print(f"训练策略: 分阶段解耦训练 (LAP -> MLP)")
    else:
        print(f"use_lap: {args.use_lap}, use_mlp: {args.use_mlp}")
    print(f"GPU: {args.gpu}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)

    # 加载数据集
    print("\n加载数据...")

    if TRAIN_MODE == 'lap_only' or TRAIN_MODE == 'hybrid' or TRAIN_MODE == 'lap_then_mlp':
        # LAP/Hybrid/LAP_then_MLP 模式：使用缓存数据集（Q_minus 实时编码，文档从缓存加载）
        print(f"使用缓存数据集: {args.cache_path}")
        dataset = LAPDatasetWithCache(
            data_dir=args.data_dir,
            cache_path=args.cache_path,
            encoder_type=args.model_type,
            num_neg_per_query=args.num_neg
        )
    else:
        # baseline/mlp_only 模式：使用预计算缓存
        print(f"使用缓存数据: {args.cache_path}")
        dataset = DSCLRDataset(args.cache_path, num_neg=args.num_neg)

    # 按 Query 划分训练集和验证集（避免数据泄露）
    def split_by_query(dataset, val_ratio, seed=42):
        """按 Query 划分数据集，确保同一个 Query 只出现在训练集或验证集中
        
        Args:
            dataset: LAPDatasetWithCache 或 DSCLRDataset 实例
            val_ratio: 验证集比例
            seed: 随机种子
        Returns:
            train_indices, val_indices: 训练集和验证集的索引列表
        """
        import random
        random.seed(seed)
        
        # 获取所有唯一的 query idx
        all_indices = list(range(len(dataset)))
        query_to_indices = {}
        
        for idx in all_indices:
            sample = dataset[idx]
            # 从样本中获取 query idx
            if 'idx' in sample:
                query_idx = sample['idx']
            else:
                # 如果没有 idx 字段，假设每个样本就是一个 query
                query_idx = idx
            
            if query_idx not in query_to_indices:
                query_to_indices[query_idx] = []
            query_to_indices[query_idx].append(idx)
        
        # 获取所有唯一的 query
        unique_queries = list(query_to_indices.keys())
        num_val_queries = int(len(unique_queries) * val_ratio)
        
        # 随机选择验证集的 query
        val_queries = set(random.sample(unique_queries, num_val_queries))
        train_queries = set(unique_queries) - val_queries
        
        # 收集对应的样本索引
        train_indices = []
        val_indices = []
        
        for query_idx in train_queries:
            train_indices.extend(query_to_indices[query_idx])
        for query_idx in val_queries:
            val_indices.extend(query_to_indices[query_idx])
        
        # 排序以保持顺序一致
        train_indices.sort()
        val_indices.sort()
        
        return train_indices, val_indices
    
    # 使用按 Query 划分
    train_indices, val_indices = split_by_query(dataset, args.val_ratio, seed=args.seed)
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_size = len(train_indices)
    val_size = len(val_indices)
    total_size = len(dataset)

    # 选择 collate_fn
    if TRAIN_MODE == 'lap_only' or TRAIN_MODE == 'hybrid' or TRAIN_MODE == 'lap_then_mlp':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lap_cache_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lap_cache_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"总数据量: {total_size}, 训练集: {train_size}, 验证集: {val_size}")

    # Wandb 初始化
    wandb_logger = None
    if args.use_wandb:
        import wandb
        wandb_logger = wandb.init(
            project="DeIR-LAP-Training",
            name=f"{TRAIN_MODE}-{args.model_type}",
            config={
                "hidden_dim": args.embed_dim,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "margin_pos": args.lap_margin_pos,
                "margin_neg": args.lap_margin_neg,
                "encoder_type": args.model_type,
                "train_mode": TRAIN_MODE,
                "use_lap": args.use_lap,
                "use_mlp": args.use_mlp
            }
        )
        print(f"Wandb 日志已启用: {wandb_logger.project_name()}")

    # 加载编码器（仅 LAP/Hybrid 模式需要）
    encoder = None
    if TRAIN_MODE == 'lap_then_mlp':
        print("\n加载编码器...")
        encoder = load_encoder(args.model_type, DEVICE, args.encode_batch_size)
        print(f"编码器已加载: {args.model_type}")
    elif TRAIN_MODE == 'hybrid':
        print("\n加载编码器...")
        encoder = load_encoder(args.model_type, DEVICE, args.encode_batch_size)
        print(f"编码器已加载: {args.model_type}")
    elif TRAIN_MODE == 'lap_only':
        print("\n跳过编码器加载（使用缓存的嵌入）")

    # 创建模型
    print("\n初始化模型...")
    model = DeIR_HybridRetriever(
        encoder=encoder,
        hidden_dim=args.embed_dim,
        use_lap=True if TRAIN_MODE == 'lap_then_mlp' else args.use_lap,
        use_mlp=True if TRAIN_MODE == 'lap_then_mlp' else args.use_mlp,
        static_alpha=args.static_alpha,
        static_tau=args.static_tau,
        encoder_type=args.model_type,
        lap_rank=32  # 极限压缩：逼迫模型只学最核心泛化特征
    ).to(DEVICE)

    model.summary_params()

    # 损失函数（💥 自适应动态锚点版）
    criterion_lap = LAPContrastiveLoss(
        margin_push=0.15,
        margin_drop=0.10
    )

    # 优化器
    trainable_mode = {
        'baseline': 'none',
        'mlp_only': 'mlp',
        'lap_only': 'lap',
        'hybrid': 'all',
        'lap_then_mlp': 'lap'  # 第一阶段：只训练 LAP
    }.get(TRAIN_MODE, 'none')

    trainable_params = model.get_trainable_params(mode=trainable_mode)
    if trainable_params:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.1)
    else:
        optimizer = None
    
    lap_optimizer = None
    lap_scheduler = None
    if TRAIN_MODE == 'lap_then_mlp':
        lap_lr = args.lap_lr
        lap_optimizer = torch.optim.AdamW(model.lap.parameters(), lr=lap_lr, weight_decay=0.01)
        lap_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lap_optimizer, T_max=args.lap_epochs, eta_min=lap_lr * 0.1)
        print(f"\n📊 LAP 阶段学习率调度: CosineAnnealingLR")
        print(f"   初始 lr={lap_lr}, 最小 lr={lap_lr * 0.1}, T_max={args.lap_epochs}")

    # 训练历史
    history = {
        'train_loss': [],
        'train_loss_pos': [],
        'train_loss_neg': [],
        'val_loss': [],
        'sim_good': [],
        'sim_bad': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n开始训练...")

    if TRAIN_MODE == 'lap_then_mlp':
        optimizer_mlp = torch.optim.AdamW(model.mlp.parameters(), lr=args.lr, weight_decay=0.01)
        
        MLP_PHASE_DIR = os.path.join(OUTPUT_DIR, "mlp_phase")
        os.makedirs(MLP_PHASE_DIR, exist_ok=True)

        # 检查是否提供了 LAP checkpoint（跳过 LAP 阶段，直接开始 MLP 阶段）
        if args.lap_checkpoint and os.path.exists(args.lap_checkpoint):
            print("\n" + "="*60)
            print("跳过 LAP 阶段，直接开始 MLP 阶段")
            print("="*60)
            print(f"\n📥 加载指定 LAP 模型: {args.lap_checkpoint}")
            checkpoint = torch.load(args.lap_checkpoint, map_location=DEVICE)
            model.lap.load_state_dict(checkpoint['lap_state_dict'])
            print(f"   LAP 权重来自 Epoch {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            # 正常流程：先训练 LAP
            model.lap.proj.requires_grad_(True)
            model.mlp.proj.requires_grad_(False) if hasattr(model.mlp, 'proj') else None

            print("\n" + "="*60)
            print("阶段 1: 训练 LAP (冻结编码器)")
            print("="*60)

            LAP_PHASE_DIR = os.path.join(OUTPUT_DIR, "lap_phase")
            os.makedirs(LAP_PHASE_DIR, exist_ok=True)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(args.lap_epochs):
                current_lr = lap_scheduler.get_last_lr()[0] if lap_scheduler else args.lap_lr
                print(f"\n[LAP Phase] Epoch {epoch+1}/{args.lap_epochs} (lr={current_lr:.2e})")

                train_metrics = train_epoch_lap_with_mid_val(
                    train_loader, model, lap_optimizer, DEVICE, criterion_lap, encoder, args.encode_batch_size, val_loader, history, wandb_logger, LAP_PHASE_DIR
                )

                if lap_scheduler:
                    lap_scheduler.step()
                    new_lr = lap_scheduler.get_last_lr()[0]
                    print(f"   📉 学习率更新: {current_lr:.2e} → {new_lr:.2e}")

                train_loss = train_metrics['loss']
                train_loss_pos = train_metrics.get('loss_pos', 0)
                train_loss_neg = train_metrics.get('loss_neg', 0)

                print(f"[LAP Epoch {epoch+1}] Final Train Loss: {train_loss:.4f} (Push: {train_loss_pos:.4f}, Pull: {train_loss_neg:.4f})")

                recent_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
                if recent_val_loss < best_val_loss:
                    best_val_loss = recent_val_loss
                    patience_counter = 0
                    lap_model_path = os.path.join(LAP_PHASE_DIR, "deir_best_lap_phase.pt")
                    torch.save({
                        'epoch': epoch, 'val_loss': recent_val_loss,
                        'lap_state_dict': model.lap.state_dict()
                    }, lap_model_path)
                    print(f"✅ 保存 LAP 阶段最佳模型: {lap_model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"\n早停: LAP 阶段 {args.patience} 个 epoch 无改善")
                        break

            final_lap_path = os.path.join(LAP_PHASE_DIR, "deir_final_lap.pt")
            torch.save({'lap_state_dict': model.lap.state_dict()}, final_lap_path)
            print(f"✅ LAP 阶段训练完成，最终模型已保存: {final_lap_path}")

        print("\n" + "="*60)
        print("阶段 2: 训练 MLP (冻结 LAP)")
        print("="*60)

        # 如果指定了不同的 MLP 缓存路径，重新加载数据集
        if args.mlp_cache_path and args.mlp_cache_path != args.cache_path:
            print(f"\n🔄 为 MLP 阶段重新加载数据集...")
            print(f"   LAP 阶段缓存: {args.cache_path}")
            print(f"   MLP 阶段缓存: {args.mlp_cache_path}")
            
            mlp_dataset = LAPDatasetWithCache(
                data_dir=args.data_dir,
                cache_path=args.mlp_cache_path,
                encoder_type=args.model_type,
                num_neg_per_query=args.num_neg
            )
            
            # 使用相同的划分索引
            mlp_train_dataset = Subset(mlp_dataset, train_indices)
            mlp_val_dataset = Subset(mlp_dataset, val_indices)
            
            train_loader = DataLoader(mlp_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lap_cache_collate_fn)
            val_loader = DataLoader(mlp_val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lap_cache_collate_fn)
            
            print(f"   MLP 数据集加载完成: {len(mlp_train_dataset)} 训练样本, {len(mlp_val_dataset)} 验证样本")

        # 重置 history 以绘制 MLP 阶段独立的损失曲线
        print("\n🔄 重置损失历史，开始 MLP 阶段独立记录...")
        history = {
            'train_loss': [],
            'train_loss_pos': [],
            'train_loss_neg': [],
            'val_loss': []
        }

        model.lap.eval()
        for param in model.lap.parameters():
            param.requires_grad = False

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(args.epochs):
            print(f"\n[MLP Phase] Epoch {epoch+1}/{args.epochs}")

            train_metrics = train_epoch_mlp_with_cache_with_mid_val(train_loader, model, optimizer_mlp, DEVICE, encoder, args.encode_batch_size, val_loader, history, MLP_PHASE_DIR)

            train_loss = train_metrics['loss']
            print(f"[MLP Epoch {epoch+1}] Final Train Loss: {train_loss:.4f}")

            recent_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
            if recent_val_loss < best_val_loss:
                best_val_loss = recent_val_loss
                patience_counter = 0
                mlp_model_path = os.path.join(MLP_PHASE_DIR, "deir_best_mlp_phase.pt")
                torch.save({
                    'epoch': epoch, 'val_loss': recent_val_loss,
                    'lap_state_dict': model.lap.state_dict(),
                    'mlp_state_dict': model.mlp.state_dict()
                }, mlp_model_path)
                print(f"✅ 保存 MLP 阶段最佳模型: {mlp_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n早停: MLP 阶段 {args.patience} 个 epoch 无改善")
                    break

        final_mlp_path = os.path.join(MLP_PHASE_DIR, "deir_final_mlp.pt")
        torch.save({
            'lap_state_dict': model.lap.state_dict(),
            'mlp_state_dict': model.mlp.state_dict()
        }, final_mlp_path)
        print(f"✅ MLP 阶段训练完成，最终模型已保存: {final_mlp_path}")

    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
        print(f"\n📊 学习率调度: CosineAnnealingLR (lr={args.lr} → {args.lr * 0.1}, T_max={args.epochs})")

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            if TRAIN_MODE == 'baseline':
                train_metrics = train_epoch_baseline(train_loader, model, optimizer, DEVICE)
            elif TRAIN_MODE == 'mlp_only':
                train_metrics = train_epoch_mlp(train_loader, model, optimizer, DEVICE, args.model_type)
            elif TRAIN_MODE == 'lap_only':
                train_metrics = train_epoch_lap(train_loader, model, optimizer, DEVICE, criterion_lap, encoder, args.encode_batch_size, wandb_logger)
            else:
                train_metrics = train_epoch_hybrid(
                    train_loader, model, optimizer, DEVICE, args.model_type, criterion_lap, encoder, args.encode_batch_size, args.lap_loss_weight, wandb_logger
                )

            val_metrics = validate(model, val_loader, DEVICE, args.model_type, TRAIN_MODE, criterion_lap, encoder, args.encode_batch_size)

            scheduler.step()

            history['train_loss'].append(train_metrics['loss'])
            history['train_loss_pos'].append(train_metrics.get('loss_pos', 0))
            history['train_loss_neg'].append(train_metrics.get('loss_neg', 0))
            history['val_loss'].append(val_metrics['loss'])
            # 记录相似度（关键监控指标！）
            sim_good = train_metrics.get('sim_good_mean', 0)
            sim_bad = train_metrics.get('sim_bad_mean', 0)
            adaptive_target = train_metrics.get('adaptive_target', 0)
            history['sim_good'] = history.get('sim_good', [])
            history['sim_bad'] = history.get('sim_bad', [])
            history['adaptive_target'] = history.get('adaptive_target', [])  # 💥 动态目标
            history['sim_good'].append(sim_good)
            history['sim_bad'].append(sim_bad)
            history['adaptive_target'].append(adaptive_target)

            print(f"Train Loss: {train_metrics['loss']:.4f} (sim_good: {sim_good:.4f}, sim_bad: {sim_bad:.4f}), Val Loss: {val_metrics['loss']:.4f}")
            print(f"  → 💥 完全自适应: sim_good 目标={adaptive_target:.3f} (动态), sim_bad 紧贴原始底线")

            plot_realtime_loss(history, OUTPUT_DIR)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                model_path = os.path.join(OUTPUT_DIR, f"deir_best_{TRAIN_MODE}.pt")
                save_dict = {'epoch': epoch, 'val_loss': val_metrics['loss']}
                if model.mlp:
                    save_dict['mlp_state_dict'] = model.mlp.state_dict()
                if model.lap:
                    save_dict['lap_state_dict'] = model.lap.state_dict()
                torch.save(save_dict, model_path)
                print(f"✅ 保存最佳模型: {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n早停: {args.patience} 个 epoch 无改善")
                    break

    # 保存历史（仅在非 lap_then_mlp 模式下保存到父目录，避免重复）
    if TRAIN_MODE != 'lap_then_mlp':
        history_path = os.path.join(OUTPUT_DIR, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    # 绘制并保存损失曲线（仅在非 lap_then_mlp 模式下保存到父目录）
    if TRAIN_MODE != 'lap_then_mlp':
        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(history['train_loss']) + 1)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 左图：总损失对比
            axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Total Loss', linewidth=2, marker='o')
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(f'LAP Training - Total Loss ({TRAIN_MODE})', fontsize=14)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)

            # 右图：Push/Pull 损失分解
            if history['train_loss_pos']:
                axes[1].plot(epochs, history['train_loss_pos'], 'g-', label='Push Away (loss_pos)', linewidth=2, marker='^')
                axes[1].plot(epochs, history['train_loss_neg'], 'orange', label='Pull Closer (loss_neg)', linewidth=2, marker='v')
                axes[1].set_xlabel('Epoch', fontsize=12)
                axes[1].set_ylabel('Loss', fontsize=12)
                axes[1].set_title('LAP - Loss Decomposition', fontsize=14)
                axes[1].legend(fontsize=11)
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 损失曲线已保存: {loss_plot_path}")
        except Exception as e:
            print(f"⚠️  无法保存损失曲线: {e}")
    else:
        print(f"📊 损失曲线已保存在 lap_phase/ 和 mlp_phase/ 子目录中")

    print("\n" + "="*60)
    print("✅ 训练完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
