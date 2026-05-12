"""
LAP (Lightweight Asymmetric Projection) 逆向对比损失

终极版：自适应动态锚点 (Adaptive Self-Anchoring)

核心创新：
- 不再使用硬编码的 anchor_target=0.80
- 而是使用原始向量的相似度作为动态锚点
- 强迫投影后的分数不得低于原始分数

设计原则：
1. 相对对比拉伸：烂文必须比好文高出 margin
2. 动态正交下压：好文必须压到底噪之下
3. 💥 自适应锚点：投影后的烂文分数 ≥ 原始烂文分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPContrastiveLoss(nn.Module):
    """
    LAP 逆向对比损失函数 - 自适应动态锚点版
    
    核心创新：使用原始向量相似度作为动态锚点，而非硬编码阈值
    
    Args:
        margin_push: 相对对比边界，默认 0.15
        margin_drop: 正交下压边距，默认 0.10
    """
    
    def __init__(
        self,
        margin_push: float = 0.15,
        margin_drop: float = 0.10
    ):
        super().__init__()
        self.margin_push = margin_push
        self.margin_drop = margin_drop
    
    def forward(
        self,
        q_neg_proj: torch.Tensor,
        q_neg_raw: torch.Tensor,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor,
        dynamic_tau: float = 0.80
    ) -> tuple:
        """
        计算自适应锚点损失
        
        Args:
            q_neg_proj: LAP 投影后的负向查询向量 [batch_size, hidden_dim]
            q_neg_raw: 原始负向查询向量（未投影）[batch_size, hidden_dim]
            d_pos: 正样本文档向量 [batch_size, hidden_dim]
            d_neg: 负样本文档向量 [batch_size, hidden_dim] 或 [batch_size * num_neg, hidden_dim]
            dynamic_tau: 动态底噪水平（用于正交下压）
        
        Returns:
            total_loss: 总损失
            base_sim_bad_mean: 原始烂文相似度均值（用于监控）
        """
        # 处理多负样本情况
        batch_size = q_neg_proj.shape[0]
        
        if d_neg.dim() == 3:
            # [batch_size, num_neg, hidden_dim]
            num_neg = d_neg.shape[1]
            sim_good_proj = F.cosine_similarity(q_neg_proj, d_pos)
            sim_bad_proj = []
            base_sim_bad = []
            
            for i in range(num_neg):
                # 投影后的烂文分数
                sim_bad_proj_i = F.cosine_similarity(q_neg_proj, d_neg[:, i, :])
                sim_bad_proj.append(sim_bad_proj_i)
                
                # 🔐 原始烂文分数（截断梯度，作为靶标）
                with torch.no_grad():
                    base_sim_bad_i = F.cosine_similarity(q_neg_raw, d_neg[:, i, :]).detach()
                    base_sim_bad.append(base_sim_bad_i)
            
            sim_bad_proj = torch.stack(sim_bad_proj, dim=1)
            sim_good_proj = sim_good_proj.unsqueeze(1).expand_as(sim_bad_proj)
            base_sim_bad = torch.stack(base_sim_bad, dim=1)
            
        elif d_neg.shape[0] != batch_size:
            # [batch_size * num_neg, hidden_dim]
            num_neg = d_neg.shape[0] // batch_size
            d_neg = d_neg.view(batch_size, num_neg, -1)
            
            sim_good_proj = F.cosine_similarity(q_neg_proj, d_pos)
            sim_bad_proj = []
            base_sim_bad = []
            
            for i in range(num_neg):
                sim_bad_proj_i = F.cosine_similarity(q_neg_proj, d_neg[:, i, :])
                sim_bad_proj.append(sim_bad_proj_i)
                
                with torch.no_grad():
                    base_sim_bad_i = F.cosine_similarity(q_neg_raw, d_neg[:, i, :]).detach()
                    base_sim_bad.append(base_sim_bad_i)
            
            sim_bad_proj = torch.stack(sim_bad_proj, dim=1)
            sim_good_proj = sim_good_proj.unsqueeze(1).expand_as(sim_bad_proj)
            base_sim_bad = torch.stack(base_sim_bad, dim=1)
            
        else:
            # [batch_size, hidden_dim]
            sim_good_proj = F.cosine_similarity(q_neg_proj, d_pos)
            sim_bad_proj = F.cosine_similarity(q_neg_proj, d_neg)
            
            with torch.no_grad():
                base_sim_bad = F.cosine_similarity(q_neg_raw, d_neg).detach()
        
        # 💥 完全自适应目标计算
        with torch.no_grad():
            # 当前批次的原始烂文相似度均值（作为动态锚点）
            batch_base_bad_mean = base_sim_bad.mean()
            # sim_good 的动态目标 = 原始烂文分数 - margin_push - margin_drop
            # 这样好文必须比原始烂文低足够大的差距
            adaptive_good_target = batch_base_bad_mean - self.margin_push - self.margin_drop
        
        # 1. 相对对比拉伸：烂文必须比好文高出 margin_push
        loss_contrastive = torch.mean(F.relu(sim_good_proj - sim_bad_proj + self.margin_push))
        
        # 2. 💥 完全自适应正交下压：好文必须低于动态目标
        # 目标 = 原始烂文分数 - margin_push - margin_drop
        loss_orthogonal = torch.mean(F.relu(sim_good_proj - adaptive_good_target) ** 2)
        
        # 3. 💥 自适应锚点惩罚：投影后的烂文分数 ≥ 原始烂文分数
        loss_anchor = torch.mean(F.relu(base_sim_bad - sim_bad_proj) ** 2)
        
        # 严刑峻法：Anchor Loss 权重拉爆，守住烂文底线是第一优先级
        total_loss = 1.0 * loss_contrastive + 1.0 * loss_orthogonal + 5.0 * loss_anchor
        
        # 返回总损失和动态目标值（用于监控）
        adaptive_target_value = adaptive_good_target.item()
        
        return total_loss, adaptive_target_value
    
    def extra_repr(self) -> str:
        return (f"margin_push={self.margin_push}, "
                f"margin_drop={self.margin_drop}, "
                f"adaptive_anchor=True")
