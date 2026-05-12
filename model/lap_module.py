"""
LAP (Lightweight Asymmetric Projection) 模块 - Low-Rank 版本

核心功能：对负向查询向量进行低秩空间投影，解决维度诅咒导致的过拟合问题

设计原则（终极版）：
1. 残差低秩投影：将 4096→256→4096 的瓶颈结构，参数量从 16M 降至 2M
2. 零初始化升维矩阵：Epoch 0 时无损穿透，数学上等价于单位矩阵
3. Dropout 正则化：进一步防止过拟合
4. L2 归一化输出：确保余弦相似度空间一致性

架构创新：
- 借鉴 LoRA (Low-Rank Adaptation) 思想
- 残差连接：output = input + B(A(input))
- 信息瓶颈强迫网络学习通用特征而非死记硬背
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LAPProjection(nn.Module):
    """
    Lightweight Asymmetric Projection Module - Low-Rank Version
    
    对负向查询向量 (q_neg_emb) 进行低秩空间扭曲投影
    
    参数量对比：
    - 原版全连接: 4096 * 4096 = 16,777,216 参数
    - 低秩版本: 4096*256 + 256*4096 = 2,097,152 参数 (减少 87.5%)
    
    Args:
        hidden_dim: 输入向量维度 (如 4096 for RepLLaMA, 1024 for BGE)
        rank: 低秩瓶颈维度 (默认 256，可调整)
    """
    
    def __init__(self, hidden_dim: int = 4096, rank: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # 💥 残差低秩投影架构 (Residual Low-Rank Projection)
        # 1. 信息瓶颈：降维 (压缩掉死记硬背的冗余空间)
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        # 2. 信息瓶颈：升维 (只还原最核心的偏差特征)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        
        # Dropout 正则化（强力神经毒素：每次毒死30%神经元）
        self.dropout = nn.Dropout(p=0.3)
        
        # 💥 残差缩放 (Residual Scaling)
        # 默认 LAP 只对基座模型做 10% 的微调，保护原始空间流形
        self.res_scale = nn.Parameter(torch.tensor([0.1]))
        
        # 💥 架构师的魔法初始化 💥
        # 降维矩阵：使用小的正态分布初始化
        nn.init.normal_(self.down_proj.weight, std=0.02)
        # 升维矩阵：初始化为全 0
        # 这样在 Epoch 0 时，B(A(x)) = 0，残差连接保证 output = input
        # 数学上等价于单位矩阵，但更加优雅！
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, q_neg_emb: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        """
        前向传播：残差低秩投影 + L2归一化

        Args:
            q_neg_emb: 负向查询向量
                Shape: [batch_size, hidden_dim]
                要求: 输入应已 L2 归一化
            return_raw: 是否返回归一化前的原始投影向量

        Returns:
            q_neg_proj: 投影后的负向查询向量
                Shape: [batch_size, hidden_dim]
                保证: 输出已 L2 归一化
        """
        # 💥 残差连接 (Residual Connection)
        # 输出 = 原生向量 + 经过瓶颈提取的微小偏差
        # 因为初始 up_proj 为 0，所以初始状态 Out = q_neg_emb
        
        # 确保数据类型一致
        weight_dtype = self.down_proj.weight.dtype
        q_neg_emb = q_neg_emb.to(weight_dtype)
        
        delta = self.up_proj(self.down_proj(q_neg_emb))
        x_proj = q_neg_emb + self.res_scale * self.dropout(delta)
        
        if return_raw:
            return x_proj
        
        return F.normalize(x_proj, p=2, dim=-1)
    
    def get_effective_weight(self) -> torch.Tensor:
        """
        获取等效的投影矩阵权重（用于分析和可视化）
        
        等效权重 = I + B @ A (残差连接 + 低秩分解)
        """
        # 单位矩阵
        identity = torch.eye(self.hidden_dim, device=self.down_proj.weight.device)
        # 低秩扰动
        low_rank_delta = self.up_proj.weight @ self.down_proj.weight
        return identity + low_rank_delta
    
    def get_compression_ratio(self) -> float:
        """获取参数压缩比例"""
        original_params = self.hidden_dim ** 2
        low_rank_params = self.hidden_dim * self.rank + self.rank * self.hidden_dim
        return 1.0 - (low_rank_params / original_params)
    
    def extra_repr(self) -> str:
        original = self.hidden_dim ** 2
        low_rank = self.hidden_dim * self.rank + self.rank * self.hidden_dim
        compression = self.get_compression_ratio()
        return (f"hidden_dim={self.hidden_dim}, rank={self.rank}, "
                f"params={low_rank:,} (vs {original:,}), "
                f"compression={compression:.1%}")
