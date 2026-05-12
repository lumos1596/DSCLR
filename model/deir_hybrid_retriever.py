"""
DeIR_HybridRetriever: 解耦双开关终极检索架构

核心设计：
- LAP (Lightweight Asymmetric Projection): 作用在"表示层"，负责投影 Q- 空间，精准锁定目标
- MLP (DSCLR_MLP): 作用在"打分层"，负责预测 α/τ，动态调整惩罚力度

双开关控制：
- use_lap: 控制是否启用 LAP 空间投影
- use_mlp: 控制是否启用 MLP 动态预测

支持四种配置：
1. (F, F): 传统双塔 Baseline
2. (F, T): 纯 MLP 动态路由
3. (T, F): 纯 LAP 矩阵投影
4. (T, T): 终极完全体 Hybrid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.lap_module import LAPProjection
from model.dsclr_mlp import DSCLR_MLP


class DeIR_HybridRetriever(nn.Module):
    """
    DeIR 混合检索器

    通过双布尔开关独立控制 LAP 和 MLP 的启用状态

    Args:
        encoder: 底座编码器（冻结）
        hidden_dim: 嵌入维度
        use_lap: 是否启用 LAP 空间投影模块
        use_mlp: 是否启用 MLP 动态预测模块
        static_alpha: 静态 alpha（当 use_mlp=False 时使用）
        static_tau: 静态 tau（当 use_mlp=False 时使用）
        encoder_type: 编码器类型 ('bge', 'mistral', 'repllama')
    """

    def __init__(
        self,
        encoder,
        hidden_dim: int,
        use_lap: bool = False,
        use_mlp: bool = False,
        static_alpha: float = 1.0,
        static_tau: float = 0.5,
        encoder_type: str = 'bge',
        lap_rank: int = 256
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.use_lap = use_lap
        self.use_mlp = use_mlp
        self.encoder_type = encoder_type

        # 冻结底座编码器
        self._freeze_encoder()

        # ========== LAP 模块（表示层投影）==========
        # 任务：把 Q- 投影到"负向空间"，使其远离好文档、靠近踩雷文档
        if self.use_lap:
            self.lap = LAPProjection(hidden_dim=hidden_dim, rank=lap_rank)
            print(f"✅ LAP 模块已启用: hidden_dim={hidden_dim}, rank={lap_rank}")
        else:
            self.lap = None
            print(f"⚠️ LAP 模块已禁用")

        # ========== MLP 模块（打分层动态预测）==========
        # 任务：动态预测惩罚系数 (alpha, tau)
        if self.use_mlp:
            self.mlp = DSCLR_MLP(input_dim=hidden_dim, hidden_dim=256)
            print(f"✅ MLP 模块已启用")
        else:
            self.mlp = None
            self.static_alpha = static_alpha
            self.static_tau = static_tau
            print(f"⚠️ MLP 模块已禁用，使用静态参数: alpha={static_alpha}, tau={static_tau}")

        # 打印配置
        self._print_config()

    def _freeze_encoder(self):
        """冻结底座编码器所有参数"""
        if hasattr(self.encoder, 'parameters'):
            for param in self.encoder.parameters():
                param.requires_grad = False

        if hasattr(self.encoder, 'model'):
            for param in self.encoder.model.parameters():
                param.requires_grad = False

        print("🔒 底座编码器已冻结")

    def _print_config(self):
        """打印当前配置"""
        print("\n" + "=" * 50)
        print("DeIR_HybridRetriever 配置")
        print("=" * 50)
        print(f"  LAP 启用: {self.use_lap}")
        print(f"  MLP 启用: {self.use_mlp}")
        print(f"  编码器类型: {self.encoder_type}")
        print("=" * 50)

    def encode_texts(
        self,
        texts: list,
        batch_size: int = 32,
        mode: str = "document"
    ) -> torch.Tensor:
        """
        编码文本为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            mode: "query" 或 "document"

        Returns:
            embeddings: [N, hidden_dim] 已 L2 归一化
        """
        from eval.models.e5_mistral_encoder import E5MistralEncoder
        from eval.models.repllama_encoder import RepLLaMAEncoder

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            if isinstance(self.encoder, E5MistralEncoder):
                if mode == "query":
                    batch_emb = self.encoder.encode_queries(batch_texts, batch_size=batch_size)
                else:
                    batch_emb = self.encoder.encode_documents(batch_texts, batch_size=batch_size)
            elif isinstance(self.encoder, RepLLaMAEncoder):
                if mode == "query":
                    batch_emb = self.encoder.encode_queries(batch_texts, batch_size=batch_size)
                else:
                    batch_emb = self.encoder.encode_documents(batch_texts, batch_size=batch_size)
            else:
                from sentence_transformers import SentenceTransformer
                if isinstance(self.encoder, SentenceTransformer):
                    batch_emb = self.encoder.encode(
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

    def forward(
        self,
        q_minus_emb: torch.Tensor,
        q_pos_emb: Optional[torch.Tensor] = None,
        q_minus_raw: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        数据流：
        1. 提取底座特征（冻结）
        2. LAP 链路：重塑负向特征空间 (Where to hit)
        3. MLP 链路：动态预测惩罚力度 (How hard to hit)

        Args:
            q_minus_emb: Q- 嵌入向量 [batch_size, hidden_dim]
            q_pos_emb: Q+ 正向嵌入向量 [batch_size, hidden_dim]，用于 MLP 计算相关性
            q_minus_raw: 原始 Q- 用于 MLP 预测（如果与 q_minus_emb 不同）

        Returns:
            q_minus_proj: 投影后的 Q-（如果 use_lap=True 则经过 LAP，否则不变）
            alpha: 惩罚系数（标量或 [batch_size]）
            tau: 容忍阈值（标量或 [batch_size]）
            info: 额外信息 dict
        """
        info = {}

        # ========== Step 1: 底座特征提取（已在外层完成）==========

        # ========== Step 2: LAP 链路 - 表示层投影 ==========
        # 任务：把 Q- 投影到"负向空间"
        # 效果：让投影后的 Q- 更像"踩雷烂文"，远离"好文档"
        if self.use_lap and self.lap is not None:
            q_minus_proj = self.lap(q_minus_emb)
            info['lap_applied'] = True
        else:
            q_minus_proj = q_minus_emb
            info['lap_applied'] = False

        # ========== Step 3: MLP 链路 - 动态预测 ==========
        # 任务：预测 (alpha, tau)
        # alpha: 惩罚力度
        # tau: 容忍底线
        if self.use_mlp and self.mlp is not None:
            if q_pos_emb is None:
                raise ValueError("q_pos_emb is required when use_mlp=True")
            alpha, tau = self.mlp(q_pos_emb, q_minus_proj)
            info['mlp_applied'] = True
        else:
            alpha = torch.tensor(self.static_alpha, device=q_minus_emb.device)
            tau = torch.tensor(self.static_tau, device=q_minus_emb.device)
            info['mlp_applied'] = False

        return q_minus_proj, alpha, tau, info

    def get_trainable_params(self, mode: str = 'all'):
        """
        获取可训练参数

        Args:
            mode: 'all', 'lap', 'mlp', 'none'
                - 'all': 所有可训练参数（LAP + MLP）
                - 'lap': 仅 LAP 参数
                - 'mlp': 仅 MLP 参数
                - 'none': 无（全部冻结）

        Returns:
            param_groups for optimizer
        """
        params = []

        if mode == 'all':
            if self.use_lap and self.lap is not None:
                params.extend(list(self.lap.parameters()))
            if self.use_mlp and self.mlp is not None:
                params.extend(list(self.mlp.parameters()))

        elif mode == 'lap':
            if self.use_lap and self.lap is not None:
                params.extend(list(self.lap.parameters()))

        elif mode == 'mlp':
            if self.use_mlp and self.mlp is not None:
                params.extend(list(self.mlp.parameters()))

        elif mode == 'none':
            pass

        return params

    def summary_params(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("\n" + "=" * 50)
        print("参数统计")
        print("=" * 50)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params:,}")
        print(f"  冻结: {frozen_params:,}")
        print("=" * 50)

        if self.use_lap:
            lap_params = sum(p.numel() for p in self.lap.parameters())
            print(f"  LAP 参数: {lap_params:,}")

        if self.use_mlp:
            mlp_params = sum(p.numel() for p in self.mlp.parameters())
            print(f"  MLP 参数: {mlp_params:,}")

        print("=" * 50)


class DeIR_HybridRetrieverConfig:
    """
    DeIR_HybridRetriever 配置类

    方便创建预定义配置
    """

    @staticmethod
    def baseline():
        """配置 1: 传统双塔 Baseline"""
        return {
            'use_lap': False,
            'use_mlp': False,
            'static_alpha': 1.0,
            'static_tau': 0.5
        }

    @staticmethod
    def mlp_only():
        """配置 2: 纯 MLP 动态路由"""
        return {
            'use_lap': False,
            'use_mlp': True,
            'static_alpha': 1.0,
            'static_tau': 0.5
        }

    @staticmethod
    def lap_only():
        """配置 3: 纯 LAP 矩阵投影"""
        return {
            'use_lap': True,
            'use_mlp': False,
            'static_alpha': 1.0,
            'static_tau': 0.5
        }

    @staticmethod
    def hybrid():
        """配置 4: 终极完全体 Hybrid"""
        return {
            'use_lap': True,
            'use_mlp': True,
            'static_alpha': 1.0,
            'static_tau': 0.5
        }


if __name__ == "__main__":
    print("=" * 60)
    print("测试 DeIR_HybridRetriever 四种配置")
    print("=" * 60)

    # 模拟编码器
    class MockEncoder(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.dim = dim

        def encode(self, x):
            return F.normalize(torch.randn(len(x), self.dim), p=2, dim=-1)

    encoder = MockEncoder(dim=128)

    # 测试四种配置
    configs = [
        ("Baseline (F, F)", DeIR_HybridRetrieverConfig.baseline()),
        ("MLP Only (F, T)", DeIR_HybridRetrieverConfig.mlp_only()),
        ("LAP Only (T, F)", DeIR_HybridRetrieverConfig.lap_only()),
        ("Hybrid (T, T)", DeIR_HybridRetrieverConfig.hybrid()),
    ]

    for name, config in configs:
        print(f"\n{'=' * 50}")
        print(f"测试: {name}")
        print("=" * 50)

        model = DeIR_HybridRetriever(
            encoder=encoder,
            hidden_dim=128,
            **config
        )

        # 测试前向传播
        q_minus_emb = F.normalize(torch.randn(4, 128), p=2, dim=-1)
        q_minus_proj, alpha, tau, info = model(q_minus_emb)

        print(f"  q_minus_proj shape: {q_minus_proj.shape}")
        print(f"  alpha: {alpha}")
        print(f"  tau: {tau}")
        print(f"  info: {info}")

        model.summary_params()
