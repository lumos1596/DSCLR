"""
DSCLR 双流检索引擎
实现 Dual-Stream Contrastive Logical Reranking 的双流打分逻辑
支持静态超参数网格搜索 (Grid Search) 寻找最佳 alpha 和 tau
支持文档向量缓存，避免重复编码
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import random
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"


def get_model_cache_dir(base_cache_dir: str, model_name: str) -> str:
    """根据模型名称获取模型专属的缓存目录"""
    if "mistral" in model_name.lower():
        model_subdir = "e5-mistral-7b"
    elif "bge" in model_name.lower():
        model_subdir = "bge-large-en"
    elif "promptriever-llama3" in model_name.lower() or "promptriever_llama3" in model_name.lower():
        model_subdir = "promptriever_llama31_8b_instruct"
    else:
        model_subdir = model_name.split("/")[-1].replace("-", "_")
    
    return os.path.join(base_cache_dir, model_subdir)


def get_model_name_short(model_name: str) -> str:
    """从模型全名获取短名称用于缓存"""
    if "mistral" in model_name.lower():
        return "e5-mistral-7b"
    elif "bge" in model_name.lower():
        return "bge-large-en"
    elif "promptriever-llama3" in model_name.lower() or "promptriever_llama3" in model_name.lower():
        return "promptriever_llama31_8b_instruct"
    else:
        # 默认使用模型名称的最后一部分
        return model_name.split("/")[-1].replace("-", "_")


def load_cached_embeddings(
    cache_dir: str,
    task_name: str,
    model_name: str
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """尝试加载缓存的文档向量"""
    # 使用模型专属的缓存目录
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    model_name_short = get_model_name_short(model_name)
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

    if os.path.exists(cache_file) and os.path.exists(ids_file):
        logger.info(f"📂 加载缓存的文档向量: {cache_file}")
        
        # 尝试加载为 numpy 数组
        try:
            embeddings = np.load(cache_file)
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            logger.info(f"✅ 缓存加载成功: {len(doc_ids)} 个文档, shape={embeddings.shape}")
            return torch.tensor(embeddings), doc_ids
        except:
            # 可能是 dict 格式（E5-Mistral 保存的格式）
            try:
                data = np.load(cache_file, allow_pickle=True)
                if data.dtype == np.object_ and len(data.shape) == 0:
                    embedding_dict = data.item()
                    with open(ids_file, 'r') as f:
                        doc_ids = json.load(f)
                    
                    # 按 doc_ids 顺序提取 embeddings
                    embeddings_list = []
                    for doc_id in doc_ids:
                        if doc_id in embedding_dict:
                            embeddings_list.append(embedding_dict[doc_id])
                    
                    if embeddings_list:
                        embeddings = torch.stack(embeddings_list)
                        logger.info(f"✅ 缓存加载成功 (dict格式): {len(doc_ids)} 个文档, shape={embeddings.shape}")
                        return embeddings, doc_ids
            except Exception as e:
                logger.warning(f"⚠️ 缓存加载失败: {e}")

    logger.info(f"⚠️ 未找到缓存: {cache_file}")
    return None


def save_embeddings_cache(
    cache_dir: str,
    task_name: str,
    model_name: str,
    embeddings: torch.Tensor,
    doc_ids: List[str]
) -> None:
    """保存文档向量到缓存"""
    # 使用模型专属的缓存目录
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    os.makedirs(model_cache_dir, exist_ok=True)
    model_name_short = get_model_name_short(model_name)
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

    np.save(cache_file, embeddings.cpu().numpy())
    with open(ids_file, 'w') as f:
        json.dump(doc_ids, f)

    logger.info(f"💾 文档向量已缓存: {cache_file}")


class DSCLRDenseRetriever:
    """DSCLR 双流稠密检索器"""

    def __init__(
        self,
        encoder,
        device: str = "cuda",
        batch_size: int = 64
    ):
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.doc_ids: List[str] = []

    def index_documents(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        batch_size: Optional[int] = None
    ) -> None:
        """构建文档索引（带 L2 归一化）"""
        batch_size = batch_size or self.batch_size
        logger.info(f"📚 索引 {len(doc_ids)} 个文档...")

        embeddings = self.encoder.encode_documents(doc_texts, batch_size=batch_size)

        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        self.doc_embeddings = embeddings
        self.doc_ids = doc_ids
        logger.info(f"✅ 文档索引构建完成 (L2 归一化)")
        
        import torch as _torch
        _torch.cuda.empty_cache()

    def set_embeddings(
        self,
        embeddings: torch.Tensor,
        doc_ids: List[str]
    ) -> None:
        """直接设置已编码的文档向量"""
        logger.info(f"   [set_embeddings] 输入设备: {embeddings.device}, 目标设备: {self.device}")
        
        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 确保在正确的设备上
        embeddings = embeddings.to(self.device)
        logger.info(f"   [set_embeddings] 转移后设备: {embeddings.device}")
        
        self.doc_embeddings = embeddings
        self.doc_ids = doc_ids
        logger.info(f"✅ 文档向量已加载 (L2 归一化)")

    def compute_base_scores(
        self,
        q_plus_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算基础得分矩阵（仅 S_base，用于 OG 查询）
        返回: S_base
        """
        # 确保在同一设备上
        device = self.doc_embeddings.device
        if q_plus_embeddings.device != device:
            q_plus_embeddings = q_plus_embeddings.to(device)
        
        # S_base: [num_queries, num_docs]
        S_base = torch.matmul(q_plus_embeddings, self.doc_embeddings.T)
        return S_base

    def compute_scores_matrix(
        self,
        q_plus_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算得分矩阵（向量化，用于 Changed 查询）
        返回: (S_base, S_neg)
        """
        # 调试设备信息
        logger.info(f"   [compute_scores_matrix] q_plus: {q_plus_embeddings.device}, doc_emb: {self.doc_embeddings.device}")
        
        # 确保所有张量在同一设备上（使用 doc_embeddings 的设备作为目标）
        device = self.doc_embeddings.device
        if q_plus_embeddings.device != device:
            logger.warning(f"   设备不匹配！将查询 embeddings 转移到 {device}")
            q_plus_embeddings = q_plus_embeddings.to(device)
            q_minus_embeddings = q_minus_embeddings.to(device)
            neg_mask = neg_mask.to(device)
        
        # 文档已在索引时归一化，查询也已归一化
        # S_base: [num_queries, num_docs]
        S_base = torch.matmul(q_plus_embeddings, self.doc_embeddings.T)

        # S_neg: [num_queries, num_docs]
        S_neg = torch.matmul(q_minus_embeddings, self.doc_embeddings.T)

        # 应用 mask（将 [NONE] 的负向得分置零）
        S_neg = S_neg * neg_mask.unsqueeze(1)

        return S_base, S_neg

    def compute_dscrl_scores(
        self,
        S_base: torch.Tensor,
        S_neg: torch.Tensor,
        alpha: float,
        tau: float
    ) -> torch.Tensor:
        """
        计算 DSCLR 最终得分（静态版本）
        S_final = S_base - alpha * ReLU(S_neg - tau)
        """
        # ReLU 惩罚项
        penalty = torch.relu(S_neg - tau)

        # 最终得分
        S_final = S_base - alpha * penalty

        return S_final


class DSCLREvaluatorEngine:
    """DSCLR 评测引擎"""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        alphas: Optional[str] = None,
        taus: Optional[str] = None,
        num_samples: int = 15,
        sbase_mode: str = "original",
        confidence_beta: float = 0.0,
        gap_temperature: float = 0.0,
        max_penalty_ratio: float = 0.0,
        anchor_lambda: float = 0.0,
        anchor_top_k: int = 0,
        preserve_lambda: float = 0.0,
        preserve_top_k: int = 0
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.sbase_mode = sbase_mode
        self.confidence_beta = confidence_beta
        self.gap_temperature = gap_temperature
        self.max_penalty_ratio = max_penalty_ratio
        self.anchor_lambda = anchor_lambda
        self.anchor_top_k = anchor_top_k
        self.preserve_lambda = preserve_lambda
        self.preserve_top_k = preserve_top_k

        # 网格搜索参数空间（支持自定义）
        default_alphas = "0.0,0.5,1.0,2.0,3.0,5.0"
        default_taus = "0.5,0.6,0.7,0.8,0.9,0.95"

        alphas_list = [float(a.strip()) for a in (alphas or default_alphas).split(",")]
        taus_list = [float(t.strip()) for t in (taus or default_taus).split(",")]
        self.num_samples = num_samples

        all_combinations = [(a, t) for a in alphas_list for t in taus_list]
        self.param_combinations = random.sample(all_combinations, min(self.num_samples, len(all_combinations)))
        logger.info(f"🎲 随机抽取 {len(self.param_combinations)} 组参数: {self.param_combinations}")

        self._setup_seed()
        self._init_components()

    def _setup_seed(self) -> None:
        """设置随机种子"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

    def _init_components(self) -> None:
        """初始化各组件"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载编码器
        from eval.models import ModelFactory
        self.encoder = ModelFactory.create(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            max_seq_length=self.max_seq_length
        )

        # 初始化 Query Reformulator (LLM API 调用)
        from model.reformulator import QueryReformulator
        self.reformulator = QueryReformulator(
            task_name=self.task_name,
            use_cache=True,
            cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v5"
        )

        # 创建检索器
        self.retriever = DSCLRDenseRetriever(self.encoder, self.device, self.batch_size)

        # 加载数据
        from eval.engine import FollowIRDataLoader
        self.data_loader = FollowIRDataLoader(self.task_name)

        # 初始化 MLP (用于动态推理)
        self.mlp = None
        self.use_mlp = False
        
        # 初始化 LAP (用于 DeIR 模式)
        self.lap = None
        self.use_lap = False

        logger.info(f"✅ DSCLR 评测引擎初始化完成")
        logger.info(f"   模型: {self.model_name}")
        logger.info(f"   任务: {self.task_name}")
        logger.info(f"   查询重构: LLM API (实时解耦)")

    def compute_dscrl_scores_dynamic(
        self,
        S_base: torch.Tensor,
        S_neg: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor,
        q_plus_embeddings: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 DSCLR 最终得分（动态 MLP 版本）
        
        使用绝对值扣分，并通过 neg_mask 保护 [NONE] 查询
        
        Args:
            S_base: 基础得分 [num_queries, num_docs]
            S_neg: 负向得分 [num_queries, num_docs]
            q_minus_embeddings: 负向查询嵌入 [num_queries, embed_dim]
            neg_mask: 负向掩码 [num_queries]
            q_plus_embeddings: 正向查询嵌入 [num_queries, embed_dim] (V2 版本需要)
        """
        if self.mlp is None:
            raise RuntimeError("MLP model not loaded!")
        
        device = next(self.mlp.parameters()).device
        if q_minus_embeddings.device != device:
            q_minus_embeddings = q_minus_embeddings.to(device)
        
        q_minus_fp32 = q_minus_embeddings.float()
        
        model_lower = self.model_name.lower()
        if 'repllama' in model_lower:
            encoder_type = 'repllama'
        elif 'mistral' in model_lower:
            encoder_type = 'mistral'
        else:
            encoder_type = 'bge'
        
        # V2 版本使用双参数模式
        if self.mlp_v2 and q_plus_embeddings is not None:
            if q_plus_embeddings.device != device:
                q_plus_embeddings = q_plus_embeddings.to(device)
            q_plus_fp32 = q_plus_embeddings.float()
            alpha, tau = self.mlp(q_plus_fp32, q_minus_fp32, encoder_type=encoder_type)
        else:
            # V1 版本或没有 q_plus_embeddings 时使用单参数模式
            alpha, tau = self.mlp(q_minus_fp32, encoder_type=encoder_type)
        
        # 扩展维度用于计算
        alpha_expanded = alpha.unsqueeze(1)
        tau_expanded = tau.unsqueeze(1)
        neg_mask_expanded = neg_mask.unsqueeze(1)
        
        # 计算惩罚项（统一使用线性 ReLU）
        penalty = torch.relu(S_neg - tau_expanded)
        
        # 应用绝对值扣分，使用 neg_mask 保护 [NONE] 查询
        # 如果 neg_mask=0（即 [NONE]），则惩罚为 0，S_final = S_base
        S_final = S_base - alpha_expanded * penalty * neg_mask_expanded

        return S_final, alpha, tau

    def compute_dynamic_tau(
        self,
        q_original_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor,
        delta: float = 0.1
    ) -> torch.Tensor:
        """计算动态 τ_q = Noise_q + Delta
        
        Noise_q = Cosine(Q_rich, Q^-_pure)
        
        Args:
            q_original_embeddings: 原始查询向量 (Q_rich) [num_queries, hidden_dim]
            q_minus_embeddings: 负向查询向量 (Q^-_pure) [num_queries, hidden_dim]
            neg_mask: 负向词掩码 [num_queries]
            delta: 偏移量
            
        Returns:
            tau_q: 每个查询的动态阈值 [num_queries]
        """
        # 确保在同一设备上
        device = q_original_embeddings.device
        if q_minus_embeddings.device != device:
            q_minus_embeddings = q_minus_embeddings.to(device)
        if neg_mask.device != device:
            neg_mask = neg_mask.to(device)
        
        # 【绝对旁路网关】如果所有查询都是 [NONE]，直接返回零阈值
        if neg_mask.sum() == 0:
            return torch.zeros(len(q_original_embeddings), device=device)
        
        # 计算 Noise_q = Cosine(Q_rich, Q^-_pure)
        # 由于向量已归一化，点积即余弦相似度
        noise_q = torch.sum(q_original_embeddings * q_minus_embeddings, dim=-1)
        
        # 对于 [NONE] 的查询，噪声设为 0（这样 τ_q = 0 + delta，但惩罚会被 neg_mask 屏蔽）
        noise_q = noise_q * neg_mask
        
        # 计算动态 τ_q = Noise_q + Delta
        tau_q = noise_q + delta
        
        return tau_q

    def compute_dscrl_scores_dynamic_tau(
        self,
        S_base: torch.Tensor,
        S_neg: torch.Tensor,
        q_original_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor,
        alpha: float,
        delta: float,
        top_k: int = 0,
        confidence_beta: float = 0.0,
        gap_temperature: float = 0.0,
        max_penalty_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 DSCLR 最终得分（动态 τ 版本）
        
        S_final = S_base - min(alpha * max(0, S_neg - tau_q), S_base * max_ratio) * weight
        tau_q = Cosine(Q_rich, Q^-_pure) + delta
        
        三种加权/限制模式（可叠加）:
        1. confidence_beta > 0: confidence_weight = (1 - S_base_norm)^beta
        2. gap_temperature > 0: gap_weight = sigmoid((S_neg - S_base) * temperature)
        3. max_penalty_ratio > 0: 惩罚上限 = S_base * max_ratio
           确保相关文档（高S_base）不会被过度降权，允许使用更高α提升p-MRR
        
        Args:
            S_base: 基础得分 [num_queries, num_docs]
            S_neg: 负向得分 [num_queries, num_docs]
            q_original_embeddings: 原始查询向量 [num_queries, hidden_dim]
            q_minus_embeddings: 负向查询向量 [num_queries, hidden_dim]
            neg_mask: 负向词掩码 [num_queries]
            alpha: 惩罚力度
            delta: τ 的偏移量
            top_k: 只对 Top-K 文档应用惩罚（0 表示全部应用）
            confidence_beta: 置信度加权指数（0 表示不加权）
            gap_temperature: 差距加权温度（0 表示不加权）
            max_penalty_ratio: 惩罚上限比例（0 表示不限制，如0.3=最多降30%）
            
        Returns:
            S_final: 最终得分
            tau_q: 每个查询的动态阈值
        """
        device = S_base.device
        if S_neg.device != device:
            S_neg = S_neg.to(device)
        if q_original_embeddings.device != device:
            q_original_embeddings = q_original_embeddings.to(device)
        if q_minus_embeddings.device != device:
            q_minus_embeddings = q_minus_embeddings.to(device)
        if neg_mask.device != device:
            neg_mask = neg_mask.to(device)
        
        if neg_mask.sum() == 0:
            return S_base, torch.zeros(len(S_base), device=S_base.device)
        
        tau_q = self.compute_dynamic_tau(q_original_embeddings, q_minus_embeddings, neg_mask, delta)
        
        tau_q_expanded = tau_q.unsqueeze(1)
        neg_mask_expanded = neg_mask.unsqueeze(1)
        
        penalty = torch.relu(S_neg - tau_q_expanded)
        
        if confidence_beta > 0:
            s_min = S_base.min(dim=1, keepdim=True)[0]
            s_max = S_base.max(dim=1, keepdim=True)[0]
            s_range = s_max - s_min
            s_range = torch.where(s_range < 1e-8, torch.ones_like(s_range), s_range)
            S_base_norm = (S_base - s_min) / s_range
            confidence_weight = (1.0 - S_base_norm).pow(confidence_beta)
            penalty = penalty * confidence_weight
        
        if gap_temperature > 0:
            gap = S_neg - S_base
            gap_weight = torch.sigmoid(gap * gap_temperature)
            penalty = penalty * gap_weight
        
        raw_penalty = alpha * penalty
        
        if max_penalty_ratio > 0:
            max_penalty = torch.relu(S_base) * max_penalty_ratio
            raw_penalty = torch.minimum(raw_penalty, max_penalty)
        
        if top_k > 0 and top_k < S_base.shape[1]:
            _, top_k_indices = torch.topk(S_base, k=top_k, dim=1)
            penalty_mask = torch.zeros_like(S_base, dtype=torch.bool)
            penalty_mask.scatter_(1, top_k_indices, True)
            raw_penalty = raw_penalty * penalty_mask.float()
        
        S_final = S_base - raw_penalty * neg_mask_expanded
        
        return S_final, tau_q

    def _build_changed_og_anchor_scores(
        self,
        S_base_og: torch.Tensor,
        query_ids_og: List[str],
        query_ids_changed: List[str]
    ) -> torch.Tensor:
        """构建与 changed 查询对齐的 OG 锚点分数矩阵。"""
        og_row_by_base_qid = {}
        for idx, qid in enumerate(query_ids_og):
            base_qid = qid.replace('-og', '').replace('-changed', '')
            og_row_by_base_qid[base_qid] = idx

        aligned_rows = []
        for qid in query_ids_changed:
            base_qid = qid.replace('-og', '').replace('-changed', '')
            if base_qid in og_row_by_base_qid:
                aligned_rows.append(S_base_og[og_row_by_base_qid[base_qid]])
            else:
                # 兜底: 若找不到对应 OG 查询，则使用零向量，避免引入随机噪声
                aligned_rows.append(torch.zeros_like(S_base_og[0]))

        return torch.stack(aligned_rows, dim=0)

    def _apply_og_anchor(
        self,
        S_base_changed: torch.Tensor,
        S_anchor_changed: torch.Tensor,
        anchor_lambda: float,
        anchor_top_k: int
    ) -> torch.Tensor:
        """将 OG 锚点注入 changed 基础分数，提升排序稳定性。"""
        if anchor_lambda <= 0:
            return S_base_changed

        if anchor_top_k > 0 and anchor_top_k < S_anchor_changed.shape[1]:
            _, top_idx = torch.topk(S_anchor_changed, k=anchor_top_k, dim=1)
            anchor_mask = torch.zeros_like(S_anchor_changed, dtype=torch.bool)
            anchor_mask.scatter_(1, top_idx, True)
            anchored = S_base_changed + anchor_lambda * torch.relu(S_anchor_changed) * anchor_mask.float()
            return anchored

        # 全局融合: anchor_lambda 越大，越偏向 OG 排序
        return (1.0 - anchor_lambda) * S_base_changed + anchor_lambda * S_anchor_changed

    def _apply_map_preserve_boost(
        self,
        S_scores: torch.Tensor,
        S_anchor_changed: torch.Tensor,
        preserve_lambda: float,
        preserve_top_k: int
    ) -> torch.Tensor:
        """对 OG 排名靠前文档施加平滑保留增益，减少 MAP 深排坍塌。"""
        if preserve_lambda <= 0 or preserve_top_k <= 0:
            return S_scores

        k = min(preserve_top_k, S_anchor_changed.shape[1])
        if k <= 0:
            return S_scores

        _, top_idx = torch.topk(S_anchor_changed, k=k, dim=1)

        # 线性衰减权重: top1 权重最高，越靠后权重越低
        rank_weights = torch.linspace(1.0, 0.1, steps=k, device=S_scores.device).unsqueeze(0)
        boost = torch.zeros_like(S_scores)
        boost.scatter_(1, top_idx, rank_weights.expand(S_scores.shape[0], -1))

        return S_scores + preserve_lambda * boost

    def run(self, mlp_model_path: Optional[str] = None, mlp_hidden_dim: int = 256, lap_model_path: Optional[str] = None, use_dynamic_tau: bool = False, alphas_dynamic: Optional[List[float]] = None, deltas: Optional[List[float]] = None, top_k: int = 0) -> Dict[str, Any]:
        """运行 DSCLR 评测流程（含网格搜索、动态 MLP 或 DeIR 模式）
        
        Args:
            mlp_model_path: 如果提供，则使用动态 MLP 推理；否则使用网格搜索
            mlp_hidden_dim: MLP隐藏层维度 (默认: 256)
            lap_model_path: 如果提供，则使用 LAP 投影负向查询（DeIR 模式）
            use_dynamic_tau: 是否使用动态 τ 模式
            alphas_dynamic: 动态 τ 模式的 Alpha 范围
            deltas: 动态 τ 模式的 Delta 范围
            top_k: 只对 Top-K 文档应用惩罚（0 表示全部应用）
        """
        logger.info("=" * 60)
        logger.info("🚀 开始 DSCLR 评测")
        logger.info("=" * 60)

        start_time = time.time()

        # 根据模型名称确定嵌入维度
        if "mistral" in self.model_name.lower() or "repllama" in self.model_name.lower():
            embed_dim = 4096
            logger.info(f"   检测到 {self.model_name} 模型，使用嵌入维度: {embed_dim}")
        else:
            embed_dim = 1024
            logger.info(f"   使用默认嵌入维度: {embed_dim}")

        # 初始化 LAP (如果提供了模型路径)
        if lap_model_path:
            logger.info(f"🧠 加载 LAP 模型: {lap_model_path}")
            from model.lap_module import LAPProjection

            lap_checkpoint = torch.load(lap_model_path, map_location=self.device)

            # 兼容不同 checkpoint 格式，并从权重自动推断 lap_rank
            lap_state_dict = lap_checkpoint['lap_state_dict'] if 'lap_state_dict' in lap_checkpoint else lap_checkpoint
            if 'down_proj.weight' not in lap_state_dict:
                raise KeyError("LAP checkpoint 缺少 down_proj.weight，无法推断 rank")

            inferred_rank = lap_state_dict['down_proj.weight'].shape[0]
            self.lap = LAPProjection(hidden_dim=embed_dim, rank=inferred_rank).to(self.device)
            incompatible = self.lap.load_state_dict(lap_state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning(f"⚠️ LAP checkpoint 缺失键，已使用默认初始化: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"⚠️ LAP checkpoint 存在多余键，已忽略: {incompatible.unexpected_keys}")
            self.lap.eval()
            self.use_lap = True
            logger.info(f"✅ LAP 模型加载成功 (rank={inferred_rank})")
        else:
            self.use_lap = False

        # 初始化 MLP (如果提供了模型路径)
        if mlp_model_path:
            logger.info(f"🧠 加载动态 MLP 模型: {mlp_model_path}")
            mlp_checkpoint = torch.load(mlp_model_path, map_location=self.device)
            
            # 检测是否为 V2 版本
            use_concat_features = mlp_checkpoint.get('use_concat_features', False)
            if use_concat_features:
                logger.info("  检测到 V2 版本 MLP (使用拼接特征)")
                from model.dsclr_mlp_v2 import DSCLR_MLP_V2
                hidden_dim = mlp_checkpoint.get('hidden_dim', 512)
                self.mlp = DSCLR_MLP_V2(input_dim=embed_dim, hidden_dim=hidden_dim, use_concat_features=True).to(self.device)
            else:
                logger.info("  检测到 V1 版本 MLP")
                from model.dsclr_mlp import DSCLR_MLP
                self.mlp = DSCLR_MLP(input_dim=embed_dim, hidden_dim=mlp_hidden_dim).to(self.device)
            
            # 兼容不同的 checkpoint 格式
            if 'mlp_state_dict' in mlp_checkpoint:
                self.mlp.load_state_dict(mlp_checkpoint['mlp_state_dict'])
            elif 'model_state_dict' in mlp_checkpoint:
                self.mlp.load_state_dict(mlp_checkpoint['model_state_dict'])
            else:
                self.mlp.load_state_dict(mlp_checkpoint)
            self.mlp.eval()
            self.use_mlp = True
            self.mlp_v2 = use_concat_features
            logger.info(f"✅ MLP 模型加载成功，进入动态推理模式")
        else:
            self.use_mlp = False
            self.mlp_v2 = False
        
        # 输出当前模式
        if self.use_lap and self.use_mlp:
            logger.info("🔬 进入 DeIR 模式 (LAP + MLP)")
        elif self.use_mlp:
            logger.info("🔬 进入 DSCLR+MLP 动态模式")
        elif self.use_lap:
            logger.info("🔬 进入 DSCLR+LAP 模式 (使用 LAP 投影 + 网格搜索)")
        else:
            logger.info("🔬 进入 DSCLR 基础模式 (网格搜索)")

        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        
        # 加载原始 query 和 instruction (用于 reformulator)
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        # 编码/加载文档
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        
        # 尝试加载缓存
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        
        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
                
                # 直接使用缓存中的顺序，不再重排
                # 缓存中的顺序: cached_doc_ids[0] -> cached_embeddings[0]
                # _extract_results 需要使用相同的顺序
                ordered_embeddings = cached_embeddings
                ordered_doc_ids = cached_doc_ids
                
                self.retriever.set_embeddings(ordered_embeddings, ordered_doc_ids)
            else:
                logger.warning(f"⚠️ 缓存文档ID不匹配，重新编码...")
                doc_texts = [corpus[did]['text'] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
                save_embeddings_cache(self.cache_dir, self.task_name, self.model_name, self.retriever.doc_embeddings, self.retriever.doc_ids)
        else:
            # 无缓存，重新编码并保存
            logger.info("📚 编码候选文档...")
            doc_texts = [corpus[did]['text'] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
            if self.use_cache:
                save_embeddings_cache(self.cache_dir, self.task_name, self.model_name, self.retriever.doc_embeddings, self.retriever.doc_ids)

        # 构建 og 查询对 (仅提取 Q+，节省算力)
        logger.info("🔍 准备 og 查询 (仅 Q+)...")
        q_plus_list_og, query_ids_og = self._prepare_single_queries(q_og, q_raw_og)
        
        # 构建 changed 查询对 (使用 reformulator 实时解耦，需要 Q+ 和 Q-)
        logger.info("🔍 准备 changed 查询 (Q+ 和 Q-)...")
        q_plus_list_changed, q_minus_list_changed, q_original_list_changed, neg_mask_changed, query_ids_changed = self._prepare_dual_queries(q_changed, q_raw_changed)

        # 编码查询
        logger.info("🔍 编码 OG 查询 (仅 Q+)...")
        q_plus_embeddings_og = self._encode_queries(q_plus_list_og)

        logger.info("🔍 编码 Changed 查询 (Q+ 和 Q- 和原始查询)...")
        q_plus_embeddings_changed = self._encode_queries(q_plus_list_changed)
        q_minus_embeddings_changed = self._encode_queries(q_minus_list_changed)
        q_original_embeddings_changed = self._encode_queries(q_original_list_changed)

        # 构建 q_minus_map 用于白盒分析和坏例分析
        q_minus_map = {}
        for i, qid in enumerate(query_ids_changed):
            q_minus_map[qid] = q_minus_list_changed[i]

        # 根据 sbase_mode 统一计算 S_base
        if self.sbase_mode == "q_plus":
            logger.info("📊 S_base_mode=q_plus: 统一使用 Q+ 计算 S_base...")
            # OG 和 Changed 都使用 Q+
            S_base_og = self.retriever.compute_base_scores(q_plus_embeddings_og)
            S_base_changed = self.retriever.compute_base_scores(q_plus_embeddings_changed)
        else:
            logger.info("📊 S_base_mode=original: 统一使用原始查询计算 S_base...")
            # 需要为 og 查询编码原始查询
            q_original_list_og = [q for q in q_plus_list_og]  # OG 的原始查询就是 Q+
            q_original_embeddings_og = self._encode_queries(q_original_list_og)
            S_base_og = self.retriever.compute_base_scores(q_original_embeddings_og)
            S_base_changed = self.retriever.compute_base_scores(q_original_embeddings_changed)
        
        # 【LAP 投影】如果使用 LAP，对负向查询向量进行投影
        # 【关键修复】保存原始 Q^- 向量用于计算 τ，τ 必须用原始向量！
        q_minus_embeddings_changed_raw = q_minus_embeddings_changed.clone()
        
        if self.use_lap:
            logger.info("🔄 使用 LAP 投影负向查询...")
            with torch.no_grad():
                if q_minus_embeddings_changed.device != self.device:
                    q_minus_embeddings_changed = q_minus_embeddings_changed.to(self.device)
                q_minus_embeddings_changed = self.lap(q_minus_embeddings_changed)
        
        # 确保数据类型和设备与文档嵌入一致
        q_minus_embeddings_changed = q_minus_embeddings_changed.to(
            device=self.retriever.doc_embeddings.device,
            dtype=self.retriever.doc_embeddings.dtype
        )
        q_minus_embeddings_changed_raw = q_minus_embeddings_changed_raw.to(
            device=self.retriever.doc_embeddings.device,
            dtype=self.retriever.doc_embeddings.dtype
        )
        
        # 【绝对旁路网关】对于 [NONE] 查询，直接返回零矩阵，避免计算 "[NONE]" 与文档的无意义相似度
        # 这可以防止 "[NONE]" 这个词的高相似度（0.80+）污染统计指标
        if neg_mask_changed.sum() == 0:
            # 所有查询都是 [NONE]，直接返回零矩阵
            S_neg_changed = torch.zeros(
                len(query_ids_changed), 
                self.retriever.doc_embeddings.shape[0], 
                device=self.retriever.doc_embeddings.device
            )
        else:
            S_neg_changed = torch.matmul(q_minus_embeddings_changed, self.retriever.doc_embeddings.T)
            S_neg_changed = S_neg_changed * neg_mask_changed.unsqueeze(1)

        # 构建 doc_id -> 列索引 的映射 (用于白盒分析和结果提取)
        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}

        # 预构建 OG 锚点分数（与 changed 查询逐行对齐）
        S_anchor_changed = self._build_changed_og_anchor_scores(
            S_base_og=S_base_og,
            query_ids_og=query_ids_og,
            query_ids_changed=query_ids_changed
        )
        S_base_changed_anchored = self._apply_og_anchor(
            S_base_changed=S_base_changed,
            S_anchor_changed=S_anchor_changed,
            anchor_lambda=self.anchor_lambda,
            anchor_top_k=self.anchor_top_k
        )
        if self.anchor_lambda > 0:
            logger.info(
                f"🧷 启用 OG 锚点: anchor_lambda={self.anchor_lambda:.3f}, anchor_top_k={self.anchor_top_k}"
            )
        if self.preserve_lambda > 0 and self.preserve_top_k > 0:
            logger.info(
                f"🛡️ 启用 MAP-preserve: preserve_lambda={self.preserve_lambda:.3f}, preserve_top_k={self.preserve_top_k}"
            )

        # 动态推理 或 静态网格搜索
        if self.use_mlp:
            logger.info("🧠 使用动态 MLP 进行推理...")
            with torch.no_grad():
                # 【物理隔离】OG 查询直接使用 S_base，不经过任何 MLP 惩罚！
                S_final_og = S_base_og
                pred_alpha_og = torch.zeros(len(query_ids_og), device=self.device)
                pred_tau_og = torch.zeros(len(query_ids_og), device=self.device)
                
                # 只有 Changed 查询才进入动态门控计算
                S_final_changed, pred_alpha_changed, pred_tau_changed = self.compute_dscrl_scores_dynamic(
                    S_base_changed, S_neg_changed, q_minus_embeddings_changed, neg_mask_changed,
                    q_plus_embeddings=q_plus_embeddings_changed
                )
            
            logger.info(f"   OG 查询: 物理隔离，直接使用 S_base (无惩罚)")
            logger.info(f"   Changed 查询: 动态预测 avg_alpha={pred_alpha_changed.mean().item():.4f}, avg_tau={pred_tau_changed.mean().item():.4f}")
            
            # 提取检索结果
            results_og = self._extract_results(S_final_og, query_ids_og, candidates)
            results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)
            
            # 评测
            from eval.metrics import FollowIREvaluator
            evaluator = FollowIREvaluator(self.task_name)
            best_metrics = evaluator.evaluate(results_og, results_changed)
            best_params = {'alpha': 'dynamic', 'tau': 'dynamic'}
            
            # 计算单查询指标
            all_query_metrics = self._compute_per_query_metrics(
                results_og, results_changed, 
                query_ids_og, query_ids_changed, candidates
            )
            
            all_results = [{
                'mode': 'dynamic_mlp',
                'p-MRR': best_metrics.get('p-MRR', 0),
                'og_nDCG@5': best_metrics.get('original', {}).get('ndcg_at_5', 0),
                'changed_nDCG@5': best_metrics.get('changed', {}).get('ndcg_at_5', 0),
                'metrics': best_metrics
            }]
        elif use_dynamic_tau:
            # 动态 τ 网格搜索 (α × Δ)
            alphas_list = alphas_dynamic if alphas_dynamic else [1.0, 2.0, 3.0]
            deltas_list = deltas if deltas else [0.0, 0.05, 0.10, 0.15]
            
            logger.info(f"🔬 动态 τ 网格搜索: {len(alphas_list)} 个 α × {len(deltas_list)} 个 Δ = {len(alphas_list) * len(deltas_list)} 组参数")
            logger.info(f"   α 值: {alphas_list}")
            logger.info(f"   Δ 值: {deltas_list}")
            
            best_metrics = None
            best_params = None
            best_results_og = None
            best_results_changed = None
            all_results = []
            all_query_metrics = []

            for alpha in alphas_list:
                for delta in deltas_list:
                    # 【物理隔离】OG 查询直接使用 S_base，不经过任何惩罚！
                    S_final_og = S_base_og
                    
                    # 【固定使用 Q_original 计算 τ】
                    # τ = Cos(Q_original, Q^-) + delta
                    # 无论 sbase_mode 如何，τ 始终使用原始查询计算
                    tau_query_embeddings = q_original_embeddings_changed
                    
                    # 【关键修复】计算 τ 时必须使用原始的 Q^- 向量，而不是 LAP 投影后的向量！
                    # τ = Cosine(Q_rich, Q^-_raw) + delta
                    # 只有 S_neg 的计算才使用 LAP 投影后的向量
                    S_final_changed, tau_q = self.compute_dscrl_scores_dynamic_tau(
                        S_base_changed_anchored, S_neg_changed,
                        tau_query_embeddings, q_minus_embeddings_changed_raw,
                        neg_mask_changed, alpha, delta, top_k,
                        confidence_beta=self.confidence_beta,
                        gap_temperature=self.gap_temperature,
                        max_penalty_ratio=self.max_penalty_ratio
                    )
                    S_final_changed = self._apply_map_preserve_boost(
                        S_scores=S_final_changed,
                        S_anchor_changed=S_anchor_changed,
                        preserve_lambda=self.preserve_lambda,
                        preserve_top_k=self.preserve_top_k
                    )

                    # 提取检索结果
                    results_og = self._extract_results(S_final_og, query_ids_og, candidates)
                    results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)

                    # 评测 - 使用正确的 og 和 changed 结果
                    from eval.metrics import FollowIREvaluator
                    evaluator = FollowIREvaluator(self.task_name)
                    metrics = evaluator.evaluate(results_og, results_changed)

                    p_mrr = metrics.get('p-MRR', 0)
                    og_ndcg = metrics.get('original', {}).get('ndcg_at_5', 0)
                    changed_ndcg = metrics.get('changed', {}).get('ndcg_at_5', 0)
                    
                    # 计算平均 tau
                    avg_tau = tau_q.mean().item()
                    logger.info(f"   α={alpha:.1f}, Δ={delta:.2f} (avg_τ={avg_tau:.3f}) => p-MRR={p_mrr:.4f}, og_nDCG@5={og_ndcg:.4f}, changed_nDCG@5={changed_ndcg:.4f}")

                    result_entry = {
                        'alpha': alpha,
                        'delta': delta,
                        'avg_tau': avg_tau,
                        'p-MRR': p_mrr,
                        'og_nDCG@5': og_ndcg,
                        'changed_nDCG@5': changed_ndcg,
                        'metrics': metrics,
                        'results_changed': {qid: list(scores.items()) for qid, scores in results_changed.items()}
                        # 注意：不保存 S_base_changed, S_neg_changed, tau_q 等大矩阵，避免文件过大
                    }
                    
                    all_results.append(result_entry)

                    # 选择最佳参数
                    if best_metrics is None:
                        best_metrics = metrics
                        best_params = (alpha, delta)
                        best_composite_score = p_mrr + og_ndcg + changed_ndcg
                        best_results_og = results_og
                        best_results_changed = results_changed
                        # 保存最佳参数的中间结果用于白盒分析
                        best_S_base_og = S_base_og
                        best_S_base_changed = S_base_changed_anchored
                        best_S_neg_changed = S_neg_changed
                        best_S_final_changed = S_final_changed
                        best_tau_q = tau_q
                    else:
                        current_composite_score = p_mrr + og_ndcg + changed_ndcg
                        best_composite_score = best_metrics.get('p-MRR', 0) + best_metrics.get('original', {}).get('ndcg_at_5', 0) + best_metrics.get('changed', {}).get('ndcg_at_5', 0)
                        if current_composite_score > best_composite_score:
                            best_metrics = metrics
                            best_params = (alpha, delta)
                            best_composite_score = current_composite_score
                            best_results_og = results_og
                            best_results_changed = results_changed
                            # 保存最佳参数的中间结果用于白盒分析
                            best_S_base_og = S_base_og
                            best_S_base_changed = S_base_changed_anchored
                            best_S_neg_changed = S_neg_changed
                            best_S_final_changed = S_final_changed
                            best_tau_q = tau_q
            
            # 为每个查询计算详细的性能指标
            all_query_metrics = self._compute_per_query_metrics(
                best_results_og, best_results_changed, 
                query_ids_og, query_ids_changed, candidates
            )
            
            # 【白盒分析】对最佳参数执行漏网烂文追踪
            if best_params is not None:
                alpha, delta = best_params
                logger.info(f"🔍 执行白盒漏网烂文追踪分析 (最佳参数: α={alpha}, Δ={delta})...")
                from eval.metrics.evaluator import DataLoader
                data_loader = DataLoader(self.task_name)
                qrels = data_loader.load_qrels()
                
                whitebox_analysis = self._whitebox_survivor_analysis(
                    target_queries=['310-changed', '419-changed'],
                    alpha=alpha,
                    delta=delta,
                    S_base_og=best_S_base_og,
                    S_base_changed=best_S_base_changed,
                    S_neg_changed=best_S_neg_changed,
                    S_final_changed=best_S_final_changed,
                    tau_q=best_tau_q,
                    query_ids_changed=query_ids_changed,
                    candidates=candidates,
                    qrels=qrels,
                    q_minus_map=q_minus_map,
                    doc_id_to_col_idx=doc_id_to_col_idx,
                    q_raw_changed=q_raw_changed
                )
                # 单独保存白盒分析结果为 Markdown 文件
                self._save_whitebox_report(whitebox_analysis, alpha, delta)
                logger.info(f"✅ 白盒分析完成: {len(whitebox_analysis)} 个查询")
            
        else:
            # 静态网格搜索
            logger.info(f"🔬 随机搜索: {len(self.param_combinations)} 组参数")
            best_metrics = None
            best_params = None
            best_results_og = None
            best_results_changed = None
            all_results = []
            all_query_metrics = []

            for alpha, tau in self.param_combinations:
                    # 【物理隔离】OG 查询直接使用 S_base，不经过任何惩罚！
                    S_final_og = S_base_og
                    # 只有 Changed 查询才应用 DSCLR 惩罚
                    S_final_changed = self.retriever.compute_dscrl_scores(S_base_changed_anchored, S_neg_changed, alpha, tau)
                    S_final_changed = self._apply_map_preserve_boost(
                        S_scores=S_final_changed,
                        S_anchor_changed=S_anchor_changed,
                        preserve_lambda=self.preserve_lambda,
                        preserve_top_k=self.preserve_top_k
                    )

                    # 提取检索结果
                    results_og = self._extract_results(S_final_og, query_ids_og, candidates)
                    results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)

                    # 评测 - 使用正确的 og 和 changed 结果
                    from eval.metrics import FollowIREvaluator
                    evaluator = FollowIREvaluator(self.task_name)
                    metrics = evaluator.evaluate(results_og, results_changed)

                    p_mrr = metrics.get('p-MRR', 0)
                    og_ndcg = metrics.get('original', {}).get('ndcg_at_5', 0)
                    changed_ndcg = metrics.get('changed', {}).get('ndcg_at_5', 0)
                    logger.info(f"   α={alpha:.1f}, τ={tau:.2f} => p-MRR={p_mrr:.4f}, og_nDCG@5={og_ndcg:.4f}, changed_nDCG@5={changed_ndcg:.4f}")

                    all_results.append({
                        'alpha': alpha,
                        'tau': tau,
                        'p-MRR': p_mrr,
                        'og_nDCG@5': og_ndcg,
                        'changed_nDCG@5': changed_ndcg,
                        'metrics': metrics,
                        'results_changed': {qid: list(scores.items()) for qid, scores in results_changed.items()},
                        'S_base_changed': S_base_changed_anchored.cpu().numpy().tolist() if S_base_changed_anchored is not None else None,
                        'S_neg_changed': S_neg_changed.cpu().numpy().tolist() if S_neg_changed is not None else None
                    })

                    # 选择最佳参数：综合考虑 p-MRR、og_nDCG 和 changed_nDCG
                    if best_metrics is None:
                        best_metrics = metrics
                        best_params = (alpha, tau)
                        best_composite_score = p_mrr + changed_ndcg
                        best_results_og = results_og
                        best_results_changed = results_changed
                    else:
                        current_composite_score = p_mrr + changed_ndcg
                        best_composite_score = best_metrics.get('p-MRR', 0) + best_metrics.get('original', {}).get('ndcg_at_5', 0) + best_metrics.get('changed', {}).get('ndcg_at_5', 0)
                        if current_composite_score > best_composite_score:
                            best_metrics = metrics
                            best_params = (alpha, tau)
                            best_composite_score = current_composite_score
                            best_results_og = results_og
                            best_results_changed = results_changed
            
            # 为每个查询计算详细的性能指标
            all_query_metrics = self._compute_per_query_metrics(
                best_results_og, best_results_changed, 
                query_ids_og, query_ids_changed, candidates
            )

        elapsed_time = time.time() - start_time

        # 输出结果
        logger.info("=" * 60)
        if self.use_mlp:
            logger.info("🧠 DSCLR 动态 MLP 推理结果:")
        elif use_dynamic_tau:
            logger.info("📊 DSCLR 动态 τ 搜索结果:")
            logger.info(f"   最佳参数: α={best_params[0]}, Δ={best_params[1]}")
        else:
            logger.info("📊 DSCLR 随机搜索结果:")
            logger.info(f"   最佳参数: α={best_params[0]}, τ={best_params[1]}")
        og_metrics = best_metrics.get('original', {})
        changed_metrics = best_metrics.get('changed', {})
        logger.info(f"   p-MRR: {best_metrics.get('p-MRR', 0):.4f}")
        logger.info(f"   OG - nDCG@1: {og_metrics.get('ndcg_at_1', 0):.4f}, nDCG@5: {og_metrics.get('ndcg_at_5', 0):.4f}, nDCG@10: {og_metrics.get('ndcg_at_10', 0):.4f}")
        logger.info(f"   Changed - nDCG@1: {changed_metrics.get('ndcg_at_1', 0):.4f}, nDCG@5: {changed_metrics.get('ndcg_at_5', 0):.4f}, nDCG@10: {changed_metrics.get('ndcg_at_10', 0):.4f}")
        logger.info(f"   耗时: {elapsed_time:.1f}秒")
        logger.info("=" * 60)

        # 保存结构化汇总文件
        self._save_structured_summary(best_metrics, all_query_metrics, q_raw_og, q_raw_changed, best_params)
        
        # 保存所有参数的简化汇总
        self._save_all_params_summary(all_results)

        # 生成坏例分析报告
        # 将 corpus 合并到 candidates 中，方便后续坏例分析使用文档文本
        candidates_with_text = {}
        for qid, doc_ids in candidates.items():
            for doc_id in doc_ids:
                if doc_id not in candidates_with_text:
                    candidates_with_text[doc_id] = corpus.get(doc_id, {'text': ''})
        
        # 构建 q_minus_map 用于坏例分析
        q_minus_map = {}
        for i, qid in enumerate(query_ids_changed):
            q_minus_map[qid] = q_minus_list_changed[i]
        
        self._generate_bad_case_analysis(
            all_query_metrics, q_raw_og, q_raw_changed, 
            candidates_with_text, all_results, q_minus_map
        )

        # 保存 TREC 格式文件
        trec_dir = os.path.join(self.output_dir, "trec")
        os.makedirs(trec_dir, exist_ok=True)
        
        run_og_path = os.path.join(trec_dir, f"run_{self.task_name}_og.trec")
        run_changed_path = os.path.join(trec_dir, f"run_{self.task_name}_changed.trec")
        
        # 保存 TREC 格式文件
        self._save_trec_format(results_og, run_og_path)
        self._save_trec_format(results_changed, run_changed_path)
        
        logger.info(f"💾 TREC 文件已保存:")
        logger.info(f"   OG: {run_og_path}")
        logger.info(f"   Changed: {run_changed_path}")

        # 保存结果
        if self.use_mlp:
            # MLP 模式：只保存测试指标结果
            result_path = os.path.join(self.output_dir, "mlp_results.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'dynamic_mlp',
                    'metrics': best_metrics
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 MLP 测试结果已保存: {result_path}")
        else:
            # 静态网格搜索模式：保存完整结果
            self._save_results(all_results, best_params, best_metrics, all_query_metrics)

        return {
            'best_params': {'alpha': 'dynamic', 'tau': 'dynamic'} if self.use_mlp else {'alpha': best_params[0], 'tau': best_params[1]},
            'best_metrics': best_metrics
        }

    def _get_all_candidate_doc_ids(self, candidates: Dict[str, List[str]]) -> List[str]:
        """获取所有候选文档ID"""
        all_doc_ids_set = set()
        for doc_ids in candidates.values():
            all_doc_ids_set.update(doc_ids)
        return list(all_doc_ids_set)

    def _get_bulletproof_mask(self, q_minus_text: str) -> float:
        """防弹级掩码生成函数
        
        拦截所有可能的 LLM 废话输出，确保 [NONE] 查询不受惩罚
        """
        if not q_minus_text:
            return 0.0
        text = str(q_minus_text).strip().upper()
        # 拦截所有可能的无效输出
        if text in ["[NONE]", "NONE", "NULL", "N/A", "", "[NONE]", "NONE"]:
            return 0.0
        return 1.0

    def _prepare_single_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """
        准备单流查询（仅 Q+）- 用于 OG 查询，节省算力
        返回: (q_plus_list, query_ids)
        """
        query_ids = []
        q_plus_list = []

        for qid in queries.keys():
            # 获取原始 query 和 instruction
            raw = raw_queries.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            
            # OG 查询使用 query + instruction 拼接（与基线评估一致）
            q_plus = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")

            query_ids.append(qid)
            q_plus_list.append(q_plus)

        logger.info(f"   单流查询准备完成: {len(query_ids)} 个")
        return q_plus_list, query_ids

    def _prepare_dual_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str], List[str], torch.Tensor, List[str]]:
        """
        准备双流查询 - 使用 reformulator 实时解耦
        返回: (q_plus_list, q_minus_list, q_original_list, neg_mask, query_ids)
        其中 q_original_list 是原始 query + instruction，用于计算 S_base
        """
        query_ids = []
        q_plus_list = []
        q_minus_list = []
        q_original_list = []

        for qid in queries.keys():
            # 获取原始 query 和 instruction
            raw = raw_queries.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            
            # 原始查询 = query + instruction（用于 S_base 计算）
            q_original = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")

            # 从 qid 提取 idx (格式: "1-og" -> idx=1)
            try:
                idx = int(qid.split('-')[0])
            except:
                idx = 0
            
            # 确定 query_type
            query_type = "og" if qid.endswith("-og") else "changed"
            
            # 使用 reformulator 进行实时解耦 (带缓存)
            q_plus, q_minus = self.reformulator.reformulate(
                qid=qid,
                idx=idx,
                query=query_text,
                instruction=instruction,
                query_type=query_type
            )

            query_ids.append(qid)
            q_plus_list.append(q_plus)
            q_minus_list.append(q_minus)
            q_original_list.append(q_original)

        # 【调试逻辑】强制 OG 查询跳过 MLP，即使 Q- 不为 None
        # 这样可以让 OG nDCG 与原始检索模型对比，验证掩码逻辑
        def get_debug_mask(qm, qid):
            base_mask = self._get_bulletproof_mask(qm)
            if qid.endswith("-og"):
                # 强制 OG 查询 mask = 0，跳过 MLP
                return 0.0
            return base_mask
        
        neg_mask = torch.tensor(
            [get_debug_mask(qm, qid) for qm, qid in zip(q_minus_list, query_ids)],
            dtype=torch.float32,
            device=self.device
        )
        
        # 统计调试信息
        og_count = sum(1 for qid in query_ids if qid.endswith("-og"))
        changed_count = len(query_ids) - og_count
        logger.info(f"【调试模式】OG 查询强制跳过 MLP: {og_count} 个, Changed 查询: {changed_count} 个")

        return q_plus_list, q_minus_list, q_original_list, neg_mask, query_ids

    def _encode_queries(self, texts: List[str]) -> torch.Tensor:
        """编码查询（带 L2 归一化）"""
        embeddings = self.encoder.encode_queries(texts, self.batch_size)

        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _extract_results(
        self,
        S_final: torch.Tensor,
        query_ids: List[str],
        candidates: Dict[str, List[str]],
        top_k: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """从得分矩阵提取检索结果
        
        Args:
            S_final: 得分矩阵
            query_ids: 查询 ID 列表
            candidates: 候选文档字典
            top_k: 返回的文档数量（默认 1000，与原始评估一致）
        """
        results = {}
        
        # 构建 doc_id -> 列索引 的映射
        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        
        for idx, qid in enumerate(query_ids):
            base_qid = qid.replace('-og', '').replace('-changed', '')

            if base_qid not in candidates or not candidates[base_qid]:
                continue

            doc_ids = candidates[base_qid]

            # 获取该查询对应的得分 (转换为 float32，避免 BFloat16 问题)
            scores = S_final[idx].cpu().float().numpy()

            # 使用 doc_id_to_col_idx 找到正确的列索引
            doc_scores = {}
            for doc_id in doc_ids:
                if doc_id in doc_id_to_col_idx:
                    col_idx = doc_id_to_col_idx[doc_id]
                    doc_scores[doc_id] = float(scores[col_idx])

            # 取 top-k（如果 top_k <= 0，则返回所有文档）
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            if top_k > 0:
                results[qid] = dict(sorted_docs[:top_k])
            else:
                results[qid] = dict(sorted_docs)

        return results

    def _whitebox_survivor_analysis(
        self,
        target_queries: List[str],
        alpha: float,
        delta: float,
        S_base_og: torch.Tensor,
        S_base_changed: torch.Tensor,
        S_neg_changed: torch.Tensor,
        S_final_changed: torch.Tensor,
        tau_q: torch.Tensor,
        query_ids_changed: List[str],
        candidates: Dict[str, Any],
        qrels: Dict[str, Dict[str, int]],
        q_minus_map: Dict[str, str],
        doc_id_to_col_idx: Dict[str, int],
        q_raw_changed: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        白盒漏网烂文追踪分析
        
        追踪为什么部分烂文能免疫惩罚（导致 pMRR 低）
        同时对比 OG 模式和 Changed 模式下相关文档的排名
        """
        from eval.metrics.evaluator import DataLoader
        
        data_loader = DataLoader(self.task_name)
        corpus = data_loader.load_corpus()
        
        analysis_results = {}
        
        for qid in target_queries:
            if qid not in query_ids_changed:
                continue
                
            idx = query_ids_changed.index(qid)
            changed_key = qid
            
            q_minus = q_minus_map.get(changed_key, "[NONE]")
            if q_minus == "[NONE]" or not q_minus:
                continue
            
            neg_words = [w.strip().lower() for w in q_minus.split(',') if w.strip()]
            
            # Changed 模式的得分
            s_base_changed_query = S_base_changed[idx].cpu().float().numpy()
            s_neg_query = S_neg_changed[idx].cpu().float().numpy() if S_neg_changed is not None else np.zeros_like(s_base_changed_query)
            s_final_changed_query = S_final_changed[idx].cpu().float().numpy()
            tau_value = tau_q[idx].item()
            
            # OG 模式的得分 (使用相同的 query index)
            s_base_og_query = S_base_og[idx].cpu().float().numpy()
            
            # 获取该查询的 qrels - 分别获取 OG 和 Changed 模式
            changed_key = qid  # "310-changed"
            og_key = qid.replace('-changed', '-og')  # "310-og"
            
            query_qrels_changed = qrels.get(changed_key, {})
            query_qrels_og = qrels.get(og_key, {})
            
            # 获取该查询的候选文档列表
            # 注意：candidates 中的 key 是原始数字（如 "310"、"419"），不是 "310-changed"
            base_qid = qid.split('-')[0] if '-' in qid else qid
            query_candidates = candidates.get(base_qid, [])
            
            # 只处理当前查询的候选文档
            if not query_candidates:
                logger.warning(f"⚠️ {qid} 没有候选文档，跳过白盒分析")
                continue
            
            # 收集候选文档的详细信息
            doc_details = []
            candidate_scores_changed = []
            candidate_scores_og = []
            for doc_id in query_candidates:
                if doc_id not in doc_id_to_col_idx:
                    continue
                col_idx = doc_id_to_col_idx[doc_id]
                
                doc = corpus.get(doc_id, {})
                doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                
                contains_neg = any(neg_word in doc_text.lower() for neg_word in neg_words)
                
                # Changed 模式得分
                s_pos_changed = float(s_base_changed_query[col_idx])
                s_neg_proj = float(s_neg_query[col_idx])
                s_final_changed = float(s_final_changed_query[col_idx])
                
                # OG 模式得分
                s_pos_og = float(s_base_og_query[col_idx])
                
                over_threshold = s_neg_proj - tau_value
                actual_penalty = alpha * max(0, over_threshold)
                
                # 分别获取 OG 和 Changed 模式下的相关性标签
                relevance_score_changed = query_qrels_changed.get(doc_id, 0)
                relevance_score_og = query_qrels_og.get(doc_id, 0)
                
                candidate_scores_changed.append(s_final_changed)
                candidate_scores_og.append(s_pos_og)
                
                doc_details.append({
                    'doc_id': doc_id,
                    'relevance_score_changed': relevance_score_changed,
                    'relevance_score_og': relevance_score_og,
                    'is_relevant_changed': relevance_score_changed > 0,
                    'is_relevant_og': relevance_score_og > 0,
                    'contains_neg': contains_neg,
                    'rank_changed': 0,
                    'rank_og': 0,
                    'S_pos_changed': round(s_pos_changed, 4),
                    'S_pos_og': round(s_pos_og, 4),
                    'S_neg_proj': round(s_neg_proj, 4),
                    'Dynamic_Tau': round(tau_value, 4),
                    'Over_Threshold': round(over_threshold, 4),
                    'Actual_Penalty': round(actual_penalty, 4),
                    'S_final_changed': round(s_final_changed, 4),
                    'text_snippet': doc_text[:200] if doc_text else ''
                })
            
            # 计算 Changed 模式排名
            sorted_by_changed = sorted(doc_details, key=lambda x: (-x['S_final_changed'], x['doc_id']))
            doc_rank_changed_map = {d['doc_id']: rank for rank, d in enumerate(sorted_by_changed, 1)}
            for doc in doc_details:
                doc['rank_changed'] = doc_rank_changed_map.get(doc['doc_id'], 1000)
            
            # 计算 OG 模式排名
            sorted_by_og = sorted(doc_details, key=lambda x: (-x['S_pos_og'], x['doc_id']))
            doc_rank_og_map = {d['doc_id']: rank for rank, d in enumerate(sorted_by_og, 1)}
            for doc in doc_details:
                doc['rank_og'] = doc_rank_og_map.get(doc['doc_id'], 1000)
            
            # 筛选：所有包含负向词的文档，按 Changed 模式排名排序
            all_neg_docs = [d for d in doc_details if d['contains_neg']]
            all_neg_docs = sorted(all_neg_docs, key=lambda x: x['rank_changed'])
            
            # 筛选：排名前10且包含负向词的文档（漏网烂文）
            survivor_docs = [d for d in all_neg_docs if d['rank_changed'] <= 10]
            
            # 获取前10名文档（无论是否含负向词）
            top10_docs = sorted(doc_details, key=lambda x: x['rank_changed'])[:10]
            
            # 获取相关文档的排名 - Changed 模式
            relevant_docs_ranking_changed = []
            for doc in doc_details:
                if doc['is_relevant_changed']:
                    relevant_docs_ranking_changed.append({
                        'doc_id': doc['doc_id'],
                        'rank_changed': doc['rank_changed'],
                        'relevance': doc['relevance_score_changed'],
                        'S_pos_changed': doc['S_pos_changed'],
                        'S_final_changed': doc['S_final_changed'],
                        'contains_neg': doc['contains_neg']
                    })
            relevant_docs_ranking_changed = sorted(relevant_docs_ranking_changed, key=lambda x: x['rank_changed'])
            
            # 获取相关文档的排名 - OG 模式
            relevant_docs_ranking_og = []
            for doc in doc_details:
                if doc['is_relevant_og']:
                    relevant_docs_ranking_og.append({
                        'doc_id': doc['doc_id'],
                        'rank_og': doc['rank_og'],
                        'relevance': doc['relevance_score_og'],
                        'S_pos_og': doc['S_pos_og'],
                        'contains_neg': doc['contains_neg']
                    })
            relevant_docs_ranking_og = sorted(relevant_docs_ranking_og, key=lambda x: x['rank_og'])
            
            # 获取查询内容
            q_plus_text = ""
            q_minus_text = ""
            if q_raw_changed and changed_key in q_raw_changed:
                q_plus_text, q_minus_text = q_raw_changed[changed_key]
            
            analysis_results[qid] = {
                'query_id': qid,
                'q_plus': q_plus_text,
                'q_minus': q_minus_text,
                'neg_words': neg_words,
                'dynamic_tau': round(tau_value, 4),
                'alpha': alpha,
                'delta': delta,
                'survivor_docs': survivor_docs[:10],  # Top-10 漏网文档
                'all_neg_docs': all_neg_docs[:20],  # Top-20 包含负向词的文档（用于分析）
                'top10_docs': top10_docs,  # 前10名文档
                'relevant_docs_ranking_changed': relevant_docs_ranking_changed[:15],  # Changed 模式相关文档排名
                'relevant_docs_ranking_og': relevant_docs_ranking_og[:15],  # OG 模式相关文档排名
                'all_docs_count': len(doc_details),
                'neg_docs_count': len(all_neg_docs),
                'survivor_count': len(survivor_docs)
            }
        
        return analysis_results

    def _save_results(
        self,
        all_results: List[Dict],
        best_params,
        best_metrics: Dict,
        all_query_metrics: Optional[List[Dict]] = None
    ) -> None:
        """保存评测结果"""
        results_path = os.path.join(self.output_dir, "random_search_results.json")
        
        if self.use_mlp:
            save_data = {
                'best_params': best_params,
                'mode': 'dynamic_mlp',
                'best_metrics': best_metrics,
                'all_results': all_results
            }
        else:
            save_data = {
                'best_params': {'alpha': best_params[0], 'tau': best_params[1]},
                'best_metrics': best_metrics,
                'all_results': all_results,
                'all_query_metrics': all_query_metrics or []
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 结果已保存: {results_path}")
        
        # 打印白盒分析报告（如果有）
        self._print_whitebox_report(all_results)

    def _print_whitebox_report(self, all_results: List[Dict]) -> None:
        """打印白盒漏网烂文追踪报告"""
        # 查找包含白盒分析的结果
        whitebox_results = [r for r in all_results if 'whitebox_analysis' in r]
        
        if not whitebox_results:
            return
        
        for result in whitebox_results:
            alpha = result.get('alpha', 0)
            delta = result.get('delta', 0)
            analysis = result.get('whitebox_analysis', {})
            
            logger.info("\n" + "="*80)
            logger.info(f"🔍 白盒漏网烂文追踪报告 (α={alpha}, Δ={delta})")
            logger.info("="*80)
            
            for qid, query_analysis in analysis.items():
                neg_words = query_analysis.get('neg_words', [])
                dynamic_tau = query_analysis.get('dynamic_tau', 0)
                survivor_docs = query_analysis.get('survivor_docs', [])
                all_neg_docs = query_analysis.get('all_neg_docs', [])
                neg_docs_count = query_analysis.get('neg_docs_count', 0)
                
                logger.info(f"\n📋 Query {qid} (负向词: {', '.join(neg_words)})")
                logger.info(f"   Dynamic_Tau (内生底噪): {dynamic_tau:.4f}")
                logger.info(f"   包含负向词的文档总数: {neg_docs_count}")
                logger.info(f"   漏网文档数 (排名前10): {len(survivor_docs)}")
                logger.info("-"*80)
                
                if survivor_docs:
                    # 打印 Top-2 漏网文档
                    logger.info(f"\n   ⚠️ 漏网烂文 (免疫惩罚的文档):")
                    for i, doc in enumerate(survivor_docs[:2], 1):
                        logger.info(f"\n   🏆 漏网文档 #{i} (排名: {doc['rank']})")
                        logger.info(f"      doc_id:           {doc['doc_id']}")
                        logger.info(f"      is_relevant:      {doc['is_relevant']} (Ground Truth)")
                        logger.info(f"      S_pos:            {doc['S_pos']:.4f} (原始正向基准分)")
                        logger.info(f"      S_neg_proj:       {doc['S_neg_proj']:.4f} (负向投影分)")
                        logger.info(f"      Dynamic_Tau:      {doc['Dynamic_Tau']:.4f} (动态护盾值)")
                        logger.info(f"      Over_Threshold:   {doc['Over_Threshold']:.4f} (S_neg_proj - Tau)")
                        logger.info(f"      Actual_Penalty:   {doc['Actual_Penalty']:.4f} (α × max(0, Over_Threshold))")
                        logger.info(f"      S_final:          {doc['S_final']:.4f} (S_pos - Actual_Penalty)")
                        logger.info(f"      文本片段:         {doc['text_snippet'][:100]}...")
                    
                    if len(survivor_docs) > 2:
                        logger.info(f"\n   ... 还有 {len(survivor_docs) - 2} 个漏网文档 ...")
                else:
                    logger.info(f"\n   ✅ 无漏网烂文！所有含负向词的文档都被惩罚到10名之外")
                
                # 打印被惩罚最严重的文档（排名最靠前但被惩罚的）
                if all_neg_docs:
                    logger.info(f"\n   📉 被惩罚最重的文档 (Top-2 含负向词但排名>10):")
                    punished_docs = [d for d in all_neg_docs if d['rank'] > 10][:2]
                    for i, doc in enumerate(punished_docs, 1):
                        logger.info(f"\n   🔻 被惩罚文档 #{i} (排名: {doc['rank']})")
                        logger.info(f"      doc_id:           {doc['doc_id']}")
                        logger.info(f"      is_relevant:      {doc['is_relevant']} (Ground Truth)")
                        logger.info(f"      S_pos:            {doc['S_pos']:.4f} (原始正向基准分)")
                        logger.info(f"      S_neg_proj:       {doc['S_neg_proj']:.4f} (负向投影分)")
                        logger.info(f"      Dynamic_Tau:      {doc['Dynamic_Tau']:.4f} (动态护盾值)")
                        logger.info(f"      Over_Threshold:   {doc['Over_Threshold']:.4f} (冒出护盾部分)")
                        logger.info(f"      Actual_Penalty:   {doc['Actual_Penalty']:.4f} (实际惩罚值)")
                        logger.info(f"      S_final:          {doc['S_final']:.4f} (惩罚后得分)")
                        logger.info(f"      文本片段:         {doc['text_snippet'][:80]}...")
                
                logger.info("-"*80)

    def _save_whitebox_report(self, analysis: Dict[str, Any], alpha: float, delta: float) -> None:
        """保存白盒分析报告为 Markdown 文件"""
        report_path = os.path.join(self.output_dir, "whitebox_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🔍 白盒检索诊断报告\n\n")
            f.write(f"**参数**: α={alpha}, Δ={delta}\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            for qid, query_analysis in analysis.items():
                neg_words = query_analysis.get('neg_words', [])
                dynamic_tau = query_analysis.get('dynamic_tau', 0)
                survivor_docs = query_analysis.get('survivor_docs', [])
                all_neg_docs = query_analysis.get('all_neg_docs', [])
                neg_docs_count = query_analysis.get('neg_docs_count', 0)
                q_plus = query_analysis.get('q_plus', '')
                q_minus = query_analysis.get('q_minus', '')
                top10_docs = query_analysis.get('top10_docs', [])
                relevant_docs_ranking_changed = query_analysis.get('relevant_docs_ranking_changed', [])
                relevant_docs_ranking_og = query_analysis.get('relevant_docs_ranking_og', [])
                
                f.write(f"## 📋 Query {qid}\n\n")
                f.write(f"### 查询内容\n\n")
                f.write(f"- **Q+ (正查询)**: {q_plus[:200] if q_plus else '(无)'}\n")
                f.write(f"- **Q- (负查询)**: {q_minus[:200] if q_minus else '(无)'}\n")
                f.write(f"- **负向词**: {', '.join(neg_words)}\n")
                f.write(f"- **Dynamic_Tau (内生底噪)**: {dynamic_tau:.4f}\n")
                f.write(f"- **包含负向词的文档总数**: {neg_docs_count}\n")
                f.write(f"- **漏网文档数 (排名前10)**: {len(survivor_docs)}\n\n")
                
                # 前10名文档分析 - 同时显示 OG 和 Changed 排名
                f.write(f"### 📊 前10名文档 (Changed 模式 S_final 排序)\n\n")
                f.write(f"| 排名(OG) | 排名(Changed) | doc_id | relevance(Changed) | 含负向词 | S_pos(OG) | S_final(Changed) |\n")
                f.write(f"|----------|---------------|--------|---------------------|----------|------------|------------------|\n")
                for doc in top10_docs:
                    rel = doc.get('relevance_score_changed', 0)
                    contains_neg = '✓' if doc.get('contains_neg') else '✗'
                    f.write(f"| {doc['rank_og']} | {doc['rank_changed']} | {doc['doc_id']} | {rel} | {contains_neg} | {doc['S_pos_og']:.4f} | {doc['S_final_changed']:.4f} |\n")
                
                # 相关文档排名分析 - OG 模式
                if relevant_docs_ranking_og:
                    f.write(f"\n### 🎯 OG 模式 - 相关文档排名 (relevance > 0)\n\n")
                    f.write(f"| 排名 | doc_id | relevance | S_pos |\n")
                    f.write(f"|------|--------|-----------|-------|\n")
                    for doc in relevant_docs_ranking_og[:15]:
                        f.write(f"| {doc['rank_og']} | {doc['doc_id']} | {doc['relevance']} | {doc['S_pos_og']:.4f} |\n")
                    
                    if relevant_docs_ranking_og:
                        avg_rank_og = sum(d['rank_og'] for d in relevant_docs_ranking_og) / len(relevant_docs_ranking_og)
                        top1_og = relevant_docs_ranking_og[0]['rank_og'] if relevant_docs_ranking_og else 'N/A'
                        f.write(f"\n**相关文档平均排名**: {avg_rank_og:.1f}, **最佳排名**: {top1_og}\n")
                
                # 相关文档排名分析 - Changed 模式
                if relevant_docs_ranking_changed:
                    f.write(f"\n### 🎯 Changed 模式 - 相关文档排名 (relevance > 0)\n\n")
                    f.write(f"| 排名 | doc_id | relevance | S_final |\n")
                    f.write(f"|------|--------|-----------|---------|\n")
                    for doc in relevant_docs_ranking_changed[:15]:
                        f.write(f"| {doc['rank_changed']} | {doc['doc_id']} | {doc['relevance']} | {doc['S_final_changed']:.4f} |\n")
                    
                    if relevant_docs_ranking_changed:
                        avg_rank_changed = sum(d['rank_changed'] for d in relevant_docs_ranking_changed) / len(relevant_docs_ranking_changed)
                        top1_changed = relevant_docs_ranking_changed[0]['rank_changed'] if relevant_docs_ranking_changed else 'N/A'
                        f.write(f"\n**相关文档平均排名**: {avg_rank_changed:.1f}, **最佳排名**: {top1_changed}\n")
                
                # 漏网烂文分析
                if survivor_docs:
                    f.write(f"\n### ⚠️ 漏网烂文 (免疫惩罚的文档)\n\n")
                    f.write(f"| 排名 | doc_id | relevance | S_pos(Changed) | S_neg_proj | Dynamic_Tau | Over_Threshold | Actual_Penalty | S_final(Changed) |\n")
                    f.write(f"|------|--------|-----------|----------------|------------|-------------|----------------|----------------|------------------|\n")
                    for doc in survivor_docs[:5]:
                        rel = doc.get('relevance_score_changed', 0)
                        f.write(f"| {doc['rank_changed']} | {doc['doc_id']} | {rel} | {doc['S_pos_changed']:.4f} | {doc['S_neg_proj']:.4f} | {doc['Dynamic_Tau']:.4f} | {doc['Over_Threshold']:.4f} | {doc['Actual_Penalty']:.4f} | {doc['S_final_changed']:.4f} |\n")
                    
                    if len(survivor_docs) > 5:
                        f.write(f"\n*... 还有 {len(survivor_docs) - 5} 个漏网文档 ...*\n")
                else:
                    f.write(f"\n### ✅ 无漏网烂文\n\n")
                    f.write(f"所有含负向词的文档都被惩罚到10名之外。\n\n")
                
                # 被惩罚最重的文档
                if all_neg_docs:
                    f.write(f"\n### 📉 被惩罚最重的文档 (Top-5 含负向词但排名>10)\n\n")
                    f.write(f"| 排名 | doc_id | relevance | S_pos(Changed) | S_neg_proj | Dynamic_Tau | Over_Threshold | Actual_Penalty | S_final(Changed) |\n")
                    f.write(f"|------|--------|-----------|----------------|------------|-------------|----------------|----------------|------------------|\n")
                    punished_docs = [d for d in all_neg_docs if d['rank_changed'] > 10][:5]
                    for doc in punished_docs:
                        rel = doc.get('relevance_score_changed', 0)
                        f.write(f"| {doc['rank_changed']} | {doc['doc_id']} | {rel} | {doc['S_pos_changed']:.4f} | {doc['S_neg_proj']:.4f} | {doc['Dynamic_Tau']:.4f} | {doc['Over_Threshold']:.4f} | {doc['Actual_Penalty']:.4f} | {doc['S_final_changed']:.4f} |\n")
                
                f.write("\n---\n\n")
        
        logger.info(f"💾 白盒分析报告已保存: {report_path}")

    def _save_trec_format(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: str
    ) -> None:
        """保存 TREC 格式结果文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for qid in sorted(results.keys()):
                doc_scores = results[qid]
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} dscrl\n")
        logger.info(f"✅ TREC 文件已保存: {output_path}")

    def _save_all_params_summary(
        self,
        all_results: List[Dict[str, Any]]
    ) -> None:
        """保存所有参数的简化汇总"""
        summary_path = os.path.join(self.output_dir, "all_params_summary.csv")
        
        # 判断是否为动态 τ 模式
        is_dynamic_tau = 'delta' in all_results[0] if all_results else False
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            if is_dynamic_tau:
                f.write("alpha,delta,avg_tau,pMRR,og_nDCG1,og_nDCG5,og_nDCG10,og_nDCG100,og_MAP1,og_MAP5,og_MAP10,og_MAP100,og_MAP1000,changed_nDCG1,changed_nDCG5,changed_nDCG10,changed_nDCG100,changed_MAP1,changed_MAP5,changed_MAP10,changed_MAP100,changed_MAP1000\n")
            else:
                f.write("alpha,tau,pMRR,og_nDCG1,og_nDCG5,og_nDCG10,og_nDCG100,og_MAP1,og_MAP5,og_MAP10,og_MAP100,og_MAP1000,changed_nDCG1,changed_nDCG5,changed_nDCG10,changed_nDCG100,changed_MAP1,changed_MAP5,changed_MAP10,changed_MAP100,changed_MAP1000\n")
            
            for result in all_results:
                alpha = result.get('alpha', 0)
                p_mrr = result.get('p-MRR', 0)
                metrics = result.get('metrics', {})
                
                if is_dynamic_tau:
                    delta = result.get('delta', 0)
                    avg_tau = result.get('avg_tau', 0)
                    line = f"{alpha},{delta},{avg_tau:.6f},{p_mrr:.6f},"
                else:
                    tau = result.get('tau', 0)
                    line = f"{alpha},{tau},{p_mrr:.6f},"
                
                full_scores = metrics.get('full_scores', {})
                og_metrics = full_scores.get('og', {})
                changed_metrics = full_scores.get('changed', {})
                
                line += f"{og_metrics.get('ndcg_at_1', 0):.6f},"
                line += f"{og_metrics.get('ndcg_at_5', 0):.6f},"
                line += f"{og_metrics.get('ndcg_at_10', 0):.6f},"
                line += f"{og_metrics.get('ndcg_at_100', 0):.6f},"
                line += f"{og_metrics.get('map_at_1', 0):.6f},"
                line += f"{og_metrics.get('map_at_5', 0):.6f},"
                line += f"{og_metrics.get('map_at_10', 0):.6f},"
                line += f"{og_metrics.get('map_at_100', 0):.6f},"
                line += f"{og_metrics.get('map_at_1000', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_1', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_5', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_10', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_100', 0):.6f},"
                line += f"{changed_metrics.get('map_at_1', 0):.6f},"
                line += f"{changed_metrics.get('map_at_5', 0):.6f},"
                line += f"{changed_metrics.get('map_at_10', 0):.6f},"
                line += f"{changed_metrics.get('map_at_100', 0):.6f},"
                line += f"{changed_metrics.get('map_at_1000', 0):.6f}\n"
                
                f.write(line)
        
        logger.info(f"💾 所有参数汇总已保存: {summary_path}")

    def _compute_per_query_metrics(
        self,
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        candidates: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """计算每个查询的详细性能指标"""
        from eval.metrics.evaluator import DataLoader
        
        data_loader = DataLoader(self.task_name)
        qrels = data_loader.load_qrels()
        
        query_metrics = []
        
        # 获取所有查询的基础ID (用于匹配 qrels)
        processed_qids = set()
        for qid in query_ids_og:
            base_qid = qid.replace('-og', '')
            # qrels 的键格式是 "{id}-og" 或 "{id}-changed"
            processed_qids.add(f"{base_qid}-og")
        
        # 过滤 qrels 只保留需要的
        filtered_qrels = {k: v for k, v in qrels.items() if k in processed_qids}
        
        # 获取所有查询的基础ID
        for idx, qid in enumerate(query_ids_og):
            base_qid = qid.replace('-og', '')
            changed_qid = qid.replace('-og', '-changed')
            
            # qrels 键格式是 "{id}-og" 或 "{id}-changed"，需要用完整键来查找
            og_qid = f"{base_qid}-og"
            
            if og_qid not in filtered_qrels:
                continue
            
            # 获取真实相关文档 - OG模式用OG qrels，Changed模式用Changed qrels
            relevant_docs_og = set(filtered_qrels.get(og_qid, {}).keys())
            relevant_docs_changed = set(filtered_qrels.get(changed_qid, {}).keys())
            
            # 获取模型返回的排序结果
            og_scores = results_og.get(qid, {})
            changed_scores = results_changed.get(changed_qid, {}) if changed_qid in results_changed else {}
            
            # 计算各个指标
            for k in [1, 3, 5, 10, 100, 1000]:
                # OG nDCG@k
                og_ndcg = self._compute_ndcg(og_scores, relevant_docs_og, k)
                # Changed nDCG@k (使用 Changed qrels)
                changed_ndcg = self._compute_ndcg(changed_scores, relevant_docs_changed, k) if changed_scores else 0
                
                # MAP@k
                og_map = self._compute_map(og_scores, relevant_docs_og, k)
                changed_map = self._compute_map(changed_scores, relevant_docs_changed, k) if changed_scores else 0
                
                # MRR@k
                og_mrr = self._compute_mrr(og_scores, relevant_docs_og, k)
                changed_mrr = self._compute_mrr(changed_scores, relevant_docs_changed, k) if changed_scores else 0
                
                query_metrics.append({
                    'qid': base_qid,
                    'query_type': 'og' if '-og' in qid else 'changed',
                    'k': k,
                    'ndcg': og_ndcg,
                    'map': og_map,
                    'mrr': og_mrr
                })
                
                if changed_scores:
                    query_metrics.append({
                        'qid': base_qid,
                        'query_type': 'changed',
                        'k': k,
                        'ndcg': changed_ndcg,
                        'map': changed_map,
                        'mrr': changed_mrr
                    })
        
        return query_metrics

    def _compute_ndcg(
        self,
        scores: Dict[str, float],
        relevant_docs: set,
        k: int
    ) -> float:
        """计算 NDCG@k"""
        if not scores:
            return 0.0
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(rank + 1)
        
        # 计算 IDCG
        num_relevant = min(len(relevant_docs), k)
        if num_relevant == 0:
            return 0.0
        
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, num_relevant + 1))
        
        return dcg / idcg if idcg > 0 else 0.0

    def _compute_map(
        self,
        scores: Dict[str, float],
        relevant_docs: set,
        k: int
    ) -> float:
        """计算 MAP@k"""
        if not scores or not relevant_docs:
            return 0.0
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        num_relevant = 0
        precision_sum = 0.0
        
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_sum += num_relevant / rank
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0

    def _compute_mrr(
        self,
        scores: Dict[str, float],
        relevant_docs: set,
        k: int
    ) -> float:
        """计算 MRR@k"""
        if not scores or not relevant_docs:
            return 0.0
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        
        return 0.0

    def _save_structured_summary(
        self,
        best_metrics: Dict[str, Any],
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]],
        best_params: Any = None
    ) -> None:
        """保存结构化汇总文件"""
        full_scores = best_metrics.get('full_scores', {})
        
        # 解析 best_params
        params_dict = {}
        if best_params is not None:
            if isinstance(best_params, dict):
                params_dict = best_params
            elif isinstance(best_params, (tuple, list)) and len(best_params) >= 2:
                params_dict = {
                    'alpha': best_params[0],
                    'delta_or_tau': best_params[1]
                }
        
        summary = {
            'task': self.task_name,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'best_params': params_dict,
            'metrics': {
                'p-MRR': full_scores.get('p-MRR', best_metrics.get('p-MRR', 0)),
                'original': {
                    'ndcg_at_1': full_scores.get('og', {}).get('ndcg_at_1', 0),
                    'ndcg_at_3': full_scores.get('og', {}).get('ndcg_at_3', 0),
                    'ndcg_at_5': full_scores.get('og', {}).get('ndcg_at_5', 0),
                    'ndcg_at_10': full_scores.get('og', {}).get('ndcg_at_10', 0),
                    'ndcg_at_100': full_scores.get('og', {}).get('ndcg_at_100', 0),
                    'ndcg_at_1000': full_scores.get('og', {}).get('ndcg_at_1000', 0),
                    'map_at_1': full_scores.get('og', {}).get('map_at_1', 0),
                    'map_at_3': full_scores.get('og', {}).get('map_at_3', 0),
                    'map_at_5': full_scores.get('og', {}).get('map_at_5', 0),
                    'map_at_10': full_scores.get('og', {}).get('map_at_10', 0),
                    'map_at_100': full_scores.get('og', {}).get('map_at_100', 0),
                    'map_at_1000': full_scores.get('og', {}).get('map_at_1000', 0),
                },
                'changed': {
                    'ndcg_at_1': full_scores.get('changed', {}).get('ndcg_at_1', 0),
                    'ndcg_at_3': full_scores.get('changed', {}).get('ndcg_at_3', 0),
                    'ndcg_at_5': full_scores.get('changed', {}).get('ndcg_at_5', 0),
                    'ndcg_at_10': full_scores.get('changed', {}).get('ndcg_at_10', 0),
                    'ndcg_at_100': full_scores.get('changed', {}).get('ndcg_at_100', 0),
                    'ndcg_at_1000': full_scores.get('changed', {}).get('ndcg_at_1000', 0),
                    'map_at_1': full_scores.get('changed', {}).get('map_at_1', 0),
                    'map_at_3': full_scores.get('changed', {}).get('map_at_3', 0),
                    'map_at_5': full_scores.get('changed', {}).get('map_at_5', 0),
                    'map_at_10': full_scores.get('changed', {}).get('map_at_10', 0),
                    'map_at_100': full_scores.get('changed', {}).get('map_at_100', 0),
                    'map_at_1000': full_scores.get('changed', {}).get('map_at_1000', 0),
                },
                'full_scores': full_scores
            }
        }
        
        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 结构化指标汇总已保存: {summary_path}")

    def _generate_bad_case_analysis(
        self,
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        q_minus_map: Dict[str, str]
    ) -> None:
        """生成极端坏例分析诊断报告"""
        report_path = os.path.join(self.output_dir, "bad_case_analysis.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🔍 DSCLR 极端坏例分析报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            report_data = self._analyze_extreme_cases(
                all_query_metrics, q_raw_og, q_raw_changed, 
                candidates, all_results, q_minus_map
            )
            
            f.write(report_data['markdown'])
        
        json_report_path = os.path.join(self.output_dir, "bad_case_analysis.json")
        with open(json_report_path, 'w', encoding='utf-8') as json_f:
            json.dump(report_data['json'], json_f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 坏例分析报告已保存: {report_path}")
        logger.info(f"💾 JSON格式报告已保存: {json_report_path}")
    
    def _analyze_extreme_cases(
        self,
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        q_minus_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """分析极端坏例"""
        
        # 兼容动态 τ 模式（使用 delta 而不是 tau）
        if 'tau' in all_results[0]:
            result_params = {r['alpha']: r['tau'] for r in all_results}
            alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
            alpha_5_tau_7_result = next((r for r in all_results if r['alpha'] == 5.0 and abs(r['tau'] - 0.7) < 0.01), None)
            alpha_1_tau_8_result = next((r for r in all_results if r['alpha'] == 1.0 and abs(r['tau'] - 0.8) < 0.01), None)
        else:
            # 动态 τ 模式
            result_params = {r['alpha']: r.get('avg_tau', 0) for r in all_results}
            alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
            alpha_5_tau_7_result = next((r for r in all_results if r['alpha'] == 5.0 and abs(r.get('delta', 0) - 0.0) < 0.01), None)
            alpha_1_tau_8_result = next((r for r in all_results if r['alpha'] == 1.0 and abs(r.get('delta', 0) - 0.0) < 0.01), None)
        
        if not alpha_0_result:
            return {'markdown': '# ❌ 错误：未找到 α=0.0 的结果\n', 'json': {'error': 'Missing alpha=0.0 results'}}
        
        og_metrics = [m for m in all_query_metrics if m['query_type'] == 'og' and m['k'] == 5]
        
        query_neg_scores = self._compute_query_negative_scores(
            q_raw_changed, candidates, all_results, q_minus_map
        )
        
        selected_queries = self._select_extreme_queries(
            og_metrics, query_neg_scores, q_raw_og, q_raw_changed, alpha_0_result
        )
        
        # 判断是否为动态 τ 模式
        is_dynamic_tau = 'delta' in all_results[0] if all_results else False
        
        markdown_output = self._generate_query_analysis_markdown(
            selected_queries, candidates, all_results,
            alpha_0_result, alpha_5_tau_7_result, alpha_1_tau_8_result,
            q_raw_changed, query_neg_scores, is_dynamic_tau
        )
        
        json_output = self._generate_query_analysis_json(
            selected_queries, candidates, all_results,
            alpha_0_result, alpha_5_tau_7_result, alpha_1_tau_8_result,
            q_raw_changed, query_neg_scores, is_dynamic_tau
        )
        
        return {'markdown': markdown_output, 'json': json_output}
    
    def _compute_query_negative_scores(
        self,
        q_raw_changed: Dict[str, Tuple[str, str]],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        q_minus_map: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """计算每个Query的正负向得分 - 使用 reformulator 提取的 Q-"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        
        if not alpha_0_result:
            return {}
        
        from eval.metrics.evaluator import DataLoader
        data_loader = DataLoader(self.task_name)
        qrels = data_loader.load_qrels()
        
        query_neg_scores = {}
        
        for qid in [k.split('-')[0] for k in q_raw_changed.keys()]:
            changed_key = f"{qid}-changed"
            
            if changed_key not in q_raw_changed:
                continue
            
            # 从 q_minus_map 获取 reformulator 提取的 Q-
            q_minus = q_minus_map.get(changed_key, "[NONE]")
            
            if q_minus == "[NONE]" or not q_minus:
                query_neg_scores[qid] = {
                    'neg_words': [],
                    'doc_scores': {},
                    'relevant_docs': set(),
                    'irrelevant_docs': set()
                }
                continue
            
            # Q- 是逗号分隔的负向词列表
            neg_words_list = [w.strip() for w in q_minus.split(',') if w.strip()]
            
            query_neg_scores[qid] = {
                'neg_words': neg_words_list,
                'doc_scores': {},
                'relevant_docs': set(),
                'irrelevant_docs': set()
            }
            
            results_changed = alpha_0_result.get('results_changed', {})
            
            changed_key_result = f"{qid}-changed"
            if changed_key_result in results_changed:
                for doc_id, score in results_changed[changed_key_result][:50]:
                    doc = candidates.get(doc_id, {})
                    doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                    
                    # 从 qrels 获取相关性标注
                    qrel_key = f"{qid}-changed"
                    relevance = qrels.get(qrel_key, {}).get(doc_id, 0)
                    
                    if relevance > 0:
                        query_neg_scores[qid]['relevant_docs'].add(doc_id)
                    else:
                        query_neg_scores[qid]['irrelevant_docs'].add(doc_id)
                    
                    # 检查文档是否包含负向词
                    neg_word_found = None
                    for neg_word in neg_words_list:
                        if neg_word.lower() in doc_text.lower():
                            neg_word_found = neg_word
                            break
                    
                    if neg_word_found:
                        query_neg_scores[qid]['doc_scores'][doc_id] = {
                            'score': score,
                            'neg_word': neg_word_found,
                            'text_snippet': doc_text[:200]
                        }
        
        return query_neg_scores
    
    def _select_extreme_queries(
        self,
        og_metrics: List[Dict[str, Any]],
        query_neg_scores: Dict[str, Dict[str, float]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, Tuple[str, str]]],
        alpha_0_result: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """选择4种极端类型的Query"""
        
        # 从 alpha_0_result 获取所有文档的 S_neg_proj 用于计算平均
        results_changed = alpha_0_result.get('results_changed', {}) if alpha_0_result else {}
        
        query_type_scores = {}
        for m in og_metrics:
            qid = str(m['qid'])
            mrr = m['mrr']
            
            neg_info = query_neg_scores.get(qid, {})
            
            # 计算平均负向得分：使用所有文档的 S_neg_proj，不只是包含负向词的文档
            changed_key = f"{qid}-changed"
            
            # 检查是否为 [NONE] 查询（没有负向约束）
            neg_words = neg_info.get('neg_words', [])
            is_none_query = len(neg_words) == 0
            
            if is_none_query:
                # [NONE] 查询没有负向约束，负向得分应为 0
                avg_neg_score = 0.0
            elif changed_key in results_changed:
                all_scores = [score for _, score in results_changed[changed_key]]
                avg_neg_score = np.mean(all_scores) if all_scores else 0.0
            else:
                avg_neg_score = 0.0
            
            query_type_scores[qid] = {
                'mrr': mrr,
                'avg_neg_score': avg_neg_score,
                'neg_info': neg_info,
                'is_none_query': is_none_query
            }
        
        high_noise = []
        low_noise = []
        entity_entangled = []
        logical_negation = []
        
        for qid, scores in query_type_scores.items():
            avg_neg = scores['avg_neg_score']
            
            neg_info = scores.get('neg_info', {})
            neg_words = neg_info.get('neg_words', [])
            
            has_entity_neg = any(
                any(c.isalpha() and len(c) > 3 for c in w.split()) 
                for w in neg_words
            )
            
            is_logical_only = all(
                any(neg in w.lower() for neg in ['not', 'no', 'without', 'except', '除了', '不要', '非', '无'])
                for w in neg_words
            ) if neg_words else True
            
            if avg_neg > 0.65:
                high_noise.append((qid, scores, 'high_noise'))
            elif avg_neg < 0.55 and avg_neg > 0:
                low_noise.append((qid, scores, 'low_noise'))
            
            if is_logical_only and neg_words:
                logical_negation.append((qid, scores, 'logical_negation'))
        
        entity_keywords = ['病', '癌', '基因', '细胞', '蛋白', '病毒', '遗传', '突', '转基因', 
                          'cancer', 'gene', 'protein', 'virus', 'genetic', 'mutant']
        
        for qid, scores in query_type_scores.items():
            og_key = f"{qid}-og"
            raw = q_raw_og.get(og_key, ("", ""))
            query_text = raw[0].lower() if raw else ""
            
            has_entity = any(kw in query_text for kw in entity_keywords)
            
            if has_entity:
                neg_info = scores.get('neg_info', {})
                if neg_info.get('neg_words'):
                    entity_entangled.append((qid, scores, 'entity_entangled'))
        
        selected = []
        
        if high_noise:
            selected.append(high_noise[0])
        if low_noise:
            selected.append(low_noise[0])
        if entity_entangled:
            selected.append(entity_entangled[0])
        if logical_negation:
            selected.append(logical_negation[0])
        
        remaining = []
        for qid, scores, qtype in high_noise[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        for qid, scores, qtype in low_noise[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        for qid, scores, qtype in entity_entangled[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        for qid, scores, qtype in logical_negation[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        
        selected.extend(remaining[:max(0, 8 - len(selected))])
        
        return selected
    
    def _generate_query_analysis_markdown(
        self,
        selected_queries: List[Tuple],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        alpha_0_result: Optional[Dict],
        alpha_5_tau_7_result: Optional[Dict],
        alpha_1_tau_8_result: Optional[Dict],
        q_raw_changed: Dict[str, Tuple[str, str]],
        query_neg_scores: Dict[str, Dict],
        is_dynamic_tau: bool = False
    ) -> str:
        """生成Markdown格式的分析报告"""
        
        markdown = "## 📋 选中的极端Query分析\n\n"
        
        type_names = {
            'high_noise': '🔴 高底噪型',
            'low_noise': '🟢 低底噪型',
            'entity_entangled': '🟠 实体纠缠型',
            'logical_negation': '🔵 纯逻辑否定型'
        }
        
        for qid, scores, qtype in selected_queries:
            changed_key = f"{qid}-changed"
            raw = q_raw_changed.get(changed_key, ("", ""))
            query_text, _ = raw
            
            neg_info = query_neg_scores.get(qid, {})
            neg_words_list = neg_info.get('neg_words', [])
            q_minus = ', '.join(neg_words_list) if neg_words_list else '[NONE]'
            
            is_none_query = scores.get('is_none_query', False)
            
            markdown += f"### {type_names.get(qtype, qtype)} - Q{qid}\n\n"
            markdown += f"**Query**: {query_text}\n\n"
            markdown += f"**负向词 (Q-)**: {q_minus}\n\n"
            markdown += f"**当前MRR**: {scores['mrr']:.4f}\n\n"
            
            if is_none_query:
                markdown += f"**平均负向得分**: N/A (无负向约束)\n\n"
            else:
                markdown += f"**平均负向得分**: {scores['avg_neg_score']:.4f}\n\n"
            
            markdown += f"---\n\n"
            markdown += f"#### 【数据组 A：假阳性（漏网的烂文）】 α=0.0\n\n"
            
            fp_docs = self._extract_false_positives(
                qid, candidates, alpha_0_result, neg_info
            )
            
            for i, doc in enumerate(fp_docs[:3], 1):
                markdown += f"**A-{i}**: `doc_id={doc['doc_id']}`\n"
                markdown += f"- Snippet: {doc['snippet']}\n"
                markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n\n"
            
            markdown += f"---\n\n"
            if is_dynamic_tau:
                markdown += f"#### 【数据组 B：高惩罚文档（排名大幅下降）】 α=5.0, Δ=0.0 (动态τ)\n\n"
            else:
                markdown += f"#### 【数据组 B：高惩罚文档（排名大幅下降）】 α=5.0, τ=0.7\n\n"
            
            fn_docs = self._extract_false_negatives(
                qid, candidates, all_results, neg_info
            )
            
            for i, doc in enumerate(fn_docs[:3], 1):
                markdown += f"**B-{i}**: `doc_id={doc['doc_id']}`\n"
                markdown += f"- Snippet: {doc['snippet']}\n"
                markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n"
                markdown += f"- Penalty: {doc['penalty']:.4f}\n"
                markdown += f"- 原排名: {doc['original_rank']}, 现排名: {doc['current_rank']}\n\n"
            
            markdown += f"---\n\n"
            if is_dynamic_tau:
                markdown += f"#### 【数据组 C：当前最优参数下的残留误差】 α=1.5, Δ=0.0 (动态τ, 最佳参数)\n\n"
            else:
                markdown += f"#### 【数据组 C：当前最优参数下的残留误差】 α=1.0, τ=0.8\n\n"
            
            if alpha_1_tau_8_result:
                c_docs = self._extract_optimal_residual_errors(
                    qid, candidates, alpha_1_tau_8_result, all_results
                )
                
                if c_docs['false_positive']:
                    doc = c_docs['false_positive']
                    markdown += f"**C-1 漏网烂文**: `doc_id={doc['doc_id']}`\n"
                    markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n"
                    markdown += f"- 排名: {doc['rank']}\n\n"
                
                if c_docs['false_negative']:
                    doc = c_docs['false_negative']
                    markdown += f"**C-2 冤枉好文**: `doc_id={doc['doc_id']}`\n"
                    markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n"
                    markdown += f"- 排名: {doc['rank']}\n\n"
            
            markdown += f"---\n\n"
            markdown += f"#### 【数据组 D：特征倒挂点】❗最高优先级\n\n"
            
            inversion = self._extract_feature_inversions(
                qid, candidates, all_results, neg_info
            )
            
            if inversion:
                markdown += f"**倒挂文档对**:\n\n"
                markdown += f"- **高S_neg被惩罚文档**: `doc_id={inversion['good']['doc_id']}`\n"
                markdown += f"  - Snippet: {inversion['good']['snippet']}\n"
                markdown += f"  - $S_{{neg\_proj}}$: {inversion['good']['S_neg_proj']:.4f}\n\n"
                markdown += f"- **低S_neg幸存文档**: `doc_id={inversion['bad']['doc_id']}`\n"
                markdown += f"  - Snippet: {inversion['bad']['snippet']}\n"
                markdown += f"  - $S_{{neg\_proj}}$: {inversion['bad']['S_neg_proj']:.4f}\n\n"
                markdown += f"- **差值**: {inversion['diff']:.4f}\n\n"
            else:
                markdown += f"未找到特征倒挂点\n\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    def _generate_query_analysis_json(
        self,
        selected_queries: List[Tuple],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        alpha_0_result: Optional[Dict],
        alpha_5_tau_7_result: Optional[Dict],
        alpha_1_tau_8_result: Optional[Dict],
        q_raw_changed: Dict[str, Tuple[str, str]],
        query_neg_scores: Dict[str, Dict],
        is_dynamic_tau: bool = False
    ) -> Dict[str, Any]:
        """生成JSON格式的分析报告"""
        
        json_output = {
            'statistics': {
                'total_selected_queries': len(selected_queries)
            },
            'queries': []
        }
        
        type_names = {
            'high_noise': '高底噪型',
            'low_noise': '低底噪型',
            'entity_entangled': '实体纠缠型',
            'logical_negation': '纯逻辑否定型'
        }
        
        for qid, scores, qtype in selected_queries:
            changed_key = f"{qid}-changed"
            raw = q_raw_changed.get(changed_key, ("", ""))
            query_text, _ = raw
            
            neg_info = query_neg_scores.get(qid, {})
            neg_words_list = neg_info.get('neg_words', [])
            q_minus = ', '.join(neg_words_list) if neg_words_list else '[NONE]'
            
            query_data = {
                'qid': qid,
                'type': type_names.get(qtype, qtype),
                'query': query_text,
                'negative_words': q_minus,
                'current_mrr': float(scores['mrr']),
                'avg_neg_score': float(scores['avg_neg_score']),
                'data_group_A': [],
                'data_group_B': [],
                'data_group_C': {},
                'data_group_D': None
            }
            
            fp_docs = self._extract_false_positives(
                qid, candidates, alpha_0_result, neg_info
            )
            query_data['data_group_A'] = fp_docs[:3]
            
            fn_docs = self._extract_false_negatives(
                qid, candidates, all_results, neg_info
            )
            query_data['data_group_B'] = fn_docs[:3]
            
            if alpha_1_tau_8_result:
                c_docs = self._extract_optimal_residual_errors(
                    qid, candidates, alpha_1_tau_8_result, all_results
                )
                query_data['data_group_C'] = c_docs
            
            inversion = self._extract_feature_inversions(
                qid, candidates, all_results, neg_info
            )
            query_data['data_group_D'] = inversion
            
            json_output['queries'].append(query_data)
        
        return json_output
    
    def _extract_false_positives(
        self,
        qid: str,
        candidates: Dict[str, Any],
        alpha_0_result: Optional[Dict],
        neg_info: Dict
    ) -> List[Dict[str, Any]]:
        """提取假阳性文档（数据组A）"""
        
        if not alpha_0_result:
            return []
        
        results_changed = alpha_0_result.get('results_changed', {})
        
        changed_key = f"{qid}-changed"
        if changed_key not in results_changed:
            return []
        
        fp_docs = []
        
        for doc_id, score in results_changed[changed_key][:10]:
            doc = candidates.get(doc_id, {})
            doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
            
            neg_words = neg_info.get('neg_words', [])
            neg_word_found = None
            
            for neg_word in neg_words:
                if neg_word.lower() in doc_text.lower():
                    neg_word_found = neg_word
                    break
            
            if neg_word_found and neg_word_found in neg_info.get('doc_scores', {}):
                doc_score_info = neg_info['doc_scores'][doc_id]
                
                fp_docs.append({
                    'doc_id': doc_id,
                    'snippet': doc_text[:150],
                    'S_pos': float(score),
                    'S_neg_proj': float(doc_score_info['score']),
                    'neg_word': neg_word_found
                })
        
        return fp_docs
    
    def _extract_false_negatives(
        self,
        qid: str,
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        neg_info: Dict
    ) -> List[Dict[str, Any]]:
        """提取高惩罚文档（数据组B）- 在α=0.0时排名靠前但在α=5.0时被严重惩罚的文档"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        # 兼容动态 τ 模式
        if 'tau' in all_results[0]:
            alpha_5_result = next((r for r in all_results if r['alpha'] == 5.0 and abs(r['tau'] - 0.7) < 0.01), None)
        else:
            alpha_5_result = next((r for r in all_results if r['alpha'] == 5.0 and abs(r.get('delta', 0) - 0.0) < 0.01), None)
        
        if not alpha_0_result or not alpha_5_result:
            return []
        
        results_0 = alpha_0_result.get('results_changed', {}).get(f"{qid}-changed", [])
        results_5 = alpha_5_result.get('results_changed', {}).get(f"{qid}-changed", [])
        
        fn_docs = []
        
        for i, (doc_id, score_0) in enumerate(results_0[:10]):
            rank_0 = i + 1
            
            rank_5 = None
            score_5 = None
            for j, (d_id, s) in enumerate(results_5):
                if d_id == doc_id:
                    rank_5 = j + 1
                    score_5 = s
                    break
            
            if rank_5 and rank_5 > 50:
                doc = candidates.get(doc_id, {})
                doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                
                neg_word = neg_info.get('doc_scores', {}).get(doc_id, {}).get('neg_word', '')
                
                penalty = abs(score_0 - score_5) if score_5 else 0
                
                fn_docs.append({
                    'doc_id': doc_id,
                    'snippet': doc_text[:150],
                    'S_pos': float(score_0),
                    'S_neg_proj': float(score_5),
                    'penalty': float(penalty),
                    'original_rank': rank_0,
                    'current_rank': rank_5
                })
        
        return fn_docs
    
    def _extract_optimal_residual_errors(
        self,
        qid: str,
        candidates: Dict[str, Any],
        alpha_1_tau_8_result: Dict,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """提取最优参数下的残留误差（数据组C）"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        
        if not alpha_0_result:
            return {}
        
        results_optimal = alpha_1_tau_8_result.get('results_changed', {}).get(f"{qid}-changed", [])
        results_0 = alpha_0_result.get('results_changed', {}).get(f"{qid}-changed", [])
        
        best_fp = None
        worst_fn = None
        
        for i, (doc_id, score) in enumerate(results_optimal[:20]):
            if doc_id in results_0:
                rank_0 = next((j for j, (d, _) in enumerate(results_0) if d == doc_id), None)
                
                if rank_0 is not None and rank_0 < 10:
                    doc = candidates.get(doc_id, {})
                    doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                    
                    best_fp = {
                        'doc_id': doc_id,
                        'S_pos': float(score),
                        'S_neg_proj': 0.0,
                        'rank': i + 1
                    }
                    break
        
        for i, (doc_id, score) in enumerate(results_optimal):
            if doc_id in results_0:
                rank_0 = next((j for j, (d, _) in enumerate(results_0) if d == doc_id), None)
                
                if rank_0 is not None and rank_0 < 10:
                    if worst_fn is None or i > worst_fn.get('rank', 0):
                        worst_fn = {
                            'doc_id': doc_id,
                            'S_pos': float(score),
                            'S_neg_proj': 0.0,
                            'rank': i + 1
                        }
        
        return {
            'false_positive': best_fp,
            'false_negative': worst_fn
        }
    
    def _extract_feature_inversions(
        self,
        qid: str,
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        neg_info: Dict
    ) -> Optional[Dict[str, Any]]:
        """提取特征倒挂点（数据组D）"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        
        if not alpha_0_result:
            return None
        
        results_changed = alpha_0_result.get('results_changed', {}).get(f"{qid}-changed", [])
        
        doc_neg_scores = neg_info.get('doc_scores', {})
        
        relevant_docs = []
        irrelevant_docs = []
        
        for doc_id, score in results_changed:
            if doc_id in doc_neg_scores:
                doc_score_info = doc_neg_scores[doc_id]
                
                relevant_docs.append({
                    'doc_id': doc_id,
                    'S_neg_proj': doc_score_info['score']
                })
        
        for doc_id, score in results_changed[:100]:
            if doc_id not in doc_neg_scores:
                doc = candidates.get(doc_id, {})
                doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                
                irrelevant_docs.append({
                    'doc_id': doc_id,
                    'S_neg_proj': float(score),  # 使用实际的 S_neg_proj 得分
                    'snippet': doc_text[:150]
                })
        
        if not relevant_docs or not irrelevant_docs:
            return None
        
        relevant_docs_sorted = sorted(relevant_docs, key=lambda x: x['S_neg_proj'], reverse=True)
        
        for rel_doc in relevant_docs_sorted[:10]:
            rel_doc['snippet'] = candidates.get(rel_doc['doc_id'], {}).get('text', '')[:150] if isinstance(candidates.get(rel_doc['doc_id'], {}), dict) else str(candidates.get(rel_doc['doc_id'], ''))[:150]
            
            for irr_doc in irrelevant_docs[:20]:
                if rel_doc['S_neg_proj'] > irr_doc['S_neg_proj']:
                    return {
                        'good': rel_doc,
                        'bad': irr_doc,
                        'diff': float(rel_doc['S_neg_proj'] - irr_doc['S_neg_proj'])
                    }
        
        return None


def run_dsclr_evaluation(
    model_name: str = "BAAI/bge-large-en-v1.5",
    task_name: str = "Core17InstructionRetrieval",
    output_dir: str = "eval/output/dsclr",
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    mlp_model_path: Optional[str] = None,
    mlp_hidden_dim: int = 256,
    lap_model_path: Optional[str] = None,
    alphas: Optional[str] = None,
    taus: Optional[str] = None,
    num_samples: int = 15,
    use_dynamic_tau: bool = False,
    alphas_dynamic: Optional[str] = None,
    deltas: Optional[str] = None,
    sbase_mode: str = "original",
    top_k: int = 0,
    confidence_beta: float = 0.0,
    gap_temperature: float = 0.0,
    max_penalty_ratio: float = 0.0,
    anchor_lambda: float = 0.0,
    anchor_top_k: int = 0,
    preserve_lambda: float = 0.0,
    preserve_top_k: int = 0
) -> Dict[str, Any]:
    """运行 DSCLR 评测的便捷函数
    
    支持四种模式：
    1. DSCLR 基础模式：不提供 mlp_model_path 和 lap_model_path
    2. DSCLR+MLP 模式：只提供 mlp_model_path
    3. DSCLR+LAP 模式：只提供 lap_model_path
    4. DeIR 模式（LAP+MLP）：同时提供 lap_model_path 和 mlp_model_path
    5. 动态 τ 模式：use_dynamic_tau=True，使用 Noise_q + Delta 作为动态阈值
    
    Args:
        top_k: 只对 Top-K 文档应用惩罚（0 表示全部应用）
    """
    engine = DSCLREvaluatorEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        alphas=alphas,
        taus=taus,
        num_samples=num_samples,
        sbase_mode=sbase_mode,
        confidence_beta=confidence_beta,
        gap_temperature=gap_temperature,
        max_penalty_ratio=max_penalty_ratio,
        anchor_lambda=anchor_lambda,
        anchor_top_k=anchor_top_k,
        preserve_lambda=preserve_lambda,
        preserve_top_k=preserve_top_k
    )
    
    # 解析动态 τ 参数
    alphas_dynamic_list = None
    if alphas_dynamic:
        alphas_dynamic_list = [float(x.strip()) for x in alphas_dynamic.split(',')]
    
    deltas_list = None
    if deltas:
        deltas_list = [float(x.strip()) for x in deltas.split(',')]
    
    return engine.run(
        mlp_model_path=mlp_model_path, 
        mlp_hidden_dim=mlp_hidden_dim, 
        lap_model_path=lap_model_path,
        use_dynamic_tau=use_dynamic_tau,
        alphas_dynamic=alphas_dynamic_list,
        deltas=deltas_list,
        top_k=top_k
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DSCLR 评测")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="模型名称")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval", help="任务名称")
    parser.add_argument("--output_dir", type=str, default="eval/output/dsclr", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    parser.add_argument("--use_cache", type=bool, default=True, help="是否使用缓存")
    parser.add_argument("--mlp_model_path", type=str, default=None, help="MLP模型路径 (可选，使用动态MLP推理)")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256, help="MLP隐藏层维度 (默认: 256)")
    parser.add_argument("--lap_model_path", type=str, default=None, help="LAP模型路径 (可选，使用LAP投影负向查询)")
    parser.add_argument("--alphas", type=str, default=None, help="Alpha 搜索范围，逗号分隔 (默认: 0.0,0.5,1.0,2.0,3.0,5.0)")
    parser.add_argument("--taus", type=str, default=None, help="Tau 搜索范围，逗号分隔 (默认: 0.5,0.6,0.7,0.8,0.9,0.95)")
    parser.add_argument("--num_samples", type=int, default=15, help="随机抽样数量 (默认: 15)")
    
    # 动态 τ 参数
    parser.add_argument("--use_dynamic_tau", action="store_true", help="使用动态 τ 模式 (τ_q = Noise_q + Delta)")
    parser.add_argument("--alphas_dynamic", type=str, default=None, help="动态 τ 模式的 Alpha 范围，逗号分隔 (默认: 1.0,2.0,3.0)")
    parser.add_argument("--deltas", type=str, default=None, help="Delta 搜索范围，逗号分隔 (默认: 0.0,0.05,0.10,0.15)")
    parser.add_argument("--sbase_mode", type=str, default="original", choices=["q_plus", "original"], help="S_base 计算模式: q_plus 使用 Q+ 计算, original 使用原始查询计算")
    
    # Top-K 重排参数
    parser.add_argument("--top_k", type=int, default=0, help="只对 Top-K 文档应用惩罚 (0 表示全部应用，推荐: 100)")
    
    # 置信度加权参数
    parser.add_argument("--confidence_beta", type=float, default=0.0, help="置信度加权指数 (0=不加权, >0=S_base高的文档惩罚减轻)")
    
    # 差距加权参数
    parser.add_argument("--gap_temperature", type=float, default=0.0, help="差距加权温度 (0=不加权, >0=S_neg>S_base时惩罚更重, 推荐10~50)")
    
    # 封顶惩罚参数
    parser.add_argument("--max_penalty_ratio", type=float, default=0.0, help="惩罚上限比例 (0=不限制, 0.3=最多降30%%的S_base)")
    parser.add_argument("--anchor_lambda", type=float, default=0.0, help="OG锚点融合强度 (0=关闭, 建议 0.05~0.3)")
    parser.add_argument("--anchor_top_k", type=int, default=0, help="仅对OG Top-K应用锚点 (0=全局融合)")
    parser.add_argument("--preserve_lambda", type=float, default=0.0, help="MAP保留增益强度 (0=关闭, 建议 0.02~0.15)")
    parser.add_argument("--preserve_top_k", type=int, default=0, help="MAP保留增益作用的OG Top-K (0=关闭)")

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results = run_dsclr_evaluation(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        mlp_model_path=args.mlp_model_path,
        mlp_hidden_dim=args.mlp_hidden_dim,
        lap_model_path=args.lap_model_path,
        alphas=args.alphas,
        taus=args.taus,
        num_samples=args.num_samples,
        use_dynamic_tau=args.use_dynamic_tau,
        alphas_dynamic=args.alphas_dynamic,
        deltas=args.deltas,
        sbase_mode=args.sbase_mode,
        top_k=args.top_k,
        confidence_beta=args.confidence_beta,
        gap_temperature=args.gap_temperature,
        max_penalty_ratio=args.max_penalty_ratio,
        anchor_lambda=args.anchor_lambda,
        anchor_top_k=args.anchor_top_k,
        preserve_lambda=args.preserve_lambda,
        preserve_top_k=args.preserve_top_k
    )
    print(f"\n最终 p-MRR: {results['best_metrics'].get('p-MRR', 0):.4f}")
