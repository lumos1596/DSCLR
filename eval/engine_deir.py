"""
DeIR 双流检索引擎
实现 Dual-Stream Explicit Intent Ranking 的双流打分逻辑
支持 LAP + 升级 MLP 架构的动态推理
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
import csv
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
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
    else:
        model_subdir = model_name.split("/")[-1].replace("-", "_")
    
    return os.path.join(base_cache_dir, model_subdir)


def get_model_name_short(model_name: str) -> str:
    """从模型全名获取短名称用于缓存"""
    if "mistral" in model_name.lower():
        return "e5-mistral-7b"
    elif "bge" in model_name.lower():
        return "bge-large-en"
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


class DeIRDenseRetriever:
    """DeIR 双流稠密检索器"""

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
        计算基础得分矩阵（仅 S_pos，用于 OG 查询）
        返回: S_pos
        """
        # 确保在同一设备上
        device = self.doc_embeddings.device
        if q_plus_embeddings.device != device:
            q_plus_embeddings = q_plus_embeddings.to(device)
        
        # S_pos: [num_queries, num_docs]
        S_pos = torch.matmul(q_plus_embeddings, self.doc_embeddings.T)
        return S_pos

    def compute_scores_matrix(
        self,
        q_plus_embeddings: torch.Tensor,
        q_minus_proj_embeddings: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算得分矩阵（向量化，用于 Changed 查询）
        返回: (S_pos, S_neg_proj)
        """
        # 调试设备信息
        logger.info(f"   [compute_scores_matrix] q_plus: {q_plus_embeddings.device}, doc_emb: {self.doc_embeddings.device}")
        
        # 确保所有张量在同一设备上（使用 doc_embeddings 的设备作为目标）
        device = self.doc_embeddings.device
        if q_plus_embeddings.device != device:
            logger.warning(f"   设备不匹配！将查询 embeddings 转移到 {device}")
            q_plus_embeddings = q_plus_embeddings.to(device)
            q_minus_proj_embeddings = q_minus_proj_embeddings.to(device)
            neg_mask = neg_mask.to(device)
        
        # 确保数据类型一致（doc_embeddings 可能是 float16）
        doc_dtype = self.doc_embeddings.dtype
        if q_plus_embeddings.dtype != doc_dtype:
            q_plus_embeddings = q_plus_embeddings.to(doc_dtype)
        if q_minus_proj_embeddings.dtype != doc_dtype:
            q_minus_proj_embeddings = q_minus_proj_embeddings.to(doc_dtype)
        
        # 文档已在索引时归一化，查询也已归一化
        # S_pos: [num_queries, num_docs]
        S_pos = torch.matmul(q_plus_embeddings, self.doc_embeddings.T)

        # S_neg_proj: [num_queries, num_docs]
        S_neg_proj = torch.matmul(q_minus_proj_embeddings, self.doc_embeddings.T)

        # 应用 mask（将 [NONE] 的负向得分置零）
        S_neg_proj = S_neg_proj * neg_mask.unsqueeze(1)

        return S_pos, S_neg_proj

    def compute_deir_scores(
        self,
        S_pos: torch.Tensor,
        S_neg_proj: torch.Tensor,
        alpha: float,
        tau: float
    ) -> torch.Tensor:
        """
        计算 DeIR 最终得分（静态版本）
        S_final = S_pos - alpha * ReLU(S_neg_proj - tau)
        """
        # ReLU 惩罚项
        penalty = torch.relu(S_neg_proj - tau)

        # 最终得分
        S_final = S_pos - alpha * penalty

        return S_final


class DeIREvaluatorEngine:
    """DeIR 评测引擎（支持 LAP + 升级 MLP）"""

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
        use_cache: bool = True
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

        # 网格搜索参数空间（用于对比模式）
        self.alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        self.taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
        # 随机抽取15组参数进行测试
        all_combinations = [(a, t) for a in self.alphas for t in self.taus]
        self.param_combinations = random.sample(all_combinations, min(15, len(all_combinations)))
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
            cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4"
        )

        # 创建检索器
        self.retriever = DeIRDenseRetriever(self.encoder, self.device, self.batch_size)

        # 加载数据
        from eval.engine import FollowIRDataLoader
        self.data_loader = FollowIRDataLoader(self.task_name)

        # 初始化 LAP 和 MLP (用于动态推理)
        self.lap = None
        self.mlp = None
        self.use_deir = False
        
        # 用于保存分析数据
        self.analysis_data = []

        logger.info(f"✅ DeIR 评测引擎初始化完成")
        logger.info(f"   模型: {self.model_name}")
        logger.info(f"   任务: {self.task_name}")
        logger.info(f"   查询重构: LLM API (实时解耦)")

    def compute_deir_scores_dynamic(
        self,
        S_pos: torch.Tensor,
        S_neg_proj: torch.Tensor,
        q_plus_embeddings: torch.Tensor,
        q_minus_proj_embeddings: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 DeIR 最终得分（动态 MLP 版本）
        
        使用升级后的 MLP，输入为 q_plus_emb 和 q_neg_proj
        """
        if self.mlp is None:
            raise RuntimeError("MLP model not loaded!")
        if self.lap is None:
            raise RuntimeError("LAP model not loaded!")
        
        # 确保 embeddings 在正确的设备上
        device = next(self.mlp.parameters()).device
        if q_plus_embeddings.device != device:
            q_plus_embeddings = q_plus_embeddings.to(device)
        if q_minus_proj_embeddings.device != device:
            q_minus_proj_embeddings = q_minus_proj_embeddings.to(device)
        
        # 转换为 float32 以匹配 MLP 权重
        q_plus_fp32 = q_plus_embeddings.float()
        q_minus_proj_fp32 = q_minus_proj_embeddings.float()
        
        # 使用升级后的 MLP（输入 q_pos_emb 和 q_neg_proj）
        alpha, tau = self.mlp(q_plus_fp32, q_minus_proj_fp32)
        
        # 扩展维度用于计算
        alpha_expanded = alpha.unsqueeze(1)
        tau_expanded = tau.unsqueeze(1)
        neg_mask_expanded = neg_mask.unsqueeze(1)
        
        # 计算惩罚项（统一使用线性 ReLU）
        penalty = torch.relu(S_neg_proj - tau_expanded)
        
        # 应用绝对值扣分，使用 neg_mask 保护 [NONE] 查询
        # 如果 neg_mask=0（即 [NONE]），则惩罚为 0，S_final = S_pos
        S_final = S_pos - alpha_expanded * penalty * neg_mask_expanded

        return S_final, alpha, tau

    def run(
        self, 
        lap_model_path: Optional[str] = None,
        mlp_model_path: Optional[str] = None, 
        mlp_hidden_dim: int = 256,
        save_analysis: bool = True
    ) -> Dict[str, Any]:
        """运行 DeIR 评测流程
        
        Args:
            lap_model_path: LAP 模型路径
            mlp_model_path: MLP 模型路径
            mlp_hidden_dim: MLP隐藏层维度 (默认: 256)
            save_analysis: 是否保存 alpha/tau 分析数据
        """
        logger.info("=" * 60)
        logger.info("🚀 开始 DeIR 评测")
        logger.info("=" * 60)

        start_time = time.time()

        # 初始化 LAP 和 MLP
        if lap_model_path and mlp_model_path:
            logger.info(f"🧠 加载 LAP 模型: {lap_model_path}")
            logger.info(f"🧠 加载 MLP 模型: {mlp_model_path}")
            
            from model.lap_module import LAPProjection
            from model.dsclr_mlp import DSCLR_MLP
            
            # 根据模型名称确定嵌入维度
            if "mistral" in self.model_name.lower() or "repllama" in self.model_name.lower():
                embed_dim = 4096
                logger.info(f"   检测到 {self.model_name} 模型，使用嵌入维度: {embed_dim}")
            else:
                embed_dim = 1024
                logger.info(f"   使用默认嵌入维度: {embed_dim}")
            
            # 加载 LAP
            self.lap = LAPProjection(hidden_dim=embed_dim).to(self.device)
            lap_checkpoint = torch.load(lap_model_path, map_location=self.device)
            self.lap.load_state_dict(lap_checkpoint['lap_state_dict'])
            self.lap.eval()
            logger.info(f"✅ LAP 模型加载成功")
            
            # 加载 MLP（升级后的版本，输入为 embed_dim + 1）
            # 升级后的 MLP 输入: [q_neg_proj, correlation]
            self.mlp = DSCLR_MLP(hidden_dim=embed_dim).to(self.device)
            mlp_checkpoint = torch.load(mlp_model_path, map_location=self.device)
            self.mlp.load_state_dict(mlp_checkpoint['mlp_state_dict'])
            self.mlp.eval()
            self.use_deir = True
            logger.info(f"✅ MLP 模型加载成功 (hidden_dim={mlp_hidden_dim})，进入 DeIR 动态推理模式")
        else:
            self.use_deir = False
            logger.info("🔬 使用静态网格搜索模式（未提供 LAP/MLP 模型）")

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
                
                # 直接使用缓存中的顺序
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
        q_plus_list_changed, q_minus_list_changed, neg_mask_changed, query_ids_changed = self._prepare_dual_queries(q_changed, q_raw_changed)

        # 编码查询
        logger.info("🔍 编码 OG 查询 (仅 Q+)...")
        q_plus_embeddings_og = self._encode_queries(q_plus_list_og)
        
        logger.info("🔍 编码 Changed 查询 (Q+ 和 Q-)...")
        q_plus_embeddings_changed = self._encode_queries(q_plus_list_changed)
        q_minus_embeddings_changed = self._encode_queries(q_minus_list_changed)

        # 【DeIR 关键步骤】使用 LAP 投影负向查询
        if self.use_deir:
            logger.info("🔄 使用 LAP 投影负向查询...")
            with torch.no_grad():
                # 确保 embeddings 在正确的设备上
                if q_minus_embeddings_changed.device != self.device:
                    q_minus_embeddings_changed = q_minus_embeddings_changed.to(self.device)
                q_minus_proj_changed = self.lap(q_minus_embeddings_changed)
        else:
            # 不使用 LAP 时，直接使用原始 q_minus
            q_minus_proj_changed = q_minus_embeddings_changed

        # 计算 og 得分矩阵 (仅 S_pos，无 S_neg)
        logger.info("📊 计算 og 得分矩阵 (仅 S_pos)...")
        S_pos_og = self.retriever.compute_base_scores(q_plus_embeddings_og)
        
        # 计算 changed 得分矩阵 (S_pos 和 S_neg_proj)
        logger.info("📊 计算 changed 得分矩阵...")
        S_pos_changed, S_neg_proj_changed = self.retriever.compute_scores_matrix(
            q_plus_embeddings_changed, q_minus_proj_changed, neg_mask_changed
        )

        # DeIR 动态推理 或 静态网格搜索
        if self.use_deir:
            logger.info("🧠 使用 DeIR 动态 MLP 进行推理...")
            with torch.no_grad():
                # 【物理隔离】OG 查询直接使用 S_pos，不经过任何 MLP 惩罚！
                S_final_og = S_pos_og
                pred_alpha_og = torch.zeros(len(query_ids_og), device=self.device)
                pred_tau_og = torch.zeros(len(query_ids_og), device=self.device)
                
                # 只有 Changed 查询才进入动态门控计算
                S_final_changed, pred_alpha_changed, pred_tau_changed = self.compute_deir_scores_dynamic(
                    S_pos_changed, S_neg_proj_changed, 
                    q_plus_embeddings_changed, q_minus_proj_changed,
                    neg_mask_changed
                )
            
            logger.info(f"   OG 查询: 物理隔离，直接使用 S_pos (无惩罚)")
            logger.info(f"   Changed 查询: 动态预测 avg_alpha={pred_alpha_changed.mean().item():.4f}, avg_tau={pred_tau_changed.mean().item():.4f}")
            
            # 保存分析数据
            if save_analysis:
                self._save_analysis_data(
                    query_ids_og, q_raw_og, pred_alpha_og, pred_tau_og,
                    query_ids_changed, q_raw_changed, pred_alpha_changed, pred_tau_changed
                )
                
                # 生成诊断报告（仅针对 changed 查询）
                self._generate_diagnostic_report(
                    query_ids_changed,
                    S_pos_changed,
                    S_neg_proj_changed,
                    S_final_changed,
                    pred_alpha_changed,
                    pred_tau_changed,
                    candidates,
                    neg_mask_changed
                )
            
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
                'mode': 'deir_dynamic',
                'p-MRR': best_metrics.get('p-MRR', 0),
                'og_nDCG@5': best_metrics.get('original', {}).get('ndcg_at_5', 0),
                'changed_nDCG@5': best_metrics.get('changed', {}).get('ndcg_at_5', 0),
                'metrics': best_metrics
            }]
        else:
            # 静态网格搜索（用于对比）
            logger.info(f"🔬 随机搜索: {len(self.param_combinations)} 组参数")
            best_metrics = None
            best_params = None
            best_results_og = None
            best_results_changed = None
            all_results = []
            all_query_metrics = []

            for alpha, tau in self.param_combinations:
                # 【物理隔离】OG 查询直接使用 S_pos，不经过任何惩罚！
                S_final_og = S_pos_og
                # 只有 Changed 查询才应用 DeIR 惩罚
                S_final_changed = self.retriever.compute_deir_scores(S_pos_changed, S_neg_proj_changed, alpha, tau)

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
                    'metrics': metrics
                })

                # 选择最佳参数：综合考虑 p-MRR、og_nDCG 和 changed_nDCG
                if best_metrics is None:
                    best_metrics = metrics
                    best_params = (alpha, tau)
                    best_composite_score = p_mrr + og_ndcg + changed_ndcg
                    best_results_og = results_og
                    best_results_changed = results_changed
                else:
                    current_composite_score = p_mrr + og_ndcg + changed_ndcg
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
        if self.use_deir:
            logger.info("🧠 DeIR 动态 MLP 推理结果:")
        else:
            logger.info("📊 DeIR 随机搜索结果:")
            logger.info(f"   最佳参数: α={best_params[0]}, τ={best_params[1]}")
        og_metrics = best_metrics.get('original', {})
        changed_metrics = best_metrics.get('changed', {})
        logger.info(f"   p-MRR: {best_metrics.get('p-MRR', 0):.4f}")
        logger.info(f"   OG - nDCG@1: {og_metrics.get('ndcg_at_1', 0):.4f}, nDCG@5: {og_metrics.get('ndcg_at_5', 0):.4f}, nDCG@10: {og_metrics.get('ndcg_at_10', 0):.4f}")
        logger.info(f"   Changed - nDCG@1: {changed_metrics.get('ndcg_at_1', 0):.4f}, nDCG@5: {changed_metrics.get('ndcg_at_5', 0):.4f}, nDCG@10: {changed_metrics.get('ndcg_at_10', 0):.4f}")
        logger.info(f"   耗时: {elapsed_time:.1f}秒")
        logger.info("=" * 60)

        # 保存结构化汇总文件
        self._save_structured_summary(best_metrics, all_query_metrics, q_raw_og, q_raw_changed)
        
        # 保存完整结果
        self._save_results(all_results, best_params, best_metrics)
        
        # 生成坏例分析
        self._generate_bad_case_analysis(all_query_metrics, q_raw_og, q_raw_changed)

        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'all_results': all_results,
            'all_query_metrics': all_query_metrics,
            'elapsed_time': elapsed_time
        }

    def _save_analysis_data(
        self,
        query_ids_og: List[str],
        q_raw_og: Dict[str, Tuple[str, str]],
        pred_alpha_og: torch.Tensor,
        pred_tau_og: torch.Tensor,
        query_ids_changed: List[str],
        q_raw_changed: Dict[str, Tuple[str, str]],
        pred_alpha_changed: torch.Tensor,
        pred_tau_changed: torch.Tensor
    ) -> None:
        """保存 alpha/tau 分析数据到 JSON 和 CSV"""
        analysis_data = []
        
        # OG 查询（alpha=0, tau=0）
        for i, qid in enumerate(query_ids_og):
            base_qid = qid.replace('-og', '')
            raw_query, instruction = q_raw_og.get(qid, ("", ""))
            analysis_data.append({
                'qid': base_qid,
                'query_type': 'og',
                'query_text': raw_query,
                'instruction': instruction,
                'alpha': float(pred_alpha_og[i].item()),
                'tau': float(pred_tau_og[i].item()),
                'is_none': False
            })
        
        # Changed 查询
        for i, qid in enumerate(query_ids_changed):
            base_qid = qid.replace('-changed', '')
            raw_query, instruction = q_raw_changed.get(qid, ("", ""))
            
            # 检查是否为 [NONE] 查询
            is_none = instruction.strip() == '[NONE]'
            
            analysis_data.append({
                'qid': base_qid,
                'query_type': 'changed',
                'query_text': raw_query,
                'instruction': instruction,
                'alpha': float(pred_alpha_changed[i].item()),
                'tau': float(pred_tau_changed[i].item()),
                'is_none': is_none
            })
        
        # 保存为 JSON
        json_path = os.path.join(self.output_dir, "alpha_tau_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Alpha/Tau 分析数据已保存 (JSON): {json_path}")
        
        # 保存为 CSV
        csv_path = os.path.join(self.output_dir, "alpha_tau_analysis.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['qid', 'query_type', 'query_text', 'instruction', 'alpha', 'tau', 'is_none'])
            writer.writeheader()
            writer.writerows(analysis_data)
        logger.info(f"💾 Alpha/Tau 分析数据已保存 (CSV): {csv_path}")
        
        # 统计信息
        changed_alphas = [d['alpha'] for d in analysis_data if d['query_type'] == 'changed' and not d['is_none']]
        changed_taus = [d['tau'] for d in analysis_data if d['query_type'] == 'changed' and not d['is_none']]
        
        if changed_alphas:
            logger.info(f"📊 Changed 查询统计 (排除 [NONE]):")
            logger.info(f"   Alpha - 均值: {np.mean(changed_alphas):.4f}, 标准差: {np.std(changed_alphas):.4f}, 范围: [{min(changed_alphas):.4f}, {max(changed_alphas):.4f}]")
            logger.info(f"   Tau   - 均值: {np.mean(changed_taus):.4f}, 标准差: {np.std(changed_taus):.4f}, 范围: [{min(changed_taus):.4f}, {max(changed_taus):.4f}]")

    def _generate_diagnostic_report(
        self,
        query_ids_changed: List[str],
        S_pos_changed: torch.Tensor,
        S_neg_proj_changed: torch.Tensor,
        S_final_changed: torch.Tensor,
        pred_alpha_changed: torch.Tensor,
        pred_tau_changed: torch.Tensor,
        candidates: Dict[str, List[str]],
        neg_mask_changed: torch.Tensor
    ) -> None:
        """
        生成 DeIR 全局分布与误差诊断报告
        
        仅针对 "changed" 组查询，分析：
        1. 正负样本的得分分布
        2. 惩罚触发统计
        3. 网络输出参数的统计学特征
        """
        logger.info("🔍 生成 DeIR 诊断报告...")
        
        # 加载 qrels 获取真实标签
        qrels = self.data_loader.metrics_loader.load_qrels()
        
        # 收集所有样本的统计信息
        pos_samples_s_pos = []
        pos_samples_s_neg = []
        neg_samples_s_pos = []
        neg_samples_s_neg = []
        
        # 惩罚触发统计
        penalty_triggered_in_pos = 0
        total_pos_samples = 0
        penalty_values = []
        
        # 遍历每个 changed 查询
        for i, qid in enumerate(query_ids_changed):
            base_qid = qid.replace('-changed', '')
            # candidates 使用 base_qid（不带后缀）
            doc_ids = candidates.get(base_qid, [])
            
            if not doc_ids:
                logger.warning(f"   查询 {qid} (base: {base_qid}) 没有找到 candidates")
                continue
            
            # 获取该查询的 qrels
            # 对于 changed 查询，使用对应的 og 查询的 qrels（因为 qrels 中只有 -og 后缀的键）
            og_qid = f"{base_qid}-og"
            query_qrels = qrels.get(og_qid, {})
            
            # 获取该查询的得分和参数
            s_pos_i = S_pos_changed[i, :len(doc_ids)].cpu().numpy()
            s_neg_i = S_neg_proj_changed[i, :len(doc_ids)].cpu().numpy()
            alpha_i = pred_alpha_changed[i].item()
            tau_i = pred_tau_changed[i].item()
            
            # 遍历每个文档
            for j, doc_id in enumerate(doc_ids):
                if j >= len(s_pos_i):
                    break
                    
                relevance = query_qrels.get(doc_id, 0)
                s_pos_val = s_pos_i[j]
                s_neg_val = s_neg_i[j]
                
                # 计算惩罚
                penalty = max(0, s_neg_val - tau_i)
                
                if relevance > 0:
                    # 正样本
                    pos_samples_s_pos.append(s_pos_val)
                    pos_samples_s_neg.append(s_neg_val)
                    total_pos_samples += 1
                    
                    # 检查是否触发惩罚（误伤）
                    if s_neg_val > tau_i:
                        penalty_triggered_in_pos += 1
                        penalty_values.append(penalty)
                else:
                    # 负样本
                    neg_samples_s_pos.append(s_pos_val)
                    neg_samples_s_neg.append(s_neg_val)
                    
                    # 记录所有触发惩罚的样本
                    if s_neg_val > tau_i:
                        penalty_values.append(penalty)
        
        # 计算统计信息
        def compute_stats(values):
            if not values:
                return {"mean": 0, "min": 0, "max": 0, "p25": 0, "p50": 0, "p75": 0}
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75))
            }
        
        # 1. 正负样本的得分分布
        score_distribution = {
            "positive_samples": {
                "count": len(pos_samples_s_pos),
                "S_pos": compute_stats(pos_samples_s_pos),
                "S_neg_proj": compute_stats(pos_samples_s_neg)
            },
            "negative_samples": {
                "count": len(neg_samples_s_pos),
                "S_pos": compute_stats(neg_samples_s_pos),
                "S_neg_proj": compute_stats(neg_samples_s_neg)
            }
        }
        
        # 2. 惩罚触发统计
        false_positive_rate = (penalty_triggered_in_pos / total_pos_samples * 100) if total_pos_samples > 0 else 0
        avg_penalty = np.mean(penalty_values) if penalty_values else 0
        
        penalty_stats = {
            "total_positive_samples": total_pos_samples,
            "penalty_triggered_in_positive": penalty_triggered_in_pos,
            "false_positive_rate_percent": float(false_positive_rate),
            "avg_penalty_value": float(avg_penalty),
            "total_penalty_triggered": len(penalty_values)
        }
        
        # 3. 网络输出参数的统计学特征
        alpha_values = pred_alpha_changed.cpu().numpy()
        tau_values = pred_tau_changed.cpu().numpy()
        
        network_params = {
            "alpha": compute_stats(alpha_values.tolist()),
            "tau": compute_stats(tau_values.tolist())
        }
        
        # 构建完整报告
        report = {
            "task": self.task_name,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "num_changed_queries": len(query_ids_changed),
            "score_distribution": score_distribution,
            "penalty_activation_stats": penalty_stats,
            "network_parameters": network_params
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "deir_diagnostic_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 DeIR 诊断报告已保存: {report_path}")
        
        # 打印关键发现
        logger.info("📊 诊断报告关键发现:")
        logger.info(f"   正样本数: {len(pos_samples_s_pos)}, 负样本数: {len(neg_samples_s_pos)}")
        logger.info(f"   正样本 S_pos 均值: {score_distribution['positive_samples']['S_pos']['mean']:.4f}")
        logger.info(f"   正样本 S_neg_proj 均值: {score_distribution['positive_samples']['S_neg_proj']['mean']:.4f}")
        logger.info(f"   负样本 S_pos 均值: {score_distribution['negative_samples']['S_pos']['mean']:.4f}")
        logger.info(f"   负样本 S_neg_proj 均值: {score_distribution['negative_samples']['S_neg_proj']['mean']:.4f}")
        logger.info(f"   误伤率 (正样本触发惩罚): {false_positive_rate:.2f}%")
        logger.info(f"   平均惩罚值: {avg_penalty:.4f}")
        logger.info(f"   Alpha - 中位数: {network_params['alpha']['p50']:.4f}, P75: {network_params['alpha']['p75']:.4f}")
        logger.info(f"   Tau - 中位数: {network_params['tau']['p50']:.4f}, P75: {network_params['tau']['p75']:.4f}")

    def _get_all_candidate_doc_ids(self, candidates: Dict[str, List[str]]) -> List[str]:
        """获取所有候选文档ID（去重）"""
        all_doc_ids = set()
        for doc_ids in candidates.values():
            all_doc_ids.update(doc_ids)
        return list(all_doc_ids)

    def _prepare_single_queries(
        self,
        q_og: Dict[str, str],
        q_raw_og: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """准备 OG 查询（仅提取 Q+）"""
        q_plus_list = []
        query_ids = []
        
        for qid, query_text in q_og.items():
            # 从 qid 提取 idx (格式: "1-og" -> idx=1)
            try:
                idx = int(qid.split('-')[0])
            except:
                idx = 0
            
            # 获取原始查询和指令
            raw_query, instruction = q_raw_og.get(qid, (query_text, ""))
            
            # 使用 reformulator 进行实时解耦 (带缓存)
            q_plus, _ = self.reformulator.reformulate(
                qid=qid,
                idx=idx,
                query=raw_query,
                instruction=instruction,
                query_type="og"
            )
            
            query_ids.append(qid)
            q_plus_list.append(q_plus)
        
        return q_plus_list, query_ids

    def _prepare_dual_queries(
        self,
        q_changed: Dict[str, str],
        q_raw_changed: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str], torch.Tensor, List[str]]:
        """准备 Changed 查询（提取 Q+ 和 Q-）"""
        q_plus_list = []
        q_minus_list = []
        query_ids = []
        
        for qid, query_text in q_changed.items():
            # 从 qid 提取 idx (格式: "1-og" -> idx=1)
            try:
                idx = int(qid.split('-')[0])
            except:
                idx = 0
            
            # 确定 query_type
            query_type = "og" if qid.endswith("-og") else "changed"
            
            # 获取原始查询和指令
            raw_query, instruction = q_raw_changed.get(qid, (query_text, ""))
            
            # 使用 reformulator 进行实时解耦 (带缓存)
            q_plus, q_minus = self.reformulator.reformulate(
                qid=qid,
                idx=idx,
                query=raw_query,
                instruction=instruction,
                query_type=query_type
            )
            
            query_ids.append(qid)
            q_plus_list.append(q_plus)
            q_minus_list.append(q_minus)
        
        # 【调试逻辑】强制 OG 查询跳过 MLP，即使 Q- 不为 None
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

        return q_plus_list, q_minus_list, neg_mask, query_ids

    def _get_bulletproof_mask(self, q_minus: str) -> float:
        """获取防弹 mask（处理 [NONE] 情况）"""
        if q_minus is None or q_minus == "[NONE]":
            return 0.0
        if isinstance(q_minus, str) and q_minus.strip().upper() == "[NONE]":
            return 0.0
        return 1.0

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
        top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """从得分矩阵提取检索结果"""
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

            # 取 top-k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            results[qid] = dict(sorted_docs[:top_k])

        return results

    def _compute_per_query_metrics(
        self,
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        candidates: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """计算每个查询的详细指标"""
        all_metrics = []
        
        # 合并所有查询结果
        all_results = {**results_og, **results_changed}
        
        # 获取所有唯一的 base_qid
        all_base_qids = set()
        for qid in query_ids_og + query_ids_changed:
            base_qid = qid.replace('-og', '').replace('-changed', '')
            all_base_qids.add(base_qid)
        
        # 加载 qrels
        # 加载 qrels - 直接使用 data_loader 中的 metrics_loader
        qrels_og = self.data_loader.metrics_loader.load_qrels()
        qrels_changed = qrels_og  # changed 查询使用相同的 qrels
        
        # 为每个查询计算指标
        for base_qid in all_base_qids:
            og_key = f"{base_qid}-og"
            changed_key = f"{base_qid}-changed"
            
            # 计算不同 k 值的指标
            for k in [1, 3, 5, 10]:
                # OG 查询指标 - qrels 的键是带 -og 后缀的
                if og_key in results_og:
                    og_metrics = self._compute_single_query_metrics(
                        results_og[og_key], qrels_og.get(og_key, {}), k
                    )
                    all_metrics.append({
                        'qid': base_qid,
                        'query_type': 'og',
                        'k': k,
                        **og_metrics
                    })
                
                # Changed 查询指标 - qrels 的键是带 -changed 后缀的
                if changed_key in results_changed:
                    changed_metrics = self._compute_single_query_metrics(
                        results_changed[changed_key], qrels_changed.get(changed_key, {}), k
                    )
                    all_metrics.append({
                        'qid': base_qid,
                        'query_type': 'changed',
                        'k': k,
                        **changed_metrics
                    })
        
        return all_metrics

    def _compute_single_query_metrics(
        self,
        results: Dict[str, float],
        qrels: Dict[str, int],
        k: int
    ) -> Dict[str, float]:
        """计算单个查询的指标"""
        # 取 top-k
        sorted_docs = sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # 计算 DCG
        dcg = 0.0
        for i, (doc_id, _) in enumerate(sorted_docs):
            rel = qrels.get(doc_id, 0)
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # 计算 ideal DCG
        ideal_rels = sorted(qrels.values(), reverse=True)[:k]
        ideal_dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        # nDCG
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        # MRR
        mrr = 0.0
        for i, (doc_id, _) in enumerate(sorted_docs):
            if qrels.get(doc_id, 0) > 0:
                mrr = 1.0 / (i + 1)
                break
        
        # MAP
        num_relevant = sum(1 for rel in qrels.values() if rel > 0)
        if num_relevant > 0:
            ap = 0.0
            num_rel_so_far = 0
            for i, (doc_id, _) in enumerate(sorted_docs):
                if qrels.get(doc_id, 0) > 0:
                    num_rel_so_far += 1
                    precision_at_i = num_rel_so_far / (i + 1)
                    ap += precision_at_i
            map_score = ap / min(num_relevant, k)
        else:
            map_score = 0.0
        
        return {
            'ndcg': ndcg,
            'mrr': mrr,
            'map': map_score
        }

    def _save_results(
        self,
        all_results: List[Dict],
        best_params,
        best_metrics: Dict
    ) -> None:
        """保存评测结果"""
        # 根据模式选择文件名
        if self.use_deir:
            results_path = os.path.join(self.output_dir, "deir_dynamic_results.json")
        else:
            results_path = os.path.join(self.output_dir, "random_search_results.json")
        
        # 构建可序列化的结果
        serializable_results = []
        for r in all_results:
            sr = {
                'p-MRR': r.get('p-MRR', 0),
                'og_nDCG@5': r.get('og_nDCG@5', 0),
                'changed_nDCG@5': r.get('changed_nDCG@5', 0),
            }
            if 'alpha' in r:
                sr['alpha'] = r['alpha']
                sr['tau'] = r['tau']
            if 'mode' in r:
                sr['mode'] = r['mode']
            serializable_results.append(sr)
        
        output = {
            'best_params': {'alpha': best_params[0], 'tau': best_params[1]} if isinstance(best_params, tuple) else best_params,
            'best_metrics': best_metrics,
            'all_results': serializable_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 评测结果已保存: {results_path}")

    def _save_structured_summary(
        self,
        best_metrics: Dict,
        all_query_metrics: List[Dict],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]]
    ) -> None:
        """保存结构化的指标汇总"""
        # 计算完整指标
        full_scores = {'og': {}, 'changed': {}}
        
        for query_type in ['og', 'changed']:
            metrics_list = [m for m in all_query_metrics if m['query_type'] == query_type]
            
            for k in [1, 3, 5, 10, 100, 1000]:
                k_metrics = [m for m in metrics_list if m['k'] == k]
                if k_metrics:
                    full_scores[query_type][f'ndcg_at_{k}'] = float(np.mean([m['ndcg'] for m in k_metrics]))
                    full_scores[query_type][f'mrr_at_{k}'] = float(np.mean([m['mrr'] for m in k_metrics]))
                    full_scores[query_type][f'map_at_{k}'] = float(np.mean([m['map'] for m in k_metrics]))
        
        summary = {
            'p-MRR': best_metrics.get('p-MRR', 0),
            'og': {
                'nDCG@1': full_scores.get('og', {}).get('ndcg_at_1', 0),
                'nDCG@3': full_scores.get('og', {}).get('ndcg_at_3', 0),
                'nDCG@5': full_scores.get('og', {}).get('ndcg_at_5', 0),
                'nDCG@10': full_scores.get('og', {}).get('ndcg_at_10', 0),
                'nDCG@100': full_scores.get('og', {}).get('ndcg_at_100', 0),
                'nDCG@1000': full_scores.get('og', {}).get('ndcg_at_1000', 0),
                'MRR1': full_scores.get('og', {}).get('mrr_at_1', 0),
                'MRR3': full_scores.get('og', {}).get('mrr_at_3', 0),
                'MRR5': full_scores.get('og', {}).get('mrr_at_5', 0),
                'MRR10': full_scores.get('og', {}).get('mrr_at_10', 0),
                'MRR100': full_scores.get('og', {}).get('mrr_at_100', 0),
                'MRR1000': full_scores.get('og', {}).get('mrr_at_1000', 0),
                'MAP1': full_scores.get('og', {}).get('map_at_1', 0),
                'MAP3': full_scores.get('og', {}).get('map_at_3', 0),
                'MAP5': full_scores.get('og', {}).get('map_at_5', 0),
                'MAP10': full_scores.get('og', {}).get('map_at_10', 0),
                'MAP100': full_scores.get('og', {}).get('map_at_100', 0),
                'MAP1000': full_scores.get('og', {}).get('map_at_1000', 0),
            },
            'changed': {
                'nDCG@1': full_scores.get('changed', {}).get('ndcg_at_1', 0),
                'nDCG@3': full_scores.get('changed', {}).get('ndcg_at_3', 0),
                'nDCG@5': full_scores.get('changed', {}).get('ndcg_at_5', 0),
                'nDCG@10': full_scores.get('changed', {}).get('ndcg_at_10', 0),
                'nDCG@100': full_scores.get('changed', {}).get('ndcg_at_100', 0),
                'nDCG@1000': full_scores.get('changed', {}).get('ndcg_at_1000', 0),
                'MRR1': full_scores.get('changed', {}).get('mrr_at_1', 0),
                'MRR3': full_scores.get('changed', {}).get('mrr_at_3', 0),
                'MRR5': full_scores.get('changed', {}).get('mrr_at_5', 0),
                'MRR10': full_scores.get('changed', {}).get('mrr_at_10', 0),
                'MRR100': full_scores.get('changed', {}).get('mrr_at_100', 0),
                'MRR1000': full_scores.get('changed', {}).get('mrr_at_1000', 0),
                'MAP1': full_scores.get('changed', {}).get('map_at_1', 0),
                'MAP3': full_scores.get('changed', {}).get('map_at_3', 0),
                'MAP5': full_scores.get('changed', {}).get('map_at_5', 0),
                'MAP10': full_scores.get('changed', {}).get('map_at_10', 0),
                'MAP100': full_scores.get('changed', {}).get('map_at_100', 0),
                'MAP1000': full_scores.get('changed', {}).get('map_at_1000', 0),
            },
            'best_params': best_metrics.get('best_params', {}),
            'task': self.task_name,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 结构化指标汇总已保存: {summary_path}")

    def _generate_bad_case_analysis(
        self,
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]]
    ) -> None:
        """生成坏例分析诊断报告"""
        # 提取 k=5 时的 p-MRR（使用 MRR 作为代理）
        og_metrics = [m for m in all_query_metrics if m['query_type'] == 'og' and m['k'] == 5]
        
        if not og_metrics:
            logger.warning("⚠️ 无法生成坏例分析：没有足够的查询指标数据")
            return
        
        mrr_values = [m['mrr'] for m in og_metrics]
        mean_mrr = np.mean(mrr_values)
        std_mrr = np.std(mrr_values)
        
        # 筛选显著低于平均的样本（低于 mean - std）
        low_performance = [m for m in og_metrics if m['mrr'] < mean_mrr - std_mrr]
        # 筛选显著高于平均的样本（高于 mean + std）
        high_performance = [m for m in og_metrics if m['mrr'] > mean_mrr + std_mrr]
        
        # 构建坏例分析报告
        report = {
            'statistics': {
                'total_queries': len(og_metrics),
                'mean_mrr': float(mean_mrr),
                'std_mrr': float(std_mrr),
                'low_performance_count': len(low_performance),
                'high_performance_count': len(high_performance),
            },
            'low_performance_samples': [],
            'high_performance_samples': [],
            'comparison_analysis': {}
        }
        
        # 添加低性能样本详情
        for m in low_performance[:10]:  # 最多10个
            qid = m['qid']
            # q_raw_og 的键是完整格式，如 "310-og"
            og_key = f"{qid}-og"
            raw = q_raw_og.get(og_key, ("", ""))
            
            sample = {
                'qid': qid,
                'mrr': float(m['mrr']),
                'ndcg': float(m['ndcg']),
                'map': float(m['map']),
                'query': raw[0] if raw else "",
                'instruction': raw[1] if raw else "",
            }
            report['low_performance_samples'].append(sample)
        
        # 添加高性能样本详情
        for m in high_performance[:10]:  # 最多10个
            qid = m['qid']
            # q_raw_og 的键是完整格式，如 "310-og"
            og_key = f"{qid}-og"
            raw = q_raw_og.get(og_key, ("", ""))
            
            sample = {
                'qid': qid,
                'mrr': float(m['mrr']),
                'ndcg': float(m['ndcg']),
                'map': float(m['map']),
                'query': raw[0] if raw else "",
                'instruction': raw[1] if raw else "",
            }
            report['high_performance_samples'].append(sample)
        
        # 对比分析
        low_query_lens = [len(s['query'].split()) for s in report['low_performance_samples']]
        high_query_lens = [len(s['query'].split()) for s in report['high_performance_samples']]
        
        low_instr_lens = [len(s['instruction'].split()) for s in report['low_performance_samples']]
        high_instr_lens = [len(s['instruction'].split()) for s in report['high_performance_samples']]
        
        report['comparison_analysis'] = {
            'avg_query_length_low': float(np.mean(low_query_lens)) if low_query_lens else 0,
            'avg_query_length_high': float(np.mean(high_query_lens)) if high_query_lens else 0,
            'avg_instruction_length_low': float(np.mean(low_instr_lens)) if low_instr_lens else 0,
            'avg_instruction_length_high': float(np.mean(high_instr_lens)) if high_instr_lens else 0,
            'key_findings': []
        }
        
        # 自动生成分析结论
        if report['comparison_analysis']['avg_query_length_high'] > report['comparison_analysis']['avg_query_length_low']:
            report['comparison_analysis']['key_findings'].append(
                "高性能样本倾向于有更长的查询文本"
            )
        else:
            report['comparison_analysis']['key_findings'].append(
                "低性能样本倾向于有更长的查询文本"
            )
        
        if report['comparison_analysis']['avg_instruction_length_high'] > report['comparison_analysis']['avg_instruction_length_low']:
            report['comparison_analysis']['key_findings'].append(
                "高性能样本倾向于有更长的指令文本"
            )
        else:
            report['comparison_analysis']['key_findings'].append(
                "低性能样本倾向于有更长的指令文本"
            )
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "bad_case_analysis.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 坏例分析报告已保存: {report_path}")


def run_deir_evaluation(
    model_name: str = "BAAI/bge-large-en-v1.5",
    task_name: str = "Core17InstructionRetrieval",
    output_dir: str = "eval/output/deir",
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    lap_model_path: Optional[str] = None,
    mlp_model_path: Optional[str] = None,
    mlp_hidden_dim: int = 256,
    save_analysis: bool = True
) -> Dict[str, Any]:
    """运行 DeIR 评测的便捷函数"""
    engine = DeIREvaluatorEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    return engine.run(
        lap_model_path=lap_model_path,
        mlp_model_path=mlp_model_path,
        mlp_hidden_dim=mlp_hidden_dim,
        save_analysis=save_analysis
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeIR 评测")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="模型名称")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval", help="任务名称")
    parser.add_argument("--output_dir", type=str, default="eval/output/deir", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    parser.add_argument("--use_cache", type=bool, default=True, help="是否使用缓存")
    parser.add_argument("--lap_model_path", type=str, required=True, help="LAP模型路径")
    parser.add_argument("--mlp_model_path", type=str, required=True, help="MLP模型路径")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256, help="MLP隐藏层维度 (默认: 256)")
    parser.add_argument("--save_analysis", type=bool, default=True, help="是否保存 alpha/tau 分析数据")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results = run_deir_evaluation(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        lap_model_path=args.lap_model_path,
        mlp_model_path=args.mlp_model_path,
        mlp_hidden_dim=args.mlp_hidden_dim,
        save_analysis=args.save_analysis
    )
    print(f"\n最终 p-MRR: {results['best_metrics'].get('p-MRR', 0):.4f}")
