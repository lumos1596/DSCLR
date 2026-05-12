"""
LAP (Lightweight Asymmetric Projection) 训练脚本

核心设计原则：
1. 底座编码器 100% 冻结（requires_grad=False）
2. 仅优化 LAP 模块的投影矩阵参数
3. 训练阶段不引入 alpha/tau 惩罚推断公式

训练流程：
1. 加载预训练编码器（BGE / E5-Mistral / RepLLaMA）
2. 冻结编码器所有参数
3. 初始化 LAPProjection 模块
4. 对每个 batch:
   - 编码 q_minus, d_pos, d_neg
   - LAP 投影 q_minus -> q_neg_proj
   - 计算 LAPContrastiveLoss
   - 反向传播（仅更新 LAP 参数）
"""

import os
import sys
import argparse
import json
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.lap_module import LAPProjection
from model.lap_loss import LAPContrastiveLoss
from model.lap_dataset import LAPDataset, lap_collate_fn
from eval.models.e5_mistral_encoder import E5MistralEncoder
from eval.models.repllama_encoder import RepLLaMAEncoder


class LAPEncoderWrapper(nn.Module):
    """
    LAP 编码器包装类
    
    整合底座编码器 + LAP 投影模块
    
    关键：底座编码器完全冻结，仅 LAP 模块可训练
    """
    
    def __init__(
        self,
        encoder,
        hidden_dim: int,
        use_lap: bool = True
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.use_lap = use_lap
        
        # 冻结底座编码器
        self._freeze_encoder()
        
        # 初始化 LAP 模块
        if self.use_lap:
            self.lap = LAPProjection(hidden_dim=hidden_dim)
            print(f"LAP 模块已初始化: hidden_dim={hidden_dim}")
        else:
            self.lap = None
    
    def _freeze_encoder(self):
        """冻结底座编码器所有参数"""
        # 对于 SentenceTransformer
        if hasattr(self.encoder, 'parameters'):
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 对于自定义编码器（E5MistralEncoder / RepLLaMAEncoder）
        if hasattr(self.encoder, 'model'):
            for param in self.encoder.model.parameters():
                param.requires_grad = False
        
        print("✅ 底座编码器已冻结")
    
    def encode_texts(
        self,
        texts: list,
        batch_size: int = 32,
        mode: str = "document"
    ) -> torch.Tensor:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            mode: "query" 或 "document"
        
        Returns:
            embeddings: [N, hidden_dim] 已 L2 归一化
        """
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
                # SentenceTransformer
                batch_emb = self.encoder.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )
            
            if isinstance(batch_emb, np.ndarray):
                batch_emb = torch.from_numpy(batch_emb)
            
            all_embeddings.append(batch_emb.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def project_q_minus(self, q_minus_emb: torch.Tensor) -> torch.Tensor:
        """
        对 q_minus 进行 LAP 投影
        
        Args:
            q_minus_emb: [batch_size, hidden_dim]
        
        Returns:
            q_neg_proj: [batch_size, hidden_dim] 投影后向量
        """
        if self.use_lap and self.lap is not None:
            return self.lap(q_minus_emb)
        return q_minus_emb
    
    def get_trainable_params(self):
        """获取可训练参数（仅 LAP 模块）"""
        if self.use_lap and self.lap is not None:
            return self.lap.parameters()
        return []


def load_encoder(model_name: str, device: str, batch_size: int):
    """
    加载编码器
    
    Args:
        model_name: 模型名称
        device: GPU 设备
        batch_size: 批处理大小
    
    Returns:
        encoder, hidden_dim
    """
    if "mistral" in model_name.lower():
        print("加载 E5-Mistral-7B 编码器...")
        encoder = E5MistralEncoder(
            model_name=model_name,
            device=device,
            batch_size=min(batch_size, 28),
            normalize_embeddings=True
        )
        hidden_dim = 4096
    elif "repllama" in model_name.lower():
        print("加载 RepLLaMA 编码器...")
        encoder = RepLLaMAEncoder(
            model_name=model_name,
            device=device,
            batch_size=min(batch_size, 28),
            normalize_embeddings=True
        )
        hidden_dim = 4096
    else:
        print("加载 BGE 编码器...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, device=device)
        encoder = encoder.half()
        hidden_dim = 1024
    
    return encoder, hidden_dim


def train_epoch(
    model: LAPEncoderWrapper,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: LAPContrastiveLoss,
    device: str,
    encode_batch_size: int
) -> dict:
    """
    训练一个 epoch
    
    Returns:
        metrics: 包含 loss, sim_good_mean, sim_bad_mean 等指标
    """
    model.train()
    total_loss = 0
    total_sim_good = 0
    total_sim_bad = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        q_minus_texts = batch['q_minus_text']
        doc_pos_texts = batch['doc_pos_text']
        doc_neg_texts = batch['doc_neg_text']
        
        # 编码（底座编码器，无需梯度）
        with torch.no_grad():
            q_minus_emb = model.encode_texts(q_minus_texts, batch_size=encode_batch_size, mode="query")
            d_pos_emb = model.encode_texts(doc_pos_texts, batch_size=encode_batch_size, mode="document")
            d_neg_emb = model.encode_texts(doc_neg_texts, batch_size=encode_batch_size, mode="document")
        
        # 移动到 GPU
        q_minus_emb = q_minus_emb.to(device)
        d_pos_emb = d_pos_emb.to(device)
        d_neg_emb = d_neg_emb.to(device)
        
        # LAP 投影
        q_neg_proj = model.project_q_minus(q_minus_emb)
        
        # 计算损失
        loss, sim_good_mean, sim_bad_mean = criterion(q_neg_proj, d_pos_emb, d_neg_emb)
        
        if torch.isnan(loss):
            print("⚠️ Loss is NaN, skipping batch")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_sim_good += sim_good_mean
        total_sim_bad += sim_bad_mean
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'sim_good_mean': total_sim_good / num_batches if num_batches > 0 else 0,
        'sim_bad_mean': total_sim_bad / num_batches if num_batches > 0 else 0
    }


@torch.no_grad()
def validate(
    model: LAPEncoderWrapper,
    dataloader: DataLoader,
    criterion: LAPContrastiveLoss,
    device: str,
    encode_batch_size: int
) -> dict:
    """
    验证
    
    Returns:
        metrics: 包含 loss, sim_good_mean, sim_bad_mean 等指标
    """
    model.eval()
    total_loss = 0
    total_sim_good = 0
    total_sim_bad = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        q_minus_texts = batch['q_minus_text']
        doc_pos_texts = batch['doc_pos_text']
        doc_neg_texts = batch['doc_neg_text']
        
        # 编码
        q_minus_emb = model.encode_texts(q_minus_texts, batch_size=encode_batch_size, mode="query")
        d_pos_emb = model.encode_texts(doc_pos_texts, batch_size=encode_batch_size, mode="document")
        d_neg_emb = model.encode_texts(doc_neg_texts, batch_size=encode_batch_size, mode="document")
        
        q_minus_emb = q_minus_emb.to(device)
        d_pos_emb = d_pos_emb.to(device)
        d_neg_emb = d_neg_emb.to(device)
        
        # LAP 投影
        q_neg_proj = model.project_q_minus(q_minus_emb)
        
        # 计算损失
        loss, sim_good_mean, sim_bad_mean = criterion(q_neg_proj, d_pos_emb, d_neg_emb)
        
        if not torch.isnan(loss):
            total_loss += loss.item()
            total_sim_good += sim_good_mean
            total_sim_bad += sim_bad_mean
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'sim_good_mean': total_sim_good / num_batches if num_batches > 0 else 0,
        'sim_bad_mean': total_sim_bad / num_batches if num_batches > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='LAP Module Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--encode_batch_size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--output_dir', type=str, default='train/output/LAP', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='dataset/FollowIR_train', help='Training data directory')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--model_type', type=str, default='repllama', 
                        choices=['bge', 'mistral', 'repllama'], help='Encoder type')
    parser.add_argument('--contrastive_margin', type=float, default=0.15, 
                        help='Relative contrastive margin (default: 0.15)')
    parser.add_argument('--orthogonal_target', type=float, default=0.65, 
                        help='Orthogonal target for good docs (default: 0.65)')
    parser.add_argument('--num_neg_per_query', type=int, default=1, help='Number of negatives per query')
    args = parser.parse_args()
    
    # 设备
    DEVICE = f"cuda:{args.gpu}"
    
    # 模型名称
    if args.model_type == "mistral":
        MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
        MODEL_SHORT = "e5-mistral-7b"
        HIDDEN_DIM = 4096
    elif args.model_type == "repllama":
        MODEL_NAME = "castorini/repllama-v1-7b-lora-passage"
        MODEL_SHORT = "repllama-v1-7b"
        HIDDEN_DIM = 4096
    else:
        MODEL_NAME = "BAAI/bge-large-en-v1.5"
        MODEL_SHORT = "bge-large-en"
        HIDDEN_DIM = 1024
    
    # 输出目录（使用时间戳命名）
    timestamp = datetime.now().strftime("%m.%d-%H%M")
    OUTPUT_DIR = os.path.join(args.output_dir, f"{timestamp}-{MODEL_SHORT}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("LAP 模块训练配置")
    print("="*60)
    print(f"编码器: {MODEL_NAME}")
    print(f"隐藏维度: {HIDDEN_DIM}")
    print(f"GPU: {args.gpu}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Margin Pos: {args.margin_pos}")
    print(f"Margin Neg: {args.margin_neg}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = LAPDataset(
        data_dir=args.data_dir,
        num_neg_per_query=args.num_neg_per_query
    )
    
    total_size = len(dataset)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lap_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lap_collate_fn
    )
    
    print(f"总数据量: {total_size}")
    print(f"训练集: {train_size}, 验证集: {val_size}")
    
    # 加载编码器
    print("\n加载编码器...")
    encoder, hidden_dim = load_encoder(MODEL_NAME, DEVICE, args.encode_batch_size)
    
    # 创建 LAP 模型
    print("\n初始化 LAP 模型...")
    model = LAPEncoderWrapper(
        encoder=encoder,
        hidden_dim=hidden_dim,
        use_lap=True
    ).to(DEVICE)
    
    # 损失函数 (重构版: 相对锚定 + 软正交损失)
    criterion = LAPContrastiveLoss(
        contrastive_margin=args.contrastive_margin,
        orthogonal_target=args.orthogonal_target
    )
    
    # 优化器（仅 LAP 参数）
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 验证只有 LAP 参数可训练
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"总参数: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"冻结参数: {total_params - trainable_params:,}")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_sim_good': [],
        'train_sim_bad': [],
        'val_loss': [],
        'val_sim_good': [],
        'val_sim_bad': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n开始训练...")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, args.encode_batch_size
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion, DEVICE, args.encode_batch_size
        )
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_sim_good'].append(train_metrics['sim_good_mean'])
        history['train_sim_bad'].append(train_metrics['sim_bad_mean'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_sim_good'].append(val_metrics['sim_good_mean'])
        history['val_sim_bad'].append(val_metrics['sim_bad_mean'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(sim_good: {train_metrics['sim_good_mean']:.4f}, sim_bad: {train_metrics['sim_bad_mean']:.4f})")
        print(f"Val Loss: {val_metrics['loss']:.4f} "
              f"(sim_good: {val_metrics['sim_good_mean']:.4f}, sim_bad: {val_metrics['sim_bad_mean']:.4f})")
        print(f"  → 目标: sim_good < 0.65, sim_bad > 0.80")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # 保存最佳模型
            model_path = os.path.join(OUTPUT_DIR, f"lap_best_{MODEL_SHORT}.pt")
            torch.save({
                'epoch': epoch,
                'lap_state_dict': model.lap.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': {
                    'hidden_dim': hidden_dim,
                    'model_type': args.model_type,
                    'margin_pos': args.margin_pos,
                    'margin_neg': args.margin_neg
                }
            }, model_path)
            print(f"✅ 保存最佳模型: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停: {args.patience} 个 epoch 无改善")
                break
    
    # 保存训练历史
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存: {history_path}")
    
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
