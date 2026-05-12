#!/usr/bin/env python
"""
训练数据准备脚本 - RepLLaMA 专用版本
使用本地模型路径加载
"""

import json
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_jsonl(path):
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def encode_with_encoder(
    texts: List[str],
    encoder: Any,
    batch_size: int,
    desc: str = "Encoding"
) -> torch.Tensor:
    """
    统一的编码接口
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]
        
        # SentenceTransformer 编码器
        batch_embeddings = encoder.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # 确保是 CPU 上的 tensor
        if isinstance(batch_embeddings, np.ndarray):
            batch_embeddings = torch.from_numpy(batch_embeddings)
        elif batch_embeddings.device != torch.device('cpu'):
            batch_embeddings = batch_embeddings.cpu()
            
        all_embeddings.append(batch_embeddings)
    
    return torch.cat(all_embeddings, dim=0)


def prepare_training_data_repllama(
    base_model_path: str = "/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf",
    adapter_path: str = "/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage",
    device: str = "cuda:0",
    batch_size: int = 28,
    output_dir: str = "dataset/FollowIR_train/embeddings",
    data_dir: str = "dataset/FollowIR_train"
):
    """
    准备 RepLLaMA 训练数据缓存
    """
    print("="*60)
    print(f"RepLLaMA 基础模型: {base_model_path}")
    print(f"RepLLaMA Adapter: {adapter_path}")
    print(f"设备: {device}")
    print(f"批次: {batch_size}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_short = "repllama-v1-7b"
    cache_filename = f"dsclr_train_embeddings_{model_short}.pt"
    cache_path = os.path.join(output_dir, cache_filename)
    
    # 检查是否已存在缓存
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
    
    distilled_queries = load_jsonl(os.path.join(data_dir, "distilled_queries.jsonl"))
    print(f"双流查询: {len(distilled_queries)} 条")
    
    q_plus_list = []
    q_minus_list = []
    for item in tqdm(distilled_queries, desc="解析双流查询"):
        output = json.loads(item['output'])
        q_plus_list.append(output['Q_plus'])
        q_minus_list.append(output['Q_minus'])
    
    print("\n" + "="*60)
    print("Step 2: 加载 RepLLaMA 编码器...")
    print("="*60)
    
    # 加载 RepLLaMA 模型 - 使用 PEFT 加载基础模型 + LoRA adapter
    print(f"加载 RepLLaMA 模型...")
    print(f"基础模型: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        print(f"❌ 基础模型路径不存在: {base_model_path}")
        sys.exit(1)
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter 路径不存在: {adapter_path}")
        sys.exit(1)
    
    # 使用 transformers 加载 PEFT 模型 (8-bit 量化)
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    print("\n加载基础模型 (LLaMA-2 7B) - 使用 8-bit 量化...")
    
    # 配置 8-bit 量化
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    base_model = AutoModel.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",  # 自动分配到可用设备
        torch_dtype=torch.float16
    )
    
    print("加载 LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # 合并 adapter 到基础模型
    model.eval()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ 模型加载完成")
    
    # 定义编码函数
    def encode_texts(texts, batch_size=28, max_length=2600):
        """编码文本为向量"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # 获取模型输出
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用 mean pooling
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                
                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                # 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    print("\n" + "="*60)
    print("Step 3: 编码 Q+ 和 Q- ...")
    print("="*60)
    
    print("\n编码 Q+ (正向查询)...")
    q_plus_embeddings = encode_texts(q_plus_list, batch_size=batch_size)
    print(f"Q+ embeddings shape: {q_plus_embeddings.shape}")
    
    print("\n编码 Q- (负向查询)...")
    q_minus_embeddings = encode_texts(q_minus_list, batch_size=batch_size)
    print(f"Q- embeddings shape: {q_minus_embeddings.shape}")
    
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
    pos_embeddings = encode_texts(pos_docs, batch_size=batch_size)
    print(f"Pos embeddings shape: {pos_embeddings.shape}")
    
    print("\n编码负样本文档...")
    neg_embeddings = encode_texts(neg_docs, batch_size=batch_size)
    print(f"Neg embeddings shape: {neg_embeddings.shape}")
    
    print("\n" + "="*60)
    print("Step 5: 保存缓存...")
    print("="*60)
    
    torch.save({
        'q_plus': q_plus_embeddings,
        'q_minus': q_minus_embeddings,
        'pos': pos_embeddings,
        'neg': neg_embeddings,
        'model_name': 'repllama-v1-7b-lora-passage',
        'embed_dim': q_plus_embeddings.shape[1]
    }, cache_path)
    
    print(f"✅ 缓存已保存: {cache_path}")
    print(f"文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")
    
    return cache_path


def main():
    parser = argparse.ArgumentParser(description='准备 RepLLaMA 训练数据缓存')
    parser.add_argument('--base_model_path', type=str, 
                        default='/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf',
                        help='LLaMA-2 7B 基础模型本地路径')
    parser.add_argument('--adapter_path', type=str,
                        default='/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage',
                        help='RepLLaMA adapter 本地路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备')
    parser.add_argument('--batch_size', type=int, default=28, help='批处理大小')
    parser.add_argument('--output_dir', type=str, default='dataset/FollowIR_train/embeddings',
                        help='输出目录')
    parser.add_argument('--data_dir', type=str, default='dataset/FollowIR_train',
                        help='训练数据目录')
    
    args = parser.parse_args()
    
    prepare_training_data_repllama(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()
