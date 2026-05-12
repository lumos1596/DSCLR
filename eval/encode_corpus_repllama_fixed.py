#!/usr/bin/env python
"""
为 RepLLaMA 编码语料库文档 - 修正版
使用正确的 prompt template 和 embedding 提取方式
从 HuggingFace 数据集加载 corpus（而不是训练数据）
支持 Core17、Robust04、News21 三个数据集
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics.evaluator import DataLoader


def encode_corpus_repllama_fixed(
    dataset_name: str = 'Core17InstructionRetrieval',
    base_model_path: str = "/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf",
    adapter_path: str = "/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage",
    device: str = "cuda:0",
    batch_size: int = 32,
    output_dir: str = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/repllama-v1-7b",
):
    """编码 RepLLaMA 语料库文档 - 使用正确的 prompt template"""
    
    print("="*60)
    print(f"RepLLaMA 语料库文档编码 (修正版) - {dataset_name}")
    print("="*60)
    print(f"基础模型: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"设备: {device}")
    print(f"批次: {batch_size}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    final_cache_path = os.path.join(output_dir, f"{dataset_name}_repllama_corpus_fixed.pt")
    
    if os.path.exists(final_cache_path):
        print(f"⚠️ 完整缓存已存在: {final_cache_path}")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("使用已有缓存，退出")
            return final_cache_path
    
    # 加载 RepLLaMA 模型
    print("\n加载 RepLLaMA 模型...")
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    print("加载基础模型 (LLaMA-2 7B)...")
    base_model = AutoModel.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("加载 LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 模型加载完成")
    
    # 从 HuggingFace 数据集加载 corpus（正确的做法）
    print(f"\n从 HuggingFace 数据集加载 {dataset_name} 语料库...")
    data_loader = DataLoader(dataset_name)
    corpus = data_loader.load_corpus()
    
    doc_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
    
    print(f"文档数量: {len(doc_ids)}")
    
    # 编码文档 - 使用正确的 prompt template
    print(f"\n编码文档 (使用 'passage: {{text}}</s>' 格式)...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding passages"):
        batch_texts = corpus_texts[i:i + batch_size]
        
        # 应用 prompt template: passage: {text}</s>
        formatted_texts = [f"passage: {text}</s>" for text in batch_texts]
        
        inputs = tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 提取最后一个 token 的 embedding (RepLLaMA 官方做法)
            batch_embeddings = []
            for j in range(len(batch_texts)):
                seq_len = inputs['attention_mask'][j].sum().item()
                embedding = outputs.last_hidden_state[j, seq_len - 1]
                batch_embeddings.append(embedding)
            
            batch_embeddings = torch.stack(batch_embeddings)
            # L2 归一化
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        all_embeddings.append(batch_embeddings.cpu())
    
    # 合并所有 embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"\n✅ 编码完成！")
    print(f"   总文档数: {len(doc_ids)}")
    print(f"   Embeddings shape: {final_embeddings.shape}")
    
    # 保存最终缓存
    print("\n保存最终缓存...")
    torch.save({
        'documents': final_embeddings,
        'doc_ids': doc_ids,
        'model_name': 'repllama-v1-7b-lora-passage',
        'embed_dim': final_embeddings.shape[1],
        'prompt_template': 'passage: {text}</s>',
        'extraction_method': 'last_token'
    }, final_cache_path)
    
    print(f"✅ 最终缓存已保存: {final_cache_path}")
    print(f"   文件大小: {os.path.getsize(final_cache_path) / 1024 / 1024:.2f} MB")
    
    return final_cache_path


def main():
    parser = argparse.ArgumentParser(description='编码 RepLLaMA 语料库文档（修正版）')
    parser.add_argument('--dataset', type=str, default='Core17InstructionRetrieval',
                        choices=['Core17InstructionRetrieval', 'Robust04InstructionRetrieval', 'News21InstructionRetrieval'])
    parser.add_argument('--base_model_path', type=str,
                        default='/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf')
    parser.add_argument('--adapter_path', type=str,
                        default='/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str,
                        default='/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/repllama-v1-7b')
    
    args = parser.parse_args()
    
    encode_corpus_repllama_fixed(
        dataset_name=args.dataset,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
