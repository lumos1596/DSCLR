#!/usr/bin/env python
"""
为 RepLLaMA 编码语料库文档
支持 Core17、Robust04、News21 三个数据集
支持实时保存，防止中断丢失进度
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 支持的数据集配置
DATASET_CONFIG = {
    'Core17InstructionRetrieval': {
        'corpus_ids_file': 'Core17InstructionRetrieval_e5-mistral-7b_corpus_ids.json',
        'num_docs': 19899
    },
    'Robust04InstructionRetrieval': {
        'corpus_ids_file': 'Robust04InstructionRetrieval_e5-mistral-7b_corpus_ids.json',
        'num_docs': 528155  # 大约数量
    },
    'News21InstructionRetrieval': {
        'corpus_ids_file': 'News21InstructionRetrieval_e5-mistral-7b_corpus_ids.json',
        'num_docs': 595037  # 大约数量
    }
}


def encode_corpus_repllama_with_checkpoint(
    dataset_name: str = 'Core17InstructionRetrieval',
    base_model_path: str = "/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf",
    adapter_path: str = "/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage",
    device: str = "cuda:0",
    batch_size: int = 16,
    output_dir: str = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/repllama-v1-7b",
    data_dir: str = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test",
    checkpoint_interval: int = 1000
):
    """编码 RepLLaMA 语料库文档，支持实时检查点保存"""
    
    if dataset_name not in DATASET_CONFIG:
        print(f"❌ 不支持的数据集: {dataset_name}")
        print(f"支持的数据集: {list(DATASET_CONFIG.keys())}")
        return None
    
    config = DATASET_CONFIG[dataset_name]
    
    print("="*60)
    print(f"RepLLaMA 语料库文档编码 - {dataset_name}")
    print("="*60)
    print(f"基础模型: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"设备: {device}")
    print(f"批次: {batch_size}")
    print(f"检查点间隔: 每 {checkpoint_interval} 个文档")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件名使用数据集名称
    final_cache_path = os.path.join(output_dir, f"{dataset_name}_repllama_corpus.pt")
    checkpoint_path = os.path.join(output_dir, f"{dataset_name}_repllama_corpus_checkpoint.pt")
    progress_path = os.path.join(output_dir, f"{dataset_name}_repllama_corpus_progress.json")
    
    # 检查是否已有完成的缓存
    if os.path.exists(final_cache_path):
        print(f"⚠️ 完整缓存已存在: {final_cache_path}")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("使用已有缓存，退出")
            return final_cache_path
    
    # 检查是否有检查点可以恢复
    start_idx = 0
    all_embeddings = []
    doc_ids = None
    
    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        print(f"\n🔄 发现检查点，尝试恢复...")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            all_embeddings = [checkpoint['embeddings']]
            doc_ids = checkpoint['doc_ids']
            
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                start_idx = progress['last_processed_idx'] + 1
            
            print(f"✅ 恢复成功！从第 {start_idx} 个文档继续")
            print(f"   已处理: {start_idx} / {len(doc_ids)}")
        except Exception as e:
            print(f"⚠️ 恢复失败: {e}")
            print("   从头开始...")
            all_embeddings = []
            start_idx = 0
    
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
    
    # 加载语料库文档内容
    if doc_ids is None:
        print(f"\n加载 {dataset_name} 语料库...")
        corpus_ids_path = os.path.join(data_dir, "embeddings", "e5-mistral-7b", config['corpus_ids_file'])
        
        if not os.path.exists(corpus_ids_path):
            print(f"❌ 未找到 corpus IDs 文件: {corpus_ids_path}")
            return None
        
        with open(corpus_ids_path, 'r') as f:
            doc_ids = json.load(f)
        
        print(f"文档数量: {len(doc_ids)}")
        
        # 加载文档内容
        corpus_texts = []
        train_corpus_path = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/corpus.jsonl"
        
        if os.path.exists(train_corpus_path):
            print(f"从训练数据加载 corpus: {train_corpus_path}")
            corpus_dict = {}
            with open(train_corpus_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    corpus_dict[item['_id']] = item['text']
            
            missing_count = 0
            for doc_id in doc_ids:
                if doc_id in corpus_dict:
                    corpus_texts.append(corpus_dict[doc_id])
                else:
                    missing_count += 1
                    corpus_texts.append("")
            
            if missing_count > 0:
                print(f"⚠️ 缺少 {missing_count} 个文档的内容")
        else:
            print(f"⚠️ 未找到 corpus 文件，使用 doc_id 作为文本")
            corpus_texts = doc_ids
    else:
        # 恢复模式：重新加载文档内容
        print("\n重新加载文档内容...")
        corpus_texts = []
        train_corpus_path = "/home/luwa/Documents/DSCLR/dataset/FollowIR_train/corpus.jsonl"
        
        if os.path.exists(train_corpus_path):
            corpus_dict = {}
            with open(train_corpus_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    corpus_dict[item['_id']] = item['text']
            
            for doc_id in doc_ids:
                if doc_id in corpus_dict:
                    corpus_texts.append(corpus_dict[doc_id])
                else:
                    corpus_texts.append("")
    
    # 编码文档
    total_docs = len(corpus_texts)
    remaining_docs = corpus_texts[start_idx:]
    
    if remaining_docs:
        print(f"\n编码文档 ({start_idx}/{total_docs} ~ {total_docs})...")
        
        current_idx = start_idx
        batch_embeddings = []
        
        for i in tqdm(range(0, len(remaining_docs), batch_size), desc="Documents"):
            batch_texts = remaining_docs[i:i + batch_size]
            
            # 编码当前批次
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            batch_embeddings.append(embeddings.cpu())
            current_idx += len(batch_texts)
            
            # 定期保存检查点
            if current_idx % checkpoint_interval == 0 or current_idx >= total_docs:
                print(f"\n💾 保存检查点 (已处理 {current_idx}/{total_docs})...")
                
                if all_embeddings:
                    prev_embeddings = torch.cat(all_embeddings, dim=0)
                    curr_embeddings = torch.cat(batch_embeddings, dim=0)
                    combined_embeddings = torch.cat([prev_embeddings, curr_embeddings], dim=0)
                else:
                    combined_embeddings = torch.cat(batch_embeddings, dim=0)
                
                torch.save({
                    'embeddings': combined_embeddings,
                    'doc_ids': doc_ids,
                    'model_name': 'repllama-v1-7b-lora-passage',
                    'embed_dim': combined_embeddings.shape[1]
                }, checkpoint_path)
                
                with open(progress_path, 'w') as f:
                    json.dump({
                        'last_processed_idx': current_idx - 1,
                        'total_docs': total_docs,
                        'dataset': dataset_name
                    }, f)
                
                print(f"   检查点已保存")
                
                all_embeddings = [combined_embeddings]
                batch_embeddings = []
        
        # 处理剩余 embeddings
        if batch_embeddings:
            if all_embeddings:
                prev_embeddings = torch.cat(all_embeddings, dim=0)
                curr_embeddings = torch.cat(batch_embeddings, dim=0)
                final_embeddings = torch.cat([prev_embeddings, curr_embeddings], dim=0)
            else:
                final_embeddings = torch.cat(batch_embeddings, dim=0)
        else:
            final_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.zeros(0, 4096)
    else:
        final_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.zeros(0, 4096)
    
    print(f"\n✅ 编码完成！")
    print(f"   总文档数: {len(doc_ids)}")
    print(f"   Embeddings shape: {final_embeddings.shape}")
    
    # 保存最终缓存
    print("\n保存最终缓存...")
    torch.save({
        'documents': final_embeddings,
        'doc_ids': doc_ids,
        'model_name': 'repllama-v1-7b-lora-passage',
        'embed_dim': final_embeddings.shape[1]
    }, final_cache_path)
    
    print(f"✅ 最终缓存已保存: {final_cache_path}")
    print(f"   文件大小: {os.path.getsize(final_cache_path) / 1024 / 1024:.2f} MB")
    
    # 清理检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print("   已清理检查点文件")
    
    return final_cache_path


def main():
    parser = argparse.ArgumentParser(description='编码 RepLLaMA 语料库文档（带实时检查点）')
    parser.add_argument('--dataset', type=str, default='Core17InstructionRetrieval',
                        choices=['Core17InstructionRetrieval', 'Robust04InstructionRetrieval', 'News21InstructionRetrieval'],
                        help='数据集名称')
    parser.add_argument('--base_model_path', type=str,
                        default='/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf')
    parser.add_argument('--adapter_path', type=str,
                        default='/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str,
                        default='/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/repllama-v1-7b')
    parser.add_argument('--data_dir', type=str,
                        default='/home/luwa/Documents/DSCLR/dataset/FollowIR_test')
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    
    args = parser.parse_args()
    
    encode_corpus_repllama_with_checkpoint(
        dataset_name=args.dataset,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()
