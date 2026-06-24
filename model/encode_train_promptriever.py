"""
Encode training set with Promptriever-LLaMA3 encoder.

Promptriever uses:
- query encoding: "query:  {text}" + EOS pooling
- document encoding: "passage:  {text}" + EOS pooling

This script encodes ALL embeddings (q_base, q_plus, q_minus, pos, neg)
with Promptriever, since document embeddings are encoder-specific.
"""

import json
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.models.repllama_encoder import RepLLaMAEncoder


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Encode training set with Promptriever-LLaMA3")
    parser.add_argument("--model_name", type=str, default="samaya-ai/promptriever-llama3.1-8b-instruct-v1")
    parser.add_argument("--qwen3_model", type=str, default="4B",
                        help="Qwen3 model name for output file naming")
    parser.add_argument("--queries_path", type=str,
                        default="dataset/FollowIR_train/train/distilled_queries_Qwen3-4B_tsc_balanced.jsonl",
                        help="Path to Qwen3-reformulated queries JSONL")
    parser.add_argument("--train_data_path", type=str,
                        default="dataset/FollowIR_train/train/dsclr_total_dataset.jsonl",
                        help="Path to training data JSONL (for pos/neg texts)")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_train/embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Promptriever Encoder: {args.model_name}")
    print(f"Qwen3 Model: {args.qwen3_model}")
    print(f"Queries: {args.queries_path}")
    print(f"Train data: {args.train_data_path}")
    print(f"Device: {args.device}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    output_filename = f"dsclr_train_embeddings_promptriever-llama31_qwen3-{args.qwen3_model}.pt"
    output_path = os.path.join(args.output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}")
        return

    # Step 1: Load Qwen3-reformulated queries
    print("\nStep 1: Load Qwen3-reformulated queries...")
    distilled_queries = load_jsonl(args.queries_path)
    print(f"Loaded {len(distilled_queries)} queries")

    q_base_list = []
    q_plus_list = []
    q_minus_list = []
    for item in distilled_queries:
        q_base_list.append(item['query'])
        output = json.loads(item['output'])
        q_plus_list.append(output['Q_plus'])
        q_minus_list.append(output['Q_minus'])

    print(f"Q_plus sample: {q_plus_list[0][:80]}...")
    print(f"Q_minus sample: {q_minus_list[0]}")

    # Step 2: Load training data for pos/neg texts
    print("\nStep 2: Load training data for pos/neg document texts...")
    train_data = load_jsonl(args.train_data_path)
    print(f"Loaded {len(train_data)} training entries")

    # Collect all unique pos and neg documents
    pos_texts = []
    neg_texts = []
    for item in train_data:
        pos_texts.extend(item['pos'])
        neg_texts.extend(item['neg'])
    print(f"Total pos docs: {len(pos_texts)}, Total neg docs: {len(neg_texts)}")

    # Step 3: Load Promptriever encoder
    print("\nStep 3: Load Promptriever encoder...")
    encoder = RepLLaMAEncoder(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        max_seq_length=args.max_seq_length
    )

    # Step 4: Encode queries
    print("\nStep 4: Encode Q_base...")
    q_base_embeddings = encoder.encode_queries(q_base_list, batch_size=args.batch_size)
    if isinstance(q_base_embeddings, np.ndarray):
        q_base_embeddings = torch.from_numpy(q_base_embeddings)
    print(f"Q_base embeddings: {q_base_embeddings.shape}")

    print("\nStep 5: Encode Q_plus...")
    q_plus_embeddings = encoder.encode_queries(q_plus_list, batch_size=args.batch_size)
    if isinstance(q_plus_embeddings, np.ndarray):
        q_plus_embeddings = torch.from_numpy(q_plus_embeddings)
    print(f"Q_plus embeddings: {q_plus_embeddings.shape}")

    print("\nStep 6: Encode Q_minus...")
    q_minus_embeddings = encoder.encode_queries(q_minus_list, batch_size=args.batch_size)
    if isinstance(q_minus_embeddings, np.ndarray):
        q_minus_embeddings = torch.from_numpy(q_minus_embeddings)
    print(f"Q_minus embeddings: {q_minus_embeddings.shape}")

    # Handle Q_minus = [NONE] samples
    print("\nStep 7: Handle Q_minus = [NONE] samples...")
    none_count = 0
    for i, q_minus_text in enumerate(q_minus_list):
        if q_minus_text == '[NONE]' or not q_minus_text.strip():
            q_minus_embeddings[i] = torch.zeros_like(q_minus_embeddings[i])
            none_count += 1
    print(f"Set {none_count} [NONE] embeddings to zero vectors")

    # Step 5: Encode documents
    print("\nStep 8: Encode positive documents...")
    pos_embeddings = encoder.encode_documents(pos_texts, batch_size=args.batch_size)
    if isinstance(pos_embeddings, np.ndarray):
        pos_embeddings = torch.from_numpy(pos_embeddings)
    print(f"Pos embeddings: {pos_embeddings.shape}")

    print("\nStep 9: Encode negative documents...")
    neg_embeddings = encoder.encode_documents(neg_texts, batch_size=args.batch_size)
    if isinstance(neg_embeddings, np.ndarray):
        neg_embeddings = torch.from_numpy(neg_embeddings)
    print(f"Neg embeddings: {neg_embeddings.shape}")

    # Step 6: Save
    print("\nStep 10: Save embeddings...")
    torch.save({
        'q_base_embeddings': q_base_embeddings,
        'q_plus_embeddings': q_plus_embeddings,
        'q_minus_embeddings': q_minus_embeddings,
        'pos_embeddings': pos_embeddings,
        'neg_embeddings': neg_embeddings,
        'model_name': args.model_name,
        'embed_dim': q_plus_embeddings.shape[1],
        'max_seq_length': args.max_seq_length,
        'reformulator': f'Qwen3-{args.qwen3_model}-TSC_BALANCED',
    }, output_path)

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved: {output_path} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()
