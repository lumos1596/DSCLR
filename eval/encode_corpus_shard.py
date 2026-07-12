"""
Sharded corpus encoder for multi-GPU parallel encoding.

Each instance encodes a shard of the corpus on a specific GPU.
Results are saved as individual shard files that can be merged later.

Usage:
  # GPU 0, shard 0/4:
  CUDA_VISIBLE_DEVICES=0 python -m eval.encode_corpus_shard \
    --dataset msmarco --shard 0 --num_shards 4 \
    --batch_size 64 --cache_dir dataset/BEIR/embeddings/msmarco_full

  # GPU 1, shard 1/4:
  CUDA_VISIBLE_DEVICES=1 python -m eval.encode_corpus_shard \
    --dataset msmarco --shard 1 --num_shards 4 \
    --batch_size 64 --cache_dir dataset/BEIR/embeddings/msmarco_full

Then merge:
  python -m eval.encode_corpus_shard --merge --num_shards 4 \
    --cache_dir dataset/BEIR/embeddings/msmarco_full
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import json
import logging
import argparse
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

import datasets

logger = logging.getLogger(__name__)

BEIR_DATASET_MAP = {
    "nq": "BeIR/nq", "hotpotqa": "BeIR/hotpotqa", "quora": "BeIR/quora",
    "fiqa": "BeIR/fiqa", "arguana": "BeIR/arguana", "scidocs": "BeIR/scidocs",
    "scifact": "BeIR/scifact", "nfcorpus": "BeIR/nfcorpus", "trec-covid": "BeIR/trec-covid",
    "msmarco": "BeIR/msmarco", "fever": "BeIR/fever", "climate-fever": "BeIR/climate-fever",
    "dbpedia-entity": "BeIR/dbpedia-entity", "webis-touche2020": "BeIR/webis-touche2020",
}


def load_corpus(dataset_name: str) -> Dict[str, Dict[str, str]]:
    full_name = BEIR_DATASET_MAP.get(dataset_name, dataset_name)
    logger.info(f"Loading corpus from {full_name}...")
    ds = datasets.load_dataset(full_name, "corpus", split="corpus")
    corpus = {}
    for d in tqdm(ds, desc="Loading corpus"):
        doc_id = str(d["_id"])
        title = str(d.get("title", ""))
        text = str(d.get("text", ""))
        if title and title != "None":
            full_text = f"{title} {text}"
        else:
            full_text = text
        corpus[doc_id] = {"text": full_text}
    logger.info(f"Loaded {len(corpus)} documents")
    return corpus


def encode_shard(args):
    cache_dir = args.cache_dir
    shard_dir = os.path.join(cache_dir, f"shard_{args.shard}")
    meta_path = os.path.join(shard_dir, "meta.json")
    os.makedirs(shard_dir, exist_ok=True)

    # Check if already completed
    final_path = os.path.join(cache_dir, f"shard_{args.shard}_embeddings.pt")
    if os.path.exists(final_path):
        logger.info(f"Shard {args.shard} already encoded at {final_path}")
        return

    # Load corpus
    corpus = load_corpus(args.dataset)
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did]["text"] for did in doc_ids]

    # Select shard
    total = len(doc_ids)
    shard_size = total // args.num_shards
    remainder = total % args.num_shards
    start = args.shard * shard_size + min(args.shard, remainder)
    end = start + shard_size + (1 if args.shard < remainder else 0)
    logger.info(f"Shard {args.shard}/{args.num_shards}: docs [{start}, {end}) = {end - start} documents")

    shard_ids = doc_ids[start:end]
    shard_texts = doc_texts[start:end]

    del corpus  # Free memory
    import gc
    gc.collect()

    # Check for checkpoint resume
    processed_count = 0
    shard_info_list = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            processed_count = int(meta.get("processed_count", 0))
            shard_info_list = meta.get("shards", [])
            logger.info(f"Resuming from {processed_count} docs")
        except Exception:
            processed_count = 0

    # Load encoder
    from eval.models import ModelFactory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_kwargs = {
        "model_name": args.model_name,
        "device": device,
        "batch_size": args.batch_size,
        "normalize_embeddings": True,
    }
    if args.max_seq_length:
        encoder_kwargs["max_seq_length"] = args.max_seq_length
    encoder = ModelFactory.create(**encoder_kwargs)

    # Encode with checkpointing
    shard_buffers = []
    shard_start_idx = processed_count
    batches_in_shard = 0
    checkpoint_interval = args.checkpoint_interval

    logger.info(f"Encoding {len(shard_texts)} docs (starting from {processed_count})...")
    start_time = time.time()

    for i in tqdm(range(processed_count, len(shard_texts), args.batch_size),
                  desc=f"Shard {args.shard}", initial=processed_count, total=len(shard_texts)):
        batch = shard_texts[i:i + args.batch_size]
        batch_emb = encoder.encode_documents(batch, batch_size=args.batch_size)
        if batch_emb.dim() == 2:
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
        shard_buffers.append(batch_emb.cpu())
        batches_in_shard += 1

        if batches_in_shard >= checkpoint_interval:
            shard_end_idx = min(i + args.batch_size, len(shard_texts))
            shard_tensor = torch.cat(shard_buffers, dim=0)
            chunk_path = os.path.join(shard_dir, f"chunk_{shard_start_idx}_{shard_end_idx}.pt")
            torch.save(shard_tensor, chunk_path)
            shard_info_list.append({"start": shard_start_idx, "end": shard_end_idx, "path": chunk_path})
            processed_count = shard_end_idx
            shard_buffers = []
            batches_in_shard = 0
            shard_start_idx = processed_count

            with open(meta_path, "w") as f:
                json.dump({"processed_count": processed_count, "shards": shard_info_list}, f)

            elapsed = time.time() - start_time
            speed = processed_count / elapsed
            eta = (len(shard_texts) - processed_count) / speed if speed > 0 else 0
            logger.info(f"Checkpoint: {processed_count}/{len(shard_texts)} docs, "
                        f"speed={speed:.1f} docs/s, ETA={eta/3600:.1f}h")

    # Save remaining
    if shard_buffers:
        shard_end_idx = len(shard_texts)
        shard_tensor = torch.cat(shard_buffers, dim=0)
        chunk_path = os.path.join(shard_dir, f"chunk_{shard_start_idx}_{shard_end_idx}.pt")
        torch.save(shard_tensor, chunk_path)
        shard_info_list.append({"start": shard_start_idx, "end": shard_end_idx, "path": chunk_path})
        with open(meta_path, "w") as f:
            json.dump({"processed_count": len(shard_texts), "shards": shard_info_list}, f)

    # Consolidate chunks into single file
    logger.info(f"Consolidating {len(shard_info_list)} chunks...")
    chunks_sorted = sorted(shard_info_list, key=lambda x: x["start"])
    embeddings_list = [torch.load(c["path"], map_location="cpu") for c in chunks_sorted]
    embeddings = torch.cat(embeddings_list, dim=0)

    torch.save({"doc_ids": shard_ids, "embeddings": embeddings}, final_path)
    logger.info(f"Shard {args.shard} complete: {embeddings.shape} saved to {final_path}")

    elapsed = time.time() - start_time
    speed = len(shard_texts) / elapsed
    logger.info(f"Total: {elapsed/3600:.1f}h, {speed:.1f} docs/s")


def merge_shards(args):
    cache_dir = args.cache_dir
    dataset_short = args.dataset.split("/")[-1]
    final_path = os.path.join(cache_dir, f"{dataset_short}_full_corpus.pt")

    if os.path.exists(final_path):
        logger.info(f"Merged file already exists: {final_path}")
        return

    all_doc_ids = []
    all_embeddings = []

    for shard_idx in range(args.num_shards):
        shard_path = os.path.join(cache_dir, f"shard_{shard_idx}_embeddings.pt")
        if not os.path.exists(shard_path):
            logger.error(f"Shard {shard_idx} not found: {shard_path}")
            return
        logger.info(f"Loading shard {shard_idx}...")
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        all_doc_ids.extend(data["doc_ids"])
        all_embeddings.append(data["embeddings"])
        logger.info(f"  Shard {shard_idx}: {len(data['doc_ids'])} docs, shape={data['embeddings'].shape}")

    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save({"doc_ids": all_doc_ids, "embeddings": embeddings}, final_path)
    logger.info(f"Merged: {len(all_doc_ids)} docs, shape={embeddings.shape}, saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="msmarco")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=4, help="Total number of shards")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Batches per checkpoint save")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--merge", action="store_true", help="Merge all shards into one file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.merge:
        merge_shards(args)
    else:
        encode_shard(args)


if __name__ == "__main__":
    main()
