import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_test_at_risk(model_dir, model_suffix, task_name, dual_queries_path):
    print(f"\n  {task_name}:")
    
    emb_path = f"{model_dir}/{task_name}_{model_suffix}_corpus_embeddings.npy"
    ids_path = f"{model_dir}/{task_name}_{model_suffix}_corpus_ids.json"
    
    if not os.path.exists(emb_path):
        print(f"    No corpus embeddings found")
        return
    
    data = np.load(emb_path, allow_pickle=True).item()
    with open(ids_path) as f:
        doc_ids = json.load(f)
    
    emb_list = []
    for did in doc_ids:
        if did in data:
            emb_list.append(data[did].float().numpy())
    doc_embs = F.normalize(torch.from_numpy(np.stack(emb_list)), p=2, dim=1)
    n_docs = doc_embs.shape[0]
    print(f"    Docs: {n_docs}")
    
    dual_queries = []
    with open(dual_queries_path) as f:
        for line in f:
            dual_queries.append(json.loads(line.strip()))
    
    q_base_texts = [dq["q_base"] for dq in dual_queries]
    q_neg_texts = [dq.get("q_changed_neg", dq.get("q_neg", "")) for dq in dual_queries]
    
    # Check if we have pre-computed query embeddings
    q_emb_dir = model_dir
    q_emb_path = None
    for fname in os.listdir(q_emb_dir):
        if "query" in fname and fname.endswith(".pt"):
            q_emb_path = os.path.join(q_emb_dir, fname)
            break
    
    if q_emb_path is None:
        # Need to encode queries - but we can't use CUDA in sandbox
        # Instead, use the results from previous evaluations
        print(f"    No pre-computed query embeddings, checking evaluation results...")
        
        # Try to find evaluation results with score matrices
        for search_dir in [
            f"results/{model_suffix}/v21_grid/{task_name.replace('InstructionRetrieval', '')}",
            f"results/{model_suffix}/v21_baseline/{task_name.replace('InstructionRetrieval', '')}",
        ]:
            if os.path.exists(search_dir):
                print(f"    Found results dir: {search_dir}")
                break
        return
    
    q_data = torch.load(q_emb_path, map_location='cpu', weights_only=False)
    print(f"    Query emb type: {type(q_data)}")
    if isinstance(q_data, dict):
        print(f"    Keys: {list(q_data.keys())[:10]}")
    
    del doc_embs, data

for model_dir, model_suffix, model_label in [
    ("dataset/FollowIR_test/embeddings/RepLLaMA_reproduced", "RepLLaMA_reproduced", "Repllama"),
    ("dataset/FollowIR_test/embeddings/e5-mistral-7b", "e5-mistral-7b", "Mistral"),
]:
    print(f"\n{'=' * 60}")
    print(f"  {model_label} - Test Set Analysis")
    print(f"{'=' * 60}")
    
    for ds, task in [
        ("Core17", "Core17InstructionRetrieval"),
        ("Robust04", "Robust04InstructionRetrieval"),
        ("News21", "News21InstructionRetrieval"),
    ]:
        dual_path = f"dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_{task}.jsonl"
        try:
            analyze_test_at_risk(model_dir, model_suffix, task, dual_path)
        except Exception as e:
            print(f"  {ds}: Error - {e}")
