import torch
import torch.nn.functional as F
import numpy as np
import json
import os

def load_test_corpus(model_dir, task_name, model_suffix):
    emb_path = f"{model_dir}/{task_name}_{model_suffix}_corpus_embeddings.npy"
    ids_path = f"{model_dir}/{task_name}_{model_suffix}_corpus_ids.json"
    
    data = np.load(emb_path, allow_pickle=True).item()
    with open(ids_path) as f:
        doc_ids = json.load(f)
    
    emb_list = []
    for did in doc_ids:
        if did in data:
            emb_list.append(data[did].float().numpy())
    
    emb_matrix = np.stack(emb_list)
    return torch.from_numpy(emb_matrix), len(doc_ids)

for model_dir_name, model_suffix, model_label in [
    ("dataset/FollowIR_test/embeddings/RepLLaMA_reproduced", "RepLLaMA_reproduced", "Repllama"),
    ("dataset/FollowIR_test/embeddings/e5-mistral-7b", "e5-mistral-7b", "Mistral"),
]:
    print(f"\n{'=' * 60}")
    print(f"  {model_label} - Test Set S_base Distribution")
    print(f"{'=' * 60}")
    
    for ds, task in [
        ("Core17", "Core17InstructionRetrieval"),
        ("Robust04", "Robust04InstructionRetrieval"),
        ("News21", "News21InstructionRetrieval"),
    ]:
        print(f"\n  {ds}:")
        
        try:
            doc_embs, n_docs = load_test_corpus(model_dir_name, task, model_suffix)
            doc_embs = F.normalize(doc_embs.float(), p=2, dim=1)
            print(f"    Docs: {n_docs}, dim: {doc_embs.shape[1]}")
            
            # Load dual queries to get q_base texts
            dual_path = f"dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_{task}.jsonl"
            dual_queries = []
            with open(dual_path) as f:
                for line in f:
                    dual_queries.append(json.loads(line.strip()))
            
            q_base_texts = [dq["q_base"] for dq in dual_queries]
            print(f"    Queries: {len(q_base_texts)}")
            
            # Check if query embeddings are cached
            q_cache_dir = f"results/{model_suffix}/v21_grid/{ds}" if model_label == "Mistral" else f"results/repllama-reproduced/v21_grid/{ds}"
            
            # Try to find cached query embeddings
            q_emb_found = False
            for search_dir in [
                f"results/e5-mistral-7b/v21_grid/{ds}",
                f"results/e5-mistral-7b/v21_baseline/{ds}",
                f"results/repllama-reproduced/v21_grid/{ds}",
                f"results/RepLLaMA_reproduced/v2_grid/{ds}",
            ]:
                metrics_path = f"{search_dir}/metrics_summary.json"
                if os.path.exists(metrics_path):
                    print(f"    Found results at: {search_dir}")
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    if "all_results" in metrics:
                        best = max(metrics["all_results"], key=lambda x: x.get("changed_map", 0))
                        print(f"    Best: alpha={best.get('alpha')}, beta={best.get('beta')}, delta={best.get('delta')}, gamma={best.get('gamma')}, ch_MAP={best.get('changed_map', 0):.4f}")
                    q_emb_found = True
                    break
            
            if not q_emb_found:
                print(f"    No cached results found")
            
            del doc_embs
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
