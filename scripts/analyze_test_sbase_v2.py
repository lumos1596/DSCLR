import torch
import torch.nn.functional as F
import numpy as np
import json
import os

def load_test_corpus_embeddings(model_dir, task_name):
    emb_path = f"{model_dir}/{task_name}_corpus_embeddings.npy"
    ids_path = f"{model_dir}/{task_name}_corpus_ids.json"
    
    data = np.load(emb_path, allow_pickle=True).item()
    with open(ids_path) as f:
        doc_ids = json.load(f)
    
    emb_list = []
    ordered_ids = []
    for did in doc_ids:
        if did in data:
            emb_list.append(data[did].float().numpy())
            ordered_ids.append(did)
    
    emb_matrix = np.stack(emb_list)
    return torch.from_numpy(emb_matrix), ordered_ids

def load_test_query_embeddings(model_dir, task_name):
    q_emb_path = f"{model_dir}/{task_name}_query_embeddings.pt"
    if os.path.exists(q_emb_path):
        data = torch.load(q_emb_path, map_location='cpu', weights_only=False)
        return data
    return None

for model_dir_name, model_label in [
    ("repllama-reproduced", "Repllama"),
    ("e5-mistral-7b", "Mistral"),
]:
    print(f"\n{'=' * 60}")
    print(f"  {model_label} - Test Set S_base Distribution")
    print(f"{'=' * 60}")
    
    base_dir = f"dataset/FollowIR_test/embeddings/{model_dir_name}"
    
    for ds, task in [
        ("Core17", "Core17InstructionRetrieval"),
        ("Robust04", "Robust04InstructionRetrieval"),
        ("News21", "News21InstructionRetrieval"),
    ]:
        print(f"\n  {ds}:")
        
        try:
            doc_embs, doc_ids = load_test_corpus_embeddings(base_dir, task)
            doc_embs = F.normalize(doc_embs.float(), p=2, dim=1)
            print(f"    Docs: {doc_embs.shape[0]}, dim: {doc_embs.shape[1]}")
            
            q_emb_path = f"{base_dir}/{task}_query_embeddings.pt"
            if os.path.exists(q_emb_path):
                q_data = torch.load(q_emb_path, map_location='cpu', weights_only=False)
                print(f"    Query emb keys: {list(q_data.keys())[:5] if isinstance(q_data, dict) else type(q_data)}")
            else:
                print(f"    No query embeddings found at {q_emb_path}")
                continue
            
            if isinstance(q_data, dict):
                for key in q_data:
                    if isinstance(q_data[key], torch.Tensor):
                        print(f"      {key}: shape={q_data[key].shape}")
                
                if 'q_base_embeddings' in q_data:
                    q_base = F.normalize(q_data['q_base_embeddings'].float(), p=2, dim=1)
                    S_base = q_base @ doc_embs.T
                    all_sbase = S_base.cpu().numpy().flatten()
                    
                    n_queries = S_base.shape[0]
                    n_docs = S_base.shape[1]
                    
                    print(f"    S_base distribution (n_queries={n_queries}, n_docs={n_docs}):")
                    print(f"      mean={all_sbase.mean():.4f}, std={all_sbase.std():.4f}")
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                        print(f"      p{p}={np.percentile(all_sbase, p):.4f}")
                    
                    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                        frac = (all_sbase > thresh).mean()
                        print(f"      S_base > {thresh}: {frac * 100:.2f}%")
                    
                    # Top-100 S_base per query
                    top100 = np.sort(all_sbase.reshape(n_queries, n_docs), axis=1)[:, -100:]
                    print(f"    Top-100 S_base: mean={top100.mean():.4f}, min={top100.min():.4f}")
                else:
                    print(f"    No q_base_embeddings in query data")
            
            del doc_embs
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
