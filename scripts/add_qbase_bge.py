"""为 bge-large-en 训练集 embeddings 补充 q_base_embeddings"""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

q_base_texts = []
with open('dataset/FollowIR_train/train/dsclr_total_dataset.jsonl') as f:
    for line in f:
        item = json.loads(line.strip())
        q_base = f"{item['query']} {item['instruction']}".strip()
        q_base_texts.append(q_base)

print(f'Loaded {len(q_base_texts)} Q_base texts')

from eval.models.encoder import SentenceTransformerEncoder
encoder = SentenceTransformerEncoder(
    model_name='BAAI/bge-large-en-v1.5',
    device='cuda',
    batch_size=128,
    normalize_embeddings=True
)

print('Encoding Q_base with BGE-large-en...')
q_base_emb = encoder.encode_queries(q_base_texts, batch_size=128)
print(f'Q_base embeddings shape: {q_base_emb.shape}')

q_base_emb = F.normalize(q_base_emb.float(), p=2, dim=1)

cache_path = 'dataset/FollowIR_train/embeddings/bge-large-en/dsclr_train_embeddings.pt'
cache = torch.load(cache_path, map_location='cpu', weights_only=False)
print(f'Existing cache keys: {list(cache.keys())}')

cache['q_base_embeddings'] = q_base_emb.cpu().to(torch.float16)
print(f'Added q_base_embeddings: shape={cache["q_base_embeddings"].shape}')

torch.save(cache, cache_path)
print(f'Saved updated cache to {cache_path}')
