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

from eval.models.e5_mistral_encoder import E5MistralEncoder
encoder = E5MistralEncoder(
    model_name='intfloat/e5-mistral-7b-instruct',
    device='cuda',
    batch_size=28,
    normalize_embeddings=True
)

print('Encoding Q_base with E5-Mistral...')
q_base_emb = encoder.encode_queries(q_base_texts, batch_size=28)
print(f'Q_base embeddings shape: {q_base_emb.shape}')

q_base_emb = F.normalize(q_base_emb.float(), p=2, dim=1)

cache_path = 'dataset/FollowIR_train/embeddings/e5-mistral-7b/dsclr_train_embeddings_e5-mistral-7b.pt'
cache = torch.load(cache_path, map_location='cpu', weights_only=False)
print(f'Existing cache keys: {list(cache.keys())}')

cache['q_base_embeddings'] = q_base_emb.cpu().to(torch.bfloat16)
print(f'Added q_base_embeddings: shape={cache["q_base_embeddings"].shape}')

torch.save(cache, cache_path)
print(f'Saved updated cache to {cache_path}')
