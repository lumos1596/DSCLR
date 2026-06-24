"""
COCO-Neg Evaluation: CLIP Baseline vs DeIR-Dual V2

Text-to-image retrieval with negated captions on COCO 2017 val set.
Supports multiple CLIP model variants via open_clip:
  - openai       : OpenAI CLIP ViT-B/32 (original)
  - datacomp     : CLIP ViT-B-32 DataComp.XL-s13B-b90K
  - laion400m    : CLIP ViT-B-32 LAION-400M
  - negclip      : NegCLIP ViT-B-32 (from vinid/neg_clip)

Usage:
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.coconeg_eval --pretrained openai

  # With dual queries:
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.coconeg_eval --pretrained openai --dual_queries

  # Other models:
  --pretrained datacomp
  --pretrained laion400m
  --pretrained negclip
"""

import os, sys, json, argparse, logging
import numpy as np, torch, torch.nn.functional as F

torch.cuda._lazy_init()

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "dataset/COCO-Neg"
NEGATED_CSV = os.path.join(DATA_DIR, "evaluation data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
ORIG_CSV = os.path.join(DATA_DIR, "evaluation data/images/COCO_val_retrieval.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "val2017")
DUAL_QUERIES_PATH = os.path.join(DATA_DIR, "dual_queries/COCO-Neg_TSC_BALANCED_t01.jsonl")

ALPHA = 1.0
BETA = 1.5
DELTA = 0.05
T_SAFETY = 20.0

PRETRAINED_CONFIG = {
    "openai": {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "desc": "OpenAI CLIP ViT-B/32",
    },
    "datacomp": {
        "model_name": "ViT-B-32",
        "pretrained": "datacomp_xl_s13b_b90k",
        "desc": "CLIP ViT-B-32 DataComp.XL-s13B-b90K",
    },
    "laion400m": {
        "model_name": "ViT-B-32",
        "pretrained": "laion400m_e32",
        "desc": "CLIP ViT-B-32 LAION-400M",
    },
    "negclip": {
        "model_name": "ViT-B-32",
        "pretrained": "Nano1337/negclip",
        "desc": "NegCLIP ViT-B-32",
    },
}


def load_clip_model(pretrained_key, device):
    import open_clip
    cfg = PRETRAINED_CONFIG[pretrained_key]
    model_name = cfg["model_name"]
    pretrained_tag = cfg["pretrained"]
    logger.info(f"Loading {cfg['desc']} via open_clip (model={model_name}, pretrained={pretrained_tag})")

    if pretrained_key == "negclip":
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download("Nano1337/negclip", "open_clip_pytorch_model.bin")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
    elif pretrained_key == "openai":
        local_path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
        if os.path.exists(local_path):
            logger.info(f"Using local OpenAI CLIP weights: {local_path}")
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=local_path)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    elif pretrained_key == "laion400m":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag, quick_gelu=True)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    logger.info(f"Model loaded: {cfg['desc']}")
    return model, preprocess, tokenizer


def load_data():
    import pandas as pd, ast
    df = pd.read_csv(NEGATED_CSV)

    image_ids = []
    image_paths = []
    captions_list = []
    pos_objects_list = []
    neg_objects_list = []

    for _, row in df.iterrows():
        img_id = str(row['image_id'])
        img_path = os.path.join(IMAGES_DIR, f"{int(row['image_id']):012d}.jpg")
        caps = ast.literal_eval(row['captions'])
        pos_objs = ast.literal_eval(row['positive_objects'])
        neg_objs = ast.literal_eval(row['negative_objects'])

        if os.path.exists(img_path):
            image_ids.append(img_id)
            image_paths.append(img_path)
            captions_list.append(caps)
            pos_objects_list.append(pos_objs)
            neg_objects_list.append(neg_objs)

    logger.info(f"Loaded {len(image_ids)} images with negated captions")
    logger.info(f"Sample caption: {captions_list[0][0]}")
    logger.info(f"Sample neg objects: {neg_objects_list[0]}")

    return image_ids, image_paths, captions_list, pos_objects_list, neg_objects_list


def load_dual_queries():
    dual_data = {}
    if os.path.exists(DUAL_QUERIES_PATH):
        with open(DUAL_QUERIES_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    dual_data[r['qid']] = r
        logger.info(f"Loaded {len(dual_data)} dual queries")
    else:
        logger.warning(f"Dual queries not found at {DUAL_QUERIES_PATH}")
    return dual_data


def encode_images_clip(model, preprocess, image_paths, device, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        logger.info(f"Loading cached image embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logger.info(f"Cached: {len(cache['image_ids'])} images, shape={cache['embeddings'].shape}")
        return cache['image_ids'], cache['embeddings']

    logger.info(f"Encoding {len(image_paths)} images...")
    all_embeddings = []
    batch_size = 64

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        end = min(start + batch_size, len(image_paths))
        batch_images = []
        for path in image_paths[start:end]:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(preprocess(img))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                batch_images.append(preprocess(Image.new('RGB', (224, 224))))

        image_input = torch.stack(batch_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = F.normalize(image_features.float(), p=2, dim=1)
        all_embeddings.append(image_features.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save({"image_ids": image_paths, "embeddings": embeddings}, cache_path)
    logger.info(f"Cached image embeddings to {cache_path}")
    return image_paths, embeddings


def encode_texts_openclip(model, tokenizer, texts, device, batch_size=256):
    import open_clip
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        if isinstance(tokenizer, open_clip.tokenizer.SimpleTokenizer):
            text_tokens = open_clip.tokenize(batch_texts).to(device)
        else:
            text_tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features.float(), p=2, dim=1)
        all_embeddings.append(text_features.cpu())
    return torch.cat(all_embeddings, dim=0)


def compute_recall_at_k(scores, positive_pairs, k):
    nb_queries, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_reshaped = positive_pairs.view(nb_queries, 1, nb_images)
    nb_true_positive = (topk_onehot * positive_reshaped).sum(dim=(1, 2))
    recall = (nb_true_positive / nb_positive)
    return recall


def batchify_recall(func, scores, positive_pairs, batch_size, device, k):
    results = []
    for start in range(0, len(scores), batch_size):
        end = start + batch_size
        s = scores[start:end].to(device)
        p = positive_pairs[start:end].to(device)
        result = func(s, p, k).cpu()
        results.append(result)
    return torch.cat(results)


def is_none_query(q):
    return not q or q.strip().lower() in ['none', '[none]', 'n/a', 'null', '']


def score_deir_dual_v2(s_base, s_req, s_neg, cos_qbase_qneg, has_req, has_neg,
                       alpha=ALPHA, beta=BETA, delta=DELTA):
    if not has_neg:
        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        return s_base + beta * s_req_eff
    tau = cos_qbase_qneg + delta
    overflow = s_neg - tau
    smooth_penalty = F.softplus(overflow)
    raw_penalty = alpha * smooth_penalty
    safety = 1.0 - torch.sigmoid((s_neg - tau) * T_SAFETY)
    s_req_eff = s_req if has_req else torch.zeros_like(s_base)
    return s_base + beta * s_req_eff * safety - raw_penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual_queries", action="store_true", help="Use dual queries for DeIR-Dual V2")
    parser.add_argument("--pretrained", type=str, default="openai",
                        choices=list(PRETRAINED_CONFIG.keys()),
                        help="CLIP pretrained variant")
    parser.add_argument("--subset", type=int, default=0, help="Use subset of data (0=all)")
    args = parser.parse_args()

    cfg = PRETRAINED_CONFIG[args.pretrained]
    logger.info("=" * 60)
    logger.info(f"COCO-Neg Evaluation: {cfg['desc']}")
    logger.info("=" * 60)

    image_ids, image_paths, captions_list, pos_objects_list, neg_objects_list = load_data()

    if args.subset > 0:
        n = min(args.subset, len(image_ids))
        image_ids = image_ids[:n]
        image_paths = image_paths[:n]
        captions_list = captions_list[:n]
        pos_objects_list = pos_objects_list[:n]
        neg_objects_list = neg_objects_list[:n]
        logger.info(f"Using subset of {n} images")

    num_images = len(image_ids)
    num_captions_per_image = len(captions_list[0])
    logger.info(f"Images: {num_images}, Captions/image: {num_captions_per_image}")

    device = "cuda"
    model, preprocess, tokenizer = load_clip_model(args.pretrained, device)

    image_emb_cache = os.path.join(DATA_DIR, f"embeddings/coconeg_{args.pretrained}_images.pt")
    _, image_embeddings = encode_images_clip(model, preprocess, image_paths, device, image_emb_cache)
    image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=1).to(device)

    query_texts = []
    query_image_indices = []
    for img_idx, caps in enumerate(captions_list):
        for cap in caps:
            query_texts.append(cap)
            query_image_indices.append(img_idx)

    total_queries = len(query_texts)
    logger.info(f"Total queries: {total_queries}")

    logger.info(f"Encoding {len(query_texts)} negated caption queries...")
    text_embeddings = encode_texts_openclip(model, tokenizer, query_texts, device, batch_size=256)

    logger.info("Computing baseline scores (negated captions → images)...")
    scores = text_embeddings.to(device) @ image_embeddings.T

    positive_pairs = torch.zeros(total_queries, num_images, dtype=torch.bool)
    for qi, img_idx in enumerate(query_image_indices):
        positive_pairs[qi, img_idx] = True

    recall_k_list = [1, 5, 10]
    baseline_metrics = {}
    for k in recall_k_list:
        recall_vals = batchify_recall(compute_recall_at_k, scores, positive_pairs.float(), 512, device, k)
        baseline_metrics[f"image_retrieval_recall@{k}"] = (recall_vals > 0).float().mean().item()

    logger.info("Baseline (negated captions) Results:")
    for k in recall_k_list:
        m = f"image_retrieval_recall@{k}"
        logger.info(f"  {m}: {baseline_metrics[m]:.4f}")

    import pandas as pd, ast
    df_orig = pd.read_csv(ORIG_CSV)
    orig_query_texts = []
    orig_query_image_indices = []
    for img_idx in range(num_images):
        img_id = int(image_ids[img_idx])
        row = df_orig[df_orig['image_id'] == img_id]
        if len(row) == 0:
            continue
        caps = ast.literal_eval(row.iloc[0]['captions'])
        for cap in caps:
            orig_query_texts.append(cap)
            orig_query_image_indices.append(img_idx)

    orig_metrics = {}
    if orig_query_texts:
        logger.info(f"\nEncoding {len(orig_query_texts)} original (positive) captions...")
        orig_text_embeddings = encode_texts_openclip(model, tokenizer, orig_query_texts, device, batch_size=256)
        orig_scores = orig_text_embeddings.to(device) @ image_embeddings.T

        orig_positive_pairs = torch.zeros(len(orig_query_texts), num_images, dtype=torch.float)
        for qi, img_idx in enumerate(orig_query_image_indices):
            orig_positive_pairs[qi, img_idx] = 1.0

        for k in recall_k_list:
            recall_vals = batchify_recall(compute_recall_at_k, orig_scores, orig_positive_pairs, 512, device, k)
            orig_metrics[f"image_retrieval_recall@{k}"] = (recall_vals > 0).float().mean().item()

        logger.info("Original (positive) captions Results:")
        for k in recall_k_list:
            m = f"image_retrieval_recall@{k}"
            logger.info(f"  {m}: {orig_metrics[m]:.4f}")

    deir_metrics = {}
    qplus_metrics = {}

    if args.dual_queries:
        dual_data = load_dual_queries()
        if not dual_data:
            logger.error("No dual queries found! Run coconeg_reformulate.py first.")
            return

        logger.info(f"\n--- DeIR-Dual V2 (alpha={ALPHA}, beta={BETA}, delta={DELTA}) ---")

        q_plus_list = []
        q_minus_list = []
        has_req_mask = []
        has_neg_mask = []

        for qi, cap in enumerate(query_texts):
            qid = f"q{qi}"
            d = dual_data.get(qid, {})
            q_plus = d.get('q_plus', '')
            q_minus = d.get('q_minus', '')

            if not q_plus or is_none_query(q_plus):
                q_plus = cap
                has_req_mask.append(0.0)
            else:
                has_req_mask.append(1.0)

            if not q_minus or is_none_query(q_minus):
                q_minus = ""
                has_neg_mask.append(0.0)
            else:
                has_neg_mask.append(1.0)

            q_plus_list.append(q_plus)
            q_minus_list.append(q_minus)

        logger.info(f"Q_plus available: {int(sum(has_req_mask))}/{len(has_req_mask)}")
        logger.info(f"Q_minus available: {int(sum(has_neg_mask))}/{len(has_neg_mask)}")

        logger.info("Encoding Q_req (Q+)...")
        q_req_emb = encode_texts_openclip(model, tokenizer, q_plus_list, device, batch_size=256)

        logger.info("Encoding Q_neg (Q-)...")
        q_neg_texts = [q if q else "nothing" for q in q_minus_list]
        q_neg_emb = encode_texts_openclip(model, tokenizer, q_neg_texts, device, batch_size=256)

        cos_qbase_qneg = F.cosine_similarity(text_embeddings, q_neg_emb, dim=1)

        TOP_K = 100
        logger.info(f"Computing DeIR-Dual V2 scores (top-{TOP_K} reranking)...")

        topk_scores, topk_indices = torch.topk(scores.cpu(), k=min(TOP_K, num_images), dim=1)

        S_base_topk = topk_scores
        S_req_topk = torch.zeros(total_queries, TOP_K)
        S_neg_topk = torch.zeros(total_queries, TOP_K)

        for i in tqdm(range(total_queries), desc="S_req/S_neg"):
            indices = topk_indices[i]
            doc_emb_selected = image_embeddings[indices]
            s_req = (q_req_emb[i].unsqueeze(0).to(device) @ doc_emb_selected.T).squeeze(0)
            S_req_topk[i] = s_req.float().cpu()

            if has_neg_mask[i] > 0:
                s_neg = (q_neg_emb[i].unsqueeze(0).to(device) @ doc_emb_selected.T).squeeze(0)
                S_neg_topk[i] = s_neg.float().cpu()

        S_final_topk = torch.zeros(total_queries, TOP_K)
        for i in range(total_queries):
            k = min(TOP_K, num_images)
            s_final = score_deir_dual_v2(
                s_base=S_base_topk[i, :k],
                s_req=S_req_topk[i, :k],
                s_neg=S_neg_topk[i, :k],
                cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
                has_req=bool(has_req_mask[i] > 0),
                has_neg=bool(has_neg_mask[i] > 0),
            )
            S_final_topk[i, :k] = s_final

        deir_scores = torch.full((total_queries, num_images), -1e9)
        for i in range(total_queries):
            k = min(TOP_K, num_images)
            deir_scores[i, topk_indices[i, :k]] = S_final_topk[i, :k]

        for k_val in recall_k_list:
            recall_vals = batchify_recall(compute_recall_at_k, deir_scores, positive_pairs.float(), 512, device, k_val)
            deir_metrics[f"image_retrieval_recall@{k_val}"] = (recall_vals > 0).float().mean().item()

        logger.info("DeIR-Dual V2 Results:")
        for k_val in recall_k_list:
            m = f"image_retrieval_recall@{k_val}"
            delta_v = deir_metrics[m] - baseline_metrics[m]
            logger.info(f"  {m}: {deir_metrics[m]:.4f} (Δ={delta_v:+.4f})")

        logger.info("\n--- Q_plus Only ---")
        S_qplus_topk = torch.zeros(total_queries, TOP_K)
        for i in range(total_queries):
            k = min(TOP_K, num_images)
            s_b = S_base_topk[i, :k]
            s_r = S_req_topk[i, :k]
            has_req = bool(has_req_mask[i] > 0)
            s_req_eff = s_r if has_req else torch.zeros_like(s_b)
            S_qplus_topk[i, :k] = s_b + BETA * s_req_eff

        qplus_scores = torch.full((total_queries, num_images), -1e9)
        for i in range(total_queries):
            k = min(TOP_K, num_images)
            qplus_scores[i, topk_indices[i, :k]] = S_qplus_topk[i, :k]

        for k_val in recall_k_list:
            recall_vals = batchify_recall(compute_recall_at_k, qplus_scores, positive_pairs.float(), 512, device, k_val)
            qplus_metrics[f"image_retrieval_recall@{k_val}"] = (recall_vals > 0).float().mean().item()

        logger.info("Q_plus Only Results:")
        for k_val in recall_k_list:
            m = f"image_retrieval_recall@{k_val}"
            delta_v = qplus_metrics[m] - baseline_metrics[m]
            logger.info(f"  {m}: {qplus_metrics[m]:.4f} (Δ={delta_v:+.4f})")

    print("\n" + "=" * 80)
    print(f"COCO-Neg EVALUATION SUMMARY ({cfg['desc']})")
    print("=" * 80)
    print(f"{'Method':<35} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 80)
    if orig_metrics:
        print(f"{'Original (positive) captions':<35} {orig_metrics.get('image_retrieval_recall@1',0):>8.4f} {orig_metrics.get('image_retrieval_recall@5',0):>8.4f} {orig_metrics.get('image_retrieval_recall@10',0):>8.4f}")
    print(f"{'Baseline (negated captions)':<35} {baseline_metrics['image_retrieval_recall@1']:>8.4f} {baseline_metrics['image_retrieval_recall@5']:>8.4f} {baseline_metrics['image_retrieval_recall@10']:>8.4f}")
    if qplus_metrics:
        print(f"{'Q_plus only':<35} {qplus_metrics['image_retrieval_recall@1']:>8.4f} {qplus_metrics['image_retrieval_recall@5']:>8.4f} {qplus_metrics['image_retrieval_recall@10']:>8.4f}")
    if deir_metrics:
        print(f"{'DeIR-Dual V2 (full)':<35} {deir_metrics['image_retrieval_recall@1']:>8.4f} {deir_metrics['image_retrieval_recall@5']:>8.4f} {deir_metrics['image_retrieval_recall@10']:>8.4f}")
        print("-" * 80)
        for k_val in [1, 5, 10]:
            m = f"image_retrieval_recall@{k_val}"
            delta_v = deir_metrics[m] - baseline_metrics[m]
            print(f"  Δ DeIR-Dual V2 vs Baseline R@{k_val}: {delta_v:+.4f}")

    results = {
        "clip_model": args.pretrained,
        "clip_desc": cfg["desc"],
        "num_images": num_images,
        "num_queries": total_queries,
        "original_captions": orig_metrics,
        "baseline_negated": baseline_metrics,
        "qplus_only": qplus_metrics,
        "deir_dual_v2": deir_metrics,
        "params": {"alpha": ALPHA, "beta": BETA, "delta": DELTA, "t_safety": T_SAFETY},
    }

    os.makedirs("results/coconeg", exist_ok=True)
    results_path = f"results/coconeg/coconeg_{args.pretrained}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
