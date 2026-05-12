import json
import numpy as np

all_data = {}
for ds in ['Core17', 'Robust04', 'News21']:
    path = f'results/e5-mistral-7b/v21_grid/{ds}/all_results.json'
    with open(path) as f:
        all_data[ds] = json.load(f)

param_combos = set()
for ds in ['Core17', 'Robust04', 'News21']:
    for r in all_data[ds]:
        key = (r['alpha'], r['beta'], r['delta'], r['gamma'])
        param_combos.add(key)

results_list = []
for combo in param_combos:
    a, b, d, g = combo
    metrics = {}
    valid = True
    for ds in ['Core17', 'Robust04', 'News21']:
        found = [r for r in all_data[ds] if r['alpha'] == a and r['beta'] == b and r['delta'] == d and r['gamma'] == g]
        if not found:
            valid = False
            break
        r = found[0]
        ch_map = r['changed_MAP@1000']
        og_map = r['og_MAP@1000']
        ch_ndcg5 = r['changed_nDCG@5']
        og_ndcg5 = r['og_nDCG@5']
        metrics[ds] = {
            'change_MAP': ch_map - og_map,
            'change_nDCG5': ch_ndcg5 - og_ndcg5,
            'p-MRR': r['p-MRR']
        }
    
    if valid:
        target_avg = (metrics['Core17']['change_MAP'] + metrics['Robust04']['change_MAP'] + metrics['News21']['change_nDCG5']) / 3
        pmrr_avg = (metrics['Core17']['p-MRR'] + metrics['Robust04']['p-MRR'] + metrics['News21']['p-MRR']) / 3
        results_list.append({
            'alpha': a, 'beta': b, 'delta': d, 'gamma': g,
            'target_avg': target_avg,
            'pmrr_avg': pmrr_avg,
            'Core17_change_MAP': metrics['Core17']['change_MAP'],
            'Robust04_change_MAP': metrics['Robust04']['change_MAP'],
            'News21_change_nDCG5': metrics['News21']['change_nDCG5'],
        })

results_list.sort(key=lambda x: x['target_avg'], reverse=True)

print('=== Top 20 by target_avg (Mistral V2.1 grid search) ===')
print(f'     a     b      d    g | C17_dMAP R04_dMAP N21_dN5 | target_avg  pMRR_avg')
for r in results_list[:20]:
    print(f'{r["alpha"]:5.1f} {r["beta"]:5.1f} {r["delta"]:6.2f} {r["gamma"]:4.1f} | {r["Core17_change_MAP"]:8.4f} {r["Robust04_change_MAP"]:8.4f} {r["News21_change_nDCG5"]:8.4f} | {r["target_avg"]:10.5f} {r["pmrr_avg"]:8.4f}')

print('\n=== Baseline (alpha=0, beta=0) ===')
for ds in ['Core17', 'Robust04', 'News21']:
    path = f'results/e5-mistral-7b/v21_baseline/{ds}/metrics_summary.json'
    with open(path) as f:
        m = json.load(f)
    metrics = m.get('metrics', {})
    ch = metrics.get('changed', {})
    og = metrics.get('original', {})
    pmrr = metrics.get('p-MRR', 0)
    ch_map = ch.get('map_at_1000', 0)
    ch_ndcg5 = ch.get('ndcg_at_5', 0)
    og_map = og.get('map_at_1000', 0)
    og_ndcg5 = og.get('ndcg_at_5', 0)
    print(f'{ds}: ch_MAP={ch_map:.4f}, og_MAP={og_map:.4f}, change_MAP={ch_map-og_map:.4f} | ch_nDCG5={ch_ndcg5:.4f}, change_nDCG5={ch_ndcg5-og_ndcg5:.4f} | pMRR={pmrr:.4f}')
