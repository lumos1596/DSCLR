import json

datasets = {
    'Core17InstructionRetrieval': 'changed_MAP@1000',
    'Robust04InstructionRetrieval': 'changed_MAP@1000',
    'News21InstructionRetrieval': 'changed_nDCG@5'
}

def load_grid(results_dir):
    param_results = {}
    for ds_name, metric in datasets.items():
        path = f'{results_dir}/{ds_name}/all_results.json'
        with open(path) as f:
            data = json.load(f)
        for item in data:
            key = (item['alpha'], item['beta'], item['delta'])
            if key not in param_results:
                param_results[key] = {'alpha': item['alpha'], 'beta': item['beta'], 'delta': item['delta']}
            if 'Core17' in ds_name:
                param_results[key]['Core17_changed_MAP@1000'] = item[metric]
            elif 'Robust04' in ds_name:
                param_results[key]['Robust04_changed_MAP@1000'] = item[metric]
            else:
                param_results[key]['News21_changed_nDCG@5'] = item[metric]
    for v in param_results.values():
        v['target_avg'] = (v['Core17_changed_MAP@1000'] + v['Robust04_changed_MAP@1000'] + v['News21_changed_nDCG@5']) / 3
    return param_results

ws = load_grid('results/ablation_safety/WITH_SAFETY')
ns = load_grid('results/ablation_safety/NO_SAFETY')

ws_best = max(ws.values(), key=lambda x: x['target_avg'])
ns_best = max(ns.values(), key=lambda x: x['target_avg'])

print("=" * 80)
print("SAFETY GATE ABLATION (4B TSC_BALANCED + RepLLaMA)")
print("=" * 80)

print("\n--- WITH SAFETY ---")
print(f"  Best: a={ws_best['alpha']}, b={ws_best['beta']}, d={ws_best['delta']}")
print(f"  C17={ws_best['Core17_changed_MAP@1000']:.5f}  R04={ws_best['Robust04_changed_MAP@1000']:.5f}  N21={ws_best['News21_changed_nDCG@5']:.5f}")
print(f"  target_avg={ws_best['target_avg']:.5f}")

print("\n--- NO SAFETY ---")
print(f"  Best: a={ns_best['alpha']}, b={ns_best['beta']}, d={ns_best['delta']}")
print(f"  C17={ns_best['Core17_changed_MAP@1000']:.5f}  R04={ns_best['Robust04_changed_MAP@1000']:.5f}  N21={ns_best['News21_changed_nDCG@5']:.5f}")
print(f"  target_avg={ns_best['target_avg']:.5f}")

diff = ws_best['target_avg'] - ns_best['target_avg']
print(f"\n  SAFETY IMPACT: {diff:+.5f} ({'HELPS' if diff > 0 else 'HURTS' if diff < 0 else 'NO EFFECT'})")

print("\n" + "=" * 80)
print("SAME PARAMS COMPARISON")
print("=" * 80)

key1 = (ws_best['alpha'], ws_best['beta'], ws_best['delta'])
if key1 in ns:
    r = ns[key1]
    d = ws_best['target_avg'] - r['target_avg']
    print(f"\n  At WITH_SAFETY best params (a={ws_best['alpha']}, b={ws_best['beta']}, d={ws_best['delta']}):")
    print(f"    WS={ws_best['target_avg']:.5f}  NS={r['target_avg']:.5f}  diff={d:+.5f}")

key2 = (ns_best['alpha'], ns_best['beta'], ns_best['delta'])
if key2 in ws:
    r = ws[key2]
    d = r['target_avg'] - ns_best['target_avg']
    print(f"\n  At NO_SAFETY best params (a={ns_best['alpha']}, b={ns_best['beta']}, d={ns_best['delta']}):")
    print(f"    WS={r['target_avg']:.5f}  NS={ns_best['target_avg']:.5f}  diff={d:+.5f}")

print("\n" + "=" * 80)
print("PER-DATASET (own best params)")
print("=" * 80)
for ds_name, metric in datasets.items():
    short = ds_name.split('Instruction')[0]
    m = short + '_' + metric
    ws_b = max(ws.values(), key=lambda x: x[m])
    ns_b = max(ns.values(), key=lambda x: x[m])
    d = ws_b[m] - ns_b[m]
    print(f"\n  {ds_name}:")
    print(f"    WS={ws_b[m]:.5f} (a={ws_b['alpha']},b={ws_b['beta']},d={ws_b['delta']})")
    print(f"    NS={ns_b[m]:.5f} (a={ns_b['alpha']},b={ns_b['beta']},d={ns_b['delta']})")
    print(f"    diff={d:+.5f}")

print("\n" + "=" * 80)
print("TOP-5 PARAMS COMPARISON")
print("=" * 80)
ws_top5 = sorted(ws.values(), key=lambda x: x['target_avg'], reverse=True)[:5]
ns_top5 = sorted(ns.values(), key=lambda x: x['target_avg'], reverse=True)[:5]
print("\n  WITH_SAFETY top-5:")
for i, r in enumerate(ws_top5):
    print(f"    {i+1}. a={r['alpha']},b={r['beta']},d={r['delta']} avg={r['target_avg']:.5f} C17={r['Core17_changed_MAP@1000']:.5f} R04={r['Robust04_changed_MAP@1000']:.5f} N21={r['News21_changed_nDCG@5']:.5f}")
print("\n  NO_SAFETY top-5:")
for i, r in enumerate(ns_top5):
    print(f"    {i+1}. a={r['alpha']},b={r['beta']},d={r['delta']} avg={r['target_avg']:.5f} C17={r['Core17_changed_MAP@1000']:.5f} R04={r['Robust04_changed_MAP@1000']:.5f} N21={r['News21_changed_nDCG@5']:.5f}")
