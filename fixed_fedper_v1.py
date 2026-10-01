#!/usr/bin/env python3
"""
fill_fedprox_run2_nulls.py -- Fill the null `per_client_fedprox_f1` values for
`run2_quantity_skew` in outputs/fedper_v2_personalized/fedper_v2_results.json.

This does NOT retrain FedPer (the existing per_client_fedper_f1 values are left
untouched) and does NOT touch run3_dirichlet_label_skew (its FedProx checkpoints
do not exist yet). It only loads the existing Run2 FedProx checkpoints and runs
per-client inference on the test splits, reusing the EXACT model / eval /
dataloader code from review_9_main.py (exec'd without its main driver) so the
evaluation protocol is identical.
"""
import os, sys, json, gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # review_9_main.py uses paths relative to this dir

SRC = os.path.join(SCRIPT_DIR, 'review_9_main.py')

# ── Load review_9_main.py definitions ONLY (everything above the
#    "# ── Main driver" marker), so we reuse its exact code without running
#    its main loop / FedPer training. ──────────────────────────────────────
with open(SRC) as f:
    src = f.read()

lines = src.split('\n')
marker_line = None
for i, ln in enumerate(lines):
    if ln.strip().startswith('#') and 'Main driver' in ln:
        marker_line = i
        break
assert marker_line is not None, 'Could not find "Main driver" marker in review_9_main.py'
defs_src = '\n'.join(lines[:marker_line])

ns = {'__name__': '__review9_defs__', '__file__': SRC}
exec(compile(defs_src, SRC, 'exec'), ns)

build_primary           = ns['build_primary']
wrap                    = ns['wrap']
unwrap                  = ns['unwrap']
per_client_macro_f1     = ns['per_client_macro_f1']
get_client_dataloaders  = ns['get_client_dataloaders']
fedprox_ckpt_path_run2  = ns['fedprox_ckpt_path_run2']
DEVICE                  = ns['DEVICE']
FEDPER_OUT              = ns['FEDPER_OUT']
NUM_CLIENTS             = ns['NUM_CLIENTS']
FOLDS                   = ns['FOLDS']
DATA_ROOT               = ns['DATA_ROOT']

import torch  # after exec so the ns torch is the same object

TAG          = 'run2_quantity_skew'
RUN_DIR_NAME = 'FL_Run2_Heterogeneous'
master_path  = os.path.join(FEDPER_OUT, 'fedper_v2_results.json')

with open(master_path) as f:
    results = json.load(f)
assert TAG in results, f'{TAG} not found in {master_path}'

print(f'Device: {DEVICE}')
print(f'Filling FedProx (Run2) nulls into {master_path}\n')

for fold in FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    ckpt = fedprox_ckpt_path_run2(fold)
    if not os.path.isfile(ckpt):
        print(f'[SKIP] {fold}: FedProx checkpoint MISSING at {ckpt}')
        continue

    print(f'{"="*60}\n  {fold}  (ckpt: {ckpt})\n{"="*60}')

    # Per-client TEST loaders only (matches review_9_main.py data layout)
    te_loaders = []
    for c in range(1, NUM_CLIENTS + 1):
        _, _, te = get_client_dataloaders(fold_dir, RUN_DIR_NAME, c)
        te_loaders.append(te)

    # Load FedProx checkpoint (strict=False, exactly as review_9_main.py does)
    fedprox_model = wrap(build_primary())
    fedprox_model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)

    f1_per_client = []
    for ci in range(NUM_CLIENTS):
        f1 = per_client_macro_f1(fedprox_model, te_loaders[ci])
        f1_per_client.append(f1)
        print(f'  {fold}/Client_{ci+1}: FedProx macro-F1 = {f1}')

    results[TAG][fold]['per_client_fedprox_f1'] = f1_per_client
    del fedprox_model
    gc.collect()
    torch.cuda.empty_cache()

    with open(master_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'[saved] {fold} FedProx values written\n')

# ── Verification ────────────────────────────────────────────────────────────
with open(master_path) as f:
    results = json.load(f)

print('=== Verification: run2_quantity_skew / per_client_fedprox_f1 ===')
all_ok = True
for fold in FOLDS:
    vals = results[TAG][fold]['per_client_fedprox_f1']
    n_null = sum(1 for v in vals if v is None)
    if n_null:
        all_ok = False
    print(f'  {fold}: {vals}  (nulls={n_null})')

print('\n=== run3_dirichlet_label_skew / per_client_fedprox_f1 (must stay all-null) ===')
for fold in FOLDS:
    vals = results['run3_dirichlet_label_skew'][fold]['per_client_fedprox_f1']
    print(f'  {fold}: {vals}')

print('\n=== per_client_fedper_f1 sanity (must be unchanged / non-null) ===')
for fold in FOLDS:
    vals = results[TAG][fold]['per_client_fedper_f1']
    print(f'  {fold}: {vals}')

print('\nDone. run2_quantity_skew FedProx nulls filled.' if all_ok
      else '\nWARNING: some nulls remain in run2_quantity_skew.')
