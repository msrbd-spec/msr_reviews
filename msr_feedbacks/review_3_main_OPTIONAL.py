#!/usr/bin/env python3
"""
fedavg_vs_fedprox_controlled.py — standalone script for a properly delivered,
controlled FedAvg vs FedProx comparison, addressing Peer-Review Major Comment 3.

WHY A FRESH RETRAIN OF BOTH, NOT JUST FedAvg:
The existing FedProx checkpoints (from the main pipeline, and the FedAvg
mu=0 checkpoints from the poisoning experiment) were never paired under a
matched-seed protocol, RNG state drifted across the many experiments that
ran earlier in the same process, so head/attention initialization and
per-round batch shuffling order were never guaranteed identical between the
two conditions. To honestly claim "identical initialization, augmentation,
local epochs, learning rates, and random seeds" (the reviewer's own fix),
BOTH conditions are retrained here, fresh, from an explicit per-fold seed
reset applied right before each of the two runs for that fold.

THE ONLY VARIABLE ISOLATED: the FedProx proximal coefficient mu.
  FedAvg  : mu = 0.0
  FedProx : mu = 0.01 (matches methodology default, Table hyperparams)
Everything else (fold split, augmentation, LR schedule per param group,
weight decay, local epochs, gradient clipping, early-stopping criterion) is
identical, and reset to the SAME seed before each of the two runs of a fold.

Run under FL_Run2_Heterogeneous, matching the manuscript's own contribution
claim ("...under heterogeneous data conditions").

Sized for RTX 4070 Super 12GB: BATCH_SIZE=16, NUM_WORKERS=4.
"""
import os, gc, copy, json, math, random, warnings, traceback
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

BASE_SEED = 42

def seed_everything(seed):
    """Reset ALL RNG streams. Called right before build_fn() for each of the
    two paired runs of a fold, so FedAvg and FedProx get bit-for-bit
    identical head/attention initialization and identical DataLoader
    shuffle order for that fold."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

NUM_CLIENTS = 5
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
FOLDS       = [f'Fold_{i}' for i in range(1, 6)]
DATA_ROOT   = 'datasets/final_5_fold_pruned/'
RUN_NAME    = 'FL_Run2_Heterogeneous'
OUT_DIR     = 'outputs/fedavg_vs_fedprox_controlled'
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE  = 384
MEAN, STD   = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH_SIZE  = 16     # RTX 4070 Super 12GB
NUM_WORKERS = 4

LR_BACKBONE, LR_ATTN, LR_HEAD = 3e-5, 1e-4, 2e-4
WEIGHT_DECAY = 2e-4
MSAF_DIM, GEM_P, DROP_PATH = 256, 3.0, 0.10

FOCAL_GAMMA = 2.0
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()

FL_ROUNDS, LOCAL_EPOCHS, PATIENCE = 50, 3, 18
CONFIGS = [('fedavg', 0.0), ('fedprox', 0.01)]

def save_json(obj, path):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=2, default=float)

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return default if default is not None else {}

# ── Model (identical to other scripts, msaf_primary only) ───────────────────
class GeMPool(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0/self.p).flatten(1)

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1, init_alpha=0.01):
        super().__init__()
        t = int(abs(math.log2(max(channels,2))/gamma + b/gamma)); k = max(t if t%2 else t+1, 3)
        self.gap = nn.AdaptiveAvgPool2d(1); self.conv = nn.Conv1d(1,1,kernel_size=k,padding=k//2,bias=False)
        self.sigmoid = nn.Sigmoid(); self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.sigmoid(self.conv(self.gap(x).view(b,1,c))).view(b,c,1,1)
        return x + self.alpha*(x*w - x)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * noise.float() / (keep + 1e-8)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7, init_alpha=0.01, drop_path=DROP_PATH):
        super().__init__()
        reduced = max(4, channels//reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1); self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc1 = nn.Linear(channels, reduced, bias=False); self.ch_fc2 = nn.Linear(reduced, channels, bias=False)
        self.ch_sig = nn.Sigmoid()
        self.sp_conv = nn.Conv2d(2,1,spatial_kernel,padding=spatial_kernel//2,bias=False); self.sp_sig = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha))); self.drop = StochasticDepth(drop_path)
    def _ch(self, x):
        b,c,_,_ = x.shape
        mx = self.max_pool(x).view(b,c); av = self.avg_pool(x).view(b,c)
        gate = self.ch_sig(self.ch_fc2(F.relu(self.ch_fc1(mx),inplace=True)) + self.ch_fc2(F.relu(self.ch_fc1(av),inplace=True))).view(b,c,1,1)
        return x*gate
    def _sp(self, x):
        sp = torch.cat([x.max(dim=1,keepdim=True)[0], x.mean(dim=1,keepdim=True)], dim=1)
        return x * self.sp_sig(self.sp_conv(sp))
    def forward(self, x):
        x_attn = self._sp(self._ch(x))
        return x + self.alpha*self.drop(x_attn - x)

class CrossScaleAttentionHead(nn.Module):
    def __init__(self, dims, d=MSAF_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.gem = nn.ModuleList([GeMPool() for _ in dims])
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(c,d,bias=False), nn.LayerNorm(d)) for c in dims])
        self.q_lin = nn.Linear(d,d,bias=False); self.k_lin = nn.Linear(d,d,bias=False); self.v_lin = nn.Linear(d,d,bias=False)
        self.scale = d**-0.5; self.norm = nn.LayerNorm(d); self.temp = nn.Parameter(torch.ones(1))
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(d,d//2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d//2,num_classes))
    def forward(self, feat_list):
        tokens = [self.proj[i](self.gem[i](f)) for i,f in enumerate(feat_list)]
        seq = torch.stack(tokens, dim=1)
        q = self.q_lin(seq[:,-1:,:]); k = self.k_lin(seq); v = self.v_lin(seq)
        attn = torch.softmax(q @ k.transpose(-2,-1) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        fused = self.norm(fused + tokens[-1]) * self.temp
        return self.head(fused)

class ConvNeXtV2MSAFv5(nn.Module):
    DIMS = [96, 192, 384, 768]
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        try:
            import timm
            bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)
            self._backend = 'timm'
            self.stem, self.stage0, self.stage1, self.stage2, self.stage3 = bb.stem, bb.stages[0], bb.stages[1], bb.stages[2], bb.stages[3]
        except Exception:
            bb = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self._backend = 'tv'
            self.stem = bb.features[0]; self.ds1, self.ds2, self.ds3 = bb.features[2], bb.features[4], bb.features[6]
            self.stage0, self.stage1, self.stage2, self.stage3 = bb.features[1], bb.features[3], bb.features[5], bb.features[7]
        dims = self.DIMS
        self.attn0, self.attn1 = ECABlock(dims[0]), ECABlock(dims[1])
        self.attn2 = CBAMBlock(dims[2], reduction=max(4,dims[2]//16), drop_path=DROP_PATH)
        self.attn3 = CBAMBlock(dims[3], reduction=max(4,dims[3]//16), drop_path=DROP_PATH)
        self.head = CrossScaleAttentionHead(dims=[dims[1],dims[2],dims[3]], d=MSAF_DIM, num_classes=num_classes)
    def _stages(self, x):
        if self._backend == 'timm':
            x = self.stem(x); s0 = self.attn0(self.stage0(x)); s1 = self.attn1(self.stage1(s0))
            s2 = self.attn2(self.stage2(s1)); s3 = self.attn3(self.stage3(s2))
        else:
            x = self.stem(x); s0 = self.attn0(self.stage0(x)); s1 = self.attn1(self.stage1(self.ds1(s0)))
            s2 = self.attn2(self.stage2(self.ds2(s1))); s3 = self.attn3(self.stage3(self.ds3(s2)))
        return s1, s2, s3
    def forward(self, x):
        s1, s2, s3 = self._stages(x)
        return self.head([s1, s2, s3])
    def get_param_groups(self):
        bb = {'stem','stage0','stage1','stage2','stage3','ds1','ds2','ds3'}
        atn = {'attn0','attn1','attn2','attn3'}
        bp, ap, hp = [], [], []
        for name, param in self.named_parameters():
            top = name.split('.')[0]
            (bp if top in bb else ap if top in atn else hp).append(param)
        return [{'params': bp, 'lr': LR_BACKBONE, 'name': 'backbone'},
                {'params': ap, 'lr': LR_ATTN, 'name': 'attention'},
                {'params': hp, 'lr': LR_HEAD, 'name': 'head'}]

def build_primary(nc=NUM_CLASSES): return ConvNeXtV2MSAFv5(nc)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, num_classes=NUM_CLASSES):
        super().__init__(); self.gamma = gamma
        self.alpha = torch.ones(num_classes)/num_classes if alpha is None else torch.tensor(alpha, dtype=torch.float32)
    def forward(self, pred, target):
        alpha = self.alpha.to(pred.device); log_prob = F.log_softmax(pred, dim=1); prob = log_prob.exp()
        focal_w = (1-prob)**self.gamma; alpha_t = alpha[target]
        return -(alpha_t * focal_w[range(len(target)), target] * log_prob[range(len(target)), target]).mean()

class EarlyStopping:
    def __init__(self, patience=PATIENCE, checkpoint_path='best.pt', mode='max', min_delta=5e-5):
        self.patience=patience; self.checkpoint=checkpoint_path; self.mode=mode; self.min_delta=min_delta
        self.counter=0; self.best=None; self.stop=False
    def _better(self, s):
        if self.best is None: return True
        return s > self.best+self.min_delta if self.mode=='max' else s < self.best-self.min_delta
    def step(self, score, model):
        if self._better(score):
            self.best=score; self.counter=0; torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter+=1
            if self.counter >= self.patience: self.stop=True

def compute_epoch_metrics(model, loader, criterion, device=DEVICE):
    model.eval(); ls=0.0; ap, al = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            ls += criterion(out, labels).item()*imgs.size(0)
            ap.extend(out.argmax(1).cpu().numpy()); al.extend(labels.cpu().numpy())
    n = len(al)
    acc = float((np.array(ap)==np.array(al)).mean())
    prec = float(precision_score(al,ap,average='macro',zero_division=0))
    rec = float(recall_score(al,ap,average='macro',zero_division=0))
    f1 = float(f1_score(al,ap,average='macro',zero_division=0))
    return ls/n, acc, prec, rec, f1

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def get_client_dataloaders(fold_dir, run_dir_name, client_id):
    base = os.path.join(fold_dir, run_dir_name, f'Client_{client_id}')
    tr_ds = datasets.ImageFolder(os.path.join(base,'Train'), transform=TRAIN_TRANSFORM)
    vl_ds = datasets.ImageFolder(os.path.join(base,'Valid'), transform=EVAL_TRANSFORM)
    te_ds = datasets.ImageFolder(os.path.join(base,'Test'), transform=EVAL_TRANSFORM)
    return (DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True))

def get_agg_test_loader(fold_dir, run_dir_name):
    te_list = []
    for c in range(1, NUM_CLIENTS+1):
        _,_,tl = get_client_dataloaders(fold_dir, run_dir_name, c)
        te_list.append(tl.dataset)
    return DataLoader(ConcatDataset(te_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def wrap(m): return m.to(DEVICE)
def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

def _fedprox_local_update(global_model, client_loader, focal_loss, mu, local_epochs=LOCAL_EPOCHS):
    local_model = wrap(copy.deepcopy(unwrap(global_model)))
    global_params = {n: p.data.clone() for n,p in unwrap(local_model).named_parameters()}
    opt = optim.AdamW(unwrap(local_model).get_param_groups(), weight_decay=WEIGHT_DECAY)
    local_model.train(); total=0.0; all_preds, all_labels = [], []
    for _ in range(local_epochs):
        ep=0.0
        for imgs, labels in client_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = local_model(imgs)
            task = focal_loss(out, labels)
            prox = sum(((p-global_params[n].to(DEVICE))**2).sum() for n,p in unwrap(local_model).named_parameters() if n in global_params)
            loss = task + (mu/2.0)*prox
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                all_preds.extend(out.argmax(1).cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            ep += loss.item()
        total += ep
    avg_acc = float((np.array(all_preds)==np.array(all_labels)).mean()) if all_preds else 0.0
    return local_model.cpu(), total/local_epochs, avg_acc

def _fedavg(gm, lms, sizes):
    total = sum(sizes); w = [n/total for n in sizes]
    target = unwrap(gm); gd = target.state_dict()
    for k in gd: gd[k] = sum(w[i]*lms[i][k].float() for i in range(len(lms)))
    target.load_state_dict(gd); return gm

def train_controlled(build_fn, fold_dir, save_dir, run_name, mu, fold_seed, focal_loss=None):
    seed_everything(fold_seed)  # THE FIX: identical starting point for FedAvg and FedProx on this fold
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f'{run_name}_best.pt')
    if focal_loss is None: focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    ce = nn.CrossEntropyLoss()
    tr_loaders, vl_loaders, sizes = [], [], []
    for c in range(1, NUM_CLIENTS+1):
        tr, vl, _ = get_client_dataloaders(fold_dir, RUN_NAME, c)
        tr_loaders.append(tr); vl_loaders.append(vl); sizes.append(len(tr.dataset))
    agg_vl = DataLoader(ConcatDataset([l.dataset for l in vl_loaders]), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    gm = wrap(build_fn())
    stopper = EarlyStopping(patience=PATIENCE, checkpoint_path=ckpt, mode='max')
    history = {k: [] for k in ['round','avg_local_loss','avg_local_acc','global_val_f1']}
    print(f'  [{run_name}] mu={mu} | seed={fold_seed} | {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs | sizes={sizes}')
    for rnd in range(1, FL_ROUNDS+1):
        lms, lls, las = [], [], []
        for ci in range(NUM_CLIENTS):
            lm, ll, la = _fedprox_local_update(gm, tr_loaders[ci], focal_loss, mu=mu)
            lms.append({k: v.cpu() for k,v in lm.state_dict().items()})
            lls.append(ll); las.append(la); del lm; gc.collect()
        gm = _fedavg(gm, lms, sizes).to(DEVICE)
        _, _, _, _, vl_f1 = compute_epoch_metrics(gm, agg_vl, ce)
        history['round'].append(rnd); history['avg_local_loss'].append(float(np.mean(lls)))
        history['avg_local_acc'].append(float(np.mean(las))); history['global_val_f1'].append(vl_f1)
        stopper.step(vl_f1, gm)
        if rnd % 5 == 0 or stopper.stop:
            print(f'    Round {rnd:3d}/{FL_ROUNDS} | ValF1={vl_f1:.4f} | ES={stopper.counter}/{PATIENCE}')
        if stopper.stop: print(f'    Early stopping at round {rnd}.'); break
        torch.cuda.empty_cache()
    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    save_json(history, os.path.join(save_dir, f'history_{run_name}.json'))
    return gm

def evaluate_model(model, loader):
    model.eval(); all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(DEVICE)); probs = torch.softmax(out, 1)
            all_probs.extend(probs.cpu().numpy()); all_preds.extend(probs.argmax(1).cpu().numpy()); all_labels.extend(labels.numpy())
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    bins = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auroc = roc_auc_score(bins, y_prob, average='macro', multi_class='ovr')
    return {
        'accuracy': float((y_pred==y_true).mean()), 'macro_f1': float(report['macro avg']['f1-score']),
        'macro_precision': float(report['macro avg']['precision']), 'macro_recall': float(report['macro avg']['recall']),
        'auroc_macro': float(auroc),
        'per_class': {cls: {'precision': float(report[cls]['precision']), 'recall': float(report[cls]['recall']), 'f1': float(report[cls]['f1-score'])} for cls in CLASSES},
    }

# ── Main loop ─────────────────────────────────────────────────────────────
master_path = os.path.join(OUT_DIR, 'fedavg_vs_fedprox_results.json')
results = load_json(master_path, default={})

for fold_idx, fold in enumerate(FOLDS):
    fold_seed = BASE_SEED + fold_idx  # same seed for BOTH configs of this fold, differs across folds
    results.setdefault(fold, {})
    fold_dir = os.path.join(DATA_ROOT, fold)
    for tag, mu in CONFIGS:
        if tag in results[fold] and results[fold][tag].get('accuracy'):
            print(f'[skip] {fold}/{tag} already done'); continue
        print(f'\n{"="*60}\n  {fold} | {tag} (mu={mu})\n{"="*60}')
        save_dir = os.path.join(OUT_DIR, fold, tag)
        run_name = f'{fold}_{tag}'
        try:
            gm = train_controlled(build_primary, fold_dir, save_dir, run_name, mu=mu, fold_seed=fold_seed)
            agg_te = get_agg_test_loader(fold_dir, RUN_NAME)
            m = evaluate_model(gm, agg_te)
            results[fold][tag] = m
            del gm; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f'ERROR: {e}'); traceback.print_exc()
            results[fold][tag] = {'error': str(e)}
        save_json(results, master_path)

# ── Aggregate + paired stats ──────────────────────────────────────────────
fedavg_f1  = [results[f]['fedavg']['macro_f1']  for f in FOLDS if 'macro_f1' in results.get(f, {}).get('fedavg', {})]
fedprox_f1 = [results[f]['fedprox']['macro_f1'] for f in FOLDS if 'macro_f1' in results.get(f, {}).get('fedprox', {})]

if len(fedavg_f1) == 5 and len(fedprox_f1) == 5:
    def agg(key):
        a = [results[f]['fedavg'][key] for f in FOLDS]; p = [results[f]['fedprox'][key] for f in FOLDS]
        return a, p
    print('\n=== Table 1 NEW ROW (paste into tab:main_cv) ===')
    for tag, key_list in [('fedavg', ['accuracy','macro_precision','macro_recall','macro_f1','auroc_macro']),
                           ('fedprox', ['accuracy','macro_precision','macro_recall','macro_f1','auroc_macro'])]:
        vals = [np.array([results[f][tag][k] for f in FOLDS]) for k in key_list]
        row = ' & '.join(f'${v.mean():.4f} \\pm {v.std():.3f}$' for v in vals)
        label = 'Exp.~2 FL (FedAvg), Heterogeneous' if tag=='fedavg' else 'Exp.~2 FL (FedProx, matched-seed rerun), Heterogeneous'
        print(f'{label} & {row} \\\\')

    print('\n=== Table 5 NEW BLOCK (paste into tab:per_class) ===')
    for tag in ['fedavg', 'fedprox']:
        label = 'Exp.~2 FL (FedAvg)' if tag=='fedavg' else 'Exp.~2 FL (FedProx, rerun)'
        print(f'\\multirow{{4}}{{*}}{{{label}}}')
        for cls in CLASSES:
            p = [results[f][tag]['per_class'][cls]['precision'] for f in FOLDS]
            r = [results[f][tag]['per_class'][cls]['recall'] for f in FOLDS]
            fs = [results[f][tag]['per_class'][cls]['f1'] for f in FOLDS]
            short = {'Chickenpox':'CP','Healthy':'H','Measles':'M','Monkeypox':'MP'}[cls]
            print(f'& {short} & ${np.mean(p):.3f} \\pm {np.std(p):.3f}$ & ${np.mean(r):.3f} \\pm {np.std(r):.3f}$ & ${np.mean(fs):.3f} \\pm {np.std(fs):.3f}$ \\\\')

    # Paired stats: Delta_k = FedProx_k - FedAvg_k
    deltas = np.array(fedprox_f1) - np.array(fedavg_f1)
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    cohens_d = mean_delta / std_delta if std_delta > 0 else float('nan')

    rng = np.random.RandomState(BASE_SEED)
    B = 10000
    boot_means = [np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(B)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    try:
        stat, p_wilcoxon = wilcoxon(fedprox_f1, fedavg_f1)
    except ValueError:
        p_wilcoxon = float('nan')

    print('\n=== Paired statistics (for new Results subsection prose) ===')
    print(f'Mean paired difference (FedProx - FedAvg): {mean_delta:+.4f}')
    print(f'Cohen\'s d (paired): {cohens_d:.3f}')
    print(f'95% bootstrap CI of mean difference: [{ci_low:+.4f}, {ci_high:+.4f}]')
    print(f'Wilcoxon signed-rank p-value (supplementary): {p_wilcoxon:.4f}')

    save_json({'fedavg_f1': fedavg_f1, 'fedprox_f1': fedprox_f1, 'per_fold_delta': deltas.tolist(),
               'mean_delta': mean_delta, 'cohens_d': cohens_d, 'bootstrap_ci_95': [float(ci_low), float(ci_high)],
               'wilcoxon_p': float(p_wilcoxon)},
              os.path.join(OUT_DIR, 'paired_stats_summary.json'))
else:
    print(f'\n[WARNING] Not all 10 runs completed (fedavg={len(fedavg_f1)}/5, fedprox={len(fedprox_f1)}/5). '
          f'Skipping aggregation until all folds finish.')

print('\nControlled FedAvg vs FedProx comparison complete.')