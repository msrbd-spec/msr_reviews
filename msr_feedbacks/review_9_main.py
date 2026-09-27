#!/usr/bin/env python3
"""
fedper_v2_personalized.py — fixes FedPer implementation and evaluation
protocol, addressing Peer-Review Major Comment 9.

BUGS FIXED vs. the original train_fedper/_fedper_avg in training_script.md:
1. Personalization was not actually persisting. The global model's head was
   never updated by aggregation, and every round each client's local training
   started from a fresh deep-copy of that SAME untouched initial head, not
   from the client's own previous round's personalized head. Personalization
   was discarded and relearned from scratch every round.
   FIX: client_heads[] list persisted across the entire round loop, each
   client's local update reads and writes its own head state dict.
2. FedProx proximal regularization was applied to ALL parameters including
   the head, pulling personalization back toward a head that (per bug 1)
   never moved anyway. A personalized head has no meaningful "global"
   reference to regularize toward.
   FIX: prox term computed only over shared (backbone+attention) params.
3. Evaluation was on the pooled test set (Comment 9's explicit complaint).
   FIX: each client's personalized model (shared params + that client's own
   head) is evaluated on that SAME client's own held-out test partition, and
   compared against the FedProx global model on the same per-client partitions.
4. Run under both the quantity-skew heterogeneous setting (Run2) AND the
   Dirichlet label-skew setting (Run3), per Comment 9's own note that this
   experiment "would also become more meaningful under genuine label or
   domain heterogeneity rather than quantity skew alone."

FedProx is NOT retrained here. Existing FedProx checkpoints (Run2 from the
main pipeline, Run3 from run3_dirichlet_training_standalone.py) are loaded
and evaluated per-client, which is all that was missing for FedProx's side
of this comparison.

Sized for RTX 4070 Super 12GB: BATCH_SIZE=16, NUM_WORKERS=4.
"""
import os, gc, copy, json, math, random, warnings, traceback
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

_TV_VERSION = tuple(int(x) for x in torchvision.__version__.split('+')[0].split('.')[:2])
if _TV_VERSION < (0, 16):
    raise RuntimeError(f"torchvision {torchvision.__version__} too old for allow_empty=True. "
                        f"pip install -U torchvision")
print(f'torchvision {torchvision.__version__} -- allow_empty supported.')

NUM_CLIENTS = 5
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
FOLDS       = [f'Fold_{i}' for i in range(1, 6)]
DATA_ROOT   = 'datasets/final_5_fold_pruned/'

# ── Settings to compare: (run_dir_name, fedprox_ckpt_dir_fn, output_tag) ─────
MAIN_OUT_DIR = 'pipeline_v2_single_gpu'   # main pipeline's OUT_DIR (Run2 FedProx checkpoints live here)
RUN3_OUT_DIR = 'outputs/run3_dirichlet'   # run3_dirichlet_training_standalone.py's RUN3_OUT

def fedprox_ckpt_path_run2(fold):
    return os.path.join(MAIN_OUT_DIR, fold, 'FL_Run2_Heterogeneous', 'fl',
                         f'{fold}_FL_Run2_Heterogeneous_fl_best.pt')

def fedprox_ckpt_path_run3(fold):
    return os.path.join(RUN3_OUT_DIR, fold, f'{fold}_FL_Run3_LabelSkew_Dirichlet_fl_best.pt')

SETTINGS = [
    {'tag': 'run2_quantity_skew', 'run_dir_name': 'FL_Run2_Heterogeneous',
     'fedprox_ckpt_fn': fedprox_ckpt_path_run2, 'label': 'Quantity-Skew (Run 2)'},
    {'tag': 'run3_dirichlet_label_skew', 'run_dir_name': 'FL_Run3_LabelSkew_Dirichlet',
     'fedprox_ckpt_fn': fedprox_ckpt_path_run3, 'label': 'Dirichlet Label-Skew (alpha=0.5)'},
]

FEDPER_OUT = os.path.join('outputs', 'fedper_v2_personalized')
os.makedirs(FEDPER_OUT, exist_ok=True)

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

FL_ROUNDS, LOCAL_EPOCHS, FEDPROX_MU, PATIENCE = 50, 3, 0.01, 18
SHARED_PREFIXES = {'stem', 'stage0', 'stage1', 'stage2', 'stage3', 'ds1', 'ds2', 'ds3',
                    'attn0', 'attn1', 'attn2', 'attn3'}

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
        bb = SHARED_PREFIXES - {'attn0','attn1','attn2','attn3'}
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

def per_client_macro_f1(model, loader):
    model.eval(); ap, al = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(DEVICE))
            ap.extend(out.argmax(1).cpu().numpy()); al.extend(labels.cpu().numpy())
    if len(al) == 0:
        return None
    report = classification_report(al, ap, labels=list(range(NUM_CLASSES)),
                                    target_names=CLASSES, output_dict=True, zero_division=0)
    return float(report['macro avg']['f1-score'])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def get_client_dataloaders(fold_dir, run_dir_name, client_id):
    """allow_empty=True: Run3 (Dirichlet) legitimately has zero-sample classes
    for some clients. Class_to_idx is verified consistent across all clients
    since create_client_dirs always creates all 4 class subfolders."""
    base = os.path.join(fold_dir, run_dir_name, f'Client_{client_id}')
    tr_ds = datasets.ImageFolder(os.path.join(base,'Train'), transform=TRAIN_TRANSFORM, allow_empty=True)
    vl_ds = datasets.ImageFolder(os.path.join(base,'Valid'), transform=EVAL_TRANSFORM, allow_empty=True)
    te_ds = datasets.ImageFolder(os.path.join(base,'Test'), transform=EVAL_TRANSFORM, allow_empty=True)
    expected = {c: i for i, c in enumerate(CLASSES)}
    for name, ds in [('Train', tr_ds), ('Valid', vl_ds), ('Test', te_ds)]:
        if ds.class_to_idx != expected:
            raise RuntimeError(f'class_to_idx mismatch for {run_dir_name}/Client_{client_id}/{name}: '
                                f'got {ds.class_to_idx}, expected {expected}')
    return (DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
            DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True))

def wrap(m): return m.to(DEVICE)
def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

# ── THE FIX: persistent per-client head, prox term on shared params only ────
def _fedper_local_update(global_model, client_head_state, client_loader, focal_loss, mu, local_epochs):
    local_model = wrap(copy.deepcopy(unwrap(global_model)))
    local_model.head.load_state_dict(client_head_state)  # personalize: load THIS client's own head
    global_shared_params = {n: p.data.clone() for n, p in unwrap(local_model).named_parameters()
                             if n.split('.')[0] in SHARED_PREFIXES}
    opt = optim.AdamW(unwrap(local_model).get_param_groups(), weight_decay=WEIGHT_DECAY)
    local_model.train()
    total = 0.0; all_preds, all_labels = [], []
    for _ in range(local_epochs):
        ep = 0.0
        for imgs, labels in client_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = local_model(imgs)
            task = focal_loss(out, labels)
            # prox term ONLY on shared params -- the head has no meaningful
            # "global" reference to regularize toward, it is purely personal
            prox = sum(((p - global_shared_params[n].to(DEVICE))**2).sum()
                       for n, p in unwrap(local_model).named_parameters()
                       if n.split('.')[0] in SHARED_PREFIXES)
            loss = task + (mu/2.0)*prox
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                all_preds.extend(out.argmax(1).cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            ep += loss.item()
        total += ep
    avg_acc = float((np.array(all_preds)==np.array(all_labels)).mean()) if all_preds else 0.0
    sd = unwrap(local_model).state_dict()
    shared_sd = {k: v.cpu() for k, v in sd.items() if k.split('.')[0] in SHARED_PREFIXES}
    head_sd = {k: v.cpu() for k, v in unwrap(local_model).head.state_dict().items()}
    del local_model
    return shared_sd, head_sd, total/local_epochs, avg_acc

def _fedper_avg_shared(gm, shared_sds, sizes):
    total = sum(sizes); w = [n/total for n in sizes]
    target = unwrap(gm); gd = target.state_dict()
    for k in gd:
        if k.split('.')[0] in SHARED_PREFIXES:
            gd[k] = sum(w[i]*shared_sds[i][k].float() for i in range(len(shared_sds)))
    target.load_state_dict(gd)
    return gm

def train_fedper_fixed(build_fn, fold_dir, run_dir_name, save_dir, run_name, focal_loss=None, mu=FEDPROX_MU):
    os.makedirs(save_dir, exist_ok=True)
    if focal_loss is None: focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    ce = nn.CrossEntropyLoss()
    tr_loaders, vl_loaders, te_loaders, sizes = [], [], [], []
    for c in range(1, NUM_CLIENTS+1):
        tr, vl, te = get_client_dataloaders(fold_dir, run_dir_name, c)
        if len(tr.dataset) == 0:
            raise RuntimeError(f'Client_{c} has ZERO total training images -- check {fold_dir}/{run_dir_name}.')
        tr_loaders.append(tr); vl_loaders.append(vl); te_loaders.append(te)
        sizes.append(len(tr.dataset))

    gm = wrap(build_fn())
    client_heads = [copy.deepcopy(unwrap(gm).head.state_dict()) for _ in range(NUM_CLIENTS)]

    best_mean_val_f1, best_shared_sd, best_client_heads = None, None, None
    patience_counter = 0
    history = {k: [] for k in ['round', 'avg_local_loss', 'avg_local_acc', 'mean_val_f1', 'per_client_val_f1']}

    print(f'FedPer (FIXED, persistent per-client heads): {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs '
          f'| client sizes={sizes}')

    for rnd in range(1, FL_ROUNDS+1):
        shared_sds, new_heads, lls, las = [], [], [], []
        for ci in range(NUM_CLIENTS):
            shared_sd, head_sd, ll, la = _fedper_local_update(
                gm, client_heads[ci], tr_loaders[ci], focal_loss, mu, LOCAL_EPOCHS)
            shared_sds.append(shared_sd); new_heads.append(head_sd)
            lls.append(ll); las.append(la); gc.collect()

        gm = _fedper_avg_shared(gm, shared_sds, sizes).to(DEVICE)
        client_heads = new_heads  # persist for next round -- THIS is the fix

        per_client_val_f1 = []
        for ci in range(NUM_CLIENTS):
            eval_model = wrap(copy.deepcopy(unwrap(gm)))
            eval_model.head.load_state_dict(client_heads[ci])
            _, _, _, _, f1 = compute_epoch_metrics(eval_model, vl_loaders[ci], ce)
            per_client_val_f1.append(f1)
            del eval_model
        mean_val_f1 = float(np.mean(per_client_val_f1))

        history['round'].append(rnd)
        history['avg_local_loss'].append(float(np.mean(lls)))
        history['avg_local_acc'].append(float(np.mean(las)))
        history['mean_val_f1'].append(mean_val_f1)
        history['per_client_val_f1'].append(per_client_val_f1)

        improved = best_mean_val_f1 is None or mean_val_f1 > best_mean_val_f1 + 5e-5
        if improved:
            best_mean_val_f1 = mean_val_f1
            best_shared_sd = {k: v.clone() for k, v in unwrap(gm).state_dict().items()}
            best_client_heads = [copy.deepcopy(h) for h in client_heads]
            patience_counter = 0
        else:
            patience_counter += 1

        if rnd % 5 == 0 or patience_counter >= PATIENCE:
            print(f'  Round {rnd:3d}/{FL_ROUNDS} | MeanValF1={mean_val_f1:.4f} | ES={patience_counter}/{PATIENCE}')
        if patience_counter >= PATIENCE:
            print(f'  Early stopping at round {rnd}.'); break
        torch.cuda.empty_cache()

    unwrap(gm).load_state_dict(best_shared_sd)
    client_heads = best_client_heads

    torch.save(unwrap(gm).state_dict(), os.path.join(save_dir, f'{run_name}_shared_best.pt'))
    for ci in range(NUM_CLIENTS):
        torch.save(client_heads[ci], os.path.join(save_dir, f'{run_name}_client{ci+1}_head_best.pt'))
    save_json(history, os.path.join(save_dir, f'history_{run_name}.json'))

    return gm, client_heads, te_loaders

# ── Main driver ───────────────────────────────────────────────────────────
master_path = os.path.join(FEDPER_OUT, 'fedper_v2_results.json')
results = load_json(master_path, default={})

for setting in SETTINGS:
    tag, run_dir_name, fedprox_ckpt_fn, label = setting['tag'], setting['run_dir_name'], setting['fedprox_ckpt_fn'], setting['label']
    results.setdefault(tag, {})
    for fold in FOLDS:
        if fold in results[tag] and results[tag][fold].get('per_client_fedper_f1'):
            print(f'[skip] {tag}/{fold} already done'); continue

        print(f'\n{"="*60}\n  FedPer v2: {label} | {fold}\n{"="*60}')
        fold_dir = os.path.join(DATA_ROOT, fold)
        save_dir = os.path.join(FEDPER_OUT, tag, fold)
        run_name = f'{fold}_{tag}_fedper'

        try:
            gm_shared, client_heads, te_loaders = train_fedper_fixed(
                build_primary, fold_dir, run_dir_name, save_dir, run_name)

            fedper_f1_per_client = []
            for ci in range(NUM_CLIENTS):
                eval_model = wrap(copy.deepcopy(unwrap(gm_shared)))
                eval_model.head.load_state_dict(client_heads[ci])
                f1 = per_client_macro_f1(eval_model, te_loaders[ci])
                fedper_f1_per_client.append(f1)
                del eval_model

            fedprox_ckpt = fedprox_ckpt_fn(fold)
            fedprox_f1_per_client = [None]*NUM_CLIENTS
            if os.path.isfile(fedprox_ckpt):
                fedprox_model = wrap(build_primary())
                fedprox_model.load_state_dict(torch.load(fedprox_ckpt, map_location=DEVICE), strict=False)
                for ci in range(NUM_CLIENTS):
                    fedprox_f1_per_client[ci] = per_client_macro_f1(fedprox_model, te_loaders[ci])
                del fedprox_model
            else:
                print(f'  [WARNING] FedProx checkpoint not found at {fedprox_ckpt} -- '
                      f'FedProx column will be null for this fold. Verify the path.')

            results[tag][fold] = {
                'per_client_fedper_f1': fedper_f1_per_client,
                'per_client_fedprox_f1': fedprox_f1_per_client,
                'fedprox_ckpt_used': fedprox_ckpt,
            }
            del gm_shared; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f'ERROR: {e}'); traceback.print_exc()
            results[tag][fold] = {'error': str(e)}

        save_json(results, master_path)

# ── Aggregate: per-client mean+-std across folds, per setting ───────────────
print('\n=== Table 8 REPLACEMENT ROWS (paste into tab:fedper) ===')
for setting in SETTINGS:
    tag, label = setting['tag'], setting['label']
    print(f'\n% -- {label} --')
    fedprox_all, fedper_all = [], []
    for ci in range(NUM_CLIENTS):
        fp = [results[tag][f]['per_client_fedprox_f1'][ci] for f in FOLDS
              if tag in results and f in results[tag] and results[tag][f].get('per_client_fedprox_f1', [None]*NUM_CLIENTS)[ci] is not None]
        pr = [results[tag][f]['per_client_fedper_f1'][ci] for f in FOLDS
              if tag in results and f in results[tag] and results[tag][f].get('per_client_fedper_f1', [None]*NUM_CLIENTS)[ci] is not None]
        fedprox_all.extend(fp); fedper_all.extend(pr)
        if fp and pr:
            fp_m, fp_s = np.mean(fp), np.std(fp)
            pr_m, pr_s = np.mean(pr), np.std(pr)
            delta = pr_m - fp_m
            print(f'& Client {ci+1} & ${fp_m:.3f} \\pm {fp_s:.3f}$ & ${pr_m:.3f} \\pm {pr_s:.3f}$ & ${delta:+.3f}$ \\\\')
        else:
            print(f'& Client {ci+1} & $TBA$ & $TBA$ & $TBA$ \\\\')
    if fedprox_all and fedper_all:
        print(f'& \\textbf{{Mean}} & $\\mathbf{{{np.mean(fedprox_all):.3f}}}$ & '
              f'$\\mathbf{{{np.mean(fedper_all):.3f}}}$ & '
              f'$\\mathbf{{{np.mean(fedper_all)-np.mean(fedprox_all):+.3f}}}$ \\\\')

print('\nFedPer v2 (fixed personalization + per-client evaluation) complete.')