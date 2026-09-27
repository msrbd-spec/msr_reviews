#!/usr/bin/env python3
"""
run3_dirichlet_training_standalone.py — trains + evaluates PoxCSAF-Net
(winning architecture, msaf_primary) via FedProx on FL_Run3_LabelSkew_Dirichlet
across all 5 folds. Identical hyperparameters/protocol to the main pipeline's
Exp.1/Exp.2 FL runs, so results merge directly into Table 1 / Table 5 as one
new row/block -- no new table needed.

FIX vs. previous version: genuine Dirichlet(alpha=0.5) label skew legitimately
produces some (client, class) pairs with ZERO raw images (confirmed in the
distribution log -- e.g. Fold_1/Client_5 = 0 Chickenpox, Fold_3/Client_3 = 0
Chickenpox and 0 Measles). torchvision's ImageFolder raises FileNotFoundError
on any class subfolder that is completely empty, even though the subfolder
itself exists (create_client_dirs() always makes all 4 class dirs for every
client). The fix is `allow_empty=True`, an ImageFolder/DatasetFolder
parameter added specifically for this case (torchvision docs: "If True, empty
folders are considered to be valid classes. An error is raised on empty
folders if False (default)."). Because all 4 class subfolders are always
created for every client (even when empty), find_classes() still discovers
and indexes all 4 classes identically across every client's dataset --
Chickenpox=0, Healthy=1, Measles=2, Monkeypox=3, alphabetical, matching the
model's fixed class order -- so label indices never get silently shifted for
clients missing a class. This is the ONLY change from the previous script.

Sized for RTX 4070 Super 12GB: BATCH_SIZE=16, NUM_WORKERS=4.
"""
import os, gc, copy, json, math, random, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torchvision
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Check torchvision supports allow_empty (needed for zero-sample classes) ──
_TV_VERSION = tuple(int(x) for x in torchvision.__version__.split('+')[0].split('.')[:2])
_SUPPORTS_ALLOW_EMPTY = _TV_VERSION >= (0, 16)
if not _SUPPORTS_ALLOW_EMPTY:
    raise RuntimeError(
        f"torchvision {torchvision.__version__} is too old to support "
        f"allow_empty=True (needed since genuine Dirichlet label skew "
        f"produces clients with zero samples for some classes). "
        f"Please upgrade: pip install -U torchvision")
print(f'torchvision {torchvision.__version__} -- allow_empty supported.')

NUM_CLIENTS = 5
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
FOLDS       = [f'Fold_{i}' for i in range(1, 6)]
DATA_ROOT   = 'datasets/final_5_fold_pruned/'
RUN3_NAME   = 'FL_Run3_LabelSkew_Dirichlet'
OUT_DIR     = 'outputs'
RUN3_OUT    = os.path.join(OUT_DIR, 'run3_dirichlet')
os.makedirs(RUN3_OUT, exist_ok=True)

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

def save_json(obj, path):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=2, default=float)

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return default if default is not None else {}

# ── Model (same architecture as main pipeline, msaf_primary only) ───────────
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
    """FIX: allow_empty=True on every ImageFolder call. Genuine Dirichlet
    label skew produces some (client, class) pairs with zero raw images
    (e.g. Fold_1/Client_5 has 0 Chickenpox) -- the class subfolder still
    exists (create_client_dirs() always creates all 4), it's just empty.
    Without allow_empty=True, ImageFolder raises FileNotFoundError on any
    empty class folder. Because all 4 class folders always exist for every
    client, class_to_idx stays identical (alphabetical: Chickenpox=0,
    Healthy=1, Measles=2, Monkeypox=3) across every client's dataset, so
    label indices are never silently shifted for a client missing a class."""
    base = os.path.join(fold_dir, run_dir_name, f'Client_{client_id}')
    tr_ds = datasets.ImageFolder(os.path.join(base,'Train'), transform=TRAIN_TRANSFORM, allow_empty=True)
    vl_ds = datasets.ImageFolder(os.path.join(base,'Valid'), transform=EVAL_TRANSFORM, allow_empty=True)
    te_ds = datasets.ImageFolder(os.path.join(base,'Test'), transform=EVAL_TRANSFORM, allow_empty=True)
    # Sanity check: class_to_idx must always be the full, consistent 4-class
    # mapping, regardless of which classes happen to be empty for this client.
    expected_class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    for ds_name, ds in [('Train', tr_ds), ('Valid', vl_ds), ('Test', te_ds)]:
        if ds.class_to_idx != expected_class_to_idx:
            raise RuntimeError(
                f"class_to_idx mismatch for Client_{client_id}/{ds_name}: "
                f"got {ds.class_to_idx}, expected {expected_class_to_idx}. "
                f"This would silently corrupt label indices -- aborting.")
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

def _fedprox_local_update(global_model, client_loader, focal_loss, mu=FEDPROX_MU, local_epochs=LOCAL_EPOCHS):
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

def train_fedprox_run3(build_fn, fold_dir, save_dir, run_name, focal_loss=None):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f'{run_name}_best.pt'); swa_ckpt = os.path.join(save_dir, f'{run_name}_swa.pt')
    if focal_loss is None: focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    ce = nn.CrossEntropyLoss()
    tr_loaders, vl_loaders, sizes = [], [], []
    for c in range(1, NUM_CLIENTS+1):
        tr, vl, _ = get_client_dataloaders(fold_dir, RUN3_NAME, c)
        if len(tr.dataset) == 0:
            raise RuntimeError(f'Client_{c} has ZERO total training images across all classes -- '
                                f'check Dirichlet partitioning for {fold_dir}/{RUN3_NAME}.')
        tr_loaders.append(tr); vl_loaders.append(vl); sizes.append(len(tr.dataset))
    agg_vl = DataLoader(ConcatDataset([l.dataset for l in vl_loaders]), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    gm = wrap(build_fn()); swa_gm = AveragedModel(unwrap(gm))
    stopper = EarlyStopping(patience=PATIENCE, checkpoint_path=ckpt, mode='max')
    history = {k: [] for k in ['round','avg_local_loss','avg_local_acc','global_val_loss','global_val_acc','global_val_prec','global_val_rec','global_val_f1']}
    print(f'FedProx (Run3, Dirichlet label-skew): {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs | client sizes={sizes}')
    SWA_START = FL_ROUNDS - 10
    for rnd in range(1, FL_ROUNDS+1):
        lms, lls, las = [], [], []
        for ci in range(NUM_CLIENTS):
            lm, ll, la = _fedprox_local_update(gm, tr_loaders[ci], focal_loss)
            lms.append({k: v.cpu() for k,v in lm.state_dict().items()})
            lls.append(ll); las.append(la); del lm; gc.collect()
        gm = _fedavg(gm, lms, sizes).to(DEVICE)
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(gm, agg_vl, ce)
        history['round'].append(rnd); history['avg_local_loss'].append(float(np.mean(lls)))
        history['avg_local_acc'].append(float(np.mean(las))); history['global_val_loss'].append(vl_loss)
        history['global_val_acc'].append(vl_acc); history['global_val_prec'].append(vl_prec)
        history['global_val_rec'].append(vl_rec); history['global_val_f1'].append(vl_f1)
        stopper.step(vl_f1, gm)
        if rnd >= SWA_START: swa_gm.update_parameters(gm)
        if rnd % 5 == 0 or stopper.stop:
            print(f'  Round {rnd:3d}/{FL_ROUNDS} | A={vl_acc:.4f} F1={vl_f1:.4f} | ES={stopper.counter}/{PATIENCE}')
        if stopper.stop: print(f'  Early stopping at round {rnd}.'); break
    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    update_bn(tr_loaders[0], swa_gm.to(DEVICE), device=DEVICE)
    torch.save(swa_gm.state_dict(), swa_ckpt)
    return gm, history, swa_ckpt

def evaluate_model(model, loader, save_dir, run_label):
    model.eval(); all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(DEVICE)); probs = torch.softmax(out, 1)
            all_probs.extend(probs.cpu().numpy()); all_preds.extend(probs.argmax(1).cpu().numpy()); all_labels.extend(labels.numpy())
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    bins = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auroc_mac = roc_auc_score(bins, y_prob, average='macro', multi_class='ovr')
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f'CM -- {run_label}'); plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'cm_{run_label}.png'), dpi=300); plt.close()
    return {
        'accuracy': float((y_pred==y_true).mean()), 'macro_f1': float(report['macro avg']['f1-score']),
        'macro_precision': float(report['macro avg']['precision']), 'macro_recall': float(report['macro avg']['recall']),
        'auroc_macro': float(auroc_mac),
        'per_class': {cls: {'precision': float(report[cls]['precision']), 'recall': float(report[cls]['recall']), 'f1': float(report[cls]['f1-score'])} for cls in CLASSES},
    }

# ── Run across all 5 folds ────────────────────────────────────────────────
run3_master = os.path.join(RUN3_OUT, 'run3_results.json')
run3_results = load_json(run3_master, default={})

for fold in FOLDS:
    if fold in run3_results and run3_results[fold].get('accuracy'):
        print(f'[skip] {fold} already done'); continue
    print(f'\n{"="*60}\n  Run3 Dirichlet: {fold}\n{"="*60}')
    fold_dir = os.path.join(DATA_ROOT, fold)
    save_dir = os.path.join(RUN3_OUT, fold)
    lbl = f'{fold}_{RUN3_NAME}_fl'
    try:
        gm, hist, swa = train_fedprox_run3(build_primary, fold_dir, save_dir, lbl)
        save_json(hist, os.path.join(save_dir, f'history_{lbl}.json'))
        agg_te = get_agg_test_loader(fold_dir, RUN3_NAME)
        m = evaluate_model(gm, agg_te, save_dir, lbl)
        save_json(m, os.path.join(save_dir, f'metrics_{lbl}.json'))
        run3_results[fold] = m
        del gm; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f'ERROR: {e}'); traceback.print_exc()
        run3_results[fold] = {'error': str(e)}
    save_json(run3_results, run3_master)

# ── Aggregate row for Table 1, and per-class block for Table 5 ──────────────
accs = [run3_results[f]['accuracy'] for f in FOLDS if 'accuracy' in run3_results.get(f, {})]
f1s  = [run3_results[f]['macro_f1'] for f in FOLDS if 'macro_f1' in run3_results.get(f, {})]
precs= [run3_results[f]['macro_precision'] for f in FOLDS if 'macro_precision' in run3_results.get(f, {})]
recs = [run3_results[f]['macro_recall'] for f in FOLDS if 'macro_recall' in run3_results.get(f, {})]
aurocs=[run3_results[f]['auroc_macro'] for f in FOLDS if 'auroc_macro' in run3_results.get(f, {})]

if len(accs) == 5:
    print('\n=== Table 1 NEW ROW (paste into tab:main_cv) ===')
    print(f'Exp.~3 FL (FedProx), Label-Skew Non-IID ($\\alpha={0.5}$) & '
          f'${np.mean(accs):.4f} \\pm {np.std(accs):.3f}$ & ${np.mean(precs):.4f} \\pm {np.std(precs):.3f}$ & '
          f'${np.mean(recs):.4f} \\pm {np.std(recs):.3f}$ & ${np.mean(f1s):.4f} \\pm {np.std(f1s):.3f}$ & '
          f'${np.mean(aurocs):.4f} \\pm {np.std(aurocs):.3f}$ \\\\')

    print('\n=== Table 5 NEW BLOCK (paste into tab:per_class) ===')
    for cls in CLASSES:
        p = [run3_results[f]['per_class'][cls]['precision'] for f in FOLDS if cls in run3_results.get(f, {}).get('per_class', {})]
        r = [run3_results[f]['per_class'][cls]['recall'] for f in FOLDS if cls in run3_results.get(f, {}).get('per_class', {})]
        fs = [run3_results[f]['per_class'][cls]['f1'] for f in FOLDS if cls in run3_results.get(f, {}).get('per_class', {})]
        short = {'Chickenpox':'CP','Healthy':'H','Measles':'M','Monkeypox':'MP'}[cls]
        print(f'& {short} & ${np.mean(p):.3f} \\pm {np.std(p):.3f}$ & ${np.mean(r):.3f} \\pm {np.std(r):.3f}$ & ${np.mean(fs):.3f} \\pm {np.std(fs):.3f}$ \\\\')
else:
    print(f'\n[WARNING] Only {len(accs)}/5 folds completed successfully -- '
          f'skipping Table 1/5 aggregation until all folds finish. Check errors above.')

# ── Paired comparison vs Exp.2 Heterogeneous (prose only, no new table) ──────
if len(accs) == 5:
    cv_summary = load_json(os.path.join(OUT_DIR, 'cv_summary.json'), default={})
    exp2_f1 = []
    for f in FOLDS:
        r = cv_summary.get('fold_results', {}).get(f, {}).get('FL_Run2_Heterogeneous', {}).get('fl', {}).get('standard', {})
        if r: exp2_f1.append(r['macro_f1'])
    if len(exp2_f1) == len(f1s) and len(f1s) >= 2:
        stat, p = wilcoxon(exp2_f1, f1s)
        print(f'\nWilcoxon Exp.2 (quantity-skew) vs Exp.3 (Dirichlet label-skew) macro-F1: p={p:.4f}')
        print(f'Mean F1 drop from Exp.2 to Exp.3: {np.mean(exp2_f1) - np.mean(f1s):+.4f}')

print('\nRun3 Dirichlet label-skew experiment complete.')