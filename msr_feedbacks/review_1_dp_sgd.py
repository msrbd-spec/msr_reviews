#!/usr/bin/env python3
"""
dp_sgd_standalone_v2.py — Standalone DP-SGD script (FIXED cumulative privacy accounting)
==========================================================================================
Runs Section 9a (DP-SGD: Differentially Private Federated Learning) with an
END-TO-END composed privacy guarantee, resolving Peer-Review Major Comment 1:

    "The privacy-preserving claim is currently too strong ... epsilon = 8 or
    epsilon = 12 is not the privacy budget of the final trained system.
    Privacy loss composes over repeated access to the same client data."

──────────────────────────────────────────────────────────────────────────────
WHAT WAS WRONG (in the original dp_sgd_standalone.py / Section 9b)
──────────────────────────────────────────────────────────────────────────────
Every round, every client, a *brand-new* `PrivacyEngine()` was instantiated and
`make_private_with_epsilon(..., epochs=LOCAL_EPOCHS, target_epsilon=eps_target)`
calibrated noise so that ONE client's 3 local epochs alone would spend
`eps_target`. Because a new engine (= new, empty RDP accountant) was created
every single round, the reported "epsilon_achieved" never composed across the
50 communication rounds that the same client's data was repeatedly touched by.
The number written to `history['epsilon']` and to Table 9 was therefore a
*single-round* budget, not the budget of the trained system — exactly the gap
the reviewer flagged.

──────────────────────────────────────────────────────────────────────────────
THE FIX
──────────────────────────────────────────────────────────────────────────────
1. ONE `PrivacyEngine` PER CLIENT, created once, BEFORE the round loop, and
   reused (never re-instantiated) for all R=50 rounds. Opacus's RDP accountant
   lives inside the PrivacyEngine object; calling `.make_private()` again on
   the SAME engine (with a fresh local model/optimizer each round, since the
   model itself is reset from the broadcast global weights every round) keeps
   appending steps to the SAME accountant, i.e. it correctly composes privacy
   loss across the whole training horizon for that client.
   [This is the standard "one engine per client, persisted across FL rounds"
   pattern; equivalent to saving/reloading the accountant state between
   rounds, as documented by Opacus and multiple FL+Opacus integrations.]
2. A FIXED per-client noise multiplier `sigma_k` is solved ONCE, up front, for
   the FULL intended horizon (R=50 rounds x E=3 local epochs x that client's
   own batches/epoch), via `opacus.accountants.utils.get_noise_multiplier`.
   We do NOT call `make_private_with_epsilon` per round anymore (that method
   implicitly assumes the given `epochs` IS the entire training horizon,
   which was the root cause of the bug).
3. The ACHIEVED cumulative epsilon is read from the persisted accountant via
   `privacy_engine.get_epsilon(delta)` — after every round (for a full
   epsilon-vs-round trajectory) and at the end of training (for the final,
   reportable, end-to-end epsilon). If early stopping fires before round 50,
   the achieved epsilon will legitimately be <= the value sigma was
   calibrated for (fewer steps were actually taken) — this is reported
   honestly rather than assumed.
4. Scope of the guarantee (stated explicitly, not implied): this is a
   PER-CLIENT (participant-level, over the full training run) DP guarantee in
   the CROSS-SILO regime, where all K=5 clients participate in every round
   (no client-subsampling amplification is claimed — only the standard
   subsampled-Gaussian / Poisson batch-sampling amplification that Opacus's
   RDP accountant already integrates). The system-level, worst-case guarantee
   reported in the paper is epsilon_max = max_k(epsilon_k), the tightest bound
   that simultaneously holds for every participating institution.
5. Full audit trail (per-client n_k, sample_rate, step budget, calibrated
   sigma, and the per-round epsilon trajectory) is saved to JSON so every
   number in Table 9 is independently re-derivable/verifiable.

This keeps "privacy-preserving" truthful in the title: epsilon is now the
genuine end-to-end guarantee for the entire federated training run, not a
per-round snapshot silently repeated 50 times.

Usage:
    pip install opacus --quiet
    python dp_sgd_standalone_v2.py 2>&1 | tee dp_sgd_standalone_v2.log
"""

import os, sys, gc, copy, json, math, random, warnings, traceback, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

# ── Make sure Opacus is available (module-not-found guard) ───────────────────
try:
    import opacus  # noqa: F401
except ImportError:
    print('[setup] Installing opacus ...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'opacus', '--quiet'], check=False)

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants.utils import get_noise_multiplier
    _OPACUS_AVAILABLE = True
except ImportError:
    _OPACUS_AVAILABLE = False
    print('WARNING: Opacus still unavailable after install attempt. '
          'DP-SGD runs will be skipped (only the no-DP baseline will run).')

# ── Offline mode: load pretrained weights from cache, no downloads ────────────
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — Configuration (must match the main pipeline exactly)
# ═══════════════════════════════════════════════════════════════════════════════

NUM_CLIENTS = 5
CLASSES     = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
NUM_CLASSES = 4
FOLDS       = [f'Fold_{i}' for i in range(1, 6)]
DATA_ROOT   = 'datasets/final_5_fold_pruned/'
OUT_DIR     = 'outputs/dp_sgd_standalone_v2/'
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE  = 384
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
BATCH_SIZE  = 32 
NUM_WORKERS = 8

LR_BACKBONE  = 3e-5
LR_ATTN      = 1e-4
LR_HEAD      = 2e-4
WEIGHT_DECAY = 2e-4

MSAF_DIM   = 256
GEM_P      = 3.0
DROP_PATH  = 0.10

FOCAL_GAMMA = 2.0
TEST_COUNTS = np.array([42.0, 94.0, 35.0, 116.0])
_inv        = 1.0 / (TEST_COUNTS + 1e-6)
FOCAL_ALPHA = (_inv / _inv.sum()).tolist()
print(f'Focal alpha (test-dist): {[f"{a:.3f}" for a in FOCAL_ALPHA]}')

AUX_W        = 0.2
FL_ROUNDS    = 50
LOCAL_EPOCHS = 3
PATIENCE     = 18

# ── DP-SGD experiment config (matches methodology: Folds 1,3,5 | eps in {8,12}) ──
DP_BASE       = os.path.join(OUT_DIR, 'dp_sgd_v2_cumulative')
os.makedirs(DP_BASE, exist_ok=True)
DP_RUN_NAME   = 'FL_Run2_Heterogeneous'
DP_FOLDS      = ['Fold_1', 'Fold_3', 'Fold_5']
DP_EPSILONS   = [None, 12.0, 8.0]     # None = no-DP baseline
TARGET_DELTA  = 1e-5                   # matches methodology delta = 1e-5
MAX_GRAD_NORM = 1.0                    # matches methodology clipping norm C = 1.0
ACCOUNTANT    = 'rdp'                  # Renyi-DP accountant (Abadi et al. moments-accountant analogue)
EPS_TOLERANCE = 0.01                   # opacus binary-search tolerance for sigma calibration

# ── JSON helpers with file locking (safe for concurrent writes) ─────────────
def save_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)

def save_json_locked(obj, path, max_retries=20, retry_delay=1.0):
    lock_path = path + '.lock'
    for attempt in range(max_retries):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if attempt == max_retries - 1:
                try: os.remove(lock_path)
                except: pass
                break
            time.sleep(retry_delay)
    try:
        save_json(obj, path)
    finally:
        try: os.remove(lock_path)
        except: pass

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default if default is not None else {}

# ── Elsevier paper-ready matplotlib style ────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth':   1.0,
    'lines.markersize':  3,
})

print('Configuration ready.')
print(f'  DP folds: {DP_FOLDS} | epsilons: {DP_EPSILONS} | delta={TARGET_DELTA} | accountant={ACCOUNTANT}')
print('Section 0 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Model Architecture (Opacus-safe: GeM+proj+stack done in parent
#              forward so every submodule sees a single Tensor, as required by
#              Opacus's per-sample-gradient hooks/vmap — unrelated to the
#              Major-Comment-1 fix, but required for DP-SGD to run at all)
# ═══════════════════════════════════════════════════════════════════════════════

class GeMPool(nn.Module):
    def __init__(self, p=GEM_P, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1, init_alpha=0.01):
        super().__init__()
        t = int(abs(math.log2(max(channels, 2)) / gamma + b / gamma))
        k = max(t if t % 2 else t + 1, 3)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.conv    = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha   = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.sigmoid(self.conv(self.gap(x).view(b, 1, c))).view(b, c, 1, 1)
        return x + self.alpha * (x * w - x)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * noise.float() / (keep + 1e-8)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7,
                 init_alpha=0.01, drop_path=DROP_PATH):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_fc1   = nn.Linear(channels, reduced, bias=False)
        self.ch_fc2   = nn.Linear(reduced, channels, bias=False)
        self.ch_sig   = nn.Sigmoid()
        pad           = spatial_kernel // 2
        self.sp_conv  = nn.Conv2d(2, 1, spatial_kernel, padding=pad, bias=False)
        self.sp_sig   = nn.Sigmoid()
        self.alpha    = nn.Parameter(torch.tensor(float(init_alpha)))
        self.drop     = StochasticDepth(drop_path)

    def _ch(self, x):
        b, c, _, _ = x.shape
        mx = self.max_pool(x).view(b, c)
        av = self.avg_pool(x).view(b, c)
        gate = self.ch_sig(
            self.ch_fc2(F.relu(self.ch_fc1(mx), inplace=True)) +
            self.ch_fc2(F.relu(self.ch_fc1(av), inplace=True))
        ).view(b, c, 1, 1)
        return x * gate

    def _sp(self, x):
        sp = torch.cat([x.max(dim=1, keepdim=True)[0],
                        x.mean(dim=1, keepdim=True)], dim=1)
        return x * self.sp_sig(self.sp_conv(sp))

    def forward(self, x):
        x_attn = self._sp(self._ch(x))
        return x + self.alpha * self.drop(x_attn - x)


class AuxHead(nn.Module):
    def __init__(self, in_ch, num_classes=NUM_CLASSES):
        super().__init__()
        self.gem = GeMPool()
        self.ln  = nn.LayerNorm(in_ch)
        self.fc  = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        return self.fc(self.ln(self.gem(x)))


class CrossScaleAttentionHead(nn.Module):
    """Opacus-safe: receives a single pre-pooled/projected/stacked (B,3,d)
    tensor (GeM-pool + projection + stacking happen in the PARENT forward)."""
    def __init__(self, dims, d=MSAF_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.gem = nn.ModuleList([GeMPool() for _ in dims])
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(c, d, bias=False), nn.LayerNorm(d))
            for c in dims
        ])
        self.q_lin = nn.Linear(d, d, bias=False)
        self.k_lin = nn.Linear(d, d, bias=False)
        self.v_lin = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5
        self.norm  = nn.LayerNorm(d)
        self.temp  = nn.Parameter(torch.ones(1))
        self.head  = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(d // 2, num_classes),
        )

    def forward(self, token_seq):
        q = self.q_lin(token_seq[:, -1:, :])
        k = self.k_lin(token_seq)
        v = self.v_lin(token_seq)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        fused = (attn @ v).squeeze(1)
        fused = self.norm(fused + token_seq[:, -1, :]) * self.temp
        return self.head(fused)


class ConvNeXtV2MSAFv5(nn.Module):
    DIMS = [96, 192, 384, 768]

    def __init__(self, num_classes=NUM_CLASSES, attn_type='msaf', use_aux=False):
        super().__init__()
        self.attn_type = attn_type
        self.use_aux   = use_aux
        dims = self.DIMS

        try:
            import timm
            bb = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)
            self._backend = 'timm'
            self.stem   = bb.stem
            self.stage0 = bb.stages[0]
            self.stage1 = bb.stages[1]
            self.stage2 = bb.stages[2]
            self.stage3 = bb.stages[3]
        except Exception:
            bb = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self._backend = 'tv'
            self.stem   = bb.features[0]
            self.ds1    = bb.features[2]
            self.ds2    = bb.features[4]
            self.ds3    = bb.features[6]
            self.stage0 = bb.features[1]
            self.stage1 = bb.features[3]
            self.stage2 = bb.features[5]
            self.stage3 = bb.features[7]

        def _eca(ch):  return ECABlock(ch)
        def _cbam(ch): return CBAMBlock(ch, reduction=max(4, ch // 16), drop_path=DROP_PATH)

        # PRIMARY config only (msaf): ECA(stage0,1) + CBAM(stage2,3)
        self.attn0, self.attn1 = _eca(dims[0]), _eca(dims[1])
        self.attn2, self.attn3 = _cbam(dims[2]), _cbam(dims[3])

        self.head = CrossScaleAttentionHead(
            dims=[dims[1], dims[2], dims[3]], d=MSAF_DIM, num_classes=num_classes)

        if use_aux:
            self.aux1 = AuxHead(dims[1], num_classes)
            self.aux2 = AuxHead(dims[2], num_classes)

    def _stages(self, x):
        if self._backend == 'timm':
            x  = self.stem(x)
            s0 = self.attn0(self.stage0(x))
            s1 = self.attn1(self.stage1(s0))
            s2 = self.attn2(self.stage2(s1))
            s3 = self.attn3(self.stage3(s2))
        else:
            x  = self.stem(x)
            s0 = self.attn0(self.stage0(x))
            s1 = self.attn1(self.stage1(self.ds1(s0)))
            s2 = self.attn2(self.stage2(self.ds2(s1)))
            s3 = self.attn3(self.stage3(self.ds3(s2)))
        return s1, s2, s3

    def forward(self, x):
        s1, s2, s3 = self._stages(x)
        # Opacus-safe: GeM-pool + project + stack HERE, then pass a single
        # (B,3,d) tensor to self.head (see CrossScaleAttentionHead docstring).
        tokens = []
        for i, feat in enumerate((s1, s2, s3)):
            pooled = self.head.gem[i](feat)
            tokens.append(self.head.proj[i](pooled))
        token_seq = torch.stack(tokens, dim=1)  # (B, 3, d)
        main = self.head(token_seq)

        if self.use_aux and self.training:
            return main, self.aux1(s1), self.aux2(s2)
        return main

    def get_param_groups(self):
        bb  = {'stem', 'stage0', 'stage1', 'stage2', 'stage3', 'ds1', 'ds2', 'ds3'}
        atn = {'attn0', 'attn1', 'attn2', 'attn3'}
        bp, ap, hp = [], [], []
        for name, param in self.named_parameters():
            top = name.split('.')[0]
            if   top in bb:  bp.append(param)
            elif top in atn: ap.append(param)
            else:            hp.append(param)
        return [
            {'params': bp, 'lr': LR_BACKBONE, 'name': 'backbone'},
            {'params': ap, 'lr': LR_ATTN,     'name': 'attention'},
            {'params': hp, 'lr': LR_HEAD,     'name': 'head'},
        ]


def build_primary(nc=NUM_CLASSES):
    return ConvNeXtV2MSAFv5(nc, attn_type='msaf', use_aux=False)

print('Dimension smoke test...')
_x = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
_m = build_primary().cpu(); _m.eval()
with torch.no_grad():
    _oe = _m(_x)
assert _oe.shape == (2, NUM_CLASSES), f'eval {_oe.shape}'
print(f'  msaf_primary  eval={_oe.shape}  params={sum(p.numel() for p in _m.parameters())/1e6:.2f}M')
del _m
print('Model classes defined. Section 1 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Losses, Early Stopping, Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, num_classes=NUM_CLASSES):
        super().__init__()
        self.gamma = gamma
        self.alpha = (torch.ones(num_classes) / num_classes) if alpha is None \
                     else torch.tensor(alpha, dtype=torch.float32)

    def forward(self, pred, target):
        alpha    = self.alpha.to(pred.device)
        log_prob = F.log_softmax(pred, dim=1)
        prob     = log_prob.exp()
        focal_w  = (1 - prob) ** self.gamma
        alpha_t  = alpha[target]
        loss = -(alpha_t * focal_w[range(len(target)), target] * log_prob[range(len(target)), target])
        return loss.mean()


class EarlyStopping:
    def __init__(self, patience=PATIENCE, checkpoint_path='best.pt', mode='max', min_delta=5e-5):
        self.patience = patience; self.checkpoint = checkpoint_path
        self.mode = mode; self.min_delta = min_delta
        self.counter = 0; self.best = None; self.stop = False

    def _better(self, s):
        if self.best is None: return True
        return s > self.best + self.min_delta if self.mode == 'max' else s < self.best - self.min_delta

    def step(self, score, model):
        if self._better(score):
            self.best = score; self.counter = 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True


def compute_epoch_metrics(model, loader, criterion, device=DEVICE):
    model.eval()
    ls = 0.0; ap, al = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            if isinstance(out, tuple): out = out[0]
            ls += criterion(out, labels).item() * imgs.size(0)
            ap.extend(out.argmax(1).cpu().numpy())
            al.extend(labels.cpu().numpy())
    n = len(al)
    loss = ls / n
    acc  = float((np.array(ap) == np.array(al)).mean())
    prec = float(precision_score(al, ap, average='macro', zero_division=0))
    rec  = float(recall_score(al,  ap, average='macro', zero_division=0))
    f1   = float(f1_score(al,      ap, average='macro', zero_division=0))
    return loss, acc, prec, rec, f1

print('Losses, EarlyStopping, compute_epoch_metrics defined. Section 2 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Data Transforms and Loaders
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_client_dataloaders(fold_dir, run_dir_name, client_id, strong_aug=False):
    tr_tf = TRAIN_TRANSFORM
    base  = os.path.join(fold_dir, run_dir_name, f'Client_{client_id}')
    tr_ds = datasets.ImageFolder(os.path.join(base, 'Train'), transform=tr_tf)
    vl_ds = datasets.ImageFolder(os.path.join(base, 'Valid'), transform=EVAL_TRANSFORM)
    te_ds = datasets.ImageFolder(os.path.join(base, 'Test'),  transform=EVAL_TRANSFORM)
    tr_l = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    vl_l = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_l = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return tr_l, vl_l, te_l

def get_agg_test_loader(fold_dir, run_dir_name):
    te_list = []
    for c in range(1, NUM_CLIENTS + 1):
        _, _, tl = get_client_dataloaders(fold_dir, run_dir_name, c)
        te_list.append(tl.dataset)
    return DataLoader(ConcatDataset(te_list), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print('Section 3 complete.')

def wrap_multigpu(model): return model.to(DEVICE)
def unwrap(model): return model.module if isinstance(model, nn.DataParallel) else model

print(f'BATCH_SIZE={BATCH_SIZE}  NUM_WORKERS={NUM_WORKERS}  single-GPU mode')
print('Section 3b complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — FL Aggregation + THE FIX: cumulative, composed DP-SGD accounting
# ═══════════════════════════════════════════════════════════════════════════════

def _fedavg(gm, lms, sizes):
    total = sum(sizes); w = [n / total for n in sizes]
    target = unwrap(gm)
    gd = target.state_dict()
    for k in gd:
        gd[k] = sum(w[i] * lms[i][k].float() for i in range(len(lms)))
    target.load_state_dict(gd)
    return gm


def _identity_forward(x):
    return x


def _disable_random_ops_for_dp(model):
    """Opacus's vmap-based per-sample-gradient computation cannot trace random
    ops (Dropout, StochasticDepth). Standard practice: disable them for DP runs
    only (the non-DP global model / other experiments keep them)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            module.forward = _identity_forward
        if type(module).__name__ == 'StochasticDepth':
            if hasattr(module, 'drop_prob'): module.drop_prob = 0.0
            if hasattr(module, 'p'): module.p = 0.0
            module.forward = _identity_forward
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FIX PART A — calibrate ONE fixed noise multiplier per client, for the FULL
# intended training horizon (R rounds x E local epochs), NOT per round.
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_client_noise_multipliers(tr_loaders, target_epsilon, target_delta,
                                        rounds, local_epochs, accountant=ACCOUNTANT,
                                        epsilon_tolerance=EPS_TOLERANCE):
    """For each client's (already-built, PLAIN) DataLoader, solve for the noise
    multiplier sigma_k such that, if that client trains for
    `rounds * local_epochs * len(tr_loaders[k])` DP-SGD steps at Poisson
    sample_rate = 1/len(tr_loaders[k]) (Opacus's own internal convention when
    poisson_sampling=True), the accountant reaches EXACTLY (target_epsilon,
    target_delta) at the END of the full horizon. Returns:
        sigmas          : list[float]           -- one sigma per client
        audit           : list[dict]             -- full audit trail per client
    """
    sigmas, audit = [], []
    for ci, loader in enumerate(tr_loaders):
        n_k = len(loader.dataset)
        batches_per_epoch = len(loader)                      # matches Opacus's own convention
        sample_rate = 1.0 / batches_per_epoch
        total_steps = int(rounds * local_epochs * batches_per_epoch)
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant=accountant,
            epsilon_tolerance=epsilon_tolerance,
        )
        sigmas.append(sigma)
        audit.append({
            'client': ci + 1, 'n_k': n_k, 'batches_per_epoch': batches_per_epoch,
            'sample_rate': sample_rate, 'total_steps_budgeted': total_steps,
            'sigma': sigma, 'target_epsilon': target_epsilon, 'target_delta': target_delta,
        })
        print(f'  [privacy-calib] Client_{ci+1}: n_k={n_k} batches/ep={batches_per_epoch} '
              f'sample_rate={sample_rate:.4f} steps_budget={total_steps} -> sigma={sigma:.4f}')
    return sigmas, audit


def _local_update_plain(global_model, client_loader, focal_loss,
                         local_epochs=LOCAL_EPOCHS, use_aux_loss=False):
    """No-DP local update (standard FedAvg local training, dropout/stoch-depth
    active) -- used only for the eps_target=None baseline."""
    local_model = copy.deepcopy(unwrap(global_model)).to(DEVICE)
    if hasattr(local_model, 'get_param_groups'):
        opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
    else:
        opt = optim.AdamW(local_model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    local_model.train()
    total = 0.0
    all_preds, all_labels = [], []
    for _ in range(local_epochs):
        ep = 0.0
        for imgs, labels in client_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = local_model(imgs)
            if isinstance(out, tuple) and use_aux_loss:
                main, a1, a2 = out
                task = focal_loss(main, labels) + AUX_W * (focal_loss(a1, labels) + focal_loss(a2, labels))
            else:
                m_out = out[0] if isinstance(out, tuple) else out
                task = focal_loss(m_out, labels)
            task.backward()
            opt.step()
            with torch.no_grad():
                pred = out[0] if isinstance(out, tuple) else out
                all_preds.extend(pred.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            ep += task.item()
        total += ep
    avg_acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if all_preds else 0.0
    return local_model.cpu(), total / local_epochs, avg_acc


# ─────────────────────────────────────────────────────────────────────────────
# FIX PART B — local update that trains under a PERSISTED, per-client
# PrivacyEngine (passed in from outside the round loop) with a FIXED sigma.
# No epsilon is computed here per call; epsilon is only ever queried from the
# persisted accountant AFTER a full round (or at the very end), so it reflects
# TRUE cumulative composition across every round the client has participated in.
# ─────────────────────────────────────────────────────────────────────────────
def _dp_local_update_persistent(privacy_engine, global_model, client_loader, focal_loss,
                                 noise_multiplier, local_epochs=LOCAL_EPOCHS,
                                 use_aux_loss=False, max_grad_norm=MAX_GRAD_NORM):
    local_model = copy.deepcopy(unwrap(global_model)).to(DEVICE)
    if hasattr(local_model, 'get_param_groups'):
        opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
    else:
        opt = optim.AdamW(local_model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    # Validate/fix model for DP compatibility (e.g. BatchNorm -> GroupNorm).
    # ConvNeXtV2 uses LayerNorm/GRN, so this is expected to be a no-op here,
    # but is kept for robustness / future backbone changes.
    errors = ModuleValidator.validate(local_model, strict=False)
    if errors:
        local_model = ModuleValidator.fix(local_model)
        if hasattr(local_model, 'get_param_groups'):
            opt = optim.AdamW(local_model.get_param_groups(), weight_decay=WEIGHT_DECAY)
        else:
            opt = optim.AdamW(local_model.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    local_model = _disable_random_ops_for_dp(local_model)
    local_model.train()

    # IMPORTANT: `client_loader` passed here must always be the ORIGINAL plain
    # DataLoader (never a previously DP-wrapped one) -- Opacus builds a fresh
    # Poisson-sampled DPDataLoader from it every round. Calling make_private()
    # again on the SAME `privacy_engine` instance (created once, outside the
    # round loop, per client) is what makes the accountant COMPOSE across
    # rounds instead of resetting.
    dp_model, dp_opt, dp_loader = privacy_engine.make_private(
        module=local_model,
        optimizer=opt,
        data_loader=client_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,
    )

    total = 0.0
    all_preds, all_labels = [], []
    for _ in range(local_epochs):
        ep = 0.0
        for imgs, labels in dp_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            dp_opt.zero_grad()
            out = dp_model(imgs)
            if isinstance(out, tuple) and use_aux_loss:
                main, a1, a2 = out
                task = focal_loss(main, labels) + AUX_W * (focal_loss(a1, labels) + focal_loss(a2, labels))
            else:
                m_out = out[0] if isinstance(out, tuple) else out
                task = focal_loss(m_out, labels)
            task.backward()
            dp_opt.step()
            with torch.no_grad():
                pred = out[0] if isinstance(out, tuple) else out
                all_preds.extend(pred.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            ep += task.item()
        total += ep

    # Unwrap GradSampleModule so state_dict keys match the plain global model
    # for _fedavg aggregation (silent key-mismatch bug fix, carried over from
    # the original script -- unrelated to Major Comment 1 but still required).
    trained_model = dp_model._module if hasattr(dp_model, '_module') else dp_model
    avg_acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if all_preds else 0.0
    return trained_model.cpu(), total / local_epochs, avg_acc


def train_fedprox_dp_v2(build_fn, fold_dir, run_dir_name, save_dir, run_name,
                         focal_loss=None, use_aux_loss=False,
                         target_epsilon=None, target_delta=TARGET_DELTA,
                         max_grad_norm=MAX_GRAD_NORM, accountant=ACCOUNTANT):
    """FedAvg with (optionally) DP-SGD local updates. When target_epsilon is not
    None, privacy loss is composed ACROSS ALL ROUNDS via one persistent
    PrivacyEngine per client (the Major-Comment-1 fix)."""
    os.makedirs(save_dir, exist_ok=True)
    ckpt     = os.path.join(save_dir, f'{run_name}_best.pt')
    swa_ckpt = os.path.join(save_dir, f'{run_name}_swa.pt')

    if focal_loss is None:
        focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    ce = nn.CrossEntropyLoss()

    tr_loaders, vl_loaders, sizes = [], [], []
    for c in range(1, NUM_CLIENTS + 1):
        tr, vl, _ = get_client_dataloaders(fold_dir, run_dir_name, c, strong_aug=False)
        tr_loaders.append(tr); vl_loaders.append(vl)
        sizes.append(len(tr.dataset))

    agg_vl = DataLoader(ConcatDataset([l.dataset for l in vl_loaders]),
                         BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    use_dp = _OPACUS_AVAILABLE and target_epsilon is not None
    privacy_engines, sigmas, privacy_audit = None, None, None
    if use_dp:
        sigmas, privacy_audit = calibrate_client_noise_multipliers(
            tr_loaders, target_epsilon=target_epsilon, target_delta=target_delta,
            rounds=FL_ROUNDS, local_epochs=LOCAL_EPOCHS, accountant=accountant)
        # ONE PrivacyEngine per client, created ONCE, reused for every round.
        privacy_engines = [PrivacyEngine(accountant=accountant) for _ in range(NUM_CLIENTS)]
    elif target_epsilon is not None and not _OPACUS_AVAILABLE:
        print('  [DP-SGD] Opacus unavailable -- falling back to non-DP FedAvg for this setting.')

    gm = wrap_multigpu(build_fn())
    swa_gm = AveragedModel(unwrap(gm))
    stopper = EarlyStopping(patience=PATIENCE, checkpoint_path=ckpt, mode='max')
    history = {k: [] for k in ['round', 'avg_local_loss', 'avg_local_acc',
                                 'global_val_loss', 'global_val_acc',
                                 'global_val_prec', 'global_val_rec', 'global_val_f1',
                                 'epsilon_mean', 'epsilon_max', 'epsilon_per_client']}

    dp_tag = f'DP(target_eps={target_epsilon}, composed over up to {FL_ROUNDS} rounds)' if use_dp else 'No-DP'
    print(f'FedAvg + DP-SGD: {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs | {dp_tag}')
    print(f'Client train sizes: {sizes}')

    SWA_FL_START = FL_ROUNDS - 10
    for rnd in range(1, FL_ROUNDS + 1):
        lms, lls, las = [], [], []
        for ci in range(NUM_CLIENTS):
            if use_dp:
                lm, ll, la = _dp_local_update_persistent(
                    privacy_engines[ci], gm, tr_loaders[ci], focal_loss,
                    noise_multiplier=sigmas[ci], local_epochs=LOCAL_EPOCHS,
                    use_aux_loss=use_aux_loss, max_grad_norm=max_grad_norm)
            else:
                lm, ll, la = _local_update_plain(
                    gm, tr_loaders[ci], focal_loss,
                    local_epochs=LOCAL_EPOCHS, use_aux_loss=use_aux_loss)
            lms.append({k: v.cpu() for k, v in lm.state_dict().items()})
            lls.append(ll); las.append(la)
            del lm; gc.collect()

        gm = _fedavg(gm, lms, sizes).to(DEVICE)
        avg_ll = float(np.mean(lls)); avg_la = float(np.mean(las))

        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = compute_epoch_metrics(gm, agg_vl, ce)
        history['round'].append(rnd)
        history['avg_local_loss'].append(avg_ll); history['avg_local_acc'].append(avg_la)
        history['global_val_loss'].append(vl_loss); history['global_val_acc'].append(vl_acc)
        history['global_val_prec'].append(vl_prec); history['global_val_rec'].append(vl_rec)
        history['global_val_f1'].append(vl_f1)

        # Query the PERSISTED, composed accountant -- this is the TRUE
        # cumulative epsilon spent by each client up through THIS round,
        # not a per-round-reset snapshot (the bug being fixed).
        if use_dp:
            eps_now = [privacy_engines[ci].get_epsilon(target_delta) for ci in range(NUM_CLIENTS)]
        else:
            eps_now = [0.0] * NUM_CLIENTS
        history['epsilon_per_client'].append(eps_now)
        history['epsilon_mean'].append(float(np.mean(eps_now)))
        history['epsilon_max'].append(float(np.max(eps_now)))

        stopper.step(vl_f1, gm)
        if rnd >= SWA_FL_START:
            swa_gm.update_parameters(gm)

        if rnd % 5 == 0 or stopper.stop:
            eps_str = f'eps(mean/max)={history["epsilon_mean"][-1]:.3f}/{history["epsilon_max"][-1]:.3f}' if use_dp else 'eps=n/a(no-DP)'
            print(f'  Round {rnd:3d}/{FL_ROUNDS} | LL={avg_ll:.4f} | '
                  f'A={vl_acc:.4f} F1={vl_f1:.4f} | {eps_str} | ES={stopper.counter}/{PATIENCE}')

        if stopper.stop:
            print(f'  Early stopping at round {rnd}.')
            break

    gm.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f'  Best val F1: {stopper.best:.4f}')

    update_bn(tr_loaders[0], swa_gm.to(DEVICE), device=DEVICE)
    torch.save(swa_gm.state_dict(), swa_ckpt)

    # Final, reportable end-to-end epsilon = whatever the accountant shows
    # after training actually stopped (<= the horizon sigma was calibrated
    # for, if early stopping triggered before round FL_ROUNDS).
    privacy_summary = None
    if use_dp:
        final_eps_per_client = history['epsilon_per_client'][-1]
        privacy_summary = {
            'target_epsilon': target_epsilon,
            'target_delta': target_delta,
            'accountant': accountant,
            'rounds_run': len(history['round']),
            'rounds_budgeted_for_sigma': FL_ROUNDS,
            'achieved_epsilon_per_client': final_eps_per_client,
            'achieved_epsilon_mean': float(np.mean(final_eps_per_client)),
            'achieved_epsilon_max': float(np.max(final_eps_per_client)),
            'sigma_per_client': sigmas,
            'audit_trail': privacy_audit,
            'scope_note': ('Per-client (participant-level), end-to-end over the full '
                            'federated training run actually executed (cross-silo: all '
                            f'{NUM_CLIENTS} clients participate every round; standard '
                            'Poisson-subsampled-Gaussian amplification only, no extra '
                            'client-subsampling amplification claimed).'),
        }
        print(f'  [privacy] FINAL achieved epsilon per client: '
              f'{[f"{e:.3f}" for e in final_eps_per_client]} '
              f'(mean={privacy_summary["achieved_epsilon_mean"]:.3f}, '
              f'max={privacy_summary["achieved_epsilon_max"]:.3f}, target={target_epsilon})')

    return gm, history, swa_ckpt, privacy_summary

print('Training functions defined (fixed cumulative DP accounting). Section 4 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, save_dir, run_label, use_tta=False, n_crops=10):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(DEVICE))
            if isinstance(out, tuple): out = out[0]
            probs = torch.softmax(out, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels); y_pred = np.array(all_preds); y_prob = np.array(all_probs)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    print(f'--- {run_label} ---')
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    bins = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auroc_mac = roc_auc_score(bins, y_prob, average='macro', multi_class='ovr')
    auroc_mic = roc_auc_score(bins, y_prob, average='micro', multi_class='ovr')
    accuracy  = float((y_pred == y_true).mean())
    macro_f1  = float(report['macro avg']['f1-score'])
    macro_prec= float(report['macro avg']['precision'])
    macro_rec = float(report['macro avg']['recall'])

    print(f'  Accuracy={accuracy:.4f}  Macro-F1={macro_f1:.4f}  AUROC(macro)={auroc_mac:.4f}')

    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'CM -- {run_label}')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'cm_{run_label}.png'), dpi=300); plt.close()

    class_aurocs = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(bins[:, i], y_prob[:, i])
        ca = auc(fpr, tpr); class_aurocs[cls] = ca
        ax.plot(fpr, tpr, label=f'{cls} (AUC={ca:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'ROC -- {run_label}')
    ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'roc_{run_label}.png'), dpi=300); plt.close()

    return {
        'accuracy': accuracy, 'macro_f1': macro_f1,
        'macro_precision': macro_prec, 'macro_recall': macro_rec,
        'weighted_f1': float(report['weighted avg']['f1-score']),
        'auroc_macro': float(auroc_mac), 'auroc_micro': float(auroc_mic),
        'per_class_auroc': {k: float(v) for k, v in class_aurocs.items()},
        'per_class': {cls: {'precision': float(report[cls]['precision']),
                             'recall': float(report[cls]['recall']),
                             'f1': float(report[cls]['f1-score']),
                             'support': int(report[cls]['support'])} for cls in CLASSES},
        'confusion_matrix': cm.tolist(), 'n_samples': int(len(y_true)),
    }


def plot_curves(history, save_dir, run_label):
    os.makedirs(save_dir, exist_ok=True)
    rr = history['round']
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(rr, history['avg_local_loss'], label='Avg Local', color='#0066CC')
    axes[0].plot(rr, history['global_val_loss'], label='Global Val', color='#CC0000')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Round'); axes[0].legend(); axes[0].grid(alpha=0.4)
    axes[1].plot(rr, history['avg_local_acc'], color='#0066CC', label='Train Acc', ls='--')
    axes[1].plot(rr, history['global_val_acc'], color='#CC0000', label='Val Acc')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Round'); axes[1].legend(); axes[1].grid(alpha=0.4)
    for k, lbl in [('global_val_prec','Val Prec'), ('global_val_rec','Val Rec'), ('global_val_f1','Val F1')]:
        axes[2].plot(rr, history[k], label=lbl)
    axes[2].set_title('Val Precision/Recall/F1'); axes[2].set_xlabel('Round')
    axes[2].legend(); axes[2].grid(alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'curves_fl_{run_label}.png'), dpi=300); plt.close()


def plot_privacy_trajectory(history, save_dir, run_label):
    """NEW: epsilon-vs-round trajectory -- the direct visual evidence that
    privacy loss composes monotonically across the federated training run."""
    if not history.get('epsilon_max') or all(e == 0.0 for e in history['epsilon_max']):
        return
    os.makedirs(save_dir, exist_ok=True)
    rr = history['round']
    fig, ax = plt.subplots(figsize=(7, 5))
    per_client = np.array(history['epsilon_per_client'])  # (rounds, NUM_CLIENTS)
    for ci in range(per_client.shape[1]):
        ax.plot(rr, per_client[:, ci], alpha=0.35, lw=0.8, color='#888888')
    ax.plot(rr, history['epsilon_mean'], color='#0066CC', lw=1.6, label='Mean over clients')
    ax.plot(rr, history['epsilon_max'], color='#CC0000', lw=1.6, label='Max over clients (worst case)')
    ax.set_xlabel('Communication round'); ax.set_ylabel('Cumulative achieved $\\varepsilon$')
    ax.set_title(f'Composed privacy loss vs. round -- {run_label}')
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epsilon_trajectory_{run_label}.png'), dpi=300)
    plt.close()

print('Evaluation utilities defined. Section 5 complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# Section 9a — DP-SGD experiment: cumulative, end-to-end privacy accounting
# ═══════════════════════════════════════════════════════════════════════════════

WINNING_BUILD_FN = build_primary
WINNING_USE_AUX  = False   # matches main pipeline: winning architecture = msaf_primary, no aux heads

dp_master = os.path.join(DP_BASE, 'dp_results_v2.json')
dp_results = load_json(dp_master, default={})

print(f'\n{"="*70}')
print(f'  DP-SGD Standalone v2 (cumulative accounting) — {len(DP_FOLDS)} folds x {len(DP_EPSILONS)} epsilons')
print(f'  Results JSON: {dp_master}')
print(f'{"="*70}')

for fold in DP_FOLDS:
    fold_dir = os.path.join(DATA_ROOT, fold)
    dp_results.setdefault(fold, {})
    for eps_target in DP_EPSILONS:
        eps_label = f'eps{int(eps_target)}' if eps_target else 'noDP'
        if eps_label in dp_results[fold] and dp_results[fold][eps_label].get('standard', {}).get('accuracy'):
            print(f'[skip -- already done] {fold} / {eps_label}')
            continue

        print(f'\n{"="*70}\n  DP-SGD: {fold} | target_epsilon={eps_target}\n{"="*70}')
        save_dir = os.path.join(DP_BASE, fold, eps_label)
        lbl = f'dp_{fold}_{eps_label}'

        try:
            gm, hist, swa, privacy_summary = train_fedprox_dp_v2(
                WINNING_BUILD_FN, fold_dir, DP_RUN_NAME, save_dir, lbl,
                use_aux_loss=WINNING_USE_AUX, target_epsilon=eps_target)

            save_json(hist, os.path.join(save_dir, f'history_{lbl}.json'))
            plot_curves(hist, save_dir, lbl)
            plot_privacy_trajectory(hist, save_dir, lbl)

            agg_te = get_agg_test_loader(fold_dir, DP_RUN_NAME)
            m = evaluate_model(gm, agg_te, save_dir, lbl, use_tta=False)
            save_json(m, os.path.join(save_dir, f'metrics_{lbl}.json'))
            if privacy_summary is not None:
                save_json(privacy_summary, os.path.join(save_dir, f'privacy_audit_{lbl}.json'))

            dp_results[fold][eps_label] = {
                'standard': m, 'epsilon_target': eps_target,
                'privacy_summary': privacy_summary,
            }
            del gm; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f'ERROR: {e}'); traceback.print_exc()
            dp_results[fold][eps_label] = {'error': str(e), 'epsilon_target': eps_target}

        save_json_locked(dp_results, dp_master)
        print(f'  Results saved to {dp_master}')

# ── Table 9 (revised): target vs. ACHIEVED cumulative epsilon ────────────────
rows = []
for fold in DP_FOLDS:
    for eps_target in DP_EPSILONS:
        eps_label = f'eps{int(eps_target)}' if eps_target else 'noDP'
        entry = dp_results.get(fold, {}).get(eps_label, {})
        r = entry.get('standard', {})
        ps = entry.get('privacy_summary') or {}
        if r:
            rows.append({
                'Fold': fold,
                'Epsilon_Target': eps_target if eps_target else float('inf'),
                'Epsilon_Achieved_Mean': ps.get('achieved_epsilon_mean', None),
                'Epsilon_Achieved_Max': ps.get('achieved_epsilon_max', None),
                'Rounds_Run': ps.get('rounds_run', None),
                'Accuracy': r.get('accuracy', ''),
                'Macro_F1': r.get('macro_f1', ''),
                'AUROC': r.get('auroc_macro', ''),
            })
df_dp = pd.DataFrame(rows)
print('\nTable 9 (revised) -- DP-SGD Privacy-Utility Tradeoff with CUMULATIVE end-to-end epsilon:')
print(df_dp.to_string(index=False))
df_dp.to_csv(os.path.join(OUT_DIR, 'table9_dp_tradeoff_v2_cumulative.csv'), index=False)

# ── Aggregate + privacy-utility figure, plotted against ACHIEVED epsilon ─────
if len(df_dp) > 0:
    df_dp_agg = df_dp.groupby('Epsilon_Target').agg(
        acc_mean=('Accuracy', 'mean'), acc_std=('Accuracy', 'std'),
        f1_mean=('Macro_F1', 'mean'), f1_std=('Macro_F1', 'std'),
        eps_achieved_max_mean=('Epsilon_Achieved_Max', 'mean')).reset_index()
    print('\nAggregate (mean +/- std across folds):')
    print(df_dp_agg.to_string(index=False))
    df_dp_agg.to_csv(os.path.join(OUT_DIR, 'table9_dp_tradeoff_v2_aggregate.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = []
    for _, row in df_dp_agg.iterrows():
        if np.isinf(row['Epsilon_Target']):
            x_labels.append('no-DP\n($\\varepsilon=\\infty$)')
        else:
            eps_ach = row['eps_achieved_max_mean']
            x_labels.append(f"target={row['Epsilon_Target']:.0f}\nachieved(max)≈{eps_ach:.1f}" if pd.notna(eps_ach) else f"target={row['Epsilon_Target']:.0f}")
    xpos = range(len(df_dp_agg))
    ax.errorbar(xpos, df_dp_agg['f1_mean'], yerr=df_dp_agg['f1_std'],
                marker='o', capsize=5, color='#0066CC', label='Macro F1')
    ax.errorbar(xpos, df_dp_agg['acc_mean'], yerr=df_dp_agg['acc_std'],
                marker='s', capsize=5, color='#CC0000', label='Accuracy')
    ax.set_xticks(list(xpos)); ax.set_xticklabels(x_labels)
    ax.set_xlabel('Privacy setting (target / achieved cumulative worst-case $\\varepsilon$)')
    ax.set_ylabel('Score')
    ax.set_title('DP-SGD Privacy-Utility Tradeoff (end-to-end composed $\\varepsilon$, 3 folds, Run2)')
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_dp_tradeoff_v2_cumulative.png'), dpi=300)
    plt.close()
    print('Figure saved: fig_dp_tradeoff_v2_cumulative.png')

print(f'\n{"="*70}')
print('  DP-SGD Standalone v2 complete.  Cumulative privacy accounting resolved (Major Comment 1).')
print(f'  Results JSON: {dp_master}')
print(f'{"="*70}')