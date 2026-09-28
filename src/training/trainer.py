import copy
import math
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LambdaLR
import logging
from tqdm import tqdm
import numpy as np

# Ensure checkpoints directory exists
os.makedirs('checkpoints', exist_ok=True)


from ..evaluation.metrics import compute_metrics
from .mixup import apply_mixup_or_cutmix, mixup_criterion

logger = logging.getLogger(__name__)


class ModelEMA:
    """
    Exponential moving average of model weights.

    Uses a ramped decay so the EMA model tracks closely at the start
    and only smooths aggressively once there's enough signal.
    """

    def __init__(self, model, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        ema_params = dict(self.ema_model.named_parameters())
        for name, p in model.named_parameters():
            if name in ema_params:
                ema_params[name].mul_(d).add_(p.detach(), alpha=1 - d)
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(b)

    def update_from_swa(self, swa_model):
        """Copy SWA-averaged weights into EMA model."""
        self.ema_model.load_state_dict(swa_model.state_dict())


class SWA:
    """
    Stochastic Weight Averaging.

    Maintains a running average of model weights after a specified epoch.
    At the end of training, the SWA model is used for evaluation/inference.
    SWA finds wider, flatter minima → better generalization.
    """

    def __init__(self, model, swa_start_epoch: int, swa_lr: float, swa_anneal_epochs: int = 5):
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.swa_anneal_epochs = swa_anneal_epochs

    @torch.no_grad()
    def update_swa(self, model):
        """Update SWA running average with current model weights."""
        swa_params = dict(self.swa_model.named_parameters())
        self.swa_n += 1
        alpha = 1.0 / self.swa_n

        for name, p in model.named_parameters():
            if name in swa_params:
                swa_params[name].mul_(1 - alpha).add_(p.detach(), alpha=alpha)

        # Copy buffers (BN stats, etc.)
        swa_buffers = dict(self.swa_model.named_buffers())
        for name, b in model.named_buffers():
            if name in swa_buffers:
                swa_buffers[name].copy_(b)

    def update_bn_stats(self, model, train_loader, device):
        """
        Update BatchNorm statistics for the SWA model.
        Must be called after SWA averaging is complete.
        """
        self.swa_model.train()
        # Reset BN running stats
        for m in self.swa_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.momentum = None

        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Updating SWA BN stats", file=sys.stdout, mininterval=30, ncols=100):
                inputs = inputs.to(device)
                with torch.amp.autocast('cuda'):
                    self.swa_model(inputs)
        self.swa_model.eval()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, min_lr_ratio=0.01):
    """
    Cosine annealing with linear warmup.

    During warmup: LR linearly increases from 0 to initial_lr
    After warmup: LR cosine anneals from initial_lr to min_lr

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of steps for linear warmup
        num_training_steps: Total number of training steps
        num_cycles: Frequency of cosine cycles (0.5 = single half-cycle)
        min_lr_ratio: Final LR as fraction of initial LR
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


class DRTrainer:
    """
    Trainer for DR classification with:
      - Cosine scheduler with linear warmup (separate backbone/head warmup)
      - Mixup/CutMix augmentation
      - EMA (Exponential Moving Average)
      - SWA (Stochastic Weight Averaging)
      - Gradient accumulation
      - Mixed precision training (AMP)
      - Deep supervision (auxiliary loss)
      - Early stopping on QWK
    """

    def __init__(self, model, train_loader, val_loader, criterion, device, config,
                 ablation='proposed', fold_idx=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.epochs = config.get("epochs", 80)
        self.ablation = ablation
        self.fold_idx = fold_idx

        # Optimizer with separate LR groups
        lr = config.get("lr", 3e-4)
        weight_decay = config.get("weight_decay", 0.01)
        head_weight_decay = config.get("head_weight_decay", 0.05)
        backbone_lr_mult = config.get("backbone_lr_mult", 0.2)

        backbone_params = [p for n, p in self.model.named_parameters() if n.startswith("backbone")]
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("backbone")]

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
                {"params": head_params, "lr": lr, "weight_decay": head_weight_decay},
            ],
        )

        # Save initial LRs
        for pg in self.optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

        # Scheduler
        scheduler_type = config.get("scheduler", "cosine")
        self.warmup_epochs = config.get("warmup_epochs", 10)
        steps_per_epoch = math.ceil(len(train_loader) / config.get("accumulation_steps", 2))
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs

        if scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[pg['lr'] for pg in self.optimizer.param_groups],
                total_steps=total_steps,
                pct_start=config.get("pct_start", 0.15),
                div_factor=config.get("div_factor", 25.0),
                final_div_factor=config.get("final_div_factor", 1000.0),
                anneal_strategy='cos'
            )
            self.scheduler_step_per_batch = True
        else:
            # Cosine with linear warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=0.01
            )
            self.scheduler_step_per_batch = True

        self.scheduler_type = scheduler_type

        # Training config
        self.patience = config.get("patience", 25)
        self.accumulation_steps = config.get("accumulation_steps", 2)
        self.scaler = torch.amp.GradScaler('cuda')
        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)

        # EMA
        self.use_ema = config.get("use_ema", True)
        self.ema = ModelEMA(self.model, decay=config.get("ema_decay", 0.999)) if self.use_ema else None

        # SWA
        self.use_swa = config.get("use_swa", True)
        if self.use_swa:
            self.swa = SWA(
                self.model,
                swa_start_epoch=config.get("swa_start_epoch", 50),
                swa_lr=config.get("swa_lr", 1e-6),
                swa_anneal_epochs=config.get("swa_anneal_epochs", 5)
            )
        else:
            self.swa = None

        # Mixup/CutMix
        self.use_mixup = config.get("use_mixup", True)
        self.mixup_alpha = config.get("mixup_alpha", 0.4)
        self.cutmix_alpha = config.get("cutmix_alpha", 1.0)
        self.mix_prob = config.get("mix_prob", 0.1)
        self.cutmix_prob = config.get("cutmix_prob", 0.3)

        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_val_qwk = -1.0
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        self.optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
            file=sys.stdout,
            mininterval=30,
            ncols=100
        )):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply Mixup/CutMix
            if self.use_mixup:
                mixed_inputs, y_a, y_b, lam = apply_mixup_or_cutmix(
                    inputs, targets,
                    mixup_alpha=self.mixup_alpha,
                    cutmix_alpha=self.cutmix_alpha,
                    mix_prob=self.mix_prob,
                    cutmix_prob=self.cutmix_prob
                )
            else:
                mixed_inputs, y_a, y_b, lam = inputs, targets, targets, 1.0

            with torch.amp.autocast('cuda'):
                outputs = self.model(mixed_inputs)
                if self.use_mixup and lam < 1.0:
                    loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                else:
                    loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                # Only step optimizer/scaler if no inf/nan gradients
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                new_scale = self.scaler.get_scale()

                self.optimizer.zero_grad()

                # Step scheduler per batch (both onecycle and cosine-with-warmup)
                if new_scale >= old_scale and self.scheduler_step_per_batch:
                    self.scheduler.step()

                if self.use_ema:
                    self.ema.update(self.model)

            running_loss += loss.item() * self.accumulation_steps * inputs.size(0)

            # For metrics, use original (unmixed) targets and classification logits
            with torch.no_grad():
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        train_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        return train_loss, metrics

    def val_epoch(self, epoch, use_ema: bool = False, use_swa: bool = False):
        """Evaluate on validation set."""
        if use_swa and self.swa is not None:
            eval_model = self.swa.swa_model
        elif use_ema and self.use_ema:
            eval_model = self.ema.ema_model
        else:
            eval_model = self.model

        eval_model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.epochs} [Val]",
                file=sys.stdout,
                mininterval=30,
                ncols=100
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.amp.autocast('cuda'):
                    outputs = eval_model(inputs)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                # Get predictions from combined classification + ordinal heads
                # Uses 0.7/0.3 weighting (classification dominant)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    ordinal_logits = outputs.get('ordinal_logits', None)
                    if ordinal_logits is not None:
                        from ..models.components import OrdinalRegressionHead
                        cls_probs = torch.softmax(logits, dim=1)
                        ord_probs = OrdinalRegressionHead.ordinal_logits_to_class_probs(ordinal_logits)
                        final_probs = 0.7 * cls_probs + 0.3 * ord_probs
                        preds = torch.argmax(final_probs, dim=1)
                    else:
                        preds = torch.argmax(logits, dim=1)
                else:
                    preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        return val_loss, metrics

    def train(self):
        logger.info("Starting training...")
        if self.fold_idx is not None:
            logger.info(f"Training Fold {self.fold_idx + 1}")

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_epoch(epoch)

            # Evaluate with EMA model
            val_loss, val_metrics = self.val_epoch(epoch, use_ema=True)

            val_qwk = val_metrics['qwk']
            val_acc = val_metrics['accuracy']
            train_acc = train_metrics['accuracy']

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1_macro']:.4f}, QWK: {train_metrics['qwk']:.4f}")
            logger.info(f"[Val] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_macro']:.4f}, QWK: {val_qwk:.4f}")

            # SWA: update running average after swa_start_epoch
            if self.use_swa and epoch >= self.swa.swa_start_epoch:
                self.swa.update_swa(self.model)
                logger.info(f"SWA updated (n={self.swa.swa_n})")

            # Early Stopping and Checkpointing on QWK
            if val_qwk > self.best_val_qwk or (val_qwk == self.best_val_qwk and val_loss < self.best_val_loss):
                self.best_val_qwk = val_qwk
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self.epochs_without_improvement = 0

                # Save EMA model
                save_model = self.ema.ema_model if self.use_ema else self.model
                suffix = f"_fold{self.fold_idx}" if self.fold_idx is not None else ""
                torch.save(save_model.state_dict(), f'checkpoints/best_model_{self.ablation}{suffix}.pth')
                logger.info(f"New best model saved at epoch {epoch+1} with QWK {val_qwk:.4f}")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Post-training: update SWA BN stats and evaluate
        if self.use_swa and self.swa.swa_n > 0:
            logger.info("Updating SWA BatchNorm statistics...")
            self.swa.update_bn_stats(self.model, self.train_loader, self.device)

            # Evaluate SWA model
            swa_val_loss, swa_val_metrics = self.val_epoch(self.epochs, use_swa=True)
            logger.info(f"[SWA Val] Loss: {swa_val_loss:.4f}, Acc: {swa_val_metrics['accuracy']:.4f}, QWK: {swa_val_metrics['qwk']:.4f}")

            # If SWA is better, save it
            if swa_val_metrics['qwk'] > self.best_val_qwk:
                logger.info(f"SWA model outperforms best EMA model (QWK: {swa_val_metrics['qwk']:.4f} > {self.best_val_qwk:.4f})")
                suffix = f"_fold{self.fold_idx}" if self.fold_idx is not None else ""
                torch.save(self.swa.swa_model.state_dict(), f'checkpoints/best_model_{self.ablation}{suffix}.pth')
                self.best_val_qwk = swa_val_metrics['qwk']

        return train_losses, val_losses, train_accs, val_accs
