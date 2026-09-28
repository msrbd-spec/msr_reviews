"""
SSL Pretraining Trainer.

Implements the training loop for Lesion-Aware Self-Supervised Pretraining
on the EyePACS dataset (Phase 3 of plan.md).

Features:
  - AdamW optimizer with cosine LR schedule + warmup
  - Mixed precision training (AMP fp16)
  - Exponential Moving Average (EMA) of model weights
  - Gradient clipping
  - Combined loss: InfoNCE contrastive + BCE lesion type + MSE lesion count + L1 reconstruction
  - Saves pretrained backbone weights for RetiNA-Net fine-tuning
  - Logs training curves for paper figures

Author: RetiNA-Net Project
"""

import os
import time
import math
import logging
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict


from ..models.ssl_model import SSLModel, SSLCombinedLoss


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Maintains a slow-moving copy of weights for more stable evaluation.
    """

    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_to(self, model):
        """Copy EMA weights into model (for saving/evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


class SSLTrainer:
    """
    Trainer for Lesion-Aware Self-Supervised Pretraining.

    Args:
        model: SSLModel instance
        train_loader: DataLoader for EyePACS SSL dataset
        config: Configuration dict (from config_ssl.yaml)
        device: torch.device
        logger: logging.Logger
    """

    def __init__(self, model, train_loader, config, device, logger=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Training hyperparameters
        self.epochs = config.get('ssl_epochs', 50)
        self.lr = config.get('ssl_lr', 1.5e-4)
        self.backbone_lr_mult = config.get('ssl_backbone_lr_mult', 0.1)
        self.weight_decay = config.get('ssl_weight_decay', 0.05)
        self.warmup_epochs = config.get('ssl_warmup_epochs', 5)
        self.grad_clip = config.get('ssl_grad_clip', 1.0)
        self.grad_accum_steps = config.get('ssl_grad_accum_steps', 1)
        self.use_ema = config.get('ssl_use_ema', True)

        self.ema_decay = config.get('ssl_ema_decay', 0.996)

        # SSL loss config
        self.temperature = config.get('ssl_temperature', 0.07)
        self.w_contrastive = config.get('ssl_w_contrastive', 1.0)
        self.w_lesion_type = config.get('ssl_w_lesion_type', 0.5)
        self.w_lesion_count = config.get('ssl_w_lesion_count', 0.3)
        self.w_reconstruction = config.get('ssl_w_reconstruction', 0.2)

        # Ablation flags
        self.use_contrastive = config.get('ssl_use_contrastive', True)
        self.use_multitask = config.get('ssl_use_multitask', True)

        # Loss function
        self.criterion = SSLCombinedLoss(
            temperature=self.temperature,
            w_contrastive=self.w_contrastive,
            w_lesion_type=self.w_lesion_type,
            w_lesion_count=self.w_lesion_count,
            w_reconstruction=self.w_reconstruction,
            use_contrastive=self.use_contrastive,
            use_multitask=self.use_multitask
        ).to(device)

        # Optimizer: separate LR for backbone vs heads
        backbone_params = list(self.model.backbone.parameters())
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith('backbone.')]

        param_groups = [
            {'params': backbone_params, 'lr': self.lr * self.backbone_lr_mult, 'weight_decay': self.weight_decay},
        ]
        if len(head_params) > 0:
            param_groups.append({'params': head_params, 'lr': self.lr, 'weight_decay': self.weight_decay})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # LR Scheduler: cosine with warmup
        self.scheduler = self._build_scheduler()

        # AMP scaler
        self.scaler = GradScaler()

        # EMA
        if self.use_ema:
            self.ema = EMAModel(self.model, decay=self.ema_decay)
        else:
            self.ema = None

        # Logging
        self.loss_history = defaultdict(list)

        # Save paths
        self.save_dir = config.get('ssl_save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

    def _build_scheduler(self):
        """Cosine annealing with linear warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _save_checkpoint(self, epoch):
        """Save full training state for resume after session interruption."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss_history': dict(self.loss_history),
        }
        if self.ema is not None:
            checkpoint['ema_shadow'] = {k: v.cpu() for k, v in self.ema.shadow.items()}

        checkpoint_path = os.path.join(self.save_dir, 'ssl_training_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"  Training checkpoint saved (epoch {epoch+1}) → {checkpoint_path}")

    def _load_checkpoint(self):
        """Load training state from checkpoint if available. Returns start_epoch."""
        checkpoint_path = os.path.join(self.save_dir, 'ssl_training_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            self.logger.info("No checkpoint found. Starting from epoch 0.")
            return 0

        self.logger.info(f"Found checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.loss_history = defaultdict(list, checkpoint['loss_history'])

        if self.ema is not None and 'ema_shadow' in checkpoint:
            for name, param in self.model.named_parameters():
                if name in checkpoint['ema_shadow']:
                    self.ema.shadow[name] = checkpoint['ema_shadow'][name].to(self.device)

        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"  Resumed from epoch {start_epoch} (checkpoint was at epoch {checkpoint['epoch']+1})")
        return start_epoch

    def train(self):
        """Run SSL pretraining for all epochs. Auto-resumes from checkpoint if available."""
        self.logger.info("=" * 60)
        self.logger.info("Starting SSL Pretraining")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Backbone LR: {self.lr * self.backbone_lr_mult:.2e}")
        self.logger.info(f"  Head LR: {self.lr:.2e}")
        self.logger.info(f"  Contrastive: {self.use_contrastive}, Multi-task: {self.use_multitask}")
        self.logger.info(f"  Temperature: {self.temperature}")
        self.logger.info(f"  Dataset size: {len(self.train_loader.dataset)} images")
        self.logger.info(f"  Batches per epoch: {len(self.train_loader)}")
        self.logger.info(f"  Grad accumulation steps: {self.grad_accum_steps}")
        self.logger.info(f"  Effective batch size: {self.train_loader.batch_size * self.grad_accum_steps}")

        # Check for existing checkpoint to resume
        start_epoch = self._load_checkpoint()
        if start_epoch >= self.epochs:
            self.logger.info(f"  Already completed {start_epoch} epochs. Training finished!")
            return dict(self.loss_history)

        self.logger.info("=" * 60)

        for epoch in range(start_epoch, self.epochs):

            epoch_start = time.time()
            self.model.train()

            epoch_losses = defaultdict(float)
            num_batches = 0
            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                view1 = batch['view1'].to(self.device, non_blocking=True)
                view2 = batch['view2'].to(self.device, non_blocking=True)
                lesion_mask = batch['lesion_mask'].to(self.device, non_blocking=True)
                lesion_count = batch['lesion_count'].to(self.device, non_blocking=True)
                lesion_presence = batch['lesion_presence'].to(self.device, non_blocking=True)

                # Forward pass with AMP
                with autocast():
                    outputs = self.model(view1, view2)
                    loss, loss_dict = self.criterion(
                        outputs, lesion_mask, lesion_count, lesion_presence
                    )
                    # Scale loss by accumulation steps
                    loss = loss / self.grad_accum_steps

                # Backward pass (accumulate gradients)
                self.scaler.scale(loss).backward()

                # Step optimizer every grad_accum_steps
                if (step + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # EMA update
                    if self.ema is not None:
                        self.ema.update(self.model)

                # Track losses
                for k, v in loss_dict.items():
                    epoch_losses[k] += v
                num_batches += 1


            # Scheduler step
            self.scheduler.step()

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= num_batches
                self.loss_history[k].append(epoch_losses[k])

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(
                f"SSL Epoch {epoch+1}/{self.epochs} "
                f"| Loss: {epoch_losses['total']:.4f} "
                f"| Contrastive: {epoch_losses['contrastive']:.4f} "
                f"| Type: {epoch_losses['lesion_type']:.4f} "
                f"| Count: {epoch_losses['lesion_count']:.4f} "
                f"| Recon: {epoch_losses['reconstruction']:.4f} "
                f"| LR: {current_lr:.2e} "
                f"| Time: {epoch_time:.1f}s"
            )

            # Save backbone + training checkpoint every 10 epochs and at the end
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                self._save_backbone(epoch)
                self._save_checkpoint(epoch)


        # Final save
        self._save_backbone(self.epochs - 1, final=True)
        self._save_loss_history()

        self.logger.info("SSL Pretraining complete!")
        self.logger.info(f"Backbone weights saved to {os.path.join(self.save_dir, 'ssl_pretrained_backbone.pth')}")

        return dict(self.loss_history)

    def _save_backbone(self, epoch, final=False):
        """Save the SSL-pretrained backbone weights."""
        # Use EMA weights if available
        if self.ema is not None:
            self.ema.apply_to(self.model)

        save_path = os.path.join(self.save_dir, 'ssl_pretrained_backbone.pth')
        self.model.save_backbone(save_path)

        if not final:
            # Also save epoch-specific checkpoint
            epoch_path = os.path.join(self.save_dir, f'ssl_backbone_epoch{epoch+1}.pth')
            self.model.save_backbone(epoch_path)

    def _save_loss_history(self):
        """Save loss history as numpy arrays for plotting."""
        save_path = os.path.join('results', 'ssl_loss_history.npz')
        os.makedirs('results', exist_ok=True)
        np.savez(save_path, **dict(self.loss_history))
        self.logger.info(f"SSL loss history saved to {save_path}")
