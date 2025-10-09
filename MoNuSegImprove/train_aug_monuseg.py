"""
Training script for Progressive Growing U-Net on the augmented MoNuSeg dataset

This script is a variant of train_monuseg.py that loads augmented training
data from the `train/aug` folder (via `AugMoNuSegDataset`) while keeping
validation on the standard `val` split. Place this file in the `MoNuSegImprove`
folder alongside `aug_monuseg_dataset.py` and `monuseg_dataset.py`.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import argparse

# Ensure project root is on path so we can import unet and progressive_trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet import ProgressiveUNet
from progressive_trainer import ProgressiveTrainer
from monuseg_dataset import MoNuSegDataset, create_train_val_split
from aug_monuseg_dataset import AugMoNuSegDataset


class AugMoNuSegTrainer(ProgressiveTrainer):
    """
    Trainer that uses the augmented training dataset (train/aug) for training.
    """

    def __init__(self, config):
        self.config = config
        super().__init__(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            device=config['device']
        )

        epochs_per_stage = config.get('num_epochs_per_stage', 50)
        self.stage_configs = {
            1: {'resolution': 32, 'epochs_per_stage': epochs_per_stage, 'lr': 3e-4},
            2: {'resolution': 64, 'epochs_per_stage': epochs_per_stage, 'lr': 1e-4},
            3: {'resolution': 128, 'epochs_per_stage': epochs_per_stage, 'lr': 1e-4},
            4: {'resolution': 256, 'epochs_per_stage': epochs_per_stage, 'lr': 1e-4}
        }

    def setup_datasets(self):
        """Setup datasets: use augmented dataset for training and standard val for validation."""
        print("Setting up augmented MoNuSeg datasets...")

        # Create train/val split if needed (operates on train/images & train/annots)
        val_dir = os.path.join(self.config['data_dir'], 'val')
        if not os.path.exists(val_dir):
            print("Creating train/validation split (will not touch train/aug)...")
            create_train_val_split(
                self.config['data_dir'],
                val_ratio=self.config.get('val_ratio', 0.2)
            )

        self.train_datasets = {}
        self.val_datasets = {}

        for stage in range(1, 5):
            image_size = self.get_image_size_for_stage(stage)

            # Training uses augmented dataset located under train/aug
            self.train_datasets[stage] = AugMoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=image_size,
                transform=True,
                augment=True
            )

            # Validation uses the standard (non-augmented) val split
            self.val_datasets[stage] = MoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=image_size,
                split='val',
                transform=True,
                augment=False
            )

        print(f"Dataset setup complete:\n  Training samples (stage1): {len(self.train_datasets[1])}\n  Validation samples (stage1): {len(self.val_datasets[1])}")

        # Compute pos_weight automatically from training masks (use non-aug variant to be conservative)
        try:
            print("Computing positive class weight from training masks (using augmented dataset without augment)...")
            stats_ds = AugMoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=self.get_image_size_for_stage(4),
                transform=True,
                augment=False
            )

            total_pos = 0.0
            total_pix = 0
            for i in range(len(stats_ds)):
                _, mask = stats_ds[i]
                total_pos += mask.sum().item()
                total_pix += mask.numel()

            pos_ratio = (total_pos / total_pix) if total_pix > 0 else 0.0
            computed_pos_weight = float((1.0 - pos_ratio) / (pos_ratio + 1e-8))

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([computed_pos_weight]).to(self.device))
            print(f"Auto pos_weight={computed_pos_weight:.3f} (positive ratio={pos_ratio:.4f}) set for BCEWithLogitsLoss")
        except Exception as e:
            print(f"Warning: failed to compute pos_weight automatically: {e}. Using default criterion.")

    def get_image_size_for_stage(self, stage):
        sizes = {1: 32, 2: 64, 3: 128, 4: 256}
        return sizes[stage]

    # re-use other methods from base ProgressiveTrainer or override as needed


def create_config():
    """Create configuration that points to the MoNuSegImprove dataset root."""
    config = {
        # Data settings - point to MoNuSegImprove where train/aug lives
        'data_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSegImprove',
        'val_ratio': 0.2,

        # Model settings
        'in_channels': 3,
        'num_classes': 1,

        # Training settings
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs_per_stage': 50,
        'num_workers': 4,
        'log_interval': 10,

        # Progressive training stages
        'stages': [1, 2, 3, 4],

        # Output settings
        'output_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSegImprove\outputs',
        'save_interval': 10,

        # Optimization settings
        'weight_decay': 1e-4,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,

        # Device settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Progressive Growing U-Net on augmented MoNuSeg')
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3, 4], help='Training stages to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per stage')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')

    args = parser.parse_args()

    config = create_config()
    config['stages'] = args.stages
    config['num_epochs_per_stage'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr

    print("=== Augmented MoNuSeg Progressive Growing U-Net Training ===")
    print(f"Device: {config['device']}")
    print(f"Training stages: {config['stages']}")
    print(f"Epochs per stage: {config['num_epochs_per_stage']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 50)

    trainer = AugMoNuSegTrainer(config)
    trainer.setup_datasets()

    # Create data loaders using highest resolution dataset for training loops
    train_loader = DataLoader(
        trainer.train_datasets[4],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )

    val_loader = DataLoader(
        trainer.val_datasets[4],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )

    try:
        trainer.train_progressive(
            train_loader=train_loader,
            val_loader=val_loader,
            max_stages=len(config['stages']),
            save_dir=config['output_dir']
        )
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
