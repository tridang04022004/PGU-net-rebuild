"""
Training script for Progressive Growing U-Net on MoNuSeg dataset

This script trains the PGU-Net architecture on the Multi-organ Nuclei 
Segmentation dataset for nuclei segmentation.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet import ProgressiveUNet
from progressive_trainer import ProgressiveTrainer
from monuseg_dataset import MoNuSegDataset, create_train_val_split


class MoNuSegTrainer(ProgressiveTrainer):
    """
    Specialized trainer for MoNuSeg nuclei segmentation task.
    Inherits from the base ProgressiveTrainer with nuclei-specific modifications.
    """
    
    def __init__(self, config):
        self.config = config
        super().__init__(
            in_channels=config['in_channels'],
            num_classes=config['num_classes'], 
            device=config['device']
        )
        
        # Override stage configurations with config values
        epochs_per_stage = config.get('num_epochs_per_stage', 50)
        self.stage_configs = {
            1: {'resolution': 32, 'epochs_per_stage': epochs_per_stage, 'lr': 3e-4},
            2: {'resolution': 64, 'epochs_per_stage': epochs_per_stage, 'lr': 1e-4},
            3: {'resolution': 128, 'epochs_per_stage': epochs_per_stage, 'lr': 1e-4},
            4: {'resolution': 256, 'epochs_per_stage': epochs_per_stage, 'lr': 1e-4}
        }
        print(f"Stage configurations updated with {epochs_per_stage} epochs per stage")
        
    def setup_datasets(self):
        """Setup MoNuSeg datasets for training and validation"""
        print("Setting up MoNuSeg datasets...")
        
        # Create train/val split if needed
        val_dir = os.path.join(self.config['data_dir'], 'val')
        if not os.path.exists(val_dir):
            print("Creating train/validation split...")
            create_train_val_split(
                self.config['data_dir'], 
                val_ratio=self.config.get('val_ratio', 0.2)
            )
        
        # Create datasets
        self.train_datasets = {}
        self.val_datasets = {}
        
        for stage in range(1, 5):
            image_size = self.get_image_size_for_stage(stage)
            
            self.train_datasets[stage] = MoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=image_size,
                split='train',
                transform=True,
                augment=True
            )
            
            self.val_datasets[stage] = MoNuSegDataset(
                data_dir=self.config['data_dir'],
                image_size=image_size,
                split='val',
                transform=True,
                augment=False
            )
        
        print(f"Dataset setup complete:")
        print(f"  Training samples: {len(self.train_datasets[1])}")
        print(f"  Validation samples: {len(self.val_datasets[1])}")
    
    def get_image_size_for_stage(self, stage):
        """Get appropriate image size for each training stage"""
        sizes = {1: 32, 2: 64, 3: 128, 4: 256}
        return sizes[stage]
    
    def calculate_nuclei_metrics(self, outputs, targets):
        """
        Calculate nuclei-specific metrics including:
        - IoU (Intersection over Union)
        - Dice coefficient
        - Pixel accuracy
        - Precision and Recall
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        # Flatten tensors
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate metrics
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        # IoU (Jaccard Index)
        iou = (intersection + 1e-8) / (union + 1e-8)
        
        # Dice coefficient
        dice = (2 * intersection + 1e-8) / (preds_flat.sum() + targets_flat.sum() + 1e-8)
        
        # Pixel accuracy
        correct = (preds_flat == targets_flat).sum()
        accuracy = correct.float() / targets_flat.numel()
        
        # Precision and Recall
        true_positives = intersection
        predicted_positives = preds_flat.sum()
        actual_positives = targets_flat.sum()
        
        precision = (true_positives + 1e-8) / (predicted_positives + 1e-8)
        recall = (true_positives + 1e-8) / (actual_positives + 1e-8)
        
        return {
            'iou': iou.item(),
            'dice': dice.item(),
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }
    
    def train_epoch(self, dataloader, stage):
        """Enhanced training with nuclei-specific metrics logging"""
        self.current_model.train()
        train_loss = 0.0
        train_metrics = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        resolution = self.stage_configs[stage]['resolution']
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Resize inputs to current stage resolution
            import torch.nn.functional as F
            data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
            target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
            
            self.optimizer.zero_grad()
            output = self.current_model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate nuclei metrics
            batch_metrics = self.calculate_nuclei_metrics(output, target)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]
            
            # Log batch progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f'Stage {stage}, Batch {batch_idx}/{len(dataloader)}: '
                      f'Loss: {loss.item():.6f}, Dice: {batch_metrics["dice"]:.4f}')
        
        # Average metrics
        num_batches = len(dataloader)
        train_loss /= num_batches
        for key in train_metrics:
            train_metrics[key] /= num_batches

        # Report batches processed for visibility
        print(f"Stage {stage} training epoch completed. Batches processed: {num_batches}")

        return train_loss, train_metrics['dice'], train_metrics['accuracy']
    
    def validate_epoch(self, dataloader, stage):
        """Enhanced validation with nuclei-specific metrics"""
        self.current_model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        resolution = self.stage_configs[stage]['resolution']
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Resize inputs to current stage resolution
                import torch.nn.functional as F
                data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
                target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
                
                output = self.current_model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                
                # Calculate nuclei metrics
                batch_metrics = self.calculate_nuclei_metrics(output, target)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        # Average metrics
        num_batches = len(dataloader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches

        # Report batches processed for visibility
        print(f"Stage {stage} validation epoch completed. Batches processed: {num_batches}")

        return val_loss, val_metrics['dice'], val_metrics['accuracy']


def create_config():
    """Create configuration for MoNuSeg training"""
    config = {
        # Data settings
        'data_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSeg',
        'val_ratio': 0.2,
        
        # Model settings
        'in_channels': 3,
        'num_classes': 1,  # Binary segmentation for nuclei
        
        # Training settings
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs_per_stage': 50,
        'num_workers': 4,
        'log_interval': 10,
        
        # Progressive training stages
        'stages': [1, 2, 3, 4],  # 32x32, 64x64, 128x128, 256x256
        
        # Output settings
        'output_dir': r'd:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSeg\outputs',
        'save_interval': 10,  # Save checkpoint every 10 epochs
        
        # Optimization settings
        'weight_decay': 1e-4,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        
        # Device settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Progressive Growing U-Net on MoNuSeg')
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='Training stages to run (default: all stages)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per stage')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config()
    config['stages'] = args.stages
    config['num_epochs_per_stage'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    print("=== MoNuSeg Progressive Growing U-Net Training ===")
    print(f"Device: {config['device']}")
    print(f"Training stages: {config['stages']}")
    print(f"Epochs per stage: {config['num_epochs_per_stage']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print("=" * 50)
    
    # Create trainer
    trainer = MoNuSegTrainer(config)
    
    # Setup datasets
    trainer.setup_datasets()
    
    # Create data loaders
    train_loader = DataLoader(
        trainer.train_datasets[4],  # Use highest resolution dataset
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        trainer.val_datasets[4],  # Use highest resolution dataset
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Start training
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