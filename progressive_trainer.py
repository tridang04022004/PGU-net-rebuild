import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from math import ceil
from pathlib import Path

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")

from unet import ProgressiveUNet, PGUNet1, PGUNet2, PGUNet3, PGUNet4


class ProgressiveTrainer:
    """
    Progressive Growing trainer for U-Net segmentation
    
    This trainer implements the progressive growing strategy where:
    1. Training starts with a simple network at low resolution
    2. Gradually increases network complexity and resolution
    3. Transfers weights between stages for stable training
    """
    
    def __init__(self, in_channels=3, num_classes=4, device='cuda'):
        self.device = device
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Stage configurations
        self.stage_configs = {
            1: {'resolution': 32, 'epochs_per_stage': 40, 'lr': 3e-4},
            2: {'resolution': 64, 'epochs_per_stage': 40, 'lr': 1e-4},
            3: {'resolution': 128, 'epochs_per_stage': 40, 'lr': 1e-4},
            4: {'resolution': 256, 'epochs_per_stage': 40, 'lr': 1e-4}
        }
        
        # Initialize models for each stage
        self.models = {
            1: PGUNet1(in_channels, num_classes).to(device),
            2: PGUNet2(in_channels, num_classes).to(device),
            3: PGUNet3(in_channels, num_classes).to(device),
            4: PGUNet4(in_channels, num_classes).to(device)
        }
        
        self.current_stage = 1
        self.current_model = self.models[1]
        
        # Training components - Use weighted loss for class imbalance
        # Nuclei are typically much smaller than background
        pos_weight = torch.tensor([5.0]).to(device)  # Weight positive class more
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = None
        self.setup_optimizer(1)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'stage_transitions': []
        }

    def setup_optimizer(self, stage):
        """Setup optimizer for the current stage"""
        lr = self.stage_configs[stage]['lr']
        self.optimizer = optim.RMSprop(
            self.current_model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )

    def dice_coefficient(self, pred, target, smooth=1):
        """Calculate Dice coefficient for binary segmentation evaluation"""
        # Ensure both tensors are on the same device
        pred = pred.to(target.device)
        target = target.to(pred.device)
        
        # Ensure pred is binary (0 or 1) and target is binary
        pred = pred.contiguous().float()
        target = target.contiguous().float()
        
        # Flatten tensors for calculation
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        
        dice = (2. * intersection + smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
        )
        
        return dice.mean()

    def get_predictions(self, output_batch):
        """Convert model output to predictions for binary segmentation"""
        # For binary segmentation with BCEWithLogitsLoss
        # Apply sigmoid to get probabilities, then threshold
        probs = torch.sigmoid(output_batch)
        preds = (probs > 0.5).float()
        return preds.squeeze(1)  # Remove channel dimension for binary segmentation

    def calculate_accuracy(self, pred, target):
        """Calculate pixel-wise accuracy"""
        # Ensure both tensors are on the same device
        pred = pred.to(target.device)
        assert pred.size() == target.size()
        bs, h, w = pred.size()
        n_pixels = bs * h * w
        incorrect = pred.ne(target).cpu().sum().numpy()
        err = incorrect / n_pixels
        return 1 - err

    def transfer_weights(self, prev_stage, new_stage):
        """
        Transfer weights from previous stage to new stage
        This is a simplified implementation - in practice, you'd need
        more sophisticated layer mapping
        """
        print(f"Transferring weights from stage {prev_stage} to stage {new_stage}")
        
        # Get state dictionaries
        prev_dict = self.models[prev_stage].state_dict()
        new_dict = self.models[new_stage].state_dict()
        
        # Simple weight transfer strategy
        # In practice, this would need careful mapping based on layer correspondence
        transfer_dict = {}
        
        # Transfer common layers (this is a simplified approach)
        prev_keys = list(prev_dict.keys())
        new_keys = list(new_dict.keys())
        
        # Map common layer names and transfer weights
        for new_key in new_keys:
            # Look for corresponding keys in previous stage
            for prev_key in prev_keys:
                if self._keys_match(prev_key, new_key):
                    if prev_dict[prev_key].shape == new_dict[new_key].shape:
                        transfer_dict[new_key] = prev_dict[prev_key]
                        break
            
            # If no match found, keep the new model's initialized weights
            if new_key not in transfer_dict:
                transfer_dict[new_key] = new_dict[new_key]
        
        # Load the transferred weights
        self.models[new_stage].load_state_dict(transfer_dict)
        print(f"Weight transfer completed. Transferred {len([k for k in transfer_dict.keys() if k in prev_dict.keys()])} layers.")

    def _keys_match(self, prev_key, new_key):
        """Check if two layer keys represent similar layers"""
        # Simple matching strategy - can be improved
        prev_parts = prev_key.split('.')
        new_parts = new_key.split('.')
        
        # Check for similar layer structure
        if len(prev_parts) >= 2 and len(new_parts) >= 2:
            # Match based on layer type and position
            return prev_parts[-1] == new_parts[-1] and prev_parts[-2] == new_parts[-2]
        
        return False

    def train_epoch(self, dataloader, stage):
        """Train for one epoch"""
        self.current_model.train()
        total_loss = 0
        total_dice = 0
        total_accuracy = 0
        
        resolution = self.stage_configs[stage]['resolution']
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Resize inputs to current stage resolution
            data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
            target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.current_model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            pred = self.get_predictions(output)
            # Ensure pred is on the same device as target for metric calculations
            pred = pred.to(self.device)
            target_squeezed = target.squeeze(1)  # Remove channel dimension for metrics
            dice = self.dice_coefficient(pred.float(), target_squeezed.float())
            accuracy = self.calculate_accuracy(pred, target_squeezed.long())
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_accuracy += accuracy
            
            if batch_idx % 10 == 0:
                print(f'Stage {stage}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Dice: {dice.item():.4f}, Acc: {accuracy:.4f}')
        
            avg_loss = total_loss / len(dataloader)
            avg_dice = total_dice / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)

            # Report number of batches processed
            print(f"Stage {stage} training epoch completed. Batches processed: {len(dataloader)}")

            return avg_loss, avg_dice, avg_accuracy

    def validate_epoch(self, dataloader, stage):
        """Validate for one epoch"""
        self.current_model.eval()
        total_loss = 0
        total_dice = 0
        total_accuracy = 0
        
        resolution = self.stage_configs[stage]['resolution']
        
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Resize inputs to current stage resolution
                data = F.interpolate(data, size=(resolution, resolution), mode='bilinear', align_corners=True)
                target = F.interpolate(target, size=(resolution, resolution), mode='nearest')
                
                # Forward pass
                output = self.current_model(data)
                loss = self.criterion(output, target)
                
                # Calculate metrics
                pred = self.get_predictions(output)
                # Ensure pred is on the same device as target for metric calculations
                pred = pred.to(self.device)
                target_squeezed = target.squeeze(1)  # Remove channel dimension for metrics
                dice = self.dice_coefficient(pred.float(), target_squeezed.float())
                accuracy = self.calculate_accuracy(pred, target_squeezed.long())
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / len(dataloader)
        avg_dice = total_dice / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        # Report number of batches processed
        print(f"Stage {stage} validation epoch completed. Batches processed: {len(dataloader)}")

        return avg_loss, avg_dice, avg_accuracy

    def train_progressive(self, train_loader, val_loader, max_stages=4, save_dir='./progressive_weights'):
        """
        Train the progressive U-Net through all stages
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print("Starting Progressive Growing U-Net Training")
        print("=" * 50)
        
        for stage in range(1, max_stages + 1):
            print(f"\nStarting Stage {stage}")
            print(f"Resolution: {self.stage_configs[stage]['resolution']}x{self.stage_configs[stage]['resolution']}")
            print("-" * 30)
            
            # Switch to new stage
            if stage > 1:
                self.transfer_weights(stage - 1, stage)
            
            self.current_stage = stage
            self.current_model = self.models[stage]
            self.setup_optimizer(stage)
            
            # Record stage transition
            self.history['stage_transitions'].append(len(self.history['train_loss']))
            
            # Training for this stage
            epochs = self.stage_configs[stage]['epochs_per_stage']
            best_val_dice = 0
            
            for epoch in range(epochs):
                start_time = time.time()
                
                # Train and validate
                train_loss, train_dice, train_acc = self.train_epoch(train_loader, stage)
                val_loss, val_dice, val_acc = self.validate_epoch(val_loader, stage)
                
                # Record history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_dice'].append(train_dice)
                self.history['val_dice'].append(val_dice)
                
                epoch_time = time.time() - start_time
                
                print(f'Stage {stage}, Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)')
                print(f'Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, Acc: {train_acc:.4f}')
                print(f'Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}')
                
                # Save best model for this stage
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save({
                        'stage': stage,
                        'epoch': epoch,
                        'model_state_dict': self.current_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_dice': val_dice,
                        'train_dice': train_dice,
                        'history': self.history
                    }, save_path / f'pgunet_stage{stage}_best.pth')
                
                print("-" * 50)
        
        print("Progressive training completed!")
        self.save_training_plots(save_path)

    def save_training_plots(self, save_path):
        """Save training history plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Skipping plot generation.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.history['train_loss']))
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Add stage transition lines
        for transition in self.history['stage_transitions']:
            ax1.axvline(x=transition, color='red', linestyle='--', alpha=0.7)
        
        # Dice plot
        ax2.plot(epochs, self.history['train_dice'], label='Train Dice')
        ax2.plot(epochs, self.history['val_dice'], label='Val Dice')
        ax2.set_title('Dice Coefficient Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.legend()
        ax2.grid(True)
        
        # Add stage transition lines
        for transition in self.history['stage_transitions']:
            ax2.axvline(x=transition, color='red', linestyle='--', alpha=0.7)
        
        # Stage-wise performance
        stage_epochs = [40, 40, 40, 40]  # epochs per stage
        stage_starts = [0] + list(np.cumsum(stage_epochs)[:-1])
        stage_dice = []
        
        for i, start in enumerate(stage_starts):
            end = start + stage_epochs[i]
            if end <= len(self.history['val_dice']):
                stage_dice.append(np.mean(self.history['val_dice'][start:end]))
        
        ax3.bar(range(1, len(stage_dice) + 1), stage_dice)
        ax3.set_title('Average Dice per Stage')
        ax3.set_xlabel('Stage')
        ax3.set_ylabel('Average Dice Coefficient')
        ax3.grid(True)
        
        # Resolution progression
        resolutions = [32, 64, 128, 256]
        ax4.plot(range(1, len(stage_dice) + 1), resolutions[:len(stage_dice)], 'o-')
        ax4.set_title('Resolution Progression')
        ax4.set_xlabel('Stage')
        ax4.set_ylabel('Resolution')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path / 'training_history.png'}")

    def load_model(self, checkpoint_path, stage):
        """Load a trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.models[stage].load_state_dict(checkpoint['model_state_dict'])
        self.current_stage = stage
        self.current_model = self.models[stage]
        print(f"Loaded model from {checkpoint_path} (Stage {stage})")
        return checkpoint

    def predict(self, data, stage=None, target_resolution=None):
        """Make predictions with the model"""
        if stage is None:
            stage = self.current_stage
        
        model = self.models[stage]
        model.eval()
        
        if target_resolution is None:
            target_resolution = self.stage_configs[stage]['resolution']
        
        with torch.no_grad():
            data = data.to(self.device)
            data = F.interpolate(data, size=(target_resolution, target_resolution), 
                               mode='bilinear', align_corners=True)
            output = model(data)
            pred = self.get_predictions(output)
        
        return pred


def create_dummy_dataloader(batch_size=8, num_samples=100, image_size=256):
    """Create dummy data loaders for testing"""
    import torch.utils.data as data
    
    class DummyDataset(data.Dataset):
        def __init__(self, num_samples, image_size):
            self.num_samples = num_samples
            self.image_size = image_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create dummy RGB image
            image = torch.randn(3, self.image_size, self.image_size)
            # Create dummy segmentation mask (4 classes)
            mask = torch.randint(0, 4, (self.image_size, self.image_size))
            return image, mask
    
    dataset = DummyDataset(num_samples, image_size)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Example usage
    print("Progressive Growing U-Net Training Example")
    print("=" * 50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data loaders (replace with your actual data loaders)
    train_loader = create_dummy_dataloader(batch_size=4, num_samples=50, image_size=256)
    val_loader = create_dummy_dataloader(batch_size=4, num_samples=20, image_size=256)
    
    # Initialize trainer
    trainer = ProgressiveTrainer(in_channels=3, num_classes=4, device=device)
    
    # Start progressive training
    trainer.train_progressive(
        train_loader=train_loader,
        val_loader=val_loader,
        max_stages=4,
        save_dir='./progressive_weights'
    )
    
    print("\nTraining completed! Check './progressive_weights' for saved models and plots.")