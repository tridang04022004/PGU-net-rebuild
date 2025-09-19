## MoNuSeg for training and testing 



# Progressive Growing U-Net for Pap Smear Segmentation

A complete implementation of Progressive Growing U-Net (PGU-Net) for medical image segmentation, specifically designed for Pap smear cell segmentation. This implementation follows the progressive growing strategy where the model is trained progressively from low resolution (32×32) to high resolution (256×256).

## 🏗️ Architecture Overview

The Progressive Growing U-Net consists of 4 stages:

- **Stage 1**: 32×32 resolution, simplified encoder-decoder
- **Stage 2**: 64×64 resolution, adds one more encoder/decoder layer
- **Stage 3**: 128×128 resolution, full encoder with 3 levels
- **Stage 4**: 256×256 resolution, complete U-Net with 4 levels

Each stage builds upon the previous one, with weight transfer mechanisms for stable training.

## 🚀 Key Features

- **Progressive Growing Training**: Train from 32×32 → 64×64 → 128×128 → 256×256 resolution
- **Multi-Scale Output Fusion**: Combines predictions from different resolution levels
- **Weight Transfer**: Transfers compatible weights between progressive stages
- **Medical Image Dataset**: Custom dataset class for Pap smear images with extensive augmentation
- **Comprehensive Training**: Complete training pipeline with logging, checkpointing, and evaluation
- **Flexible Architecture**: Supports both progressive and standard training modes

## 📋 Requirements

```
numpy==2.2.1
matplotlib==3.10.5
scikit-image==0.24.0
scipy==1.13.1
pyparsing==3.1.4
pytorch==2.5.1
torchvision
torchaudio
pytorch-cuda=11.8  # if using CUDA
```

## 🛠️ Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd pgu-net-rebuild
```

2. Install dependencies:

```bash
# Using conda (recommended)
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy==2.2.1 matplotlib==3.10.5 scikit-image==0.24.0 scipy==1.13.1 pyparsing==3.1.4

# Or using pip
pip install -r requirements.txt
```

## 📁 Project Structure

```text
pgu-net-rebuild/
├── unet_parts.py            # U-Net building blocks
├── unet.py                  # Progressive U-Net model definitions
├── progressive_trainer.py   # Training framework with progressive strategy
├── pap_smear_dataset.py     # Dataset class for Pap smear data
├── train_papsmear.py        # Complete training script
├── test_implementation.py   # Test suite for all components
├── test_compatibility.py    # Compatibility tests
├── example_usage.py         # Usage examples
├── requirements.txt         # Python dependencies
└── data/                    # Dataset directory
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── valid/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

## 🛠️ Installation

1. Install dependencies:

```bash
# Using conda (recommended)
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy==2.2.1 matplotlib==3.10.5 scikit-image==0.24.0 scipy==1.13.1 pyparsing==3.1.4

# Or using pip
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Expected Data Structure

```text
data/
├── train/
│   ├── images/          # Training images (.jpg, .png)
│   └── masks/           # Corresponding masks (.jpg, .png)
├── valid/
│   ├── images/          # Validation images
│   └── masks/           # Corresponding masks
└── test/
    ├── images/          # Test images
    └── masks/           # Corresponding masks
```

### Data Requirements

- **Image Format**: JPG, JPEG, or PNG
- **Mask Format**: Binary masks (0 for background, 255 for foreground)
- **Naming**: Image and mask files must have identical names
- **Size**: Any size (will be resized during training)

### Supported Augmentations

The dataset automatically detects and includes pre-augmented images with prefixes:

- `flipped_*`: Horizontally flipped images
- `mirrored_*`: Vertically flipped images
- `rotated_X_*`: Rotated images (where X is the rotation angle)

## 🏃‍♂️ Quick Start

### 1. Test the Implementation

```bash
python test_implementation.py
```

### 2. Progressive Training (Recommended)

```bash
python train_papsmear.py \
    --data_root data \
    --progressive \
    --epochs_per_stage 50 \
    --output_dir outputs \
    --experiment_name my_pgunet_experiment
```

### 3. Standard Training (Single Resolution)

```bash
python train_papsmear.py \
    --data_root data \
    --progressive false \
    --epochs_per_stage 200 \
    --output_dir outputs
```

## 📚 Usage Examples

### Basic Progressive Training

```python
from unet import ProgressiveUNet
from progressive_trainer import ProgressiveTrainer
from pap_smear_dataset import create_progressive_dataloaders

# Create model
model = ProgressiveUNet(in_channels=1, out_channels=1)

# Create trainer
trainer = ProgressiveTrainer(model=model, device='cuda')

# Create dataloaders for all resolutions
dataloaders = create_progressive_dataloaders(
    data_root='data',
    batch_sizes={32: 64, 64: 32, 128: 16, 256: 8}
)

# Training loop
for stage in [1, 2, 3, 4]:
    resolution = [32, 64, 128, 256][stage-1]
    train_loader, valid_loader, _ = dataloaders[resolution]

    trainer.set_stage(stage)

    for epoch in range(50):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate_epoch(valid_loader, epoch)
        print(f"Stage {stage}, Epoch {epoch}: Loss={val_metrics['loss']:.4f}")
```

### Custom Dataset Usage

```python
from pap_smear_dataset import PapSmearDataset

# Create dataset
dataset = PapSmearDataset(
    data_root='data',
    split='train',
    target_size=256,
    use_augmented=True,
    augment_probability=0.3
)

# Get a sample
sample = dataset[0]
image = sample['image']  # [1, 256, 256]
mask = sample['mask']    # [1, 256, 256]

# Update resolution for progressive training
dataset.update_target_size(128)
```

## ⚙️ Configuration Options

### Training Arguments

| Argument             | Default                    | Description                                        |
| -------------------- | -------------------------- | -------------------------------------------------- |
| `--data_root`        | `data`                     | Root directory containing train/valid/test folders |
| `--progressive`      | `True`                     | Use progressive growing training                   |
| `--epochs_per_stage` | `50`                       | Number of epochs per progressive stage             |
| `--lr`               | `1e-3`                     | Learning rate                                      |
| `--batch_sizes`      | `32:64,64:32,128:16,256:8` | Batch sizes for each resolution                    |
| `--output_dir`       | `outputs`                  | Output directory for logs and checkpoints          |
| `--use_augmented`    | `True`                     | Use pre-augmented training data                    |

### Model Architecture

- **Progressive Stages**: 4 stages (32×32, 64×64, 128×128, 256×256)
- **Input Channels**: 1 (grayscale medical images)
- **Output Channels**: 1 (binary segmentation mask)
- **Architecture**: U-Net with progressive growing capability

## 📈 Training Process

### Progressive Growing Strategy

1. **Stage 1 (32×32)**: Learn basic structure and shape information
2. **Stage 2 (64×64)**: Add more detailed features, transfer weights from Stage 1
3. **Stage 3 (128×128)**: Refine boundaries and textures, transfer weights from Stage 2
4. **Stage 4 (256×256)**: Final high-resolution details, transfer weights from Stage 3

### Loss Function

- **Primary**: Binary Cross-Entropy Loss
- **Regularization**: Dice Loss for better segmentation
- **Combined**: BCE + Dice Loss for optimal performance

### Metrics

- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Area overlap metric
- **Loss**: Combined BCE + Dice loss

### Key Training Features

- **Weight Transfer**: Automatically transfers compatible weights between stages
- **Multi-scale Loss**: Each stage outputs predictions at multiple scales
- **Adaptive Resolution**: Input images are automatically resized for each stage
- **Comprehensive Metrics**: Tracks loss, Dice coefficient, and accuracy

## 📊 Results and Logging

### Output Structure

```text
outputs/
└── experiment_name/
    ├── config.json              # Training configuration
    ├── logs/                    # TensorBoard logs
    ├── best_stage_1.pth         # Best model for each stage
    ├── best_stage_2.pth
    ├── best_stage_3.pth
    ├── best_stage_4.pth
    ├── final_model.pth          # Final trained model
    └── stage_X_epoch_Y.pth      # Regular checkpoints
```

### TensorBoard Visualization

```bash
tensorboard --logdir outputs/experiment_name/logs
```

View training progress with:

- Loss curves for each stage
- Dice coefficient progression
- IoU metrics
- Training time per epoch

## 🔧 Advanced Usage

### Multi-Scale Inference

```python
# Get predictions at all scales
model.set_stage(4)
outputs = model(input_image, return_all_scales=True)

# outputs[0]: 32×32 prediction
# outputs[1]: 64×64 prediction
# outputs[2]: 128×128 prediction
# outputs[3]: 256×256 prediction
```

## � Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch sizes: `--batch_sizes 32:32,64:16,128:8,256:4`
   - Use CPU training: `--device cpu`

2. **Dataset Not Found**

   - Check data directory structure
   - Ensure image/mask pairs have identical names
   - Verify file extensions (.jpg, .png)

3. **Poor Convergence**

   - Increase epochs per stage: `--epochs_per_stage 100`
   - Adjust learning rate: `--lr 5e-4`
   - Enable data augmentation: `--use_augmented`

4. **Memory Issues**
   - Reduce number of workers: `--num_workers 2`
   - Disable image caching in dataset

### Performance Tips

1. **For Small Datasets**

   - Enable image caching: `cache_images=True` in dataset
   - Use more epochs per stage
   - Higher augmentation probability

2. **For Large Datasets**
   - Use more workers: `--num_workers 8`
   - Larger batch sizes if memory allows
   - Consider gradient accumulation

## 📚 References

1. **PGU-net+**: Progressive Growing of U-net+ for Automated Cervical Nuclei Segmentation (MICCAI 2019)
2. **Progressive GAN**: Progressive Growing of GANs for Improved Quality, Stability, and Variation
3. **U-Net**: Convolutional Networks for Biomedical Image Segmentation

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original U-Net paper by Ronneberger et al.
- Progressive GAN concept by Karras et al.
- PyTorch deep learning framework
- Medical imaging community for Pap smear datasets

---

## Getting Started

Happy training with Progressive Growing U-Net! 🚀
