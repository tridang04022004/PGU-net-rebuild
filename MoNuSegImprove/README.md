# MoNuSeg Progressive Growing U-Net Implementation

This directory contains a complete implementation of Progressive Growing U-Net for the Multi-organ Nuclei Segmentation (MoNuSeg) challenge dataset.

https://drive.google.com/file/d/19uiEHEwMO46YZCf-d8OWIGiTkkQMnd53/view?usp=sharing

## Overview

The MoNuSeg dataset contains 37 H&E stained tissue images (1000×1000 pixels) from multiple organs with pixel-wise nuclear boundary annotations in XML format. Our implementation uses a progressive growing strategy to train U-Net models at multiple resolutions.

## Dataset Structure

```
MoNuSeg/
├── train/
│   ├── images/           # 30 RGB TIFF images (1000×1000)
│   │   ├── 0001.tif
│   │   ├── 0002.tif
│   │   └── ...
│   └── annots/           # XML polygon annotations
│       ├── 0001.xml
│       ├── 0002.xml
│       └── ...
├── val/                  # 7 images (auto-created from train split)
│   ├── images/
│   └── annots/
└── outputs/              # Training outputs and checkpoints
```

## Features

### 🎯 **Multi-Resolution Training**

- **Stage 1**: 32×32 → Basic nuclear patterns
- **Stage 2**: 64×64 → Enhanced detail recognition
- **Stage 3**: 128×128 → Fine boundary detection
- **Stage 4**: 256×256 → High-resolution segmentation

### 🔬 **Nuclei-Specific Metrics**

- **IoU (Intersection over Union)**: Overlap accuracy
- **Dice Coefficient**: Segmentation similarity
- **Pixel Accuracy**: Overall correctness
- **Precision & Recall**: Detection performance
- **Specificity**: True negative rate

### 📊 **Comprehensive Evaluation**

- XML to binary mask conversion
- Progressive resolution support
- Data augmentation (rotation, flipping, color jitter)
- Visualization with overlay comparisons
- Quality analysis and integrity checks

## File Descriptions

| File                     | Purpose                                         |
| ------------------------ | ----------------------------------------------- |
| `monuseg_dataset.py`     | PyTorch dataset loader with XML parsing         |
| `train_monuseg.py`       | Progressive training script with nuclei metrics |
| `test_monuseg.py`        | Model evaluation and visualization              |
| `preprocessing_utils.py` | Data analysis and quality check utilities       |

## Quick Start

### 1. Dataset Preparation

```bash
# The dataset should be structured as shown above
# If you only have a train folder, the train/val split will be created automatically
```

### 2. Training

```bash
# Train all stages (32→64→128→256)
python train_monuseg.py

# Train specific stages only
python train_monuseg.py --stages 1 2 3

# Custom parameters
python train_monuseg.py --epochs 100 --batch_size 16 --lr 0.0005
```

### 3. Evaluation

```bash
# Quick test on random images
python test_monuseg.py --model outputs/stage_4_final.pth --data . --num_test 10

# Full dataset evaluation
python test_monuseg.py --model outputs/stage_4_final.pth --data . --eval_full --split val
```

### 4. Data Analysis

```bash
# Run preprocessing utilities for dataset analysis
python preprocessing_utils.py
```

## Training Configuration

### Default Parameters

```python
config = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'num_epochs_per_stage': 50,
    'val_ratio': 0.2,          # 20% for validation
    'weight_decay': 1e-4,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5
}
```

### Progressive Stages

- **Stage 1**: 32×32, 512 channels, 2 levels
- **Stage 2**: 64×64, 256 channels, 3 levels
- **Stage 3**: 128×128, 128 channels, 4 levels
- **Stage 4**: 256×256, 64 channels, 5 levels

## Expected Results

### Training Metrics (Stage 4)

- **Training Loss**: ~0.3-0.5 (BCEWithLogitsLoss)
- **Validation Dice**: ~0.75-0.85
- **Validation IoU**: ~0.65-0.75
- **Pixel Accuracy**: ~0.90-0.95

### Performance Characteristics

- **Training Time**: ~2-4 hours per stage (GPU)
- **Memory Usage**: ~6-8 GB (batch_size=8, 256×256)
- **Convergence**: Usually within 30-50 epochs per stage

## Nuclei Segmentation Challenges

### Dataset Characteristics

- **480 nuclei per image** (average)
- **Mask Coverage**: ~15-25% of image pixels
- **Multi-organ Sources**: Breast, liver, kidney, prostate, bladder, colon, stomach
- **Staining Variation**: H&E with varying intensity

### Technical Considerations

- **Boundary Precision**: Critical for downstream analysis
- **Small Object Detection**: Nuclei vary significantly in size
- **Overlapping Regions**: Dense tissue areas with touching nuclei
- **Annotation Quality**: Hand-drawn polygons with varying precision

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch size
   python train_monuseg.py --batch_size 4
   ```

2. **Dataset Not Found**

   ```bash
   # Check path structure matches expected format
   python preprocessing_utils.py  # Run integrity check
   ```

3. **Poor Convergence**
   ```bash
   # Try lower learning rate
   python train_monuseg.py --lr 0.0001
   ```

## Directory Structure After Training

```
MoNuSeg/
├── outputs/
│   ├── stage_1_epoch_50.pth
│   ├── stage_2_epoch_50.pth
│   ├── stage_3_epoch_50.pth
│   ├── stage_4_final.pth
│   └── training_log.txt
├── analysis/
│   ├── data_quality_report.png
│   └── sample_visualization.png
└── test_results/
    └── random_test_YYYYMMDD_HHMMSS/
        ├── prediction_001.png
        ├── prediction_002.png
        └── ...
```

## Citations

If you use this implementation, please cite:

```bibtex
@article{monuseg2018,
  title={Multi-organ nuclei segmentation challenge},
  author={Kumar, Neeraj and Verma, Ruchika and Sharma, Sanuj and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2018}
}
```

## Contact

For questions or issues related to this implementation, please check the main repository documentation or create an issue.

---

**Note**: This implementation is designed for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
