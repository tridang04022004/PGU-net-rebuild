"""
MoNuSeg Dataset Loader for Progressive Growing U-Net

This module provides a PyTorch Dataset class for loading the MoNuSeg 
(Multi-organ Nuclei Segmentation) dataset with XML polygon annotations.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Any
import cv2


class MoNuSegDataset(Dataset):
    """
    MoNuSeg dataset loader for nuclei segmentation.
    
    The dataset contains:
    - RGB TIFF images (1000x1000)
    - XML annotations with polygon coordinates for nuclei boundaries
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        split: str = 'train',
        transform: bool = True,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Path to MoNuSeg directory
            image_size: Target size for progressive training (32, 64, 128, 256)
            split: Dataset split ('train', 'val', 'test')
            transform: Whether to apply transforms
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # Setup paths
        self.images_dir = os.path.join(data_dir, split, 'images')
        self.annotations_dir = os.path.join(data_dir, split, 'annots')
        
        # Get file lists
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
        self.annotation_files = sorted([f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')])
        
        # Verify matching files
        assert len(self.image_files) == len(self.annotation_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.annotation_files)} annotations"
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"MoNuSeg {split} dataset: {len(self.image_files)} samples")
    
    def _setup_transforms(self):
        """Setup image transforms based on current settings"""
        # Base transforms for all images
        self.base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms for training
        if self.augment and self.split == 'train':
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.aug_transform = None
    
    def _parse_xml_annotations(self, xml_path: str, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Parse XML file and create binary mask from polygon annotations.
        
        Args:
            xml_path: Path to XML annotation file
            image_size: (width, height) of the image
            
        Returns:
            Binary mask as numpy array
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Create blank mask
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # PIL uses (W,H), numpy uses (H,W)
        
        # Find all regions (nuclei)
        regions = root.findall('.//Region')
        
        for region in regions:
            vertices = region.findall('.//Vertex')
            if len(vertices) < 3:  # Need at least 3 points for a polygon
                continue
                
            # Extract polygon coordinates
            polygon_points = []
            for vertex in vertices:
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
                polygon_points.append((x, y))
            
            # Draw filled polygon on mask using PIL
            mask_pil = Image.fromarray(mask)
            draw = ImageDraw.Draw(mask_pil)
            draw.polygon(polygon_points, fill=1)
            mask = np.array(mask_pil)
        
        return mask
    
    def _apply_joint_transforms(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms that need to be applied to both image and mask consistently.
        """
        # Convert to tensor and resize
        image_tensor = self.base_transform(image)
        
        # Resize mask to match image size
        mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized)).float()
        
        # Add channel dimension to mask
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # Apply augmentations if enabled (apply same random transform to both)
        if self.aug_transform and self.split == 'train':
            # For joint transforms, we need to use the same random state
            seed = torch.randint(0, 2**32, (1,)).item()
            
            # Apply to image
            torch.manual_seed(seed)
            image_tensor = self.aug_transform(transforms.ToPILImage()(image_tensor))
            image_tensor = transforms.ToTensor()(image_tensor)
            
            # Apply to mask
            torch.manual_seed(seed)
            mask_pil = transforms.ToPILImage()(mask_tensor)
            mask_tensor = self.aug_transform(mask_pil)
            mask_tensor = transforms.ToTensor()(mask_tensor)
        
        return image_tensor, mask_tensor
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Load and parse annotations
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        mask_array = self._parse_xml_annotations(annotation_path, image.size)
        mask = Image.fromarray(mask_array)
        
        # Apply transforms
        if self.transform:
            image_tensor, mask_tensor = self._apply_joint_transforms(image, mask)
        else:
            # Just convert to tensors without transforms
            image_tensor = transforms.ToTensor()(image)
            mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)
        
        return image_tensor, mask_tensor
    
    def update_image_size(self, new_size: int):
        """Update image size for progressive training"""
        self.image_size = new_size
        self._setup_transforms()
        print(f"Updated dataset image size to {new_size}x{new_size}")
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample"""
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        
        # Load image to get original size
        image = Image.open(image_path)
        
        # Parse XML to count nuclei
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        regions = root.findall('.//Region')
        
        return {
            'image_file': self.image_files[idx],
            'annotation_file': self.annotation_files[idx],
            'original_size': image.size,
            'num_nuclei': len(regions),
            'microns_per_pixel': float(root.attrib.get('MicronsPerPixel', 0.252))
        }


def create_train_val_split(data_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """
    Create train/validation split from the training data.
    Since MoNuSeg only provides a train folder, we need to split it.
    
    Args:
        data_dir: Path to MoNuSeg directory
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducible splits
    """
    import shutil
    import random
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Create validation directory structure
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'annots'), exist_ok=True)
    
    # Get all files
    image_files = sorted([f for f in os.listdir(os.path.join(train_dir, 'images')) if f.endswith('.tif')])
    
    # Split files
    random.seed(seed)
    n_val = int(len(image_files) * val_ratio)
    val_files = random.sample(image_files, n_val)
    
    print(f"Moving {n_val} files to validation set...")
    
    # Move validation files
    for img_file in val_files:
        # Corresponding annotation file
        annot_file = img_file.replace('.tif', '.xml')
        
        # Move image
        src_img = os.path.join(train_dir, 'images', img_file)
        dst_img = os.path.join(val_dir, 'images', img_file)
        shutil.move(src_img, dst_img)
        
        # Move annotation
        src_annot = os.path.join(train_dir, 'annots', annot_file)
        dst_annot = os.path.join(val_dir, 'annots', annot_file)
        shutil.move(src_annot, dst_annot)
    
    print(f"Train/Val split complete:")
    print(f"  Training: {len(os.listdir(os.path.join(train_dir, 'images')))} samples")
    print(f"  Validation: {len(os.listdir(os.path.join(val_dir, 'images')))} samples")


if __name__ == "__main__":
    # Test the dataset loader
    data_dir = r"d:\DangTri\Uni\NCKH\PGUnetPlus\project\pgu-net-rebuild\MoNuSeg"
    
    # Create train/val split if validation doesn't exist
    if not os.path.exists(os.path.join(data_dir, 'val')):
        print("Creating train/validation split...")
        create_train_val_split(data_dir, val_ratio=0.2)
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    dataset = MoNuSegDataset(data_dir, image_size=256, split='train')
    
    # Test a sample
    image, mask = dataset[0]
    print(f"Sample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask range: {mask.min():.3f} - {mask.max():.3f}")
    print(f"  Nuclei pixels: {mask.sum().item():.0f}")
    
    # Get sample info
    info = dataset.get_sample_info(0)
    print(f"  Sample info: {info}")