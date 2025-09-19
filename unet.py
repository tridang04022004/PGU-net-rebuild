import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_parts import DoubleConv, DownSample, UpSample, InConv, Down, Up, OutConv


class PGUNet1(nn.Module):
    """Progressive Growing U-Net Stage 1 - 32x32 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.outc = OutConv(256, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # Initial conv, maintains resolution
        x2 = self.down4(x1)     # Downsample by 2
        x3 = self.up1(x2, x1)   # Upsample by 2, skip connection
        x = self.outc(x3)       # Final output - raw logits
        return x


class PGUNet2(nn.Module):
    """Progressive Growing U-Net Stage 2 - 64x64 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.outc1 = OutConv(256, num_classes)
        self.outc2 = OutConv(128, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # 64x64
        x2 = self.down3(x1)     # 32x32
        x3 = self.down4(x2)     # 16x16
        x4 = self.up1(x3, x2)   # 32x32
        x5 = self.up2(x4, x1)   # 64x64
        
        # Multi-scale outputs
        x4_out = self.outc1(x4)
        x5_out = self.outc2(x5)
        
        # Interpolate and combine
        x4_out = F.interpolate(x4_out, scale_factor=2, mode='bilinear', align_corners=True)
        x = x4_out + x5_out  # Raw logits for BCEWithLogitsLoss
        return x


class PGUNet3(nn.Module):
    """Progressive Growing U-Net Stage 3 - 128x128 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.outc1 = OutConv(256, num_classes)
        self.outc2 = OutConv(128, num_classes)
        self.outc3 = OutConv(64, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # 128x128
        x2 = self.down2(x1)     # 64x64
        x3 = self.down3(x2)     # 32x32
        x4 = self.down4(x3)     # 16x16
        x5 = self.up1(x4, x3)   # 32x32
        x6 = self.up2(x5, x2)   # 64x64
        x7 = self.up3(x6, x1)   # 128x128
        
        # Multi-scale outputs
        x5_out = self.outc1(x5)
        x6_out = self.outc2(x6)
        x7_out = self.outc3(x7)
        
        # Interpolate and combine
        x5_out = F.interpolate(x5_out, scale_factor=4, mode='bilinear', align_corners=True)
        x6_out = F.interpolate(x6_out, scale_factor=2, mode='bilinear', align_corners=True)
        x = x5_out + x6_out + x7_out  # Raw logits for BCEWithLogitsLoss
        return x


class PGUNet4(nn.Module):
    """Progressive Growing U-Net Stage 4 - 256x256 resolution"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc1 = OutConv(256, num_classes)
        self.outc2 = OutConv(128, num_classes)
        self.outc3 = OutConv(64, num_classes)
        self.outc4 = OutConv(64, num_classes)
        # Removed LogSoftmax for binary segmentation with BCEWithLogitsLoss

    def forward(self, x):
        x1 = self.inc(x)        # 256x256
        x2 = self.down1(x1)     # 128x128
        x3 = self.down2(x2)     # 64x64
        x4 = self.down3(x3)     # 32x32
        x5 = self.down4(x4)     # 16x16
        x6 = self.up1(x5, x4)   # 32x32
        x7 = self.up2(x6, x3)   # 64x64
        x8 = self.up3(x7, x2)   # 128x128
        x9 = self.up4(x8, x1)   # 256x256
        
        # Multi-scale outputs
        x6_out = self.outc1(x6)
        x7_out = self.outc2(x7)
        x8_out = self.outc3(x8)
        x9_out = self.outc4(x9)
        
        # Interpolate and combine
        x6_out = F.interpolate(x6_out, scale_factor=8, mode='bilinear', align_corners=True)
        x7_out = F.interpolate(x7_out, scale_factor=4, mode='bilinear', align_corners=True)
        x8_out = F.interpolate(x8_out, scale_factor=2, mode='bilinear', align_corners=True)
        x = x6_out + x7_out + x8_out + x9_out  # Raw logits for BCEWithLogitsLoss
        return x


class ProgressiveUNet(nn.Module):
    """
    Progressive Growing U-Net with dynamic stage switching
    
    This implementation allows progressive training where the network starts
    with a simple architecture and progressively adds layers and increases
    resolution during training.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.current_stage = 1
        self.stage_resolutions = {1: 32, 2: 64, 3: 128, 4: 256}
        
        # Initialize all stages
        self.stage1 = PGUNet1(in_channels, num_classes)
        self.stage2 = PGUNet2(in_channels, num_classes)
        self.stage3 = PGUNet3(in_channels, num_classes)
        self.stage4 = PGUNet4(in_channels, num_classes)
        
        self.stages = {
            1: self.stage1,
            2: self.stage2,
            3: self.stage3,
            4: self.stage4
        }

    def set_stage(self, stage):
        """Set the current progressive stage (1-4)"""
        if stage not in [1, 2, 3, 4]:
            raise ValueError("Stage must be 1, 2, 3, or 4")
        self.current_stage = stage

    def get_current_resolution(self):
        """Get the target resolution for the current stage"""
        return self.stage_resolutions[self.current_stage]

    def transfer_weights(self, prev_stage_dict, current_stage_dict, stage):
        """
        Transfer weights from previous stage to current stage
        This is a simplified version - in practice, you'd need more sophisticated
        weight mapping based on layer correspondence
        """
        # This is a placeholder for weight transfer logic
        # The actual implementation would need careful mapping of layers
        # between stages based on the architectural differences
        return current_stage_dict

    def forward(self, x, target_resolution=None):
        """
        Forward pass with optional resolution specification
        If target_resolution is provided, input will be resized accordingly
        """
        if target_resolution is not None:
            x = F.interpolate(x, size=(target_resolution, target_resolution), 
                            mode='bilinear', align_corners=True)
        else:
            target_resolution = self.get_current_resolution()
            x = F.interpolate(x, size=(target_resolution, target_resolution), 
                            mode='bilinear', align_corners=True)
        
        return self.stages[self.current_stage](x)


# Legacy UNet for compatibility
class UNet(nn.Module):
    """Original U-Net implementation for compatibility"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out