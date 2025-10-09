import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    import albumentations as A
except Exception as e:
    raise ImportError("albumentations is required for this script. Install it (e.g. pip install albumentations) and retry.") from e
import xml.etree.ElementTree as ET
import warnings
import inspect

# Use local train directory (this script lives in MoNuSegImprove/train)
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "annots"
OUT_BASE = BASE_DIR / "aug"
OUT_IMG = OUT_BASE / "images"
OUT_MASK = OUT_BASE / "annots"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

# Parameters
PATCH_SIZE = 256
STRIDE = 128  # overlap between patches
AUG_PER_PATCH = 3  # number of augmentations per patch

# Augmentation setup
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    # Some albumentations versions don't accept `alpha_affine` for ElasticTransform.
    # Keep only widely-supported args (alpha, sigma) to avoid warnings.
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),
    A.GridDistortion(p=0.3),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Diagnostics: print albumentations version and inspect created transforms
try:
    print(f"albumentations {A.__version__}")
    print("ElasticTransform signature:", inspect.signature(A.ElasticTransform))
except Exception:
    pass

for t in transform.transforms:
    try:
        print(f"Transform: {type(t).__name__}")
        # print constructor params if available
        if hasattr(t, '__dict__'):
            # show only keys that are not callable and small
            info = {k: v for k, v in t.__dict__.items() if not callable(v) and (isinstance(v, (int, float, str, bool)) or (hasattr(v, '__len__') and len(str(v))<200))}
            print('  attrs:', info)
    except Exception:
        pass

# Helper: extract overlapping patches
def extract_patches(img, mask, patch_size=256, stride=128):
    h, w = img.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = img[y:y+patch_size, x:x+patch_size]
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            patches.append((patch_img, patch_mask))
    return patches


def xml_to_mask(xml_path, image_shape):
    """Rasterize polygon Regions from XML into a single-channel uint8 mask matching image_shape.

    xml_path: Path or str to annotation XML
    image_shape: (h, w, ...) from the corresponding image
    Returns mask with 255 for annotated regions, 0 elsewhere.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse XML {xml_path}: {e}")
        return mask

    # Regions can be nested under Annotation -> Regions -> Region
    for region in root.findall('.//Region'):
        vertices = region.find('Vertices')
        if vertices is None:
            continue
        pts = []
        for v in vertices.findall('Vertex'):
            try:
                x = float(v.get('X'))
                y = float(v.get('Y'))
            except Exception:
                continue
            ix = int(round(x))
            iy = int(round(y))
            # Clip to image bounds
            ix = max(0, min(w-1, ix))
            iy = max(0, min(h-1, iy))
            pts.append([ix, iy])
        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts_arr], color=255)
    return mask


def xml_to_regions(xml_path):
    """Return list of regions, each region is a list of (x,y) floats (image coords)."""
    regions = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse XML {xml_path}: {e}")
        return regions

    for region in root.findall('.//Region'):
        vertices = region.find('Vertices')
        if vertices is None:
            continue
        pts = []
        for v in vertices.findall('Vertex'):
            try:
                x = float(v.get('X'))
                y = float(v.get('Y'))
            except Exception:
                continue
            pts.append((x, y))
        if pts:
            regions.append(pts)
    return regions


def regions_to_xml(regions, out_path):
    """Write a minimal XML with the provided regions (list of list of (x,y))."""
    annotations = ET.Element('Annotations')
    annotation = ET.SubElement(annotations, 'Annotation')
    regions_el = ET.SubElement(annotation, 'Regions')
    for i, region in enumerate(regions, start=1):
        region_el = ET.SubElement(regions_el, 'Region', Id=str(i))
        verts_el = ET.SubElement(region_el, 'Vertices')
        for (x, y) in region:
            v = ET.SubElement(verts_el, 'Vertex')
            # Store with reasonable precision similar to originals
            v.set('X', f"{float(x):.6f}")
            v.set('Y', f"{float(y):.6f}")
    tree = ET.ElementTree(annotations)
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)

# Main loop
img_files = sorted(IMG_DIR.glob("*.tif"))
if not img_files:
    print(f"No .tif images found in {IMG_DIR}. Check your image directory.")

for img_file in tqdm(img_files):
    # prefer a raster mask file (same name .tif/.png) but fallback to .xml annotations
    mask_file = MASK_DIR / img_file.name
    xml_file = MASK_DIR / f"{img_file.stem}.xml"

    image = cv2.imread(str(img_file))
    if image is None:
        print(f"Failed to read image {img_file}. Skipping")
        continue

    if mask_file.exists():
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    elif xml_file.exists():
        # create mask from xml annotation
        mask = xml_to_mask(xml_file, image.shape)
    else:
        print(f"Warning: mask not found for {img_file.name} at {mask_file} and no XML at {xml_file}. Skipping")
        continue

    base_name = img_file.stem
    patches = extract_patches(image, mask, PATCH_SIZE, STRIDE)

    # If xml exists for this image, get regions once
    xml_file = MASK_DIR / f"{base_name}.xml"
    image_regions = xml_to_regions(xml_file) if xml_file.exists() else None

    for j, (patch_img, patch_mask) in enumerate(patches):
        y0 = (j * STRIDE) // ((image.shape[1] - PATCH_SIZE) // STRIDE + 1) if False else None
        # Save original patch (image + mask)
        patch_img_path = OUT_IMG / f"{base_name}_{j:03d}.tif"
        patch_mask_path = OUT_MASK / f"{base_name}_{j:03d}.tif"
        cv2.imwrite(str(patch_img_path), patch_img)
        patch_mask_to_save = patch_mask.astype(np.uint8)
        cv2.imwrite(str(patch_mask_path), patch_mask_to_save)

        # Save original patch XML: crop polygons to patch coordinates
        if image_regions is not None:
            # compute top-left of this patch in source image
            # find patch coordinates by scanning for the patch in the image grid
            # Simpler: compute grid x,y from index j
            h, w = image.shape[:2]
            # compute number of steps in x and y
            nx = (w - PATCH_SIZE) // STRIDE + 1
            ny = (h - PATCH_SIZE) // STRIDE + 1
            py = j // nx
            px = j % nx
            x0 = px * STRIDE
            y0 = py * STRIDE

            cropped_regions = []
            for region in image_regions:
                # shift coordinates to patch-local and include only points inside the patch
                local_pts = []
                for (x, y) in region:
                    lx = x - x0
                    ly = y - y0
                    if 0 <= lx < PATCH_SIZE and 0 <= ly < PATCH_SIZE:
                        local_pts.append((lx, ly))
                if len(local_pts) >= 3:
                    cropped_regions.append(local_pts)

            if cropped_regions:
                regions_to_xml(cropped_regions, OUT_MASK / f"{base_name}_{j:03d}.xml")

        # Augment and save multiple copies (image + mask + xml regions if present)
        for k in range(AUG_PER_PATCH):
            if image_regions is not None and cropped_regions:
                # For augmentation, convert polygon to keypoints by taking vertices
                # albumentations will transform keypoints (we treat polygons as lists of keypoints)
                # We flatten all polygons into a single keypoints list and keep indices to rebuild
                kps = []
                splits = []
                for region in cropped_regions:
                    splits.append(len(region))
                    kps.extend([(float(x), float(y)) for (x, y) in region])

                augmented = transform(image=patch_img, mask=patch_mask, keypoints=kps)
                aug_img, aug_mask = augmented["image"], augmented["mask"]
                aug_kps = augmented.get('keypoints', [])

                # rebuild polygons from aug_kps and splits
                rebuilt = []
                idx = 0
                for s in splits:
                    pts = aug_kps[idx:idx+s]
                    idx += s
                    if len(pts) >= 3:
                        rebuilt.append(pts)

                out_img_path = OUT_IMG / f"{base_name}_{j:03d}_aug{k}.tif"
                out_mask_path = OUT_MASK / f"{base_name}_{j:03d}_aug{k}.tif"
                cv2.imwrite(str(out_img_path), aug_img)
                cv2.imwrite(str(out_mask_path), aug_mask.astype(np.uint8))

                if rebuilt:
                    regions_to_xml(rebuilt, OUT_MASK / f"{base_name}_{j:03d}_aug{k}.xml")
            else:
                # no xml regions; just augment image+mask
                augmented = transform(image=patch_img, mask=patch_mask)
                aug_img, aug_mask = augmented["image"], augmented["mask"]
                out_img_path = OUT_IMG / f"{base_name}_{j:03d}_aug{k}.tif"
                out_mask_path = OUT_MASK / f"{base_name}_{j:03d}_aug{k}.tif"
                cv2.imwrite(str(out_img_path), aug_img)
                cv2.imwrite(str(out_mask_path), aug_mask.astype(np.uint8))
