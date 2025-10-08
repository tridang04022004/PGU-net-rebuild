import os

# Define the directories with raw strings
image_dir = r"D:\DangTri\Uni\NCKH\SPRN\project\project\try test\Mask-RCNN-TF2\kangaroo-transfer-learning\kangaroo\images"
annot_dir = r"D:\DangTri\Uni\NCKH\SPRN\project\project\try test\Mask-RCNN-TF2\kangaroo-transfer-learning\kangaroo\annots"

# Ensure both directories exist
if not (os.path.exists(image_dir) and os.path.exists(annot_dir)):
    print("One or both directories do not exist!")
    exit()

# Get list of image files (assuming .tif extension)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
annot_files = sorted([f for f in os.listdir(annot_dir) if f.endswith('.xml')])

# Check if the number of image and annotation files match
if len(image_files) != len(annot_files):
    print("Mismatch in number of image and annotation files!")
    exit()

# Rename files
for i, (img_file, annot_file) in enumerate(zip(image_files, annot_files), start=1):
    # Extract base names without extensions
    img_base = os.path.splitext(img_file)[0]
    annot_base = os.path.splitext(annot_file)[0]
    
    # Check if the base names match (to ensure correspondence)
    if img_base != annot_base:
        print(f"Mismatch found: {img_file} does not correspond to {annot_file}")
        exit()
    
    # Generate new file names (e.g., 0001.tif and 0001.xml)
    new_name = f"{i:04d}"  # Format number as 4-digit string (e.g., 0001)
    new_img_file = f"{new_name}.tif"
    new_annot_file = f"{new_name}.xml"
    
    # Rename the files
    os.rename(
        os.path.join(image_dir, img_file),
        os.path.join(image_dir, new_img_file)
    )
    os.rename(
        os.path.join(annot_dir, annot_file),
        os.path.join(annot_dir, new_annot_file)
    )
    
    print(f"Renamed {img_file} to {new_img_file} and {annot_file} to {new_annot_file}")

print("Renaming completed successfully!")