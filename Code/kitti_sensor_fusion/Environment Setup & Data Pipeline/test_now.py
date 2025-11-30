"""
Quick test script - Run this NOW to verify everything works!
No complex setup, just tests the basics.
"""

import sys
from pathlib import Path

# Your KITTI path
KITTI_ROOT = r"D:\KITTI Dataset"

print("\n" + "="*70)
print("QUICK KITTI VERIFICATION TEST")
print("="*70 + "\n")

# Test 1: Check path exists
print(f"Testing path: {KITTI_ROOT}")
kitti_path = Path(KITTI_ROOT)

if not kitti_path.exists():
    print(f"✗ ERROR: Path does not exist: {KITTI_ROOT}")
    print(f"\nPlease verify your KITTI dataset is at: {KITTI_ROOT}")
    print(f"Current contents of {kitti_path.parent}:")
    try:
        for item in kitti_path.parent.iterdir():
            print(f"  - {item.name}")
    except:
        print(f"  Cannot read directory")
    sys.exit(1)

print(f"✓ Path exists: {KITTI_ROOT}\n")

# Test 2: Check required directories
required_dirs = [
    'data_tracking_image_2',
    'data_tracking_velodyne',
    'data_tracking_calib',
    'data_tracking_label_2'
]

print("Checking required directories...")
all_found = True
for dir_name in required_dirs:
    dir_path = kitti_path / dir_name
    if dir_path.exists():
        print(f"  ✓ {dir_name}/")
    else:
        print(f"  ✗ Missing: {dir_name}/")
        all_found = False

if not all_found:
    print(f"\n✗ Some directories are missing!")
    print(f"\nContents of {KITTI_ROOT}:")
    try:
        for item in kitti_path.iterdir():
            print(f"  - {item.name}")
    except:
        pass
    sys.exit(1)

print(f"\n✓ All required directories found!\n")

# Test 3: Check sequence 0000
print("Checking sequence 0000...")
seq_dirs = {
    'Images': kitti_path / 'data_tracking_image_2' / 'training' / 'image_02' / '0000',
    'LiDAR': kitti_path / 'data_tracking_velodyne' / 'training' / 'velodyne' / '0000',
}

# Labels are sequence-level files, not per-frame folders
label_file = kitti_path / 'data_tracking_label_2' / 'training' / 'label_02' / '0000.txt'

for name, path in seq_dirs.items():
    if path.exists():
        count = len(list(path.glob('*')))
        print(f"  ✓ {name}: {count} files")
    else:
        print(f"  ✗ Missing: {name}")
        sys.exit(1)

# Check labels (sequence-level file)
if label_file.exists():
    print(f"  ✓ Labels: 0000.txt")
else:
    print(f"  ✗ Missing: Labels (0000.txt)")
    sys.exit(1)

# Check calib file
calib_file = kitti_path / 'data_tracking_calib' / 'training' / 'calib' / '0000.txt'
if calib_file.exists():
    print(f"  ✓ Calibration: 0000.txt")
else:
    print(f"  ✗ Missing: Calibration file")
    sys.exit(1)

print(f"\n✓ Sequence 0000 ready!\n")

# Test 4: Try to import and load
print("Testing data loader import...")
try:
    from kitti_dataloader import KITTIDataLoader
    print(f"  ✓ Import successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print(f"\nMake sure kitti_dataloader.py is in the same directory as this script!")
    sys.exit(1)

# Test 5: Try to initialize
print("Initializing data loader...")
try:
    loader = KITTIDataLoader(KITTI_ROOT, sequence="0000", split="training")
    num_frames = len(loader)
    print(f"  ✓ Loaded: {num_frames} frames\n")
except Exception as e:
    print(f"  ✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Try to load frame
print("Loading frame 0...")
try:
    frame = loader[0]
    img_shape = frame['image'].shape
    lidar_count = len(frame['lidar'])
    obj_count = len(frame['labels'])
    
    print(f"  ✓ Image: {img_shape}")
    print(f"  ✓ LiDAR points: {lidar_count}")
    print(f"  ✓ Objects: {obj_count}\n")
except Exception as e:
    print(f"  ✗ Failed to load frame: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Try transformations
print("Testing transformations...")
try:
    points_cam = loader.project_lidar_to_camera(frame['lidar'])
    pixels, depth = loader.project_to_image(points_cam)
    
    in_fov = sum((pixels[:, 0] >= 0) & (pixels[:, 0] < 1242) & 
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < 375) & 
                 (depth > 0))
    
    print(f"  ✓ Projection works")
    print(f"  ✓ Points in FOV: {in_fov}/{lidar_count}\n")
except Exception as e:
    print(f"  ✗ Transformation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("="*70)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("="*70)
print("\nYour data loader is working correctly!")
print("You're ready for Day 2: Detection Pipeline\n")
print("Next steps:")
print("  1. Run: python visualize_kitti.py")
print("  2. Review the visualization")
print("  3. Start Day 2: Detection Pipeline\n")

