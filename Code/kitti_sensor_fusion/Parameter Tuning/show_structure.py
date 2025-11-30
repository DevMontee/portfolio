"""
Directory Structure Viewer

Simple script to display all folders, subfolders, and files in a directory.

Usage:
  python show_structure.py
  python show_structure.py /path/to/directory
  python show_structure.py D:\kitti_sensor_fusion
"""

import os
from pathlib import Path
import sys


def show_tree(directory, prefix="", max_depth=None, current_depth=0):
    """
    Recursively display directory tree structure.
    
    Args:
        directory: Path to directory
        prefix: Prefix for tree lines
        max_depth: Maximum depth to traverse (None = unlimited)
        current_depth: Current depth level
    """
    if max_depth and current_depth >= max_depth:
        return
    
    try:
        items = sorted(Path(directory).iterdir())
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    
    # Separate directories and files
    dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
    files = [item for item in items if item.is_file() and not item.name.startswith('.')]
    
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last = i == len(all_items) - 1
        
        # Determine tree characters
        if is_last:
            current = "└── "
            extension = "    "
        else:
            current = "├── "
            extension = "│   "
        
        # Print item
        if item.is_dir():
            print(f"{prefix}{current}{item.name}/")
            # Recurse into subdirectories
            show_tree(item, prefix + extension, max_depth, current_depth + 1)
        else:
            print(f"{prefix}{current}{item.name}")


def show_flat_list(directory, indent=0):
    """
    Display directory structure as flat list.
    
    Args:
        directory: Path to directory
        indent: Indentation level
    """
    try:
        items = sorted(Path(directory).iterdir())
    except PermissionError:
        return
    
    # Separate directories and files
    dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
    files = [item for item in items if item.is_file() and not item.name.startswith('.')]
    
    # Print directories first
    for dir_item in dirs:
        print("  " * indent + f"📁 {dir_item.name}/")
        show_flat_list(dir_item, indent + 1)
    
    # Print files
    for file_item in files:
        print("  " * indent + f"📄 {file_item.name}")


def count_items(directory):
    """Count directories and files recursively."""
    total_dirs = 0
    total_files = 0
    
    try:
        items = Path(directory).rglob('*')
        for item in items:
            if not item.name.startswith('.'):
                if item.is_dir():
                    total_dirs += 1
                else:
                    total_files += 1
    except PermissionError:
        pass
    
    return total_dirs, total_files


def main():
    """Main entry point."""
    # Get directory from argument or use current directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "."
    
    # Convert to Path object
    target_path = Path(target_dir)
    
    # Check if directory exists
    if not target_path.exists():
        print(f"Error: Directory '{target_dir}' does not exist")
        return
    
    if not target_path.is_dir():
        print(f"Error: '{target_dir}' is not a directory")
        return
    
    # Get absolute path
    abs_path = target_path.resolve()
    
    print("\n" + "="*70)
    print("DIRECTORY STRUCTURE")
    print("="*70)
    print(f"\nDirectory: {abs_path}\n")
    
    # Count items
    num_dirs, num_files = count_items(abs_path)
    print(f"Subdirectories: {num_dirs}")
    print(f"Files: {num_files}")
    print("\n" + "-"*70 + "\n")
    
    # Show tree structure
    print(f"{abs_path.name}/")
    show_tree(abs_path)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
