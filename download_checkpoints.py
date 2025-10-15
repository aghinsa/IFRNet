#!/usr/bin/env python3
"""
download_checkpoints.py
=======================
IFRNet checkpoint auto-download script

Downloads pre-trained IFRNet checkpoints from Dropbox and organizes them
in the checkpoints/ directory at the repo root.

Usage:
    python download_checkpoints.py
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    print(f"[INFO] Downloading to {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract ZIP file with progress."""
    print(f"[INFO] Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files for progress
        file_list = zip_ref.namelist()
        
        with tqdm(desc="Extracting", total=len(file_list)) as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)


def main():
    # Get repo root directory
    repo_root = Path(__file__).parent
    checkpoints_dir = repo_root / "checkpoints"
    
    print("=" * 60)
    print("🚀 IFRNet Checkpoint Download Script")
    print("=" * 60)
    
    # Dropbox download link (force ZIP mode with ?dl=1)
    dropbox_url = "https://www.dropbox.com/scl/fo/gvfjc8bq259l4cre2ai0k/AIxkWTcEOcvIIYe7RDlZpag?rlkey=x4lxph520gbt0tjy839gmwoc0&e=1&dl=1"
    
    # Temporary ZIP file path
    zip_path = checkpoints_dir / "ifrnet_checkpoints.zip"
    
    try:
        # Step 1: Create checkpoints directory
        print(f"[INFO] Creating checkpoints directory: {checkpoints_dir}")
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Step 2: Download the ZIP file
        print(f"[INFO] Downloading checkpoints from Dropbox...")
        download_file(dropbox_url, zip_path)
        
        # Step 3: Extract the ZIP file
        print(f"[INFO] Extracting checkpoints...")
        extract_zip(zip_path, checkpoints_dir)
        
        # Step 4: Clean up ZIP file
        print(f"[INFO] Cleaning up temporary ZIP file...")
        zip_path.unlink()
        
        # Step 5: Verify and list downloaded files
        print("\n✅ Download complete!")
        print(f"📁 Checkpoints location: {checkpoints_dir}")
        print("\n📋 Downloaded files:")
        
        # Recursively list all files in checkpoints directory
        for root, dirs, files in os.walk(checkpoints_dir):
            root_path = Path(root)
            level = len(root_path.relative_to(checkpoints_dir).parts)
            indent = "  " * level
            print(f"{indent}📂 {root_path.name}/")
            
            sub_indent = "  " * (level + 1)
            for file in files:
                file_path = root_path / file
                file_size = file_path.stat().st_size
                size_mb = file_size / (1024 * 1024)
                print(f"{sub_indent}📄 {file} ({size_mb:.1f} MB)")
        
        print(f"\n🎯 Usage example:")
        print(f"python interpolate_video.py --input video.mp4 --target_fps 24 --model ./checkpoints/IFRNet/IFRNet_Vimeo90K.pth")
        
    except requests.RequestException as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)
    except zipfile.BadZipFile as e:
        print(f"❌ ZIP extraction failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()