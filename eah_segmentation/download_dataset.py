import os
import requests
import zipfile
from pathlib import Path
import argparse
import shutil

"""
eah_segmentation/download_dataset.py

This module handles the downloading and organization of the ADE20K dataset.
It provides functionality to download the dataset from official sources,
extract it, and organize it into a consistent directory structure.

Key Features:
- Downloads dataset with progress tracking
- Handles both images and annotations
- Verifies downloads and extracts
- Organizes files into a standardized structure
- Provides detailed progress and error reporting
"""

def download_file(url, output_path):
    """
    Downloads a file from a URL with progress tracking.
    
    Key Features:
        - Shows download progress bar
        - Handles network errors gracefully
        - Verifies downloaded file integrity
        
    Args:
        url (str): URL to download from
        output_path (Path): Path to save the downloaded file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"📥 Starting download from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            print(f"⚠️ Warning: Could not determine file size for {url}")
            total_size = 1  # Prevent division by zero
        
        block_size = 1024
        downloaded = 0
        
        print(f"📥 Downloading to {output_path}...")
        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
        print("\n✅ Download completed")
        
        # Verify the downloaded file
        if output_path.stat().st_size == 0:
            print(f"❌ Error: Downloaded file is empty")
            output_path.unlink()
            return False
            
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network error downloading {url}: {str(e)}")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        print(f"\n❌ Error downloading {url}: {str(e)}")
        if output_path.exists():
            output_path.unlink()
        return False

def verify_dataset(data_dir):
    """
    Verifies that the ADE20K dataset is properly organized.
    
    Checks for two possible structures:
    1. Organized: validation/images and validation/annotations
    2. Unorganized: ADEChallengeData2016/images/validation and annotations/validation
    
    Args:
        data_dir (Path): Directory containing the dataset
        
    Returns:
        bool: True if dataset structure is valid
    """
    data_dir = Path(data_dir)
    
    # Check for organized structure (validation/images and validation/annotations)
    val_dir = data_dir / "validation"
    if val_dir.exists():
        val_images = list((val_dir / "images").glob("*.jpg"))
        val_annotations = list((val_dir / "annotations").glob("*.png"))
        if val_images and val_annotations:
            print("✅ Found organized dataset structure")
            return True
    
    # Check for unorganized structure (ADEChallengeData2016/images/validation and ADEChallengeData2016/annotations/validation)
    ade_dir = data_dir / "ADEChallengeData2016"
    if ade_dir.exists():
        val_images = list((ade_dir / "images" / "validation").glob("*.jpg"))
        val_annotations = list((ade_dir / "annotations" / "validation").glob("*.png"))
        if val_images and val_annotations:
            print("✅ Found unorganized dataset structure")
            return True
    
    return False

def download_ade20k(output_dir):
    """
    Downloads and organizes the complete ADE20K dataset.
    
    Key Operations:
        1. Downloads dataset files from MIT CSAIL servers
        2. Extracts downloaded zip files
        3. Organizes files into standardized structure
        4. Verifies dataset integrity
        5. Provides detailed progress information
    
    Args:
        output_dir (Path): Directory to save the dataset
        
    Returns:
        bool: True if download and organization successful
    """
    print("🚀 Starting ADE20K dataset download...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir = output_dir.resolve()  # Convert to absolute path
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dataset will be saved to: {output_dir}")
    
    # Check if dataset is already downloaded and organized
    if verify_dataset(output_dir):
        print("✅ Dataset already exists and is properly organized")
        return True
    
    try:
        # ADE20K dataset URLs
        urls = {
            'images': 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
            'annotations': 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016_Annotations.zip'
        }
        
        # Download and extract each file
        for name, url in urls.items():
            zip_path = output_dir / f"{name}.zip"
            
            # Check if zip exists and is valid
            should_download = True
            if zip_path.exists():
                size = zip_path.stat().st_size
                print(f"📦 Found {name}.zip ({size} bytes)")
                if size < 1000:  # If file is too small, it's probably corrupted
                    print(f"⚠️ {name}.zip seems corrupted (too small), will re-download")
                    zip_path.unlink()
                else:
                    should_download = False
                    print(f"📦 {name}.zip exists and seems valid, skipping download")
            
            if should_download:
                print(f"📥 Downloading {name}...")
                if not download_file(url, zip_path):
                    print(f"❌ Failed to download {name}, skipping...")
                    continue
                
                # Verify the downloaded file
                if not zip_path.exists() or zip_path.stat().st_size < 1000:
                    print(f"❌ Downloaded {name}.zip is invalid, skipping...")
                    continue
            
            # Extract if not already extracted
            ade_dir = output_dir / "ADEChallengeData2016"
            if not ade_dir.exists() or not (ade_dir / "annotations").exists():
                print(f"📦 Extracting {name}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        print(f"📦 Extracting to {output_dir}")
                        zip_ref.extractall(output_dir)
                    # Remove zip file after successful extraction
                    zip_path.unlink()
                    print(f"✅ Extraction completed, removed {zip_path}")
                except Exception as e:
                    print(f"❌ Error extracting {name}.zip: {str(e)}")
                    print(f"Current directory: {Path.cwd()}")
                    print(f"Target directory: {output_dir}")
                    print(f"Zip file: {zip_path}")
                    # If extraction fails, remove the corrupted zip
                    if zip_path.exists():
                        zip_path.unlink()
                        print(f"🗑️ Removed corrupted {zip_path}")
                    continue
            else:
                print(f"📦 {name} already extracted, skipping")
        
        # Organize files if not already organized
        if not verify_dataset(output_dir):
            ade_dir = output_dir / "ADEChallengeData2016"
            if ade_dir.exists():
                print(f"📦 Organizing files from {ade_dir}")
                # Move validation files to a separate directory
                val_dir = output_dir / "validation"
                val_dir.mkdir(exist_ok=True)
                
                # Move validation images
                val_images_dir = val_dir / "images"
                val_images_dir.mkdir(exist_ok=True)
                for img in (ade_dir / "images" / "validation").glob("*.jpg"):
                    shutil.move(str(img), str(val_images_dir / img.name))
                
                # Move validation annotations
                val_annotations_dir = val_dir / "annotations"
                val_annotations_dir.mkdir(exist_ok=True)
                for ann in (ade_dir / "annotations" / "validation").glob("*.png"):
                    shutil.move(str(ann), str(val_annotations_dir / ann.name))
                
                # Remove original directory
                shutil.rmtree(ade_dir)
                print("✅ Organization completed")
        
        # Final verification
        if verify_dataset(output_dir):
            print("\n📊 Dataset information:")
            print(f"  - Name: ADE20K")
            # Check both organized and unorganized structures for file counts
            val_dir = output_dir / "validation"
            ade_dir = output_dir / "ADEChallengeData2016"
            
            if val_dir.exists():
                val_images = list((val_dir / "images").glob("*.jpg"))
                val_annotations = list((val_dir / "annotations").glob("*.png"))
            else:
                val_images = list((ade_dir / "images" / "validation").glob("*.jpg"))
                val_annotations = list((ade_dir / "annotations" / "validation").glob("*.png"))
                
            print(f"  - Number of validation images: {len(val_images)}")
            print(f"  - Number of validation annotations: {len(val_annotations)}")
            return True
        else:
            print("❌ Dataset verification failed")
            print(f"Current directory: {Path.cwd()}")
            print(f"Target directory: {output_dir}")
            return False
        
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Target directory: {output_dir}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download ADE20K dataset')
    parser.add_argument('--output_dir', type=str, default='datasets',
                      help='Directory to save the dataset (default: datasets)')
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
        print(f"📁 Using absolute path: {output_dir}")
    
    # Download dataset
    success = download_ade20k(output_dir)
    
    if success:
        print("\n🎉 Dataset download completed successfully!")
        print("\nYou can now run the test script:")
        print("python -m eah_segmentation.evaluate --models deeplabv3plus_edgetpu segformer_b0")
    else:
        print("\n❌ Dataset download failed. Please check the error message above.")

if __name__ == '__main__':
    main() 