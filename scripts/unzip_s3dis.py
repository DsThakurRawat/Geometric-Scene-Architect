#!/usr/bin/env python3
import os
import time
import zipfile
import subprocess

ZIP_PATH = "data/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip"
EXTRACT_DIR = "data/s3dis/"

def is_wget_running():
    try:
        # Check if there is any wget downloading Stanford3dDataset
        output = subprocess.check_output(["pgrep", "-f", "wget.*Stanford3dDataset"])
        return len(output) > 0
    except subprocess.CalledProcessError:
        return False

def main():
    print("Monitoring S3DIS dataset download...")
    
    # Wait for the download to start and for wget to finish
    while True:
        if not os.path.exists(ZIP_PATH):
            print("Zip file not found yet. Waiting...")
            time.sleep(30)
            continue
            
        if is_wget_running():
            size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
            print(f"Download in progress. Current size: {size_mb:.2f} MB. Waiting...")
            time.sleep(30)
            continue
            
        # If wget is not running, wait a bit and check if file size is stable
        size_before = os.path.getsize(ZIP_PATH)
        time.sleep(10)
        size_after = os.path.getsize(ZIP_PATH)
        
        if size_before != size_after:
            print("File size is still changing. Waiting...")
            continue
            
        # Wget finished and size is stable. Let's try to unzip.
        print(f"Download finished. File size: {size_after / (1024 * 1024 * 1024):.2f} GB.")
        print("Extracting S3DIS dataset (this may take a few minutes)...")
        
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                # Get total number of files for progress logging
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                print(f"Total files in zip: {total_files:,}")
                
                # Extract all
                zip_ref.extractall(EXTRACT_DIR)
                
            print("Extraction completed successfully!")
            
            # Clean up the zip file to save space
            print("Cleaning up ZIP file...")
            os.remove(ZIP_PATH)
            print("ZIP file deleted. S3DIS dataset is ready under data/s3dis/!")
            break
            
        except zipfile.BadZipFile:
            print("Warning: Bad zip file (might be incomplete or corrupted). Waiting for download to resume or restart...")
            time.sleep(30)
        except Exception as e:
            print(f"Error during extraction: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
