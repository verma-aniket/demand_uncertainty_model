import os
import zipfile
import urllib.request
from pathlib import Path

# 1. Define paths relative to this script location
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"

# 2. Zenodo configuration (Replace with your actual Zenodo Record ID once published)
ZENODO_RECORD_ID = "YOUR_ZENODO_RECORD_ID" 
DATA_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/simulation_data.zip/content"

def setup_data_pipeline():
    print("Initializing Pipeline Environment Setup...")
    
    # Ensure the base directory exists
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    zip_target_path = BASE_DIR / "simulation_data.zip"
    
    # Skip downloading if data structures are already present
    if DATA_DIR.exists() or RESULTS_DIR.exists():
        print("Pipeline folders (Data/ or Results/) already exist locally. Skipping download to protect local files.")
        return

    # Download the zipped dataset from Zenodo
    print(f"Downloading unified data and results bundle from Zenodo (Record: {ZENODO_RECORD_ID})...")
    try:
        urllib.request.urlretrieve(DATA_URL, zip_target_path)
        print("Download completed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download archive from Zenodo. Details: {e}")
        return

    # Extracting the files safely to the repository root
    print("Extracting archive directly to repository root...")
    try:
        with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
            # Unzips directly into BASE_DIR, generating Data/ and Results/ folders
            zip_ref.extractall(BASE_DIR)
        print("Archive successfully extracted.")
    except Exception as e:
        print(f"ERROR: Extraction failed. Details: {e}")
    finally:
        # Clean up the zip file to save local drive space
        if zip_target_path.exists():
            os.remove(zip_target_path)
            
    print("Pipeline environment successfully deployed and verified!")

if __name__ == "__main__":
    setup_data_pipeline()