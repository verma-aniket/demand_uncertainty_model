# scripts/UST/save_housing_data.py
import os
import sys
from pathlib import Path
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add root to python path so it can find the 'src' directory
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.utils import read_housing_data

def main():
    # route directories
    data_dir = REPO_ROOT / "data" / "Housing" / "Raw_Data"
    output_dir = REPO_ROOT / "data" / "Housing" / "Clean_Data"

    # Get all housing data files in the directory
    file_list = os.listdir(data_dir)

    # Execute pipeline
    print("Building Housing Data Dataset...")
    housing_data = read_housing_data.get_housing_data(data_dir, file_list)

    # Save data
    output_file = output_dir / "UST_Data.csv"
    housing_data.to_csv(output_file, index=False)
    
    print(f"Housing data saved.")

if __name__ == "__main__":
    main()