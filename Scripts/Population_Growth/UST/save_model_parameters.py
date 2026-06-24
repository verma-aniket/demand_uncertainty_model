# import base libraries
import sys
from pathlib import Path
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.models import build_UST_model

def main():
    # route directories
    data_file = REPO_ROOT / "data" / "Housing" / "Clean_Data" / "UST_Data.csv"
    output_dir = REPO_ROOT / "data" / "Housing" / "Params"

    # read housing and population data
    data = pd.read_csv(data_file)

    # Define default data columns to pass
    def_cols = ['Year', 'City', 'Population']

    # 5. EXECUTE REGRESSION RUNNERS
    print("Balancing panel data for Single-Family households...")
    sf_data = build_UST_model.balance_data(
        data[def_cols + ['Single']].copy(), 
        feature_col='Single'
    )

    print("Executing Fama-MacBeth Regression for Single-Family models...")
    build_UST_model.fama_macbeth(
        df=sf_data, 
        feature_col='Single', 
        output_dir=output_dir
    )

    print(f"Regression complete.")

if __name__ == "__main__":
    main()