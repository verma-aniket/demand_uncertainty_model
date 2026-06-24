# import root libraries
import sys
from pathlib import Path
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.utils.climate_functions import compute_spi

def main():
    # route directories
    data_dir = REPO_ROOT / "Data" / "Climate"
    output_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs"

    # read precipitation data from NOAA weather station
    input_file = data_dir / "NOAA_data.csv"
    df = pd.read_csv(input_file)
    
    # Process and build continuous monthly datetime timeline index
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
    df = df.set_index("Date")
    precip = df["Precip"]

    # execute custom spi computation function
    print("Computing 12-month Standardized Precipitation Index (SPI-12) & Gamma metrics...")
    df['SPI12'], spi_params = compute_spi(precip, k=12, return_params=True)

    # Convert parameter definitions dictionary into an indexed summary DataFrame
    params_df = pd.DataFrame(
        [spi_params[month] for month in range(1, 13)], 
        index=range(1, 13), 
        columns=['a', 'scale', 'q']
    )
    params_df.index.name = 'month'

    # save spi parameter values
    params_output_file = output_dir / "spi_params.csv"
    params_df.to_csv(params_output_file, index=True)
    print(f"SPI parameters saved.")

if __name__ == "__main__":
    main()