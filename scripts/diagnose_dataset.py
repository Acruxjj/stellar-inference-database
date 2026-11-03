from pathlib import Path
import pandas as pd
from src.data_diagnostic import general_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "all_stars.parquet"


def main():
    df = pd.read_parquet(DATA_PATH)
    report = general_diagnostics(df)

    structure = report["structure"]

    print("Dataset Overview")
    print(f"File: {DATA_PATH.name}")
    print(f"Shape: {structure['shape']}")
    print(f"Memory: {structure['memory_mb']:.2f} MB")

    print("\nColumn Types:")
    print(pd.Series(structure["column_types"]))

    print("\nFirst five rows:")
    print(report["preview"].to_string(index=False))

    print("\nMissing values:")
    print(pd.Series(report["missing_values"]))

    print("\nDuplicate rows")
    print(report["duplicate_rows"])

    print("\nNumeric Summary")
    print(report["numeric_summary"].to_string())

    print("\nInfinite Values")
    print(pd.Series(report["non_finite_values"]))


if __name__ == "__main__":
    main()