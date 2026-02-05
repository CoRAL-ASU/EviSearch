# table_definitions/definitions.py
import pandas as pd
from pathlib import Path
from src.config.config import DEFINITIONS_CSV_PATH   # <-- NEW import

def load_definitions(
    csv_path: str | None = None,
    cols_to_test_path: str | None = None
):
    """
    Load column groups from Definitions.csv.
    If csv_path is None → uses the path from config.
    If cols_to_test_path is None → includes **all** groups.
    """
    # -------------------------------------------------- Resolve CSV path
    if csv_path is None:
        csv_path = Path(DEFINITIONS_CSV_PATH)
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Definitions.csv not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")

    # -------------------------------------------------- Filter by Labels_to_Test (optional)
    if cols_to_test_path and Path(cols_to_test_path).exists():
        to_test = pd.read_csv(cols_to_test_path, encoding="utf-8")
        included_labels = to_test[to_test['included'] == 1]['Label'].unique()
    else:
        included_labels = df["Label"].unique()          # <-- ALL groups

    # -------------------------------------------------- Build groups dict
    groups = {}
    for label, gdf in df.groupby("Label"):
        if label in included_labels:
            groups[label] = gdf[["Column Name", "Definition"]].to_dict(orient="records")

    return groups


if __name__ == "__main__":
    groups = load_definitions()          # uses config path
    print(f"Loaded {len(groups)} groups")