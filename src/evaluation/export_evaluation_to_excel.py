"""
Export evaluation_results.json to an Excel workbook for easy comparison.
Creates: evaluation_results.xlsx in the same directory as the JSON file.

Usage:
  python -m src.evaluation.export_evaluation_to_excel [path_to_evaluation_results.json]
  If no path given, searches for evaluation_results.json in current dir or common result dirs.
"""
import json
import sys
from pathlib import Path

import pandas as pd


def load_evaluation_results(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def verdict(overall: float) -> str:
    if overall >= 1.0:
        return "Correct"
    if overall > 0.0:
        return "Partial"
    return "Wrong"


def build_rows(data: dict) -> list:
    doc_name = data.get("document_name", "")
    cols = data.get("columns", {})
    rows = []
    for col_name, col_data in cols.items():
        c = col_data.get("correctness")
        comp = col_data.get("completeness")
        ov = col_data.get("overall")
        rows.append({
            "Column Name": col_name,
            "Category": col_data.get("category", ""),
            "Ground Truth": col_data.get("ground_truth", ""),
            "Predicted": col_data.get("predicted", ""),
            "Correctness": c,
            "Completeness": comp,
            "Overall": ov,
            "Verdict": verdict(ov) if ov is not None else "",
            "Reason": col_data.get("reason", ""),
        })
    return rows, doc_name


def write_excel(json_path: Path, out_path: Path | None = None) -> Path:
    data = load_evaluation_results(json_path)
    rows, doc_name = build_rows(data)
    df = pd.DataFrame(rows)

    if out_path is None:
        out_path = json_path.parent / "evaluation_results.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="By column", index=False)
        # Summary sheet
        summary_rows = [
            {"Metric": "Document", "Value": doc_name},
            {"Metric": "Total columns", "Value": len(df)},
            {"Metric": "Correct", "Value": (df["Verdict"] == "Correct").sum()},
            {"Metric": "Partial", "Value": (df["Verdict"] == "Partial").sum()},
            {"Metric": "Wrong", "Value": (df["Verdict"] == "Wrong").sum()},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        # By category: total, correct, partial, wrong per category
        by_cat = df.groupby("Category").agg(
            Total=("Column Name", "count"),
            Correct=("Verdict", lambda s: (s == "Correct").sum()),
            Partial=("Verdict", lambda s: (s == "Partial").sum()),
            Wrong=("Verdict", lambda s: (s == "Wrong").sum()),
        ).reset_index()
        by_cat.to_excel(writer, sheet_name="By category", index=False)

    return out_path


def main():
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Default: look for evaluation_results.json in cwd or in a results folder
        cwd = Path.cwd()
        candidates = [
            cwd / "evaluation_results.json",
            cwd / "evaluation" / "evaluation_results.json",
        ]
        json_path = None
        for p in candidates:
            if p.exists():
                json_path = p
                break
        if json_path is None:
            print("Usage: python -m src.evaluation.export_evaluation_to_excel <path_to_evaluation_results.json>")
            sys.exit(1)

    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    out_path = write_excel(json_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
