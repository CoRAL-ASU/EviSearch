"""Appendix A: the reference audit. Reference values found, on reading their source page, to contradict it.

  python experiment-analysis/reference_audit.py

Every reported accuracy is scored against the reference annotations as released; these cells are listed, not applied.
The share is of all reference cells (papers x 133 columns).
"""
import sys

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

# (document prefix, column, what the cited page states)
CORRECTIONS = [
    ("NCT01809691_Aggarwal", "COE_RCT_IND_OVERALL_RJ", "Yes (randomised phase 3)"),
    ("NCT02446405_Sweeney", "Median Age (years) | Control", "69 (64-75)"),
    ("NCT02446405_Sweeney", "Median Age (years) | Treatment", "69 (63-74)"),
    ("NCT02799602_Hussain", "COE_RCT_IND_OVERALL_RJ", "Yes (randomised, double-blind, placebo-controlled phase 3)"),
    ("NCT02799602_Smith", "Add-on Treatment", "Darolutamide"),
    ("NCT02799602_Smith", "Region - N (%) | South America | Control", "Included in Rest of the world"),
    ("NCT02799602_Smith", "Region - N (%) | South America | Treatment", "Included in Rest of the world"),
    ("NCT01957436_Fizazi", "PS - N (%) | 0 | Control", "412 (70%)"),
    ("NCT01957436_Fizazi", "PS - N (%) | 0 | Treatment", "412 (71%)"),
    ("NCT01957436_Fizazi", "PS - N (%) | 1-2 | Control", "177 (30%)"),
    ("NCT01957436_Fizazi", "PS - N (%) | 1-2 | Treatment", "171 (29%)"),
]


if __name__ == "__main__":
    gold = cs.load_gold()
    docs = cs.select_docs("all", gold)
    cells = len(docs) * len(cs.load_columns())
    for doc, col, page_says in CORRECTIONS:
        print(f"  {doc:22} {col[:52]:52} page: {page_says}")
    print(f"{len(CORRECTIONS)} of {cells} reference cells contradict their cited page: {100 * len(CORRECTIONS) / cells:.1f}%")
