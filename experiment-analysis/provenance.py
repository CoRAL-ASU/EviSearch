"""What a curator can actually check: how many shipped values carry a citation, and how good those citations are.

Accuracy is not the only thing a cell has to be. A value with no page and no quotation cannot be audited, and in
this system it also cannot be highlighted on the PDF - `_resolve_candidate_chunk_ids` returns nothing for a cell
whose attribution list is empty, so the viewer has nowhere to point. Three numbers per run:

  coverage       of the values shipped, the share carrying a page and a quotation
  corroboration  of those, the share a second reading independently arrived at (verdict "supported")
  fidelity       of those, the share whose quoted text is actually present in the cited page's parsed text

Fidelity is a lower bound and deliberately so: a value read out of a page image is often missing from the parsed
text, and is counted here as unconfirmed rather than wrong. The gap between coverage and fidelity is therefore
mostly figures and picture-tables, which is worth reporting as such.

Absences are excluded throughout. "Not reported" has nothing to cite, and counting it as uncited would let a run
improve its coverage by answering less.

  python experiment-analysis/provenance.py <run> [<run> ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evisearch.columns import is_no_value  # noqa: E402
from src.evaluation import claude_scoring as cs  # noqa: E402
from src.retrieval import embedding_retriever as retriever  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
PAPERS = 10
N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda v: "".join(ch for ch in N(v).lower() if ch.isalnum())


def absence(value: str) -> bool:
    """An absence in the sense the table means it: no value, or a stated "not reported"/"no"."""
    v = N(value).lower()
    return is_no_value(value) or v in {"no", "none"} or v.startswith(("not reported", "not found", "n/a", "not applicable"))


def page_text(doc: str, page: int, cache: dict) -> str:
    if (doc, page) not in cache:
        try:
            cache[(doc, page)] = SQ(retriever.get_page_content(doc, [page]).get(page, ""))
        except Exception:
            cache[(doc, page)] = ""
    return cache[(doc, page)]


def measure(run: str, docs, cache) -> dict:
    t = {k: 0 for k in ("values", "cited", "page", "quote", "corroborated", "fidelity", "fig_or_table", "flagged")}
    for doc in docs:
        path = ROOT / doc / "runs" / run / "reconciliation_agent" / "reconciled_results.json"
        if not path.exists():
            continue
        for name, entry in (json.loads(path.read_text()).get("columns") or {}).items():
            if entry.get("needs_review"):
                t["flagged"] += 1
            if absence(entry.get("value")):
                continue
            t["values"] += 1
            attr = entry.get("attribution") or []
            source = entry.get("source") or {}
            if not attr:
                continue
            t["cited"] += 1
            if source.get("page"):
                t["page"] += 1
            quote = N(source.get("verbatim_quote"))
            if quote:
                t["quote"] += 1
            if entry.get("verified"):
                t["corroborated"] += 1
            if source.get("modality") in ("figure", "table"):
                t["fig_or_table"] += 1
            # is the quoted text findable in the cited page's parsed text?
            if quote and source.get("page"):
                needle = SQ(quote)[:60]
                if needle and needle in page_text(doc, int(source["page"]), cache):
                    t["fidelity"] += 1
    return t


def main() -> None:
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit(__doc__)
    docs = cs.select_docs("all", cs.load_gold())
    cache: dict = {}
    print(f"{'run':20} {'values':>7} {'cited':>13} {'page+quote':>12} {'corroborated':>14} "
          f"{'quote in text':>14} {'flagged/paper':>14}")
    print("-" * 100)
    for run in runs:
        t = measure(run, docs, cache)
        v, c = max(t["values"], 1), max(t["cited"], 1)
        print(f"{run:20} {t['values']:7} {t['cited']:6} {100*t['cited']/v:5.1f}% "
              f"{t['quote']:6} {100*t['quote']/v:4.1f}% "
              f"{t['corroborated']:6} {100*t['corroborated']/c:5.1f}% "
              f"{t['fidelity']:6} {100*t['fidelity']/c:5.1f}% "
              f"{t['flagged']/PAPERS:13.1f}")
    print("\ncited        = the value carries a non-empty attribution list, so the viewer has a page to open")
    print("corroborated = a second reading of the cited page arrived at this value independently")
    print("quote in text= the quoted text is present in the cited page's PARSED text; a value read off a page")
    print("               image is legitimately absent from it, so this is a lower bound, not an error rate")


if __name__ == "__main__":
    main()
