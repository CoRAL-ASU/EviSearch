"""Build and audit evidence locations: where each admitted value is boxed on its source pages.

  python experiment-analysis/evidence_locations.py build <run> [<run> ...]     write evidence_locations.json per paper
  python experiment-analysis/evidence_locations.py audit <run> [<run> ...]     how well the boxes cover the values
  python experiment-analysis/evidence_locations.py sheet <run> <picks.json> <out.png>   draw boxes for inspection

Locating reads only the PDF and the stored outputs, so it runs on any finished run without a new extraction.

audit reports, per run: how each value was located (the value as printed, its printed parts, the numbers it was
computed from, the supporting quotation, a table or figure box, or the cited page alone), the share of cells with a
highlight, boxes per cell, and - for numeric values - whether every number of the value falls inside a box. Text
values and identifiers are left out of that last figure, since they are matched by their words.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evisearch.services import evidence_locator as L  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")


def docs_of(run):
    return sorted(p.name for p in ROOT.iterdir() if (L.stage_dir(ROOT, p.name, run) / "reconciled_results.json").exists())


def build(runs):
    for run in runs:
        n = sum(1 for d in docs_of(run) if L.build_run(d, run, ROOT))
        print(f"{run}: evidence locations written for {n} papers")


def audit(runs):
    for run in runs:
        st, boxes, inside, n = collections.Counter(), [], collections.Counter(), 0
        for d in docs_of(run):
            for c in L.load_locations(d, run, ROOT).values():
                n += 1
                st[c["status"]] += 1
                if c["regions"]:
                    boxes.append(sum(len(r["rects"]) for r in c["regions"]))
                if L.text_dominant(c["value"]) or L.identifiers(c["value"]):
                    continue
                vn = [x for x in L.numbers(c["value"]) if not L._weak(x)] or L.numbers(c["value"])
                if vn:
                    got = set(L.numbers(" ".join(r["text"] for r in c["regions"] if r["kind"] in ("value", "value_parts"))))
                    inside["all" if all(x in got for x in vn) else ("some" if any(x in got for x in vn) else "none")] += 1
        if not n:
            print(f"{run}: no locations (run `build` first)")
            continue
        t = max(1, sum(inside.values()))
        print(f"\n{run}: {n} admitted values")
        for k, v in st.most_common():
            print(f"  {k:12} {100 * v / n:5.1f}%")
        print(f"  highlighted {100 * (n - st['page'] - st['none']) / n:.1f}%   boxes per highlighted cell: "
              f"median {statistics.median(boxes)}, mean {statistics.mean(boxes):.1f}")
        print(f"  numeric values: every number boxed {100 * inside['all'] / t:.1f}%, some {100 * inside['some'] / t:.1f}%, "
              f"none {100 * inside['none'] / t:.1f}%")


def sheet(run, picks_json, out):
    import fitz
    from PIL import Image, ImageDraw
    from src.evisearch.services.highlight import resolve_pdf_path

    tiles = []
    for tag, d, col in json.loads(Path(picks_json).read_text()):
        c = L.load_locations(d, run, ROOT).get(col)
        if not c:
            continue
        page = c["regions"][0]["page"] if c["regions"] else (c["pages_checked"] or [1])[0]
        with fitz.open(resolve_pdf_path(d)) as doc:
            pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img)
        rects = L.regions_on(c, page, 2.0)
        for x0, y0, x1, y1 in rects:
            draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(220, 30, 30), width=3)
        cy = (min(r[1] for r in rects) + max(r[3] for r in rects)) / 2 if rects else img.height / 3
        top = max(0, int(cy - 100))
        crop = img.crop((0, top, img.width, min(img.height, top + 200)))
        crop = crop.resize((900, int(900 * crop.height / crop.width)))
        head = Image.new("RGB", (900, 34), "white")
        hd = ImageDraw.Draw(head)
        hd.text((6, 3), f"[{tag} -> {c['status']}] {d.split('_')[1]} p{page} | {col[:58]}", fill="black")
        hd.text((6, 18), f"value: {c['value'][:60]}   boxed: {' / '.join(r['text'][:26] for r in c['regions'][:3])}", fill=(160, 0, 0))
        tile = Image.new("RGB", (900, head.height + crop.height), "white")
        tile.paste(head, (0, 0))
        tile.paste(crop, (0, head.height))
        tiles.append(tile)
    out_img = Image.new("RGB", (900, sum(t.height + 4 for t in tiles)), (90, 90, 90))
    y = 0
    for t in tiles:
        out_img.paste(t, (0, y))
        y += t.height + 4
    out_img.save(out)
    print(out, len(tiles), "tiles")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"build": lambda: build(rest), "audit": lambda: audit(rest), "sheet": lambda: sheet(*rest)}[cmd]()
