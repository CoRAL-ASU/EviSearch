"""Where a value sits on its source pages: the boxes a reviewer is shown.

For every admitted value the locator returns *regions* - a page, rectangles in PDF points, what the region is, the
text it covers, and the parser chunks it falls in. Regions are chosen in this order:

  value        the value as printed. Every number in it (or, for a text value, most of its words) found together on a
               cited page. When the same numbers occur more than once, the occurrence nearest the supporting
               quotation, and then nearest the column's own words, is chosen.
  value_parts  some of the value's numbers are printed and the rest are not (a count with a computed percentage):
               the printed ones.
  operand      the value is derived - a sum, a difference, a percentage computed from counts - so it is not printed
               at all. The numbers it was computed from, taken from its reasoning and quotations, each found on the
               cited pages.
  quote        the supporting quotation, matched to the exact text of the page: the whole quotation, then its
               sentences, then the narrowest window of words that holds most of it. Prefixes of the quotation are
               never used - they highlight the opening of a sentence rather than the evidence.
  chunk        a value read from a figure or a table image, absent from the page text: the parser's box of that type
               on the cited page that best matches the column.

Only pages the extraction cited are searched, so a highlight is never placed on a page nobody pointed to. Word
rectangles are merged into one box per line. The quotation actually found on the page is kept as `quote_on_page`,
which is the verbatim span the attribution should show.

`build_run` writes the result for a run next to its reconciled output, as evidence_locations.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOCATIONS_FILE = "evidence_locations.json"
VERSION = 1
QUOTE_WINDOW = 24          # words: the narrowest window that may stand in for a quotation not found verbatim
VALUE_WINDOW = 14          # words: how far apart the numbers of one value may be printed
ANCHOR_RADIUS = 60         # words: how near a column word must be to count as anchoring an occurrence
MAX_OPERANDS = 6
STOP = set("""the and for with from that this were was are not per any all into than then also only been have
has had its their which when where while each both after before between within without about above below over under
more less most least other such these those they them there here what who whom whose treatment control overall arm
arms group groups patients patient number value values reported column""".split())

# ---- numbers --------------------------------------------------------------------------------------------------------

_DECIMAL = str.maketrans({"·": ".", "∙": ".", "⋅": ".", "−": "-", "–": "-", "—": "-"})
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_REF = re.compile(r"(?:page|pages|p\.|pp\.|table|tables|fig\.?|figure|figures|ref\.?|appendix|cycle|cycles|grade|grades|"
                  r"version|v)\s*$", re.I)


def canon(number: str) -> str:
    """One spelling per number: thousands separators dropped, trailing decimal zeros dropped (68.0 and 68 agree)."""
    s = number.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    s = s.lstrip("0") or "0"
    return "0" + s if s.startswith(".") else s


_STANDALONE = re.compile(r"^[(\[{<>=~\u2264\u2265\u00b1+\-]*(?:[nN]\s*=\s*)?\d[\d.,]*(?:[-]\d[\d.,]*)?[%)\]}.,;:*\u2020\u2021]*$")
_IDENT = re.compile(r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{4,}\b")


def word_numbers(word: str) -> List[str]:
    """The numbers a single printed word stands for. Only a word that is a number, perhaps with brackets, a sign, a
    percent or "n =", counts: a digit glued to letters is an affiliation mark, a superscript, or part of a name
    (MD3, E3805), and a group of zeros is the tail of a thousands-separated number (100 000), not the number 0."""
    out = []
    for part in str(word or "").translate(_DECIMAL).split("/"):  # events/N, "years/47%": each side is its own number
        if not _STANDALONE.match(part):
            continue
        for m in _NUM.finditer(part):
            raw = m.group(0)
            if len(raw) > 1 and set(raw.replace(",", "")) == {"0"}:
                continue
            out.append(canon(raw))
    return out


def identifiers(value: str) -> List[str]:
    """Alphanumeric tokens in a value (trial numbers such as E3805, NCT registrations), matched as whole words."""
    return [m.group(0).lower() for m in _IDENT.finditer(str(value or ""))]


def numbers(text: str, skip_references: bool = False) -> List[str]:
    """The numbers in a text, in order, canonical. With skip_references, numbers naming a page, table, figure or
    grade are left out, since they are not values anything was computed from."""
    text = str(text or "").translate(_DECIMAL)
    out = []
    for m in _NUM.finditer(text):
        if skip_references and _REF.search(text[max(0, m.start() - 12):m.start()]):
            continue
        out.append(canon(m.group(0)))
    return out


def _weak(number: str) -> bool:
    """A bare small integer (0-9) matches far too many places to locate a value by itself."""
    return "." not in number and len(number) == 1


_ARITH = re.compile(r"(\d[\d.,]*)\s*(?:[+\-*/x\u00d7\u00f7=]|minus|plus|out of|of|divided by|times)\s*(\d[\d.,]*)", re.I)


def arithmetic_operands(text: str) -> List[str]:
    """The numbers that take part in arithmetic written out in a text - "4/192 = 2.08%", "117 of 397", "289 + 108" -
    which is how a derived value's reasoning names what it was computed from. A number merely mentioned (a dose, a
    schedule, a page) is not an operand."""
    text = str(text or "").translate(_DECIMAL)
    out: List[str] = []
    for m in _ARITH.finditer(text):
        for g in (m.group(1), m.group(2)):
            n = canon(g.rstrip(".,"))
            if n not in out:
                out.append(n)
    return out


def text_dominant(value: str) -> bool:
    """A value that is mostly words with an incidental number (a list of secondary endpoints that mentions a PSA
    threshold) is located as text: its numbers do not identify it. A value whose numbers carry labels ("56 months in
    the abiraterone trial; 54 months in the abiraterone and enzalutamide trial") is still numeric - the labels say
    which number is which - so a value counts as text only when its words outnumber its numbers more than four to one."""
    words = content_words(value)
    strong = [n for n in numbers(value) if not _weak(n)]
    return len(words) >= 3 and len(words) > 4 * max(1, len(strong))


def content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", str(text or "").lower()) if w not in STOP]


# ---- the page as words ----------------------------------------------------------------------------------------------

class PageWords:
    """A page's words in reading order, each with its rectangle, the numbers it holds, and its line."""

    def __init__(self, page: Any):
        self.page = page
        raw = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
        self.words = raw
        self.text = [str(w[4]) for w in raw]
        self.lower = [re.sub(r"[^a-z0-9.%\-]+", "", str(w[4]).lower().translate(_DECIMAL)) for w in raw]
        self.nums = [word_numbers(w[4]) for w in raw]
        self._join_thousands()
        self.bare = [re.sub(r"[^a-z0-9\-]+", "", str(w[4]).lower()) for w in raw]

    def _join_thousands(self) -> None:
        """A number printed with a thin-space thousands separator ("100 000") comes out of the PDF as two words. Join
        them only when the second is a group of zeros, which no table cell prints. Any wider rule is wrong on real
        papers: JCO sets "n = 497" with the equals sign encoded as the glyph 5 ("n 5 497"), and numbers-at-risk rows
        print adjacent columns a space apart ("193 103") - both were being merged into one number."""
        for i in range(len(self.words) - 1):
            a, b = self.words[i], self.words[i + 1]
            if (a[5], a[6]) != (b[5], b[6]):
                continue
            if re.fullmatch(r"\d{1,3}", str(a[4])) and re.fullmatch(r"000[.,;:)%]*", str(b[4])):
                gap, height = b[0] - a[2], max(1.0, a[3] - a[1])
                if gap < 0.45 * height:
                    joined = canon(str(a[4]) + "000")
                    self.nums[i], self.nums[i + 1] = [joined], [joined]

    def __len__(self) -> int:
        return len(self.words)

    def positions(self, number: str) -> List[int]:
        return [i for i, ns in enumerate(self.nums) if number in ns]

    def boxes(self, idx: Iterable[int]) -> List[List[float]]:
        """One rectangle per printed line over the given words."""
        lines: Dict[Tuple[int, int], List[float]] = {}
        order: List[Tuple[int, int]] = []
        for i in sorted(set(idx)):
            x0, y0, x1, y1, _, block, line, _ = self.words[i]
            key = (block, line)
            if key not in lines:
                lines[key] = [x0, y0, x1, y1]
                order.append(key)
            else:
                b = lines[key]
                lines[key] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
        return [[round(v, 2) for v in lines[k]] for k in order]

    def span_text(self, idx: Iterable[int]) -> str:
        return " ".join(self.text[i] for i in sorted(set(idx)))

    def column_hits(self, center: int, tokens: Sequence[str]) -> int:
        if not tokens:
            return 0
        lo, hi = max(0, center - ANCHOR_RADIUS), min(len(self.lower), center + ANCHOR_RADIUS)
        window = set(self.lower[lo:hi])
        return sum(1 for t in tokens if any(t in w for w in window))


def _merge(rects: Sequence[Sequence[float]]) -> List[List[float]]:
    """Merge rectangles that sit on the same line and touch, so a phrase is one box rather than one per word."""
    out: List[List[float]] = []
    for r in sorted(([float(v) for v in r] for r in rects), key=lambda r: (round(r[1]), r[0])):
        if out:
            last = out[-1]
            same_line = abs(last[1] - r[1]) < 3 and abs(last[3] - r[3]) < 3
            if same_line and r[0] - last[2] < 1.5 * (r[3] - r[1]):
                out[-1] = [last[0], min(last[1], r[1]), max(last[2], r[2]), max(last[3], r[3])]
                continue
        out.append(r)
    return [[round(v, 2) for v in r] for r in out]


# ---- finding things on a page ---------------------------------------------------------------------------------------

def _sentences(quote: str) -> List[str]:
    parts = sorted((" ".join(p.split()) for p in re.split(r"(?<=[.;:])\s+|\n", quote)), key=len, reverse=True)
    return [p for p in parts if len(p) >= 16]


def find_quote(pw: PageWords, quote: str) -> Optional[Dict[str, Any]]:
    """The quotation on the page, as the words it covers: whole, then by sentence, then the narrowest window that
    holds most of its words. Returns {idx, text, how} or None."""
    quote = " ".join(str(quote or "").split())
    if len(quote) < 8 or not len(pw):
        return None
    for probe, how in [(quote, "exact")] + [(s, "sentence") for s in _sentences(quote)]:
        hits = pw.page.search_for(probe, quads=False)
        if hits:
            idx = [i for i, w in enumerate(pw.words)
                   if any(w[0] >= h.x0 - 1 and w[2] <= h.x1 + 1 and w[1] >= h.y0 - 2 and w[3] <= h.y1 + 2 for h in hits[:1] + hits[1:])]
            # keep only the first occurrence's words: the hits for one match are consecutive lines
            if idx:
                first = _consecutive(idx)
                return {"idx": first, "text": pw.span_text(first), "how": how}
    wanted = [w for w in (re.sub(r"[^a-z0-9.%\-]+", "", t.lower().translate(_DECIMAL)) for t in quote.split()) if len(w) > 1]
    if len(wanted) < 4:
        return None
    size = min(len(wanted), QUOTE_WINDOW)
    target = set(wanted)
    best, best_at = 0, -1
    for start in range(0, max(1, len(pw) - size + 1)):
        score = sum(1 for w in pw.lower[start:start + size] if w in target)
        if score > best:
            best, best_at = score, start
    if best_at < 0 or best < max(4, int(0.6 * size)):
        return None
    idx = list(range(best_at, min(len(pw), best_at + size)))
    # trim words at either end that the quotation does not contain
    while idx and pw.lower[idx[0]] not in target:
        idx.pop(0)
    while idx and pw.lower[idx[-1]] not in target:
        idx.pop()
    return {"idx": idx, "text": pw.span_text(idx), "how": "window"} if idx else None


def _consecutive(idx: List[int], gap: int = 3) -> List[int]:
    run = [idx[0]]
    for i in idx[1:]:
        if i - run[-1] > gap:
            break
        run.append(i)
    return run


def find_value(pw: PageWords, value_nums: Sequence[str], anchor: Optional[int], tokens: Sequence[str],
               span: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    """All of the value's numbers printed together (within VALUE_WINDOW words). Among several occurrences, the one
    nearest the quotation wins, then the one with most column words around it, then the tightest."""
    wanted = list(dict.fromkeys(value_nums))
    if not wanted or all(_weak(n) for n in wanted) and anchor is None and not tokens:
        return None
    pos = {n: pw.positions(n) for n in wanted}
    if any(not p for p in pos.values()):
        return None
    lead = min(wanted, key=lambda n: len(pos[n]))  # the rarest number seeds the search
    candidates = []
    for p0 in pos[lead]:
        chosen = [p0]
        ok = True
        for n in wanted:
            if n == lead:
                continue
            near = [p for p in pos[n] if abs(p - p0) <= VALUE_WINDOW]
            if not near:
                ok = False
                break
            chosen.append(min(near, key=lambda p: abs(p - p0)))
        if ok:
            lo, hi = min(chosen), max(chosen)
            center = (lo + hi) // 2
            dist = abs(center - anchor) if anchor is not None else 10 ** 6
            candidates.append((dist, -pw.column_hits(center, tokens), hi - lo, sorted(set(chosen))))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[:3])
    idx = candidates[0][3]
    if all(_weak(n) for n in wanted):
        # only small integers: they must sit together and next to a column word, or inside the quotation
        inside = lambda ix: span is not None and span[0] - 3 <= min(ix) and max(ix) <= span[1] + 3
        tight = [c for c in candidates if c[2] <= 3 and (inside(c[3]) or c[0] <= 12 or _next_to_column(pw, c[3], tokens))]
        if not tight:
            return None
        idx = tight[0][3]
    return {"idx": idx, "text": pw.span_text(idx), "occurrences": len(candidates)}


def _with_neighbours(pw: PageWords, idx: Sequence[int], extra: Sequence[str], reach: int = 2) -> List[int]:
    """Add the value's small numbers when they are printed right beside the located ones ("7 (1.1)")."""
    out = set(idx)
    for n in extra:
        lo, hi = max(0, min(idx) - reach), min(len(pw), max(idx) + reach + 1)
        near = [i for i in range(lo, hi) if n in pw.nums[i]]
        if near:
            out.add(min(near, key=lambda i: min(abs(i - j) for j in idx)))
    return sorted(out)


def _next_to_column(pw: PageWords, idx: Sequence[int], tokens: Sequence[str], reach: int = 4) -> bool:
    lo, hi = max(0, min(idx) - reach), min(len(pw.lower), max(idx) + reach + 1)
    return any(t in pw.lower[i] for i in range(lo, hi) for t in tokens)


def find_number(pw: PageWords, number: str, anchor: Optional[int], tokens: Sequence[str]) -> Optional[int]:
    """The best single occurrence of one number: nearest the anchor, then with most column words around it."""
    pos = pw.positions(number)
    if not pos:
        return None
    ranked = sorted(pos, key=lambda p: (abs(p - anchor) if anchor is not None else 10 ** 6, -pw.column_hits(p, tokens)))
    best = ranked[0]
    if _weak(number) and (anchor is None or abs(best - anchor) > ANCHOR_RADIUS) and pw.column_hits(best, tokens) == 0:
        return None
    return best


def find_text_value(pw: PageWords, value: str, anchor: Optional[int]) -> Optional[Dict[str, Any]]:
    """A text value (a drug, an arm, a design): the narrowest window holding most of its content words."""
    words = list(dict.fromkeys(content_words(value)))
    if not words:
        return None
    need = max(1, int(round(0.7 * len(words))))
    hits = [i for i, w in enumerate(pw.lower) if any(t in w for t in words)]
    if not hits:
        return None
    best = None
    for s in hits:
        window = [i for i in hits if s <= i <= s + max(8, 3 * len(words))]
        found = {t for i in window for t in words if t in pw.lower[i]}
        if len(found) >= need:
            dist = abs(s - anchor) if anchor is not None else 10 ** 6
            key = (-len(found), dist, window[-1] - window[0])
            if best is None or key < best[0]:
                best = (key, window)
    if not best:
        return None
    return {"idx": best[1], "text": pw.span_text(best[1])}


def _cluster(parts: List[Tuple[int, int, str]], words, anchor_of, tokens) -> List[Tuple[int, int, str]]:
    """Keep the printed parts of a value that belong together: on one page, within a few lines of each other, and
    tied to the quotation or the column. A lone part far from everything else is dropped rather than boxed."""
    if not parts:
        return []
    best: List[Tuple[int, int, str]] = []
    for p in {pp for pp, _, _ in parts}:
        on = sorted((x for x in parts if x[0] == p), key=lambda x: x[1])
        group = [on[0]]
        groups = [group]
        for x in on[1:]:
            if x[1] - group[-1][1] <= 40:
                group.append(x)
            else:
                group = [x]
                groups.append(group)
        for g in groups:
            center = (g[0][1] + g[-1][1]) // 2
            anchored = (anchor_of(p) is not None and abs(center - anchor_of(p)) <= ANCHOR_RADIUS) \
                or words(p).column_hits(center, tokens) > 0
            if anchored and len(g) > len(best):
                best = g
    return best


# ---- parser chunks --------------------------------------------------------------------------------------------------

def _chunks_on(parse: Optional[Dict[str, Any]], page_no: int) -> List[Dict[str, Any]]:
    if not parse:
        return []
    out = []
    for c in parse.get("chunks") or []:
        g = c.get("grounding") or {}
        if g.get("page") == page_no - 1 and isinstance(g.get("box"), dict) and c.get("id"):
            out.append(c)
    return out


def _chunk_rect(chunk: Dict[str, Any], width: float, height: float) -> List[float]:
    b = chunk["grounding"]["box"]
    return [round(b["left"] * width, 2), round(b["top"] * height, 2), round(b["right"] * width, 2), round(b["bottom"] * height, 2)]


def chunk_ids_for(parse: Optional[Dict[str, Any]], page_no: int, rects: Sequence[Sequence[float]],
                  width: float, height: float) -> List[str]:
    """The parser chunks whose boxes contain the centre of any of the rectangles."""
    ids: List[str] = []
    for c in _chunks_on(parse, page_no):
        x0, y0, x1, y1 = _chunk_rect(c, width, height)
        for r in rects:
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
                ids.append(c["id"])
                break
    return ids


def _best_chunk(parse: Optional[Dict[str, Any]], page_no: int, modality: str, value: str, column: str
                ) -> Optional[Dict[str, Any]]:
    kind = {"table": "table", "figure": "figure"}.get(modality)
    if not kind:
        return None
    pool = [c for c in _chunks_on(parse, page_no) if kind in str(c.get("type") or "").lower()
            or (kind == "figure" and any(k in str(c.get("type") or "").lower() for k in ("figure", "chart", "image")))]
    if not pool:
        return None
    vn, ct = set(numbers(value)), set(content_words(column))
    def score(c):
        md = str(c.get("markdown") or "").lower()
        return (sum(1 for n in vn if n in numbers(md)), sum(1 for t in ct if t in md))
    return max(pool, key=score)


# ---- one cell -------------------------------------------------------------------------------------------------------

def _region(pw: PageWords, page_no: int, idx: Sequence[int], kind: str, origin: str, parse, **extra) -> Dict[str, Any]:
    rects = _merge(pw.boxes(idx))
    w, h = pw.page.rect.width, pw.page.rect.height
    return {"page": page_no, "kind": kind, "rects": rects, "text": pw.span_text(idx),
            "chunk_ids": chunk_ids_for(parse, page_no, rects, w, h), "from": origin, **extra}


def locate(doc: Any, parse: Optional[Dict[str, Any]], value: str, column: str,
           sources: Sequence[Dict[str, Any]], reasoning: Sequence[str] = ()) -> Dict[str, Any]:
    """Regions for one value.

    `doc` is an open fitz document; `sources` are the attributions to try, most trusted first, each
    {pages: [int], quote: str, modality: str, origin: str}; `reasoning` holds texts that may name the numbers a
    derived value was computed from.
    """
    value = " ".join(str(value or "").split())
    tokens = [t for t in content_words(column) if len(t) >= 4]
    value_nums = [n for n in numbers(value)]
    strong = [n for n in value_nums if not _weak(n)] or value_nums
    pages: List[int] = []
    quotes: Dict[int, List[Tuple[str, str]]] = {}
    modality = "text"
    for s in sources:
        for p in s.get("pages") or []:
            if isinstance(p, int) and 1 <= p <= len(doc):
                if p not in pages:
                    pages.append(p)
                if s.get("quote"):
                    quotes.setdefault(p, []).append((str(s["quote"]), str(s.get("origin") or "")))
        if s.get("modality") in ("table", "figure") and modality == "text":
            modality = s["modality"]
    result: Dict[str, Any] = {"value": value, "status": "none", "regions": [], "quote_on_page": "", "pages_checked": pages}
    if not pages:
        return result

    cache: Dict[int, PageWords] = {}
    def words(p: int) -> PageWords:
        if p not in cache:
            cache[p] = PageWords(doc[p - 1])
        return cache[p]

    # the quotation anchors everything else: where on each cited page the extraction said the evidence was
    anchors: Dict[int, Dict[str, Any]] = {}
    for p in pages:
        for q, origin in quotes.get(p, []):
            hit = find_quote(words(p), q)
            if hit:
                anchors[p] = {**hit, "origin": origin}
                break
    anchor_of = lambda p: (sum(anchors[p]["idx"]) // len(anchors[p]["idx"])) if p in anchors else None
    span_of = lambda p: (min(anchors[p]["idx"]), max(anchors[p]["idx"])) if p in anchors else None

    # 1. the value as printed. An identifier (a trial number) is matched as a whole word; a text-dominant value by its
    # words, not its incidental numbers; a numeric value by its numbers printed together.
    ids = identifiers(value)
    if ids:
        for p in sorted(pages, key=lambda p: p not in anchors):
            pw = words(p)
            idx = [i for i, b in enumerate(pw.bare) if any(b == t or b.startswith(t) for t in ids)]
            if idx:
                pick = min(idx, key=lambda i: abs(i - anchor_of(p)) if anchor_of(p) is not None else 0)
                result["regions"] = [_region(pw, p, [pick], "value", "value", parse, occurrences=len(idx))]
                result["status"] = "value"
                break
    if result["status"] == "none" and value_nums and not text_dominant(value):
        for p in sorted(pages, key=lambda p: p not in anchors):
            hit = find_value(words(p), strong, anchor_of(p), tokens, span_of(p))
            if hit:
                idx = _with_neighbours(words(p), hit["idx"], [n for n in value_nums if n not in strong])
                result["regions"] = [_region(words(p), p, idx, "value", "value", parse,
                                             occurrences=hit["occurrences"])]
                result["status"] = "value"
                break
    else:
        for p in sorted(pages, key=lambda p: p not in anchors):
            hit = find_text_value(words(p), value, anchor_of(p))
            if hit:
                result["regions"] = [_region(words(p), p, hit["idx"], "value", "value", parse)]
                result["status"] = "value"
                break

    # 2. the printed parts of a value, and 3. the numbers a derived value was computed from
    if result["status"] == "none" and value_nums and not text_dominant(value) and any(not _weak(n) for n in value_nums):
        parts: List[Tuple[int, int, str]] = []
        for n in dict.fromkeys(n for n in value_nums if not _weak(n)):
            for p in sorted(pages, key=lambda p: p not in anchors):
                i = find_number(words(p), n, anchor_of(p), tokens)
                if i is not None:
                    parts.append((p, i, n))
                    break
        parts = _cluster(parts, words, anchor_of, tokens)
        if parts:
            result["regions"] = [_region(words(p), p, [i], "value_parts", "value", parse, number=n) for p, i, n in parts]
            result["status"] = "value_parts"
        else:
            texts = list(reasoning) + [q for p in pages for q, _ in quotes.get(p, [])]
            operands = [n for t in texts for n in arithmetic_operands(t) if n not in value_nums]
            found = []
            for n in list(dict.fromkeys(operands))[:MAX_OPERANDS * 3]:
                for p in sorted(pages, key=lambda p: p not in anchors):
                    pw = words(p)
                    i = find_number(pw, n, anchor_of(p), tokens)
                    near = i is not None and ((anchor_of(p) is not None and abs(i - anchor_of(p)) <= ANCHOR_RADIUS)
                                              or pw.column_hits(i, tokens) > 0)
                    if near:
                        found.append(_region(pw, p, [i], "operand", "reasoning", parse, number=n))
                        break
                if len(found) >= MAX_OPERANDS:
                    break
            if found:
                result["regions"], result["status"] = found, "operand"

    # 4. the quotation, matched to the page
    if result["status"] == "none" and anchors:
        p = next(iter(anchors))
        a = anchors[p]
        result["regions"] = [_region(words(p), p, a["idx"], "quote", a["origin"], parse, how=a["how"])]
        result["status"] = "quote"

    # 5. a value read from a figure or a table image
    if result["status"] == "none" and modality in ("table", "figure"):
        for p in pages:
            c = _best_chunk(parse, p, modality, value, column)
            if c:
                pg = doc[p - 1]
                result["regions"] = [{"page": p, "kind": "chunk", "rects": [_chunk_rect(c, pg.rect.width, pg.rect.height)],
                                      "text": "", "chunk_ids": [c["id"]], "from": "parser"}]
                result["status"] = "chunk"
                break

    if result["status"] == "none" and pages:
        result["status"] = "page"  # read from a figure, or computed without stated arithmetic: the page, unboxed
    if anchors:
        first = next(iter(anchors))
        result["quote_on_page"] = anchors[first]["text"]
    return result


# ---- a run ----------------------------------------------------------------------------------------------------------

def _is_absence(value: str) -> bool:
    v = " ".join(str(value or "").split()).lower()
    return not v or v.startswith(("not reported", "not found", "n/a", "not applicable")) or v in ("no", "none")


def sources_for(cell: Dict[str, Any], arms: Dict[str, Dict[str, Any]], column: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The attributions to try for a reconciled cell, most trusted first, and the texts that may name its operands:
    the reconciled source, the verifier's readings of the same value, the stage's own finding, then each agent whose
    answer is the shipped value."""
    value = " ".join(str(cell.get("value") or "").split())
    same = lambda v: " ".join(str(v or "").split()).lower() == value.lower()
    out: List[Dict[str, Any]] = []
    reasoning = [str(cell.get("reasoning") or "")]
    src = cell.get("source") or {}
    pages = [int(a["page"]) for a in (cell.get("attribution") or []) if isinstance(a, dict) and a.get("page")]
    if src.get("page"):
        pages = [int(src["page"])] + [p for p in pages if p != int(src["page"])]
    if pages:
        out.append({"pages": pages, "quote": src.get("verbatim_quote") or "", "modality": src.get("modality") or "text",
                    "origin": "reconciled"})
    for chk in cell.get("checks") or []:
        if same(chk.get("value")) and chk.get("pages"):
            out.append({"pages": [int(p) for p in chk["pages"]], "quote": chk.get("evidence") or "",
                        "modality": chk.get("modality") or "text", "origin": "verifier"})
            reasoning.append(str(chk.get("evidence") or ""))
    own = cell.get("own_finding") or {}
    if same(own.get("value")) and own.get("pages"):
        out.append({"pages": [int(p) for p in own["pages"]], "quote": own.get("evidence") or "", "origin": "own reading"})
    for name, arm in arms.items():
        entry = arm.get(column)
        if isinstance(entry, dict) and same(entry.get("value")):
            reasoning.append(str(entry.get("reasoning") or ""))
            for a in entry.get("attribution") or []:
                if isinstance(a, dict) and a.get("page"):
                    out.append({"pages": [int(a["page"])], "quote": a.get("evidence") or a.get("verbatim_quote") or "",
                                "modality": a.get("modality") or "text", "origin": name})
    return out, reasoning


def build_doc(doc_id: str, stage_dir: Path, pdf_path: Path, parse: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    import fitz

    rec = json.loads((stage_dir / "reconciled_results.json").read_text())["columns"]
    arms = {}
    for name, folder in (("A", "agent_extractor"), ("B", "search_agent")):
        arms[name] = {}
        for f in (stage_dir.parent / folder).glob("*results*.json"):
            try:
                arms[name].update(json.loads(f.read_text()).get("columns") or {})
            except (OSError, json.JSONDecodeError):
                continue
    out: Dict[str, Any] = {}
    with fitz.open(str(pdf_path)) as doc:
        for column, cell in rec.items():
            if not isinstance(cell, dict) or _is_absence(cell.get("value")):
                continue
            srcs, reasoning = sources_for(cell, arms, column)
            out[column] = locate(doc, parse, cell.get("value"), column, srcs, reasoning)
    return {"version": VERSION, "doc_id": doc_id, "columns": out}


def stage_dir(results_root: Path, doc_id: str, run: Optional[str]) -> Path:
    base = results_root / doc_id / "runs" / run if run else results_root / doc_id
    return base / "reconciliation_agent"


def build_stage(doc_id: str, stage: Path) -> Optional[Path]:
    """Write evidence_locations.json into a reconciliation stage folder. Returns its path, or None with nothing to do."""
    from src.evisearch.services.highlight import load_landing_ai_parse, resolve_pdf_path

    if not (stage / "reconciled_results.json").exists():
        return None
    pdf = resolve_pdf_path(doc_id)
    if not pdf or not Path(pdf).exists():
        return None
    data = build_doc(doc_id, stage, Path(pdf), load_landing_ai_parse(doc_id))
    path = stage / LOCATIONS_FILE
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def build_run(doc_id: str, run: Optional[str], results_root: Path) -> Optional[Path]:
    return build_stage(doc_id, stage_dir(results_root, doc_id, run))


def load_locations(doc_id: str, run: Optional[str], results_root: Path) -> Dict[str, Any]:
    path = stage_dir(results_root, doc_id, run) / LOCATIONS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("columns") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def regions_on(cell: Dict[str, Any], page: int, scale: float = 1.0) -> List[List[float]]:
    """The rectangles of a located cell that fall on one page, scaled to a rendered image."""
    return [[v * scale for v in r] for reg in cell.get("regions") or [] if reg.get("page") == page for r in reg["rects"]]
