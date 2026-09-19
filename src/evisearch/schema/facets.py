"""What a column header asks for, read deterministically from its text.

"Median OS (mo) | High volume | Treatment" -> characteristic "Median OS", statistic "median", unit "months",
subgroup "High volume", arm "Treatment". Headers are split on " | "; the first part carries the characteristic, the
statistic and the unit, the others are arms, subgroups or categories. Abbreviations that are not in the glossary (or
tokens such as COE_RCT_IND_OVERALL_RJ) are listed as cryptic so the schema agent asks about them instead of guessing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

ARMS = {"treatment": "Treatment", "control": "Control", "experimental": "Treatment", "placebo": "Control"}
SUBGROUPS = {
    "overall", "high volume", "low volume", "high", "low", "synchronous", "metachronous", "de novo", "recurrent",
    "high risk", "low risk",
}
UNITS = {"mo": "months", "months": "months", "years": "years", "yrs": "years", "%": "percent", "weeks": "weeks", "days": "days"}
GLOSSARY = {
    "N", "OS", "PFS", "ORR", "CR", "PR", "SD", "PD", "PS", "ECOG", "PSA", "ADT", "AE", "AEs", "NCT", "QoL", "HR", "CI",
    "RCT", "mo", "Y/N", "ID",
}


def _statistic(base: str) -> Dict[str, Any]:
    text = base
    out: Dict[str, Any] = {"statistic": "", "unit": ""}
    if re.search(r"\bN\s*\(%\)", text):
        out["statistic"], out["unit"] = "count (percent)", "patients"
        text = re.sub(r"\s*-?\s*\bN\s*\(%\)", "", text)
    elif re.search(r"-\s*N\b", text) or re.search(r"\bNo\.\s+of\b", text):
        out["statistic"], out["unit"] = "count", "patients"
        text = re.sub(r"\s*-\s*N\b", "", text)
    elif re.search(r"\bY/N\b", text):
        out["statistic"] = "yes/no"
        text = re.sub(r"\s*-?\s*\bY/N\b", "", text)
    unit = re.search(r"\(([^)]+)\)\s*$", text)
    if unit and unit.group(1).strip().lower() in UNITS:
        out["unit"] = UNITS[unit.group(1).strip().lower()]
        text = text[: unit.start()]
        if out["unit"] == "percent" and not out["statistic"]:
            out["statistic"] = "rate (percent)" if re.search(r"\brate\b", text, re.I) else "percent"
    if not out["statistic"]:
        if re.match(r"\s*median\b", text, re.I):
            out["statistic"] = "median"
        elif re.search(r"\breported\b", text, re.I):
            out["statistic"] = "yes/no"
    out["characteristic"] = re.sub(r"\s+", " ", text).strip(" -")
    return out


def cryptic_tokens(header: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_/]*", header)
    out = []
    for token in tokens:
        if "_" in token or (token.isupper() and len(token) >= 2 and token not in GLOSSARY):
            out.append(token)
    return sorted(set(out))


def parse_header(header: str) -> Dict[str, Any]:
    parts = [p.strip() for p in header.split("|")]
    facets = _statistic(parts[0])
    facets.update({"arm": "", "subgroup": "", "category": "", "family": parts[0], "parts": parts})
    for part in parts[1:]:
        key = part.lower()
        if key in ARMS:
            facets["arm"] = ARMS[key]
        elif key in SUBGROUPS:
            facets["subgroup"] = part
        else:
            facets["category"] = (facets["category"] + " / " + part).strip(" /")
    facets["cryptic"] = cryptic_tokens(header)
    return facets
