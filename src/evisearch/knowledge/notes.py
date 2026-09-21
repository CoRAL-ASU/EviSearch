"""Knowledge notes: the extraction knowledge base as markdown documents instead of one-line rules.

A note is a markdown file with YAML frontmatter under KNOWLEDGE_DIR/notes/<role>/<name>.md:

    ---
    id: endpoints
    role: definitions          # the directory, repeated for readability
    scope: global | table | family | column
    family: Region - N (%)     # for scope: family
    also_families: [...]       # further families the note governs
    columns: [...]             # for scope: column
    supersedes: [cv-0002, ...] # the conventions this note replaces
    changed: why it differs from those conventions (free text, for the audit trail)
    ---
    # Heading
    - the text the prompts receive

This is the only knowledge format. It replaced a `conventions.jsonl` of one-line rules, and two things
about the change mattered:

* **A note is delivered whole.** The unit of knowledge is a document about one topic, not a sentence, so the model
  reads a rule together with its scope and its exceptions instead of meeting twenty context-free bullets.
* **A note is addressed to a role.** `definitions/` says what a column means; `extraction/` says how to find and
  assemble a value. The auditor (the reconciliation stage's own reading pass) gets `definitions/` only, so it can
  disagree with the agents instead of inheriting their method. Every wrong rule in the previous knowledge base was
  confirmed unanimously because the same text went to all six prompts, including the checker.

Provenance lives in `notes_log.jsonl`, which records every edit: the note, the text, who asked and why. The
`supersedes` frontmatter ties a note back to the numbered conventions it absorbed, which is why those ids still appear.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from src.config import runtime_paths

_WRITE_LOCK = threading.Lock()

ROLE_DIRS: Dict[str, Sequence[str]] = {
    "agent": ("definitions", "extraction"),  # Agent A, Agent B, the markdown baseline, the reader
    "auditor": ("definitions",),             # the reconciliation stage's own reading pass
    "all": ("definitions", "extraction"),
}
SCOPES = ("column", "family", "table", "global")  # most specific first
HEADER = (
    "\n\nEXTRACTION KNOWLEDGE BASE\n"
    "These notes are how this table is read. Each note covers one topic and states its own scope and exceptions.\n"
    "When a note written for specific columns differs from a general one, the specific note governs those columns.\n"
)


@dataclass(frozen=True)
class Note:
    id: str
    role: str
    scope: str
    body: str
    path: str
    family: Optional[str] = None
    also_families: Sequence[str] = field(default_factory=tuple)
    columns: Sequence[str] = field(default_factory=tuple)
    supersedes: Sequence[str] = field(default_factory=tuple)
    changed: str = ""

    @property
    def families(self) -> Sequence[str]:
        return tuple(f for f in (self.family, *self.also_families) if f)


def notes_dir() -> Path:
    return runtime_paths.KNOWLEDGE_DIR / "notes"


def log_path() -> Path:
    """The append-only record of every edit to the notes: what changed, who asked, and why.

    This replaces conventions.jsonl. The knowledge itself now lives only in the markdown, because a rule is only
    readable next to its scope and its exceptions; what a log is still good for is provenance, so each entry says
    which note was edited, which feedback event prompted it, and what the text was before.
    """
    return runtime_paths.KNOWLEDGE_DIR / "notes_log.jsonl"


def _parse(path: Path, role: str) -> Note:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path} has no frontmatter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    scope = str(meta.get("scope", "global"))
    if scope not in SCOPES:
        raise ValueError(f"{path}: scope must be one of {SCOPES}, got {scope!r}")
    return Note(
        id=str(meta.get("id") or path.stem),
        role=role,
        scope=scope,
        body=body.strip(),
        path=str(path.relative_to(notes_dir())) if path.is_relative_to(notes_dir()) else str(path),
        family=meta.get("family"),
        also_families=tuple(meta.get("also_families") or ()),
        columns=tuple(meta.get("columns") or ()),
        supersedes=tuple(meta.get("supersedes") or ()),
        changed=str(meta.get("changed") or "").strip(),
    )


def load_notes(role: str = "all") -> List[Note]:
    """Every note a role may receive, ordered general before specific so a specific note is read last."""
    out: List[Note] = []
    for sub in ROLE_DIRS.get(role, ROLE_DIRS["all"]):
        directory = notes_dir() / sub
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            out.append(_parse(path, sub))
    out.sort(key=lambda n: (-SCOPES.index(n.scope), n.id))
    return out


def select_for(notes: Iterable[Note], columns: Optional[Iterable[str]] = None) -> List[Note]:
    """The notes a batch of columns receives. Global and table notes always; a family or column note only when the
    batch holds one of its columns, so a narrow rule never reaches a prompt for other columns (the failure Q10
    measured: an eligibility rule for the docetaxel columns was generalised to previous local therapy)."""
    notes = list(notes)
    if columns is None:
        return notes
    names = set(columns)

    def applies(note: Note) -> bool:
        if note.scope in ("global", "table"):
            return True
        if names & set(note.columns):
            return True
        return bool(note.families) and any(n.startswith(f) for f in note.families for n in names)

    return [n for n in notes if applies(n)]


def render(notes: Iterable[Note]) -> str:
    """The prompt text: the header, then each note's body verbatim under a scope label."""
    notes = list(notes)
    if not notes:
        return ""
    blocks = []
    for note in notes:
        where = ""
        if note.scope == "family" and note.families:
            where = f"  [applies to the {', '.join(note.families)} columns]"
        elif note.scope == "column" and note.columns:
            where = f"  [applies to: {', '.join(note.columns)}]"
        blocks.append(f"### {note.id}{where}\n{note.body}")
    return HEADER + "\n\n".join(blocks)


def fingerprint(notes: Optional[Iterable[Note]] = None) -> str:
    """Short content hash of the notes a run used, recorded in run settings so runs with different knowledge never mix."""
    notes = load_notes("all") if notes is None else list(notes)
    blob = json.dumps(sorted((n.role, n.id, n.body) for n in notes), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def snapshot() -> Path:
    """Freeze the notes for a run, content-addressed, so editing a note mid-run never changes its prompts."""
    notes = load_notes("all")
    path = runtime_paths.KNOWLEDGE_DIR / "note_snapshots" / f"{fingerprint(notes)}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"id": n.id, "role": n.role, "scope": n.scope, "family": n.family, "also_families": list(n.also_families),
             "columns": list(n.columns), "supersedes": list(n.supersedes), "changed": n.changed, "body": n.body,
             "path": n.path}
            for n in notes
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_snapshot(path: str | Path) -> List[Note]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    notes = [
        Note(id=r["id"], role=r["role"], scope=r["scope"], body=r["body"], path=r.get("path", ""),
             family=r.get("family"), also_families=tuple(r.get("also_families") or ()),
             columns=tuple(r.get("columns") or ()), supersedes=tuple(r.get("supersedes") or ()),
             changed=r.get("changed", ""))
        for r in payload
    ]
    notes.sort(key=lambda n: (-SCOPES.index(n.scope), n.id))
    return notes


def for_role(notes: Iterable[Note], role: str) -> List[Note]:
    """Filter loaded (or snapshotted) notes down to the directories a role may read."""
    allowed = set(ROLE_DIRS.get(role, ROLE_DIRS["all"]))
    return [n for n in notes if n.role in allowed]


def coverage() -> Dict[str, List[str]]:
    """Which conventions the notes tree claims to replace, for checking the migration left nothing behind."""
    claimed: Dict[str, List[str]] = {}
    for note in load_notes("all"):
        for cid in note.supersedes:
            claimed.setdefault(cid, []).append(note.id)
    return claimed


# ---- editing: how a reviewer's correction becomes knowledge ----------------------------------------------------------

def governing(columns: Iterable[str], role: str = "agent") -> List[Note]:
    """The notes that already govern these columns, most specific last.

    This is what replaces the old integrity gate. That gate existed because a one-line convention arrived with no
    context, so duplicates, overlaps and contradictions had to be detected by a model comparing sentence pairs. A note
    is a document about one topic: the way to avoid contradicting it is to read it and edit it, which is what this
    returns for a proposer or a reviewer to work from.
    """
    return select_for(for_role(load_notes("all"), role), columns)


def _log(entry: Dict[str, Any]) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z", **entry}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_entries() -> List[Dict[str, Any]]:
    path = log_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _path_of(note_id: str) -> Optional[Path]:
    for note in load_notes("all"):
        if note.id == note_id:
            return notes_dir() / note.path
    return None


def apply_edit(note_id: str, text: str, *, heading: Optional[str] = None, by: str = "",
               why: str = "", event: str = "", role: str = "extraction") -> Dict[str, Any]:
    """Add or replace text in a note, and record the edit.

    `heading` names the section the text belongs under; a heading that is not in the note is appended as a new one,
    which is how a note grows a topic it did not cover. Without a heading the text goes to the end of the note. A note
    id that does not exist is created under `role`, so a genuinely new topic does not have to be forced into an
    existing document.

    Returns the note's id, its path, and the fingerprint of the whole tree after the edit - the fingerprint being what
    a run records, so any result can be traced to the exact knowledge that produced it.
    """
    text = text.strip()
    if not text:
        raise ValueError("an edit needs text")
    with _WRITE_LOCK:
        path = _path_of(note_id)
        created = path is None
        if created:
            if role not in ("definitions", "extraction"):
                raise ValueError(f"role must be definitions or extraction, got {role!r}")
            path = notes_dir() / role / f"{note_id}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            front = f"---\nid: {note_id}\nrole: {role}\nscope: global\n---\n"
            path.write_text(front + f"\n# {heading or note_id}\n\n{text}\n", encoding="utf-8")
            before = ""
        else:
            before = path.read_text(encoding="utf-8")
            body = before
            if heading:
                pattern = re.compile(r"^(#+\s*" + re.escape(heading) + r"\s*)$", re.MULTILINE)
                match = pattern.search(body)
                if match:
                    start = match.end()
                    nxt = re.compile(r"^#+\s", re.MULTILINE).search(body, start)
                    end = nxt.start() if nxt else len(body)
                    body = body[:end].rstrip() + "\n" + text + "\n\n" + body[end:].lstrip("\n")
                else:
                    body = body.rstrip() + f"\n\n# {heading}\n\n{text}\n"
            else:
                body = body.rstrip() + "\n" + text + "\n"
            path.write_text(body, encoding="utf-8")
        after = fingerprint(load_notes("all"))
        _log({"note": note_id, "path": str(path.relative_to(notes_dir())), "created": created,
              "heading": heading or "", "text": text, "by": by, "why": why, "event": event,
              "fingerprint": after, "bytes_before": len(before)})
        return {"note": note_id, "path": str(path), "created": created, "fingerprint": after}
