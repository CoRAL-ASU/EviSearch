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

Two things distinguish notes from `conventions.jsonl`:

* **A note is delivered whole.** The unit of knowledge is a document about one topic, not a sentence, so the model
  reads a rule together with its scope and its exceptions instead of meeting twenty context-free bullets.
* **A note is addressed to a role.** `definitions/` says what a column means; `extraction/` says how to find and
  assemble a value. The auditor (the reconciliation stage's own reading pass) gets `definitions/` only, so it can
  disagree with the agents instead of inheriting their method. Every wrong rule in the previous knowledge base was
  confirmed unanimously because the same text went to all six prompts, including the checker.

`conventions.jsonl` stays as the append-only decision log: it records where a rule came from and who approved it.
The notes tree is what the prompts read. `supersedes` ties each note back to the conventions it replaced.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from src.evisearch.knowledge import conventions

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
    return conventions.kb_dir() / "notes"


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
    path = conventions.kb_dir() / "note_snapshots" / f"{fingerprint(notes)}.json"
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
