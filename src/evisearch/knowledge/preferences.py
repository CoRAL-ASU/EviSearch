from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREFERENCES_PATH = PROJECT_ROOT / "src" / "evisearch" / "knowledge" / "preferences.md"


def load_extraction_preferences(path: Path | None = None) -> str:
    """Return human extraction preferences, or an empty string when unset."""
    pref_path = path or PREFERENCES_PATH
    try:
        if not pref_path.exists():
            return ""
        return pref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
