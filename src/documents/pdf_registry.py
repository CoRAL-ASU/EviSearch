from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.config.runtime_paths import DATASET_DIR, RESULTS_ROOT, UPLOADS_DIR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_doc_id_from_sha(sha256_hex: str) -> str:
    return f"pdf_{sha256_hex[:12]}"


def dataset_doc_id_from_path(pdf_path: Path, dataset_dir: Path = DATASET_DIR) -> str:
    rel = pdf_path.relative_to(dataset_dir)
    return str(rel.with_suffix("")).replace("/", "_")


def _registry_dir(results_root: Path) -> Path:
    return results_root / "_registry"


def _sha_index_path(results_root: Path) -> Path:
    return _registry_dir(results_root) / "sha_index.json"


def _upload_metadata_dir(uploads_dir: Path) -> Path:
    return uploads_dir / "_metadata"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_sha_index(results_root: Path = RESULTS_ROOT) -> Dict[str, Dict[str, Any]]:
    raw = _read_json(_sha_index_path(results_root))
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _save_sha_index(index: Dict[str, Dict[str, Any]], results_root: Path = RESULTS_ROOT) -> None:
    _write_json(_sha_index_path(results_root), index)


def _document_has_pdf(doc_id: str, results_root: Path, dataset_dir: Path) -> bool:
    results_dir = results_root / doc_id
    if (results_dir / f"{doc_id}.pdf").exists():
        return True
    if any(results_dir.glob("*.pdf")):
        return True
    dataset_exact = dataset_dir / f"{doc_id}.pdf"
    if dataset_exact.exists():
        return True
    for dataset_doc_id, pdf_path in _iter_dataset_pdfs(dataset_dir):
        if dataset_doc_id == doc_id and pdf_path.exists():
            return True
    return False


def get_upload_record(upload_doc_id: str, uploads_dir: Path = UPLOADS_DIR) -> Dict[str, Any]:
    return _read_json(_upload_metadata_dir(uploads_dir) / f"{upload_doc_id}.json")


def resolve_canonical_doc_id(
    doc_id: str,
    uploads_dir: Path = UPLOADS_DIR,
    results_root: Path = RESULTS_ROOT,
    dataset_dir: Path = DATASET_DIR,
) -> str:
    if not doc_id.startswith("upload_"):
        return doc_id
    record = get_upload_record(doc_id, uploads_dir=uploads_dir)
    canonical_doc_id = str(record.get("canonical_doc_id") or "").strip()
    if canonical_doc_id and _document_has_pdf(canonical_doc_id, results_root, dataset_dir):
        return canonical_doc_id
    return doc_id


def get_registered_document(doc_id: str, results_root: Path = RESULTS_ROOT) -> Optional[Dict[str, Any]]:
    index = _load_sha_index(results_root=results_root)
    for sha256_hex, entry in index.items():
        canonical_doc_id = str(entry.get("canonical_doc_id") or "").strip()
        if canonical_doc_id != doc_id:
            continue
        merged = dict(entry)
        merged.setdefault("sha256", sha256_hex)
        return merged
    return None


def _iter_dataset_pdfs(dataset_dir: Path) -> Iterable[tuple[str, Path]]:
    if not dataset_dir.exists():
        return
    for pdf_path in sorted(dataset_dir.glob("**/*.pdf")):
        if pdf_path.is_file():
            yield dataset_doc_id_from_path(pdf_path, dataset_dir=dataset_dir), pdf_path


def _iter_results_pdfs(results_root: Path) -> Iterable[tuple[str, Path]]:
    if not results_root.exists():
        return
    for doc_dir in sorted(results_root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.startswith("_"):
            continue
        primary = doc_dir / f"{doc_dir.name}.pdf"
        if primary.exists():
            yield doc_dir.name, primary
            continue
        pdfs = sorted(doc_dir.glob("*.pdf"))
        if pdfs:
            yield doc_dir.name, pdfs[0]


def _upsert_sha_index_entry(
    sha256_hex: str,
    entry: Dict[str, Any],
    results_root: Path = RESULTS_ROOT,
) -> Dict[str, Any]:
    index = _load_sha_index(results_root=results_root)
    current = index.get(sha256_hex) or {}
    merged = dict(current)
    merged.update(entry)
    aliases = list(dict.fromkeys((current.get("upload_aliases") or []) + (entry.get("upload_aliases") or [])))
    if aliases:
        merged["upload_aliases"] = aliases
    merged["updated_at"] = _utc_now_iso()
    if not current.get("created_at"):
        merged.setdefault("created_at", merged["updated_at"])
    else:
        merged["created_at"] = current["created_at"]
    index[sha256_hex] = merged
    _save_sha_index(index, results_root=results_root)
    merged.setdefault("sha256", sha256_hex)
    return merged


def _match_existing_document(
    sha256_hex: str,
    results_root: Path,
    dataset_dir: Path,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    index = _load_sha_index(results_root=results_root)
    entry = index.get(sha256_hex)
    if entry:
        canonical_doc_id = str(entry.get("canonical_doc_id") or "").strip()
        if canonical_doc_id and _document_has_pdf(canonical_doc_id, results_root, dataset_dir):
            merged = dict(entry)
            merged.setdefault("sha256", sha256_hex)
            return canonical_doc_id, merged

    for doc_id, pdf_path in _iter_dataset_pdfs(dataset_dir):
        if sha256_file(pdf_path) != sha256_hex:
            continue
        entry = _upsert_sha_index_entry(
            sha256_hex,
            {
                "canonical_doc_id": doc_id,
                "display_name": pdf_path.stem,
                "source": "dataset",
                "pdf_path": str(pdf_path),
            },
            results_root=results_root,
        )
        return doc_id, entry

    for doc_id, pdf_path in _iter_results_pdfs(results_root):
        if sha256_file(pdf_path) != sha256_hex:
            continue
        existing = get_registered_document(doc_id, results_root=results_root) or {}
        entry = _upsert_sha_index_entry(
            sha256_hex,
            {
                "canonical_doc_id": doc_id,
                "display_name": str(existing.get("display_name") or doc_id),
                "source": str(existing.get("source") or "upload"),
                "pdf_path": str(pdf_path),
            },
            results_root=results_root,
        )
        return doc_id, entry

    return None, None


def register_uploaded_pdf(
    pdf_bytes: bytes,
    original_filename: str,
    uploads_dir: Path = UPLOADS_DIR,
    results_root: Path = RESULTS_ROOT,
    dataset_dir: Path = DATASET_DIR,
) -> Dict[str, Any]:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_doc_id = "upload_" + uuid.uuid4().hex[:12]
    upload_path = uploads_dir / f"{upload_doc_id}.pdf"
    upload_path.write_bytes(pdf_bytes)

    sha256_hex = sha256_bytes(pdf_bytes)
    canonical_doc_id, entry = _match_existing_document(
        sha256_hex,
        results_root=results_root,
        dataset_dir=dataset_dir,
    )
    reused_existing_doc = canonical_doc_id is not None

    if canonical_doc_id is None:
        canonical_doc_id = canonical_doc_id_from_sha(sha256_hex)
        canonical_dir = results_root / canonical_doc_id
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_pdf_path = canonical_dir / f"{canonical_doc_id}.pdf"
        if not canonical_pdf_path.exists():
            canonical_pdf_path.write_bytes(pdf_bytes)
        entry = _upsert_sha_index_entry(
            sha256_hex,
            {
                "canonical_doc_id": canonical_doc_id,
                "display_name": Path(original_filename).stem or canonical_doc_id,
                "source": "upload",
                "pdf_path": str(canonical_pdf_path),
            },
            results_root=results_root,
        )
    else:
        entry = _upsert_sha_index_entry(
            sha256_hex,
            {
                "canonical_doc_id": canonical_doc_id,
                "display_name": str((entry or {}).get("display_name") or canonical_doc_id),
                "source": str((entry or {}).get("source") or "upload"),
                "pdf_path": str((entry or {}).get("pdf_path") or ""),
            },
            results_root=results_root,
        )

    entry = _upsert_sha_index_entry(
        sha256_hex,
        {
            "canonical_doc_id": canonical_doc_id,
            "display_name": str((entry or {}).get("display_name") or canonical_doc_id),
            "source": str((entry or {}).get("source") or "upload"),
            "pdf_path": str((entry or {}).get("pdf_path") or ""),
            "upload_aliases": [upload_doc_id],
        },
        results_root=results_root,
    )

    upload_record = {
        "upload_doc_id": upload_doc_id,
        "canonical_doc_id": canonical_doc_id,
        "sha256": sha256_hex,
        "original_filename": original_filename,
        "stored_path": str(upload_path),
        "reused_existing_doc": reused_existing_doc,
        "uploaded_at": _utc_now_iso(),
    }
    _write_json(_upload_metadata_dir(uploads_dir) / f"{upload_doc_id}.json", upload_record)

    return {
        "upload_doc_id": upload_doc_id,
        "upload_path": upload_path,
        "canonical_doc_id": canonical_doc_id,
        "sha256": sha256_hex,
        "reused_existing_doc": reused_existing_doc,
        "display_name": str((entry or {}).get("display_name") or canonical_doc_id),
        "source": str((entry or {}).get("source") or "upload"),
    }
