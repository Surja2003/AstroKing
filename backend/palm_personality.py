from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PersonalityMatch:
    label: str
    score: float
    summary: Optional[str] = None
    traits: Optional[list[str]] = None


_index_lock = threading.Lock()
_index_data: Optional[dict[str, Any]] = None


def _lazy_import_numpy():
    import numpy as np  # type: ignore

    return np


def _default_index_path() -> str:
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "palm_trait_index.npz")


def _get_index_path() -> str:
    return os.environ.get("PALM_TRAIT_INDEX", "").strip() or _default_index_path()


def _load_index() -> dict[str, Any]:
    global _index_data
    if _index_data is not None:
        return _index_data

    with _index_lock:
        if _index_data is not None:
            return _index_data

        path = _get_index_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trait index not found at '{path}'. "
                "Build one from your curated archetype images and set PALM_TRAIT_INDEX if needed."
            )

        np = _lazy_import_numpy()
        data = np.load(path, allow_pickle=True)
        if "X" not in data or "labels" not in data:
            raise ValueError("Index file must contain keys: X, labels")

        X = data["X"].astype("float32")
        labels = [str(x) for x in data["labels"].tolist()]
        summaries = None
        if "summaries" in data:
            summaries = [None if x is None else str(x) for x in data["summaries"].tolist()]
        traits_json = None
        if "traits_json" in data:
            traits_json = [None if x is None else str(x) for x in data["traits_json"].tolist()]

        _index_data = {
            "X": X,
            "labels": labels,
            "summaries": summaries,
            "traits_json": traits_json,
        }
        return _index_data


def _cosine_similarity_matrix(x, X):
    np = _lazy_import_numpy()
    x = np.asarray(x, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)

    x_norm = np.linalg.norm(x) + 1e-12
    X_norm = np.linalg.norm(X, axis=1) + 1e-12
    sims = (X @ x) / (X_norm * x_norm)
    return sims


def match_personality(embedding: list[float], *, top_k: int = 3) -> dict[str, Any]:
    """Return top-k archetype matches for a given embedding vector."""

    try:
        np = _lazy_import_numpy()
        idx = _load_index()

        X = idx["X"]
        labels = idx["labels"]
        summaries = idx.get("summaries")
        traits_json = idx.get("traits_json")

        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError("embedding must be a 1D vector")
        if X.shape[1] != emb.shape[0]:
            raise ValueError(f"embedding dim mismatch: got {emb.shape[0]}, expected {X.shape[1]}")

        sims = _cosine_similarity_matrix(emb, X)
        k = int(max(1, min(int(top_k), int(sims.shape[0]))))
        top_idx = np.argsort(-sims)[:k]

        matches: list[PersonalityMatch] = []
        for i in top_idx.tolist():
            label = labels[i]
            summary = summaries[i] if summaries is not None else None

            traits = None
            if traits_json is not None and traits_json[i]:
                try:
                    traits = json.loads(traits_json[i])
                    if not isinstance(traits, list):
                        traits = None
                    else:
                        traits = [str(t) for t in traits]
                except Exception:
                    traits = None

            matches.append(
                PersonalityMatch(
                    label=label,
                    score=float(sims[i]),
                    summary=summary,
                    traits=traits,
                )
            )

        # Flatten traits for convenience.
        trait_list: list[str] = []
        for m in matches:
            for t in (m.traits or []):
                if t not in trait_list:
                    trait_list.append(t)

        return {
            "status": "ok",
            "matches": [m.__dict__ for m in matches],
            "traits": trait_list,
        }

    except FileNotFoundError as e:
        return {"status": "unavailable", "error": str(e), "matches": [], "traits": []}
    except Exception as e:
        return {"status": "error", "error": str(e), "matches": [], "traits": []}


def get_index_status() -> dict[str, Any]:
    """Lightweight status probe for health/debug screens."""

    try:
        idx = _load_index()
        X = idx["X"]
        labels = idx["labels"]
        dim = int(X.shape[1]) if getattr(X, "ndim", 0) == 2 else None
        return {
            "status": "ok",
            "count": int(len(labels)),
            "dim": dim,
            "path": _get_index_path(),
        }
    except FileNotFoundError as e:
        return {"status": "unavailable", "error": str(e), "path": _get_index_path()}
    except Exception as e:
        return {"status": "error", "error": str(e), "path": _get_index_path()}
