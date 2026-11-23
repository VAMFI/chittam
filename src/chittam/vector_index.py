"""Pluggable ANN / quantized vector index backends (USearch HNSW + Numpy fallback)."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    from usearch.index import Index as _USearchIndex

    USEARCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback path
    _USearchIndex = None
    USEARCH_AVAILABLE = False

_VALID_DTYPES = ("i8", "f16", "f32")

_NP_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "i8": np.dtype(np.int8),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
}


def validate_vector_dtype(dtype: str) -> str:
    """Validate and normalize vector quantization dtype ('i8', 'f16', or 'f32')."""
    norm = dtype.strip().lower()
    if norm not in _VALID_DTYPES:
        raise ValueError(f"Unsupported vector dtype {dtype!r}; expected one of {_VALID_DTYPES}")
    return norm


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.size == 0:
        return mat
    if mat.ndim == 1:
        norm = float(np.linalg.norm(mat))
        if norm > 0.0:
            return (mat / norm).astype(np.float32)
        return np.zeros_like(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return np.where(norms > 0.0, mat / safe, 0.0).astype(np.float32)


def quantize_array(vectors: np.ndarray, dtype: str = "i8") -> np.ndarray:
    """L2-normalize and encode float32 vectors into i8, f16, or f32 representation."""
    dt = validate_vector_dtype(dtype)
    arr = np.asarray(vectors)
    target_np = _NP_DTYPE_MAP[dt]
    if arr.dtype == target_np and dt == "i8":
        return np.ascontiguousarray(arr)
    normed = _l2_normalize_rows(arr)
    if dt == "i8":
        return np.ascontiguousarray(np.clip(np.round(normed * 127.0), -127, 127).astype(np.int8))
    if dt == "f16":
        return np.ascontiguousarray(normed.astype(np.float16))
    return np.ascontiguousarray(normed.astype(np.float32))


def quantize_vector(vec: np.ndarray, dtype: str = "i8") -> bytes:
    """Quantize a 1D vector into raw bytes in the configured storage dtype."""
    encoded = quantize_array(np.asarray(vec), dtype=dtype).reshape(-1)
    return encoded.tobytes()


def decode_stored_array(blobs: list[bytes], dim: int, dtype: str = "i8") -> np.ndarray:
    """Decode raw stored vector blobs directly in their encoded numpy dtype (N, dim)."""
    dt = validate_vector_dtype(dtype)
    np_dt = _NP_DTYPE_MAP[dt]
    if not blobs:
        return np.empty((0, dim), dtype=np_dt)
    expected = dim * np_dt.itemsize
    valid = [b for b in blobs if len(b) == expected]
    if len(valid) != len(blobs):
        raise ValueError(f"Vector blob size mismatch for dim={dim}, dtype={dt}")
    joined = b"".join(valid)
    return np.ascontiguousarray(np.frombuffer(joined, dtype=np_dt).reshape(len(blobs), dim))


def dequantize_vector(blob: bytes, dim: int, dtype: str = "i8") -> np.ndarray:
    """Decode a stored vector blob back into an L2-normalized float32 vector."""
    dt = validate_vector_dtype(dtype)
    np_dt = _NP_DTYPE_MAP[dt]
    expected = dim * np_dt.itemsize
    if len(blob) != expected:
        raise ValueError(f"Expected {expected} bytes for dim={dim}, dtype={dt}, got {len(blob)}")
    arr = np.frombuffer(blob, dtype=np_dt).astype(np.float32)
    if dt == "i8":
        arr = arr / 127.0
    return _l2_normalize_rows(arr)


def dequantize_matrix(blobs: list[bytes], dim: int, dtype: str = "i8") -> np.ndarray:
    """Decode multiple stored blobs into an L2-normalized float32 matrix (N, dim)."""
    dt = validate_vector_dtype(dtype)
    if not blobs:
        return np.empty((0, dim), dtype=np.float32)
    encoded = decode_stored_array(blobs, dim=dim, dtype=dt)
    mat = encoded.astype(np.float32)
    if dt == "i8":
        mat = mat / 127.0
    return _l2_normalize_rows(mat)


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for pluggable ANN / vector similarity backends."""

    dim: int
    dtype: str
    backend_name: str

    def count(self) -> int:
        """Return total indexed vector keys."""
        ...

    def add(self, keys: list[int] | np.ndarray, vectors: np.ndarray) -> None:
        """Insert or update vectors keyed by integer IDs."""
        ...

    def remove(self, keys: list[int] | np.ndarray) -> None:
        """Remove keys from the index if present."""
        ...

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return top-k (key, cosine_similarity) sorted descending."""
        ...

    def save(self) -> None:
        """Persist index changes atomically if file-backed."""
        ...

    def compact(self) -> None:
        """Compact index structure and persist."""
        ...

    def reload_if_changed(self) -> bool:
        """Reload index from disk if modified externally; return True if reloaded."""
        ...

    def clear(self) -> None:
        """Reset index to empty."""
        ...


class USearchVectorIndex:
    """USearch HNSW cosine vector index with quantized i8/f16/f32 storage and atomic saves."""

    backend_name = "usearch"

    def __init__(self, index_path: Path | str, dim: int, dtype: str = "i8") -> None:
        if not USEARCH_AVAILABLE or _USearchIndex is None:  # pragma: no cover
            raise RuntimeError("usearch package is not installed")
        self.index_path = Path(index_path).expanduser()
        self.dim = int(dim)
        self.dtype = validate_vector_dtype(dtype)
        self._lock = threading.RLock()
        self._loaded_stat: tuple[int, int] | None = None
        self._index = self._open_or_create()

    def _matches_schema(self, idx: Any) -> bool:
        if idx is None:
            return False
        if int(getattr(idx, "ndim", -1)) != self.dim:
            return False
        dt_str = str(getattr(idx, "dtype", "")).lower()
        return self.dtype in dt_str

    def _new_empty(self) -> Any:
        return _USearchIndex(
            ndim=self.dim,
            metric="cos",
            dtype=self.dtype,
            expansion_add=128,
            expansion_search=128,
        )

    def _open_or_create(self) -> Any:
        if self.index_path.exists():
            try:
                st = self.index_path.stat()
                if st.st_size > 0:
                    restored = _USearchIndex.restore(str(self.index_path), view=False)
                    if self._matches_schema(restored):
                        self._loaded_stat = (st.st_mtime_ns, st.st_size)
                        return restored
            except Exception:
                pass
        self._loaded_stat = None
        return self._new_empty()

    def count(self) -> int:
        with self._lock:
            return int(len(self._index))

    def add(self, keys: list[int] | np.ndarray, vectors: np.ndarray) -> None:
        key_arr = np.asarray(keys, dtype=np.uint64).reshape(-1)
        if key_arr.size == 0:
            return
        mat = np.asarray(vectors)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.shape != (key_arr.size, self.dim):
            raise ValueError(
                f"Expected vectors shape ({key_arr.size}, {self.dim}), got {mat.shape}"
            )
        encoded = quantize_array(mat, dtype=self.dtype)

        with self._lock:
            if len(self._index) > 0:
                present_mask = np.asarray(self._index.contains(key_arr), dtype=bool).reshape(-1)
                if np.any(present_mask):
                    self._index.remove(key_arr[present_mask], compact=False)
            self._index.add(key_arr, encoded)

    def remove(self, keys: list[int] | np.ndarray) -> None:
        key_arr = np.asarray(keys, dtype=np.uint64).reshape(-1)
        if key_arr.size == 0:
            return
        with self._lock:
            if len(self._index) == 0:
                return
            present_mask = np.asarray(self._index.contains(key_arr), dtype=bool).reshape(-1)
            if np.any(present_mask):
                self._index.remove(key_arr[present_mask], compact=True)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        q_arr = np.asarray(query_vec).reshape(-1)
        if q_arr.size != self.dim:
            return []
        q_enc = quantize_array(q_arr, dtype=self.dtype).reshape(-1)

        with self._lock:
            total = int(len(self._index))
            if total == 0:
                return []
            top_k = min(int(k), total)
            matches = self._index.search(q_enc, top_k, exact=(total <= 1024))
            out: list[tuple[int, float]] = []
            for key, dist in zip(matches.keys, matches.distances):
                sim = max(-1.0, min(1.0, 1.0 - float(dist)))
                if sim > 0.0:
                    out.append((int(key), sim))
            out.sort(key=lambda item: item[1], reverse=True)
            return out

    def save(self) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.index_path.with_name(
                f".{self.index_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                self._index.save(str(tmp_path))
                os.replace(tmp_path, self.index_path)
                st = self.index_path.stat()
                self._loaded_stat = (st.st_mtime_ns, st.st_size)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def compact(self) -> None:
        with self._lock:
            self._index = self._index.copy()
            self.save()

    def reload_if_changed(self) -> bool:
        with self._lock:
            if not self.index_path.exists():
                if self._loaded_stat is not None or len(self._index) > 0:
                    self._index = self._new_empty()
                    self._loaded_stat = None
                    return True
                return False
            try:
                st = self.index_path.stat()
            except OSError:
                return False
            cur_stat = (st.st_mtime_ns, st.st_size)
            if cur_stat == self._loaded_stat or st.st_size == 0:
                return False
            try:
                restored = _USearchIndex.restore(str(self.index_path), view=False)
                if self._matches_schema(restored):
                    self._index = restored
                    self._loaded_stat = cur_stat
                    return True
            except Exception:
                return False
            return False

    def clear(self) -> None:
        with self._lock:
            self._index = self._new_empty()
            self._loaded_stat = None


class NumpyVectorIndex:
    """In-memory quantized cosine fallback vector index."""

    backend_name = "numpy"

    def __init__(self, dim: int, dtype: str = "i8") -> None:
        self.dim = int(dim)
        self.dtype = validate_vector_dtype(dtype)
        self._lock = threading.RLock()
        self._vectors: dict[int, np.ndarray] = {}

    def count(self) -> int:
        with self._lock:
            return len(self._vectors)

    def add(self, keys: list[int] | np.ndarray, vectors: np.ndarray) -> None:
        key_arr = np.asarray(keys, dtype=np.int64).reshape(-1)
        if key_arr.size == 0:
            return
        mat = np.asarray(vectors)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.shape != (key_arr.size, self.dim):
            raise ValueError(
                f"Expected vectors shape ({key_arr.size}, {self.dim}), got {mat.shape}"
            )
        encoded = quantize_array(mat, dtype=self.dtype)
        with self._lock:
            for k, row in zip(key_arr, encoded):
                self._vectors[int(k)] = row.copy()

    def remove(self, keys: list[int] | np.ndarray) -> None:
        key_arr = np.asarray(keys, dtype=np.int64).reshape(-1)
        with self._lock:
            for k in key_arr:
                self._vectors.pop(int(k), None)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        q = _l2_normalize_rows(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        if q.size != self.dim or float(np.linalg.norm(q)) == 0.0:
            return []

        with self._lock:
            if not self._vectors:
                return []
            keys = list(self._vectors.keys())
            mat = np.vstack([self._vectors[key] for key in keys]).astype(np.float32)

        if self.dtype == "i8":
            mat = mat / 127.0
        mat = _l2_normalize_rows(mat)
        sims = mat @ q
        order = np.argsort(-sims)[: min(int(k), len(keys))]
        out: list[tuple[int, float]] = []
        for pos in order:
            sim = float(sims[int(pos)])
            if sim > 0.0:
                out.append((keys[int(pos)], sim))
        return out

    def save(self) -> None:
        return None

    def compact(self) -> None:
        return None

    def reload_if_changed(self) -> bool:
        return False

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()


def build_vector_index(
    index_path: Path | None,
    dim: int,
    dtype: str = "i8",
    backend: str = "usearch",
) -> VectorIndex:
    """Instantiate USearchVectorIndex when requested & available, else NumpyVectorIndex."""
    norm_backend = backend.strip().lower()
    norm_dtype = validate_vector_dtype(dtype)
    if norm_backend == "usearch" and index_path is not None and USEARCH_AVAILABLE:
        return USearchVectorIndex(index_path=index_path, dim=dim, dtype=norm_dtype)
    return NumpyVectorIndex(dim=dim, dtype=norm_dtype)
