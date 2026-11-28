"""Embedding, vision, and reranking provider abstractions for Chittam."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from chittam.models import AppConfig, ExternalProviderConfig, LocalProviderConfig

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

_DEFAULT_VISION_PROMPT = (
    "Analyze this image for local semantic memory indexing.\n"
    "Respond concisely with:\n"
    "1. Description: what the image depicts.\n"
    "2. Visible Text / Labels: any verbatim UI text, code, headings, or diagram labels.\n"
    "3. Keywords: 5-10 relevant technical or topical keywords."
)


class ConsentRequiredError(RuntimeError):
    """Raised when an external provider is requested without explicit consent and opt-in."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for text and query embedding providers."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into an L2-normalized float32 matrix (N, D)."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query into an L2-normalized float32 vector (D,)."""
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for image captioning and OCR/diagram extraction providers."""

    def describe_image(self, image_path: Path, context_hint: str = "") -> str:
        """Return a concise searchable description of an image, or '' if unreadable."""
        ...


@runtime_checkable
class RerankProvider(Protocol):
    """Protocol for optional cross-encoder / LLM reranking providers."""

    def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Return list of (doc_index, score) sorted descending by relevance."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D vector or 2D batch of vectors in float32, guarding zero norms."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            return (arr / norm).astype(np.float32)
        return np.zeros_like(arr, dtype=np.float32)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = np.where(norms > 0.0, arr / safe_norms, 0.0)
    return normalized.astype(np.float32)


def _postprocess_embeddings(
    raw_vectors: list[list[float]] | np.ndarray,
    matryoshka_dim: int | None = None,
) -> np.ndarray:
    """Apply optional Matryoshka truncation and mandatory L2 re-normalization."""
    arr = np.asarray(raw_vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if matryoshka_dim is not None:
        if matryoshka_dim <= 0:
            raise ValueError(f"matryoshka_dim must be positive, got {matryoshka_dim}")
        if arr.shape[1] < matryoshka_dim:
            raise ValueError(
                f"Cannot truncate embedding dim {arr.shape[1]} to matryoshka_dim={matryoshka_dim}"
            )
        arr = arr[:, :matryoshka_dim]
    return l2_normalize(arr)


def _prepare_image_png_bytes(image_path: Path, max_dimension: int = 1024) -> bytes | None:
    """Open image via Pillow, convert to RGB, resize max side <= max_dimension, encode PNG."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        logger.warning("Image path missing or not a file: %s", path)
        return None

    try:
        with Image.open(path) as img:
            img.load()
            transposed = ImageOps.exif_transpose(img)
            if transposed.mode in ("RGBA", "LA") or (
                transposed.mode == "P" and "transparency" in transposed.info
            ):
                rgba = transposed.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                rgb = background
            elif transposed.mode != "RGB":
                rgb = transposed.convert("RGB")
            else:
                rgb = transposed.copy()

            if max(rgb.size) > max_dimension:
                rgb.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            rgb.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        logger.warning("Skipping corrupt or unreadable image %s: %s", path, exc)
        return None


def _build_vision_prompt(context_hint: str = "") -> str:
    hint = context_hint.strip()
    if not hint:
        return _DEFAULT_VISION_PROMPT
    return f"Context hint: {hint}\n\n{_DEFAULT_VISION_PROMPT}"


def _clean_caption_output(raw: Any) -> str:
    """Normalize dict or string caption payloads into a clean single string."""
    if isinstance(raw, dict):
        parts = [f"{k}: {v}" for k, v in raw.items() if v]
        text = "\n".join(parts)
    else:
        text = str(raw or "")
    text = _THINK_TAG_RE.sub("", text)
    return text.strip()


def _validate_external_access(
    cfg: AppConfig | ExternalProviderConfig,
    default_api_key_env: str,
    allow_fallback_google_env: bool = False,
) -> tuple[ExternalProviderConfig, str]:
    """Enforce explicit user consent and API key presence before external calls."""
    ext = cfg.external_provider if isinstance(cfg, AppConfig) else cfg
    if not (ext.enabled and ext.user_consented):
        raise ConsentRequiredError(
            "External provider access denied: both external_provider.enabled and "
            "external_provider.user_consented must be True."
        )

    env_name = ext.api_key_env.strip() or default_api_key_env
    api_key = os.environ.get(env_name, "").strip()
    if not api_key and allow_fallback_google_env and not ext.api_key_env.strip():
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"External provider enabled and consented, but required API key environment "
            f"variable '{env_name}' is unset or empty."
        )
    return ext, api_key


class OllamaEmbeddingProvider:
    """Local Ollama embedding provider using /api/embed with /api/embeddings fallback."""

    def __init__(
        self,
        config: LocalProviderConfig | AppConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        embedding_dim: int | None = None,
        matryoshka_dim: int | None = None,
        timeout_seconds: float | None = None,
        batch_size: int = 32,
        keep_alive: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if isinstance(config, AppConfig):
            local_cfg = config.local_provider
        elif isinstance(config, LocalProviderConfig):
            local_cfg = config
        else:
            local_cfg = LocalProviderConfig()

        self.base_url = (base_url or local_cfg.ollama_base_url).rstrip("/")
        self.model_name = model or local_cfg.embedding_model
        self.embedding_dim = embedding_dim if embedding_dim is not None else local_cfg.embedding_dim
        if matryoshka_dim is not None:
            self.matryoshka_dim = matryoshka_dim
        elif config is None and embedding_dim is not None:
            self.matryoshka_dim = None
        else:
            self.matryoshka_dim = local_cfg.matryoshka_dim
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else local_cfg.timeout_seconds
        )
        self.batch_size = max(1, batch_size)
        self.keep_alive = keep_alive if keep_alive is not None else local_cfg.keep_alive
        self._client = client

    @property
    def dim(self) -> int:
        return self.matryoshka_dim or self.embedding_dim

    def _post_json(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}{path}"
        if self._client is not None:
            return self._client.post(url, json=payload, timeout=self.timeout_seconds)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(url, json=payload)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model_name, "input": batch}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        resp = self._post_json("/api/embed", payload)
        if resp.status_code in (404, 405, 501):
            return self._embed_legacy_batch(batch)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data:
            vectors = data["embeddings"]
        elif "embedding" in data:
            vectors = [data["embedding"]]
        else:
            raise ValueError(f"Unexpected Ollama /api/embed response keys: {list(data.keys())}")
        if len(vectors) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} embeddings from Ollama, received {len(vectors)}"
            )
        return vectors

    def _embed_legacy_batch(self, batch: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in batch:
            resp = self._post_json("/api/embeddings", {"model": self.model_name, "prompt": text})
            resp.raise_for_status()
            data = resp.json()
            if "embedding" not in data:
                raise ValueError(
                    f"Unexpected Ollama /api/embeddings response keys: {list(data.keys())}"
                )
            vectors.append(data["embedding"])
        return vectors

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings into an L2-normalized (N, D) float32 array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vectors.extend(self._embed_batch(batch))

        return _postprocess_embeddings(all_vectors, matryoshka_dim=self.matryoshka_dim)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into an L2-normalized (D,) float32 vector."""
        return self.embed_texts([query])[0]


class OllamaVisionProvider:
    """Local Ollama vision provider using Pillow preprocessing + /api/generate."""

    def __init__(
        self,
        config: LocalProviderConfig | AppConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        max_dimension: int = 1024,
        num_ctx: int = 8192,
        num_predict: int = 1536,
        timeout_seconds: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if isinstance(config, AppConfig):
            local_cfg = config.local_provider
        elif isinstance(config, LocalProviderConfig):
            local_cfg = config
        else:
            local_cfg = LocalProviderConfig()

        self.base_url = (base_url or local_cfg.ollama_base_url).rstrip("/")
        self.model_name = model or local_cfg.vision_model
        self.max_dimension = max_dimension
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else local_cfg.timeout_seconds
        )
        self._client = client

    def _post_json(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}{path}"
        if self._client is not None:
            return self._client.post(url, json=payload, timeout=self.timeout_seconds)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(url, json=payload)

    def describe_image(self, image_path: Path, context_hint: str = "") -> str:
        """Preprocess image and call Ollama /api/generate; returns '' on corrupt images."""
        png_bytes = _prepare_image_png_bytes(Path(image_path), max_dimension=self.max_dimension)
        if png_bytes is None:
            return ""

        b64_image = base64.b64encode(png_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": _build_vision_prompt(context_hint),
            "images": [b64_image],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        resp = self._post_json("/api/generate", payload)
        resp.raise_for_status()
        data = resp.json()
        raw_caption = data.get("response") or data.get("caption") or ""
        return _clean_caption_output(raw_caption)


class ExternalOpenAICompatibleProvider:
    """Opt-in external OpenAI-compatible provider (strictly consent-gated)."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        matryoshka_dim: int | None = None,
        timeout_seconds: float = 90.0,
        batch_size: int = 32,
        client: httpx.Client | None = None,
    ) -> None:
        ext, api_key = _validate_external_access(cfg, default_api_key_env="OPENAI_API_KEY")
        self._ext = ext
        self._api_key = api_key
        self.base_url = (ext.base_url.strip() or "https://api.openai.com/v1").rstrip("/")
        self.embedding_model = ext.embedding_model.strip() or "text-embedding-3-small"
        self.vision_model = ext.vision_model.strip() or "gpt-4o-mini"
        self.model_name = self.embedding_model
        self.matryoshka_dim = matryoshka_dim
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)
        self._client = client

    def _post_json(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._client is not None:
            return self._client.post(
                url, headers=headers, json=payload, timeout=self.timeout_seconds
            )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(url, headers=headers, json=payload)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self.matryoshka_dim or 1536
            return np.empty((0, dim), dtype=np.float32)

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._post_json(
                "/embeddings", {"model": self.embedding_model, "input": batch}
            )
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            all_vectors.extend(item["embedding"] for item in items)

        return _postprocess_embeddings(all_vectors, matryoshka_dim=self.matryoshka_dim)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def describe_image(self, image_path: Path, context_hint: str = "") -> str:
        png_bytes = _prepare_image_png_bytes(Path(image_path), max_dimension=1024)
        if png_bytes is None:
            return ""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_vision_prompt(context_hint)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        resp = self._post_json("/chat/completions", payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _clean_caption_output(content)


class ExternalGeminiProvider:
    """Opt-in external Google Gemini API provider (strictly consent-gated)."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        matryoshka_dim: int | None = None,
        timeout_seconds: float = 90.0,
        batch_size: int = 32,
        client: httpx.Client | None = None,
    ) -> None:
        ext, api_key = _validate_external_access(
            cfg,
            default_api_key_env="GEMINI_API_KEY",
            allow_fallback_google_env=True,
        )
        self._ext = ext
        self._api_key = api_key
        self.base_url = (
            ext.base_url.strip() or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.embedding_model = ext.embedding_model.strip() or "text-embedding-004"
        self.vision_model = ext.vision_model.strip() or "gemini-2.5-flash"
        self.model_name = self.embedding_model
        self.matryoshka_dim = matryoshka_dim
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)
        self._client = client

    def _post_json(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = {"x-goog-api-key": self._api_key}
        if self._client is not None:
            return self._client.post(
                url, headers=headers, json=payload, timeout=self.timeout_seconds
            )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(url, headers=headers, json=payload)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self.matryoshka_dim or 768
            return np.empty((0, dim), dtype=np.float32)

        model_id = self.embedding_model.removeprefix("models/")
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{model_id}",
                        "content": {"parts": [{"text": text}]},
                    }
                    for text in batch
                ]
            }
            resp = self._post_json(f"/models/{model_id}:batchEmbedContents", payload)
            resp.raise_for_status()
            data = resp.json()
            all_vectors.extend(item["values"] for item in data["embeddings"])

        return _postprocess_embeddings(all_vectors, matryoshka_dim=self.matryoshka_dim)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def describe_image(self, image_path: Path, context_hint: str = "") -> str:
        png_bytes = _prepare_image_png_bytes(Path(image_path), max_dimension=1024)
        if png_bytes is None:
            return ""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        model_id = self.vision_model.removeprefix("models/")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": _build_vision_prompt(context_hint)},
                        {"inline_data": {"mime_type": "image/png", "data": b64}},
                    ]
                }
            ]
        }
        resp = self._post_json(f"/models/{model_id}:generateContent", payload)
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
        return _clean_caption_output(text)


def _require_external_consent(cfg: AppConfig) -> ExternalProviderConfig:
    ext = cfg.external_provider
    if not (ext.enabled and ext.user_consented):
        raise ConsentRequiredError(
            "External provider access denied: both external_provider.enabled and "
            "external_provider.user_consented must be True."
        )
    return ext


def build_embedding_provider(cfg: AppConfig, use_external: bool = False) -> EmbeddingProvider:
    """Build the configured local (default) or consent-gated external embedding provider."""
    if not use_external:
        return OllamaEmbeddingProvider(cfg.local_provider)

    ext = _require_external_consent(cfg)
    if ext.provider_type == "openai_compatible":
        return ExternalOpenAICompatibleProvider(
            cfg, matryoshka_dim=cfg.local_provider.matryoshka_dim
        )
    if ext.provider_type == "gemini":
        return ExternalGeminiProvider(
            cfg, matryoshka_dim=cfg.local_provider.matryoshka_dim
        )
    if ext.provider_type == "remote_ollama":
        return OllamaEmbeddingProvider(
            cfg.local_provider,
            base_url=ext.base_url.strip() or cfg.local_provider.ollama_base_url,
            model=ext.embedding_model.strip() or cfg.local_provider.embedding_model,
        )
    raise ValueError(f"Unsupported external provider_type: {ext.provider_type}")


def build_vision_provider(cfg: AppConfig, use_external: bool = False) -> VisionProvider:
    """Build the configured local (default) or consent-gated external vision provider."""
    if not use_external:
        return OllamaVisionProvider(cfg.local_provider)

    ext = _require_external_consent(cfg)
    if ext.provider_type == "openai_compatible":
        return ExternalOpenAICompatibleProvider(cfg)
    if ext.provider_type == "gemini":
        return ExternalGeminiProvider(cfg)
    if ext.provider_type == "remote_ollama":
        return OllamaVisionProvider(
            cfg.local_provider,
            base_url=ext.base_url.strip() or cfg.local_provider.ollama_base_url,
            model=ext.vision_model.strip() or cfg.local_provider.vision_model,
        )
    raise ValueError(f"Unsupported external provider_type: {ext.provider_type}")


DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
_RANKER_CACHE: dict[tuple[str, str], Any] = {}
_RANKER_CACHE_LOCK = threading.Lock()


class _SimpleRerankRequest:
    def __init__(self, query: str, passages: list[dict[str, Any]]) -> None:
        self.query = query
        self.passages = passages


class FlashRankRerankProvider:
    """Local ONNX cross-encoder reranker backed by flashrank with cached model instances."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        cache_dir: Path | str | None = None,
        max_length: int = 512,
        ranker: Any | None = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_RERANK_MODEL
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else Path("~/.local/share/chittam/models/flashrank").expanduser()
        )
        self.max_length = int(max_length)
        self._ranker = ranker

    def _get_ranker(self) -> Any:
        if self._ranker is not None:
            return self._ranker
        key = (self.model_name, str(self.cache_dir))
        with _RANKER_CACHE_LOCK:
            cached = _RANKER_CACHE.get(key)
            if cached is not None:
                self._ranker = cached
                return cached
            from flashrank import Ranker

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            instance = Ranker(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                max_length=self.max_length,
            )
            _RANKER_CACHE[key] = instance
            self._ranker = instance
            return instance

    def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Score query against candidate passages and return [(idx, score), ...] descending."""
        if not documents:
            return []
        passages = [{"id": idx, "text": doc} for idx, doc in enumerate(documents)]
        try:
            from flashrank import RerankRequest

            req = RerankRequest(query=query, passages=passages)
        except ImportError:
            req = _SimpleRerankRequest(query=query, passages=passages)

        ranker = self._get_ranker()
        raw_results = ranker.rerank(req)
        scored: list[tuple[int, float]] = [
            (int(item["id"]), float(item.get("score", 0.0))) for item in raw_results
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if top_k is not None and top_k > 0:
            return scored[:top_k]
        return scored


def build_rerank_provider(cfg: AppConfig) -> RerankProvider | None:
    """Return configured local cross-encoder reranker if enabled & installed, else None."""
    model_name = (cfg.local_provider.rerank_model or "").strip()
    if not model_name:
        return None
    cache_dir = Path(cfg.data_dir).expanduser() / "models" / "flashrank"
    try:
        import flashrank  # noqa: F401
    except ImportError:
        logger.warning(
            "rerank_model=%s configured, but 'flashrank' is not installed (install chittam[rerank])",
            model_name,
        )
        return None
    return FlashRankRerankProvider(model_name=model_name, cache_dir=cache_dir)


def build_query_embedding_provider(
    cfg: AppConfig,
    index_meta: dict[str, str] | None = None,
    *,
    provider_builder: Callable[[AppConfig, bool], EmbeddingProvider] | None = None,
) -> EmbeddingProvider:
    """Select an EmbeddingProvider whose source & dimension match the stored vector index."""
    meta = index_meta or {}
    source = meta.get("active_embedding_source", "local")
    dim_raw = meta.get("active_embedding_dim")
    eff_cfg = cfg.model_copy(deep=True)

    if dim_raw and str(dim_raw).isdigit() and int(dim_raw) > 0:
        eff_cfg.local_provider.matryoshka_dim = int(dim_raw)

    use_external = (
        source == "external"
        and cfg.external_provider.enabled
        and cfg.external_provider.user_consented
    )
    builder = provider_builder or build_embedding_provider
    return builder(eff_cfg, use_external=use_external)

