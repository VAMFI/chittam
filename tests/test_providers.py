"""Unit tests for Chittam embedding, vision, and consent-gated external providers."""

from __future__ import annotations

import base64
import inspect
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from chittam.models import AppConfig
from chittam.providers import (
    ConsentRequiredError,
    EmbeddingProvider,
    ExternalGeminiProvider,
    ExternalOpenAICompatibleProvider,
    OllamaEmbeddingProvider,
    OllamaVisionProvider,
    RerankProvider,
    VisionProvider,
    build_embedding_provider,
    build_vision_provider,
    l2_normalize,
)


def _make_png_file(path: Path, size: tuple[int, int], mode: str = "RGB") -> Path:
    color = (12, 120, 220, 180) if mode == "RGBA" else (12, 120, 220)
    img = Image.new(mode, size, color=color)
    img.save(path, format="PNG")
    return path


def test_factory_signatures_and_protocols() -> None:
    emb_sig = inspect.signature(build_embedding_provider)
    assert list(emb_sig.parameters.keys()) == ["cfg", "use_external"]
    assert emb_sig.parameters["use_external"].default is False

    vis_sig = inspect.signature(build_vision_provider)
    assert list(vis_sig.parameters.keys()) == ["cfg", "use_external"]
    assert vis_sig.parameters["use_external"].default is False

    assert issubclass(ConsentRequiredError, RuntimeError)
    assert hasattr(RerankProvider, "rerank")

    cfg = AppConfig()
    emb = build_embedding_provider(cfg)
    vis = build_vision_provider(cfg)
    assert isinstance(emb, EmbeddingProvider)
    assert isinstance(vis, VisionProvider)
    assert isinstance(emb, OllamaEmbeddingProvider)
    assert isinstance(vis, OllamaVisionProvider)


def test_ollama_embedding_provider_embed_and_matryoshka_renormalization() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        # Non-unit raw vectors: prefix [3, 4] has norm 5 -> after truncation to 2 dims must be [0.6, 0.8]
        vectors = [[3.0, 4.0, 100.0, 200.0] for _ in payload["input"]]
        return httpx.Response(200, json={"model": payload["model"], "embeddings": vectors})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model="embeddinggemma:latest",
        matryoshka_dim=2,
        batch_size=2,
        client=client,
    )

    mat = provider.embed_texts(["a", "b", "c"])
    assert len(seen_payloads) == 2  # batched into 2 + 1
    assert mat.dtype == np.float32
    assert mat.shape == (3, 2)
    np.testing.assert_allclose(mat, np.array([[0.6, 0.8]] * 3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), np.ones(3), atol=1e-6)

    q = provider.embed_query("search term")
    assert q.dtype == np.float32
    assert q.shape == (2,)
    np.testing.assert_allclose(q, np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)


def test_ollama_embedding_provider_fallback_to_legacy_embeddings() -> None:
    paths_called: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_called.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(404, text="404 page not found")
        if request.url.path == "/api/embeddings":
            payload = json.loads(request.content.decode("utf-8"))
            prompt = payload["prompt"]
            vec = [0.0, 3.0, 4.0] if prompt == "first" else [5.0, 12.0, 0.0]
            return httpx.Response(200, json={"embedding": vec})
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaEmbeddingProvider(client=client, embedding_dim=3)
    res = provider.embed_texts(["first", "second"])

    assert paths_called == ["/api/embed", "/api/embeddings", "/api/embeddings"]
    assert res.shape == (2, 3)
    np.testing.assert_allclose(res[0], np.array([0.0, 0.6, 0.8], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        res[1], np.array([5.0 / 13.0, 12.0 / 13.0, 0.0], dtype=np.float32), atol=1e-6
    )


def test_ollama_embedding_edge_cases() -> None:
    provider = OllamaEmbeddingProvider(matryoshka_dim=128)
    empty = provider.embed_texts([])
    assert empty.shape == (0, 128)
    assert empty.dtype == np.float32

    zeros = l2_normalize(np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    assert not np.isnan(zeros).any()
    np.testing.assert_allclose(zeros[0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(zeros[1], np.array([0.0, 0.6, 0.8], dtype=np.float32), atol=1e-6)

    def small_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})

    bad_matryoshka = OllamaEmbeddingProvider(
        matryoshka_dim=16,
        client=httpx.Client(transport=httpx.MockTransport(small_handler)),
    )
    with pytest.raises(ValueError, match="Cannot truncate"):
        bad_matryoshka.embed_query("hi")


def test_ollama_vision_provider_resizes_converts_and_cleans_caption(tmp_path: Path) -> None:
    large_rgba = _make_png_file(tmp_path / "diagram.png", size=(2048, 1024), mode="RGBA")
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "response": "<think>internal reasoning</think>\nDescription: Architecture diagram.\nKeywords: architecture, api"
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaVisionProvider(client=client)
    caption = provider.describe_image(large_rgba, context_hint="docs/arch.png")

    assert len(captured_payloads) == 1
    req = captured_payloads[0]
    assert req["model"] == "qwen3-vl:2b"
    assert req["stream"] is False
    assert "docs/arch.png" in req["prompt"]
    assert len(req["images"]) == 1

    sent_bytes = base64.b64decode(req["images"][0])
    with Image.open(io.BytesIO(sent_bytes)) as sent_img:
        assert sent_img.mode == "RGB"
        assert max(sent_img.size) <= 1024
        assert sent_img.size == (1024, 512)

    assert "internal reasoning" not in caption
    assert "Architecture diagram." in caption


def test_ollama_vision_provider_handles_dict_caption_payload(tmp_path: Path) -> None:
    img = _make_png_file(tmp_path / "ui.png", size=(400, 300))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": {
                    "description": "Login dialog",
                    "visible_text": "Sign In",
                    "keywords": "auth, form",
                }
            },
        )

    provider = OllamaVisionProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    caption = provider.describe_image(img)
    assert "Login dialog" in caption
    assert "Sign In" in caption


def test_remote_ollama_external_dispatch_respects_consent() -> None:
    cfg = AppConfig()
    cfg.external_provider.provider_type = "remote_ollama"
    cfg.external_provider.base_url = "http://10.0.0.50:11434"

    with pytest.raises(ConsentRequiredError):
        build_embedding_provider(cfg, use_external=True)

    cfg.external_provider.enabled = True
    cfg.external_provider.user_consented = True
    emb = build_embedding_provider(cfg, use_external=True)
    vis = build_vision_provider(cfg, use_external=True)
    assert isinstance(emb, OllamaEmbeddingProvider)
    assert isinstance(vis, OllamaVisionProvider)
    assert emb.base_url == "http://10.0.0.50:11434"
    assert vis.base_url == "http://10.0.0.50:11434"


def test_ollama_vision_provider_handles_corrupt_and_missing_images(tmp_path: Path) -> None:
    calls = 0

    def fail_if_called(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP should not be called for corrupt or missing images")

    client = httpx.Client(transport=httpx.MockTransport(fail_if_called))
    provider = OllamaVisionProvider(client=client)

    missing = tmp_path / "does_not_exist.png"
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"definitely not valid png image bytes")
    truncated = tmp_path / "truncated.png"
    valid = _make_png_file(tmp_path / "valid.png", size=(64, 64))
    truncated.write_bytes(valid.read_bytes()[:24])

    assert provider.describe_image(missing) == ""
    assert provider.describe_image(corrupt) == ""
    assert provider.describe_image(truncated) == ""
    assert calls == 0


@pytest.mark.parametrize(
    ("enabled", "user_consented"),
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_external_providers_enforce_consent_and_opt_in(
    enabled: bool, user_consented: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-1234567890")

    cfg = AppConfig()
    cfg.external_provider.enabled = enabled
    cfg.external_provider.user_consented = user_consented

    with pytest.raises(ConsentRequiredError):
        ExternalOpenAICompatibleProvider(cfg)
    with pytest.raises(ConsentRequiredError):
        ExternalGeminiProvider(cfg)

    cfg.external_provider.provider_type = "openai_compatible"
    with pytest.raises(ConsentRequiredError):
        build_embedding_provider(cfg, use_external=True)
    with pytest.raises(ConsentRequiredError):
        build_vision_provider(cfg, use_external=True)

    cfg.external_provider.provider_type = "gemini"
    with pytest.raises(ConsentRequiredError):
        build_embedding_provider(cfg, use_external=True)
    with pytest.raises(ConsentRequiredError):
        build_vision_provider(cfg, use_external=True)

    # Local default remains allowed even without external consent
    assert isinstance(build_embedding_provider(cfg, use_external=False), OllamaEmbeddingProvider)
    assert isinstance(build_vision_provider(cfg, use_external=False), OllamaVisionProvider)


def test_external_providers_require_api_key_and_succeed_when_consented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHITTAM_TEST_OPENAI_KEY", raising=False)
    monkeypatch.delenv("CHITTAM_TEST_GEMINI_KEY", raising=False)

    cfg = AppConfig()
    cfg.external_provider.enabled = True
    cfg.external_provider.user_consented = True
    cfg.external_provider.api_key_env = "CHITTAM_TEST_OPENAI_KEY"

    with pytest.raises(RuntimeError, match="CHITTAM_TEST_OPENAI_KEY"):
        ExternalOpenAICompatibleProvider(cfg)

    cfg.external_provider.provider_type = "gemini"
    cfg.external_provider.api_key_env = "CHITTAM_TEST_GEMINI_KEY"
    with pytest.raises(RuntimeError, match="CHITTAM_TEST_GEMINI_KEY"):
        ExternalGeminiProvider(cfg)

    # Now provide keys and verify factory dispatch + HTTP formatting
    monkeypatch.setenv("CHITTAM_TEST_OPENAI_KEY", "sk-consented")
    monkeypatch.setenv("CHITTAM_TEST_GEMINI_KEY", "gemini-consented")
    img_path = _make_png_file(tmp_path / "shot.png", size=(300, 200))

    def openai_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-consented"
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "OpenAI caption"}}]}
            )
        raise AssertionError(request.url)

    cfg.external_provider.provider_type = "openai_compatible"
    cfg.external_provider.base_url = "https://api.example.test/v1"
    cfg.external_provider.api_key_env = "CHITTAM_TEST_OPENAI_KEY"
    assert isinstance(build_embedding_provider(cfg, use_external=True), ExternalOpenAICompatibleProvider)
    assert isinstance(build_vision_provider(cfg, use_external=True), ExternalOpenAICompatibleProvider)

    oa = ExternalOpenAICompatibleProvider(
        cfg, client=httpx.Client(transport=httpx.MockTransport(openai_handler))
    )
    np.testing.assert_allclose(oa.embed_query("q"), np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)
    assert oa.describe_image(img_path) == "OpenAI caption"

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "gemini-consented"
        if request.url.path.endswith(":batchEmbedContents"):
            return httpx.Response(200, json={"embeddings": [{"values": [0.0, 5.0, 12.0]}]})
        if request.url.path.endswith(":generateContent"):
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Gemini caption"}]}}]},
            )
        raise AssertionError(request.url)

    cfg.external_provider.provider_type = "gemini"
    cfg.external_provider.base_url = "https://gemini.example.test/v1beta"
    cfg.external_provider.api_key_env = "CHITTAM_TEST_GEMINI_KEY"
    assert isinstance(build_embedding_provider(cfg, use_external=True), ExternalGeminiProvider)
    assert isinstance(build_vision_provider(cfg, use_external=True), ExternalGeminiProvider)

    gem = ExternalGeminiProvider(
        cfg, client=httpx.Client(transport=httpx.MockTransport(gemini_handler))
    )
    np.testing.assert_allclose(
        gem.embed_query("q"),
        np.array([0.0, 5.0 / 13.0, 12.0 / 13.0], dtype=np.float32),
        atol=1e-6,
    )
    assert gem.describe_image(img_path) == "Gemini caption"
