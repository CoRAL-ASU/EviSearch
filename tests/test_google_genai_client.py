from __future__ import annotations

import sys
import types

import pytest

import src.LLMProvider.google_genai_client as google_client


def _install_fake_google_genai(monkeypatch):
    client_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(HttpOptions=FakeHttpOptions)
    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = FakeClient
    fake_genai_module.types = fake_types

    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setattr(google_client, "_genai", None)
    monkeypatch.setattr(google_client, "_types", None)
    return client_calls


def test_create_vertex_client_prefers_api_key(monkeypatch):
    client_calls = _install_fake_google_genai(monkeypatch)
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    client = google_client.create_vertex_genai_client(timeout_ms=1234)

    assert client.kwargs["vertexai"] is True
    assert client.kwargs["api_key"] == "vertex-key"
    assert "project" not in client.kwargs
    assert "location" not in client.kwargs
    assert client_calls[0]["http_options"].kwargs == {"api_version": "v1", "timeout": 1234}


def test_create_vertex_client_uses_project_and_location(monkeypatch):
    _install_fake_google_genai(monkeypatch)
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "mayo-evisearch")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = google_client.create_vertex_genai_client()

    assert client.kwargs["vertexai"] is True
    assert client.kwargs["project"] == "mayo-evisearch"
    assert client.kwargs["location"] == "us-central1"
    assert client.kwargs["http_options"].kwargs == {"api_version": "v1"}


def test_create_vertex_client_requires_auth(monkeypatch):
    _install_fake_google_genai(monkeypatch)
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    with pytest.raises(ValueError, match="Vertex AI Gemini authentication is not configured"):
        google_client.create_vertex_genai_client()
