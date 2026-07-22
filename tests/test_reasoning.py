# Empromptu FreeAgent - The free, local, entirely private agent coding system, by Empromptu!
# Copyright (C) 2025  Empromptu, Sean Robinson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Reasoning-level normalization, per-provider translation, and backend wiring."""

import json
import urllib.request

import pytest

from free_agent.config import LLMConfig
from free_agent.llm import build_backend
from free_agent.llm.ollama import OllamaBackend
from free_agent.llm.reasoning import normalize, params_for


# --- normalize --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("off", "off"),
        ("OFF", "off"),
        ("None", "off"),
        ("false", "off"),
        ("0", "off"),
        ("disabled", "off"),
        ("low", "low"),
        ("High", "high"),
        ("  Medium  ", "medium"),
        ("xhigh", "xhigh"),  # unknown levels pass through for forward-compat
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


# --- params_for -------------------------------------------------------------


def test_params_for_none_is_noop_everywhere():
    for provider in ("ollama", "ollama-openai", "openai", "anthropic", "weird"):
        assert params_for(provider, None) == {}


def test_params_for_openai_style_uses_reasoning_effort():
    for provider in ("ollama-openai", "openai"):
        assert params_for(provider, "high") == {"reasoning_effort": "high"}
        assert params_for(provider, "off") == {"reasoning_effort": "none"}


def test_params_for_ollama_native_uses_think():
    assert params_for("ollama", "medium") == {"think": "medium"}
    assert params_for("ollama", "off") == {"think": False}


def test_params_for_anthropic_maps_level_to_budget():
    assert params_for("anthropic", "low") == {
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    }
    # off -> no thinking key at all (disabled)
    assert params_for("anthropic", "off") == {}
    # unknown level -> leave default rather than guess a budget
    assert params_for("anthropic", "xhigh") == {}


def test_params_for_unknown_provider_is_noop():
    assert params_for("mystery", "high") == {}


# --- backend wiring ---------------------------------------------------------


def test_build_backend_threads_reasoning_into_ollama():
    backend = build_backend(
        LLMConfig(provider="ollama", base_url="http://x", model="m", reasoning="High")
    )
    assert isinstance(backend, OllamaBackend)
    # LLMConfig stores the raw value; the proxy normalizes before constructing
    # the config, so accept whatever was passed through here.
    assert backend.reasoning == "High"


class _FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_backend_injects_think_into_request_body(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"message": {"content": "ok"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    backend = OllamaBackend(base_url="http://x", model="m", reasoning="medium")
    assert backend.complete("sys", "hi") == "ok"
    assert captured["body"]["think"] == "medium"


def test_ollama_backend_omits_think_when_unset(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"message": {"content": "ok"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    backend = OllamaBackend(base_url="http://x", model="m")  # reasoning defaults None
    backend.complete("sys", "hi")
    assert "think" not in captured["body"]
