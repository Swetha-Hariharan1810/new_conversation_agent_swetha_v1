"""
test_phase0_flags.py — Phase 0 feature-flag defaults + parsing.

Locks the OLD defaults (so the flags reproduce today's behavior when nothing is
set) and the lenient env parsing. A later phase that flips a default flips the
corresponding assertion here.
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.core import flags

pytestmark = pytest.mark.regression


def _clear(monkeypatch):
    for name in (
        flags.ENV_UNIFIED_VOICE,
        flags.ENV_TURNPLAN_DECODE,
        flags.ENV_MULTI_INTENT_LIVE,
        flags.ENV_STREAM_GENERATION,
        flags.ENV_PARK_ANSWERABLE,
    ):
        monkeypatch.delenv(name, raising=False)


def test_rollout_defaults(monkeypatch):
    """Phase 4 flip: the rebuild is ON by default; parking answerable side
    questions stays opt-in and streaming is unchanged."""
    _clear(monkeypatch)
    assert flags.unified_voice() is True
    assert flags.turnplan_decode() == flags.TURNPLAN_LIVE
    assert flags.multi_intent_live() is True
    assert flags.stream_generation() is False
    assert flags.park_answerable() is False


def test_snapshot_matches_defaults(monkeypatch):
    _clear(monkeypatch)
    assert flags.snapshot() == {
        "UNIFIED_VOICE": True,
        "TURNPLAN_DECODE": "live",
        "MULTI_INTENT_LIVE": True,
        "STREAM_GENERATION": False,
        "PARK_ANSWERABLE": False,
        "TURNPLAN_TIMEOUT_MS": 2000,
    }


@pytest.mark.parametrize("token", ["1", "true", "TRUE", "Yes", "on", " t "])
def test_bool_truthy(monkeypatch, token):
    monkeypatch.setenv(flags.ENV_UNIFIED_VOICE, token)
    assert flags.unified_voice() is True


@pytest.mark.parametrize("token", ["0", "false", "no", "off", ""])
def test_bool_falsy(monkeypatch, token):
    monkeypatch.setenv(flags.ENV_MULTI_INTENT_LIVE, token)
    assert flags.multi_intent_live() is False


def test_bool_unknown_clamps_to_default(monkeypatch):
    # A malformed value falls back to the default (True since the Phase 4 flip).
    monkeypatch.setenv(flags.ENV_MULTI_INTENT_LIVE, "garbage")
    assert flags.multi_intent_live() is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", "off"),
        ("shadow", "shadow"),
        ("live", "live"),
        ("SHADOW", "shadow"),
        ("  live  ", "live"),
        ("nonsense", "live"),  # unknown clamps to the (Phase 4) default
    ],
)
def test_turnplan_decode_modes(monkeypatch, value, expected):
    monkeypatch.setenv(flags.ENV_TURNPLAN_DECODE, value)
    assert flags.turnplan_decode() == expected


def test_reads_are_live(monkeypatch):
    """Getters read the environment on each call (no import-time snapshot)."""
    _clear(monkeypatch)
    assert flags.park_answerable() is False
    monkeypatch.setenv(flags.ENV_PARK_ANSWERABLE, "true")
    assert flags.park_answerable() is True
