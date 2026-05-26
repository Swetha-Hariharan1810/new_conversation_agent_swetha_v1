"""
conftest.py — Shared pytest configuration for all agent tests.

Responsibilities:
  - Register custom markers
  - Auto-record every test result into the HTML reporter (when RECORD_RESPONSES=1)
  - Tear down the global Salesforce client cleanly so Python 3.13 does not
    emit "RuntimeError: Event loop is closed" during GC after the test session
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    for mark, desc in [
        ("happy", "happy-path tests"),
        ("unhappy", "non-happy-path tests"),
        ("latency", "latency measurement tests"),
        ("stress", "concurrent stress tests"),
        ("slot_retry", "max slot attempt escalation tests"),
        ("corrections", "slot correction tests"),
        ("multi_slot", "multiple slots extracted in one turn"),
        ("lookup", "Salesforce lookup retry and escalation tests"),
        ("response_check", "response content generation tests"),
        ("ambiguous", "ambiguous event counter tests"),
        ("correction_retry", "correction + retry interaction tests"),
        ("post_lookup", "post-lookup slot collection tests"),
        ("guards", "conversation guard trigger tests"),
        ("bonus_extraction", "bonus slot pre-extraction tests"),
        ("regression", "bug regression tests"),
        ("latency_correction", "latency tests for correction and clarification turns"),
        ("zip_retry", "ZIP confirmation retry tests"),
        ("sf_fail", "Salesforce failure escalation tests"),
        ("retry", "agent retry exhaustion tests"),
        ("dispatch_fail", "provider list dispatch failure tests"),
        ("live", "live integration tests requiring credentials"),
    ]:
        config.addinivalue_line("markers", f"{mark}: {desc}")


# ---------------------------------------------------------------------------
# Auto-recorder: capture every test's outcome for the HTML report
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _auto_record(request):
    """
    After each test, push a minimal record to the ResponseRecorder so the
    HTML report shows all tests (pass AND fail), not just ones that called
    rec.record() explicitly.
    """
    yield
    try:
        from agent.tests.recorder import ResponseRecord, get_recorder

        rec = get_recorder()
        test_name = request.node.nodeid
        if any(r.test_name == test_name for r in rec.records):
            return  # already has detail records from explicit rec.record() calls
        outcome = "pass"
        if hasattr(request.node, "rep_call"):
            rep = request.node.rep_call
            if rep.failed:
                outcome = "fail"
            elif rep.skipped:
                outcome = "skip"
        rec.records.append(
            ResponseRecord(
                test_name=test_name,
                turn=0,
                scenario="summary",
                user_input="",
                ai_response="",
                awaiting_slot="",
                event_type="",
                attempt_count=0,
                ambiguous_count=0,
                guard_fired="NONE",
                outcome=outcome,
            )
        )
    except Exception:
        pass  # Never let recorder errors break a test


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach call report to the node so _auto_record can read the outcome."""
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item.rep_call = rep


# ---------------------------------------------------------------------------
# Salesforce client teardown — prevents "Event loop is closed" on Python 3.13
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _teardown_gemini_client():
    """
    Close the Gemini LLM client's underlying httpx connection pool after the
    test session. Without this, Python 3.13 raises RuntimeError('Event loop
    is closed') when the GC collects open TLS connections during teardown.

    This only affects LLM 2 (generation LLM). LLM 1 (extraction) uses Azure
    OpenAI which does not hold a persistent async client at module level.
    """
    yield
    try:
        from agent.llm.config import get_generation_llm

        llm = get_generation_llm()
        # langchain_google_genai wraps the underlying google-genai client.
        # The async HTTP client may be nested several levels deep.
        # Walk the common attribute paths and close whatever is found.
        for attr_path in [
            ("client", "_async_httpx_client"),
            ("client", "aio", "_client"),
            ("_async_client",),
        ]:
            obj = llm
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "aclose"):
                import asyncio

                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(obj.aclose())
                except Exception:
                    pass
                finally:
                    loop.close()
                break
    except Exception:
        pass  # Never let teardown errors break the test session report


@pytest.fixture(autouse=True, scope="session")
def _teardown_sf_client():
    """
    Close the global SalesforceClient after the entire test session.

    Without this, Python's GC collects the httpx.AsyncClient after the event
    loop is gone and emits RuntimeError('Event loop is closed') during cleanup.
    """
    yield
    try:
        import asyncio

        from agent.storage.db import _SF_CLIENT, reset_salesforce_client

        if _SF_CLIENT is not None:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.run_until_complete(_SF_CLIENT.aclose())
            except Exception:
                pass
            finally:
                reset_salesforce_client()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTML report: save once at session end
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Save the HTML response report if RECORD_RESPONSES=1."""
    try:
        from agent.tests.recorder import maybe_save

        maybe_save("test_responses.html")
    except Exception:
        pass
