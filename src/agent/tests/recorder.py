"""
recorder.py — Response recorder for offline analysis of agent turn data.

Writes nothing to disk during normal pytest runs.
Set RECORD_RESPONSES=1 to generate an HTML report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class ResponseRecord:
    test_name: str
    turn: int
    scenario: str
    user_input: str
    ai_response: str
    awaiting_slot: str
    event_type: str
    attempt_count: int
    ambiguous_count: int
    guard_fired: str
    outcome: str
    corrections_applied: dict = field(default_factory=dict)
    next_awaiting: str = ""


class ResponseRecorder:
    def __init__(self) -> None:
        self.records: list[ResponseRecord] = []

    def record(
        self,
        test_name: str,
        turn: int,
        scenario: str,
        user_input: str,
        state_before: dict,
        result: dict,
        decision=None,
    ) -> ResponseRecord:
        awaiting = state_before.get("awaiting_slot", "")
        slot_attempts = result.get("slot_attempts") or state_before.get("slot_attempts") or {}
        ambiguous_counts = result.get("ambiguous_counts") or {}

        msg = result.get("messages", {})
        if isinstance(msg, dict):
            ai_response = msg.get("content", "")
        elif isinstance(msg, list) and msg:
            last = msg[-1]
            ai_response = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            ai_response = ""

        slot_state = slot_attempts.get(awaiting, {})
        attempt_count = slot_state.get("attempt_count", 0) if isinstance(slot_state, dict) else 0
        ambiguous_count = ambiguous_counts.get(awaiting, 0)

        guard_fired = "NONE"
        if decision is not None:
            g = getattr(decision, "guard", None)
            guard_fired = g.value if g is not None else "NONE"

        ev = ""
        if decision is not None:
            et = getattr(decision, "event_type", None)
            ev = et.value if et is not None else ""

        next_node = result.get("next_node", "")
        is_interrupt = result.get("is_interrupt", False)
        if next_node == "escalation_agent" and not is_interrupt:
            outcome = "escalate"
        elif result.get("member_status_verify") and not is_interrupt and next_node != "escalation_agent":
            outcome = "complete"
        elif is_interrupt and ("updated" in ai_response.lower() or "i've" in ai_response.lower()):
            outcome = "correction_ack"
        elif is_interrupt and "what would you like" in ai_response.lower():
            outcome = "clarify"
        else:
            outcome = "ask"

        corrections_applied = {}
        if decision is not None:
            corrections_applied = dict(getattr(decision, "corrections", None) or {})

        rec = ResponseRecord(
            test_name=test_name,
            turn=turn,
            scenario=scenario,
            user_input=user_input,
            ai_response=ai_response,
            awaiting_slot=awaiting,
            event_type=ev,
            attempt_count=attempt_count,
            ambiguous_count=ambiguous_count,
            guard_fired=guard_fired,
            outcome=outcome,
            corrections_applied=corrections_applied,
            next_awaiting=result.get("awaiting_slot", ""),
        )
        self.records.append(rec)
        return rec

    def to_html(self) -> str:
        by_test: dict[str, list[ResponseRecord]] = {}
        for r in self.records:
            by_test.setdefault(r.test_name, []).append(r)

        COLORS = {
            "escalate": "#ffcccc",
            "complete": "#ccffcc",
            "correction_ack": "#cce5ff",
            "clarify": "#fff3cd",
            "ask": "#f8f9fa",
        }

        summary_rows = []
        for test_name, recs in by_test.items():
            last = recs[-1]
            bg = COLORS.get(last.outcome, "#fff")
            pass_icon = "✓" if last.outcome in ("complete", "ask") else "✗"
            summary_rows.append(
                f'<tr style="background:{bg}"><td>{test_name}</td>'
                f"<td>{len(recs)}</td><td>{last.outcome}</td><td>{pass_icon}</td></tr>"
            )

        detail_sections = []
        for test_name, recs in by_test.items():
            rows = []
            for r in recs:
                bg = COLORS.get(r.outcome, "#fff")
                attempt_style = ' style="background:#fff3cd"' if r.attempt_count >= 2 else ""
                ai_short = r.ai_response[:80] + ("…" if len(r.ai_response) > 80 else "")
                rows.append(
                    f'<tr style="background:{bg}">'
                    f"<td>{r.turn}</td><td>{r.scenario}</td><td>{r.user_input}</td>"
                    f'<td title="{r.ai_response}">{ai_short}</td>'
                    f"<td>{r.awaiting_slot}</td><td>{r.event_type}</td>"
                    f"<td{attempt_style}>{r.attempt_count}</td><td>{r.outcome}</td></tr>"
                )
            detail_sections.append(
                f"<details><summary><b>{test_name}</b></summary>"
                f"<table border='1' cellpadding='4' cellspacing='0'>"
                f"<tr><th>Turn</th><th>Scenario</th><th>User</th><th>AI Response</th>"
                f"<th>Awaiting</th><th>Event</th><th>Attempt</th><th>Outcome</th></tr>"
                + "\n".join(rows)
                + "</table></details>"
            )

        records_json = json.dumps(
            [
                {
                    "test_name": r.test_name,
                    "turn": r.turn,
                    "scenario": r.scenario,
                    "user_input": r.user_input,
                    "ai_response": r.ai_response,
                    "awaiting_slot": r.awaiting_slot,
                    "event_type": r.event_type,
                    "attempt_count": r.attempt_count,
                    "ambiguous_count": r.ambiguous_count,
                    "guard_fired": r.guard_fired,
                    "outcome": r.outcome,
                    "corrections_applied": r.corrections_applied,
                    "next_awaiting": r.next_awaiting,
                }
                for r in self.records
            ],
            indent=2,
        )

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Agent Test Report</title>
<style>
  body{{font-family:monospace;padding:1em}}
  table{{border-collapse:collapse;width:100%;margin-bottom:1em}}
  th{{background:#333;color:#fff;padding:4px 8px}}
  td{{padding:4px 8px;vertical-align:top;word-break:break-word;max-width:300px}}
  details{{margin-bottom:.5em}} summary{{cursor:pointer;padding:4px;background:#eee}}
  #btn{{padding:6px 12px;margin-bottom:1em;cursor:pointer}}
</style></head><body>
<h1>Agent Test Report</h1>
<button id="btn" onclick="exportJSON()">Export JSON</button>
<h2>Summary</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Test</th><th>Turns</th><th>Final Outcome</th><th>Pass</th></tr>
{"".join(summary_rows)}
</table>
<h2>Turn Details</h2>
{"".join(detail_sections)}
<script>
const records={records_json};
function exportJSON(){{
  const b=new Blob([JSON.stringify(records,null,2)],{{type:'application/json'}});
  const u=URL.createObjectURL(b);const a=document.createElement('a');
  a.href=u;a.download='test_records.json';a.click();URL.revokeObjectURL(u);
}}
</script></body></html>"""

    def save(self, path: str = "test_responses.html") -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())


_recorder = ResponseRecorder()


def get_recorder() -> ResponseRecorder:
    return _recorder


def maybe_save(path: str = "test_responses.html") -> None:
    if os.getenv("RECORD_RESPONSES"):
        _recorder.save(path)
