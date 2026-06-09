"""
transcript_cursor.py — Position-anchored ground truth delivery.

Replaces Jaccard keyword matching with a stateful cursor that walks
the static transcript in lockstep with the live conversation.

Matching strategy (in order):
  1. Exact position: return transcript[cursor].human if
     keyword_overlap(live_ai, transcript[cursor].ai) >= CONFIRM_THRESHOLD
  2. Look-ahead: scan up to LOOKAHEAD turns forward for a match;
     advance cursor to that position (handles agent rephrasing)
  3. Look-back: check the previous turn in case of agent retry/re-ask
     (cursor stays, returns same ground truth again)
  4. Fallback to slot_ground_truth (for truly off-script turns)

This is fundamentally more robust than scanning the entire transcript
because it eliminates false positive matches from similar phrases that
appear multiple times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

BASE_PATH = Path(__file__).parent / "static_transcripts"

CONFIRM_THRESHOLD = 0.25  # min overlap to accept cursor position
LOOKAHEAD = 4  # how many turns ahead to scan
LOOKBACK = 1  # how many turns back to check


@dataclass
class TranscriptTurn:
    ai: str
    human: str


@dataclass
class TranscriptCursor:
    """
    Stateful cursor over a loaded transcript.
    One instance per conversation run — store in runner state.
    """

    scenario_tag: str
    turns: List[TranscriptTurn] = field(default_factory=list)
    position: int = 0  # current expected turn index
    exhausted: bool = False  # True after last turn consumed

    @classmethod
    def load(cls, scenario_tag: str) -> "TranscriptCursor":
        filename = _SCENARIO_FILE_MAP.get(scenario_tag)
        if not filename:
            return cls(scenario_tag=scenario_tag)
        path = BASE_PATH / filename
        if not path.exists():
            return cls(scenario_tag=scenario_tag)
        turns = _parse_transcript(path)
        return cls(scenario_tag=scenario_tag, turns=turns)

    def get_ground_truth(self, ai_message: str) -> Optional[str]:
        """
        Return ground truth for this AI message, advancing cursor on match.
        Returns None if no transcript match found (caller falls back to slot map).
        """
        if not self.turns or self.exhausted:
            return None

        # 1. Try current position
        if self.position < len(self.turns):
            current = self.turns[self.position]
            if _overlap(ai_message, current.ai) >= CONFIRM_THRESHOLD:
                gt = current.human
                self._advance()
                return gt

        # 2. Look-ahead: agent may have rephrased or skipped a beat
        for offset in range(1, LOOKAHEAD + 1):
            idx = self.position + offset
            if idx >= len(self.turns):
                break
            if _overlap(ai_message, self.turns[idx].ai) >= CONFIRM_THRESHOLD:
                # Jump cursor forward, return matched turn's human
                self.position = idx
                gt = self.turns[self.position].human
                self._advance()
                return gt

        # 3. Look-back: agent re-asked (retry/clarification), cursor already passed
        for offset in range(1, LOOKBACK + 1):
            idx = self.position - offset
            if idx < 0:
                break
            if _overlap(ai_message, self.turns[idx].ai) >= CONFIRM_THRESHOLD:
                # Don't advance — same position gives same ground truth on retry
                return self.turns[idx].human

        return None  # no match — caller uses slot fallback

    def _advance(self):
        self.position += 1
        if self.position >= len(self.turns):
            self.exhausted = True

    def reset(self):
        """Reset for verification restart."""
        self.position = 0
        self.exhausted = False


def _overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _parse_transcript(path: Path) -> List[TranscriptTurn]:
    turns = []
    ai_msg: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if re.match(r"(?i)^ai\s*:", line):
            ai_msg = re.sub(r"(?i)^ai\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^(human|user)\s*:", line) and ai_msg is not None:
            human = re.sub(r"(?i)^(human|user)\s*:\s*", "", line).strip()
            turns.append(TranscriptTurn(ai=ai_msg, human=human))
            ai_msg = None
    return turns


# Keep in sync with ground_truth_builder.py SCENARIO_FILE_MAP
_SCENARIO_FILE_MAP: Dict[str, str] = {
    "pcp_happy_path": "pcp_happy_path.txt",
    "pcp_clarification_zip": "pcp_clarification_zip.txt",
    "pcp_correction_first_name": "pcp_correction_first_name.txt",
    "pcp_correction_member_id": "pcp_correction_member_id.txt",
    "pcp_clarification_fax": "pcp_clarification_fax.txt",
    "claim_adjustment_happy_path": "claim_adjustment_happy_path.txt",
    "claim_adjustment_no_proceed": "claim_adjustment_no_proceed.txt",
    "claim_adjustment_upload_only": "claim_adjustment_upload_only.txt",
    "claim_adjustment_guide_only": "claim_adjustment_guide_only.txt",
}
