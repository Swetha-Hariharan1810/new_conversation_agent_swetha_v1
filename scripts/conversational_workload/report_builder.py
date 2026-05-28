"""
Saves per-conversation evaluation reports to the results directory.
"""

import os
from datetime import datetime
from pathlib import Path

from scripts.conversational_workload.models import ConversationReport

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(
    os.getenv("RESULTS_DIR", str(REPO_ROOT / "scripts/conversational_workload/results"))
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_report(report: ConversationReport) -> Path:
    filename = f"{report.scenario_tag}_{report.conversation_id}.txt"
    path = RESULTS_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("CONVERSATION EVALUATION REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"Conversation ID: {report.conversation_id}\n")
        f.write(f"Flow: {report.flow}\n")
        f.write(f"Scenario: {report.scenario_tag}\n")
        f.write(f"Completed: {report.completed}\n")
        f.write(f"Final Score: {report.final_score}\n\n")
        f.write("-" * 40 + "\n\n")

        for i, turn in enumerate(report.turns, 1):
            scores = turn.scores
            f.write(f"--- Turn {i} ---\n")
            f.write(f"Slot: {turn.slot}\n")
            f.write(f"AI Prompt:\n{turn.ai_prompt}\n\n")
            f.write(f"User Response:\n{turn.user_response}\n\n")
            f.write(f"Ground Truth:\n{turn.ground_truth}\n\n")
            f.write("Scores:\n")
            f.write(f"  Intent:       {scores.get('intent_score')}\n")
            f.write(f"  Constraint:   {scores.get('constraint_score')}\n")
            f.write(f"  Completeness: {scores.get('completeness_score')}\n")
            f.write(f"  Naturalness:  {scores.get('naturalness_score')}\n")
            f.write(f"  Overall:      {scores.get('overall')}\n")
            f.write(f"  Verdict:      {scores.get('verdict')}\n")
            f.write("\n" + "-" * 44 + "\n\n")

        f.write("=" * 40 + "\n END OF REPORT\n" + "=" * 40 + "\n")

    return path
