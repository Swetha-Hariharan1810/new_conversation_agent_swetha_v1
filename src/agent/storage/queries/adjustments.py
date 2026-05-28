"""_adjustments.py — async claim adjustment queries."""

import datetime
from typing import Dict, Optional

from agent.storage.db import query_store
from agent.storage.queries.members import normalize_member_id


def _now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


async def find_adjustment(reference_number: str, member_id: str) -> Optional[Dict]:
    return await query_store(
        "find_one",
        entity="adjustment_requests",
        where={"reference_number": reference_number, "member_id": normalize_member_id(member_id)},
    )


async def set_claim_request_delivery(
    member_id: str,
    reference_number: str,
    *,
    run_id: Optional[str] = None,
    method: Optional[str] = None,
    destination: Optional[str] = None,
    update_status: Optional[str] = None,
) -> Dict:
    member_id = normalize_member_id(member_id)
    now = _now()
    row = await query_store(
        "find_one",
        entity="claim_request_delivery_updates",
        where={"member_id": member_id, "reference_number": reference_number, "run_id": run_id},
    )

    if not row or (method is not None and row.get("method") != method):
        return await query_store(
            "insert_one",
            entity="claim_request_delivery_updates",
            record={
                "member_id": member_id,
                "reference_number": reference_number,
                "run_id": run_id,
                "method": method,
                "destination": destination,
                "update_status": update_status,
                "updated_at": now,
            },
        )

    update = {
        k: v
        for k, v in {"destination": destination, "update_status": update_status, "updated_at": now}.items()
        if v is not None
    }
    if update:
        await query_store(
            "update_one", entity="claim_request_delivery_updates", where={"id": row["id"]}, update=update
        )
        row.update(update)
    return row


async def fetch_claim_request_delivery(
    member_id: str,
    reference_number: str,
    *,
    run_id: Optional[str] = None,
    method: Optional[str] = None,
) -> Optional[Dict]:
    return await query_store(
        "find_one",
        entity="claim_request_delivery_updates",
        where={
            "member_id": normalize_member_id(member_id),
            "reference_number": reference_number,
            "run_id": run_id,
            "method": method,
        },
    )
