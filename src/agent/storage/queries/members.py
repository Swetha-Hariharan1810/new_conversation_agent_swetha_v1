"""_members.py — async member queries."""

from typing import Dict, Optional

from agent.logger import get_logger
from agent.slots.normalizers import normalize_name as _normalize_name_slot
from agent.storage.db import query_store

logger = get_logger(__name__)


def normalize_member_id(member_id: Optional[str]) -> Optional[str]:
    return member_id.strip().upper() if member_id else None


def normalize_name(value: str) -> str:
    return _normalize_name_slot(value)


async def find_member_by_identity(
    *, member_id: str, first_name: str, last_name: str, dob: str
) -> Optional[Dict]:
    where = {
        "first_name": normalize_name(first_name),
        "last_name": normalize_name(last_name),
        "dob": dob_to_db_format(dob),
    }
    if member_id:
        where["member_id"] = normalize_member_id(member_id)
    return await query_store("find_one", entity="members", where=where)


async def get_member_contact(member_id: str) -> Optional[Dict]:
    return await query_store(
        "find_one", entity="members", where={"member_id": normalize_member_id(member_id)}
    )


async def update_member_contact(
    member_id: str,
    *,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    fax: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> bool:
    payload = {
        k: v
        for k, v in {"phone_number": phone, "email": email, "fax": fax, "zip_code": zip_code}.items()
        if v is not None
    }
    if not payload:
        return False
    updated = await query_store(
        "update_one", entity="members", where={"member_id": normalize_member_id(member_id)}, update=payload
    )
    return bool(updated)


def dob_to_db_format(dob: str) -> str:
    """Convert MM/DD/YYYY (normalizer output) to YYYY-MM-DD (DB format)."""
    if not dob:
        return ""
    if len(dob) == 10 and dob[4] == "-":
        return dob  # already YYYY-MM-DD
    try:
        # Lazy import: avoids circular dependency
        from datetime import datetime

        return datetime.strptime(dob, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return dob
