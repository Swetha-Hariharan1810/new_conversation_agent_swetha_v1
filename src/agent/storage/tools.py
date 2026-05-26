"""
tools.py — Storage tools (core agents only: lookup_member, update_member_contact).
"""

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agent.logger import get_logger

logger = get_logger(__name__)


@tool
async def lookup_member(
    member_id: str,
    first_name: str = "",
    last_name: str = "",
    dob: str = "",
) -> Optional[Dict[str, Any]]:
    """Full identity verification against Salesforce member record."""
    from agent.storage.queries.members import find_member_by_identity

    try:
        record = await find_member_by_identity(
            member_id=member_id,
            first_name=first_name,
            last_name=last_name,
            dob=dob,
        )
        logger.info(
            "lookup_member: verification attempt",
            extra={"verified": bool(record), "member_tail": member_id[-4:]},
        )
        if not record:
            return {"verified": False}
        return {
            "verified": True,
            "member_id": record.get("member_id"),
            "phone_number": record.get("phone_number", ""),
            "zip_code": record.get("zip_code", ""),
            "fax": record.get("fax", ""),
            "email": record.get("email", ""),
            "relationship": record.get("relationship", ""),
            "record": record,
        }
    except Exception:
        logger.exception("lookup_member failed")
        return {"verified": False}


@tool
async def update_member_contact(
    member_id: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    fax: Optional[str] = None,
    zip_code: Optional[str] = None,
) -> bool:
    """Update contact fields for a verified member."""
    from agent.storage.queries.members import update_member_contact as _update

    try:
        return await _update(member_id, phone=phone, email=email, fax=fax, zip_code=zip_code)
    except Exception:
        logger.exception("update_member_contact failed")
        return False
