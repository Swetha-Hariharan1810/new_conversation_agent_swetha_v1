"""providers.py — async provider list dispatch queries."""

from agent.storage.queries.communication import set_provider_request_delivery
from agent.storage.queries.members import normalize_member_id


async def send_provider_list(
    member_id: str,
    provider_type: str,
    zip_code: str,
    delivery_method: str,
    delivery_address: str,
) -> bool:
    """
    Record a provider list dispatch request in Salesforce.
    Returns True on success.
    """
    try:
        await set_provider_request_delivery(
            normalize_member_id(member_id),
            provider_type=provider_type,
            method=delivery_method,
            destination=delivery_address,
            update_status="sent",
        )
        return True
    except Exception:
        return False
