"""
backend/api/emergency_router.py — Emergency contacts endpoint
"""
from fastapi import APIRouter
from backend.config import settings

router = APIRouter(prefix="/api/emergency", tags=["Emergency"])


@router.get("/contacts")
async def get_emergency_contacts():
    """Get emergency contact numbers."""
    return {
        "emergency_number": "112",
        "police": "110",
        "doctor_on_call": "116 117",
        "poison_control_nrw": "0228 19240",
        "mental_health_crisis": "0800 111 0 111",
        "youth_crisis": "0800 111 0 333",
        "oberhausen": {
            "evangelisches_krankenhaus": "+49 208 881-0",
            "st_marien_hospital": "+49 208 8999-0",
            "gesundheitsamt": "+49 208 825-3620",
            "sozialamt": "+49 208 825-0",
        },
        "note": "For life-threatening emergencies, always call 112 first.",
    }
