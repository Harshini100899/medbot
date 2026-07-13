"""
backend/api/doctor_router.py — Doctor & Pharmacy search endpoints
"""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter
from backend.tools.doctor_search_tool import find_doctors
from backend.db.mongodb import search_doctors, get_hospitals, get_pharmacies

router = APIRouter(prefix="/api/doctors", tags=["Doctors"])


@router.get("/search")
async def search_doctors_endpoint(
    query: Optional[str] = None,
    specialization: Optional[str] = None,
    language: Optional[str] = None,
    city: str = "Oberhausen",
    limit: int = 5,
):
    """Search for doctors by specialisation, language, and city."""
    result = await find_doctors(
        query=query or "",
        language=language,
        specialization=specialization,
        city=city,
        limit=limit,
    )
    return result


@router.get("/hospitals")
async def get_hospitals_endpoint(city: str = "Oberhausen"):
    """Get list of hospitals in the city."""
    hospitals = await get_hospitals(city=city)
    return {"hospitals": hospitals, "city": city}


@router.get("/pharmacies")
async def get_pharmacies_endpoint(city: str = "Oberhausen", night_only: bool = False):
    """Get list of pharmacies."""
    pharmacies = await get_pharmacies(city=city, night_only=night_only)
    return {"pharmacies": pharmacies, "city": city}
