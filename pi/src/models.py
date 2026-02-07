"""Data models for API responses"""
from pydantic import BaseModel


class PhaseDecision(BaseModel):
    """Phase decision response.

    Phase mapping:
    - phase 0 : East, West => G
    - phase 2 : North, South => G
    """
    phase: int
    duration: int
    yellow_required: bool
