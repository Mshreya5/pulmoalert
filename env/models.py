from enum import Enum
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    WAIT = "WAIT"
    ALERT_REFILL = "ALERT_REFILL"
    EMERGENCY_ALERT = "EMERGENCY_ALERT"
    NOTIFY_CAREGIVER = "NOTIFY_CAREGIVER"


class Observation(BaseModel):
    oxygen_level: float = Field(..., ge=0.0, le=100.0, description="Current oxygen cylinder level (%)")
    spo2: float = Field(..., ge=0.0, le=100.0, description="Patient SpO2 level (%)")
    heart_rate: float = Field(..., ge=0.0, description="Patient heart rate (bpm)")
    consumption_rate: float = Field(..., ge=0.0, description="Current oxygen consumption rate (%/step)")
    time_step: int = Field(..., ge=0, description="Elapsed simulation steps")
    risk_level: str = Field(..., description="risk state: low/moderate/high/critical")


class Action(BaseModel):
    action: ActionType


class Reward(BaseModel):
    value: float
    reason: str
