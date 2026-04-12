from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    WAIT = "WAIT"
    ALERT_REFILL = "ALERT_REFILL"
    EMERGENCY_ALERT = "EMERGENCY_ALERT"
    NOTIFY_CAREGIVER = "NOTIFY_CAREGIVER"


class Observation(BaseModel):
    oxygen_level: float = Field(..., ge=0.0, le=100.0, description="Cylinder fill level (%)")
    consumption_rate: float = Field(..., gt=0.0, description="O2 consumed per step")
    predicted_depletion_time: float = Field(..., description="Steps until O2 hits zero")
    spo2: float = Field(..., ge=70.0, le=100.0, description="Blood oxygen saturation (%)")
    heart_rate: float = Field(..., ge=30.0, le=200.0, description="Heart rate (bpm)")
    respiratory_rate: float = Field(..., ge=5.0, le=60.0, description="Breaths per minute")
    spo2_trend: int = Field(..., ge=-1, le=1, description="SpO2 direction: -1 falling, 0 stable, +1 rising")
    time_elapsed: int = Field(..., ge=0, description="Steps elapsed in episode")
    risk_level: str = Field(..., description="Derived risk: low | moderate | high | critical")
    refill_arriving_in: int = Field(default=0, ge=0, description="Steps until refill arrives (0 = not ordered)")
    patient_type: str = Field(default="unknown", description="Patient clinical profile")

    class Config:
        use_enum_values = True


class Action(BaseModel):
    action: ActionType

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    value: float = Field(..., description="Scalar reward for this step")
    reason: str = Field(default="", description="Human-readable explanation")


class StepRecord(BaseModel):
    step: int
    observation: Observation
    action: str
    reward: float
    reason: str
    done: bool
    risk: str


class EpisodeSummary(BaseModel):
    task: str
    total_steps: int
    score: float = Field(..., ge=0.0, le=1.0)
    total_reward: float
    false_alarms: int
    critical_misses: int
    oxygen_depletions: int
    history: Optional[List[StepRecord]] = None
