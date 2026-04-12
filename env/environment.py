import random
from collections import deque
from typing import Tuple, Dict, Any

from env.models import Observation, Action, ActionType, Reward, StepRecord


TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "oxygen_init": (80.0, 100.0),
        "spo2_init": (96.0, 99.0),
        "hr_init": (65.0, 85.0),
        "rr_init": (12.0, 18.0),
        "consumption_init": (0.5, 1.0),
        "noise_scale": 0.3,
        "deterioration_prob": 0.02,
        "emergency_prob": 0.0,
        "refill_delay": 3,
        "patient_type": "post_surgery",
        "max_steps": 60,
    },
    "medium": {
        "oxygen_init": (55.0, 80.0),
        "spo2_init": (92.0, 97.0),
        "hr_init": (60.0, 105.0),
        "rr_init": (16.0, 24.0),
        "consumption_init": (1.2, 2.5),
        "noise_scale": 0.9,
        "deterioration_prob": 0.08,
        "emergency_prob": 0.03,
        "refill_delay": 5,
        "patient_type": "copd",
        "max_steps": 80,
    },
    "hard": {
        "oxygen_init": (30.0, 65.0),
        "spo2_init": (86.0, 94.0),
        "hr_init": (50.0, 130.0),
        "rr_init": (22.0, 35.0),
        "consumption_init": (2.8, 5.0),
        "noise_scale": 1.8,
        "deterioration_prob": 0.18,
        "emergency_prob": 0.10,
        "refill_delay": 8,
        "patient_type": "icu_critical",
        "max_steps": 100,
    },
}

_ACTION_MAP = {a.value: a for a in ActionType}


class PulmoAlertEnv:
    def __init__(self, task_name: str = "easy", seed: int = 42):
        if task_name not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task_name!r}. Choose from {list(TASK_CONFIG)}")
        self.task_name = task_name
        self.seed = seed
        self._cfg = TASK_CONFIG[task_name]
        self.rng = random.Random(seed)
        self._init_state()

    def _init_state(self) -> None:
        cfg = self._cfg
        self.oxygen_level: float = self.rng.uniform(*cfg["oxygen_init"])
        self.spo2: float = self.rng.uniform(*cfg["spo2_init"])
        self.heart_rate: float = self.rng.uniform(*cfg["hr_init"])
        self.respiratory_rate: float = self.rng.uniform(*cfg["rr_init"])
        self.consumption_rate: float = self.rng.uniform(*cfg["consumption_init"])
        self._base_consumption: float = self.consumption_rate
        self.time_elapsed: int = 0
        self.episode_over: bool = False
        self.history: list = []
        self._deterioration_timer: int = 0
        self._spo2_window: deque = deque([self.spo2] * 3, maxlen=3)
        self._refill_countdown: int = 0

    def _risk_level(self) -> str:
        if self.oxygen_level <= 10 or self.spo2 <= 87:
            return "critical"
        if self.oxygen_level <= 25 or self.spo2 <= 91:
            return "high"
        if self.oxygen_level <= 40 or self.spo2 <= 94:
            return "moderate"
        return "low"

    def _spo2_trend(self) -> int:
        delta = self._spo2_window[-1] - self._spo2_window[0]
        if delta < -0.5:
            return -1
        if delta > 0.5:
            return 1
        return 0

    def _apply_physiology(self) -> None:
        noise = self._cfg["noise_scale"]

        actual_consumption = max(0.1, self.consumption_rate + self.rng.gauss(0, noise * 0.1))
        self.oxygen_level = max(0.0, self.oxygen_level - actual_consumption)

        spo2_drift = -actual_consumption * 0.06
        self.spo2 = max(70.0, min(100.0, self.spo2 + spo2_drift + self.rng.gauss(0, noise * 0.7)))
        self._spo2_window.append(self.spo2)

        hr_noise = self.rng.gauss(0, noise * 2.0)
        if self.spo2 < 91:
            hr_noise += self.rng.gauss(6, 2)
        elif self.spo2 > 98:
            hr_noise -= self.rng.gauss(2, 1)
        self.heart_rate = max(30.0, min(200.0, self.heart_rate + hr_noise))

        rr_noise = self.rng.gauss(0, noise * 0.5)
        if self.spo2 < 91:
            rr_noise += self.rng.gauss(3, 1)
        self.respiratory_rate = max(5.0, min(60.0, self.respiratory_rate + rr_noise))

        if self.rng.random() < self._cfg["deterioration_prob"]:
            self._deterioration_timer = self.rng.randint(3, 7)

        if self._deterioration_timer > 0:
            self.consumption_rate = min(self.consumption_rate * 1.4, self._base_consumption * 3)
            self.spo2 = max(72.0, self.spo2 - abs(self.rng.gauss(2.5, 1.0)))
            self._spo2_window.append(self.spo2)
            self._deterioration_timer -= 1
        else:
            self.consumption_rate = max(self._base_consumption, self.consumption_rate * 0.97)

        if self.task_name == "hard" and self.rng.random() < self._cfg["emergency_prob"]:
            self.spo2 = max(72.0, self.spo2 - abs(self.rng.gauss(9, 3)))
            self._spo2_window.append(self.spo2)
            if self.rng.random() < 0.35:
                self.heart_rate = self.rng.uniform(148, 185)
                self.respiratory_rate = min(60.0, self.respiratory_rate + self.rng.gauss(8, 2))

    def _compute_reward(self, action_type: ActionType, risk: str) -> Tuple[float, str]:
        is_critical = risk == "critical"
        is_high = risk in ("high", "critical")
        is_moderate = risk == "moderate"
        oxygen_low = self.oxygen_level < 25
        spo2_danger = self.spo2 < 90
        trend_falling = self._spo2_trend() == -1

        if action_type == ActionType.WAIT:
            if is_critical:
                return -2.0, "Dangerous wait during critical state"
            if is_high:
                return -1.0, "Risky wait during high-risk state"
            if is_moderate and trend_falling:
                return -0.4, "Wait while SpO2 is falling — moderate risk"
            if is_moderate:
                return -0.2, "Suboptimal wait during moderate risk"
            return 0.1, "Safe wait — low risk"

        if action_type == ActionType.ALERT_REFILL:
            cost = -0.05
            if self._refill_countdown > 0:
                return cost - 0.4, "Refill already ordered — duplicate alert"
            if is_critical and oxygen_low:
                return cost + 1.8, "Critical + low O2: timely refill alert"
            if is_high and oxygen_low:
                return cost + 1.2, "High risk + low O2: correct refill alert"
            if oxygen_low:
                return cost + 0.8, "Proactive refill before depletion"
            if self.oxygen_level < 40:
                return cost + 0.4, "Early refill in moderate range"
            return cost - 0.5, "Premature refill — O2 still adequate"

        if action_type == ActionType.EMERGENCY_ALERT:
            if is_critical or (oxygen_low and spo2_danger):
                return 1.5, "Correct emergency response"
            if is_high and trend_falling:
                return 0.6, "Emergency alert — high risk with falling SpO2"
            if is_high:
                return 0.4, "Emergency alert for high-risk state"
            return -1.5, "False emergency — unnecessary alarm"

        if action_type == ActionType.NOTIFY_CAREGIVER:
            if is_critical:
                return 0.3, "Caregiver notified during critical state"
            if is_high:
                return 0.8, "Appropriate caregiver notification"
            if is_moderate and (self.heart_rate > 110 or self.heart_rate < 55
                                or self.respiratory_rate > 25):
                return 0.5, "Caregiver notified for abnormal vitals"
            return -0.3, "Unnecessary caregiver notification"

        return 0.0, "Unknown action"

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed)
        self._init_state()
        return self.state()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.episode_over:
            raise RuntimeError("Episode is done — call reset() before stepping again")

        self.time_elapsed += 1

        raw = action.action
        action_type: ActionType = _ACTION_MAP.get(raw if isinstance(raw, str) else raw.value)

        risk = self._risk_level()
        reward_value, reason = self._compute_reward(action_type, risk)

        if action_type == ActionType.ALERT_REFILL and self._refill_countdown == 0:
            self._refill_countdown = self._cfg["refill_delay"]

        if self._refill_countdown > 0:
            self._refill_countdown -= 1
            if self._refill_countdown == 0:
                refill_amount = self.rng.uniform(40.0, 60.0)
                self.oxygen_level = min(100.0, self.oxygen_level + refill_amount)

        self._apply_physiology()

        if self.oxygen_level <= 0.0:
            reward_value -= 5.0
            reason = "Oxygen fully depleted — patient at risk"
            self.episode_over = True
        elif self.time_elapsed >= self._cfg["max_steps"]:
            self.episode_over = True

        obs = self.state()
        r = Reward(value=round(reward_value, 3), reason=reason)

        self.history.append(StepRecord(
            step=self.time_elapsed,
            observation=obs,
            action=action_type.value,
            reward=r.value,
            reason=reason,
            done=self.episode_over,
            risk=risk,
        ))

        return obs, r, self.episode_over, {
            "risk": risk,
            "step": self.time_elapsed,
            "spo2_trend": self._spo2_trend(),
            "respiratory_rate": round(self.respiratory_rate, 1),
            "deterioration_active": self._deterioration_timer > 0,
            "refill_arriving_in": self._refill_countdown,
            "patient_type": self._cfg["patient_type"],
        }

    def state(self) -> Observation:
        predicted_depletion = self.oxygen_level / max(0.01, self.consumption_rate)
        return Observation(
            oxygen_level=round(self.oxygen_level, 3),
            spo2=round(self.spo2, 3),
            heart_rate=round(self.heart_rate, 2),
            respiratory_rate=round(self.respiratory_rate, 1),
            consumption_rate=round(self.consumption_rate, 4),
            time_elapsed=self.time_elapsed,
            risk_level=self._risk_level(),
            predicted_depletion_time=round(predicted_depletion, 2),
            spo2_trend=self._spo2_trend(),
            refill_arriving_in=self._refill_countdown,
            patient_type=self._cfg["patient_type"],
        )
