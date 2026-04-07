import random
from typing import Tuple, Dict

from env.models import Observation, Action, ActionType, Reward


class PulmoAlertEnv:
    TASK_CONFIG = {
        "easy": {
            "oxygen_level": (70.0, 100.0),
            "spo2": (95.0, 99.0),
            "heart_rate": (60.0, 90.0),
            "consumption_rate": (0.8, 1.5),
            "noise_scale": 0.5,
        },
        "medium": {
            "oxygen_level": (55.0, 90.0),
            "spo2": (92.0, 98.0),
            "heart_rate": (55.0, 105.0),
            "consumption_rate": (1.5, 2.8),
            "noise_scale": 1.0,
        },
        "hard": {
            "oxygen_level": (40.0, 80.0),
            "spo2": (88.0, 96.0),
            "heart_rate": (50.0, 120.0),
            "consumption_rate": (2.5, 4.5),
            "noise_scale": 2.0,
        },
    }

    def __init__(self, task_name: str = "easy", seed: int = 42):
        if task_name not in self.TASK_CONFIG:
            raise ValueError("Unknown task name: %s" % task_name)
        self.task_name = task_name
        self.seed = seed
        self.rng = random.Random(seed)
        self._reset_vals()

    def _reset_vals(self):
        self.time_step = 0
        self.episode_over = False
        self.history = []
        cfg = self.TASK_CONFIG[self.task_name]

        self.oxygen_level = self.rng.uniform(*cfg["oxygen_level"])
        self.spo2 = self.rng.uniform(*cfg["spo2"])
        self.heart_rate = self.rng.uniform(*cfg["heart_rate"])
        self.consumption_rate = self.rng.uniform(*cfg["consumption_rate"])
        self.noise_scale = cfg["noise_scale"]

        self.last_refill_step = None
        self.emergency_triggered = False

    def _calc_risk(self) -> str:
        if self.oxygen_level <= 10 or self.spo2 <= 88:
            return "critical"
        if self.oxygen_level <= 25 or self.spo2 <= 92:
            return "high"
        if self.oxygen_level <= 40 or self.spo2 <= 94:
            return "moderate"
        return "low"

    def _apply_physiology(self):
        if self.episode_over:
            return

        shift = self.rng.gauss(0, self.noise_scale * 0.2)
        self.consumption_rate = max(0.5, self.consumption_rate + shift * 0.2)

        decrement = self.consumption_rate * (1 + self.rng.gauss(0, self.noise_scale * 0.04))
        self.oxygen_level = max(0.0, self.oxygen_level - decrement)

        spo2_shift = self.rng.gauss(0, self.noise_scale * 0.6)
        self.spo2 = max(75.0, min(100.0, self.spo2 - decrement * 0.03 + spo2_shift))

        hr_shift = self.rng.gauss(0, self.noise_scale * 1.5)
        self.heart_rate = max(30.0, min(180.0, self.heart_rate + hr_shift))

    def reset(self) -> Observation:
        self._reset_vals()
        self.episode_over = False
        self.history = []
        return self.state()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.episode_over:
            raise RuntimeError("Step called after episode terminated")

        self.time_step += 1

        reward_value = 0.0
        reason = ""

        risk = self._calc_risk()
        is_critical = risk == "critical"
        is_high = risk in ("high", "critical")

        if action.action == ActionType.WAIT:
            if is_high:
                reward_value -= 3.0
                reason = "WAIT under high risk"
            else:
                reward_value += 1.0
                reason = "SAFE wait"

        elif action.action == ActionType.ALERT_REFILL:
            if self.oxygen_level < 20.0:
                reward_value += 2.5
                reason = "Timely refill alert"
                self.consumption_rate = max(0.8, self.consumption_rate * 0.8)
            elif self.oxygen_level > 40.0:
                reward_value -= 2.0
                reason = "Unnecessary early refill"
            else:
                reward_value += 0.5
                reason = "Proactive refill in moderate range"

        elif action.action == ActionType.EMERGENCY_ALERT:
            if is_critical:
                reward_value += 4.0
                reason = "Correct emergency detection"
                self.emergency_triggered = True
            else:
                reward_value -= 3.0
                reason = "False emergency alert"

        elif action.action == ActionType.NOTIFY_CAREGIVER:
            if is_high and not is_critical:
                reward_value += 1.5
                reason = "Responsible caregiver notification"
            elif is_critical:
                reward_value += 1.0
                reason = "Critical caregiver notification"
            else:
                reward_value -= 1.0
                reason = "Unnecessary caregiver alert"

        # Progress time and physiology after action.
        self._apply_physiology()

        if self.oxygen_level <= 0.0:
            reward_value -= 20.0
            reason = "Oxygen depletion failure"
            self.episode_over = True

        if self.emergency_triggered and not self.episode_over:
            self.episode_over = True
            reward_value += 1.0
            reason = reason or "Emergency procedure executed"

        obs = self.state()
        done = self.episode_over
        info = {
            "risk": risk,
            "step": self.time_step,
            "oxygen_level": self.oxygen_level,
            "spo2": self.spo2,
            "heart_rate": self.heart_rate,
            "consumption_rate": self.consumption_rate,
        }

        r = Reward(value=round(reward_value, 3), reason=reason)

        self.history.append({
            "observation": obs.dict(),
            "action": action.action.value,
            "reward": r.value,
            "done": done,
            "info": info,
        })

        return obs, r, done, info

    def state(self) -> Observation:
        return Observation(
            oxygen_level=round(self.oxygen_level, 3),
            spo2=round(self.spo2, 3),
            heart_rate=round(self.heart_rate, 2),
            consumption_rate=round(self.consumption_rate, 3),
            time_step=self.time_step,
            risk_level=self._calc_risk(),
        )
