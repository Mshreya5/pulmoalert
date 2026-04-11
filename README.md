---
title: PulmoAlert
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# PulmoAlert — OpenEnv ICU Oxygen Monitoring

Production-grade reinforcement learning environment for ICU oxygen cylinder and patient vitals monitoring.
An agent observes real-time patient state and decides when to wait, request a refill, notify a caregiver, or trigger an emergency — with shaped rewards and deterministic grading.

---

## Project Structure

```
pulmoalert-openenv/
├── env/
│   ├── __init__.py
│   ├── environment.py   # PulmoAlertEnv — core simulation
│   ├── models.py        # Pydantic typed models
│   ├── tasks.py         # Task factory (easy / medium / hard)
│   ├── grader.py        # Deterministic weighted grader
│   └── logger.py        # JSON-lines episode logger
├── server/
│   ├── __init__.py
│   └── app.py           # FastAPI session-isolated REST API
├── inference.py         # Evaluator entry point
├── openenv.yaml         # OpenEnv specification
├── Dockerfile           # Python 3.10-slim container
├── requirements.txt     # Pinned dependencies
└── README.md
```

---

## OpenEnv Compliance Checklist

| Requirement | Status |
|---|---|
| Pydantic typed `Observation`, `Action`, `Reward` models | ✅ |
| `reset()` → `Observation` | ✅ |
| `step(action)` → `(Observation, Reward, bool, dict)` | ✅ |
| `state()` → current `Observation` | ✅ |
| 3 distinct tasks with different behavior | ✅ easy / medium / hard |
| Reward varies across steps and tasks | ✅ |
| Deterministic grader → score ∈ [0.0, 1.0] | ✅ |
| `inference.py` prints exact structured logs | ✅ |
| `[END]` format: `task= score= steps=` | ✅ |
| `flush=True` on all prints | ✅ |
| `openenv.yaml` with `entry_point` + `server` keys | ✅ |
| `Dockerfile` CMD starts server only | ✅ |
| `PYTHONUNBUFFERED=1` in container | ✅ |

---

## Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `oxygen_level` | float | 0–100 | Cylinder fill level (%) |
| `spo2` | float | 70–100 | Blood oxygen saturation (%) |
| `heart_rate` | float | 30–200 | Heart rate (bpm) |
| `respiratory_rate` | float | 5–60 | Breaths per minute |
| `consumption_rate` | float | > 0 | O2 consumed per step |
| `predicted_depletion_time` | float | ≥ 0 | Steps until O2 = 0 |
| `spo2_trend` | int | -1 / 0 / +1 | SpO2 direction over last 3 steps |
| `time_elapsed` | int | ≥ 0 | Steps elapsed in episode |
| `risk_level` | str | low / moderate / high / critical | Composite risk label |

---

## Action Space

| Action | Correct when |
|---|---|
| `WAIT` | Risk is low, vitals stable |
| `ALERT_REFILL` | O2 < 25% or risk is high |
| `NOTIFY_CAREGIVER` | High risk, abnormal HR / RR, SpO2 falling |
| `EMERGENCY_ALERT` | Critical risk or O2 depleting + SpO2 < 90 |

---

## Tasks

| Task | O2 Start | Consumption/step | Noise | Deterioration prob | Emergency prob | Max steps |
|---|---|---|---|---|---|---|
| `easy` | 80–100% | 0.5–1.0 | low | 2% | 0% | 60 |
| `medium` | 55–80% | 1.2–2.5 | medium | 8% | 3% | 80 |
| `hard` | 30–65% | 2.8–5.0 | high | 18% | 10% | 100 |

Each task uses its own fixed seed (easy=100, medium=200, hard=300) for deterministic replay.

---

## Reward Function

| Situation | Reward |
|---|---|
| WAIT at low risk | +0.10 |
| WAIT at moderate risk | −0.20 |
| WAIT at moderate risk + SpO2 falling | −0.40 |
| WAIT at high risk | −1.00 |
| WAIT at critical risk | −2.00 |
| ALERT_REFILL when O2 < 25 + critical | +1.75 |
| ALERT_REFILL when O2 < 25 + high | +1.15 |
| ALERT_REFILL when O2 < 25 | +0.75 |
| ALERT_REFILL when O2 < 40 | +0.35 |
| ALERT_REFILL when O2 ≥ 40 (premature) | −0.55 |
| EMERGENCY_ALERT at critical | +1.50 |
| EMERGENCY_ALERT at high + SpO2 falling | +0.60 |
| EMERGENCY_ALERT at high | +0.40 |
| EMERGENCY_ALERT at low/moderate (false) | −1.50 |
| NOTIFY_CAREGIVER at critical | +0.30 |
| NOTIFY_CAREGIVER at high | +0.80 |
| NOTIFY_CAREGIVER at moderate + abnormal vitals | +0.50 |
| NOTIFY_CAREGIVER unnecessary | −0.30 |
| O2 fully depleted (terminal) | −5.00 |

---

## Grader

```
Score = sum(weighted correct decisions) / (total_steps × 2.0)
Clamped to [0.0, 1.0]
```

**Weights by risk level:**
- `critical` → 2.0
- `high` → 1.5
- `moderate` → 1.0
- `low` → 0.5

**Penalties:** each false alarm subtracts 1.0 point.

**Returns `EpisodeSummary`** with: `score`, `total_reward`, `false_alarms`, `critical_misses`, `oxygen_depletions`.

---

## Realistic Physiology

- SpO2 drifts down as cylinder empties, with per-task noise
- SpO2 trend tracked over a 3-step rolling window → `spo2_trend`
- Heart rate rises (tachycardia) when SpO2 < 91
- Respiratory rate rises (tachypnea) when SpO2 < 91
- Random deterioration events last 3–7 steps, spike consumption × 1.4
- Hard task only: sudden SpO2 crashes (−9 ± 3) and arrhythmia (HR 148–185)
- Consumption rate recovers toward baseline between deterioration events

---

## Running Inference

```bash
pip install -r requirements.txt
python inference.py
```

**Actual output (verified):**

```
[START] task=easy
[STEP] step=1 reward=0.100
[STEP] step=2 reward=0.100
...
[STEP] step=40 reward=-0.200
...
[END] task=easy score=0.33 steps=60

[START] task=medium
[STEP] step=1 reward=-0.200
...
[STEP] step=10 reward=-3.500
[END] task=medium score=0.82 steps=10

[START] task=hard
[STEP] step=1 reward=0.400
...
[STEP] step=9 reward=-3.500
[END] task=hard score=0.97 steps=9
```

Episode logs are written to `logs/<task>_<run_id>.jsonl` (excluded from git).

---

## REST API

Start the server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Environment info (name, version, tasks) |
| GET | `/health` | Liveness + active session count |
| GET | `/tasks` | List available tasks |
| POST | `/reset?task=easy&seed=42` | Create session → `session_id` + initial observation |
| GET | `/state?session_id=` | Current observation |
| POST | `/step?session_id=` | Take action → obs / reward / done / info |
| GET | `/grade?session_id=` | Grade episode history mid-run |
| GET | `/metrics` | Sessions in flight per task |
| DELETE | `/session?session_id=` | Clean up session |

**Example:**
```bash
# Start episode
curl -X POST "http://localhost:7860/reset?task=hard&seed=42"
# → {"session_id": "abc-123", "observation": {...}}

# Take a step
curl -X POST "http://localhost:7860/step?session_id=abc-123" \
  -H "Content-Type: application/json" \
  -d '{"action": "ALERT_REFILL"}'

# Grade so far
curl "http://localhost:7860/grade?session_id=abc-123"
```

---

## LLM Policy

Set env vars to enable the LLM policy (heuristic baseline used if unset):
```bash
export API_BASE_URL="https://your-litellm-proxy/v1"
export API_KEY="your-key"
```

The LLM receives a structured clinical prompt with all 5 vitals + trend + risk level and must reply with exactly one action name.

---

## Docker

```bash
docker build -t pulmoalert .
docker run -p 7860:7860 pulmoalert
```

The container starts the FastAPI server on port 7860.
The evaluator runs `python inference.py` separately.

---

## Dependencies

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic>=1.10.16,<2.0.0
aiofiles==23.2.1
openai>=1.0.0
pyyaml>=6.0
```
