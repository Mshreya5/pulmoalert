---
title: PulmoAlert
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# PulmoAlert (OpenEnv)

## 🫁 Problem Description

PulmoAlert simulates real-world medical support: an AI agent monitors an oxygen cylinder and patient vitals in a homecare or field hospital scenario.

The environment tracks:
- Oxygen tank level (% remaining)
- Patient SpO2
- Heart rate
- Oxygen consumption rate
- Time progression

Agent actions:
- `WAIT`
- `ALERT_REFILL`
- `EMERGENCY_ALERT`
- `NOTIFY_CAREGIVER`

Objective: keep patient safe by avoiding oxygen depletion and emergencies while minimizing unnecessary alerts.

## 📊 Observation Space

`Observation` model fields:
- `oxygen_level`: 0–100
- `spo2`: 0–100
- `heart_rate`: bpm
- `consumption_rate`: % per step
- `time_step`: integer
- `risk_level`: one of `low`, `moderate`, `high`, `critical`

## 🎯 Action Space

`Action` model can be one of:
- `WAIT`
- `ALERT_REFILL`
- `EMERGENCY_ALERT`
- `NOTIFY_CAREGIVER`

## 🧮 Reward Design

Dense per-step reward with penalties and incentives:
- safe waits in low risk;+1
- waiting in high risk;-3
- timely refill when oxygen<20;+2.5
- early unnecessary refill;-2
- correct emergency response;+4
- false emergency;-3
- caregiver notification in high risk;+1.5
- oxygen depletion failure;-20
- emergency termination bonus +1

## 🧪 Tasks

Three tasks increase difficulty by initial conditions and noise:

- `easy`:
  - Stable vitals
  - Low consumption
  - Goal: refill before oxygen <20

- `medium`:
  - Moderate fluctuation and consumption
  - Balances safety vs unnecessary alerts

- `hard`:
  - High volatility in vitals
  - Critical emergency detection required

## 🧩 Project Structure

```
pulmoalert-openenv/
├── env/
│   ├── environment.py
│   ├── models.py
│   ├── tasks.py
│   ├── grader.py
├── openenv.yaml
├── baseline.py
├── requirements.txt
├── Dockerfile
├── README.md
```

## ▶️ Quick Start (Local)

```bash
cd pulmoalert-openenv
python -m pip install -r requirements.txt
python baseline.py
```

## 🐳 Docker

```bash
docker build -t pulmoalert-openenv .
docker run --rm pulmoalert-openenv
```

## 🤖 OpenAI Policy

If `OPENAI_API_KEY` is set, `baseline.py` runs a GPT-guided policy fallback each step; otherwise uses deterministic heuristic.

## 🧾 Grade Evaluation

`env/grader.py` exposes:
- `grade_episode(history)` returns 0.0–1.0
- `grade_run(episodes)` returns average score

Safety, correct alerts, and efficiency are measured.

## 📌 Baseline Scores

Baseline run prints average reward and OpenEnv scores per difficulty.

---

## 🪪 Validations

`openenv.yaml` provides metadata for OpenEnv validator requiring observation, action, reward ranges, and tasks.
=======
---
title: Pulmoalert
emoji: 📉
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 3bb332b58c0c48c51cbbab6ddb70b3e86b7c8f19
