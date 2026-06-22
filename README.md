# Block B Retake (Opportunity 2), 244437

This repo holds my retake work for Block B, split into the two halves of the brief: a
traditional and deep-learning **computer vision** pipeline for measuring plant roots, and
the **robotics** half where an OT-2 pipetting robot is driven to those roots by a PID
controller and a reinforcement-learning policy.

## Where each ILO's evidence lives

| ILO | What it is | Where to look |
|---|---|---|
| 6.4 B | Petri-dish extraction (traditional CV) | [`computer_vision/task_2.ipynb`](computer_vision/task_2.ipynb) |
| 6.4 C | Plant instance segmentation (traditional CV) | [`computer_vision/task_3.ipynb`](computer_vision/task_3.ipynb) |
| 8.5 A | U-Net root segmentation: training + inference | [`computer_vision/task_5_training.ipynb`](computer_vision/task_5_training.ipynb), [`task_5_inference.ipynb`](computer_vision/task_5_inference.ipynb), prediction PNG, model (on Drive) |
| 8.5 B | Kaggle root-length pipeline | [`computer_vision/task_8.ipynb`](computer_vision/task_8.ipynb), Kaggle leaderboard screenshot, model (on Drive) |
| 8.6 A | OT-2 environment interaction | [`robotics/task09_robotics_environment/`](robotics/task09_robotics_environment) (README, scripts, `corners.gif`) |
| 8.6 B | PID controller | [`robotics/task10_pid_controller/`](robotics/task10_pid_controller) (README, code, `pid_demo.gif`) |
| 8.6 C | RL (PPO) controller | [`robotics/task11_rl_controller/`](robotics/task11_rl_controller) (README, code, `rl_demo.gif`, weights) |
| 8.6 D | CV + robot integration and benchmarking | [`robotics/task12_13_integration_benchmarking/`](robotics/task12_13_integration_benchmarking) (README, code, demo GIFs, benchmark) |

ILO 1.6 / 2.6 / 5.1 are evidenced through the presentation and the work log, not code, so
there is nothing in this repo for those.

## The two halves

- **`computer_vision/`**: five notebooks (Tasks 2, 3, 5-train, 5-infer, 8) plus the
  prediction image and the Kaggle screenshot. The trained U-Net is ~370 MB so it is not
  committed here, see the note in `computer_vision/README.md` for the Drive link.
- **`robotics/`**: one Python package (`src/ot2`) shared by four task folders. Each task
  folder has its own README explaining what it does and a GIF of the result.

## Running the robotics code

The robotics scripts import the `ot2` package and the OT-2 simulator. The simulator
(`Y2B-2023-OT2_Twin/`) is BUas-provided and is not redistributed here. Drop it into
`robotics/` from the LMS zip, then:

```bash
cd robotics
python -m venv .venv && .venv\Scripts\activate
pip install -e .
```

Each task README has the exact command to reproduce its GIF and results.
