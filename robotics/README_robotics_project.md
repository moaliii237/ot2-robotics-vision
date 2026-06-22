# Robotics Block B, OT-2 Controllers

BUas Year 2 Block B robotics retake. PID and RL controllers for the Opentrons OT-2 PyBullet digital twin.

## Layout

```
robotics/
  Y2B-2023-OT2_Twin/        upstream BUas simulator (vendored, do not modify)
  src/ot2/                  shared code: sim wrapper, envelope, PID, Gym env
  task09_robotics_environment/        Task 9 deliverable (done)
  task10_pid_controller/              Task 10 deliverable
  task11_rl_controller/               Task 11 deliverable
  task12_13_integration_benchmarking/ Tasks 12-13 deliverable (CV pipeline gated)
```

## Setup

```powershell
# from the robotics/ folder, with the existing .venv already in place
.\.venv\Scripts\Activate.ps1

# install this project in editable mode so `import ot2` works from any task folder
pip install -e .

# (if torch is CPU-only) install CUDA build for the RTX 4070
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Per-task entry points

| task | run | what it does |
|---|---|---|
| 9 | `python Y2B-2023-OT2_Twin\visit_corners.py` | regenerate `task09_*/corners.gif`, etc. |
| 10 | `python task10_pid_controller\tune_pid.py` | sweep gains, write tuning log + plots |
| 10 | `python task10_pid_controller\run_pid_demo.py` | record `pid_demo.gif`, hit N targets |
| 11 | `python task11_rl_controller\test_wrapper.py` | smoke test the Gym env (1000 random steps) |
| 11 | `python task11_rl_controller\train_rl.py --total_timesteps 100000` | PPO training, logs to TB + W&B (optional) |
| 11 | `python task11_rl_controller\test_rl.py --model weights\latest.zip` | eval a checkpoint, record GIF |

See `task*/README.md` for the full deliverable writeup for each task.

## Where things live

- `src\ot2\sim.py`: the only sanctioned way to construct a `Simulation`. Handles the `textures/` chdir trap.
- `src\ot2\envelope.py`: practical work envelope constants from Task 9.
- `src\ot2\teleport.py`: sign-bug-free pipette teleport (the upstream `set_start_position` is broken; see Task 9 README).
- `src\ot2\control\pid.py`: per-axis PID with anti-windup and derivative-on-measurement.
- `src\ot2\envs\ot2_env.py`: Gymnasium env for RL training.
