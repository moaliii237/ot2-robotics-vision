# ot2-robotics-vision

This is my Year 2 Block B project at BUas. The goal is one end-to-end system: take a
photograph of a petri dish with plant seedlings, find the tip of each plant's primary
root, and drive a simulated Opentrons OT-2 pipetting robot to every tip so it could be
inoculated there. So it is two halves joined together, a computer vision pipeline that
turns a photo into world coordinates, and a robotics pipeline that moves the robot to
those coordinates.

I built the whole thing myself. The OT-2 simulator and the U-Net training data came from
the course, everything else here (the segmentation training, the calibration, both
controllers, the integration layer, the tests) is my work.

## What it does

Given one raw dish photo (about 4202 x 3006 pixels), the system:

1. Crops the petri dish out of the photo with traditional CV (threshold, morphological
   close and erode, largest contour).
2. Runs a U-Net over 256 px patches of the cropped region to get a binary root mask.
3. Splits the mask into separate plants with connected components, filters out dish-rim
   artefacts, skeletonises each plant, and walks the skeleton to find the primary-root
   tip pixel.
4. Converts each tip pixel into OT-2 world coordinates (metres) using a calibration I
   defined and validated.
5. Drives the simulated pipette to every tip with one of two controllers I wrote, a PID
   controller and a PPO reinforcement-learning agent, and benchmarks them against each
   other.

Both controllers hit every target on every dish under the 1 mm spec. The PID gets to
sub-micron, the RL agent holds around half a millimetre.

## How it works, and why I made the choices I did

### Computer vision

The U-Net is trained on 256 px patches cut from the dish images with a stride of 192, so
patches overlap. Batch size 32, 40 epochs, Adam at 1e-4, and a loss that combines binary
cross-entropy with a Dice term because the roots are thin and the mask is very
imbalanced (mostly background), so plain BCE alone under-weights the foreground. Best
validation F1 was 0.877. The trained model is `unet_256px.keras` (the same weights are
also saved as a `.h5`).

At inference time I do not run the U-Net on the whole raw photo. The raw image has a
black border and a white legend strip on the right, and running the network over that
just produces noise. So I crop to the dish first, predict patch by patch inside the
crop, then lift the predicted mask back into raw-image pixel coordinates. This matters
later because the calibration assumes the mask lives in the original photo frame.

Two CV bugs I had to fix while integrating:

- On one test dish the network produced a thin 42 x 2028 pixel strip down the dish rim
  whose mask area (5074 px) was larger than the smallest real plant (4235 px). My
  "keep the 5 largest blobs" rule then threw away a real plant and kept the rim. I added
  an aspect-ratio filter that rejects any blob more extreme than 10:1 before the keep-N
  step. Real seedlings are roughly 3:1 to 6:1, the rim strip was 48:1, so 10:1 separates
  them on every dish in the set.
- For tip extraction I first used a FIFO breadth-first search on the skeleton, which is
  not a true shortest path and over-estimates on looped skeletons. I replaced it with
  Dijkstra so the tip and the root length are exact geodesics. The tip is the farthest
  skeleton pixel from the shoot base, and I reconstruct the whole primary-root path by
  backtracking parent pointers so I can draw it on the demo and check it by eye.

### From pixels to robot coordinates

There is no real 3D ground truth for where a root tip is, it is only a pixel in a photo.
So the calibration is not "derived" from anything, it is a defined linear convention: the
dish photo maps onto the 150 mm square footprint of the plate, centred on the plate's
known world position (0.18270, 0.13700, 0.08700) m taken from the sim's `custom.urdf`.
Image columns map to world +y, image rows map to world -x. I have a `verify_calibration.py`
that checks the forward and inverse transforms are exact inverses (round-trip error about
6e-14 px) and that the dish-centre pixel lands on the plate centre exactly.

One honest caveat I document in the code: I first tried to justify the axis signs from
the way PyBullet renders the plate, but that reasoning was wrong, PyBullet cube-unwraps a
box texture so the render cannot pin an absolute orientation. The mapping is a convention
I fixed and validated, not something physics forced.

### Characterising the simulator first (Task 9)

Before writing any controller I measured the robot. I found:

- The action input is a velocity command in m/s, world frame, with a gain of about
  1.0 m/s per unit on each axis (system ID later confirmed K close to 1.0).
- The reachable workspace is smaller than the joint limits because the chassis floor sits
  at a uniform plane around z = 0.1695 m. Anything below that is not reachable by
  descending from above, only 5 of the 8 envelope corners are reachable from rest. So for
  controller training I sample targets from a "practical envelope" with z_min raised to
  0.170 m.
- The sim's default `get_pipette_position` rounds to 4 decimal places, a hard 0.1 mm
  floor. For sub-mm work I read the joint states directly and rebuild the pose at float64
  (`precise_pipette_position`). Without this neither controller can see, or fix, errors
  below 0.1 mm.

I also found and fixed a sign bug in the simulator's `set_start_position`. The upstream
function computes `adjusted_x = x - robot_x - offset_x` and resets the joint to that, but
`get_pipette_position` returns `robot_x - joint_0 + offset_x`, with a minus on the joint.
So asking the pipette to teleport to a target actually mirrors it about `robot_x + offset_x`
and it lands in the wrong place. My `teleport_pipette` inverts the sign correctly with
`p.resetJointState`. This mattered a lot for RL, my first env called the broken function on
every reset, the observation drifted out of sync with the world, and learning never
worked.

### PID controller (Task 10)

I designed the PID in three stages instead of just guessing gains:

1. System identification. I fit the per-axis velocity step responses to a first-order
   model `v(t) = K (1 - exp(-t/tau))` with `scipy.optimize.curve_fit`. The plant is a
   clean first-order velocity loop, K close to 1.0 on every axis, time constant tau
   between 1.6 ms (z, stronger motor) and 3.7 ms (x). R squared above 0.95 on x/y, 0.89
   on z (z rises faster than the sim timestep so there are fewer samples on the edge).
2. Theoretical gains. The plant from action to position is an integrator plus a
   first-order lag, `G(s) = K / (s (tau s + 1))`. Skogestad's SIMC rule for that form
   gives a PI controller, `Kp = 1/(K(tau+tau_c))`, `Ki = Kp/(4(tau+tau_c))`. With a
   closed-loop time constant tau_c = 10 ms this gives Kp about 76, Ki about 1400, no
   derivative needed.
3. Cross-check with a search. I let Optuna's TPE sampler explore the full (Kp, Ki, Kd,
   kff) space for 80 trials. It landed at Kp 106, Ki 418, Kd 0.34, the same regime as
   SIMC. Two independent methods agreeing is good evidence the gains are right.

Final per-axis gains (SIMC, tau_c = 10 ms): x Kp 72.7 Ki 1323, y Kp 77.1 Ki 1485, z Kp
86.3 Ki 1860. The controller itself (`src/ot2/control/pid.py`) does derivative on the
measurement, not the error, so a setpoint jump does not cause a derivative kick when I
step through a sequence of targets, and conditional anti-windup, it stops integrating
when the output is saturated and the error still points the same way.

The biggest lesson here was about my own first attempt. My first PID had a 0.5 mm settle
threshold and a 0.12 minimum-output floor to punch through static friction. Together they
created a deadband oscillation and the controller settled around 0.47 mm. When I measured
the actual static-friction dead zone in Task 9 it was at most 0.02 action units, far
smaller than I assumed, so the floor was doing more harm than good. Removing both and
adding the integrator took the max error from 0.0036 mm down to 0.00017 mm (170 nm).

### RL controller (Task 11)

The RL side is a custom Gymnasium environment (`src/ot2/envs/ot2_env.py`) trained with PPO
from Stable-Baselines3. Observation is 9-D: pipette position normalised to [-1, 1], the
three joint velocities, and the vector from pipette to target in metres. I leave the
target out of the observation on purpose, the relative vector already carries it and it
keeps an absolute frame out of the policy. The action is a 3-D continuous velocity command,
the sim's fourth "drop" action is pinned to 0 since dispensing is not part of this task.
Network is a small two-layer 64-unit MLP, 36,103 parameters, a bigger net never helped, the
bottleneck was always reward shaping.

The reward is where I spent most of my time. My first reward gave a flat +1.0 per step
while inside 1 mm plus a +20 terminal bonus. The problem: once the policy parked inside
1 mm it got the same reward at 0.05 mm as at 0.95 mm, so it had no gradient pulling it
tighter and plateaued. My final reward (I called it Path B) replaces the flat bonus with
three exponential proximity wells at 10 mm, 0.3 mm, and 0.05 mm length scales, so the
pull keeps sharpening as the error shrinks, plus a much larger +500 terminal so committing
to the target beats hovering near it. After that change the episode length dropped from
about 600 steps (truncated) to about 100 (terminated on dwell success), which is the sign
of a policy that actually commits.

Training was three runs: a 100k-step sanity run on my laptop GPU to prove the pipeline,
a 1.8M-step run that met the spec but failed about 5% of targets, then the Path B run from
scratch on a Colab A100, around 3.25M steps in about 70 minutes, which hit 100%. I stopped
at 3.25M because a late KL-divergence spike suggested the learning rate schedule was going
unstable and more training risked a regression.

## Results, honestly

CV (U-Net root segmentation): best validation F1 0.877.

PID controller on 10 random targets: 100% under 1 mm, median final error effectively
float64 zero, max error 0.00017 mm. It reaches literal zero in float64 when given enough
convergence time, because the sim is bit-exact deterministic and the integral term drives
the residual to the target. This is a statement about the sim's noise floor, not real
hardware. A real OT-2 would have backlash and friction around 100 micrometres that the
sim does not model.

RL controller on 60 random targets (3 seeds x 20 targets): 100% under 1 mm, max error
0.97 mm, median 0.59 mm.

End-to-end on the real CV-derived targets (82 root tips across all 19 test dishes), both
controllers re-homed to the same start with an equal step budget so it is a fair
comparison:

| controller | success under 1 mm | median error | max error | median settle | effort/step |
|---|---|---|---|---|---|
| PID (SIMC) | 82/82 | sub-micron | 0.0003 mm | 74 steps (~0.31 s) | 0.153 |
| RL (PPO) | 82/82 | 0.466 mm | 0.881 mm | 93 steps (~0.39 s) | 0.325 |

PID is far more accurate because it has a system-identified plant model and an analytical
tuning. RL matches the spec with no plant model at all, at about twice the control effort
and a bit slower to settle. For this task (a well-behaved linear plant, no contact) PID is
the right tool. RL is the answer when you cannot model the plant or do not have a system-ID
pipeline.

One thing worth being upfront about: my first version of the integration benchmark
reported "PID 2.6x faster, 22x less effort", but that was an artefact of an unfair test
where PID started each target from the previous tip while RL reset to the dish centre. I
found that, made both controllers re-home to the same pose, and the corrected numbers are
above. The positioning accuracy was correct all along, what changed was making the
comparison fair.

### Known limitations

- The CV pipeline is root-mask-only, no shoot model. If the U-Net under-segments, a plant
  can be missed, and touching roots could merge. CV found fewer than 5 plants on 6 of the
  19 dishes, mostly because there genuinely were fewer plants visible.
- The aspect-ratio filter is a heuristic. 10:1 worked on every dish in the set but could
  mis-reject an unusual long thin lateral root.
- The calibration is specific to this sim's plate geometry, not an arbitrary physical rig.
- The pipette hovers at z = 0.175 m and drops, it does not touch the agar surface, because
  the chassis floor blocks descent lower in the sim.

## Tech stack

- Robotics: PyBullet (the OT-2 digital twin), Gymnasium (custom env), Stable-Baselines3
  (PPO), simple-pid concepts (my own PID class), Optuna (gain search), SciPy
  (`curve_fit` for system ID), NumPy, imageio (GIFs).
- Computer vision: TensorFlow / Keras (U-Net), OpenCV (dish crop, morphology), scikit-image
  (skeletonisation, region properties), NumPy.
- Tooling: pytest (a small core test suite), pyproject-based editable install, matplotlib,
  TensorBoard.

## Repo layout

```
computer_vision/        five notebooks: dish crop, plant segmentation, U-Net train + infer,
                        Kaggle root-length pipeline
robotics/
  src/ot2/              the shared package
    sim.py              the only safe way to construct the sim (handles the textures chdir)
    teleport.py         sign-bug-free pipette teleport
    envelope.py         work-envelope constants from Task 9
    control/pid.py      per-axis PID, anti-windup, derivative on measurement
    envs/ot2_env.py     the Gymnasium env for RL
  task09_robotics_environment/        sim characterisation
  task10_pid_controller/              PID design, tuning, comparison
  task11_rl_controller/               PPO env, training, eval
  task12_13_integration_benchmarking/ CV -> world calibration, demos, benchmark
  tests/test_core.py    fast pure-numpy tests (calibration round-trip, PID anti-windup,
                        envelope containment)
```

## Setup and running

This repo does not contain two external pieces:

- The OT-2 simulator (`Y2B-2023-OT2_Twin/`) is provided by the course and I do not
  redistribute it. Drop it into `robotics/` from the course zip. The code expects it next
  to `robotics/src/`.
- The trained U-Net weights are about 370 MB, over GitHub's 100 MB file limit, so they are
  not committed. The CV notebooks expect the model file (`unet_256px.keras` or the matching
  `.h5`) and the path is set near the top of `task_5_inference.ipynb` and `task_8.ipynb`.

To run the robotics code:

```powershell
cd robotics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# RL inference uses torch; for the RTX 4070 install the CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

A few entry points:

```powershell
# characterise the sim (Task 9)
python task09_robotics_environment\smoke_test.py
python task09_robotics_environment\find_envelope.py

# tune and demo the PID (Task 10)
python task10_pid_controller\sysid.py
python task10_pid_controller\imc_tuning.py
python task10_pid_controller\run_pid_demo.py

# smoke-test the RL env, then evaluate the trained model (Task 11)
python task11_rl_controller\test_wrapper.py
python task11_rl_controller\eval_robust.py --model task11_rl_controller\weights\latest.zip

# full CV-to-robot benchmark across all 19 dishes (Task 13)
python task12_13_integration_benchmarking\benchmark_pid_vs_rl_cv.py
```

The core tests need neither the sim nor the model:

```powershell
pytest robotics\tests\test_core.py
```

The CV notebooks run on Colab with a GPU. Each task folder under `robotics/` has its own
README with the full write-up, the result numbers, and a GIF of the controller in action.
