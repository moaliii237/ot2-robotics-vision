# Retake (Opportunity 2)

For each ILO you wish to retake, provide evidence and a clear explanation of how the ILO requirements are met in your retake. If you received feedback from your mentor, explicitly describe how you addressed this feedback. Clearly indicate the specific ILO criterion for which you are submitting a retake. If you are not submitting a retake for a particular ILO, remove the corresponding header and template text from this document.

## ILO 1.6 Presentation

## Criterion A
My slides are clean and consistent. Every slide has a clear title and a number, the text is kept short, and the visuals are sized to read easily, for example the pipeline diagram (slide 4), the predicted root mask (slide 6), and the two native bar charts on the results slide (slide 11).

Evidence
[Presentation slides](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)

## Criterion B
I prepared and rehearsed the talk, started on time, and delivered it within the 10-minute limit at a steady pace, covering the full brief from problem to summary.

Evidence
[Presentation slides](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)
## Criterion C
I tried to answered Dean's questions clearly and accurately. 

Evidence
[Presentation slides](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)

## Criterion D
I delivered the presentation confidently, kept eye contact, used clear language, and engaged the audience throughout.

Evidence
[Presentation slides](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)

## Criterion E
I organised the talk logically so it is easy to follow: the problem (slides 2-3), the proposed solution and approach (slides 4-10), the results and their interpretation (slide 11), then iterations, assumptions and next steps (slides 12-14), and a summary (slide 15).

Evidence
[Presentation slides](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)
## ILO 2.6 Critical Thinking

## Criterion A
I logged at least two work items mapped to ILO 2.6 each week in my worklog, and each week I reflected on them in Section B of my learning log. In my presentation (slide 12) I highlighted three crucial iterations: (1) reshaping the RL reward to escape a 0.9 mm plateau, which moved success from 95% to 100% of targets under 1 mm; (2) removing a minimum-output friction hack from the PID and instead tuning it from a measured first-order plant model, which improved the error from about 0.4 mm to sub-micron; and (3) fixing a CV rim-shadow that was mistaken for a plant and correcting an unfair benchmark, which restored 5/5 plant detection.

Evidence
[Worklog](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/adsai_y2b_25_26_worklog_Mohammadali%20Jaberi.xlsx)
[Presentation slides](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)

## ILO 6.4 Image Data

## Criterion A (Brightspace)
No evidence link is required. I delivered the three segmentation masks (shoot, seed, root) for my assigned image and completed the peer review with the template. Result: masks [fill in, e.g. 2/3] and peer review [fill in, e.g. 2/2].

## Criterion B (Task 2)
I crop the petri dish using only traditional computer vision (no ML/DL). I convert the image to grayscale, blur it to reduce noise, apply adaptive thresholding so the bright dish separates from the dark background under uneven lighting, clean the result with a morphological close, find the contours, keep the largest roughly-square contour, and crop to that square. It works on Y2B_23, Y2B_24 and Y2B_25 within the ±30 pixel tolerance.

Evidence
[task_2.ipynb](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/task_2.ipynb)
## Criterion C (Task 3)
I separate the five plants as individual instances using only traditional computer vision (no ML/DL). After isolating the plant pixels from the background, I label the connected regions and keep the five largest as the individual plants, then display the result on task_3_image_1 and task_3_image_2.

Evidence
[task_3.ipynb](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/task_3.ipynb)


ILO 8.5 B
## Criterion A
I implemented a root segmentation pipeline and evaluated its performance using the F1 metric, achieving an F1 score of 0.88 on the validation set. In addition, I developed inference code that processes raw images from the Hades dataset and automatically generates the corresponding segmentation masks. I attempted to upload my model to GitHub, but encountered an error due to the file size, so it is shared on Google Drive.

Evidence
[task_5_training.ipynb](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/task_5_training.ipynb)
[task_5_inference.ipynb](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/task_5_inference.ipynb)
[Trained U-Net model (.h5)](https://edubuas-my.sharepoint.com/:f:/g/personal/244437_buas_nl/IgAc095pcCqJSqzCmbbWboFfAdApi5hHaQZSaqQYuwwUEik?e=go4IIz)
[Predicted mask for the test image](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/task5_test_image_prediction_MoaliJaberiiii_244437.png)
## Criterion B
I built an end-to-end computer vision pipeline that takes a directory of raw images and outputs the primary root length for each plant: it crops the dish, runs the U-Net for root segmentation, extracts the root system architecture, and measures the primary root length. I submitted my predictions to the Kaggle competition.

Evidence
[task_8.ipynb](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/task_8.ipynb)
[Root segmentation model](https://claude.ai/epitaxy/local_86a646d2-99a7-421a-a98a-6bff78113757)
[Kaggle private leaderboard screenshot](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/computer_vision/kaggle_leaderboard_screenshot.png)
[Presentation describing my best Kaggle submission](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)
## ILO 8.6 Robotic Control and Reinforcement Learning

ILO 8.6 A

## Criterion A (Task 9)
I interfaced with the OT-2 PyBullet simulation to send velocity commands and read back the pipette state (position, velocity, distance to target), and I determined the work envelope by driving the pipette to all corners. I found that only 5 of the 8 corners are reachable because of the chassis floor, and I documented the envelope, setup, dependencies, and key findings in the README.

Evidence
[task09_robotics_environment/README.md](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task09_robotics_environment/README.md)
[Interfacing code (sends commands, reads observations)](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/src/ot2/sim.py)
[GIF of the pipette moving to all 8 corners](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task09_robotics_environment/corners.gif)
## Criterion B (Task 10)
I designed a PID controller (one per axis), tuned it from a measured first-order plant model using the SIMC rule, cross-checked the gains with an Optuna search, and benchmarked it. It reaches the target to sub-micron in simulation, well inside the 1 mm requirement. The README documents the libraries, implementation steps, tuning strategy, final gains, and error analysis.

Evidence
[task10_pid_controller/README.md](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task10_pid_controller/README.md)
[PID implementation code](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/src/ot2/control/pid.py)
[PID tuning & testing code](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task10_pid_controller/tune_pid.py) (also sysid.py, imc_tuning.py, run_pid_demo.py in the same folder)
[GIF of the robot controlled by the PID](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task10_pid_controller/pid_demo.gif)
[Presentation slides (tuning results and final gains)](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)
## Criterion C (Task 11)
I implemented a Gymnasium-compatible environment and trained a PPO controller with Stable Baselines 3. I defined the observation and action spaces, a shaped reward, and termination conditions, and I evaluated the trained policy across 60 random targets, hitting all of them under 1 mm. I also compared the RL controller against the PID across defined metrics. The README documents the design, libraries, training, and results.

Evidence
[task11_rl_controller/README.md](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/README.md)
[ot2_gym_wrapper.py](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/ot2_gym_wrapper.py)
[test_wrapper.py](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/test_wrapper.py)
[RL training code](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/train_rl.py)
[RL testing code](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/test_rl.py)
[Best individual model weights](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/weights/latest.zip)
[Best group model weights](https://edubuas-my.sharepoint.com/:f:/g/personal/244437_buas_nl/IgAc095pcCqJSqzCmbbWboFfAdApi5hHaQZSaqQYuwwUEik?e=go4IIz)
[GIF of the robot controlled by the RL agent](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task11_rl_controller/rl_demo.gif)
[Presentation slides (performance + hyperparameter search)](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)

## Criterion D (Tasks 12 and 13)
I integrated the CV pipeline with both controllers: the U-Net finds each root tip, calibration converts the tip pixel to robot millimetres, and the PID or RL controller drives the pipette there and inoculates it. I benchmarked both controllers across 19 dishes and 82 root tips (both hit 100% under 1 mm), reporting accuracy, settle time, control effort, and wall time, plus limitations and recommendations. The README documents the integration, coordinate transformation, and results.

Evidence
[task12_13_integration_benchmarking/README.md](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task12_13_integration_benchmarking/README.md)
[Integrated controller code (PID and RL)](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task12_13_integration_benchmarking/demo_cv_to_controller.py) (with calibration.py and cv/root_pipeline.py)
[GIF, autonomous inoculation (RL)](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task12_13_integration_benchmarking/results/demo_rl_09.gif)
[GIF, autonomous inoculation (PID)](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task12_13_integration_benchmarking/results/demo_pid_09.gif)
[Benchmarking code](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/robotics/task12_13_integration_benchmarking/benchmark_pid_vs_rl_cv.py)
[Presentation with benchmarking results](https://github.com/BredaUniversityADSAI/2025-26b-fai2-adsai-MohammadaliJaberi244437/blob/main/Block_B_Retake_Presentation.pdf)

_provide evidence or remove this header_