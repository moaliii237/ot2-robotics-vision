# ILO 8.6 Robotic Control and Reinforcement Learning

The student is able to design, implement, and evaluate control algorithms and reinforcement learning algorithms, integrating computer vision techniques for perception, to effectively address business objectives.

|pre-requisites|A: 6 points|B: 7 points|C: 8 points|D: 9 points|
|--|--|--|--|--|
|The student exhibits professional behaviour during DataLab sessions. Student presented their findings during the final block presentation.|The student is able to interact with a simulation environment to understand and manipulate observations and actions relevant to a robotics system, including deriving and interpreting key parameters such as position, velocity, and distance to the target, extracting actionable insights to inform controller design. Furthermore, the student is able to document observations and key findings during their interaction with the simulation environment.|The student is able to design and implement traditional controllers for a robotics system, quantify their performance, and optimize parameters to improve system behavior. Moreover, the student is able to create visualizations of the controller's response to different inputs, providing a clear representation of system behavior and aiding in performance evaluation. Additionally, the student is able to document the setup comprehensively, including required libraries, implementation steps, and tuning strategies, to ensure reproducibility.|The student is able to design and implement reinforcement learning-based controllers for a robotics system, quantify their performance, and optimize parameters to improve system behavior. Moreover, the student is able to create visualizations of the controller's response to different inputs, providing a clear representation of system behavior and aiding in performance evaluation. Furthermore, the student is able to evaluate and compare the RL controller’s performance with traditional methods across a set of defined metrics. Additionally, the student is able to document the setup comprehensively, including required libraries, implementation steps, and tuning strategies, to ensure reproducibility.|The student is able to integrate controllers with a computer vision pipeline, enabling the robotic system to adapt its actions based on visual input. Moreover, the student is encouraged to design modular systems that allow the seamless swapping of computer vision or control components without significant rework, thereby enhancing flexibility and maintainability. Additionally, the student is able to evaluate the overall system performance, ensuring that the system meets the desired specifications, and provide recommendations for system improvements.|

## Notes on this ILO

This ILO is a must pass ILO.

This ILO is linked to DataLab Tasks 9, 10, 11, 12, and 13. The presentation (Task 14) is a pre-requisite.

We want you to **document your approach clearly using README files.** Create the following folders in your repository:

* `task09_robotics_environment`
* `task10_pid_controller`
* `task11_rl_controller`
* `task12_13_integration_benchmarking`

In each folder, create a `README.md` file. Document your approach, implementation steps, and key results in the corresponding task’s `README.md` file. Include relevant code snippets, visualizations, and short explanations linking your work to the deliverables. Ensure that all specific requirements defined in each task are documented in the corresponding `README.md` file. For example, the work envelope should be included in `task09_robotics_environment/README.md`.

## Criterion A

This criterion is linked to DataLab Task 9. Justify why you meet this criterion.

### Evidence

- [Link to task09_robotics_environment/README.md]()
- [Link to the code for interfacing with the simulation, sending commands, and processing observations]()
- [GIF of the pipette moving to all 8 corners]()

## Criterion B

This criterion is linked to DataLab Task 10. Justify why you meet this criterion.

### Evidence

- [Link to task10_pid_controller/README.md]()
- [Link to the PID controller implementation code]()
- [Link to the PID controller tuning & testing code]()
- [GIF(s) of robot moving to target coordinates controlled by the PID controller]()
- [Link to presentation slides showing tuning results and final gains]()

## Criterion C

This criterion is linked to DataLab Task 11. Justify why you meet this criterion.

### Evidence

- [Link to task11_rl_controller/README.md]()
- [Link to ot2_gym_wrapper.py]()
- [Link to test_wrapper.py]()
- [Link to RL training code]()
- [Link to RL testing code]()
- [Link to the best individual model weights]()
- [Link to the best group model weights]()
- [GIF(s) of robot moving to target coordinates controlled by the RL agent]()
- [Link to presentation slides with controller performance visualizations and documenting hyperparameter search results]()

## Criterion D

This criterion is linked to DataLab Tasks 12 and 13. Justify why you meet this criterion.

### Evidence

- [Link to task12_13_integration_benchmarking/README.md]()
- [Link to integrated controller code PID]()
- [Link to integrated controller code RL]()
- [GIF(s) of autonomous root tip inoculation with both controllers]()
- [Link to the benchmarking code]()
- [Link to the presentation with benchmarking results]()