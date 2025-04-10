# Enhancing Urban Traffic Flow with Branching Dueling Q-Network Adaptive Signal Control

This repository presents an implementation of the **Branching Dueling Q-Network (BDQ)** for adaptive traffic signal control using the **SUMO** traffic simulation environment.

BDQ addresses the scalability issues in traditional Q-learning algorithms by avoiding the exponential increase in action space as more control dimensions are added. Instead, it decomposes the action space into multiple branches—each responsible for one action dimension—and merges them into a single, interpretable action. In the context of traffic control, each branch represents an individual intersection, enabling efficient learning and decision-making across complex road networks.

![1_MINUTE_INFERENCE-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/703a9949-b93a-436d-9791-b96ad4ea0d00)

---

## Installation
### Prerequisites
- Python >= 3.10
- [SUMO](https://www.eclipse.org/sumo/) 1.22.0+
- The following Python packages:
  - `numpy==2.0.2`
  - `torch==2.6.0`
  - `matplotlib==3.10.1`
  - `sumolib==1.22.0`


### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rl-project.git
   cd rl-project
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Configure the environment:
   - Set simulation and training parameters in `config.py`.
   - Set the correct `map_name` and `n_branches` values in `config.py` (under `ENV`).
   - Ensure the same map name is referenced in `data/cfg/sumo_config.sumocfg`.
3. Specifically set the map name in config.py ENV map_name and n_branches, and the map name in sumo_config.sumocfg.
4. Run the training:
   ```bash
   python main.py
   ```
5. Run testing simulations with cyclic_simulation.py, for regular fixed time simulation, and inference.py, to run a BDQ trained controler.

## Branching Dueling Q-Network
Traditional Q-learning struggles with multi-dimensional discrete action spaces due to exponential growth. BDQ addresses this by:

- Decomposing the action space into separate branches, each handling one action dimension.

- Learning value and advantage functions per branch.

- Merging the outputs into a joint action that represents the overall policy.

In this project, each branch controls one traffic light intersection, enabling scalable multi-intersection learning without inflating the action space.

### State Representation

The state vector captures real-time traffic conditions at the intersection and is defined as:

- **Vehicle Queue Lengths**: The number of halted vehicles per lane, aggregated across all incoming lanes  
  \( L = \{l_1, l_2, \ldots, l_n\} \).

---

### Action Space

The action space \( A \) defines the set of admissible traffic light phases:

- **Phase 0 (North-South)**: Green signal for straight and right-turning movements  
- **Phase 1 (East-West)**: Green signal for straight and right-turning movements

These phases are subject to the following operational constraints to ensure compliance with traffic engineering standards:

- A **minimum green time duration** of **10 seconds**.  
- A **minimum yellow time duration** of **3 seconds**.


## Results
### Fixed-Time vs BDQ Control  
**Table comparing the queue length and waiting time of the fixed-time controller and BDQ for different numbers of traffic signals.**  

| Nº Intersections | Control Strategy | Queue Length | Waiting Time (s) | Throughput |
|------------------|------------------|--------------|------------------|------------|
| 1                | Fixed-Time       | 0.95         | 14.02            | 329        |
| 1                | BDQ              | 0.46         | 2.83             | 346        |
| 4                | Fixed-Time       | 2.71         | 42.21            | 289        |
| 4                | BDQ              | 0.97         | 6.74             | 310        |
| 9                | Fixed-Time       | 5.59         | 87.33            | 332        |
| 9                | BDQ              | 1.63         | 10.16            | 366        |

At nine intersections, Fixed-Time control records an average queue length of 5.59 and a
waiting time of 87.33 seconds. With BDQ, the queue length drops to 1.63, reflecting a 70.8%
improvement, while the waiting time is reduced to 10.16 seconds, achieving an 88.4% im-
provement. Throughput increases from 332 to 366, a 10.24% improvement. This substantial
reduction in both queue length and waiting time in a larger, more complex network reinforces
the potential of the BDQ approach to significantly improve traffic flow and reduce delays.

---

[Watch this 1-minute video](https://www.youtube.com/watch?v=UFfd9yIV97k&ab_channel=Baki) comparing traditional fixed-time traffic signal control with the Branching Dueling Q-Network (BDQ) approach. The video highlights the differences in traffic flow efficiency and responsiveness between the two methods.


## License
This project is licensed under the MIT License.
