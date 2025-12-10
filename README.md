# Angajman-RL: Autonomous Drone Swarm Kamikaze Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MARL-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🛸 Project Overview

**Angajman-RL** is an advanced Multi-Agent Reinforcement Learning (MARL) project designed to simulate and train a swarm of autonomous drones for coordinated kamikaze attacks. The project utilizes a hierarchical control system combining a global **Swarm Coordinator** for high-level strategy and individual **RL Agents** for low-level tactical execution.

The goal is to optimize the swarm's ability to neutralize high-value targets (Anti-Air Defense Systems - HSS) while navigating a hostile environment filled with dynamic threats.

## ✨ Key Features

*   **Hierarchical Control Architecture**:
    *   **Swarm Coordinator**: A rule-based strategic layer that assigns targets, manages formation, and prioritizes threats based on the battlefield state.
    *   **RL Agents (Hybrid A2C/PPO)**: Neural network-based agents that control individual drone navigation and engagement maneuvers.
*   **Dynamic Curriculum Learning ("Bootcamp")**:
    *   **Stage 1: Basic Maneuvers**: Drones learn to fly and avoid collisions.
    *   **Stage 2: Static Target Engagement**: Attacking undefended, stationary targets.
    *   **Stage 3: Moving Targets**: Engaging mobile enemy units.
    *   **Stage 4: Full Combat**: Complex scenarios with active enemy defenses and coordinated strikes.
*   **Strategic Coordination**:
    *   **Focus-to-Kill**: Concentrating swarm firepower on high-HP targets to overwhelm defenses.
    *   **Opportunistic Kill**: Diverting free agents to eliminate low-health or isolated targets.
    *   **Wingman Formation**: Encouraging drones to fly in dense packs for mutual support.

---

## 🧠 Technical Architecture

### 1. The Algorithm: PPO (Proximal Policy Optimization)
The project implements a custom PPO agent with a **Shared Feature Extractor** architecture. This ensures efficient feature learning while maintaining separate policy (Actor) and value (Critic) heads.

*   **Network Structure**:
    *   **Input**: 45-Dimensional State Vector.
    *   **Feature Extractor**: MLP (Linear -> LayerNorm -> ReLU -> Dropout) x 2.
    *   **Actor Head**: Outputs action distribution means (tanh) and standard deviations.
    *   **Critic Head**: Outputs value estimates ($V(s)$) for advantage estimation.
*   **Optimization**:
    *   **Loss Function**: Clipped Surrogate Objective ($L^{CLIP}$) to prevent destructive policy updates.
    *   **Advantage Estimation**: Generalized Advantage Estimation (GAE) for stable learning.
    *   **Entropy Bonus**: Encourages exploration in the early stages.

### 2. Observation Space (45-Dimensions)
Each drone perceives an Agent-Centric Local View combined with Global Strategic Directives.

| Index | Category | Description |
| :--- | :--- | :--- |
| **0-5** | **Self State** | Position (x,y), Battery, Health, Status (Free/Engaged), Team ID. |
| **6-8** | **Sensors** | Count of Visible Targets, Shared Targets (via Comms), and Nearby Teammates. |
| **9-23** | **Target Data** | Detailed info for the **Best 3 Targets** (Distance, Importance, Relative Direction X/Y, HP). |
| **24-29** | **Swarm Awareness** | Info on the Closest Teammate (Distance, Direction, Avg Battery, Avg Health) for flocking behavior. |
| **30-34** | **Spatial** | Attack density, Mission Progress, Dist. to Center/Edges. |
| **35-44** | **Coordination** | **(Crucial)** Assigned Target ID, Priority, Global GPS Vector to Target, Formation Role, Attack Command. |

### 3. Action Space (Continuous)
The agents output a 4-dimensional vector:
1.  **Move X** $[-1, 1]$: Horizontal acceleration/thrust.
2.  **Move Y** $[-1, 1]$: Vertical acceleration/thrust.
3.  **Attack Trigger** $[0, 1]$: Sigmoid output; fires main weapon if $> 0.5$.
4.  **Target Selection**: (Internal) While the network suggests target IDs, the Swarm Coordinator overrides this in the hierarchical setup to ensure strategic compliance.

### 4. Engineered Reward Function
The reward function is meticulously shaped to balance individual survival with collective mission success.

| Behavior | Reward / Penalty | Note |
| :--- | :--- | :--- |
| **Destroy High-Value Target** | **+15.0** (scaled) | Main objective (e.g., Tank destruction). |
| **Damage Enemy** | **+0.5** | Encourages engagement even if no kill is secured. |
| **Navigation (GPS)** | **+5.0 * delta** | Heavy incentive to move towards the *assigned* coordinate. |
| **Wingman Bonus** | **+0.05** / step | Reward for flying close to a squadmate sharing the same target. |
| **Keşif (Discovery)** | **+2.0** | Bonus for spotting a previously hidden enemy. |
| **Death/Collision** | **-0.15** | Small penalty. Kamikaze missions prioritize damage over survival. |
| **Idleness** | **-0.05** | Penalty for hovering without moving. |

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | **Entry Point**. Configures the simulation, initializes the `SwarmTrainingManager`, and runs the training loop. |
| `Environment.py` | Contains the `SwarmBattlefield2D` class. Defines the physics, game logic, reward calculation, and state observation space. |
| `SwarmCoordinator.py` | Implements the **SwarmCoordinator** logic. Handles target prioritization, drone allocation algorithms, and mission directives. |
| `TrainLoop.py` | Hosts the `HierarchicalSwarmTrainer` and the `ImprovedA2CAgent`. Manages the neural network updates, experience collection, and backpropagation. |
| `Visualization.py` | Provides real-time rendering tools, creating the battle map and performance dashboards (saved as images). |

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed. The project relies on standard data science and ML libraries.

```bash
# Clone the repository
git clone https://github.com/YakupTekeli/Angajman-RL.git
cd Angajman-RL

# Install dependencies
pip install numpy torch matplotlib pandas
```

## 🚀 Usage

To start the training simulation with the default configuration:

```bash
python main.py
```

### Configuration
You can modify simulation parameters directly in `main.py`. Key parameters include:
*   `num_drones`: Size of the swarm (default: 30).
*   `num_targets`: Number of enemies (default: 15).
*   `difficulty`: Environment difficulty level (0=Easy, 1=Medium, 2=Hard).

## 📊 Visualization

The training process generates real-time visualizations in the `swarm_training_results` directory:
*   **Battle Map**: Top-down view of the swarm and enemies.
*   **Dashboards**: Graphs showing win rate, average reward, and loss over time.

---
*Developed by Yakup Tekeli*
