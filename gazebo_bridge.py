
import asyncio
import numpy as np
import torch
import torch.nn as nn
import math
import json
import os
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw)

# --- CONFIGURATION ---
MODEL_PATH = "swarm_training_results/models/model_episode_1600_fixed.pth" 
NUM_DRONES = 3 
START_PORT = 14540
SCALE_FACTOR = 0.2  
MAX_SPEED = 10.0 
SENSOR_RANGE_M = 40.0
TARGETS_FILE = "targets.json"

# --- MODEL DEFINITION (Copied from TrainLoop) ---
class ImprovedA2CAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state):
        features = self.feature_extractor(state)
        actor_output = self.actor(features)
        action_mean = torch.tanh(actor_output)
        return action_mean

def process_observation(observation):
    """
    State reconstruction matching TrainLoop.py logic (45 dims).
    """
    state_dim = 45
    state = torch.zeros(state_dim)

    if not isinstance(observation, dict):
        return state.unsqueeze(0)

    try:
        # [0-8] Basic
        if 'position' in observation:
            state[0] = float(observation['position'][0])
            state[1] = float(observation['position'][1])
        state[2] = float(observation.get('battery', 0.5))
        state[3] = float(observation.get('health', 1.0))
        state[4] = float(observation.get('status', 0.0))
        state[5] = float(observation.get('team', 0.0))
        state[6] = min(float(observation.get('visible_target_count', 0.0)), 1.0)
        state[7] = min(float(observation.get('shared_target_count', 0.0)), 1.0)
        state[8] = float(observation.get('teammate_count', 0.0))

        # [9-23] Best 3 Targets
        visible_targets = observation.get('visible_targets', [])
        if visible_targets:
            # Sort by importance (simple version)
            sorted_targets = sorted(visible_targets, key=lambda t: t.get('distance', 1.0))
            for idx in range(min(3, len(sorted_targets))):
                t = sorted_targets[idx]
                base_idx = 9 + idx * 5
                dist_norm = float(t.get('distance', 1.0))
                state[base_idx] = max(0.0, 1.0 - dist_norm)
                state[base_idx + 1] = float(t.get('importance', 0.0))
                state[base_idx + 2] = float(t.get('direction_x', 0.0))
                state[base_idx + 3] = float(t.get('direction_y', 0.0))
                state[base_idx + 4] = float(t.get('hp', 1.0))

        # [33-34] Spatial Awareness (Simplified)
        pos_x = state[0].item()
        pos_y = state[1].item()
        dist_to_center = np.sqrt((pos_x - 0.5) ** 2 + (pos_y - 0.5) ** 2)
        state[33] = min(dist_to_center, 1.0)
        state[34] = min(pos_x, pos_y, 1.0 - pos_x, 1.0 - pos_y)

    except Exception as e:
        print(f"[PROCESS] Error: {e}")

    return state.unsqueeze(0)

# --- BRIDGE CODE ---
class TargetManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.targets = []
        self.last_mtime = 0

    def update(self):
        try:
            if not os.path.exists(self.filepath): return
            mtime = os.path.getmtime(self.filepath)
            if mtime > self.last_mtime:
                with open(self.filepath, 'r') as f:
                    self.targets = json.load(f)
                self.last_mtime = mtime
                print(f"[TARGETS] Updated: {len(self.targets)} targets.")
        except: pass

    def get_targets(self):
        return self.targets

class DroneAgent:
    def __init__(self, system_id, port):
        self.system_id = system_id
        self.drone = System()
        self.current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.connection_string = f"udp://:{port}"
        self.battery = 1.0

    async def connect(self):
        print(f"[DRONE {self.system_id}] Connecting to {self.connection_string}...")
        await self.drone.connect(system_address=self.connection_string)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"[DRONE {self.system_id}] Connected!")
                break
        asyncio.create_task(self.telemetry_loop())

    async def telemetry_loop(self):
        async for odom in self.drone.telemetry.position_velocity_ned():
            self.current_pos['x'] = odom.position.north_m
            self.current_pos['y'] = odom.position.east_m
            self.current_pos['z'] = -odom.position.down_m

    def construct_observation(self, all_targets):
        norm_x = (self.current_pos['x'] / SCALE_FACTOR) / 1000.0
        norm_y = (self.current_pos['y'] / SCALE_FACTOR) / 1000.0
        
        visible_targets = []
        for t in all_targets:
            dist_x = t['x'] - self.current_pos['x']
            dist_y = t['y'] - self.current_pos['y']
            dist = math.sqrt(dist_x**2 + dist_y**2)
            if dist <= SENSOR_RANGE_M:
                visible_targets.append({
                    'id': t['id'],
                    'type': self._encode_target_type(t.get('type', 'tank')),
                    'distance': dist / SENSOR_RANGE_M,
                    'direction_x': dist_x / SENSOR_RANGE_M,
                    'direction_y': dist_y / SENSOR_RANGE_M,
                    'importance': 1.0, 
                    'hp': t.get('hp', 1.0),
                    'required_drones': 0.3,
                    'attackers': 0.0
                })
        
        return {
            'drone_id': self.system_id,
            'position': [norm_x, norm_y],
            'status': 0.25, 'battery': self.battery, 'health': 1.0,
            'team': (self.system_id % 3) / 3.0,
            'visible_targets': visible_targets,
            'visible_target_count': len(visible_targets) / 10.0,
            'shared_target_count': 0, 'teammate_count': 0
        }

    def _encode_target_type(self, t_type):
        return {'infantry': 0.2, 'radar': 0.4, 'artillery': 0.6, 'aircraft': 0.8, 'tank': 1.0}.get(t_type, 0.0)

    async def execute_action(self, action):
        vel_n = action[0] * MAX_SPEED
        vel_e = action[1] * MAX_SPEED
        vel_d = 5.0 if action[2] > 0.5 else 0.0
        try:
             await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vel_n, vel_e, vel_d, 0.0))
        except: pass

class SwarmBridge:
    def __init__(self):
        self.target_manager = TargetManager(TARGETS_FILE)
        self.agents = []
        self.model = None

        # --- REAL MODEL LOADING ---
        try:
            device = torch.device("cpu")
            # 45 state dim, 4 action dim
            self.model = ImprovedA2CAgent(45, 4).to(device)
            
            if os.path.exists(MODEL_PATH):
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                # Load weights from first agent in swarm
                agent_weights = checkpoint['agents_state_dict'][0]
                self.model.load_state_dict(agent_weights)
                self.model.eval()
                print(f"[SWARM] ✅ Loaded Real Model: {MODEL_PATH}")
            else:
                print(f"[SWARM] ⚠️ Model file not found at {MODEL_PATH}. Using random weights.")
        except Exception as e:
            print(f"[SWARM] ❌ Model load error: {e}")
            self.model = None

    async def start(self):
        for i in range(NUM_DRONES):
            self.agents.append(DroneAgent(i, START_PORT + i))
        
        await asyncio.gather(*[a.connect() for a in self.agents])
        
        print("[SWARM] Arming & Takeoff...")
        for a in self.agents:
            try:
                await a.drone.action.arm()
                await a.drone.action.takeoff()
                await asyncio.sleep(0.5)
            except: pass
        await asyncio.sleep(5)

        for a in self.agents: 
            try: await a.drone.offboard.start() 
            except: pass
        
        await self.control_loop()

    async def control_loop(self):
        print("[SWARM] Control Loop Active")
        while True:
            self.target_manager.update()
            targets = self.target_manager.get_targets()
            
            observations = [a.construct_observation(targets) for a in self.agents]
            actions = self.get_model_actions(observations)
            
            await asyncio.gather(*[a.execute_action(act) for a, act in zip(self.agents, actions)])
            await asyncio.sleep(0.1)

    def get_model_actions(self, observations):
        actions = []
        with torch.no_grad():
            for obs in observations:
                if self.model:
                    state = process_observation(obs)
                    action_mean = self.model(state)
                    # Convert tensor to list: [vx, vy, attack, target (ignored)]
                    # action_mean is [1, 4]
                    act = action_mean[0].tolist() 
                    actions.append(act)
                else:
                    actions.append([0.0, 0.0, 0.0])
        return actions

if __name__ == "__main__":
    bridge = SwarmBridge()
    asyncio.get_event_loop().run_until_complete(bridge.start())
