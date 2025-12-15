
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from Environment import SwarmBattlefield2D
from TrainLoop import HierarchicalSwarmTrainer
from SwarmCoordinator import SwarmCoordinator
import math

def load_model_and_config(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Check if config is empty or just has 'training' keys, we might need default env keys
    if 'env' not in config:
        print("Warning: Config not found in checkpoint or missing 'env' key. Using defaults.")
        config['env'] = {
            'width': 1200,
            'height': 800,
            'num_drones': 8,
            'num_targets': 12,
            'max_steps': 1000
        }
    
    if 'training' not in config:
         print("Warning: 'training' config key missing. Using defaults.")
         config['training'] = {
            'hidden_dim': 256,
            'gamma': 0.99,
            'learning_rate': 0.0001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.995,
            'tau': 0.001,
            'buffer_size': 10000,
            'batch_size': 64
         }
         
    if 'coordination' not in config:
        config['coordination'] = {'enabled': True}

    return checkpoint, config

def run_evaluation_episode(env, trainer, coordinator=None):
    observations = env.reset()
    if coordinator:
        coordinator.reset()
        
    drone_paths = {i: {'x': [], 'y': []} for i in range(env.num_drones)}
    
    # Store initial target positions
    targets_info = [t.copy() for t in env.targets]
    
    hits = []
    done = False
    step_count = 0
    
    print("Running simulation...")
    while not done:
        # Record positions
        for i, drone in enumerate(env.drones):
            drone_paths[i]['x'].append(drone['x'])
            drone_paths[i]['y'].append(drone['y'])
            
        # Get actions
        directives = None
        if coordinator:
            directives = coordinator.get_strategic_actions(observations)
            
        actions = []
        for drone_id, obs in enumerate(observations):
            if obs.get('health', 1.0) <= 0 or obs.get('battery', 0) <= 0:
                actions.append([0.0, 0.0, 0, -1])
                continue
                
            directive = directives[drone_id] if directives else None
            
            # --- GPS INJECTION PATCH ---
            if directive and directive.get('target_id', -1) >= 0:
                target_id = directive['target_id']
                target = next((t for t in env.targets if t['id'] == target_id), None)
                if target:
                    drone = env.drones[drone_id]
                    dx = target['x'] - drone['x']
                    dy = target['y'] - drone['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    obs['target_distance'] = dist / env.height 
                    obs['target_direction_x'] = dx / dist if dist > 0 else 0.0
                    obs['target_direction_y'] = dy / dist if dist > 0 else 0.0
                    
                    env.drones[drone_id]['target_id'] = target_id
            # ---------------------------

            state = trainer._process_observation(obs, directive)
            
            # Deterministic action (epsilon=0)
            action, _, _, _ = trainer.agents[drone_id].get_action(state, epsilon=0.0)
            
            move_x = float(action[0])
            move_y = float(action[1])
            attack = int(action[2])
            target_id = -1
            
            if directive and directive.get('target_id', -1) >= 0:
                target_id = directive['target_id']
                if directive.get('should_attack', False):
                    attack = 1
            else:
                 if attack == 1:
                    visible = obs.get('visible_targets', [])
                    if isinstance(visible, list) and len(visible) > 0:
                        def target_key(t):
                             imp = float(t.get('importance', 0.0))
                             dist = float(t.get('distance', 1.0))
                             return (imp, -(1.0 - dist))
                        best = max(visible, key=target_key)
                        target_id = int(best.get('id', -1))
            
            actions.append([move_x, move_y, attack, target_id])
            
        # Store previous damage stats
        prev_damages = {d['id']: d['total_damage'] for d in env.drones}
        
        observations, rewards, done, info = env.step(actions)
        
        # Check hits
        for d in env.drones:
            if d['total_damage'] > prev_damages.get(d['id'], 0):
                hits.append((d['x'], d['y'], d['id']))
        
        step_count += 1
        
    print(f"Simulation finished in {step_count} steps.")
    
    # Debug Positions
    if len(env.drones) > 3:
        print(f"DEBUG: Drone 0 End Pos: ({env.drones[0]['x']:.1f}, {env.drones[0]['y']:.1f})")
        print(f"DEBUG: Drone 3 End Pos: ({env.drones[3]['x']:.1f}, {env.drones[3]['y']:.1f})")
    
    return drone_paths, targets_info, env.targets, hits

def plot_paths(drone_paths, targets_initial, targets_final, hits, width, height, output_file="path_visualization.png"):
    plt.figure(figsize=(12, 8))
    
    # Set background color
    plt.gca().set_facecolor('#f0f0f0')
    
    # Colors for targets
    target_colors = {
        'tank': 'darkgreen',
        'artillery': 'brown',
        'infantry': 'blue',
        'aircraft': 'gray',
        'radar': 'orange'
    }
    
    # Plot Targets
    for t in targets_initial:
        # Check if destroyed in final state
        final_state = next((tf for tf in targets_final if tf['id'] == t['id']), None)
        is_destroyed = final_state['destroyed'] if final_state else False
        
        color = target_colors.get(t['type'], 'black')
        marker = 'X' if is_destroyed else 'o'
        edgecolor = 'red' if is_destroyed else 'black'
        size = 150 if t['type'] in ['tank', 'aircraft'] else 80
        
        plt.scatter(t['x'], t['y'], c=color, marker=marker, s=size, edgecolors=edgecolor, zorder=2)
        
        # Add ID label
        plt.annotate(f"{t['type'][0].upper()}{t['id']}", (t['x'], t['y']), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot Drone Paths
    cmap = plt.get_cmap('tab10')
    for drone_id, path in drone_paths.items():
        if len(path['x']) == 0: continue
        
        color = cmap(drone_id % 10)
        
        # Plot path line
        plt.plot(path['x'], path['y'], color=color, alpha=0.6, linewidth=1.5, zorder=1)
        
        # Add arrows
        if len(path['x']) > 20:
            for i in range(0, len(path['x']), 20):
                 if i+1 < len(path['x']):
                     plt.arrow(path['x'][i], path['y'][i], 
                               path['x'][i+1]-path['x'][i], path['y'][i+1]-path['y'][i], 
                               shape='full', lw=0, length_includes_head=True, head_width=15, color=color, alpha=0.8)
        
        # Plot start point
        plt.scatter(path['x'][0], path['y'][0], color=color, marker='^', s=50, zorder=3, label=f'Drone {drone_id}')
        
        # Plot end point
        plt.scatter(path['x'][-1], path['y'][-1], color=color, marker='s', s=30, zorder=3)

    # Plot Hits
    if hits:
        hx, hy, _ = zip(*hits)
        plt.scatter(hx, hy, marker='*', s=200, color='gold', edgecolors='black', zorder=4, label='Hit/Damage', alpha=0.9)

    # Create unique legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Tank (Green)', markerfacecolor='darkgreen', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Artillery (Brown)', markerfacecolor='brown', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Infantry (Blue)', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Aircraft (Gray)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Radar (Orange)', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Destroyed', markerfacecolor='black', markeredgecolor='red', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Hit/Damage', markerfacecolor='gold', markeredgecolor='black', markersize=15),
        Line2D([0], [0], color='black', lw=2, label='Drone Path'),
        Line2D([0], [0], marker='^', color='w', label='Start', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='End', markerfacecolor='gray', markersize=10),
    ]
    
    # Place legend in upper right, small font
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small', framealpha=0.9)

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(f"Drone Swarm Trajectories & Hits (Map: {width}x{height})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Drone Paths from Trained Model")
    parser.add_argument("--model", type=str, default="swarm_training_results/models/model_episode_1600.pth", help="Path to model file")
    parser.add_argument("--output", type=str, default="drone_paths.png", help="Output image file base name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (generates separate plots)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load
    checkpoint, config = load_model_and_config(args.model, device)
    
    # Setup Env
    env_config = config['env']
    env = SwarmBattlefield2D(
        width=env_config['width'],
        height=env_config['height'],
        num_drones=env_config['num_drones'],
        num_targets=env_config['num_targets']
    )
    
    # Setup Trainer
    trainer = HierarchicalSwarmTrainer(env, config['training'])
    trainer.agents = torch.nn.ModuleList([
        agent.to(device) for agent in trainer.agents
    ])
    
    # Load state dicts
    for i, state_dict in enumerate(checkpoint['agents_state_dict']):
        trainer.agents[i].load_state_dict(state_dict)
        
    coordinator = None
    if config.get('coordination', {}).get('enabled', False):
        print("Coordination enabled.")
        coordinator = SwarmCoordinator(env)
    
    # Prepare output filename base
    base_name, ext = os.path.splitext(args.output)
    
    # Run loop
    for i in range(args.episodes):
        print(f"\n--- Generating Map {i+1}/{args.episodes} ---")
        drone_paths, targets_init, targets_final, hits = run_evaluation_episode(env, trainer, coordinator)
        
        # Determine unique filename
        if args.episodes > 1:
            current_output = f"{base_name}_{i+1}{ext}"
        else:
            current_output = args.output
            
        # Plot
        plot_paths(drone_paths, targets_init, targets_final, hits, env.width, env.height, current_output)

if __name__ == "__main__":
    main()
