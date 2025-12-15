
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def smooth(scalars, weight=0.9):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def generate_plots(df):
    # Set professional style
    # plt.style.use('seaborn-v0_8-paper')
    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['reward'], alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(df['episode'], smooth(df['reward'], 0.95), color='#E63946', linewidth=2.5, label='Smoothed Reward')
    plt.title("Swarm Learning Curve: Average Reward per Episode", fontweight='bold', pad=20)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("presentation_reward.png", dpi=300, bbox_inches='tight')
    print("Saved presentation_reward.png")

    # 2. Success Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], smooth(df['success_rate'], 0.95), color='#2A9D8F', linewidth=2.5)
    plt.fill_between(df['episode'], smooth(df['success_rate'], 0.95), color='#2A9D8F', alpha=0.1)
    plt.title("Mission Success Rate Trend", fontweight='bold', pad=20)
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("presentation_success.png", dpi=300, bbox_inches='tight')
    print("Saved presentation_success.png")

    # 3. Destroyed Targets
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], smooth(df['destroyed_targets'], 0.98), color='#457B9D', linewidth=2.5, label='Enemy Units Destroyed')
    plt.title("Operational Efficiency: Targets Destroyed per Episode", fontweight='bold', pad=20)
    plt.xlabel("Episodes")
    plt.ylabel("Count")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("presentation_efficiency.png", dpi=300, bbox_inches='tight')
    print("Saved presentation_efficiency.png")

if __name__ == "__main__":
    log_path = "swarm_training_results/logs/training_log_ep1600.json"
    print(f"Reading log from {log_path}...")
    df = load_data(log_path)
    generate_plots(df)
