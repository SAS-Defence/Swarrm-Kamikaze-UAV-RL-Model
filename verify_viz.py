
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Visualization import SwarmVisualization
import os

def test_visualization():
    print("Testing Visualization...")
    viz = SwarmVisualization()
    
    # Mock history data
    history = {
        'episode_rewards': np.random.rand(10) * 100,
        'success_rate': np.linspace(0, 100, 10),
        'epsilon': np.linspace(1.0, 0.1, 10),
        'episode_length': np.random.randint(50, 200, 10),
        'tank_rate': [50] * 10,
        'artillery_rate': [30] * 10,
        'infantry_rate': [80] * 10,
        'aircraft_rate': [10] * 10,
        'radar_rate': [60] * 10,
        # Mock loss data
        'loss_drone_0': np.random.rand(10),
        'actor_loss_drone_0': np.random.rand(10),
        'critic_loss_drone_0': np.random.rand(10)
    }
    
    # Test Static Dashboard
    try:
        fig = viz.create_training_dashboard(history, current_episode=10)
        fig.savefig("test_dashboard.png")
        print("✅ Static Dashboard created successfully.")
        plt.close(fig)
    except Exception as e:
        print(f"❌ Static Dashboard Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Interactive Dashboard
    try:
        fig = viz.create_interactive_dashboard(history)
        if fig:
            fig.write_html("test_interactive.html")
            print("✅ Interactive Dashboard created successfully.")
        else:
            print("⚠️ Interactive Dashboard returned None (might be empty history check).")
    except Exception as e:
        print(f"❌ Interactive Dashboard Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()
