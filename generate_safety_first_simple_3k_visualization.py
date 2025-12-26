#!/usr/bin/env python3
"""
Safety-First Strategy Training Results Visualization (3000 Episodes) - Simple Layout
Same layout as 1000-episode version but with 3000 episodes data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from pathlib import Path

def create_safety_first_simple_visualization():
    """Create simple 4-subplot visualization matching the 1000-episode layout"""
    
    try:
        # Load training data
        data_path = "outputs_pump_cbm_v047_safety_first/training_history.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        rewards = np.array(data['episode_rewards'])
        costs = np.array(data['episode_costs'])
        
        print(f"[SUCCESS] Loaded Safety-First training data: {len(rewards)} episodes")
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Safety-First Strategy - Training Results (3000 Episodes)', fontsize=16, fontweight='bold')
        
        # Moving averages
        window = 50
        reward_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        cost_ma = np.convolve(costs, np.ones(window)/window, mode='valid')
        
        # 1. Reward Progress (top-left)
        ax1.plot(rewards, alpha=0.3, color='lightcoral', linewidth=0.5)
        ax1.plot(range(window-1, len(rewards)), reward_ma, color='darkred', linewidth=2, label=f'Moving Average ({window}ep)')
        ax1.set_title('Reward Progress', fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add final reward annotation
        final_reward = reward_ma[-1] if len(reward_ma) > 0 else rewards[-1]
        ax1.annotate(f'Final: {final_reward:.1f}', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # 2. Cost Progress (top-right)  
        ax2.plot(costs, alpha=0.3, color='lightgreen', linewidth=0.5)
        ax2.plot(range(window-1, len(costs)), cost_ma, color='darkgreen', linewidth=2, label=f'Moving Average ({window}ep)')
        ax2.set_title('Cost Progress', fontweight='bold')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Total Cost')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add final cost annotation
        final_cost = cost_ma[-1] if len(cost_ma) > 0 else costs[-1]
        ax2.annotate(f'Final: {final_cost:.0f}', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # 3. Final 100 Episodes - Reward Distribution (bottom-left)
        final_100_rewards = rewards[-100:]
        mean_reward = np.mean(final_100_rewards)
        std_reward = np.std(final_100_rewards)
        
        ax3.hist(final_100_rewards, bins=15, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=1)
        ax3.axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        ax3.set_title('Final 100 Episodes - Reward Distribution', fontweight='bold')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'Mean: {mean_reward:.2f}\n+1σ: {mean_reward + std_reward:.2f}\n-1σ: {mean_reward - std_reward:.2f}'
        ax3.text(0.75, 0.75, stats_text, transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                fontsize=10, verticalalignment='top')
        
        # 4. Learning Stability (bottom-right)
        # Calculate rolling standard deviation with window=100
        stability_window = 100
        rolling_std = []
        for i in range(stability_window-1, len(rewards)):
            window_data = rewards[i-stability_window+1:i+1]
            rolling_std.append(np.std(window_data))
        
        ax4.plot(range(stability_window-1, len(rewards)), rolling_std, color='purple', linewidth=2)
        ax4.set_title(f'Learning Stability (Moving Std, window={stability_window})', fontweight='bold')
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Reward Standard Deviation')
        ax4.grid(True, alpha=0.3)
        
        # Add final stability annotation
        final_std = rolling_std[-1] if rolling_std else 0
        ax4.annotate(f'Final Std: {final_std:.1f}', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path("run_3K_episodes/outputs_pump_cbm_v047_safety_first")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "safety_first_simple_training_results.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[SUCCESS] Simple visualization saved: {output_path}")
        
        # Performance summary
        print("[PERFORMANCE] Summary:")
        print(f"   Final Performance (last 100): {mean_reward:.2f}")
        print(f"   Peak Performance: {np.max(rewards):.2f}")
        print(f"   Final Stability (Std): {final_std:.1f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"[ERROR] Training data file not found: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error creating visualization: {e}")
        return False

if __name__ == "__main__":
    print("[INFO] Generating Safety-First Simple 3000-Episode Training Results Visualization...")
    success = create_safety_first_simple_visualization()
    if success:
        print("[SUCCESS] Simple visualization generation completed successfully!")
    else:
        print("[ERROR] Simple visualization generation failed!")