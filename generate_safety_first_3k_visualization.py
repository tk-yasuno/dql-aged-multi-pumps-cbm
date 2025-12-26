#!/usr/bin/env python3
"""
Safety-First 3000-Episode Training Results Visualization
Outputs: run_3K_episodes/outputs_pump_cbm_v047_safety_first/safety_first_training_results.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os

def create_safety_first_training_visualization():
    """Create comprehensive training results visualization for Safety-First strategy"""
    
    # Create output directory structure
    output_dir = Path("run_3K_episodes/outputs_pump_cbm_v047_safety_first")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    input_path = "outputs_pump_cbm_v047_safety_first/training_history.json"
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        rewards = data.get('episode_rewards', [])
        if len(rewards) == 0:
            raise ValueError("No episode rewards found in training data")
            
        print(f"[SUCCESS] Loaded Safety-First training data: {len(rewards)} episodes")
        
    except FileNotFoundError:
        print(f"❌ Training data not found: {input_path}")
        return False
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return False
    
    # Set up the visualization
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Safety-First Strategy: 3000-Episode Training Results', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust spacing to prevent title overlap
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    episodes = list(range(1, len(rewards) + 1))
    color_primary = '#1f77b4'
    color_secondary = '#ff7f0e'
    
    # Plot 1: Raw and Smoothed Training Progress
    axes[0, 0].plot(episodes, rewards, alpha=0.3, color=color_primary, linewidth=0.8, label='Raw Rewards')
    
    # Multiple smoothing windows
    for window, alpha, linewidth in [(50, 0.8, 2), (100, 0.9, 2.5), (200, 1.0, 3)]:
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smooth_episodes = episodes[window-1:]
            axes[0, 0].plot(smooth_episodes, smoothed, 
                           alpha=alpha, linewidth=linewidth, 
                           label=f'MA-{window}')
    
    axes[0, 0].set_title('Training Progress with Multi-Scale Smoothing', fontweight='bold')
    axes[0, 0].set_xlabel('Episodes')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add performance annotations
    max_reward = max(rewards)
    max_episode = episodes[rewards.index(max_reward)]
    axes[0, 0].annotate(f'Peak: {max_reward:.0f}', 
                       xy=(max_episode, max_reward), 
                       xytext=(max_episode + len(episodes)*0.1, max_reward + 200),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       fontsize=10, fontweight='bold')
    
    # Plot 2: Performance Distribution by Training Phases
    n_phases = 6
    phase_size = len(rewards) // n_phases
    phase_names = [f'Phase {i+1}\n({i*phase_size+1}-{min((i+1)*phase_size, len(rewards))}ep)' 
                   for i in range(n_phases)]
    phase_data = []
    phase_means = []
    
    for i in range(n_phases):
        start_idx = i * phase_size
        end_idx = min((i+1) * phase_size, len(rewards))
        phase_rewards = rewards[start_idx:end_idx]
        phase_data.append(phase_rewards)
        phase_means.append(np.mean(phase_rewards))
    
    bp = axes[0, 1].boxplot(phase_data, labels=range(1, n_phases+1), patch_artist=True)
    for patch, mean in zip(bp['boxes'], phase_means):
        patch.set_facecolor(color_primary)
        patch.set_alpha(0.7)
    
    axes[0, 1].plot(range(1, n_phases+1), phase_means, 'ro-', linewidth=2, markersize=6, label='Mean')
    axes[0, 1].set_title('Performance Evolution by Training Phases', fontweight='bold')
    axes[0, 1].set_xlabel('Training Phase')
    axes[0, 1].set_ylabel('Episode Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Performance Analysis
    cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    final_avg = cumulative_avg[-1]
    
    axes[0, 2].plot(episodes, cumulative_avg, color=color_primary, linewidth=2.5, label='Cumulative Average')
    axes[0, 2].axhline(y=final_avg, color=color_secondary, linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'Final Avg: {final_avg:.0f}')
    
    # Add convergence analysis
    convergence_threshold = final_avg * 0.95
    convergence_episode = next((i for i, val in enumerate(cumulative_avg) 
                               if val >= convergence_threshold), len(episodes))
    
    axes[0, 2].axvline(x=convergence_episode, color='red', linestyle=':', 
                       alpha=0.8, label=f'95% Convergence: Ep {convergence_episode}')
    
    axes[0, 2].set_title('Cumulative Average & Convergence Analysis', fontweight='bold')
    axes[0, 2].set_xlabel('Episodes')
    axes[0, 2].set_ylabel('Cumulative Average Reward')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Learning Stability Analysis (Rolling Statistics)
    window = 100
    rolling_means = []
    rolling_stds = []
    rolling_episodes = []
    
    for i in range(window, len(rewards)):
        window_rewards = rewards[i-window:i]
        rolling_means.append(np.mean(window_rewards))
        rolling_stds.append(np.std(window_rewards))
        rolling_episodes.append(i)
    
    axes[1, 0].plot(rolling_episodes, rolling_means, color=color_primary, linewidth=2, label='Rolling Mean')
    axes[1, 0].fill_between(rolling_episodes,
                           np.array(rolling_means) - np.array(rolling_stds),
                           np.array(rolling_means) + np.array(rolling_stds),
                           alpha=0.3, color=color_primary, label='±1 Std Dev')
    
    # Stability metric (coefficient of variation)
    cv = np.array(rolling_stds) / np.array(rolling_means) * 100
    stability_score = 100 - np.mean(cv[-100:])  # Last 100 windows
    
    axes[1, 0].set_title(f'Learning Stability (Stability Score: {stability_score:.1f}/100)', fontweight='bold')
    axes[1, 0].set_xlabel('Episodes')
    axes[1, 0].set_ylabel('Reward Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Performance Improvement Rate
    window = 200
    improvement_rates = []
    improvement_episodes = []
    
    for i in range(window, len(rewards), 25):  # Every 25 episodes
        x = np.arange(window)
        y = rewards[i-window:i]
        slope = np.polyfit(x, y, 1)[0]  # Linear trend slope
        improvement_rates.append(slope)
        improvement_episodes.append(i)
    
    axes[1, 1].plot(improvement_episodes, improvement_rates, color=color_primary, 
                   linewidth=2, marker='o', markersize=4)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
    axes[1, 1].set_title('Learning Rate (Performance Improvement Slope)', fontweight='bold')
    axes[1, 1].set_xlabel('Episodes')
    axes[1, 1].set_ylabel('Improvement Rate (Reward/Episode)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Key Performance Metrics Summary
    metrics = {
        'Final Performance\n(Last 100 avg)': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        'Peak Performance': max(rewards),
        'Overall Average': np.mean(rewards),
        'Stability Score': stability_score,
        'Improvement Rate\n(Final 500 episodes)': np.mean(improvement_rates[-20:]) if improvement_rates else 0
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = axes[1, 2].barh(metric_names, metric_values, color=color_primary, alpha=0.7)
    axes[1, 2].set_title('Key Performance Metrics Summary', fontweight='bold')
    axes[1, 2].set_xlabel('Score/Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        width = bar.get_width()
        axes[1, 2].text(width + max(metric_values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='left', va='center', fontweight='bold')
    
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / "safety_first_training_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"🎨 Safety-First training results saved: {output_path}")
    print(f"📊 Performance Summary:")
    final_perf_key = 'Final Performance\n(Last 100 avg)'
    print(f"   Final Performance (last 100): {metrics[final_perf_key]:.2f}")
    print(f"   Peak Performance: {metrics['Peak Performance']:.2f}")
    print(f"   Stability Score: {metrics['Stability Score']:.1f}/100")
    print(f"   Convergence Episode (95%): {convergence_episode}")
    
    plt.show()
    return True

if __name__ == "__main__":
    print("[INFO] Generating Safety-First 3000-Episode Training Results Visualization...")
    success = create_safety_first_training_visualization()
    if success:
        print("[SUCCESS] Visualization generation completed successfully!")
    else:
        print("[ERROR] Visualization generation failed!")