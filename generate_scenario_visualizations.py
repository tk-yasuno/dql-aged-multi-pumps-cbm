"""
Generate individual scenario visualizations for pump CBM v0.4.7
Creates detailed training progress plots for each scenario
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_scenario_visualization(scenario_dir, scenario_name, color_scheme):
    """Create visualization for a single scenario"""
    
    # Load data
    data_file = Path(scenario_dir) / "training_history.json"
    if not data_file.exists():
        print(f"❌ No training data found for {scenario_name}")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    episode_rewards = data['episode_rewards']
    episode_costs = data['episode_costs'] 
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{scenario_name} Strategy - Training Results ({len(episodes)} Episodes)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Reward Progress
    ax1.plot(episodes, episode_rewards, alpha=0.6, color=color_scheme['primary'], linewidth=0.8)
    if len(episode_rewards) >= 50:
        smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
        ax1.plot(episodes[49:], smoothed, color=color_scheme['smooth'], linewidth=2.5, 
                label='Moving Average (50ep)')
    ax1.set_title('Reward Progress', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Final: {episode_rewards[-1]:.1f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor=color_scheme['box']), 
             va='top', fontweight='bold')
    
    # 2. Cost Progress
    ax2.plot(episodes, episode_costs, alpha=0.6, color=color_scheme['cost'], linewidth=0.8)
    if len(episode_costs) >= 50:
        smoothed_cost = np.convolve(episode_costs, np.ones(50)/50, mode='valid')
        ax2.plot(episodes[49:], smoothed_cost, color=color_scheme['smooth'], linewidth=2.5, 
                label='Moving Average (50ep)')
    ax2.set_title('Cost Progress', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Total Cost')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Final: {episode_costs[-1]:.0f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor=color_scheme['cost_box']), 
             va='top', fontweight='bold')
    
    # 3. Final Episode Reward Distribution
    final_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    ax3.hist(final_rewards, bins=20, alpha=0.7, color=color_scheme['primary'], edgecolor='black')
    mean_final = np.mean(final_rewards)
    std_final = np.std(final_rewards)
    ax3.axvline(mean_final, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_final:.2f}')
    ax3.axvline(mean_final + std_final, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_final + std_final:.2f}')
    ax3.axvline(mean_final - std_final, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_final - std_final:.2f}')
    ax3.set_title(f'Final {len(final_rewards)} Episodes - Reward Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=10)
    
    # 4. Learning Stability (Moving Standard Deviation)
    window = min(100, len(episode_rewards)//4)
    moving_std = []
    for i in range(len(episode_rewards)):
        start = max(0, i-window)
        window_rewards = episode_rewards[start:i+1]
        moving_std.append(np.std(window_rewards))
    
    ax4.plot(episodes, moving_std, color=color_scheme['stability'], linewidth=2)
    ax4.set_title(f'Learning Stability (Moving Std, window={window})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Reward Standard Deviation')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, f'Final Std: {moving_std[-1]:.1f}', 
             transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor=color_scheme['stability_box']), 
             va='top', fontweight='bold')
    
    # Save plot
    plt.tight_layout()
    output_path = Path(scenario_dir) / f'{scenario_name.lower()}_training_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f'✅ {scenario_name} scenario visualization saved: {output_path.name}')
    print(f'   📊 Final Reward: {episode_rewards[-1]:.2f}')
    print(f'   📊 Average (Last 100): {mean_final:.2f} ± {std_final:.2f}')
    print(f'   💰 Final Cost: {episode_costs[-1]:.0f}')
    print(f'   💰 Average Cost (Last 100): {np.mean(episode_costs[-100:]):.0f}')
    print()

def main():
    """Generate visualizations for all scenarios"""
    
    scenarios = [
        {
            'dir': 'outputs_pump_cbm_v047_balanced',
            'name': 'Balanced',
            'colors': {
                'primary': 'blue',
                'smooth': 'darkblue', 
                'cost': 'green',
                'cost_box': 'lightgreen',
                'stability': 'purple',
                'stability_box': 'plum',
                'box': 'lightblue'
            }
        },
        {
            'dir': 'outputs_pump_cbm_v047_cost_efficient', 
            'name': 'Cost-Efficient',
            'colors': {
                'primary': 'orange',
                'smooth': 'darkorange',
                'cost': 'green', 
                'cost_box': 'lightgreen',
                'stability': 'purple',
                'stability_box': 'plum',
                'box': 'wheat'
            }
        },
        {
            'dir': 'outputs_pump_cbm_v047_safety_first',
            'name': 'Safety-First', 
            'colors': {
                'primary': 'red',
                'smooth': 'darkred',
                'cost': 'green',
                'cost_box': 'lightgreen', 
                'stability': 'purple',
                'stability_box': 'plum',
                'box': 'lightcoral'
            }
        }
    ]
    
    print("🎨 Generating individual scenario visualizations...\n")
    
    for scenario in scenarios:
        create_scenario_visualization(scenario['dir'], scenario['name'], scenario['colors'])
    
    print("✅ All scenario visualizations completed!")
    
    # List generated files
    print("\n📁 Generated visualization files:")
    for scenario in scenarios:
        viz_file = Path(scenario['dir']) / f"{scenario['name'].lower()}_training_results.png"
        if viz_file.exists():
            size_mb = viz_file.stat().st_size / (1024*1024)
            print(f"   - {viz_file}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()