"""
Multi-Equipment CBM Results Visualization for MVP v0.4

Visualizes:
1. Training reward progression
2. Cost leveling performance (monthly cost variance)
3. Per-equipment condition analysis
4. Cost distribution analysis
5. Maintenance strategy optimization
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
import yaml

# 日本語フォント設定（Windows対応 - 警告完全抑制）
import matplotlib
import warnings
import logging
# 全ての警告を抑制
warnings.filterwarnings('ignore')
# matplotlib のログレベルを ERROR に設定
matplotlib.set_loglevel('ERROR')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# Windows確実対応フォント設定（存在確認不要な汎用設定）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_training_results(results_dir: str = "outputs_pump_cbm_v047_enhanced"):
    """Load training results from output directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
        
    history_file = results_path / "training_history.json"
    
    if not history_file.exists():
        print(f"❌ Training history not found: {history_file}")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print(f"✅ Loaded training results: {len(history['episode_rewards'])} episodes")
    return history


def plot_training_progress(history, save_dir="outputs_pump_cbm_v047_enhanced"):
    """Plot training progress with cost leveling metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Equipment CBM Training Progress (MVP v0.4)', fontsize=16, fontweight='bold')
    
    episodes = range(len(history['episode_rewards']))
    
    # 1. Episode Rewards
    axes[0, 0].plot(episodes, history['episode_rewards'], alpha=0.7, color='blue')
    # Moving average
    window = 50
    if len(history['episode_rewards']) >= window:
        moving_avg = pd.Series(history['episode_rewards']).rolling(window).mean()
        axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2, label=f'{window}-ep Moving Avg')
        axes[0, 0].legend()
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cost Variance (Key metric for MVP v0.4)
    if 'episode_cost_variances' in history:
        axes[0, 1].plot(episodes, history['episode_cost_variances'], alpha=0.7, color='orange')
        if len(history['episode_cost_variances']) >= window:
            cost_var_ma = pd.Series(history['episode_cost_variances']).rolling(window).mean()
            axes[0, 1].plot(episodes, cost_var_ma, color='red', linewidth=2, label=f'{window}-ep Moving Avg')
            axes[0, 1].legend()
        
        # Add target threshold line if available
        if 'config' in history and 'reward' in history['config']:
            threshold = history['config']['reward']['cost_leveling'].get('variance_threshold', 25.0)
            axes[0, 1].axhline(y=threshold, color='green', linestyle='--', 
                             label=f'Target Threshold ({threshold})')
            axes[0, 1].legend()
    
    axes[0, 1].set_title('💡 Cost Variance (Cost Leveling)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cost Variance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Monthly Costs
    if 'episode_costs' in history:
        axes[1, 0].plot(episodes, history['episode_costs'], alpha=0.7, color='green')
        if len(history['episode_costs']) >= window:
            cost_ma = pd.Series(history['episode_costs']).rolling(window).mean()
            axes[1, 0].plot(episodes, cost_ma, color='red', linewidth=2, label=f'{window}-ep Moving Avg')
            axes[1, 0].legend()
        
        # Add target budget line if available
        if 'config' in history and 'reward' in history['config']:
            target_budget = history['config']['reward']['cost_leveling'].get('target_monthly_budget', 50.0)
            axes[1, 0].axhline(y=target_budget, color='purple', linestyle='--', 
                             label=f'Target Budget ({target_budget})')
            axes[1, 0].legend()
    
    axes[1, 0].set_title('Monthly Maintenance Costs')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cost')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Training Loss
    if 'loss_history' in history and history['loss_history']:
        loss_episodes = range(len(history['loss_history']))
        axes[1, 1].plot(loss_episodes, history['loss_history'], alpha=0.7, color='purple')
        
        if len(history['loss_history']) >= window:
            loss_ma = pd.Series(history['loss_history']).rolling(window).mean()
            axes[1, 1].plot(loss_episodes, loss_ma, color='red', linewidth=2, label=f'{window}-ep Moving Avg')
            axes[1, 1].legend()
    
    axes[1, 1].set_title('Training Loss (QR-DQN)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / "training_progress_v04.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training progress plot saved: {save_path}")
    
    plt.show()


def plot_cost_risk_tradeoff(history, save_dir="outputs_pump_cbm_v047_enhanced"):
    """
    Cost vs Risk Tradeoff Analysis
    設備保全におけるコストとリスクのトレードオフを可視化
    """
    
    if 'episode_rewards' not in history or 'episode_costs' not in history:
        print("❌ Cost or reward data not available for tradeoff analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('💰⚠️ Cost vs Risk Tradeoff Analysis (MVP v0.4.2)', fontsize=16, fontweight='bold')
    
    # Data preparation
    rewards = np.array(history['episode_rewards'])
    costs = np.array(history['episode_costs'])
    
    # Calculate risk metrics (negative rewards indicate higher risk)
    risk_scores = -rewards  # Higher positive value = higher risk
    
    # 1. Cost vs Risk Scatter Plot
    episodes = range(len(rewards))
    
    # Color by episode progression
    scatter = axes[0, 0].scatter(costs, risk_scores, c=episodes, cmap='viridis', 
                                alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    
    # Add trend line
    if len(costs) > 1:
        try:
            # Check if data has sufficient variation
            cost_std = np.std(costs)
            risk_std = np.std(risk_scores)
            
            if cost_std > 1e-6 and risk_std > 1e-6:  # Sufficient variation
                z = np.polyfit(costs, risk_scores, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(costs, p(costs), "r--", alpha=0.8, linewidth=2, 
                               label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
                axes[0, 0].legend()
            else:
                print("⚠️ Warning: Insufficient data variation for trend line")
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"⚠️ Warning: Could not compute trend line: {e}")
            # Continue without trend line
    
    axes[0, 0].set_title('💰 Cost vs Risk Relationship')
    axes[0, 0].set_xlabel('Monthly Maintenance Cost')
    axes[0, 0].set_ylabel('Risk Score (Higher = More Risk)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0, 0])
    cbar.set_label('Episode Progression')
    
    # Add correlation info
    try:
        correlation = np.corrcoef(costs, risk_scores)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
            correlation_text = 'No correlation (insufficient variation)'
        else:
            correlation_text = ('Positive: Higher cost → Higher risk' if correlation > 0.1 else
                               'Negative: Higher cost → Lower risk' if correlation < -0.1 else
                               'Weak correlation')
    except (ValueError, np.linalg.LinAlgError):
        correlation = 0.0
        correlation_text = 'Cannot compute correlation'
    
    axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}\\n{correlation_text}', 
                   transform=axes[0, 0].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top')
    
    # 2. Pareto Efficiency Analysis
    # Create efficiency frontier
    def is_pareto_efficient(costs, risks):
        """Find Pareto efficient points (minimize both cost and risk)"""
        points = np.column_stack((costs, risks))
        pareto_front = []
        
        for i, point in enumerate(points):
            dominated = False
            for other_point in points:
                if (other_point[0] <= point[0] and other_point[1] <= point[1] and 
                    (other_point[0] < point[0] or other_point[1] < point[1])):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((i, point))
        
        return pareto_front
    
    pareto_points = is_pareto_efficient(costs, risk_scores)
    pareto_indices = [p[0] for p in pareto_points]
    pareto_costs = [costs[i] for i in pareto_indices]
    pareto_risks = [risk_scores[i] for i in pareto_indices]
    
    # Plot all points
    axes[0, 1].scatter(costs, risk_scores, alpha=0.4, c='lightblue', s=20, label='All Episodes')
    
    # Highlight Pareto frontier
    if pareto_points:
        axes[0, 1].scatter(pareto_costs, pareto_risks, c='red', s=50, 
                          marker='*', label='Pareto Efficient', edgecolor='black')
        
        # Connect Pareto points
        sorted_pareto = sorted(zip(pareto_costs, pareto_risks))
        if len(sorted_pareto) > 1:
            px, py = zip(*sorted_pareto)
            axes[0, 1].plot(px, py, 'r--', alpha=0.7, linewidth=2, label='Pareto Frontier')
    
    axes[0, 1].set_title('⚠️ Pareto Efficiency Frontier')
    axes[0, 1].set_xlabel('Monthly Maintenance Cost')
    axes[0, 1].set_ylabel('Risk Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Action Strategy Analysis
    if 'actions' in history:
        actions = history['actions']
        
        # Decode actions (3^6 = 729 combinations for 6 equipment)
        action_counts = {'do_nothing': 0, 'repair': 0, 'replace': 0}
        
        for episode_actions in actions:
            for action in episode_actions:
                # Decode 3-ary action to individual equipment actions
                individual_actions = []
                temp_action = action
                for i in range(6):  # 6 equipment
                    individual_actions.append(temp_action % 3)
                    temp_action //= 3
                
                for ind_action in individual_actions:
                    if ind_action == 0:
                        action_counts['do_nothing'] += 1
                    elif ind_action == 1:
                        action_counts['repair'] += 1
                    elif ind_action == 2:
                        action_counts['replace'] += 1
        
        # Create pie chart
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        colors = ['lightgreen', 'yellow', 'red']
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                 explode=explode, shadow=True, startangle=90)
        
        axes[1, 0].set_title('🛠️ Maintenance Strategy Distribution')
        
        # Add strategy assessment
        total_actions = sum(sizes)
        repair_ratio = sizes[1] / total_actions if total_actions > 0 else 0
        replace_ratio = sizes[2] / total_actions if total_actions > 0 else 0
        
        strategy_text = ""
        if repair_ratio > 0.3:
            strategy_text = "Preventive Strategy"
        elif replace_ratio > 0.3:
            strategy_text = "Reactive Strategy"
        else:
            strategy_text = "Balanced Strategy"
            
        axes[1, 0].text(0, -1.3, f'Dominant Strategy: {strategy_text}', 
                       ha='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 4. Cost-Risk Efficiency Over Time
    # Calculate efficiency metric: minimize (normalized_cost + normalized_risk)
    cost_range = np.max(costs) - np.min(costs)
    risk_range = np.max(risk_scores) - np.min(risk_scores)
    
    # Handle cases with zero range (all values are the same)
    if cost_range > 1e-6:
        normalized_costs = (costs - np.min(costs)) / cost_range
    else:
        normalized_costs = np.zeros_like(costs)
        
    if risk_range > 1e-6:
        normalized_risks = (risk_scores - np.min(risk_scores)) / risk_range
    else:
        normalized_risks = np.zeros_like(risk_scores)
    
    efficiency_scores = -(normalized_costs + normalized_risks)  # Higher is better
    
    axes[1, 1].plot(episodes, efficiency_scores, alpha=0.7, color='purple', linewidth=1.5)
    
    # Moving average
    window = min(20, len(efficiency_scores) // 4)
    if len(efficiency_scores) >= window:
        efficiency_ma = pd.Series(efficiency_scores).rolling(window).mean()
        axes[1, 1].plot(episodes, efficiency_ma, color='red', linewidth=3, 
                       label=f'{window}-episode Moving Average')
        axes[1, 1].legend()
    
    axes[1, 1].set_title('📈 Cost-Risk Efficiency Over Time')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Efficiency Score (Higher = Better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add improvement trend
    if len(efficiency_scores) > 10:
        early_avg = np.mean(efficiency_scores[:len(efficiency_scores)//3])
        late_avg = np.mean(efficiency_scores[-len(efficiency_scores)//3:])
        improvement = late_avg - early_avg
        
        improvement_text = f"Improvement: {improvement:+.3f}"
        color = 'green' if improvement > 0 else 'red'
        axes[1, 1].text(0.05, 0.95, improvement_text, 
                       transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                       verticalalignment='top')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / "cost_risk_tradeoff_analysis_v04.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cost-Risk Tradeoff analysis plot saved: {save_path}")
    
    # Generate detailed report
    report_path = Path(save_dir) / "cost_risk_tradeoff_report_v04.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Cost vs Risk Tradeoff Analysis Report\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Total Episodes Analyzed: {len(rewards):,}\\n")
        
        try:
            f.write(f"Cost-Risk Correlation: {correlation:.3f}\\n")
        except:
            f.write("Cost-Risk Correlation: Unable to compute\\n")
            
        f.write(f"Number of Pareto Efficient Points: {len(pareto_points)}\\n")
        
        if 'actions' in history:
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                f.write(f"\\nMaintenance Strategy Distribution:\\n")
                f.write(f"  Do Nothing: {action_counts['do_nothing']:,} ({100*action_counts['do_nothing']/total_actions:.1f}%)\\n")
                f.write(f"  Repair: {action_counts['repair']:,} ({100*action_counts['repair']/total_actions:.1f}%)\\n")
                f.write(f"  Replace: {action_counts['replace']:,} ({100*action_counts['replace']/total_actions:.1f}%)\\n")
        
        f.write(f"\\nEfficiency Metrics:\\n")
        f.write(f"  Final Efficiency Score: {efficiency_scores[-1]:.4f}\\n")
        if len(efficiency_scores) >= 5:
            f.write(f"  Average Efficiency (Last 20%): {np.mean(efficiency_scores[-len(efficiency_scores)//5:]):.4f}\\n")
        else:
            f.write(f"  Average Efficiency (All episodes): {np.mean(efficiency_scores):.4f}\\n")
    
    print(f"✅ Cost-Risk Tradeoff report saved: {report_path}")
    
    plt.show()


def plot_cost_leveling_analysis(history, save_dir="outputs_pump_cbm_v047_enhanced"):
    """Detailed cost leveling analysis"""
    
    if 'episode_costs' not in history or 'episode_cost_variances' not in history:
        print("❌ Cost data not available for analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cost Leveling Analysis (MVP v0.4 Key Feature)', fontsize=16, fontweight='bold')
    
    costs = np.array(history['episode_costs'])
    cost_variances = np.array(history['episode_cost_variances'])
    
    # 1. Cost Distribution
    axes[0, 0].hist(costs, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(np.mean(costs), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(costs):.2f}')
    axes[0, 0].axvline(np.median(costs), color='blue', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(costs):.2f}')
    
    # Target budget line
    if 'config' in history:
        target_budget = history['config']['reward']['cost_leveling'].get('target_monthly_budget', 50.0)
        axes[0, 0].axvline(target_budget, color='purple', linestyle=':', linewidth=2, 
                          label=f'Target: {target_budget}')
    
    axes[0, 0].set_title('Monthly Cost Distribution')
    axes[0, 0].set_xlabel('Cost')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Variance Distribution
    axes[0, 1].hist(cost_variances, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(np.mean(cost_variances), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(cost_variances):.2f}')
    
    # Variance threshold line
    if 'config' in history:
        threshold = history['config']['reward']['cost_leveling'].get('variance_threshold', 25.0)
        axes[0, 1].axvline(threshold, color='green', linestyle=':', linewidth=2, 
                          label=f'Threshold: {threshold}')
    
    axes[0, 1].set_title('Cost Variance Distribution')
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cost vs Variance Scatter
    axes[1, 0].scatter(costs, cost_variances, alpha=0.5, s=10)
    axes[1, 0].set_title('Cost vs Variance Relationship')
    axes[1, 0].set_xlabel('Monthly Cost')
    axes[1, 0].set_ylabel('Cost Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(costs, cost_variances)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 0].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Cost improvement over time
    episodes = range(len(costs))
    
    # Split into early and late episodes
    split_point = len(costs) // 2
    early_costs = costs[:split_point]
    late_costs = costs[split_point:]
    early_variances = cost_variances[:split_point]
    late_variances = cost_variances[split_point:]
    
    improvement_data = {
        'Period': ['Early Episodes', 'Late Episodes'] * 2,
        'Metric': ['Mean Cost', 'Mean Cost', 'Mean Variance', 'Mean Variance'],
        'Value': [np.mean(early_costs), np.mean(late_costs), 
                 np.mean(early_variances), np.mean(late_variances)]
    }
    
    df = pd.DataFrame(improvement_data)
    
    # Create bar plot
    x_pos = [0, 1, 3, 4]
    colors = ['lightblue', 'blue', 'lightcoral', 'red']
    bars = axes[1, 1].bar(x_pos, df['Value'], color=colors, alpha=0.7)
    
    axes[1, 1].set_title('Early vs Late Episode Performance')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Early\\nCost', 'Late\\nCost', 'Early\\nVariance', 'Late\\nVariance'])
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, df['Value']):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / "cost_leveling_analysis_v04.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cost leveling analysis plot saved: {save_path}")
    
    plt.show()


def generate_summary_report(history, save_dir="outputs_pump_cbm_v047_enhanced"):
    """Generate summary report for MVP v0.4"""
    
    print("\\n" + "="*80)
    print("📊 MVP v0.4 TRAINING SUMMARY REPORT")
    print("="*80)
    
    # Basic training info
    n_episodes = len(history['episode_rewards'])
    final_reward = history['episode_rewards'][-1]
    avg_reward_last_100 = np.mean(history['episode_rewards'][-100:])
    
    print(f"\\n🎯 Training Overview:")
    print(f"   Total Episodes: {n_episodes:,}")
    print(f"   Final Episode Reward: {final_reward:.2f}")
    print(f"   Avg Reward (Last 100 episodes): {avg_reward_last_100:.2f}")
    
    # Cost leveling performance (key for MVP v0.4)
    if 'episode_cost_variances' in history:
        cost_variances = np.array(history['episode_cost_variances'])
        final_variance = cost_variances[-1]
        avg_variance_last_100 = np.mean(cost_variances[-100:])
        
        print(f"\\n💡 Cost Leveling Performance:")
        print(f"   Final Cost Variance: {final_variance:.2f}")
        print(f"   Avg Variance (Last 100 episodes): {avg_variance_last_100:.2f}")
        
        if 'config' in history:
            threshold = history['config']['reward']['cost_leveling'].get('variance_threshold', 25.0)
            success_rate = np.sum(cost_variances[-100:] <= threshold) / 100 * 100
            print(f"   Variance Below Threshold ({threshold}): {success_rate:.1f}% of last 100 episodes")
    
    # Cost efficiency
    if 'episode_costs' in history:
        costs = np.array(history['episode_costs'])
        final_cost = costs[-1]
        avg_cost_last_100 = np.mean(costs[-100:])
        
        print(f"\\n💰 Cost Performance:")
        print(f"   Final Monthly Cost: {final_cost:.2f}")
        print(f"   Avg Cost (Last 100 episodes): {avg_cost_last_100:.2f}")
        
        if 'config' in history:
            target_budget = history['config']['reward']['cost_leveling'].get('target_monthly_budget', 50.0)
            print(f"   Target Budget: {target_budget:.2f}")
            budget_efficiency = (target_budget / avg_cost_last_100) * 100
            print(f"   Budget Efficiency: {budget_efficiency:.1f}%")
    
    # Equipment info
    if 'config' in history:
        equipment_count = len(history['config']['multi_equipment']['target_equipment_list'])
        print(f"\\n🏭 Equipment Configuration:")
        print(f"   Total Equipment: {equipment_count} HVAC units")
        print(f"   Equipment Types: R-series (冷凍機) + AHU-series (エアハンドリング)")
        print(f"   Age Range: 15.3 - 19.7 years")
    
    # Training efficiency
    print(f"\\n🚀 Training Insights:")
    if 'loss_history' in history and history['loss_history']:
        final_loss = history['loss_history'][-1] if history['loss_history'] else 'N/A'
        print(f"   Final Training Loss: {final_loss}")
    
    print(f"   Convergence: {'✅ Good' if avg_reward_last_100 > -50 else '⚠️ Needs improvement'}")
    
    # Save report
    report_path = Path(save_dir) / "training_summary_report_v04.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MVP v0.4 Training Summary Report\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"Total Episodes: {n_episodes:,}\\n")
        f.write(f"Final Reward: {final_reward:.2f}\\n")
        f.write(f"Average Reward (Last 100): {avg_reward_last_100:.2f}\\n")
        
        if 'episode_cost_variances' in history:
            f.write(f"\\nCost Leveling Performance:\\n")
            f.write(f"Final Variance: {final_variance:.2f}\\n")
            f.write(f"Average Variance (Last 100): {avg_variance_last_100:.2f}\\n")
    
    print(f"\\n✅ Summary report saved: {report_path}")
    print("="*80)


def main():
    """Main visualization function"""
    results_dir = "outputs_pump_cbm_v047_enhanced"
    
    # Load results
    history = load_training_results(results_dir)
    if history is None:
        print("❌ Cannot load training results. Make sure training has been completed.")
        return
    
    # Generate visualizations
    print("\\n🎨 Generating visualizations...")
    
    # 1. Training progress
    plot_training_progress(history, results_dir)
    
    # 2. Cost leveling analysis (key for MVP v0.4)
    plot_cost_leveling_analysis(history, results_dir)
    
    # 3. NEW: Cost-Risk Tradeoff Analysis
    plot_cost_risk_tradeoff(history, results_dir)
    
    # 4. Summary report
    generate_summary_report(history, results_dir)
    
    print("\\n✅ All visualizations completed!")


if __name__ == "__main__":
    main()