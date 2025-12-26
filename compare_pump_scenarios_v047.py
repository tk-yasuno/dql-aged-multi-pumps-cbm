"""
Pump CBM v0.4.7 - シナリオ比較分析スクリプト
3つの保全戦略シナリオを比較分析：
1. バランス型（Balanced）：安全とコストの最適バランス
2. コスト重視型（Cost-Efficient）：予算節約を最優先
3. 安全重視型（Safety-First）：継続運転の安全を最優先
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings

# 警告抑制とフォント設定
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.set_loglevel('ERROR')
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class PumpScenarioComparison:
    def __init__(self):
        """ポンプCBMシナリオ比較分析の初期化"""
        self.scenarios = {
            "balanced": {
                "name": "Balanced",
                "output_dir": "outputs_pump_cbm_v047_balanced",
                "description": "安全とコストの最適バランス保全戦略",
                "color": "#4ECDC4"  # 青緑系 - バランスの色
            },
            "cost_efficient": {
                "name": "Cost-Efficient", 
                "output_dir": "outputs_pump_cbm_v047_cost_efficient",
                "description": "予算節約を最優先とする保全戦略",
                "color": "#45B7D1"  # 青系 - 効率の色
            },
            "safety_first": {
                "name": "Safety-First",
                "output_dir": "outputs_pump_cbm_v047_safety_first", 
                "description": "継続運転の安全を最優先とする保全戦略",
                "color": "#FF6B6B"  # 赤系 - 安全の色
            }
        }
        
        self.pump_case_dir = Path("C:/Users/yasun/RL/dql-aged-multi-equipment-cbm/pump_case")
        self.output_dir = self.pump_case_dir / "comparison_results_v047"
        self.output_dir.mkdir(exist_ok=True)  # フォルダが存在しない場合は作成
        self.results = {}
        self.comparison_metrics = {}
        
    def load_scenario_data(self, scenario_key):
        """指定シナリオの学習データを読み込み"""
        scenario = self.scenarios[scenario_key]
        output_dir = self.pump_case_dir / scenario["output_dir"]
        
        try:
            # training_history.jsonを読み込み
            history_file = output_dir / "training_history.json"
            if not history_file.exists():
                print(f"⚠️ {scenario_key}: training_history.jsonが見つかりません")
                return None
                
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # データ形式を統一（episode_rewardsをrewardsに、episode_costsをcostsに）
            if 'episode_rewards' in data:
                data['rewards'] = data['episode_rewards']
                data['episodes'] = list(range(1, len(data['episode_rewards']) + 1))
            if 'episode_costs' in data:
                data['costs'] = data['episode_costs']
            if 'loss_history' in data:
                data['losses'] = data['loss_history']
                
            print(f"✅ {scenario['name']}: データ読み込み完了 ({len(data.get('rewards', []))}エピソード)")
            return data
            
        except Exception as e:
            print(f"❌ {scenario_key}のデータ読み込みエラー: {e}")
            return None
    
    def calculate_performance_metrics(self, data, scenario_key):
        """パフォーマンス指標を計算"""
        if not data:
            return {}
            
        episodes = data.get('episodes', [])
        rewards = data.get('rewards', [])
        losses = data.get('losses', [])
        costs = data.get('costs', [])
        
        if not rewards:
            return {}
        
        # 最後の100エピソードの平均（収束性能）
        final_episodes = episodes[-100:] if len(episodes) >= 100 else episodes
        final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        final_costs = costs[-100:] if len(costs) >= 100 else costs
        
        metrics = {
            'total_episodes': len(episodes),
            'final_avg_reward': np.mean(final_rewards) if final_rewards else 0,
            'final_reward_std': np.std(final_rewards) if final_rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
            'min_reward': min(rewards) if rewards else 0,
            'final_avg_cost': np.mean(final_costs) if final_costs else 0,
            'total_cost': sum(costs) if costs else 0,
            'convergence_episode': self._find_convergence_point(rewards),
            'stability_score': self._calculate_stability(final_rewards),
            'learning_efficiency': self._calculate_learning_efficiency(rewards)
        }
        
        return metrics
    
    def _find_convergence_point(self, rewards, window=50, threshold=0.05):
        """学習の収束点を推定"""
        if len(rewards) < window * 2:
            return len(rewards)
            
        for i in range(window, len(rewards) - window):
            recent_mean = np.mean(rewards[i:i+window])
            future_mean = np.mean(rewards[i+window:i+window*2])
            
            if abs(recent_mean - future_mean) / abs(recent_mean) < threshold:
                return i
                
        return len(rewards)
    
    def _calculate_stability(self, rewards):
        """学習の安定性スコアを計算（変動係数ベース、低い変動ほど高得点）"""
        if len(rewards) < 2:
            return 0
        
        mean_reward = np.mean(rewards)
        if mean_reward == 0:
            return 0
            
        # 変動係数（CV）を使用：標準偏差/平均の百分率
        cv = np.std(rewards) / abs(mean_reward) * 100
        
        # CV値を0-100スコアに変換（低いCVほど高スコア）
        # CV < 10%: 90-100点, CV < 20%: 70-90点, CV < 50%: 30-70点
        if cv < 10:
            return max(90, 100 - cv)
        elif cv < 20:
            return max(70, 90 - (cv - 10))
        elif cv < 50:
            return max(30, 70 - (cv - 20) * 1.33)
        else:
            return max(0, 30 - (cv - 50) * 0.6)
    
    def _calculate_learning_efficiency(self, rewards):
        """学習効率スコア（早期収束ほど高得点）"""
        if len(rewards) < 100:
            return 50
            
        # 前半平均と後半平均の改善度
        half_point = len(rewards) // 2
        first_half = np.mean(rewards[:half_point])
        second_half = np.mean(rewards[half_point:])
        
        if first_half == 0:
            return 50
            
        improvement = (second_half - first_half) / abs(first_half) * 100
        return max(0, min(100, improvement))
        
    def create_comparison_plots(self):
        """比較グラフを作成"""
        if not self.results:
            print("❌ 比較するデータがありません")
            return
            
        # フィギュアサイズを調整
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pump CBM Scenario Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. 学習曲線比較
        ax1 = axes[0, 0]
        for scenario_key, data in self.results.items():
            if data and 'rewards' in data:
                # 移動平均でスムージング
                rewards = data['rewards']
                window = min(50, len(rewards) // 10)
                if window > 1:
                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                    ax1.plot(smoothed, label=self.scenarios[scenario_key]['name'], 
                            color=self.scenarios[scenario_key]['color'], linewidth=2)
                else:
                    ax1.plot(rewards, label=self.scenarios[scenario_key]['name'],
                            color=self.scenarios[scenario_key]['color'], linewidth=2)
        
        ax1.set_title('Learning Curves (Smoothed)', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最終性能比較（棒グラフ）
        ax2 = axes[0, 1]
        scenario_names = []
        final_rewards = []
        colors = []
        
        for scenario_key, metrics in self.comparison_metrics.items():
            if metrics:
                scenario_names.append(self.scenarios[scenario_key]['name'])
                final_rewards.append(metrics.get('final_avg_reward', 0))
                colors.append(self.scenarios[scenario_key]['color'])
        
        if scenario_names:
            bars = ax2.bar(scenario_names, final_rewards, color=colors, alpha=0.8)
            ax2.set_title('Final Average Reward', fontweight='bold')
            ax2.set_ylabel('Reward')
            
            # 値をバーの上に表示
            for bar, value in zip(bars, final_rewards):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(final_rewards)*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. コスト比較
        ax3 = axes[0, 2]
        total_costs = []
        for scenario_key, metrics in self.comparison_metrics.items():
            if metrics:
                total_costs.append(metrics.get('total_cost', 0))
            else:
                total_costs.append(0)
        
        if scenario_names and any(c > 0 for c in total_costs):
            bars = ax3.bar(scenario_names, total_costs, color=colors, alpha=0.8)
            ax3.set_title('Total Maintenance Cost', fontweight='bold')
            ax3.set_ylabel('Cost')
            
            for bar, value in zip(bars, total_costs):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(total_costs)*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 学習安定性比較
        ax4 = axes[1, 0]
        stability_scores = []
        for scenario_key, metrics in self.comparison_metrics.items():
            if metrics:
                stability_scores.append(metrics.get('stability_score', 0))
            else:
                stability_scores.append(0)
        
        if scenario_names:
            bars = ax4.bar(scenario_names, stability_scores, color=colors, alpha=0.8)
            ax4.set_title('Learning Stability Score', fontweight='bold')
            ax4.set_ylabel('Stability Score')
            ax4.set_ylim(0, 100)
            
            for bar, value in zip(bars, stability_scores):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 収束エピソード比較
        ax5 = axes[1, 1]
        convergence_episodes = []
        for scenario_key, metrics in self.comparison_metrics.items():
            if metrics:
                convergence_episodes.append(metrics.get('convergence_episode', 1000))
            else:
                convergence_episodes.append(1000)
        
        if scenario_names:
            bars = ax5.bar(scenario_names, convergence_episodes, color=colors, alpha=0.8)
            ax5.set_title('Convergence Episode', fontweight='bold')
            ax5.set_ylabel('Episodes to Converge')
            
            for bar, value in zip(bars, convergence_episodes):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(convergence_episodes)*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 総合スコア（レーダーチャート風）
        ax6 = axes[1, 2]
        
        # 各指標を正規化してスコア化
        metrics_names = ['Performance', 'Cost Efficiency', 'Stability', 'Learning Speed']
        
        if self.comparison_metrics:
            scenarios_for_radar = []
            values_for_radar = []
            
            max_reward = max([m.get('final_avg_reward', 0) for m in self.comparison_metrics.values() if m]) or 1
            max_cost = max([m.get('total_cost', 1) for m in self.comparison_metrics.values() if m]) or 1
            
            for scenario_key, metrics in self.comparison_metrics.items():
                if metrics:
                    scenarios_for_radar.append(self.scenarios[scenario_key]['name'])
                    # 正規化されたスコア（0-100）
                    perf_score = (metrics.get('final_avg_reward', 0) / max_reward) * 100
                    cost_score = 100 - (metrics.get('total_cost', 0) / max_cost) * 100  # 低コストほど高スコア
                    stability_score = metrics.get('stability_score', 0)
                    speed_score = 100 - (metrics.get('convergence_episode', 1000) / 1000) * 100
                    
                    values_for_radar.append([perf_score, cost_score, stability_score, speed_score])
            
            # 積み上げ棒グラフで表示
            bottom = np.zeros(len(scenarios_for_radar))
            colors_list = [self.scenarios[key]['color'] for key in self.comparison_metrics.keys() if self.comparison_metrics[key]]
            
            for i, metric in enumerate(metrics_names):
                values = [v[i] for v in values_for_radar]
                ax6.bar(scenarios_for_radar, values, bottom=bottom, 
                       label=metric, alpha=0.8)
                bottom += values
            
            ax6.set_title('Comprehensive Score', fontweight='bold')
            ax6.set_ylabel('Score')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / f"pump_scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 比較グラフを保存: {output_path}")
        
        plt.show()
        
    def generate_comparison_report(self):
        """比較レポートを生成"""
        if not self.comparison_metrics:
            print("❌ レポート生成用データがありません")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"pump_scenario_comparison_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pump CBM Scenario Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Executive Summary\n\n")
            
            # 最高性能シナリオを特定
            best_reward_scenario = max(self.comparison_metrics.keys(), 
                                     key=lambda k: self.comparison_metrics[k].get('final_avg_reward', 0) if self.comparison_metrics[k] else 0)
            best_stability_scenario = max(self.comparison_metrics.keys(),
                                        key=lambda k: self.comparison_metrics[k].get('stability_score', 0) if self.comparison_metrics[k] else 0)
            
            f.write(f"- **Best Performance:** {self.scenarios[best_reward_scenario]['name']}\n")
            f.write(f"- **Most Stable:** {self.scenarios[best_stability_scenario]['name']}\n\n")
            
            f.write(f"## Detailed Analysis\n\n")
            
            for scenario_key, metrics in self.comparison_metrics.items():
                if not metrics:
                    continue
                    
                scenario = self.scenarios[scenario_key]
                f.write(f"### {scenario['name']}\n")
                f.write(f"**Strategy:** {scenario['description']}\n\n")
                f.write(f"**Key Metrics:**\n")
                f.write(f"- Final Average Reward: {metrics.get('final_avg_reward', 0):.2f}\n")
                f.write(f"- Total Cost: {metrics.get('total_cost', 0):.2f}\n")
                f.write(f"- Stability Score: {metrics.get('stability_score', 0):.2f}/100\n")
                f.write(f"- Convergence Episode: {metrics.get('convergence_episode', 0)}\n")
                f.write(f"- Learning Efficiency: {metrics.get('learning_efficiency', 0):.2f}/100\n\n")
            
            f.write(f"## Recommendations\n\n")
            
            # 最高報酬シナリオ
            if best_reward_scenario:
                best_scenario = self.scenarios[best_reward_scenario]
                f.write(f"**Primary Recommendation:** {best_scenario['name']}\n")
                f.write(f"- {best_scenario['description']}\n")
                f.write(f"- Achieved highest average reward: {self.comparison_metrics[best_reward_scenario].get('final_avg_reward', 0):.2f}\n\n")
            
            # トレードオフ分析
            f.write(f"**Trade-off Analysis:**\n")
            for scenario_key, metrics in self.comparison_metrics.items():
                if metrics:
                    scenario = self.scenarios[scenario_key]
                    reward = metrics.get('final_avg_reward', 0)
                    cost = metrics.get('total_cost', 0)
                    stability = metrics.get('stability_score', 0)
                    
                    f.write(f"- **{scenario['name']}:** ")
                    if reward > 7000:
                        f.write("High performance, ")
                    else:
                        f.write("Moderate performance, ")
                    
                    if stability > 80:
                        f.write("Very stable learning")
                    elif stability > 60:
                        f.write("Stable learning")
                    else:
                        f.write("Less stable learning")
                    f.write(f" (Cost: {cost:.0f})\n")
            
            f.write(f"\n---\n")
            f.write(f"*Report generated by Pump CBM Scenario Comparison v0.4.7*")
        
        print(f"📝 比較レポートを保存: {report_path}")
        return report_path
        
    def run_comparison(self):
        """比較分析を実行"""
        print(f"\n{'='*80}")
        print(f"🔍 Pump CBM Scenario Comparison Analysis")
        print(f"{'='*80}")
        
        # 各シナリオのデータを読み込み
        for scenario_key in self.scenarios.keys():
            print(f"\n📊 {scenario_key}シナリオの分析中...")
            data = self.load_scenario_data(scenario_key)
            self.results[scenario_key] = data
            
            if data:
                metrics = self.calculate_performance_metrics(data, scenario_key)
                self.comparison_metrics[scenario_key] = metrics
                print(f"    最終平均報酬: {metrics.get('final_avg_reward', 0):.2f}")
                print(f"    安定性スコア: {metrics.get('stability_score', 0):.2f}/100")
            else:
                self.comparison_metrics[scenario_key] = None
        
        # 結果が少なくとも1つあれば比較実行
        valid_results = sum(1 for r in self.results.values() if r is not None)
        
        if valid_results >= 2:
            print(f"\n📈 比較グラフを生成中...")
            self.create_comparison_plots()
            
            print(f"\n📝 比較レポートを生成中...")
            report_path = self.generate_comparison_report()
            
            print(f"\n✅ 比較分析完了！")
            print(f"📁 結果ファイル:")
            print(f"   - 比較グラフ: pump_scenario_comparison_*.png")
            print(f"   - 比較レポート: {report_path.name}")
            
        else:
            print(f"❌ 有効な結果が{valid_results}個しかありません（最低2個必要）")
            print(f"各シナリオの学習が正常に完了しているか確認してください。")

def main():
    """メイン実行関数"""
    try:
        comparison = PumpScenarioComparison()
        comparison.run_comparison()
        
    except KeyboardInterrupt:
        print(f"\n⚠️ ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()