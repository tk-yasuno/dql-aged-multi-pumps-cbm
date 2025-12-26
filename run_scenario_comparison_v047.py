#!/usr/bin/env python3
"""
Pump CBM v0.4.7 - 3シナリオ比較学習スクリプト
バランス型、コスト重視型、安全重視型の3つのシナリオで学習を実行し、結果を比較
"""

import subprocess
import time
import os
from pathlib import Path

def run_scenario_training(scenario_name: str, config_file: str, episodes: int = 1000):
    """指定されたシナリオで学習を実行"""
    print(f"\n{'='*80}")
    print(f"🚀 {scenario_name}シナリオ学習開始")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 学習コマンド実行
    cmd = [
        "python", 
        "train_pump_cbm_v047_enhanced.py", 
        "--config", config_file,
        "--episodes", str(episodes)
    ]
    
    print(f"📝 実行コマンド: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"✅ {scenario_name}シナリオ学習完了！")
            print(f"⏱️  実行時間: {elapsed_time:.1f}秒")
            print(f"📊 出力結果:")
            # 最後の数行を表示
            output_lines = result.stdout.split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"    {line}")
        else:
            print(f"❌ {scenario_name}シナリオ学習に失敗")
            print(f"エラー出力: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ {scenario_name}シナリオ実行中にエラー: {e}")
        return False
    
    return True

def compare_scenarios():
    """3シナリオの結果比較を実行"""
    print(f"\n{'='*80}")
    print(f"📊 シナリオ比較分析開始")
    print(f"{'='*80}")
    
    # 比較スクリプト実行（空調設備用をベースに作成予定）
    cmd = ["python", "compare_pump_scenarios_v047.py"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ シナリオ比較分析完了！")
            print("📈 比較結果:")
            # 出力結果を表示
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.strip():
                    print(f"    {line}")
        else:
            print("⚠️ シナリオ比較は手動で実行してください")
            print(f"比較用ファイル: compare_pump_scenarios_v047.py")
            
    except FileNotFoundError:
        print("⚠️ 比較スクリプトが見つかりません。手動で結果を比較してください。")
        print("📁 各シナリオの結果は以下のディレクトリに保存されています:")
        print("    - outputs_pump_cbm_v047_enhanced_balanced/")
        print("    - outputs_pump_cbm_v047_enhanced_cost_efficient/")  
        print("    - outputs_pump_cbm_v047_enhanced_safety_first/")

def main():
    """メイン実行関数"""
    print(f"\n{'='*80}")
    print(f"🏭 PUMP CBM v0.4.7 - 3シナリオ比較学習")
    print(f"{'='*80}")
    print(f"📋 実行予定:")
    print(f"  1️⃣ バランス型シナリオ (balanced)")
    print(f"  2️⃣ コスト重視シナリオ (cost_efficient)")
    print(f"  3️⃣ 安全重視シナリオ (safety_first)")
    print(f"  4️⃣ 結果比較分析")
    
    # 各シナリオの設定
    scenarios = [
        {
            "name": "バランス型",
            "config": "config_pump_cbm_v047_balanced.yaml",
            "key": "balanced"
        },
        {
            "name": "コスト重視型", 
            "config": "config_pump_cbm_v047_cost_efficient.yaml",
            "key": "cost_efficient"
        },
        {
            "name": "安全重視型",
            "config": "config_pump_cbm_v047_safety_first.yaml", 
            "key": "safety_first"
        }
    ]
    
    episodes = 1000  # 各シナリオのエピソード数
    
    # 全体開始時刻
    total_start_time = time.time()
    successful_scenarios = []
    
    # 各シナリオを順次実行
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🔄 {i}/3 シナリオ実行中...")
        
        success = run_scenario_training(
            scenario_name=scenario["name"],
            config_file=scenario["config"],
            episodes=episodes
        )
        
        if success:
            successful_scenarios.append(scenario["key"])
        
        # 次のシナリオとの間に少し間隔を置く
        if i < len(scenarios):
            print(f"\n⏳ 次のシナリオまで5秒待機...")
            time.sleep(5)
    
    # 全体完了報告
    total_elapsed_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"🎉 全シナリオ学習完了！")
    print(f"{'='*80}")
    print(f"⏱️  総実行時間: {total_elapsed_time:.1f}秒")
    print(f"✅ 成功したシナリオ: {len(successful_scenarios)}/3")
    
    for scenario_key in successful_scenarios:
        print(f"    ✓ {scenario_key}")
    
    if len(successful_scenarios) >= 2:
        print(f"\n📊 シナリオ比較を実行中...")
        compare_scenarios()
    
    print(f"\n🔍 各シナリオの詳細結果を確認するには:")
    print(f"    python visualize_pump_results_v047.py --scenario [balanced|cost_efficient|safety_first]")

if __name__ == "__main__":
    main()