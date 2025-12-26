"""
Pump Equipment List Generator for CBM v0.4.7

設置年月日データ有りの3台ポンプ設備リスト生成
Based on AgedRL_Lesson.md findings
"""

import pandas as pd
from pathlib import Path

def generate_pump_equipment_list():
    """3台ポンプ設備の詳細情報をリスト化"""
    
    pump_equipment_data = [
        {
            'equipment_id': 265715,
            'equipment_name': '薬注ポンプCP-500-5',
            'current_age': 19.7,
            'main_measurement_id': 526571,
            'measurement_name': 'タンク残量',
            'aging_factor': 0.018,
            'expected_performance': '+19.32',
            'convergence_episodes': 303,
            'stability_std': 20.97,
            'learning_type': 'stable_learner',
            'output_directory': 'outputs_pump_265715',
            '实用化_readiness': 'A (実用化推奨)',
            'notes': '老朽化設備だが安定した学習性能、大幅改善+9.02達成'
        },
        {
            'equipment_id': 137953,
            'equipment_name': '冷却水ポンプCDP-A5',
            'current_age': 3.0,
            'main_measurement_id': 137953,
            'measurement_name': '電力点検',
            'aging_factor': 0.005,
            'expected_performance': '-3.07',
            'convergence_episodes': 271,
            'stability_std': 42.45,
            'learning_type': 'special_difficulty',
            'output_directory': 'outputs_pump_137953',
            '実用化_readiness': 'C (要更なる改善)',
            'notes': '特殊困難要因、電力系測定項目、大幅改善+55.33も依然マイナス'
        },
        {
            'equipment_id': 519177,
            'equipment_name': '薬注ポンプCP-500-3',
            'current_age': 0.5,
            'main_measurement_id': 519177,
            'measurement_name': 'タンク残量',
            'aging_factor': 0.003,
            'expected_performance': '+11.34',
            'convergence_episodes': 100,
            'stability_std': 63.53,
            'learning_type': 'fast_learner',
            'output_directory': 'outputs_pump_519177',
            '実用化_readiness': 'B (改善継続)',
            'notes': '最新設備、高速学習型、劇的改善+56.54達成も高変動性'
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(pump_equipment_data)
    
    # Save as CSV
    output_path = Path(__file__).parent / "pump_equipment_list_v047.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Save as Markdown
    md_output_path = Path(__file__).parent / "Pump_Equipment_List_v047.md"
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write("# Pump Equipment List for CBM v0.4.7\\n\\n")
        f.write("設置年月日データ有り3台ポンプ設備の詳細情報\\n")
        f.write(f"作成日時: {pd.Timestamp.now().strftime('%Y年%m月%d日')}\\n\\n")
        
        f.write("## 対象ポンプ設備一覧\\n\\n")
        f.write("| 設備名 | 設備ID | 年数 | 測定項目 | aging_factor | 出力ディレクトリ | 実用化準備度 |\\n")
        f.write("|--------|--------|------|----------|-------------|-----------------|-------------|\\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['equipment_name']} | {row['equipment_id']} | {row['current_age']}年 | "
                   f"{row['measurement_name']} | {row['aging_factor']} | {row['output_directory']} | "
                   f"{row['実用化_readiness']} |\\n")
        
        f.write("\\n## 学習特性による分類\\n\\n")
        f.write("### 高速学習型（<200ep）\\n")
        f.write("- **薬注ポンプCP-500-3**: 100ep収束、高変動性、新設備の潜在力発現\\n\\n")
        
        f.write("### 標準学習型（200-400ep）\\n")
        f.write("- **冷却水ポンプCDP-A5**: 271ep収束、特殊困難要因\\n")
        f.write("- **薬注ポンプCP-500-5**: 303ep収束、安定学習型\\n\\n")
        
        f.write("### ポンプ設備特有の知見\\n\\n")
        f.write("#### 薬注ポンプ系（タンク残量測定）\\n")
        f.write("- **優位性**: 同タイプでの安定した学習成功\\n")
        f.write("- **老朽化耐性**: 19.7年設備でも高性能達成\\n")
        f.write("- **新設備学習遅延**: 0.5年設備は初期学習困難だが最終的に改善\\n\\n")
        
        f.write("#### 冷却水ポンプ系（電力測定）\\n")
        f.write("- **困難な測定項目**: 電力系は外部要因多、特殊な困難要因\\n")
        f.write("- **改善継続型**: 大幅改善を示すも依然課題残存\\n")
        f.write("- **追加学習必要**: 3000エピソード以上または手法変更検討\\n\\n")
        
        f.write("## v0.4.7学習戦略\\n\\n")
        f.write("### 設備別学習時間最適化\\n")
        f.write("```yaml\\n")
        f.write("equipment_specific_episodes:\\n")
        f.write("  265715: 400       # 薬注ポンプCP-500-5：安定学習型\\n")
        f.write("  137953: 2000      # 冷却水ポンプCDP-A5：特殊困難要因\\n")
        f.write("  519177: 200       # 薬注ポンプCP-500-3：高速学習型\\n")
        f.write("```\\n\\n")
        
        f.write("### 年数別学習特性調整\\n")
        f.write("- **新設備（0-5年）**: 学習率向上、探索強化\\n")
        f.write("- **老朽設備（15年以上）**: 安定性重視、収束時間延長\\n\\n")
        
        f.write("## 期待される成果\\n\\n")
        f.write("### 実証済み性能指標（AgedRL_Lesson based）\\n")
        f.write("- **薬注ポンプCP-500-5**: +19.32 ✅ 大幅改善（+9.02）\\n")
        f.write("- **薬注ポンプCP-500-3**: +11.34 ✅ 劇的回復（+56.54）\\n")
        f.write("- **冷却水ポンプCDP-A5**: -3.07 ⚠️ 改善も課題残（+55.33）\\n\\n")
        
        f.write("### 実用化ロードマップ\\n")
        f.write("**フェーズ1（監視下実用化）**: 薬注ポンプ2台\\n")
        f.write("- 目標: 安定運用確認、タンク系測定の最適化\\n")
        f.write("- 期間: 3-6ヶ月\\n\\n")
        
        f.write("**フェーズ2（特殊対応）**: 冷却水ポンプ\\n")
        f.write("- 目標: 追加学習またはハイブリッド手法検討\\n")
        f.write("- 期間: 6-12ヶ月\\n")
    
    print(f"✅ Pump equipment list generated:")
    print(f"  - CSV: {output_path}")
    print(f"  - Markdown: {md_output_path}")
    print(f"\\n📊 Equipment Summary:")
    print(df[['equipment_name', 'current_age', 'expected_performance', 'learning_type']].to_string(index=False))

if __name__ == "__main__":
    generate_pump_equipment_list()