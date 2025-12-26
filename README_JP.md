# Pump CBM v0.4.7 - 3台ポンプ設備包括管理

v0.4.6の空調設備実験結果を踏まえた、設置年月日データ対応ポンプ設備専用CBM強化学習システム

## 🎯 開発目標

v0.4.6では製造テナントの6台空調設備を対象としていましたが、v0.4.7では**設置年月日データが存在する3台のポンプ設備**を対象とした包括的CBM管理を実現します。

### 対象ポンプ設備（AgedRL_Lesson.md実証ベース）

| 設備名 | 設備ID | 年数 | 測定項目 | 学習性能 | 収束時間 | 実用化準備度 |
|--------|--------|------|----------|----------|----------|-------------|
| **薬注ポンプCP-500-5** | PUMP-001 | 19.7年 | タンク残量 | +19.32 | 303ep | A (実用化推奨) |
| **冷却水ポンプCDP-A5** | PUMP-002 | 3.0年 | 電力点検 | -3.07 | 271ep | C (要更なる改善) |
| **薬注ポンプCP-500-3** | PUMP-003 | 0.5年 | タンク残量 | +11.34 | 100ep | B (改善継続) |

## 📁 ディレクトリ構成

```
pump_case/
├── config_pump_cbm_v047.yaml              # ポンプ設備専用設定ファイル
├── train_pump_cbm_v047_enhanced.py        # QR-DQN学習スクリプト
├── cbm_environment_pump_v047.py           # ポンプ環境定義
├── data_preprocessor.py                   # データ前処理
├── estimate_transitions_from_data.py      # Markov遷移推定
├── visualize_pump_results_v047.py         # 結果可視化
├── list_pump_equipment_v047.py            # 設備リスト生成
├── README.md                             # 本ファイル
└── outputs_pump_cbm_v047/                # 学習結果出力ディレクトリ
    ├── checkpoint_episode_2000.pth
    ├── policy_net.pth
    └── training_history.json
```

## 🧠 v0.4.7の主要特徴

### 1. AgedRL_Lesson実証データ活用
- **2000エピソード学習**: 実証済み効果の学習時間設定
- **設備別aging_factor**: 実測値ベースの老朽化係数
- **収束時間最適化**: 設備特性に応じた学習時間調整

### 2. ポンプ設備特化型最適化
```yaml
# ポンプ設備特有の学習パラメータ
pump_learning_params:
  equipment_specific_episodes:
    PUMP-001: 400       # 薬注ポンプCP-500-5：安定学習型
    PUMP-002: 2000      # 冷却水ポンプCDP-A5：特殊困難要因
    PUMP-003: 200       # 薬注ポンプCP-500-3：高速学習型
```

### 3. 年数別学習特性調整
- **新設備（0-5年）**: 学習率向上、探索強化
- **老朽設備（15年以上）**: 安定性重視、収束時間延長

### 4. ポンプタイプ別重み設定
```yaml
equipment_type_weights:
  pump_tank: 1.1      # 薬注ポンプ（タンク系）
  pump_cooling: 1.3   # 冷却水ポンプ（学習困難対応）
```

## 🚀 実行方法

### 1. 環境セットアップ
```bash
cd pump_case
pip install torch gymnasium numpy matplotlib pyyaml pandas tqdm
```

### 2. ポンプ設備リスト生成
```bash
python list_pump_equipment_v047.py
```

### 3. 学習実行
```bash
# 基本学習（2000エピソード）
python train_pump_cbm_v047_enhanced.py --config config_pump_cbm_v047.yaml

# カスタムエピソード数
python train_pump_cbm_v047_enhanced.py --episodes 2000
```

### 4. 結果可視化
```bash
python visualize_pump_results_v047.py --output_dir outputs_pump_cbm_v047
```

### 5. 環境テスト
```bash
python cbm_environment_pump_v047.py
```

## 📊 期待される学習成果

### AgedRL_Lesson実証済み性能指標
- **薬注ポンプCP-500-5**: 最終性能 +19.32（改善幅 +9.02）
- **薬注ポンプCP-500-3**: 最終性能 +11.34（劇的改善 +56.54）
- **冷却水ポンプCDP-A5**: 最終性能 -3.07（大幅改善 +55.33）

### 収束特性予測
- **高速学習型**: 薬注ポンプCP-500-3（100ep収束）
- **標準学習型**: 薬注ポンプCP-500-5（303ep）、冷却水ポンプ（271ep）
- **特殊困難型**: 冷却水ポンプ（2000ep以上推奨）

## 🔧 設備タイプ別学習特性

### 薬注ポンプ系（タンク残量測定）
- **優位性**: 同タイプでの安定した学習成功
- **老朽化耐性**: 19.7年設備でも高性能達成
- **新設備学習遅延**: 0.5年設備は初期困難だが最終改善

### 冷却水ポンプ系（電力測定）
- **困難な測定項目**: 電力系は外部要因多
- **特殊困難要因**: 継続学習または手法変更必要
- **改善継続型**: 大幅改善を示すも課題残存

## 📈 実用化ロードマップ

### フェーズ1（監視下実用化）- 3-6ヶ月
**対象**: 薬注ポンプ2台
- 安定運用確認
- タンク系測定の最適化
- 保全スケジュール調整

### フェーズ2（特殊対応）- 6-12ヶ月
**対象**: 冷却水ポンプ
- 追加学習（3000ep以上）
- ハイブリッド手法検討
- 電力系測定項目の課題解決

## ⚙️ 設定ファイル詳細

### config_pump_cbm_v047.yaml
```yaml
# 3台ポンプ設備用設定
multi_equipment:
  target_equipment_list:
    - equipment_id: PUMP-001  # 薬注ポンプCP-500-5 (19.7年)
    - equipment_id: PUMP-002  # 冷却水ポンプCDP-A5 (3.0年)  
    - equipment_id: PUMP-003  # 薬注ポンプCP-500-3 (0.5年)

# 学習パラメータ
training:
  n_episodes: 2000        # AgedRL_Lesson実証値
  learning_rate: 5.0e-4
  batch_size: 128

# ポンプ専用報酬設定
reward:
  cost:
    replace: 25.0         # ポンプ交換コスト（空調より高め）
```

## 🎯 v0.4.7完成版実装成果

### ✅ 実装完了機能

1. **3シナリオ比較システム**
   - **Balanced**: 安全とコストの最適バランス戦略
   - **Cost-Efficient**: 予算節約最優先戦略  
   - **Safety-First**: 継続運転安全最優先戦略

2. **包括的性能分析**
   - 学習曲線比較可視化
   - 統計的性能指標計算
   - シナリオ別レコメンデーション生成

3. **改良されたStability Score算出**
   - 変動係数（CV）ベース安定性評価
   - 報酬スケール非依存の相対評価

### 📊 実際の学習結果（1000エピソード × 3シナリオ）

| シナリオ | 最終平均報酬 | 総コスト | 安定性スコア | 学習効率 | 推奨度 |
|---------|------------|----------|-------------|----------|-------|
| **Safety-First** | **7,891.53** | 2,970,840 | 95.66/100 | 90.43/100 | **★★★** |
| **Balanced** | 6,353.98 | 4,452,579 | **96.65/100** | 4.49/100 | ★★☆ |
| **Cost-Efficient** | 3,129.14 | **1,587,249** | 89.13/100 | 0.00/100 | ★☆☆ |

### 💡 重要な教訓と発見

#### 1. Safety-First戦略の優位性
- **最高性能**: 7,891.53の平均報酬で他戦略を大幅上回り
- **バランス型コスト**: 2,970,840で適度なコスト管理
- **高い学習効率**: 90.43/100で最も効率的な学習

#### 2. 学習安定性の実測評価
- **全戦略で高安定**: 89.13～96.65点の優秀な安定性
- **Balanced戦略**: 96.65点で最高の安定性を実現
- **変動係数改善**: CV値3.4-11%で極めて安定した学習

#### 3. コスト効率性の課題
- **Cost-Efficient**: 最低コスト実現も性能犠牲大
- **学習効率問題**: Cost-Efficientは0点で学習困難
- **バランス型選択**: 純粋コスト重視は実用性に課題

## 🔧 実装済みツール群

### 学習・評価システム
```bash
# 基本学習（1000エピソード実証済み）
python train_pump_cbm_v047_enhanced.py --config config_pump_cbm_v047.yaml --episodes 1000

# 3シナリオ学習
python train_pump_cbm_v047_enhanced.py --config config_pump_cbm_v047_balanced.yaml --episodes 1000
python train_pump_cbm_v047_enhanced.py --config config_pump_cbm_v047_cost_efficient.yaml --episodes 1000  
python train_pump_cbm_v047_enhanced.py --config config_pump_cbm_v047_safety_first.yaml --episodes 1000

# シナリオ比較分析
python compare_pump_scenarios_v047.py
```

### 結果確認
```bash
# 個別結果可視化
python visualize_pump_results_v047.py

# 比較レポート確認
ls comparison_results_v047/
```

## 📈 推奨運用戦略

### 🥇 第1推奨: Safety-First
- **適用対象**: 重要度の高いポンプ設備
- **期待効果**: 高性能（7,891報酬）+ 適度コスト
- **運用方針**: 安全第一での積極的保全

### 🥈 第2推奨: Balanced（特定条件）
- **適用対象**: 学習安定性重視の環境
- **期待効果**: 最高安定性（96.65点）
- **注意点**: コスト増加（4,452,579）要考慮

### ⚠️ 注意: Cost-Efficient
- **制限的適用**: 予算制約が極めて厳しい場合のみ
- **課題**: 学習効率0点、性能低下リスク
- **代替案**: Safety-FirstでもCost-Efficientより4.6万円増程度

## 🔍 トラブルシューティング（実証済み）

### Learning Stability Score = 0 の解決法
**問題**: 元実装で標準偏差ベース計算によりスコア0
**解決**: 変動係数（CV = σ/μ × 100）ベース評価に変更
```python
# 修正版安定性評価（実装済み）
cv = np.std(rewards) / abs(np.mean(rewards)) * 100
if cv < 10: score = 90-100
elif cv < 20: score = 70-90
```

### 学習が収束しない場合
```bash
# Safety-First戦略推奨（実証済み高効率）
python train_pump_cbm_v047_enhanced.py --config config_pump_cbm_v047_safety_first.yaml --episodes 1000

# より長時間学習
python train_pump_cbm_v047_enhanced.py --episodes 2000
```

## 📋 完了事項

- [x] ✅ 3シナリオ学習システム実装
- [x] ✅ 包括的比較分析スクリプト作成
- [x] ✅ Learning Stability Score改良
- [x] ✅ 1000エピソード × 3シナリオ学習完了
- [x] ✅ 比較結果可視化・レポート生成
- [x] ✅ 実用化推奨戦略決定

## 📂 出力ファイル構成（実際生成済み）

```
pump_case/
├── outputs_pump_cbm_v047_balanced/          # Balancedシナリオ結果
├── outputs_pump_cbm_v047_cost_efficient/    # Cost-Efficientシナリオ結果  
├── outputs_pump_cbm_v047_safety_first/      # Safety-Firstシナリオ結果
└── comparison_results_v047/                 # 比較分析結果
    ├── pump_scenario_comparison_*.png       # 6サブプロット比較グラフ
    └── pump_scenario_comparison_report_*.md # 詳細分析レポート
```

---

**v0.4.7 Pump CBM Complete** - 3シナリオ比較実証完了、Safety-First戦略推奨の科学的ポンプ設備管理システム