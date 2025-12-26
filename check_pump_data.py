import pandas as pd

# 実測データ読み込み
df = pd.read_csv('data/private_benchmark/測定値examples_3設備_測定項目_実測値_20251217.csv', encoding='utf-8-sig')

# 対象ポンプ設備ID
target_pumps = [265715, 137953, 519177]

# 実データに存在するかチェック
found = [id for id in target_pumps if id in df['設備id'].unique()]

print(f'対象ポンプ設備: {target_pumps}')
print(f'実データに存在: {found}')
print(f'実データ件数:')
for pump_id in found:
    count = len(df[df['設備id'] == pump_id])
    print(f'  - 設備ID {pump_id}: {count}件')

# 各設備の測定項目も確認
print('\n測定項目詳細:')
for pump_id in found:
    subset = df[df['設備id'] == pump_id]
    measurement_ids = subset['状態測定項目id'].unique()
    print(f'  設備ID {pump_id}: 測定項目ID {measurement_ids}')
    
    # 実測値の範囲
    min_val = subset['実測値'].min()
    max_val = subset['実測値'].max()
    mean_val = subset['実測値'].mean()
    print(f'    実測値範囲: {min_val:.1f} ～ {max_val:.1f} (平均: {mean_val:.1f})')
    
    # 閾値情報
    smin = subset['下限値Smin'].iloc[0]
    smax = subset['上限値Smax'].iloc[0]
    print(f'    正常範囲: {smin} ～ {smax}')