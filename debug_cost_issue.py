#!/usr/bin/env python3
"""
コスト計算の問題をデバッグするスクリプト（簡易版）
補修アクションが実行されているか、コストが正しく計算されているかを確認
"""

import yaml
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbm_environment_pump_v047 import MultiEquipmentCBMEnvironment, ACTION_NAMES

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def debug_environment_and_costs():
    print("=" * 60)
    print("🔧 Pump CBM v0.4.7 - Cost Calculation Debug")
    print("=" * 60)
    
    # Load configuration
    config_path = "config_pump_cbm_v047.yaml"
    
    # Create environment
    env = MultiEquipmentCBMEnvironment(config_path)
    
    # Load config for display
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📊 Environment Configuration:")
    print(f"  Equipment count: {len(env.equipment_list)}")
    print(f"  Cost parameters:")
    print(f"    Do nothing: {env.cost_do_nothing}")
    print(f"    Repair: {env.cost_repair}")
    print(f"    Replace: {env.cost_replace}")
    
    print(f"\n🎯 Testing Manual Actions:")
    
    # Test 1: All do nothing (should cost 3 * 10.0 = 30.0)
    state = env.reset()
    print(f"\n  Test 1 - All Do Nothing (0,0,0):")
    action = 0  # First equipment: do nothing, others: do nothing
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"    Monthly cost: {info['current_month_cost']}")
    print(f"    Expected: {3 * env.cost_do_nothing} (3 equipments × {env.cost_do_nothing})")
    
    # Test 2: One repair (should cost 1*5.0 + 2*10.0 = 25.0)
    state = env.reset()
    print(f"\n  Test 2 - One Repair (1,0,0):")
    action = 1  # First equipment: repair, others: do nothing  
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"    Monthly cost: {info['current_month_cost']}")
    print(f"    Expected: {env.cost_repair + 2 * env.cost_do_nothing} (1 repair + 2 do_nothing)")
    
    # Test 3: One replace (should cost 1*25.0 + 2*10.0 = 45.0)
    state = env.reset()
    print(f"\n  Test 3 - One Replace (2,0,0):")
    action = 2  # First equipment: replace, others: do nothing
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"    Monthly cost: {info['current_month_cost']}")
    print(f"    Expected: {env.cost_replace + 2 * env.cost_do_nothing} (1 replace + 2 do_nothing)")
    
    # Test 4: Mixed actions
    state = env.reset()
    print(f"\n  Test 4 - Mixed Actions (2,1,0):")
    action = 2*9 + 1*3 + 0  # Replace, Repair, Do Nothing
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"    Monthly cost: {info['current_month_cost']}")
    print(f"    Expected: {env.cost_replace + env.cost_repair + env.cost_do_nothing} (replace + repair + do_nothing)")
    
    print(f"\n📈 Action Decoding Test:")
    for test_action in [0, 1, 2, 9, 10, 26]:
        decoded = env._decode_action(test_action)
        print(f"    Action {test_action:2d} -> {decoded} -> {[ACTION_NAMES[a] for a in decoded]}")

def analyze_action_patterns():
    print("\n" + "=" * 60)
    print("🎲 Random Action Testing")
    print("=" * 60)
    
    config_path = "config_pump_cbm_v047.yaml"
    env = MultiEquipmentCBMEnvironment(config_path)
    
    total_cost = 0.0
    action_count = {0: 0, 1: 0, 2: 0}  # do_nothing, repair, replace
    num_tests = 100
    
    print(f"\n🎯 Testing {num_tests} random actions:")
    
    for i in range(num_tests):
        state = env.reset()
        action = np.random.randint(0, env.action_space.n)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_cost += info['current_month_cost']
        decoded_actions = env._decode_action(action)
        
        for equipment_action in decoded_actions:
            action_count[equipment_action] += 1
        
        if i < 5:  # Show first 5 examples
            print(f"  Test {i+1}: Action {action} -> {decoded_actions} -> Cost: {info['current_month_cost']:.1f}")
    
    print(f"\n📊 Action Distribution:")
    total_actions = sum(action_count.values())
    for action_type, count in action_count.items():
        percentage = (count / total_actions) * 100
        print(f"    {ACTION_NAMES[action_type]}: {count} times ({percentage:.1f}%)")
    
    avg_cost = total_cost / num_tests
    print(f"\n💰 Average Cost per Step: {avg_cost:.2f}")
    
    expected_min = 3 * env.cost_do_nothing
    expected_max = 3 * env.cost_replace
    print(f"    Expected range: {expected_min:.1f} - {expected_max:.1f}")
    
    if avg_cost < expected_min * 0.5:
        print(f"⚠️  WARNING: Average cost is much lower than expected!")
        print(f"    This suggests a problem in cost calculation.")
    elif avg_cost == 0.0:
        print(f"🚨 CRITICAL: All costs are 0.0!")
        print(f"    The cost calculation is completely broken.")

if __name__ == "__main__":
    # Debug environment costs
    debug_environment_and_costs()
    
    # Test with random actions
    analyze_action_patterns()