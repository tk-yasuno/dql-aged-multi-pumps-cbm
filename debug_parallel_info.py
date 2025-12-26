#!/usr/bin/env python3
"""
並列環境のinfo取得をテストして問題を特定
"""

import yaml
from cbm_environment_pump_v047 import MultiEquipmentCBMEnvironment
from gymnasium.vector import AsyncVectorEnv

def test_parallel_info():
    print("=" * 60)
    print("🔧 Parallel Environment Info Debug")
    print("=" * 60)
    
    # Create single environment
    single_env = MultiEquipmentCBMEnvironment("config_pump_cbm_v047.yaml")
    print(f"\n📊 Single Environment Test:")
    
    state = single_env.reset()
    action = 1  # Repair first equipment
    next_state, reward, terminated, truncated, info = single_env.step(action)
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Info keys: {list(info.keys())}")
    print(f"  Current month cost: {info.get('current_month_cost', 'NOT FOUND')}")
    
    # Create parallel environment
    print(f"\n📊 Parallel Environment Test:")
    def make_env():
        return MultiEquipmentCBMEnvironment("config_pump_cbm_v047.yaml")
    
    n_envs = 2
    envs = AsyncVectorEnv([make_env for _ in range(n_envs)])
    
    states = envs.reset()
    print(f"  States shape: {states[0].shape}")
    
    actions = [1, 2]  # Different actions for each environment
    next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
    
    print(f"  Actions: {actions}")
    print(f"  Rewards: {rewards}")
    print(f"  Infos type: {type(infos)}")
    print(f"  Infos length: {len(infos) if hasattr(infos, '__len__') else 'No length'}")
    
    if isinstance(infos, (list, tuple)):
        for i, info in enumerate(infos):
            print(f"  Info[{i}] type: {type(info)}")
            print(f"  Info[{i}] keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
            print(f"  Info[{i}] cost: {info.get('current_month_cost', 'NOT FOUND') if isinstance(info, dict) else 'N/A'}")
    elif isinstance(infos, dict):
        print(f"  Infos as dict keys: {list(infos.keys())}")
        
        # Let's look at the actual values for cost keys
        cost_keys = [key for key in infos.keys() if 'current_month_cost' in key]
        print(f"  Cost keys found: {cost_keys}")
        
        for key in cost_keys:
            value = infos[key]
            print(f"  {key}: type={type(value)}, value={value}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")
            if hasattr(value, '__len__'):
                print(f"    Length: {len(value)}")
        
        # Check variance keys too
        variance_keys = [key for key in infos.keys() if 'cost_variance' in key]
        print(f"  Variance keys found: {variance_keys}")
        
        for key in variance_keys:
            value = infos[key]
            print(f"  {key}: type={type(value)}, value={value}")
    else:
        print(f"  Unknown infos format: {infos}")
    
    envs.close()

if __name__ == "__main__":
    test_parallel_info()