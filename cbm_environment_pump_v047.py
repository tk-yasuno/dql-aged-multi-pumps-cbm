"""
Multi-Pump CBM Environment with Cost Leveling for v0.4.7

Features:
- Multiple pump equipment management (3 pump equipment)
- 2x2 state transition per equipment: Normal / Anomalous  
- Actions: DoNothing, Repair, Replace (per equipment)
- 💡 NEW: Cost leveling penalty - 保全費用の平準化
- Reward: Risk suppression + Cost minimization + Cost leveling penalty

State Definition:
- Per equipment: [condition, normalized_temperature, normalized_age]
- Global state: Concatenation of all equipment states + cost history

Action Space:
- 3^3 = 27 discrete actions (3 actions per 3 equipment)
- 0: Do Nothing, 1: Repair, 2: Replace (for each equipment)

Reward Function:
- Risk component: +1 for normal, -10 for anomalous (per equipment)
- Cost component: Equipment-specific costs
- 💡 Cost leveling penalty: Variance penalty for monthly maintenance costs
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
from collections import deque
from pathlib import Path


# ----- Constants -----

STATE_NAMES = ["Normal", "Anomalous"]  # 0, 1
ACTION_NAMES = ["DoNothing", "Repair", "Replace"]  # 0, 1, 2

# Default 2x2 transition matrix per equipment
DEFAULT_TRANSITIONS = {
    0: np.array([  # DoNothing
        [0.85, 0.15],  # from Normal → [Normal, Anomalous]
        [0.10, 0.90],  # from Anomalous → [Normal, Anomalous]
    ], dtype=np.float32),
    1: np.array([  # Repair
        [0.95, 0.05],  # from Normal → [Normal, Anomalous] 
        [0.75, 0.25],  # from Anomalous → [Normal, Anomalous]
    ], dtype=np.float32),
    2: np.array([  # Replace
        [0.98, 0.02],  # from Normal → [Normal, Anomalous]
        [0.90, 0.10],  # from Anomalous → [Normal, Anomalous]
    ], dtype=np.float32),
}


class MultiEquipmentCBMEnvironment(gym.Env):
    """
    Multi-Pump CBM Environment with Cost Leveling
    
    Manages 3 pump equipment simultaneously with cost leveling optimization.
    """
    
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}
    
    def __init__(
        self,
        config_path: str = "config_pump_cbm_v047.yaml",
        seed: Optional[int] = None
    ):
        """Initialize Multi-Equipment CBM Environment
        
        Args:
            config_path: Configuration file path
            seed: Random seed
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.equipment_list = self.config['multi_equipment']['target_equipment_list']
        self.n_equipment = len(self.equipment_list)
        
        # Environment parameters
        env_config = self.config['environment']
        self.horizon = env_config['horizon']
        self.gamma = env_config['gamma']
        self.cost_lambda = env_config['cost_lambda']
        
        # Temperature settings
        temp_config = env_config['temperature_range']
        self.temp_min = temp_config['min']
        self.temp_max = temp_config['max']
        
        normal_temp_config = env_config['normal_temp_range']
        self.normal_temp_min = normal_temp_config['min']
        self.normal_temp_max = normal_temp_config['max']
        
        # Aging parameters
        aging_config = env_config['aging']
        self.aging_factor = aging_config['aging_factor']
        self.max_equipment_age = aging_config['max_age']
        
        # Reward parameters
        reward_config = self.config['reward']
        self.risk_normal = reward_config['risk']['normal']
        self.risk_anomalous = reward_config['risk']['anomalous']
        
        # Cost parameters
        cost_config = reward_config['cost']
        self.cost_do_nothing = cost_config['do_nothing']
        self.cost_repair = cost_config['repair']
        self.cost_replace = cost_config['replace']
        
        # 💡 Cost leveling parameters
        leveling_config = reward_config.get('cost_leveling', {})
        self.cost_leveling_enabled = leveling_config.get('enabled', False)
        self.cost_window_size = leveling_config.get('window_size', 12)
        self.target_monthly_budget = leveling_config.get('target_monthly_budget', 50.0)
        self.leveling_penalty_weight = leveling_config.get('leveling_penalty_weight', 2.0)
        self.variance_threshold = leveling_config.get('variance_threshold', 25.0)
        
        # Load data-driven transitions if available
        self.use_real_data_transitions = env_config.get('use_real_data_transitions', False)
        self.real_transitions = None
        
        if self.use_real_data_transitions:
            try:
                from data_preprocessor import CBMDataPreprocessor
                data_processor = CBMDataPreprocessor()
                self.real_transitions = data_processor.load_real_data_transitions()
                print(f"✓ 実データベースの遷移行列を読み込み: {self.n_equipment}設備")
            except Exception as e:
                print(f"⚠️ 実データの遷移行列読み込みに失敗: {e}")
                self.use_real_data_transitions = False
        
        # Multi-equipment settings
        multi_config = self.config.get('multi_equipment_settings', {})
        self.simultaneous_discount = multi_config.get('simultaneous_maintenance_discount', 0.15)
        
        # State and action spaces
        self._setup_spaces()
        
        # Initialize equipment states
        self._initialize_equipment_states()
        
        # Cost history for leveling (monthly costs)
        self.monthly_cost_history = deque(maxlen=self.cost_window_size)
        
        # Episode tracking
        self.current_step = 0
        self.np_random = np.random.RandomState(seed)
        
        print(f"[OK] Multi-Equipment CBM Environment initialized")
        print(f"   - Equipment count: {self.n_equipment}")
        print(f"   - Cost leveling: {'Enabled' if self.cost_leveling_enabled else 'Disabled'}")
        print(f"   - Action space size: {self.action_space.n}")
    
    def _setup_spaces(self):
        """Set up action and observation spaces"""
        # Action space: 3^n_equipment discrete actions
        self.action_space = spaces.Discrete(3 ** self.n_equipment)
        
        # Observation space: [condition, norm_temp, norm_age] per equipment + cost history
        equipment_obs_dim = 3  # condition, normalized_temp, normalized_age
        cost_history_dim = self.cost_window_size  # monthly cost history
        
        total_obs_dim = (equipment_obs_dim * self.n_equipment) + cost_history_dim
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
    def _initialize_equipment_states(self):
        """Initialize equipment states from configuration"""
        self.equipment_conditions = np.zeros(self.n_equipment, dtype=int)  # All start normal
        self.equipment_temperatures = np.zeros(self.n_equipment)
        self.equipment_ages = np.zeros(self.n_equipment)
        
        # Set initial ages from configuration
        for i, equipment in enumerate(self.equipment_list):
            self.equipment_ages[i] = equipment['current_age']
            # Sample initial temperature
            self.equipment_temperatures[i] = self._sample_temperature(0, i)
    
    def _sample_temperature(self, condition: int, equipment_idx: int) -> float:
        """Sample temperature based on condition and equipment"""
        if condition == 0:  # Normal
            # Sample from normal range
            temp = self.np_random.uniform(self.normal_temp_min, self.normal_temp_max)
        else:  # Anomalous
            # Sample from outside normal range
            if self.np_random.random() < 0.5:
                # Below normal range
                temp = self.np_random.uniform(self.temp_min, self.normal_temp_min)
            else:
                # Above normal range  
                temp = self.np_random.uniform(self.normal_temp_max, self.temp_max)
        
        return temp
    
    def _normalize_temperature(self, temp: float) -> float:
        """Normalize temperature to [0, 1]"""
        return (temp - self.temp_min) / (self.temp_max - self.temp_min)
    
    def _decode_action(self, action: int) -> List[int]:
        """Decode single action integer to per-equipment actions
        
        Args:
            action: Integer in [0, 3^n_equipment)
            
        Returns:
            List of actions for each equipment [0, 1, 2]
        """
        actions = []
        remaining = action
        
        for i in range(self.n_equipment):
            equipment_action = remaining % 3
            actions.append(equipment_action)
            remaining //= 3
        
        return actions
    
    def _get_data_driven_transition(self, equipment_action: int, equipment_idx: int) -> np.ndarray:
        """実データベースの遷移行列を取得し、年数調整を適用"""
        if not self.use_real_data_transitions or self.real_transitions is None:
            return self._get_age_adjusted_transition(DEFAULT_TRANSITIONS[equipment_action], equipment_idx)
        
        # 設備IDを取得
        equipment_list = self.config['multi_equipment']['target_equipment_list']
        equipment_id = equipment_list[equipment_idx]['equipment_id']
        
        if equipment_id not in self.real_transitions:
            return self._get_age_adjusted_transition(DEFAULT_TRANSITIONS[equipment_action], equipment_idx)
        
        equipment_data = self.real_transitions[equipment_id]
        action_names = ['do_nothing', 'repair', 'replace']
        action_key = action_names[equipment_action]
        
        if action_key not in equipment_data:
            return self._get_age_adjusted_transition(DEFAULT_TRANSITIONS[equipment_action], equipment_idx)
        
        # 実データベースの遷移行列を使用
        base_transition = equipment_data[action_key]
        
        # 年数でさらに調整
        return self._apply_age_adjustment(base_transition, equipment_idx)
    
    def _apply_age_adjustment(self, transition: np.ndarray, equipment_idx: int) -> np.ndarray:
        """年数による遷移行列の微調整（実データベース用）"""
        equipment_list = self.config['multi_equipment']['target_equipment_list']
        current_age = equipment_list[equipment_idx]['current_age']
        
        adjusted = transition.copy()
        
        # 実データベースの場合は、より綾やかな調整
        age_effect = min(0.05, (current_age - 15.0) * 0.005)  # 15年を基準として微調整
        
        if age_effect > 0:
            # Normal → Anomalous 遷移率をわずかに増加
            adjusted[0, 1] = min(0.95, adjusted[0, 1] + age_effect)
            adjusted[0, 0] = 1.0 - adjusted[0, 1]
        
        return adjusted
    
    def _get_age_adjusted_transition(self, base_transition: np.ndarray, equipment_idx: int) -> np.ndarray:
        """Get age-adjusted transition matrix for specific equipment"""
        equipment_age = self.equipment_ages[equipment_idx]
        aging_effect = equipment_age * self.aging_factor
        
        adjusted = base_transition.copy()
        
        # Increase Normal → Anomalous transition probability
        if adjusted[0, 1] + aging_effect < 1.0:
            adjusted[0, 1] += aging_effect
            adjusted[0, 0] = 1.0 - adjusted[0, 1]
        else:
            adjusted[0, 1] = 0.99
            adjusted[0, 0] = 0.01
            
        # Slightly decrease Anomalous → Normal recovery probability
        recovery_penalty = aging_effect * 0.3
        if adjusted[1, 0] - recovery_penalty > 0.0:
            adjusted[1, 0] -= recovery_penalty
            adjusted[1, 1] = 1.0 - adjusted[1, 0]
        
        return adjusted
    
    def _calculate_cost_leveling_penalty(self, current_month_cost: float) -> float:
        """Calculate cost leveling penalty based on cost variance
        
        Args:
            current_month_cost: Cost for current month
            
        Returns:
            Penalty value (negative reward)
        """
        if not self.cost_leveling_enabled or len(self.monthly_cost_history) < 2:
            return 0.0
        
        # Add current month cost to history
        history_with_current = list(self.monthly_cost_history) + [current_month_cost]
        
        # Calculate variance from target budget
        costs = np.array(history_with_current)
        variance = np.var(costs)
        
        # Calculate penalty if variance exceeds threshold
        if variance > self.variance_threshold:
            penalty = (variance - self.variance_threshold) * self.leveling_penalty_weight
            return -penalty
        
        return 0.0
    
    def _calculate_safety_bonus(self) -> float:
        """Calculate safety bonus for maintaining equipment in normal state
        
        Returns:
            Bonus reward for safety performance
        """
        normal_count = np.sum(self.equipment_conditions == 0)  # Count normal equipment
        total_equipment = len(self.equipment_conditions)
        
        # Safety ratio bonus
        safety_ratio = normal_count / total_equipment
        
        # Exponential bonus for high safety performance
        if safety_ratio >= 1.0:  # All equipment normal
            return 10.0  # Strong safety bonus
        elif safety_ratio >= 0.8:  # Most equipment normal  
            return 5.0 * safety_ratio
        else:
            return 0.0  # No bonus for poor safety
    
    def _calculate_maintenance_action_bonus(self, action_vector):
        """
        保全実行ボーナス（適切な保全行動への追加報酬）
        """
        action_bonus = 0.0
        
        for i, action in enumerate(action_vector):
            if action == 1:  # 修理実行
                if self.equipment_conditions[i] == 1:  # 異常設備の修理
                    action_bonus += 8.0  # 適切な修理への高いボーナス
                else:  # 正常設備の予防保全
                    action_bonus += 3.0  # 予防保全への適度なボーナス
            elif action == 2:  # 交換実行
                if self.equipment_conditions[i] == 1:  # 異常設備の交換
                    action_bonus += 12.0  # 適切な交換への最高ボーナス
                else:  # 正常設備の更新
                    action_bonus += 2.0  # 設備更新への小さなボーナス
        
        return action_bonus

    def _calculate_simultaneous_discount(self, actions: List[int]) -> float:
        """Calculate discount for simultaneous maintenance actions"""
        maintenance_count = sum(1 for action in actions if action > 0)  # Repair or Replace
        
        if maintenance_count > 1:
            return self.simultaneous_discount * (maintenance_count - 1)
        
        return 0.0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.current_step = 0
        self._initialize_equipment_states()
        
        # Reset cost history
        self.monthly_cost_history.clear()
        for _ in range(self.cost_window_size):
            self.monthly_cost_history.append(0.0)
        
        obs = self._get_observation()
        info = {
            'equipment_conditions': [STATE_NAMES[c] for c in self.equipment_conditions],
            'equipment_temperatures': self.equipment_temperatures.copy(),
            'equipment_ages': self.equipment_ages.copy(),
            'monthly_cost_history': list(self.monthly_cost_history)
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs_parts = []
        
        # Per-equipment observations
        for i in range(self.n_equipment):
            condition = float(self.equipment_conditions[i])
            norm_temp = self._normalize_temperature(self.equipment_temperatures[i])
            norm_age = min(self.equipment_ages[i] / self.max_equipment_age, 1.0)
            
            obs_parts.extend([condition, norm_temp, norm_age])
        
        # Cost history
        obs_parts.extend(list(self.monthly_cost_history))
        
        return np.array(obs_parts, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment forward
        
        Args:
            action: Encoded action for all equipment
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action
        equipment_actions = self._decode_action(action)
        
        # Store old states
        old_conditions = self.equipment_conditions.copy()
        
        # Calculate rewards and apply actions
        total_risk_reward = 0.0
        total_cost_reward = 0.0
        current_month_cost = 0.0
        
        info_per_equipment = []
        
        for i in range(self.n_equipment):
            equipment_action = equipment_actions[i]
            old_condition = old_conditions[i]
            
            # --- Risk Reward Component ---
            if old_condition == 0:
                risk_reward = self.risk_normal
            else:
                risk_reward = self.risk_anomalous
            
            total_risk_reward += risk_reward
            
            # --- Cost Component ---
            if equipment_action == 0:  # Do Nothing
                action_cost = self.cost_do_nothing
            elif equipment_action == 1:  # Repair
                action_cost = self.cost_repair
            else:  # Replace
                action_cost = self.cost_replace
            
            current_month_cost += action_cost
            
            # --- State Transition ---
            if self.use_real_data_transitions:
                # 実データベースの遷移行列を使用
                trans_matrix = self._get_data_driven_transition(equipment_action, i)
                prob = trans_matrix[old_condition]
            else:
                # デフォルトの遷移行列を使用
                base_trans = DEFAULT_TRANSITIONS[equipment_action]
                age_adjusted_trans = self._get_age_adjusted_transition(base_trans, i)
                prob = age_adjusted_trans[old_condition]
            
            new_condition = self.np_random.choice([0, 1], p=prob)
            
            # Update equipment state
            self.equipment_conditions[i] = new_condition
            self.equipment_temperatures[i] = self._sample_temperature(new_condition, i)
            
            # Update equipment age
            if equipment_action == 2:  # Replace - reset age
                self.equipment_ages[i] = 0.0
            else:
                # Age increment (assuming 1 step = 1 month)
                self.equipment_ages[i] += (1.0 / 12.0)  # monthly aging
            
            info_per_equipment.append({
                'equipment_id': self.equipment_list[i]['equipment_id'],
                'equipment_name': self.equipment_list[i]['equipment_name'],
                'action': ACTION_NAMES[equipment_action],
                'old_condition': STATE_NAMES[old_condition],
                'new_condition': STATE_NAMES[new_condition],
                'temperature': self.equipment_temperatures[i],
                'age': self.equipment_ages[i],
                'risk_reward': risk_reward,
                'action_cost': action_cost
            })
        
        # Apply simultaneous maintenance discount
        simultaneous_discount = self._calculate_simultaneous_discount(equipment_actions)
        current_month_cost *= (1.0 - simultaneous_discount)
        
        total_cost_reward = -current_month_cost * self.cost_lambda
        
        # 💡 Cost leveling penalty
        cost_leveling_penalty = self._calculate_cost_leveling_penalty(current_month_cost)
        
        # 🛡️ Safety bonus: Reward for maintaining all equipment in normal state
        safety_bonus = self._calculate_safety_bonus()
        
        # 🆕 Maintenance action bonus (encourage appropriate maintenance actions)
        action_bonus = self._calculate_maintenance_action_bonus(equipment_actions)
        
        # Update cost history
        self.monthly_cost_history.append(current_month_cost)
        
        # --- Total Reward (4 Components) ---
        # 1) Safety (risk + safety_bonus): Maintain normal state
        # 2) Cost Efficiency (total_cost_reward): Optimize maintenance costs  
        # 3) Cost Leveling (cost_leveling_penalty): Stabilize monthly budgets
        # 4) Action Bonus: Encourage appropriate maintenance actions
        total_reward = total_risk_reward + total_cost_reward + cost_leveling_penalty + safety_bonus + action_bonus
        
        # Update step
        self.current_step += 1
        
        # Episode termination
        terminated = False
        truncated = self.current_step >= self.horizon
        
        # Info
        info = {
            'equipment_info': info_per_equipment,
            'total_risk_reward': total_risk_reward,
            'total_cost_reward': total_cost_reward,
            'cost_leveling_penalty': cost_leveling_penalty,
            'safety_bonus': safety_bonus,
            'action_bonus': action_bonus,
            'total_reward': total_reward,
            'current_month_cost': current_month_cost,
            'simultaneous_discount': simultaneous_discount,
            'monthly_cost_history': list(self.monthly_cost_history),
            'cost_variance': np.var(list(self.monthly_cost_history)),
            'safety_ratio': np.sum(self.equipment_conditions == 0) / len(self.equipment_conditions),
            'step': self.current_step
        }
        
        obs = self._get_observation()
        
        return obs, total_reward, terminated, truncated, info
    
    def render(self):
        """Render environment state"""
        if hasattr(self, 'render_mode') and self.render_mode in ['human', 'ansi']:
            print(f"\\n=== Step {self.current_step} ===")
            for i in range(self.n_equipment):
                equipment_name = self.equipment_list[i]['equipment_name']
                condition = STATE_NAMES[self.equipment_conditions[i]]
                temp = self.equipment_temperatures[i]
                age = self.equipment_ages[i]
                print(f"  {equipment_name}: {condition}, {temp:.1f}°C, {age:.1f}年")
            
            if self.cost_leveling_enabled:
                cost_variance = np.var(list(self.monthly_cost_history))
                print(f"  Cost Variance: {cost_variance:.2f} (Target: <{self.variance_threshold})")


def test_multi_equipment_environment():
    """Test Multi-Equipment CBM Environment"""
    print("="*80)
    print("🧪 Multi-Equipment CBM Environment Test")
    print("="*80)
    
    # Test with sample configuration
    env = MultiEquipmentCBMEnvironment(
        config_path="config_pump_cbm_v047.yaml",
        seed=42
    )
    
    print(f"\\n[OK] Environment created")
    print(f"  - Equipment count: {env.n_equipment}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Cost leveling: {env.cost_leveling_enabled}")
    
    # Test episode
    obs, info = env.reset(seed=42)
    print(f"\\n🎬 Initial state:")
    for i, eq_info in enumerate(info['equipment_conditions']):
        eq_name = env.equipment_list[i]['equipment_name']
        print(f"  - {eq_name}: {eq_info}")
    
    total_reward = 0.0
    
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        print(f"\\n📊 Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Monthly cost: {info['current_month_cost']:.2f}")
        print(f"  Cost variance: {info['cost_variance']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\\n📈 Episode Summary:")
    print(f"  - Total steps: {env.current_step}")
    print(f"  - Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    test_multi_equipment_environment()