"""
Pump Equipment QR-DQN Training Script for CBM v0.4.7 (Enhanced)

Complete integration of v0.2's advanced algorithms for pump equipment:
- [OK] N-step Learning (n=3) with PrioritizedNStepBuffer
- [OK] AsyncVectorEnv parallelization (16 environments)
- [OK] Mixed Precision Training (AMP) with GradScaler
- [OK] Advanced PER with dynamic beta adjustment
- [OK] Sophisticated QR-DQN loss with importance sampling weights

Optimized for 3 pump equipment management with cost leveling.
Based on AgedRL_Lesson findings: pump-specific learning characteristics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import json
import yaml
from torch.amp import autocast, GradScaler
from collections import deque
from gymnasium.vector import AsyncVectorEnv
import sys
import random

sys.path.insert(0, str(Path(__file__).parent))
from cbm_environment_pump_v047 import MultiEquipmentCBMEnvironment


# ===== Noisy Networks (from v0.2) =====

class NoisyLinear(nn.Module):
    """Noisy Linear for parameter-space exploration (from v0.2)"""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


# ===== QR-DQN Network (Enhanced) =====

class MultiEquipmentQRDQN(nn.Module):
    """QR-DQN for Multi-Equipment CBM with Dueling Architecture and Noisy Networks (from v0.2)"""
    
    def __init__(self, state_dim: int, n_actions: int, n_quantiles: int = 51, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream (Noisy)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_quantiles)
        )
        
        # Advantage stream (Noisy)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions * n_quantiles)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Returns:
            q_values: (batch, n_actions) - mean Q-values
            quantiles: (batch, n_actions, n_quantiles) - full distributions
        """
        batch_size = x.size(0)
        features = self.feature(x)
        
        # Dueling architecture
        value = self.value_stream(features)  # (batch, n_quantiles)
        advantage = self.advantage_stream(features)  # (batch, n_actions * n_quantiles)
        
        value = value.view(batch_size, 1, self.n_quantiles)
        advantage = advantage.view(batch_size, self.n_actions, self.n_quantiles)
        
        # Combine: Q = V + (A - mean(A))
        quantiles = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_values = quantiles.mean(dim=2)  # Average over quantiles
        
        return q_values, quantiles
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ===== Prioritized N-step Replay Buffer (from v0.2) =====

class PrioritizedNStepBuffer:
    """
    Prioritized Experience Replay with N-step returns (from v0.2)
    High-performance implementation with dynamic beta adjustment
    """
    
    def __init__(self, capacity: int, n_steps: int = 3, gamma: float = 0.95, 
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # N-step buffer
        self.n_step_buffer = []
    
    def push(self, state, action, reward, next_state, done):
        """Store transition with n-step return"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return
        
        # Compute n-step return
        n_step_state, n_step_action = self.n_step_buffer[0][:2]
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                break
        
        n_step_next_state = self.n_step_buffer[-1][3]
        n_step_done = self.n_step_buffer[-1][4]
        
        # Store with max priority
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done))
        else:
            self.buffer[self.position] = (n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Remove oldest from n-step buffer
        if self.n_step_buffer[0][4]:  # If done
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.pop(0)
    
    def sample(self, batch_size: int):
        """Sample batch with prioritized sampling"""
        if self.size < batch_size:
            return None
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return self.size


# ===== Advanced QR-DQN Loss Function (from v0.2) =====

def quantile_huber_loss_per(
    policy_net, target_net, 
    states, actions, rewards, next_states, dones, 
    weights, gamma, kappa=1.0, n_steps=3
):
    """
    Compute QR-DQN loss with importance sampling weights (from v0.2)
    High-performance implementation with PER integration
    
    Returns:
        loss: Weighted quantile Huber loss
        td_errors: TD errors for priority updates
    """
    batch_size = states.size(0)
    n_quantiles = policy_net.n_quantiles
    
    # Current quantiles
    _, current_quantiles = policy_net(states)
    current_quantiles = current_quantiles.gather(1, actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, n_quantiles)).squeeze(1)
    
    # Target quantiles (Double DQN)
    with torch.no_grad():
        next_q_values, _ = policy_net(next_states)
        next_actions = next_q_values.argmax(dim=1)
        _, next_quantiles = target_net(next_states)
        next_quantiles = next_quantiles.gather(1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, n_quantiles)).squeeze(1)
        
        # N-step target
        target_quantiles = rewards.unsqueeze(-1) + (gamma ** n_steps) * next_quantiles * (1 - dones.unsqueeze(-1))
    
    # Quantile Huber loss
    tau = torch.linspace(0.0, 1.0, n_quantiles + 1, device=states.device)
    tau = (tau[:-1] + tau[1:]) / 2.0
    tau = tau.view(1, 1, n_quantiles)
    
    td_errors_matrix = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    huber_loss = torch.where(td_errors_matrix.abs() <= kappa, 
                             0.5 * td_errors_matrix ** 2,
                             kappa * (td_errors_matrix.abs() - 0.5 * kappa))
    
    quantile_loss = (tau - (td_errors_matrix < 0).float()).abs() * huber_loss
    loss_per_sample = quantile_loss.sum(dim=2).mean(dim=1)
    
    # Apply importance sampling weights
    weighted_loss = (loss_per_sample * weights).mean()
    
    # TD errors for priority updates
    td_errors = loss_per_sample.detach()
    
    return weighted_loss, td_errors


# ===== Environment Factory (from v0.2, adapted for multi-equipment) =====

def make_multi_equipment_cbm_env(config_path: str, seed: int = 42):
    """Factory function for creating Multi-Equipment CBM environments"""
    def _init():
        env = MultiEquipmentCBMEnvironment(config_path=config_path, seed=seed)
        env.reset(seed=seed)
        return env
    return _init


def train_multi_equipment_dqn_enhanced(
    config_path: str = "config_pump_cbm_v047.yaml",
    n_episodes: int = 2000,
    n_envs: int = 16,
    lr: float = 5e-4,
    batch_size: int = 128,
    buffer_capacity: int = 200000,
    target_sync_steps: int = 1000,
    n_quantiles: int = 51,
    n_steps: int = 3,
    kappa: float = 1.0,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Train Multi-Equipment QR-DQN for CBM with v0.2's Advanced Algorithms
    
    Features from v0.2:
    - N-step Learning (n=3) with PrioritizedNStepBuffer
    - AsyncVectorEnv parallelization (16 environments)
    - Mixed Precision Training (AMP) with GradScaler
    - Advanced PER with dynamic beta adjustment
    - Sophisticated QR-DQN loss with importance sampling weights
    """
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    save_dir = Path(config['output']['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    print(f"\\n{'='*80}")
    print(f"MULTI-EQUIPMENT CBM QR-DQN TRAINING v0.4.2 (Enhanced with v0.2)")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Episodes: {n_episodes}, Parallel Envs: {n_envs}")
    print(f"  Equipment: {len(config['multi_equipment']['target_equipment_list'])}")
    print(f"  Device: {device}, LR: {lr}")
    print(f"  Buffer: {buffer_capacity}, Batch: {batch_size}")
    print(f"  Target Sync: {target_sync_steps} steps")
    print(f"  N-step: {n_steps}, Cost Leveling: {config['reward']['cost_leveling']['enabled']}")
    print(f"\\nv0.2 Advanced Features:")
    print(f"  [OK] N-step Learning (n={n_steps}) with PrioritizedNStepBuffer")
    print(f"  [OK] AsyncVectorEnv ({n_envs} parallel environments)")
    print(f"  [OK] Mixed Precision Training (AMP) with GradScaler")
    print(f"  [OK] Advanced PER with dynamic beta adjustment")
    print(f"  [OK] Sophisticated QR-DQN loss with importance sampling")
    print(f"{'='*80}\\n")
    
    # Create vectorized environments (AsyncVectorEnv from v0.2)
    env_fns = [
        make_multi_equipment_cbm_env(config_path=config_path, seed=seed + i)
        for i in range(n_envs)
    ]
    envs = AsyncVectorEnv(env_fns)
    
    # Single environment for reference
    single_env = MultiEquipmentCBMEnvironment(config_path=config_path, seed=seed)
    
    state_dim = single_env.observation_space.shape[0]
    n_actions = single_env.action_space.n
    
    print(f"Environment Info:")
    print(f"  State dim: {state_dim}")
    print(f"  Action space: {n_actions}")
    print(f"  Equipment count: {single_env.n_equipment}\\n")
    
    # Initialize networks
    agent_net = MultiEquipmentQRDQN(
        state_dim=state_dim, 
        n_actions=n_actions, 
        n_quantiles=n_quantiles,
        hidden_dim=256  # Larger for multi-equipment
    ).to(device)
    
    target_net = MultiEquipmentQRDQN(
        state_dim=state_dim, 
        n_actions=n_actions, 
        n_quantiles=n_quantiles,
        hidden_dim=256
    ).to(device)
    
    target_net.load_state_dict(agent_net.state_dict())
    target_net.eval()
    
    # Optimizer with weight decay (from v0.2)
    optimizer = optim.AdamW(agent_net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Mixed Precision Training setup (from v0.2)
    scaler = GradScaler('cuda') if device == 'cuda' else None
    
    # Advanced replay buffer (from v0.2)
    buffer = PrioritizedNStepBuffer(
        buffer_capacity, n_steps=n_steps, gamma=single_env.gamma,
        alpha=0.6, beta=0.4, beta_increment=0.001
    )
    
    # Training metrics
    episode_rewards = []
    episode_costs = []
    episode_cost_variances = []
    loss_history = []
    
    # Training tracking (v0.2 style)
    total_steps = 0
    episodes_completed = 0
    start_time = time.time()
    
    # Reset environments
    observations, infos = envs.reset()
    states = observations.astype(np.float32)
    
    # Episode tracking per environment
    env_episode_rewards = np.zeros(n_envs)
    env_episode_costs = np.zeros(n_envs)
    env_episode_cost_variances = np.zeros(n_envs)
    
    pbar = tqdm(total=n_episodes, desc="Training")
    
    while episodes_completed < n_episodes:
        # Reset noise for exploration (Noisy Networks)
        agent_net.reset_noise()
        target_net.reset_noise()
        
        # Select actions for all environments
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(device)
            q_values, _ = agent_net(states_t)
            actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Step all environments
        next_observations, rewards, terminateds, truncateds, infos = envs.step(actions)
        next_states = next_observations.astype(np.float32)
        
        # Store transitions and track metrics
        for i in range(n_envs):
            done = terminateds[i] or truncateds[i]
            buffer.push(states[i], actions[i], rewards[i], next_states[i], done)
            
            env_episode_rewards[i] += rewards[i]
            
            # Handle info dictionary access for parallel environments
            # AsyncVectorEnv returns infos as dict with arrays containing values from all environments
            if isinstance(infos, dict):
                # For parallel environments, info values are arrays with one element per env
                if 'current_month_cost' in infos and len(infos['current_month_cost']) > i:
                    env_episode_costs[i] += infos['current_month_cost'][i]
                if 'cost_variance' in infos and len(infos['cost_variance']) > i:
                    env_episode_cost_variances[i] = infos['cost_variance'][i]
            else:
                # Fallback for list format
                info = infos[i] if i < len(infos) else {}
                if 'current_month_cost' in info:
                    env_episode_costs[i] += info['current_month_cost']
                if 'cost_variance' in info:
                    env_episode_cost_variances[i] = info['cost_variance']
            
            # Episode done
            if done and episodes_completed < n_episodes:
                episode_rewards.append(env_episode_rewards[i])
                episode_costs.append(env_episode_costs[i])
                episode_cost_variances.append(env_episode_cost_variances[i])
                
                env_episode_rewards[i] = 0.0
                env_episode_costs[i] = 0.0
                env_episode_cost_variances[i] = 0.0
                
                episodes_completed += 1
                pbar.update(1)
                
                # Progress logging
                if episodes_completed % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_cost_var = np.mean(episode_cost_variances[-100:]) if episode_cost_variances else 0.0
                    avg_loss = np.mean(loss_history[-1000:]) if loss_history else 0.0
                    elapsed = time.time() - start_time
                    
                    pbar.write(f"\\n[INFO] Episode {episodes_completed}/{n_episodes}")
                    pbar.write(f"   Avg Reward (last 100): {avg_reward:.2f}")
                    pbar.write(f"   Avg Cost Variance (last 100): {avg_cost_var:.2f}")
                    pbar.write(f"   Avg Loss (last 1000): {avg_loss:.4f}")
                    pbar.write(f"   Time: {elapsed:.1f}s ({elapsed/episodes_completed:.3f}s/ep)")
        
        states = next_states
        total_steps += n_envs
        
        # Optimization step
        if len(buffer) >= batch_size:
            sample = buffer.sample(batch_size)
            if sample is not None:
                s_b, a_b, r_b, sn_b, d_b, indices, weights = sample
                
                # Convert to tensors
                s_b_t = torch.FloatTensor(s_b).to(device)
                a_b_t = torch.LongTensor(a_b).to(device)
                r_b_t = torch.FloatTensor(r_b).to(device)
                sn_b_t = torch.FloatTensor(sn_b).to(device)
                d_b_t = torch.FloatTensor(d_b).to(device)
                w_b_t = torch.FloatTensor(weights).to(device)
                
                # Mixed precision training (from v0.2)
                if scaler:
                    with autocast('cuda'):
                        loss, td_errors = quantile_huber_loss_per(
                            agent_net, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t,
                            w_b_t, single_env.gamma, kappa=kappa, n_steps=n_steps
                        )
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(agent_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, td_errors = quantile_huber_loss_per(
                        agent_net, target_net, s_b_t, a_b_t, r_b_t, sn_b_t, d_b_t,
                        w_b_t, single_env.gamma, kappa=kappa, n_steps=n_steps
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent_net.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Update priorities
                buffer.update_priorities(indices, td_errors.cpu().numpy())
                loss_history.append(loss.item())
        
        # Target network sync (step-based like v0.2)
        if total_steps % target_sync_steps == 0:
            target_net.load_state_dict(agent_net.state_dict())
        
        # Save checkpoint (episode-based)
        if episodes_completed % config['output']['save_frequency'] == 0 and episodes_completed > 0:
            torch.save({
                'episode': episodes_completed,
                'total_steps': total_steps,
                'agent_state_dict': agent_net.state_dict(),
                'target_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_costs': episode_costs,
                'episode_cost_variances': episode_cost_variances,
                'loss_history': loss_history,
                'config': config
            }, save_dir / f'checkpoint_episode_{episodes_completed}.pth')
    
    pbar.close()
    
    # Final save
    torch.save(agent_net.state_dict(), save_dir / 'policy_net.pth')
    
    # Save training history
    training_history = {
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs, 
        'episode_cost_variances': episode_cost_variances,
        'loss_history': loss_history,
        'config': config,
        'final_episode': episodes_completed,
        'total_steps': total_steps,
        'training_time': time.time() - start_time
    }
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\\n[COMPLETED] Training completed!")
    print(f"   Final episodes: {episodes_completed}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Training time: {time.time() - start_time:.1f}s")
    print(f"   Final reward: {episode_rewards[-1]:.2f}")
    if episode_cost_variances:
        print(f"   Final cost variance: {episode_cost_variances[-1]:.2f}")
    print(f"   Models saved to: {save_dir}")
    
    # Clean up
    envs.close()
    single_env.close()
    
    return agent_net, training_history


def test_enhanced_environment():
    """Test Multi-Equipment Environment with v0.2 enhancements"""
    env = MultiEquipmentCBMEnvironment(config_path="config_pump_cbm_v047.yaml", seed=42)
    obs, info = env.reset()
    
    print(f"Enhanced Environment test:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Equipment count: {env.n_equipment}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step}: Action {action}, Reward {reward:.2f}, Cost {info['current_month_cost']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Multi-Equipment CBM QR-DQN Training")
    parser.add_argument("--config", type=str, default="config_pump_cbm_v047.yaml", help="Config file path")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes")
    parser.add_argument("--envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--test", action="store_true", help="Test environment only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if args.test:
        test_enhanced_environment()
    else:
        train_multi_equipment_dqn_enhanced(
            config_path=args.config,
            n_episodes=args.episodes,
            n_envs=args.envs,
            lr=args.lr,
            device=args.device
        )