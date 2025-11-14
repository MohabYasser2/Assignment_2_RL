"""
Double Deep Q-Network (DDQN) Agent Implementation.

This module implements DDQN, an improvement over DQN that addresses
the overestimation bias by decoupling action selection from action evaluation.

Key difference from DQN:
- DQN:  Q_target = r + γ * max_a' Q_target(s', a')
- DDQN: Q_target = r + γ * Q_target(s', argmax_a' Q_policy(s', a'))

The policy network selects the action, while the target network evaluates it.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dqn_agent import DQNAgent
from replay_memory import Transition

# ============================================================================
# DDQN AGENT
# ============================================================================

class DDQNAgent(DQNAgent):
    """
    Double DQN Agent - inherits all methods from DQNAgent except train_step.
    
    The only difference from DQN is in the Q-learning update:
    - Uses policy network to select the best action
    - Uses target network to evaluate that action's value
    
    This reduces overestimation bias present in standard DQN.
    """
    
    def train_step(self):
        """
        Perform one step of Double Q-learning update.
        
        DDQN update rule:
        Q(s,a) <- Q(s,a) + α * (r + γ * Q_target(s', argmax_a' Q_policy(s',a')) - Q(s,a))
        
        Returns:
            float: Loss value, or None if not enough samples
        """
        if not self.can_optimize():
            return None
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create mask for non-terminal states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.as_tensor(np.stack([s for s in batch.next_state if s is not None]), dtype=torch.float32, device=self.device) if non_final_mask.any() else None
        
        # Convert batch to tensors
        state_batch = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Compute Q(s,a) for taken actions
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # DDQN: Use policy network to select actions, target network to evaluate
        next_q = torch.zeros((self.batch_size, 1), device=self.device)
        if non_final_mask.any():
            # Select best actions using policy network
            next_actions = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
            # Evaluate selected actions using target network
            target_vals = self.target_net(non_final_next_states).gather(1, next_actions)
            next_q[non_final_mask] = target_vals
        
        # Compute target: r + γ * Q_target(s', argmax_a' Q_policy(s',a'))
        target = reward_batch + self.gamma * next_q
        
        # Compute Smooth L1 loss (Huber loss)
        loss = F.smooth_l1_loss(q_values, target)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()
        
        return float(loss.item())
