"""
Deep Q-Network (DQN) Agent Implementation.

This module implements a DQN agent with:
- Epsilon-greedy exploration strategy
- Experience replay memory
- Target network with soft updates
- Smooth L1 loss (Huber loss)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from DQN import DQN
from replay_memory import ReplayMemory, Transition

# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration and soft target updates.
    
    The agent uses two networks:
    - Policy network: Used for action selection and updated every step
    - Target network: Used for computing target Q-values, updated slowly
    
    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Number of possible actions
        hparams (dict): Hyperparameters dictionary
        device (torch.device, optional): Device to run computations on
    """
    
    def __init__(self, state_dim, action_dim, hparams, device=None):
        # Environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Training hyperparameters
        self.gamma = hparams.get('gamma', 0.99)                      # Discount factor
        self.epsilon_start = hparams.get('epsilon_start', 0.9)       # Initial exploration rate
        self.epsilon_end = hparams.get('epsilon_end', 0.01)          # Minimum exploration rate
        self.epsilon_decay = hparams.get('epsilon_decay', 2500)      # Decay rate for epsilon
        self.lr = hparams.get('learning_rate', 1e-3)                 # Learning rate
        self.batch_size = hparams.get('batch_size', 128)             # Batch size for training
        self.tau = hparams.get('tau', 0.005)                         # Soft update parameter
        
        # Device setup (GPU if available, otherwise CPU)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize policy and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer with AMSGrad variant of Adam
        self.optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        
        # Experience replay memory
        self.memory = ReplayMemory(hparams.get('memory_size', 10000))
        
        # Step counter for epsilon decay
        self.steps_done = 0

    # ========================================================================
    # EXPLORATION STRATEGY
    # ========================================================================
    
    def eps_threshold(self):
        """
        Calculate current epsilon value using exponential decay.
        
        Returns:
            float: Current epsilon threshold
        """
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)

    def select_action(self, state, epsilon_unused=None):
        """
        Select action using epsilon-greedy strategy.
        
        With probability epsilon: random action (exploration)
        With probability 1-epsilon: greedy action (exploitation)
        
        Args:
            state (np.ndarray): Current state observation
            epsilon_unused: Unused parameter (for compatibility)
            
        Returns:
            int: Selected action
        """
        eps = self.eps_threshold()
        self.steps_done += 1
        
        # Exploration: random action
        if random.random() < eps:
            return random.randrange(self.action_dim)
        
        # Exploitation: greedy action based on Q-values
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in replay memory.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode ended
        """
        ns = None if done else next_state
        self.memory.push(state, action, ns, reward)

    # ========================================================================
    # NETWORK UPDATES
    # ========================================================================
    
    def soft_update(self):
        """
        Soft update of target network parameters.
        
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def can_optimize(self):
        """Check if enough experiences are available for training."""
        return len(self.memory) >= self.batch_size

    # ========================================================================
    # TRAINING
    # ========================================================================
    
    def train_step(self):
        """
        Perform one step of Q-learning update.
        
        DQN update rule:
        Q(s,a) <- Q(s,a) + α * (r + γ * max_a' Q_target(s',a') - Q(s,a))
        
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
        
        # Compute max_a' Q_target(s',a') for next states
        next_q = torch.zeros((self.batch_size, 1), device=self.device)
        if non_final_mask.any():
            next_vals = self.target_net(non_final_next_states).max(dim=1, keepdim=True)[0]
            next_q[non_final_mask] = next_vals
        
        # Compute target: r + γ * max_a' Q_target(s',a')
        target = reward_batch + self.gamma * next_q
        
        # Compute Smooth L1 loss (Huber loss)
        loss = F.smooth_l1_loss(q_values, target)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()
        
        return float(loss.item())

    def update_target(self):
        """Update target network (called after each training step)."""
        self.soft_update()
