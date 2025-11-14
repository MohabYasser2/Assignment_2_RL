"""
Deep Q-Network (DQN) Neural Network Architecture.

This module defines the neural network structure used for approximating
the Q-value function in DQN and DDQN agents.
"""

import torch.nn as nn

# ============================================================================
# DQN NETWORK
# ============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network with two hidden layers.
    
    Architecture:
    - Input layer: state_dim neurons
    - Hidden layer 1: 256 neurons with ReLU activation
    - Hidden layer 2: 256 neurons with ReLU activation
    - Output layer: action_dim neurons (Q-values for each action)
    
    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Number of possible actions
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.net(x)
