"""
Experience Replay Memory for DQN/DDQN Training.

This module implements a circular buffer to store and sample experiences
for training deep reinforcement learning agents.
"""

import random
from collections import deque, namedtuple

# ============================================================================
# TRANSITION DEFINITION
# ============================================================================

# Named tuple for storing a single transition (experience)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ============================================================================
# REPLAY MEMORY
# ============================================================================

class ReplayMemory:
    """
    Fixed-size buffer to store experience tuples.
    
    The replay memory stores transitions and provides random sampling
    for breaking correlation between consecutive experiences during training.
    
    Args:
        capacity (int): Maximum number of transitions to store
    """
    
    def __init__(self, capacity):
        """Initialize replay buffer with fixed capacity."""
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """
        Store a new transition in memory.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation (or None if terminal)
            reward: Reward received
        """
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            list: Random sample of transitions
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
