"""
Configuration file for DQN/DDQN training on various Gymnasium environments.

This file contains:
- Default hyperparameters for all agents
- Environment-specific configurations including success thresholds and max episodes
"""

# ============================================================================
# DEFAULT HYPERPARAMETERS
# ============================================================================
# These hyperparameters are used as defaults for all environments unless overridden

HYPERPARAMS = {
    'gamma': 0.99,                  # Discount factor for future rewards
    'epsilon_start': 1,              # Initial exploration rate (100%)
    'epsilon_end': 0.01,             # Minimum exploration rate (1%)
    'epsilon_decay': 2500,           # Steps over which epsilon decays exponentially
    'learning_rate': 0.001,          # Adam optimizer learning rate
    'memory_size': 10000,            # Replay buffer capacity
    'batch_size': 64,                # Number of transitions sampled for each training step
    'target_update_freq': 10,        # Frequency of hard target network updates (not used with soft updates)
    'tau': 0.005                     # Soft update parameter (target = tau*policy + (1-tau)*target)
}

# ============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ============================================================================
# Each environment has its own success criteria and can override default hyperparameters

CONFIG = {
    # CartPole: Balance a pole on a cart
    'CartPole-v1': {
        'max_episodes': 500,
        'success_threshold': 475,    # Average reward over last 100 episodes
        'hyperparams': {
            **HYPERPARAMS,
            'epsilon_decay': 2000,   # Faster exploration decay
            'learning_rate': 0.0007  # Lower learning rate for stability
        }
    },
    
    # Acrobot: Swing up a two-link robot arm
    'Acrobot-v1': {
        'max_episodes': 1000,
        'success_threshold': -100,   # Negative rewards; closer to 0 is better
        'hyperparams': {
            **HYPERPARAMS,
            'epsilon_end': 0.05,      # Keep 5% exploration to avoid policy collapse
            'epsilon_decay': 5000,     # Slower epsilon decay
            'learning_rate': 0.0005    # Lower learning rate for stability
        }
    },
    
    # MountainCar: Drive up a steep hill
    'MountainCar-v0': {
        'max_episodes': 1000,
        'success_threshold': -110,   # Negative rewards; closer to 0 is better
        'hyperparams': {
            **HYPERPARAMS
        }
    },
    
    # Pendulum: Keep a pendulum upright (discretized version)
    'Pendulum-v1': {
        'max_episodes': 500,
        'success_threshold': -200,   # Negative rewards; closer to 0 is better
        'hyperparams': {
            **HYPERPARAMS
        }
    }
}
