"""
Test Script for Evaluating Trained DQN/DDQN Agents.

This script:
- Loads trained models from the 'models/' directory
- Evaluates them for multiple episodes
- Records videos of the first few episodes
- Logs results to Weights & Biases
"""

import os
import torch
import wandb
import gymnasium as gym
import numpy as np
from datetime import datetime
from gymnasium.wrappers import RecordVideo

from config import CONFIG
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from discrete_pendulum import make_pendulum

# Weights & Biases project name
PROJECT = "RL_ASSIGNMENT2_TEST"

# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_env(env_id, record=False):
    """
    Create the environment, with optional video recording.
    
    Args:
        env_id (str): Gymnasium environment ID
        record (bool): Whether to enable video recording
        
    Returns:
        gym.Env: Gymnasium environment
    """
    render_mode = "rgb_array" if record else None
    if env_id == "Pendulum-v1":
        return make_pendulum(num_discrete_actions=5, render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_agent(env_id, AgentClass, model_path):
    """
    Load a saved model into the given AgentClass.
    
    Args:
        env_id (str): Environment ID
        AgentClass (class): Agent class (DQNAgent or DDQNAgent)
        model_path (str): Path to saved model weights
        
    Returns:
        Agent: Loaded agent with trained weights
    """
    env = make_env(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hyperparams = CONFIG[env_id]["hyperparams"]

    # Create agent and load weights
    agent = AgentClass(state_dim, action_dim, hyperparams)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()  # Set to evaluation mode
    
    # Verify model loaded correctly
    print(f"Model loaded from: {model_path}")
    print(f"Network has {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
    
    env.close()
    return agent

# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_agent(env_id, AgentClass, model_path, num_episodes=100, num_recordings=5):
    """
    Evaluate the trained model for a number of episodes.
    
    Args:
        env_id (str): Environment ID
        AgentClass (class): Agent class (DQNAgent or DDQNAgent)
        model_path (str): Path to saved model weights
        num_episodes (int): Number of episodes to test
        num_recordings (int): Number of episodes to record as video
    """
    # Create the environment with video recording
    video_folder = os.path.join("videos", f"{AgentClass.__name__}_{env_id}")
    os.makedirs(video_folder, exist_ok=True)
    env = make_env(env_id, record=True)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda i: i < num_recordings,  # Record first N episodes
        name_prefix=f"{AgentClass.__name__}_{env_id}"
    )

    # Load trained agent
    agent = load_agent(env_id, AgentClass, model_path)
    
    # Set seed for reproducible testing (same as training)
    test_seed = 42
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)

    # Setup WandB run
    run = wandb.init(
        project=PROJECT,
        name=f"Test-{AgentClass.__name__}-{env_id}",
        config={
            "env_id": env_id,
            "agent": AgentClass.__name__,
            "episodes": num_episodes,
            "model_path": model_path,
        },
    )

    # Run test episodes
    test_start_time = datetime.now()
    all_returns = []
    all_episode_times = []
    for ep in range(1, num_episodes + 1):
        episode_start_time = datetime.now()
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        # Run episode using greedy policy (no exploration)
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.policy_net(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            obs = next_obs

        episode_duration = (datetime.now() - episode_start_time).total_seconds()
        all_returns.append(ep_return)
        all_episode_times.append(episode_duration)
        wandb.log({"episode": ep, "return": ep_return, "episode_time": episode_duration})

        # Progress logging every 10 episodes
        if ep % 10 == 0:
            avg_time = np.mean(all_episode_times[-10:])
            print(f"[{AgentClass.__name__} | {env_id}] Episode {ep}/{num_episodes} avg_return={np.mean(all_returns[-10:]):.2f} avg_time={avg_time:.2f}s")

    # Compute and log final statistics
    test_duration = (datetime.now() - test_start_time).total_seconds()
    avg_return = np.mean(all_returns)
    avg_episode_time = np.mean(all_episode_times)
    total_episode_time = np.sum(all_episode_times)
    
    wandb.summary["avg_return"] = avg_return
    wandb.summary["num_recorded_episodes"] = num_recordings
    wandb.summary["test_duration"] = test_duration
    wandb.summary["avg_episode_time"] = avg_episode_time
    wandb.summary["total_episode_time"] = total_episode_time

    print(f"\n[{AgentClass.__name__} | {env_id}] Test completed.")
    print(f"  Avg return: {avg_return:.2f}")
    print(f"  Test duration: {test_duration:.2f}s")
    print(f"  Avg episode time: {avg_episode_time:.3f}s")
    print(f"  Videos saved in: {video_folder}")

    env.close()
    run.finish()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)

    # Test all agents on all environments
    for AgentClass in [DQNAgent, DDQNAgent]:
        for env_id in CONFIG.keys():
            model_path = os.path.join("models", AgentClass.__name__, f"{env_id}.pth")
            if os.path.exists(model_path):
                print(f"\nTesting {AgentClass.__name__} on {env_id} ...")
                test_agent(env_id, AgentClass, model_path)
            else:
                print(f"Warning: Model not found: {model_path}")