"""
Training Script for DQN/DDQN Agents on Gymnasium Environments.

This script:
- Trains DQN and DDQN agents on various Gymnasium environments
- Logs training metrics to Weights & Biases
- Saves successful models to disk
- Provides comprehensive console and file logging
"""

import os
import numpy as np
import random
import torch
import wandb
import gymnasium as gym
from collections import deque
from datetime import datetime
import shutil

from config import CONFIG
from discrete_pendulum import make_pendulum
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent

# Weights & Biases project name
PROJECT = "RL_ASSIGNMENT2"

# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_env(env_id, render_mode=None):
    """
    Create a Gymnasium environment.
    
    Args:
        env_id (str): Environment identifier
        render_mode (str, optional): Rendering mode
        
    Returns:
        gym.Env: Gymnasium environment
    """
    if env_id == "Pendulum-v1":
        return make_pendulum(num_discrete_actions=5, render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one(env_id, AgentClass, log_file):
    """
    Train a single agent on a single environment.
    
    Args:
        env_id (str): Environment identifier
        AgentClass (class): Agent class to train (DQNAgent or DDQNAgent)
        log_file (file): File handle for logging
        
    Returns:
        tuple: (success, episodes_trained, mean_reward_100, model_path)
    """
    # Display training banner
    print(f"\n{'='*60}")
    print(f"  {AgentClass.__name__} on {env_id}")
    print(f"{'='*60}")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    env = make_env(env_id)
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    
    # Additional seeds for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load hyperparameters and create agent
    hyperparams = CONFIG[env_id]["hyperparams"]
    agent = AgentClass(state_dim, action_dim, hyperparams)
    
    # Get training configuration
    threshold = CONFIG[env_id]['success_threshold']
    max_episodes = CONFIG[env_id]['max_episodes']
    
    # Display minimal configuration
    print(f"  Target: {threshold} | Max Episodes: {max_episodes}")
    
    # Initialize Weights & Biases (suppress verbose output)
    import os
    os.environ['WANDB_SILENT'] = 'true'
    run = wandb.init(project=PROJECT, name=f"{AgentClass.__name__}-{env_id}", config={**hyperparams, "env_id": env_id, "agent": AgentClass.__name__})
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    # Training state variables
    last100 = deque(maxlen=100)  # Rolling window of last 100 episode returns
    last100_times = deque(maxlen=100)  # Rolling window of episode durations
    success = False
    episodes_trained = 0
    model_path = None
    total_steps = 0
    
    for ep in range(1, max_episodes + 1):
        # Reset environment for new episode
        episode_start_time = datetime.now()
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_reward_sum = 0.0
        ep_steps = 0
        ep_losses = []
        
        # Episode loop
        while not done:
            # Select and execute action
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward shaping for MountainCar
            if env_id == "MountainCar-v0":
                reward += abs(next_obs[0] + 0.5)
            
            done = terminated or truncated
            
            # Store transition in replay memory
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train if enough experiences collected
            if len(agent.memory) > agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)
            
            # Update target network
            agent.update_target()
            
            # Accumulate reward and move to next state
            ep_return += reward
            ep_reward_sum += reward
            ep_steps += 1
            obs = next_obs if not done else obs
            total_steps += 1
        
        # Calculate episode metrics
        episode_duration = (datetime.now() - episode_start_time).total_seconds()
        last100.append(ep_return)
        last100_times.append(episode_duration)
        mean_reward_100 = float(np.mean(last100))
        avg_episode_time = float(np.mean(last100_times))
        avg_loss = float(np.mean(ep_losses)) if len(ep_losses) > 0 else 0.0
        current_epsilon = agent.eps_threshold()
        
        # Sample Q-values for collapse detection
        with torch.no_grad():
            sample_obs = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            sample_q = agent.policy_net(sample_obs)[0]
            q_max = float(sample_q.max().item())
            q_min = float(sample_q.min().item())
            q_mean = float(sample_q.mean().item())
            q_std = float(sample_q.std().item())
        
        # Log metrics to wandb with organized naming for better visualization
        wandb.log({
            # Episode Metrics
            "Training/Episode": ep,
            "Training/Total_Steps": total_steps,
            
            # Performance Metrics
            "Performance/Episode_Return": ep_return,
            "Performance/Avg_Return_100": mean_reward_100,
            "Performance/Best_Return_100": max(last100) if len(last100) > 0 else ep_return,
            
            # Learning Metrics
            "Learning/Loss": avg_loss,
            "Learning/Epsilon": current_epsilon,
            "Learning/Memory_Size": len(agent.memory),
            
            # Q-Value Statistics (to detect collapse)
            "Q_Values/Mean": q_mean,
            "Q_Values/Std": q_std,
            "Q_Values/Max": q_max,
            "Q_Values/Min": q_min,
            "Q_Values/Range": q_max - q_min,
            
            # Episode Stats
            "Stats/Episode_Length": ep_steps,
            "Stats/Episode_Duration": episode_duration
        })
        
        episodes_trained = ep
        
        # Progress logging every 50 episodes
        if ep % 50 == 0:
            q_range = q_max - q_min
            collapse_warning = " [WARNING: Q-collapse!]" if q_std < 0.1 else ""
            print(f"  [{ep:3d}] Avg: {mean_reward_100:6.2f} | Loss: {avg_loss:.4f} | Eps: {current_epsilon:.3f} | Q_std: {q_std:.3f}{collapse_warning}")
            log_file.write(f"[{AgentClass.__name__} | {env_id}] ep={ep} mean_reward_100={mean_reward_100:.2f} loss={avg_loss:.4f} epsilon={current_epsilon:.3f} q_std={q_std:.3f}\n")
            log_file.flush()
        
        # Check for success (threshold reached)
        if len(last100) == 100 and mean_reward_100 > threshold:
            success = True
            print(f"\n  SUCCESS at episode {ep} | Mean Reward (last 100): {mean_reward_100:.2f}")
            
            # Save model
            save_dir = os.path.join("models", AgentClass.__name__)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{env_id}.pth")
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"  Model saved: {model_path}")
            
            log_file.write(f"[{AgentClass.__name__} | {env_id}] success achieved at ep={ep}. model saved to {model_path}\n")
            log_file.flush()
            break
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    
    # Update wandb summary
    wandb.summary['success'] = success
    wandb.summary['episodes_trained'] = episodes_trained
    wandb.summary['mean_reward_100'] = float(np.mean(last100)) if len(last100) > 0 else None
    
    # Upload model to wandb if successful
    if model_path is not None:
        wandb_model_dir = os.path.join(wandb.run.dir, "models", AgentClass.__name__)
        os.makedirs(wandb_model_dir, exist_ok=True)
        wandb_model_path = os.path.join(wandb_model_dir, os.path.basename(model_path))
        shutil.copy2(model_path, wandb_model_path)
    else:
        print(f"  FAILED | Final Mean Reward (last 100): {float(np.mean(last100)) if len(last100) > 0 else 'N/A'}")
    
    # Cleanup
    run.finish()
    env.close()
    print(f"  Completed\n")
    
    return success, episodes_trained, float(np.mean(last100)) if len(last100) > 0 else None, model_path

# ============================================================================
# BATCH TRAINING FUNCTION
# ============================================================================

def train_all(log_file):
    """
    Train all agents on all environments (unused in current main).
    
    Args:
        log_file (file): File handle for logging
        
    Returns:
        list: List of result dictionaries
    """
    results = []
    for AgentClass in [DQNAgent, DDQNAgent]:
        for env_id in CONFIG.keys():
            success, episodes, mean_reward_100, model_path = train_one(env_id, AgentClass, log_file)
            results.append({
                "agent": AgentClass.__name__,
                "env": env_id,
                "success": success,
                "episodes_trained": episodes,
                "mean_reward_100": mean_reward_100,
                "model_path": model_path
            })
            log_file.write(f"{AgentClass.__name__} on {env_id}: success={success}, episodes={episodes}, mean_reward_100={mean_reward_100}, model_path={model_path}\n")
            log_file.flush()
    log_file.write("\nFinal Training Summary:\n")
    for r in results:
        log_file.write(f"{r['agent']} on {r['env']}: success={r['success']}, episodes={r['episodes_trained']}, mean_reward_100={r['mean_reward_100']}, model_path={r['model_path']}\n")
    log_file.flush()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    with open("output.log", "w", encoding="utf-8") as log_file:
        # Training pipeline start
        start_time = datetime.now()
        print(f"\n{'#'*60}")
        print(f"Training Pipeline Started")
        print(f"Time: {start_time}")
        print(f"{'#'*60}\n")
        log_file.write(f"Training started at {start_time}\n\n")
        
        # Train all agents on all environments
        results = []
        for AgentClass in [DQNAgent, DDQNAgent]:
            for env_id in CONFIG.keys():
                success, episodes, mean_reward_100, model_path = train_one(env_id, AgentClass, log_file)
                results.append({
                    "agent": AgentClass.__name__,
                    "env": env_id,
                    "success": success,
                    "episodes_trained": episodes,
                    "mean_reward_100": mean_reward_100,
                    "model_path": model_path
                })
                log_file.write(f"{AgentClass.__name__} on {env_id}: success={success}, episodes={episodes}, mean_reward_100={mean_reward_100}, model_path={model_path}\n")
                log_file.flush()
        
        # Training pipeline completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Display summary
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*60}")
        for r in results:
            status = "SUCCESS" if r['success'] else "FAILED "
            print(f"  [{status}] {r['agent']:10s} | {r['env']:15s} | Mean Reward: {r['mean_reward_100']:6.2f} | Ep: {r['episodes_trained']:3d}")
        print(f"  Duration: {duration}")
        print(f"{'='*60}\n")
        
        # Write summary to log file
        log_file.write("\nFinal Training Summary:\n")
        for r in results:
            log_file.write(f"{r['agent']} on {r['env']}: success={r['success']}, episodes={r['episodes_trained']}, mean_reward_100={r['mean_reward_100']}, model_path={r['model_path']}\n")
        log_file.write(f"\nTraining completed at {end_time}\n")
        log_file.write(f"Total duration: {duration}\n")
        log_file.flush()
