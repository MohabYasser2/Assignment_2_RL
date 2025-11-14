# Deep Q-Network (DQN) & Double DQN (DDQN) - Comprehensive Documentation

## Table of Contents

1. [Algorithm Overview & Flow](#algorithm-overview--flow)
2. [Hyperparameters Explained](#hyperparameters-explained)
3. [File-by-File Analysis](#file-by-file-analysis)
4. [Network Architecture Details](#network-architecture-details)

---

## Algorithm Overview & Flow

### **DQN Algorithm (Deep Q-Network)**

DQN combines Q-Learning with deep neural networks to handle high-dimensional state spaces. The algorithm learns an action-value function Q(s,a) that estimates the expected cumulative reward.

**Core Update Rule:**

```
Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q_target(s',a') - Q(s,a)]
```

**Key Components:**

1. **Policy Network** - Makes decisions and gets updated frequently
2. **Target Network** - Provides stable Q-value targets
3. **Experience Replay** - Stores and randomly samples past experiences
4. **Epsilon-Greedy** - Balances exploration vs exploitation

**Training Flow:**

```
1. Initialize policy network and target network (same weights)
2. Initialize replay memory buffer
3. For each episode:
   a. Reset environment
   b. For each step:
      - Select action (ε-greedy: random or best Q-value)
      - Execute action, observe reward and next state
      - Store transition in replay memory
      - Sample random batch from memory
      - Compute target: r + γ * max Q_target(s',a')
      - Compute loss: Huber(Q_policy(s,a), target)
      - Update policy network via gradient descent
      - Soft update target network: θ_target ← τ*θ_policy + (1-τ)*θ_target
   c. Check if success threshold reached
```

### **DDQN Algorithm (Double Deep Q-Network)**

DDQN improves upon DQN by addressing **overestimation bias**. Standard DQN uses the same network for both selecting and evaluating actions, leading to optimistic Q-values.

**Key Difference:**

- **DQN:** Uses target network for both action selection AND evaluation
- **DDQN:** Uses policy network to SELECT action, target network to EVALUATE it

**DDQN Update Rule:**

```
Q(s,a) ← Q(s,a) + α [r + γ * Q_target(s', argmax_a' Q_policy(s',a')) - Q(s,a)]
                                    ↑                    ↑
                                 evaluate          select action
                              (target net)        (policy net)
```

**Why This Helps:**

- Policy network may be noisy and overestimate some actions
- Target network provides more conservative evaluation
- Decoupling reduces positive bias in Q-value estimates

---

## Hyperparameters Explained

### **1. Gamma (γ) = 0.99**

**What:** Discount factor for future rewards  
**Range:** [0, 1]  
**Why 0.99:**

- Values near 1 make agent far-sighted (considers long-term consequences)
- γ=0.99 means rewards 100 steps away are worth ~37% of immediate rewards (0.99^100)
- Perfect for episodic tasks where final goal matters (CartPole: stay balanced long-term)
- Lower values (0.9) would make agent myopic, only caring about immediate rewards
- Standard in RL literature; proven effective across many domains

**Mathematical Impact:**

```
Return = r_0 + γ*r_1 + γ²*r_2 + γ³*r_3 + ...
With γ=0.99: Return = r_0 + 0.99*r_1 + 0.98*r_2 + 0.97*r_3 + ...
```

### **2. Epsilon Start = 1.0 (100%)**

**What:** Initial exploration probability  
**Why 1.0:**

- Agent knows nothing at start → must explore randomly
- Pure exploration in early episodes prevents premature convergence
- Ensures diverse experiences fill replay buffer
- Prevents agent from exploiting random, suboptimal patterns

### **3. Epsilon End = 0.01 (1%)**

**What:** Minimum exploration probability  
**Why 0.01 (not 0):**

- Always maintain small exploration to discover better strategies
- Environments may have stochastic elements
- Prevents getting stuck in local optima
- 1% is low enough to mostly exploit learned policy
- **Exception:** Acrobot uses 0.05 (5%) because it's harder and needs more exploration

### **4. Epsilon Decay = 2500 steps**

**What:** Rate of exponential decay from epsilon_start to epsilon_end  
**Formula:** `ε = ε_end + (ε_start - ε_end) * exp(-steps / decay)`  
**Why 2500:**

- Gradual transition from exploration to exploitation
- After 2500 steps: ε ≈ 0.37 (middle ground)
- After 5000 steps: ε ≈ 0.14 (mostly exploitation)
- After 10000 steps: ε ≈ 0.02 (nearly pure exploitation)
- Matches typical episode length × number of episodes for convergence

**Environment-Specific Adjustments:**

- **CartPole:** 2000 (faster decay) - simpler environment, learns quickly
- **Acrobot:** 5000 (slower decay) - harder environment, needs more exploration time

### **5. Learning Rate = 0.001 (1e-3)**

**What:** Step size for gradient descent optimizer  
**Why 0.001:**

- Standard for Adam optimizer (proven effective)
- Too high (0.01): unstable training, overshooting minima, oscillations
- Too low (0.0001): very slow learning, may not converge in reasonable time
- Neural networks are non-convex → need careful tuning

**Environment Adjustments:**

- **CartPole:** 0.0007 - Lower for stability (simple but sensitive)
- **Acrobot:** 0.0005 - Even lower (complex dynamics, negative rewards)

**Why Lower Rates for Harder Environments:**

- Acrobot has sparse rewards (-1 per step, 0 at goal)
- Small gradients need careful updates
- Prevents Q-value collapse (all Q-values → same number)

### **6. Memory Size = 10,000 transitions**

**What:** Replay buffer capacity  
**Why 10,000:**

- **Memory efficiency:** Each transition stores (state, action, next_state, reward)
  - CartPole: 4D state → ~8 floats per transition → ~80KB total
  - Stores enough diversity without excessive RAM
- **Statistical coverage:** 10K samples provide good state-space coverage
- **Recency balance:** Old experiences gradually replaced, keeping data fresh
- **Computational cost:** Larger buffers don't significantly improve performance but slow sampling

**Alternative Considerations:**

- Too small (1000): Limited diversity, may overfit recent experiences
- Too large (100K): Slow outdated experiences removal, more RAM, no benefit
- Research shows diminishing returns beyond 10K-50K for small state spaces

### **7. Batch Size = 64**

**What:** Number of transitions sampled per training step  
**Why 64:**

- **Gradient stability:** Averages over 64 samples reduces variance
- **Hardware efficiency:** Power of 2 → GPU optimization, memory alignment
- **Breaking correlations:** Random samples from 10K pool → i.i.d. assumption
- **Update frequency:** Small enough for frequent updates (every step)

**Size Tradeoffs:**

- **Small (16):** High variance gradients, unstable learning, but faster
- **Large (256):** Stable but slower, needs bigger replay buffer
- **64 is sweet spot:** Used in original DQN paper, proven effective

### **8. Tau (τ) = 0.005**

**What:** Soft update coefficient for target network  
**Formula:** `θ_target ← τ*θ_policy + (1-τ)*θ_target`  
**Why 0.005:**

- **Stability:** Target network updates slowly, providing stable TD targets
- **Prevents moving target problem:** If target updates too fast, chasing moving goal
- **Mathematical insight:** After N steps, new weights contribute: 1 - (1-τ)^N
  - After 200 steps: ~63% from policy network
  - After 500 steps: ~92% from policy network
  - After 1000 steps: ~99% from policy network

**Alternative: Hard Updates:**

- Old approach: Copy weights every N steps (e.g., N=10)
- Soft updates (τ=0.005) are smoother, more stable, modern standard

**Why Not 0.001 or 0.01:**

- 0.001: Too slow, target lags behind too much
- 0.01: Too fast, loses stability benefits
- 0.005: Proven optimal in DDQN paper

### **9. Target Update Frequency = 10 (UNUSED)**

**What:** Legacy parameter for hard target updates  
**Why Unused:**

- Project uses soft updates (tau) instead
- Hard updates: copy every 10 steps
- Soft updates: blend every step
- Soft updates are superior (smoother, more stable)

---

## File-by-File Analysis

### **1. config.py - Configuration Central**

**Purpose:** Single source of truth for all hyperparameters and environment settings.

**Key Components:**

#### `HYPERPARAMS` Dictionary

- **Default values** applied to all environments
- Creates consistency across experiments
- Easy to modify for global changes

#### `CONFIG` Dictionary

- **Environment-specific overrides**
- Each environment has unique:
  - `max_episodes`: Training budget
  - `success_threshold`: Convergence criterion
  - `hyperparams`: Custom tuning

#### Environment-Specific Rationale:

**CartPole-v1:**

```python
'success_threshold': 475  # Nearly perfect (max ~500)
'max_episodes': 500       # Simple task, converges fast
'epsilon_decay': 2000     # Faster exploration→exploitation
'learning_rate': 0.0007   # Lower for stability
```

- Simple 4D state, 2 actions
- Learns quickly but sensitive to instability
- Lower LR prevents oscillations

**Acrobot-v1:**

```python
'success_threshold': -100  # Sparse rewards
'max_episodes': 1000       # Harder, needs more time
'epsilon_end': 0.05        # More exploration to avoid collapse
'epsilon_decay': 5000      # Slow decay
'learning_rate': 0.0005    # Very careful updates
```

- 6D state, sparse negative rewards
- Requires careful tuning to avoid Q-collapse
- Higher epsilon_end maintains exploration

**MountainCar-v0:**

```python
'success_threshold': -110  # Maximum penalty
'max_episodes': 1000       # Sparse rewards, difficult
# Uses default hyperparams
```

- 2D state but challenging dynamics
- Reward shaping applied in train.py (adds |x+0.5|)
- Default params sufficient with reward shaping

**Pendulum-v1:**

```python
'success_threshold': -200  # Continuous penalty
'max_episodes': 500        # Moderate difficulty
# Uses default hyperparams
# CONTINUOUS → DISCRETE conversion via wrapper
```

- Discretized to 5 actions
- Default params work well

---

### **2. DQN.py - Neural Network Architecture**

**Purpose:** Defines the function approximator that estimates Q(s,a).

#### Class: `DQN(nn.Module)`

**Architecture:**

```
Input: state_dim (e.g., 4 for CartPole)
  ↓
Linear(state_dim → 256)  [Weight matrix: state_dim × 256]
  ↓
ReLU()                    [max(0, x) - introduces non-linearity]
  ↓
Linear(256 → 256)         [Weight matrix: 256 × 256]
  ↓
ReLU()
  ↓
Linear(256 → action_dim)  [Weight matrix: 256 × action_dim]
  ↓
Output: Q-values [Q(s,a₀), Q(s,a₁), ..., Q(s,aₙ)]
```

**Why This Architecture:**

**Two Hidden Layers:**

- **Not too shallow:** Single layer can't capture complex patterns
- **Not too deep:** 2 layers sufficient for small state spaces (4-6D)
- Deeper networks risk overfitting on small datasets
- Proven effective in original DQN paper

**256 Neurons per Layer:**

- **Capacity:** Enough to represent complex Q-functions
- **Not excessive:** Avoids overfitting, trains faster
- **Standard choice:** Used in many RL implementations
- **Scaling:** For image inputs, would use Conv layers instead

**ReLU Activation:**

- **Formula:** f(x) = max(0, x)
- **Why not Sigmoid/Tanh:**
  - No vanishing gradient problem
  - Faster computation
  - Sparsity (many neurons = 0)
- **Why ReLU specifically:**
  - Standard for deep learning since 2012
  - Dead ReLU problem minimal with good initialization

**No Activation on Output:**

- Q-values can be positive or negative
- Need full range (-∞, +∞)
- Activation would constrain values

**Functions:**

#### `__init__(state_dim, action_dim)`

- **Initializes network** with PyTorch's default initialization (Kaiming uniform for Linear layers)
- **Sequential container** simplifies forward pass
- **Automatically registers parameters** for optimizer

#### `forward(x)`

- **Simple pass-through:** x → net(x) → Q-values
- **No explicit softmax:** Not computing policy, just Q-values
- **Batch compatible:** Works for single state or batch of states

---

### **3. replay_memory.py - Experience Replay Buffer**

**Purpose:** Store and sample past experiences to break temporal correlations.

#### `Transition` (NamedTuple)

```python
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
```

- **Immutable container** for a single experience
- **Named access:** transition.state instead of transition[0]
- **Memory efficient:** Faster than dict, cleaner than tuple

#### Class: `ReplayMemory`

**Data Structure:**

```python
self.memory = deque(maxlen=capacity)
```

- **Deque (double-ended queue):** O(1) append and pop
- **maxlen:** Automatically removes oldest when full (circular buffer)
- **FIFO:** New experiences push out old ones

**Why Experience Replay:**

**Problem Without Replay:**

```
Step t:   state s₁ → train on (s₁, a₁, r₁, s₂)
Step t+1: state s₂ → train on (s₂, a₂, r₂, s₃)
Step t+2: state s₃ → train on (s₃, a₃, r₃, s₄)
```

- **Highly correlated:** Consecutive states are similar
- **Catastrophic forgetting:** Network forgets old states
- **Non-stationary distribution:** Data distribution shifts as policy changes
- **Violates i.i.d. assumption:** SGD assumes independent samples

**Solution With Replay:**

```
Random sample from buffer:
  (s₁₀₀, a₁₀₀, r₁₀₀, s₁₀₁)  from episode 5
  (s₂₅,  a₂₅,  r₂₅,  s₂₆)   from episode 1
  (s₇₈,  a₇₈,  r₇₈,  s₇₉)   from episode 3
  ...
```

- **Decorrelated:** Random sampling breaks temporal structure
- **Efficient reuse:** Each experience used multiple times
- **Stable distribution:** Mix of old and new policies

**Methods:**

#### `push(state, action, next_state, reward)`

- **Stores new transition** in memory
- **O(1) complexity:** Append to deque
- **Automatic eviction:** When full, oldest removed

#### `sample(batch_size)`

- **Random sampling without replacement** from buffer
- **Returns list of transitions**
- Used to create training batch

#### `__len__()`

- **Returns current buffer size**
- Used to check if enough samples (must be ≥ batch_size)

---

### **4. discrete_pendulum.py - Environment Wrapper**

**Purpose:** Convert Pendulum-v1 from continuous to discrete actions.

**Problem:**

- **Pendulum-v1:** Continuous action space [-2, 2] (infinite possibilities)
- **DQN/DDQN:** Only work with discrete actions (finite set)

**Solution:** Discretize continuous range into fixed bins.

#### Class: `DiscretePendulum(gym.ActionWrapper)`

**Initialization:**

```python
self.action_space = gym.spaces.Discrete(num_actions)  # e.g., 5 actions
self.action_map = np.linspace(-2, 2, num_actions)     # [-2, -1, 0, 1, 2]
```

**Action Mapping (num_actions=5):**

```
Discrete → Continuous
   0    →    -2.0     (strong left torque)
   1    →    -1.0     (medium left)
   2    →     0.0     (no torque)
   3    →    +1.0     (medium right)
   4    →    +2.0     (strong right torque)
```

**Methods:**

#### `action(action)` - Override

- **Input:** Discrete action index (0-4)
- **Output:** Continuous action for base environment
- **Called automatically** by Gym before env.step()

#### `reverse_action(action)` - Helper

- **Input:** Continuous action
- **Output:** Closest discrete index
- **Use case:** Converting continuous actions back (rarely used)

#### `make_pendulum(num_discrete_actions, render_mode)`

- **Factory function** for easy environment creation
- **Wraps base environment** with discretization
- **Consistent interface** with other environments

**Why 5 Actions:**

- **Balance:** Enough granularity but not too many actions
- **Odd number:** Includes "no torque" (0) as middle action
- **Performance:** More actions = larger output layer, slower learning
- **3 actions:** Too coarse, can't balance well
- **11 actions:** Too fine, harder to learn

---

### **5. dqn_agent.py - DQN Agent Implementation**

**Purpose:** Complete DQN agent with exploration, learning, and network updates.

#### Class: `DQNAgent`

**Initialization Components:**

```python
# Networks
self.policy_net = DQN(state_dim, action_dim)  # Current Q-function
self.target_net = DQN(state_dim, action_dim)  # Stable target
self.target_net.load_state_dict(self.policy_net.state_dict())  # Same initial weights

# Optimizer
self.optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

# Memory
self.memory = ReplayMemory(memory_size)

# Device
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Why Two Networks:**

- **Moving target problem:** If same network computes target and prediction, target shifts during training
- **Instability:** Chasing a moving goal leads to oscillations/divergence
- **Solution:** Target network updates slowly, provides stable reference

**Why AdamW with AMSGrad:**

- **AdamW:** Adam with decoupled weight decay (better generalization)
- **AMSGrad:** Fixes convergence issues in Adam
- **Adaptive learning rates:** Different rates for different parameters
- **Momentum:** Uses past gradients to smooth updates

**Key Methods:**

#### `eps_threshold()` - Epsilon Calculation

```python
ε = ε_end + (ε_start - ε_end) * exp(-steps_done / ε_decay)
```

**Exponential Decay Schedule:**

- **Start:** ε = 1.0 (100% random)
- **Gradual decrease** as steps increase
- **Asymptote:** ε → ε_end (1% random)
- **Smooth curve:** No sudden changes

#### `select_action(state)` - Epsilon-Greedy Policy

```python
if random() < ε:
    return random_action      # Exploration
else:
    return argmax Q(s,a)      # Exploitation
```

**Two-phase behavior:**

1. **Early training:** Mostly random (explore state space)
2. **Late training:** Mostly greedy (exploit learned policy)

**Why Epsilon-Greedy:**

- **Simple:** Easy to implement and understand
- **Effective:** Proven to work well in practice
- **Balanced:** Gradual shift from exploration to exploitation
- **Alternative:** Softmax/Boltzmann (more complex, similar results)

#### `store_transition()` - Memory Management

- **Converts done → None:** Terminal states have no next_state
- **Pushes to replay buffer**
- **Called every step** during training

#### `train_step()` - Q-Learning Update

**Step-by-Step Process:**

1. **Check if ready:**

```python
if len(self.memory) < self.batch_size:
    return None  # Not enough samples yet
```

2. **Sample batch:**

```python
transitions = self.memory.sample(batch_size)
batch = Transition(*zip(*transitions))
```

3. **Process terminal states:**

```python
non_final_mask = [next_state is not None]
# Terminals have no next Q-value
```

4. **Compute current Q-values:**

```python
q_values = policy_net(states).gather(1, actions)
# Get Q(s,a) for actions actually taken
```

5. **Compute target Q-values (DQN):**

```python
next_q = target_net(next_states).max(dim=1)[0]
target = reward + gamma * next_q
# TD target: r + γ * max_a' Q_target(s',a')
```

6. **Compute loss:**

```python
loss = smooth_l1_loss(q_values, target)
# Huber loss: |x| < 1 → 0.5*x²
#             |x| ≥ 1 → |x| - 0.5
```

**Why Smooth L1 (Huber) Loss:**

- **Combines MSE and MAE:**
  - Small errors: Acts like MSE (x²) → faster convergence
  - Large errors: Acts like MAE (|x|) → robust to outliers
- **Gradient clipping effect:** Large errors don't dominate gradients
- **More stable** than pure MSE
- **Formula:**

```
L(x) = { 0.5 * x²      if |x| < 1
       { |x| - 0.5     otherwise
```

7. **Gradient descent:**

```python
optimizer.zero_grad()
loss.backward()
clip_grad_value_(parameters, 100.0)  # Prevent exploding gradients
optimizer.step()
```

**Gradient Clipping:**

- **Prevents exploding gradients** (gradients > 100 clipped to 100)
- **Stability:** Large Q-values can cause large gradients
- **Conservative:** Doesn't hurt performance, adds safety

#### `soft_update()` - Target Network Update

```python
θ_target = τ * θ_policy + (1 - τ) * θ_target
```

- **Polyak averaging:** Exponential moving average of policy weights
- **Called every step** after training
- **Smooth transition:** Target slowly tracks policy

---

### **6. ddqn_agent.py - Double DQN Agent**

**Purpose:** Reduce overestimation bias in Q-value estimates.

#### Class: `DDQNAgent(DQNAgent)`

**Inheritance:** Inherits everything from DQNAgent except `train_step()`

**Key Difference in `train_step()`:**

**DQN Approach:**

```python
next_q = target_net(next_states).max(dim=1)[0]
# Same network selects AND evaluates action
```

**DDQN Approach:**

```python
# Step 1: Policy network selects best action
next_actions = policy_net(next_states).argmax(dim=1)

# Step 2: Target network evaluates that action
next_q = target_net(next_states).gather(1, next_actions)
```

**Why This Reduces Bias:**

**Overestimation in DQN:**

- **Max operator is biased:** E[max(X₁, X₂)] ≥ max(E[X₁], E[X₂])
- **Noise amplification:** If one Q-value is randomly high, max selects it
- **Accumulation:** Bias compounds over multiple steps
- **Example:**

```
True Q-values:  [5.0, 5.1, 5.2]
Noisy estimates: [5.3, 4.9, 5.5]  ← noise ±0.3
Max selects 5.5 (overestimate by 0.3)
```

**DDQN Solution:**

- **Decoupling:** Selection uses different network than evaluation
- **Policy network:** May overestimate but selects action
- **Target network:** Provides more conservative evaluation
- **Errors partially cancel:** If policy overestimates action 2, target might underestimate it
- **Empirical result:** More accurate Q-values, better performance

**When DDQN Helps Most:**

- Environments with many actions (more max bias)
- Noisy rewards (more estimation variance)
- Long horizons (bias accumulates)

**When DQN vs DDQN Similar:**

- Simple environments (CartPole)
- Deterministic rewards
- Short episodes

---

### **7. train.py - Training Pipeline**

**Purpose:** Orchestrate agent training, logging, and model saving.

#### Key Functions:

#### `make_env(env_id, render_mode)`

- **Factory pattern** for environment creation
- **Special handling** for Pendulum (discretization)
- **Render mode** for video recording

#### `train_one(env_id, AgentClass, log_file)`

**Complete Training Loop:**

**Phase 1: Initialization**

```python
# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# Ensures reproducibility
```

**Why Seeds Matter:**

- **Reproducibility:** Same results across runs
- **Debugging:** Can isolate issues
- **Fair comparison:** DQN vs DDQN on same random experiences
- **Scientific validity:** Published results must be reproducible

**CUDA Seeds:**

```python
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```

- **GPU operations** can be non-deterministic
- **Forces deterministic algorithms** (slower but reproducible)

**Phase 2: Training Loop**

```python
for ep in range(1, max_episodes + 1):
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # MountainCar reward shaping
        if env_id == "MountainCar-v0":
            reward += abs(next_obs[0] + 0.5)

        agent.store_transition(obs, action, reward, next_obs, done)

        if len(agent.memory) > agent.batch_size:
            loss = agent.train_step()

        agent.update_target()
```

**MountainCar Reward Shaping:**

- **Original reward:** -1 every step until goal
- **Problem:** No signal about progress
- **Solution:** Add |position + 0.5|
  - Position range: [-1.2, 0.6]
  - Center: -0.5
  - Reward moving right (toward goal)
- **Critical for learning:** Without this, nearly impossible to learn

**Phase 3: Logging**

```python
wandb.log({
    "Performance/Episode_Return": ep_return,
    "Learning/Loss": avg_loss,
    "Learning/Epsilon": current_epsilon,
    "Q_Values/Std": q_std,  # Detect Q-collapse
    ...
})
```

**Q-Value Monitoring:**

```python
q_std = Q_values.std()
if q_std < 0.1:
    print("WARNING: Q-collapse!")
```

- **Q-collapse:** All Q-values → same value
- **Symptom:** std ≈ 0
- **Causes:** Learning rate too high, poor exploration
- **Solution:** Lower LR, increase epsilon_end

**Phase 4: Success Check**

```python
if len(last100) == 100 and mean_reward_100 > threshold:
    success = True
    torch.save(agent.policy_net.state_dict(), model_path)
    break
```

**Rolling Window (deque):**

```python
last100 = deque(maxlen=100)
last100.append(ep_return)
mean_reward_100 = np.mean(last100)
```

- **Smooths noisy returns**
- **Standard metric** in RL benchmarks
- **Success criterion:** Average over 100 episodes

**Weights & Biases Integration:**

- **Experiment tracking:** All runs logged
- **Visualization:** Real-time plots
- **Comparison:** Easy to compare DQN vs DDQN
- **Reproducibility:** Hyperparams saved with results

---

### **8. test.py - Model Evaluation**

**Purpose:** Load trained models and evaluate performance.

#### Key Functions:

#### `load_agent(env_id, AgentClass, model_path)`

- **Recreates agent** with same hyperparams
- **Loads saved weights:** `policy_net.load_state_dict()`
- **Sets eval mode:** `policy_net.eval()`
  - Disables dropout (not used here)
  - Signals no gradient computation

#### `test_agent(env_id, AgentClass, model_path, num_episodes, num_recordings)`

**Evaluation Process:**

1. **Video Recording:**

```python
env = RecordVideo(env, video_folder, episode_trigger=lambda i: i < 5)
```

- **Records first 5 episodes** as MP4 videos
- **Visual verification** of learned behavior

2. **Greedy Policy:**

```python
with torch.no_grad():
    action = policy_net(obs).argmax()
# No exploration, pure exploitation
```

- **Epsilon = 0:** Always choose best action
- **Deterministic:** Same state → same action
- **True performance test**

3. **Statistics:**

```python
all_returns = []
for ep in range(num_episodes):
    # Run episode
    all_returns.append(ep_return)

avg_return = np.mean(all_returns)
```

- **100 episodes:** Reliable performance estimate
- **Mean and std:** Characterize consistency

**Why Test Separately:**

- **Training mode:** Exploration adds noise
- **Eval mode:** Pure learned policy
- **Scientific standard:** Separate train/test

---

## Network Architecture Details

### **Weight Initialization**

PyTorch uses **Kaiming He initialization** for Linear layers:

```
W ~ Uniform(-√(k), √(k)) where k = 1 / fan_in
```

**Why Kaiming:**

- **Designed for ReLU:** Maintains variance through layers
- **Prevents vanishing/exploding activations**
- **Standard since 2015**

### **Forward Pass Example (CartPole)**

**Input state:** [x, x_dot, θ, θ_dot] (4 values)

**Layer 1:**

```
z₁ = W₁ * input + b₁     [256 values]
a₁ = ReLU(z₁)            [~50% sparse due to ReLU]
```

**Layer 2:**

```
z₂ = W₂ * a₁ + b₂        [256 values]
a₂ = ReLU(z₂)            [~50% sparse]
```

**Output:**

```
Q_values = W₃ * a₂ + b₃  [2 values: Q(s,left), Q(s,right)]
```

**Total Parameters:**

```
Layer 1: 4 * 256 + 256 = 1,280
Layer 2: 256 * 256 + 256 = 65,792
Layer 3: 256 * 2 + 2 = 514
Total: 67,586 parameters
```

### **Backpropagation Flow**

**Loss:** L = Huber(Q(s,a), r + γ\*max Q_target(s',a'))

**Gradient computation:**

```
dL/dW₃ = dL/dQ * dQ/dW₃
dL/dW₂ = dL/dQ * dQ/da₂ * da₂/dW₂
dL/dW₁ = dL/dQ * dQ/da₂ * da₂/da₁ * da₁/dW₁
```

**Automatic differentiation:** PyTorch computes all gradients automatically

**Gradient clipping:**

```python
torch.nn.utils.clip_grad_value_(parameters, 100.0)
```

- Prevents any gradient from exceeding ±100
- Essential for stability with large Q-values

---

## Summary of Design Choices

### **What Makes This Implementation Effective:**

1. **Soft updates (τ=0.005)** instead of hard updates → stability
2. **Huber loss** instead of MSE → robust to outliers
3. **Experience replay (10K)** → breaks correlations
4. **Epsilon decay** → smooth exploration-exploitation transition
5. **AdamW + AMSGrad** → adaptive, stable optimization
6. **Gradient clipping** → prevents exploding gradients
7. **Environment-specific tuning** → respects problem difficulty
8. **Reward shaping (MountainCar)** → provides learning signal
9. **Two network architecture (256-256)** → sufficient capacity
10. **DDQN variant** → reduces overestimation bias

### **Trade-offs Made:**

- **Batch size 64:** Could be larger (more stable) but slower
- **Memory 10K:** Could be larger (more diversity) but more RAM
- **Two layers:** Could be deeper (more capacity) but risk overfitting
- **Discrete Pendulum:** Loses continuous control precision
- **Fixed hyperparams:** Could use hyperparameter optimization

---

**End of Documentation**
