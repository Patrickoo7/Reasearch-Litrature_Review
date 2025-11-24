# Lesson 12: Production RL & Sim-to-Real Transfer 🚀

**Module 12: Reinforcement Learning | Lesson 12 of 12**

Master deploying RL in production - from Stable-Baselines3 to real robots!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Master Stable-Baselines3 for production-quality RL
2. ✅ Deploy RL policies with FastAPI and Docker
3. ✅ Understand sim-to-real transfer and domain randomization
4. ✅ Learn distributed training (IMPALA, Ape-X) for scaling
5. ✅ Implement monitoring, logging, and reproducibility
6. ✅ Build complete end-to-end RL systems

---

## 1. Why Production RL is Different

### Research vs Production

Research RL: Algorithm development, benchmarks
Production RL: Real-world deployment, reliability!

```python
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import gym
import matplotlib.pyplot as plt

"""
Production RL Challenges:

Research:
- Focus: Algorithm performance
- Environment: Simulations
- Deployment: Not required

Production:
- Focus: Reliability, latency, safety
- Environment: Real world
- Deployment: Critical!

Production Requirements:
✅ Reproducibility
✅ Monitoring & logging
✅ Version control
✅ A/B testing
✅ Deployment infrastructure
✅ Sim-to-real transfer
✅ Safety guarantees
✅ Scalability
"""

print("=== Production RL ===")
print("\nProduction Requirements:")
print("✅ Reproducible training")
print("✅ Robust policies")
print("✅ Low-latency inference")
print("✅ Monitoring and logging")
print("✅ Sim-to-real transfer")
print("✅ Continuous improvement")
```

---

## 2. Stable-Baselines3: Production-Ready RL

The industry standard for applied RL!

```python
"""
Stable-Baselines3 (SB3):

Production-quality implementations of:
- DQN, Double DQN, Dueling DQN
- PPO (most popular!)
- A2C
- SAC (continuous control)
- TD3 (continuous control)
- DDPG

Features:
✅ Well-tested, documented
✅ Reproducible results
✅ Easy-to-use API
✅ Tensorboard integration
✅ Callbacks for customization
✅ Model saving/loading
"""

class StableBaselines3Guide:
    """
    Comprehensive Stable-Baselines3 guide.
    """

    @staticmethod
    def basic_training():
        """Basic SB3 training workflow."""
        # Create environment
        env = gym.make("CartPole-v1")

        # Create model
        model = PPO(
            "MlpPolicy",  # Policy type
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./ppo_tensorboard/"
        )

        # Train
        model.learn(total_timesteps=100000)

        # Save
        model.save("ppo_cartpole")

        # Load
        model = PPO.load("ppo_cartpole")

        # Evaluate
        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

        env.close()

        print("✅ Basic SB3 training complete")

    @staticmethod
    def advanced_training():
        """Advanced SB3 with callbacks and monitoring."""
        # Create vectorized environments (parallel training!)
        env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
        eval_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs/best_model",
            log_path="./logs/results",
            eval_freq=5000,
            deterministic=True,
            render=False
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./logs/checkpoints",
            name_prefix="ppo_model"
        )

        # Train with callbacks
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard/")
        model.learn(
            total_timesteps=200000,
            callback=[eval_callback, checkpoint_callback]
        )

        print("✅ Advanced SB3 training with monitoring")

    @staticmethod
    def hyperparameter_tuning():
        """Hyperparameter optimization with Optuna."""
        import optuna
        from stable_baselines3.common.evaluation import evaluate_policy

        def optimize_ppo(trial):
            """Optuna objective function."""
            # Sample hyperparameters
            learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
            n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
            gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.995])
            clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])

            # Create environment
            env = gym.make("CartPole-v1")

            # Create model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                clip_range=clip_range,
                verbose=0
            )

            # Train
            model.learn(total_timesteps=50000)

            # Evaluate
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

            return mean_reward

        # Optimize
        # study = optuna.create_study(direction="maximize")
        # study.optimize(optimize_ppo, n_trials=50)

        print("✅ Hyperparameter tuning with Optuna")


# Demonstrate SB3
sb3_guide = StableBaselines3Guide()
print("\n=== Stable-Baselines3 Guide ===")
print("✅ Production-ready RL library")
print("✅ PPO, SAC, TD3, DQN implementations")
print("✅ Callbacks, monitoring, hyperparameter tuning")
```

---

## 3. Deploying RL Policies

Serve policies in production with FastAPI!

```python
"""
Policy Deployment:

Requirements:
- Low latency (<100ms)
- High throughput
- REST API or gRPC
- Load balancing
- Monitoring

Tools:
- FastAPI: Modern Python web framework
- Docker: Containerization
- Kubernetes: Orchestration
- MLflow: Model versioning
"""

# FastAPI deployment example
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class DeployedRLPolicy:
    """
    Production RL policy deployment.
    """
    def __init__(self, model_path):
        # Load trained model
        self.model = PPO.load(model_path)
        self.model.policy.eval()  # Set to evaluation mode

        print(f"Model loaded from {model_path}")

    def predict(self, observation):
        """
        Predict action for observation.

        Returns:
            action: Predicted action
            latency: Inference time (ms)
        """
        import time

        start = time.time()

        # Predict
        action, _ = self.model.predict(observation, deterministic=True)

        latency = (time.time() - start) * 1000  # ms

        return action, latency


# FastAPI application
app = FastAPI(title="RL Policy API")

# Load policy
policy = None  # Will be loaded on startup


class ObservationInput(BaseModel):
    """Request schema."""
    observation: list


class ActionOutput(BaseModel):
    """Response schema."""
    action: int
    latency_ms: float


@app.on_event("startup")
def load_model():
    """Load model on startup."""
    global policy
    policy = DeployedRLPolicy("ppo_cartpole.zip")


@app.post("/predict", response_model=ActionOutput)
def predict_action(obs_input: ObservationInput):
    """
    Predict action endpoint.

    Example:
    POST /predict
    {
        "observation": [0.1, 0.2, 0.3, 0.4]
    }
    """
    observation = np.array(obs_input.observation)
    action, latency = policy.predict(observation)

    return ActionOutput(
        action=int(action),
        latency_ms=latency
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000

print("\n=== RL Policy Deployment ===")
print("✅ FastAPI for REST API")
print("✅ Low-latency inference")
print("✅ Docker containerization")
print("✅ Kubernetes orchestration")
```

---

## 4. Sim-to-Real Transfer

Bridge the reality gap!

```python
"""
Sim-to-Real Transfer:

Problem: Policies trained in simulation fail in reality.

Reality Gap:
- Physics differences
- Sensor noise
- Actuator delays
- Visual appearance

Solutions:
1. Domain Randomization
2. System Identification
3. Domain Adaptation
4. Learning in real world
"""

class DomainRandomization:
    """
    Domain Randomization for sim-to-real transfer.

    Randomize simulation parameters:
    - Physics (mass, friction, damping)
    - Sensor noise
    - Visual appearance (textures, lighting)
    - Delays

    Forces policy to be robust!
    """
    def __init__(self, base_env):
        self.base_env = base_env

        # Randomization ranges
        self.mass_range = (0.8, 1.2)  # ±20%
        self.friction_range = (0.5, 1.5)
        self.noise_std = 0.01

        print("Domain Randomization initialized")

    def randomize_physics(self):
        """Randomize physics parameters."""
        # Randomize mass
        mass_scale = np.random.uniform(*self.mass_range)
        # self.base_env.model.body_mass *= mass_scale

        # Randomize friction
        friction_scale = np.random.uniform(*self.friction_range)
        # self.base_env.model.geom_friction *= friction_scale

        print(f"Randomized: mass={mass_scale:.2f}x, friction={friction_scale:.2f}x")

    def add_sensor_noise(self, observation):
        """Add sensor noise to observations."""
        noise = np.random.normal(0, self.noise_std, size=observation.shape)
        noisy_obs = observation + noise
        return noisy_obs

    def reset(self):
        """Reset with randomized parameters."""
        self.randomize_physics()
        obs = self.base_env.reset()
        return self.add_sensor_noise(obs)

    def step(self, action):
        """Step with sensor noise."""
        obs, reward, done, info = self.base_env.step(action)
        return self.add_sensor_noise(obs), reward, done, info


class SystemIdentification:
    """
    System Identification for sim-to-real.

    1. Collect data from real robot
    2. Fit simulation parameters to match reality
    3. Train in calibrated simulation
    """
    def __init__(self):
        self.real_trajectories = []
        self.sim_params = {}

        print("System Identification initialized")

    def collect_real_data(self, robot, num_episodes=10):
        """Collect trajectories from real robot."""
        for episode in range(num_episodes):
            trajectory = []

            # Random actions on real robot
            for step in range(100):
                action = np.random.uniform(-1, 1, size=robot.action_dim)
                obs, _, _, _ = robot.step(action)
                trajectory.append((action, obs))

            self.real_trajectories.append(trajectory)

        print(f"Collected {num_episodes} real trajectories")

    def fit_simulation(self, simulator):
        """
        Fit simulation parameters to match real data.

        Optimize: min ||sim_trajectory - real_trajectory||²
        """
        # In practice: Use optimization (gradient descent, CMA-ES)
        # to find sim parameters that best match real data

        print("Fitting simulation to real data...")
        # self.sim_params = optimize_sim_params(...)

        return self.sim_params


print("\n=== Sim-to-Real Transfer ===")
print("✅ Domain Randomization")
print("✅ System Identification")
print("✅ Domain Adaptation")
print("✅ Critical for robotics!")
```

---

## 5. Distributed Training

Scale RL to multiple machines!

```python
"""
Distributed RL:

Why distributed?
- Faster training (parallel experience collection)
- Larger batch sizes
- More exploration

Architectures:
1. IMPALA: Experience collection + learner separation
2. Ape-X: Prioritized distributed experience replay
3. Ray RLlib: General distributed RL framework
"""

class IMPALAArchitecture:
    """
    IMPALA: Importance Weighted Actor-Learner Architecture.

    Components:
    - Actors: Collect experience in parallel
    - Learner: Updates policy
    - Experience queue: Actors → Learner

    Key: V-trace for off-policy correction!
    """
    def __init__(self, num_actors=16):
        self.num_actors = num_actors

        # Shared policy (actors copy periodically)
        self.policy = None

        # Experience queue
        self.experience_queue = []

        print(f"IMPALA with {num_actors} actors")

    def actor_rollout(self, actor_id):
        """
        Actor: Collect experience.

        Runs in separate process/machine!
        """
        # Copy latest policy
        local_policy = self.policy.copy()

        # Collect trajectory
        env = gym.make("CartPole-v1")
        obs = env.reset()

        trajectory = []
        for step in range(100):
            action = local_policy.predict(obs)
            next_obs, reward, done, _ = env.step(action)

            trajectory.append((obs, action, reward, next_obs, done))

            if done:
                break
            obs = next_obs

        # Send to learner
        self.experience_queue.append(trajectory)

    def learner_update(self):
        """
        Learner: Update policy from experience queue.
        """
        if not self.experience_queue:
            return

        # Get batch of trajectories
        batch = self.experience_queue[:32]
        self.experience_queue = self.experience_queue[32:]

        # V-trace correction (off-policy)
        # Update policy
        # self.policy.update(batch)

        print("Learner update complete")


# Ray RLlib example
"""
Ray RLlib: Scalable RL library

Features:
- Distributed training (multi-node)
- All major algorithms (PPO, SAC, DQN, etc.)
- Custom models and environments
- Hyperparameter tuning
- Production deployment

Example:
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

ray.init()

config = {
    "env": "CartPole-v1",
    "num_workers": 4,  # Parallel actors
    "num_gpus": 1,
    "framework": "torch"
}

tune.run(
    PPOTrainer,
    config=config,
    stop={"episode_reward_mean": 200},
    checkpoint_freq=10
)
"""

print("\n=== Distributed Training ===")
print("✅ IMPALA: Actor-learner architecture")
print("✅ Ape-X: Prioritized distributed replay")
print("✅ Ray RLlib: Production-scale RL")
print("✅ 10-100x speedup!")
```

---

## 6. Monitoring and Reproducibility

Critical for production!

```python
"""
Monitoring & Logging:

Track:
- Episode rewards (mean, std, max, min)
- Episode lengths
- Policy loss, value loss
- Learning rate
- Exploration rate (epsilon, entropy)
- Custom metrics

Tools:
- Tensorboard
- Weights & Biases (wandb)
- MLflow
- Prometheus + Grafana
"""

class ProductionMonitoring:
    """
    Production monitoring for RL.
    """
    def __init__(self):
        # Initialize logging
        import wandb

        wandb.init(
            project="rl-production",
            config={
                "algorithm": "PPO",
                "env": "CartPole-v1",
                "learning_rate": 3e-4
            }
        )

        print("Monitoring initialized")

    def log_training(self, step, metrics):
        """Log training metrics."""
        import wandb

        wandb.log({
            "episode_reward": metrics["reward"],
            "episode_length": metrics["length"],
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"]
        }, step=step)

    def log_evaluation(self, step, eval_metrics):
        """Log evaluation metrics."""
        import wandb

        wandb.log({
            "eval/mean_reward": eval_metrics["mean_reward"],
            "eval/std_reward": eval_metrics["std_reward"],
            "eval/success_rate": eval_metrics["success_rate"]
        }, step=step)


# Reproducibility
class ReproducibleRL:
    """
    Ensure reproducible RL experiments.
    """
    @staticmethod
    def set_seed(seed=42):
        """Set all random seeds."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # gym.seed(seed)  # Deprecated in newer versions

        # Deterministic behavior (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"Seed set to {seed} for reproducibility")

    @staticmethod
    def save_config(config, path="config.yaml"):
        """Save experiment configuration."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(config, f)

        print(f"Config saved to {path}")


print("\n=== Monitoring & Reproducibility ===")
print("✅ Tensorboard / Weights & Biases")
print("✅ MLflow for experiment tracking")
print("✅ Seed setting for reproducibility")
print("✅ Config versioning")
```

---

## 7. Complete End-to-End System

Putting it all together!

```python
"""
End-to-End Production RL System:

1. Training:
   - Stable-Baselines3 or Ray RLlib
   - Distributed training
   - Hyperparameter tuning
   - Monitoring with wandb

2. Evaluation:
   - Sim evaluation
   - Sim-to-real with domain randomization
   - Real-world testing

3. Deployment:
   - FastAPI REST API
   - Docker containerization
   - Kubernetes deployment
   - Load balancing

4. Monitoring:
   - Policy performance metrics
   - Latency monitoring
   - Error rates
   - A/B testing

5. Continuous Improvement:
   - Collect real-world data
   - Offline RL fine-tuning
   - Model updates
"""

class EndToEndRLSystem:
    """
    Complete production RL system.
    """
    def __init__(self):
        self.model = None
        self.monitoring = ProductionMonitoring()

        print("End-to-end RL system initialized")

    def train(self, env_name, total_timesteps=1000000):
        """Phase 1: Training."""
        print("\n=== Phase 1: Training ===")

        # Set seed
        ReproducibleRL.set_seed(42)

        # Create environment with domain randomization
        env = gym.make(env_name)
        env = DomainRandomization(env)

        # Train with SB3
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/"
        )

        self.model.learn(total_timesteps=total_timesteps)

        # Save model
        self.model.save("production_model")

        print("✅ Training complete")

    def evaluate(self, num_episodes=100):
        """Phase 2: Evaluation."""
        print("\n=== Phase 2: Evaluation ===")

        # Evaluate in simulation
        env = gym.make("CartPole-v1")

        rewards = []
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print("✅ Evaluation complete")

    def deploy(self):
        """Phase 3: Deployment."""
        print("\n=== Phase 3: Deployment ===")

        # Save model for deployment
        # Deploy with FastAPI (see earlier example)
        # Containerize with Docker
        # Deploy to Kubernetes

        print("✅ Deployed to production")

    def monitor(self):
        """Phase 4: Continuous monitoring."""
        print("\n=== Phase 4: Monitoring ===")

        # Monitor performance
        # Collect real-world data
        # Offline RL fine-tuning
        # A/B testing

        print("✅ Monitoring active")


# Example: Run end-to-end system
system = EndToEndRLSystem()
# system.train("CartPole-v1", total_timesteps=100000)
# system.evaluate()
# system.deploy()
# system.monitor()

print("\n=== Complete End-to-End RL System ===")
print("✅ Training with domain randomization")
print("✅ Evaluation in sim and real")
print("✅ Deployment with FastAPI/Docker/K8s")
print("✅ Continuous monitoring and improvement")
```

---

## 8. Real-World Success Stories

```python
"""
Production RL Deployments:

1. Google Data Centers:
   - Deep RL for cooling optimization
   - 40% energy reduction
   - Millions saved annually

2. Amazon Warehouses:
   - Multi-robot coordination
   - Inventory management
   - Route optimization

3. Waymo / Tesla:
   - Autonomous vehicle planning
   - Sim-to-real transfer
   - Safety-critical deployment

4. Microsoft Azure:
   - Resource allocation
   - Load balancing
   - Cost optimization

5. DeepMind AlphaFold:
   - Protein folding
   - Scientific discovery
   - Real-world impact

6. OpenAI Codex:
   - Code generation
   - RLHF for alignment
   - Millions of users
"""

print("\n=== Real-World Success Stories ===")
print("\n1. Google Data Centers:")
print("   - 40% energy reduction with Deep RL")

print("\n2. Amazon Warehouses:")
print("   - Multi-robot coordination")
print("   - Route optimization")

print("\n3. Autonomous Vehicles:")
print("   - Waymo, Tesla")
print("   - Sim-to-real transfer")

print("\n4. OpenAI Codex:")
print("   - RLHF for code generation")
print("   - Production deployment")
```

---

## Practice Exercises

### Exercise 1: End-to-End Deployment

```python
"""
Build complete end-to-end RL system.

Requirements:
1. Train PPO on a task (e.g., LunarLander)
2. Implement FastAPI deployment
3. Dockerize the application
4. Add monitoring with Tensorboard
5. Deploy locally

Deliverables:
- Trained model
- FastAPI server
- Dockerfile
- README with instructions
"""

# Your implementation here
```

### Exercise 2: Sim-to-Real Transfer

```python
"""
Implement sim-to-real transfer for a simple robot task.

Requirements:
1. Create simulation environment
2. Implement domain randomization
3. Train policy in randomized sim
4. (If possible) Deploy to real robot or realistic simulator
5. Compare: Standard vs domain randomization

Metrics:
- Sim performance
- Real-world performance
- Transfer gap
"""

# Your implementation here
```

### Exercise 3: Distributed Training

```python
"""
Implement distributed training with Ray RLlib.

Requirements:
1. Install Ray and RLlib
2. Configure distributed PPO (4+ workers)
3. Train on Atari or MuJoCo
4. Compare: Single-node vs distributed
5. Measure speedup

Track:
- Training time
- Sample efficiency
- Final performance
- Resource utilization
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Stable-Baselines3** 📦
   - Production-ready RL
   - PPO, SAC, TD3, DQN
   - Industry standard
   - Easy to use

2. **Deployment** 🚀
   - FastAPI for serving
   - Docker containerization
   - Kubernetes orchestration
   - Low-latency inference

3. **Sim-to-Real** 🌉
   - Domain randomization
   - System identification
   - Reality gap
   - Critical for robotics

4. **Distributed Training** ⚡
   - IMPALA, Ape-X
   - Ray RLlib
   - 10-100x speedup
   - Scalability

5. **Monitoring** 📊
   - Tensorboard, wandb
   - MLflow tracking
   - Reproducibility
   - Continuous improvement

6. **End-to-End Systems** 🏗️
   - Train → Evaluate → Deploy → Monitor
   - Complete pipeline
   - Production-grade
   - Real-world impact

### Production Checklist

✅ **Training**: Reproducible, monitored, versioned
✅ **Evaluation**: Sim + real, safety checks
✅ **Deployment**: API, containerized, scalable
✅ **Monitoring**: Metrics, logging, alerts
✅ **Improvement**: Continuous learning, A/B testing

### Real-World Impact

✅ **Energy**: 40% reduction (Google data centers)
✅ **Logistics**: Amazon warehouse optimization
✅ **Autonomous**: Waymo, Tesla deployments
✅ **AI Products**: OpenAI Codex, ChatGPT (RLHF)

### Module 12 Complete! 🎓

**Congratulations!** You've completed Module 12: Reinforcement Learning!

You've learned:
1. ✅ Bandits & Exploration
2. ✅ Tabular Methods (Q-Learning, SARSA)
3. ✅ Deep Q-Networks (DQN, Rainbow)
4. ✅ Policy Gradients (PPO, SAC)
5. ✅ Model-Based RL (PETS, Dreamer)
6. ✅ Imitation Learning (BC, GAIL)
7. ✅ Offline RL (CQL, Decision Transformer)
8. ✅ Advanced Exploration (ICM, RND)
9. ✅ RLHF (ChatGPT training!)
10. ✅ Multi-Agent RL (AlphaGo, StarCraft)
11. ✅ Hierarchical & Meta-RL
12. ✅ Production Deployment

**You're now ready to build real-world RL systems!** 🚀

---

## Additional Resources

### Libraries
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/ ⭐
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/
- **CleanRL**: https://github.com/vwxyzjn/cleanrl
- **Tianshou**: https://github.com/thu-ml/tianshou

### Deployment
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://www.docker.com/
- **Kubernetes**: https://kubernetes.io/

### Monitoring
- **Weights & Biases**: https://wandb.ai/
- **MLflow**: https://mlflow.org/
- **Tensorboard**: https://www.tensorflow.org/tensorboard

### Sim-to-Real
- **OpenAI Safety Gym**: https://github.com/openai/safety-gym
- **PyBullet**: https://pybullet.org/ (robotics simulation)
- **MuJoCo**: https://mujoco.org/

### Courses
- **Spinning Up (OpenAI)**: https://spinningup.openai.com/ ⭐
- **Deep RL Course (Berkeley)**: http://rail.eecs.berkeley.edu/deeprlcourse/
- **Deep RL (Hugging Face)**: https://huggingface.co/deep-rl-course

---

## What's Next?

**Continue your RL journey:**

1. **Build Projects**: Apply RL to real problems
2. **Contribute**: Open-source RL libraries
3. **Research**: Read latest papers (NeurIPS, ICML, ICLR)
4. **Compete**: Kaggle, AIcrowd RL competitions
5. **Deploy**: Put RL in production!

**Key Resources:**
- Papers with Code: https://paperswithcode.com/area/reinforcement-learning
- RL Discord/Slack communities
- Arxiv Sanity: http://arxiv-sanity-lite.com/

**The field is rapidly evolving - keep learning!** 📚

---

**Congratulations on completing Module 12: Reinforcement Learning!** 🎉

You now have comprehensive knowledge of RL from fundamentals to production deployment. Go build amazing RL systems! 🚀
