# AgentRL API 参考文档

## 📚 核心API概览

AgentRL框架提供了简洁而强大的API接口，支持多种强化学习算法的训练和推理。本文档详细介绍了所有主要类、方法和配置选项。

## 🎯 快速开始

```python
from agentrl import Agent, Environment, PPOTrainer
from agentrl.utils.config import create_default_config

# 创建配置
config = create_default_config()

# 创建环境和智能体
env = Environment("CartPole-v1")
agent = Agent("PPO", env.observation_space, env.action_space)

# 开始训练
trainer = PPOTrainer(agent, env, config)
trainer.train(episodes=1000)
```

## 🏗️ 核心类 API

### 1. Agent 类

智能体工厂类，用于创建和管理不同算法的智能体。

#### 初始化

```python
Agent(algorithm, observation_space, action_space, config=None)
```

**参数：**
- `algorithm` (str): 算法名称，支持 "PPO", "DQN", "SAC", "A3C"
- `observation_space`: 观测空间，Gymnasium格式
- `action_space`: 动作空间，Gymnasium格式  
- `config` (dict, optional): 智能体配置参数

**示例：**
```python
# 创建PPO智能体
agent = Agent("PPO", env.observation_space, env.action_space, {
    "lr": 3e-4,
    "gamma": 0.99,
    "hidden_dims": [64, 64]
})

# 创建DQN智能体
agent = Agent("DQN", env.observation_space, env.action_space, {
    "lr": 1e-4,
    "epsilon": 0.1,
    "target_update_freq": 1000
})
```

#### 方法

##### `act(observation, training=True)`

选择动作。

**参数：**
- `observation` (np.ndarray): 当前观测
- `training` (bool): 是否为训练模式

**返回：**
- `action`: 选择的动作（int或np.ndarray）

**示例：**
```python
obs = env.reset()
action = agent.act(obs, training=True)  # 训练时
action = agent.act(obs, training=False) # 评估时
```

##### `learn(batch)`

学习更新智能体。

**参数：**
- `batch` (dict): 训练批次数据

**返回：**
- `dict`: 训练损失信息

##### `save(filepath)` / `load(filepath)`

保存/加载模型。

**参数：**
- `filepath` (str): 文件路径

**示例：**
```python
agent.save("models/ppo_cartpole.pt")
agent.load("models/ppo_cartpole.pt")
```

### 2. Environment 类

环境封装类，提供Gymnasium环境的增强功能。

#### 初始化

```python
Environment(env_id, config=None, **kwargs)
```

**参数：**
- `env_id` (str): 环境ID，如 "CartPole-v1"
- `config` (dict, optional): 环境配置
- `**kwargs`: 传递给gymnasium.make的额外参数

**配置选项：**
```python
env_config = {
    "frame_stack": 4,           # 帧堆叠数量
    "frame_skip": 1,            # 帧跳跃数量
    "normalize_obs": True,      # 观测归一化
    "normalize_reward": True,   # 奖励归一化
    "clip_reward": 1.0,         # 奖励裁剪
    "max_episode_steps": 1000   # 最大步数
}
```

#### 方法

##### `reset()`

重置环境。

**返回：**
- `observation`: 初始观测

##### `step(action)`

执行动作。

**参数：**
- `action`: 要执行的动作

**返回：**
- `observation`: 下一个观测
- `reward`: 奖励
- `done`: 是否结束
- `info`: 额外信息

**示例：**
```python
env = Environment("CartPole-v1", {
    "normalize_obs": True,
    "max_episode_steps": 500
})

obs = env.reset()
action = agent.act(obs)
next_obs, reward, done, info = env.step(action)
```

### 3. 训练器类

#### BaseTrainer

训练器基类，定义了通用的训练流程。

```python
BaseTrainer(agent, env, config=None)
```

**通用配置：**
```python
training_config = {
    "max_episodes": 1000,
    "max_steps_per_episode": 1000,
    "eval_freq": 100,
    "save_freq": 100,
    "log_freq": 10,
    "early_stopping_patience": 100,
    "target_reward": None
}
```

##### `train(max_episodes=None, early_stopping_patience=100, target_reward=None)`

开始训练过程。

**参数：**
- `max_episodes` (int, optional): 最大训练回合数
- `early_stopping_patience` (int): 早停耐心值
- `target_reward` (float, optional): 目标奖励阈值

**返回：**
- `dict`: 训练历史数据

##### `evaluate(episodes=10, render=False)`

评估智能体性能。

**参数：**
- `episodes` (int): 评估回合数
- `render` (bool): 是否渲染

**返回：**
- `dict`: 评估结果

#### PPOTrainer

PPO算法专用训练器。

```python
PPOTrainer(agent, env, config=None)
```

**PPO特定配置：**
```python
ppo_config = {
    "update_interval": 2048,    # 更新间隔
    "ppo_epochs": 4,           # PPO训练轮数
    "clip_ratio": 0.2,         # 裁剪比例
    "value_loss_coef": 0.5,    # 价值损失系数
    "entropy_coef": 0.01       # 熵系数
}
```

#### DQNTrainer

DQN算法专用训练器。

```python
DQNTrainer(agent, env, config=None)
```

**DQN特定配置：**
```python
dqn_config = {
    "memory_size": 100000,      # 经验回放缓冲区大小
    "batch_size": 32,           # 批量大小
    "target_update_freq": 1000, # 目标网络更新频率
    "epsilon_decay": 0.995,     # epsilon衰减
    "min_epsilon": 0.01         # 最小epsilon
}
```

### 4. InferenceEngine 类

高效推理引擎，用于模型部署和批量推理。

#### 初始化

```python
InferenceEngine(agent, device=None, batch_size=1, optimize=True)
```

**参数：**
- `agent`: 训练好的智能体
- `device` (str, optional): 推理设备 ("cpu" 或 "cuda")
- `batch_size` (int): 批量推理大小
- `optimize` (bool): 是否启用模型优化

#### 方法

##### `predict(observations, deterministic=True)`

批量预测动作。

**参数：**
- `observations` (np.ndarray): 观测数组
- `deterministic` (bool): 是否确定性预测

**返回：**
- `actions`: 预测的动作

##### `benchmark(num_samples=1000)`

性能基准测试。

**参数：**
- `num_samples` (int): 测试样本数

**返回：**
- `dict`: 性能统计

**示例：**
```python
# 创建推理引擎
engine = InferenceEngine(agent, device="cuda", batch_size=32)

# 批量推理
observations = np.random.randn(32, 4)  # 32个观测
actions = engine.predict(observations)

# 性能测试
stats = engine.benchmark(num_samples=10000)
print(f"平均推理时间: {stats['avg_time']:.3f}ms")
```

## 🔧 工具类 API

### 1. Config 类

配置管理工具。

#### 初始化

```python
Config(config=None)
```

**参数：**
- `config` (dict/str, optional): 配置字典或文件路径

#### 方法

##### `get(key, default=None)`

获取配置值，支持点分割路径。

**示例：**
```python
config = Config()
lr = config.get("agent.lr", 3e-4)
hidden_dims = config.get("agent.hidden_dims", [64, 64])
```

##### `set(key, value)`

设置配置值。

**示例：**
```python
config.set("agent.lr", 1e-4)
config.set("training.max_episodes", 2000)
```

##### `load_from_file(filepath)` / `save_to_file(filepath)`

从文件加载/保存配置。

**支持格式：** JSON, YAML

### 2. Logger 类

日志记录工具。

#### 初始化

```python
Logger(log_dir, use_tensorboard=True, use_wandb=False, wandb_config=None)
```

#### 方法

##### `log_scalar(tag, value, step)`

记录标量值。

##### `log_histogram(tag, values, step)`

记录直方图。

##### `log_image(tag, image, step)`

记录图像。

**示例：**
```python
logger = Logger("./logs/experiment_1")
logger.log_scalar("train/reward", reward, episode)
logger.log_histogram("train/actions", actions, episode)
```

## 📊 算法特定配置

### PPO 配置参数

```python
ppo_config = {
    # 网络架构
    "hidden_dims": [64, 64],
    
    # 学习参数
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    
    # PPO参数
    "clip_ratio": 0.2,
    "ppo_epochs": 4,
    "mini_batch_size": 64,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    
    # 训练参数
    "update_interval": 2048,
    "normalize_advantages": True
}
```

### DQN 配置参数

```python
dqn_config = {
    # 网络架构
    "hidden_dims": [128, 128],
    "dueling": True,
    
    # 学习参数
    "lr": 1e-4,
    "gamma": 0.99,
    
    # DQN参数
    "memory_size": 100000,
    "batch_size": 32,
    "target_update_freq": 1000,
    "double_dqn": True,
    
    # 探索参数
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    
    # 训练参数
    "learning_starts": 1000,
    "train_freq": 4
}
```

### SAC 配置参数

```python
sac_config = {
    # 网络架构
    "hidden_dims": [256, 256],
    
    # 学习参数
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    
    # SAC参数
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "target_entropy": None,
    
    # 训练参数
    "memory_size": 1000000,
    "batch_size": 256,
    "learning_starts": 10000,
    "train_freq": 1,
    "gradient_steps": 1
}
```

### A3C 配置参数

```python
a3c_config = {
    # 网络架构
    "hidden_dims": [128, 128],
    
    # 学习参数
    "lr": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    
    # A3C参数
    "num_workers": 8,
    "n_steps": 5,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 40.0,
    
    # 训练参数
    "update_interval": 5
}
```

## 🌍 环境配置选项

### 基础环境配置

```python
env_config = {
    # 预处理
    "frame_stack": 1,           # 帧堆叠
    "frame_skip": 1,            # 帧跳跃
    "resize": None,             # 图像缩放尺寸 (height, width)
    "grayscale": False,         # 转换为灰度图
    
    # 归一化
    "normalize_obs": False,     # 观测归一化
    "normalize_reward": False,  # 奖励归一化
    "clip_obs": 10.0,          # 观测裁剪
    "clip_reward": 1.0,        # 奖励裁剪
    
    # 回合控制
    "max_episode_steps": None,  # 最大步数
    "noop_max": 30,            # 初始无操作步数
    
    # 向量化环境
    "num_envs": 1,             # 并行环境数量
    "async_envs": False        # 异步环境
}
```

### 视觉环境配置

```python
visual_env_config = {
    "frame_stack": 4,
    "frame_skip": 4,
    "resize": (84, 84),
    "grayscale": True,
    "normalize_obs": True,
    "clip_obs": 1.0
}
```

## 📈 监控和日志配置

### TensorBoard 配置

```python
tensorboard_config = {
    "log_dir": "./logs",
    "use_tensorboard": True,
    "log_freq": 10,
    "histogram_freq": 100,
    "write_graph": True,
    "write_images": False
}
```

### Wandb 配置

```python
wandb_config = {
    "use_wandb": True,
    "project": "agentrl-experiments",
    "name": "ppo-cartpole",
    "tags": ["ppo", "cartpole"],
    "notes": "PPO training on CartPole environment",
    "config": {
        "algorithm": "PPO",
        "environment": "CartPole-v1"
    }
}
```

## 🔍 回调系统

### 自定义回调

```python
class CustomCallback:
    def on_training_start(self, **kwargs):
        """训练开始时调用"""
        pass
    
    def on_episode_end(self, episode, reward, **kwargs):
        """回合结束时调用"""
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")
    
    def on_training_end(self, **kwargs):
        """训练结束时调用"""
        pass

# 使用回调
trainer = PPOTrainer(agent, env)
trainer.add_callback(CustomCallback())
trainer.train()
```

### 内置回调

- `EarlyStoppingCallback`: 早停回调
- `CheckpointCallback`: 模型保存回调
- `ProgressCallback`: 进度显示回调
- `WandbCallback`: Wandb集成回调

## 🔧 高级用法

### 1. 自定义网络架构

```python
from agentrl.core.network import MLP
import torch.nn as nn

class CustomNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.features = MLP(input_dim, [128, 128], 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        features = self.features(x)
        return self.actor(features), self.critic(features)

# 使用自定义网络
agent_config = {
    "network_class": CustomNetwork,
    "network_kwargs": {"input_dim": 4, "output_dim": 2}
}
```

### 2. 多GPU训练

```python
# 启用多GPU支持
config.set("training.use_gpu", True)
config.set("training.gpu_ids", [0, 1])
config.set("training.data_parallel", True)

trainer = PPOTrainer(agent, env, config)
trainer.train()
```

### 3. 分布式训练

```python
# 分布式配置
config.set("distributed.backend", "nccl")
config.set("distributed.init_method", "tcp://localhost:23456")
config.set("distributed.world_size", 4)
config.set("distributed.rank", 0)
```

## ❗ 异常处理

### 常见异常

- `ValueError`: 参数配置错误
- `RuntimeError`: 运行时错误
- `NotImplementedError`: 功能未实现
- `ImportError`: 依赖库未安装

### 调试模式

```python
# 启用调试模式
config.set("debug.enabled", True)
config.set("debug.log_level", "DEBUG")
config.set("debug.save_intermediate", True)
```

这个API文档提供了AgentRL框架的完整使用指南，涵盖了所有主要功能和配置选项。通过这些接口，用户可以轻松地训练、评估和部署强化学习智能体。