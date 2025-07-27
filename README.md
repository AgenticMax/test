# AgentRL - 高效强化学习智能体框架

一个专为提升推理和训练效率而设计的现代化强化学习智能体框架。

## 🚀 特性

- **高效训练**: 支持多进程并行训练，GPU加速，内存优化
- **灵活架构**: 模块化设计，支持多种RL算法（DQN, PPO, SAC, A3C等）
- **智能推理**: 优化的推理引擎，支持批量推理和模型压缩
- **实时监控**: 集成TensorBoard和Wandb，实时训练监控
- **易于扩展**: 插件化算法接口，便于添加新算法
- **分布式支持**: 支持多机分布式训练

## 📦 安装

```bash
pip install -r requirements.txt
```

## 🎯 快速开始

```python
from agentrl import Agent, Environment, PPOTrainer

# 创建环境和智能体
env = Environment("CartPole-v1")
agent = Agent("PPO", env.observation_space, env.action_space)

# 开始训练
trainer = PPOTrainer(agent, env)
trainer.train(episodes=1000)

# 评估智能体
rewards = trainer.evaluate(episodes=100)
print(f"Average reward: {rewards.mean()}")
```

## 📁 项目结构

```
agentrl/
├── core/           # 核心组件
├── algorithms/     # RL算法实现
├── environments/   # 环境接口
├── utils/          # 工具函数
├── training/       # 训练相关
├── inference/      # 推理引擎
└── examples/       # 示例代码
```

## 🔧 支持的算法

- **Deep Q-Network (DQN)**: 离散动作空间的值函数方法
- **Proximal Policy Optimization (PPO)**: 稳定的策略梯度方法
- **Soft Actor-Critic (SAC)**: 连续控制的最大熵方法
- **Asynchronous Advantage Actor-Critic (A3C)**: 异步并行训练
- **Twin Delayed Deep Deterministic (TD3)**: 改进的确定性策略梯度

## 📊 性能优化

- 向量化环境处理
- 经验回放缓冲区优化
- 动态批量大小调整
- 混合精度训练
- 模型量化和剪枝

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License