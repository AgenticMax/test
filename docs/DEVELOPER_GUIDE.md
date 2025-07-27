# AgentRL 开发者指南

## 🎯 开发环境设置

### 1. 克隆和安装

```bash
# 克隆仓库
git clone https://github.com/agentrl/agentrl.git
cd agentrl

# 创建开发环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -e .  # 可编辑安装

# 安装开发工具
pip install pytest black flake8 mypy pre-commit
```

### 2. 开发工具配置

#### Pre-commit 钩子

```bash
# 安装 pre-commit 钩子
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

#### VSCode 配置

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests/"]
}
```

## 🏗️ 框架扩展指南

### 1. 添加新算法

#### 步骤1：创建算法类

```python
# agentrl/algorithms/new_algorithm.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Union
from ..core.agent import BaseAgent
from ..core.network import ActorCriticNetwork
from ..core.memory import ReplayBuffer

class NewAlgorithmAgent(BaseAgent):
    """新算法智能体实现"""
    
    def __init__(self, 
                 observation_space: Any,
                 action_space: Any,
                 config: Dict[str, Any]):
        super().__init__(observation_space, action_space, config)
        
        # 算法特定配置
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        # ... 其他参数
        
        # 创建网络
        self.network = self._build_network()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.lr
        )
        
        # 创建内存
        self.memory = ReplayBuffer(
            capacity=config.get("memory_size", 100000),
            obs_shape=self.obs_shape,
            action_shape=(self.action_dim,),
            device=self.device
        )
    
    def _build_network(self):
        """构建神经网络"""
        hidden_dims = self.config.get("hidden_dims", [64, 64])
        return ActorCriticNetwork(
            input_dim=np.prod(self.obs_shape),
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            continuous_actions=self.continuous_actions
        )
    
    def act(self, observation: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """选择动作"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous_actions:
                # 连续动作空间
                action_logits, _ = self.network(obs_tensor)
                action = torch.tanh(action_logits)
                
                if training:
                    # 添加探索噪声
                    noise = torch.randn_like(action) * 0.1
                    action = torch.clamp(action + noise, -1, 1)
                
                return action.cpu().numpy()[0]
            else:
                # 离散动作空间
                action_logits, _ = self.network(obs_tensor)
                
                if training:
                    # epsilon-greedy 探索
                    if np.random.random() < self.epsilon:
                        return np.random.randint(self.action_dim)
                
                return torch.argmax(action_logits, dim=1).item()
    
    def learn(self, batch: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """学习更新"""
        if batch is None:
            if len(self.memory) < self.config.get("batch_size", 32):
                return {}
            batch = self.memory.sample(self.config.get("batch_size", 32))
        
        # 实现算法特定的学习逻辑
        obs = batch["observations"]
        actions = batch["actions"] 
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]
        
        # 计算损失
        loss = self._compute_loss(obs, actions, rewards, next_obs, dones)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def _compute_loss(self, obs, actions, rewards, next_obs, dones):
        """计算损失函数"""
        # 实现具体的损失计算
        pass
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

#### 步骤2：注册算法

在 `agentrl/core/agent.py` 中注册新算法：

```python
def _create_agent(self) -> BaseAgent:
    """根据算法类型创建智能体"""
    if self.algorithm == "PPO":
        from ..algorithms.ppo import PPOAgent
        return PPOAgent(self.observation_space, self.action_space, self.config)
    # ... 其他算法
    elif self.algorithm == "NEW_ALGORITHM":
        from ..algorithms.new_algorithm import NewAlgorithmAgent
        return NewAlgorithmAgent(self.observation_space, self.action_space, self.config)
    else:
        raise ValueError(f"不支持的算法: {self.algorithm}")
```

#### 步骤3：更新默认配置

在 `agentrl/utils/config.py` 中添加默认配置：

```python
def create_default_config() -> Config:
    default_config = {
        # ... 现有配置
        
        # 新算法配置
        "new_algorithm": {
            "lr": 3e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "memory_size": 100000,
            "hidden_dims": [64, 64],
            # ... 其他参数
        }
    }
    return Config(default_config)
```

#### 步骤4：创建专用训练器（可选）

```python
# agentrl/training/new_algorithm_trainer.py
from .trainer import BaseTrainer

class NewAlgorithmTrainer(BaseTrainer):
    """新算法专用训练器"""
    
    def __init__(self, agent, env, config=None):
        super().__init__(agent, env, config)
        # 算法特定的训练配置
    
    def train_step(self, episode: int) -> Dict[str, Any]:
        """训练步骤实现"""
        # 实现算法特定的训练逻辑
        pass
```

### 2. 添加新网络架构

#### 创建网络类

```python
# agentrl/core/network.py 中添加
class CustomNetwork(nn.Module):
    """自定义网络架构"""
    
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        
        # 实现网络结构
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(128, output_dim)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.output_layer(features)
```

### 3. 添加新环境支持

#### 创建环境包装器

```python
# agentrl/core/environment.py 中添加
class CustomEnvironmentWrapper:
    """自定义环境包装器"""
    
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or {}
        
        # 实现特定的环境增强
        self.observation_space = self._modify_observation_space()
        self.action_space = self._modify_action_space()
    
    def reset(self):
        obs = self.env.reset()
        return self._preprocess_observation(obs)
    
    def step(self, action):
        action = self._preprocess_action(action)
        obs, reward, done, info = self.env.step(action)
        
        obs = self._preprocess_observation(obs)
        reward = self._preprocess_reward(reward)
        
        return obs, reward, done, info
    
    def _preprocess_observation(self, obs):
        """预处理观测"""
        # 实现观测预处理逻辑
        return obs
    
    def _preprocess_action(self, action):
        """预处理动作"""
        # 实现动作预处理逻辑
        return action
    
    def _preprocess_reward(self, reward):
        """预处理奖励"""
        # 实现奖励预处理逻辑
        return reward
```

## 🧪 测试指南

### 1. 测试结构

```
tests/
├── unit/                   # 单元测试
│   ├── test_agent.py
│   ├── test_network.py
│   ├── test_memory.py
│   └── test_environment.py
├── integration/            # 集成测试
│   ├── test_training.py
│   └── test_inference.py
├── algorithms/             # 算法测试
│   ├── test_ppo.py
│   ├── test_dqn.py
│   └── test_sac.py
└── conftest.py            # 测试配置
```

### 2. 编写测试

#### 单元测试示例

```python
# tests/unit/test_agent.py
import pytest
import numpy as np
import gymnasium as gym
from agentrl import Agent

class TestAgent:
    """智能体测试类"""
    
    @pytest.fixture
    def cartpole_env(self):
        """CartPole环境fixture"""
        return gym.make("CartPole-v1")
    
    @pytest.fixture
    def ppo_agent(self, cartpole_env):
        """PPO智能体fixture"""
        return Agent("PPO", cartpole_env.observation_space, cartpole_env.action_space)
    
    def test_agent_creation(self, ppo_agent):
        """测试智能体创建"""
        assert ppo_agent is not None
        assert hasattr(ppo_agent, 'act')
        assert hasattr(ppo_agent, 'learn')
    
    def test_action_selection(self, ppo_agent, cartpole_env):
        """测试动作选择"""
        obs, _ = cartpole_env.reset()
        action = ppo_agent.act(obs)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < cartpole_env.action_space.n
    
    def test_save_load(self, ppo_agent, tmp_path):
        """测试模型保存和加载"""
        filepath = tmp_path / "test_model.pt"
        
        # 保存模型
        ppo_agent.save(str(filepath))
        assert filepath.exists()
        
        # 加载模型
        ppo_agent.load(str(filepath))
```

#### 集成测试示例

```python
# tests/integration/test_training.py
import pytest
from agentrl import Agent, Environment, PPOTrainer

class TestTraining:
    """训练集成测试"""
    
    def test_ppo_training(self):
        """测试PPO训练流程"""
        env = Environment("CartPole-v1")
        agent = Agent("PPO", env.observation_space, env.action_space)
        trainer = PPOTrainer(agent, env)
        
        # 短期训练测试
        history = trainer.train(max_episodes=10)
        
        assert len(history["episode_rewards"]) == 10
        assert all(isinstance(r, (int, float)) for r in history["episode_rewards"])
    
    def test_evaluation(self):
        """测试评估功能"""
        env = Environment("CartPole-v1")
        agent = Agent("PPO", env.observation_space, env.action_space)
        trainer = PPOTrainer(agent, env)
        
        # 评估未训练的智能体
        results = trainer.evaluate(episodes=5)
        
        assert "mean_reward" in results
        assert "std_reward" in results
        assert len(results["episode_rewards"]) == 5
```

### 3. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_agent.py

# 运行特定测试方法
pytest tests/unit/test_agent.py::TestAgent::test_agent_creation

# 生成覆盖率报告
pytest --cov=agentrl --cov-report=html

# 并行运行测试
pytest -n auto  # 需要安装 pytest-xdist
```

## 📝 代码规范

### 1. 代码风格

#### Python 编码规范

- 遵循 PEP 8 标准
- 使用 Black 格式化代码
- 行长度限制为 88 字符
- 使用类型提示

#### 示例代码

```python
from typing import Dict, List, Optional, Union
import numpy as np
import torch


class ExampleClass:
    """示例类的文档字符串
    
    详细描述类的功能和用途。
    
    Args:
        param1: 参数1的描述
        param2: 参数2的描述
    """
    
    def __init__(self, param1: int, param2: Optional[str] = None) -> None:
        self.param1 = param1
        self.param2 = param2
    
    def example_method(
        self, 
        data: np.ndarray, 
        config: Dict[str, Union[int, float]]
    ) -> List[float]:
        """示例方法的文档字符串
        
        Args:
            data: 输入数据
            config: 配置参数
            
        Returns:
            处理后的结果列表
            
        Raises:
            ValueError: 当输入数据格式错误时
        """
        if data.ndim != 2:
            raise ValueError("数据必须是二维数组")
        
        # 方法实现
        result = []
        for row in data:
            processed = self._process_row(row, config)
            result.append(processed)
        
        return result
    
    def _process_row(self, row: np.ndarray, config: Dict) -> float:
        """私有方法示例"""
        # 实现细节
        return float(np.mean(row))
```

### 2. 文档字符串规范

使用 Google 风格的文档字符串：

```python
def train_agent(
    agent: BaseAgent,
    env: Environment,
    episodes: int,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[float]]:
    """训练智能体
    
    在指定环境中训练智能体指定的回合数。
    
    Args:
        agent: 要训练的智能体
        env: 训练环境
        episodes: 训练回合数
        config: 可选的训练配置
        
    Returns:
        包含训练历史的字典，键包括：
        - "episode_rewards": 每回合奖励列表
        - "episode_lengths": 每回合长度列表
        
    Raises:
        ValueError: 当episodes小于等于0时
        RuntimeError: 当训练过程中发生错误时
        
    Examples:
        >>> agent = Agent("PPO", env.observation_space, env.action_space)
        >>> env = Environment("CartPole-v1")
        >>> history = train_agent(agent, env, episodes=1000)
        >>> print(f"平均奖励: {np.mean(history['episode_rewards'])}")
    """
    # 实现
    pass
```

### 3. 错误处理

```python
def validate_config(config: Dict[str, Any]) -> None:
    """验证配置参数"""
    required_keys = ["lr", "gamma", "hidden_dims"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"缺少必需的配置参数: {key}")
    
    if not 0 < config["lr"] <= 1:
        raise ValueError(f"学习率必须在(0, 1]范围内，当前值: {config['lr']}")
    
    if not 0 <= config["gamma"] <= 1:
        raise ValueError(f"折扣因子必须在[0, 1]范围内，当前值: {config['gamma']}")

def safe_train(agent, env, episodes):
    """安全训练包装器"""
    try:
        return train_agent(agent, env, episodes)
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"运行时错误: {e}")
        # 清理资源
        cleanup_resources()
        raise
    except Exception as e:
        logger.error(f"未知错误: {e}")
        raise RuntimeError(f"训练失败: {e}") from e
```

## 🚀 性能优化

### 1. 性能分析

```python
# 使用 cProfile 进行性能分析
import cProfile
import pstats

def profile_training():
    """分析训练性能"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 运行训练代码
    train_agent(agent, env, episodes=100)
    
    profiler.disable()
    
    # 分析结果
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 显示前20个最耗时的函数
```

### 2. 内存优化

```python
import tracemalloc
import gc

def monitor_memory():
    """监控内存使用"""
    tracemalloc.start()
    
    # 运行代码
    train_agent(agent, env, episodes=100)
    
    # 检查内存使用
    current, peak = tracemalloc.get_traced_memory()
    print(f"当前内存使用: {current / 1024 / 1024:.2f} MB")
    print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    # 强制垃圾回收
    gc.collect()
```

### 3. GPU 优化

```python
def optimize_gpu_usage():
    """优化GPU使用"""
    if torch.cuda.is_available():
        # 启用自动混合精度
        scaler = torch.cuda.amp.GradScaler()
        
        # 使用autocast
        with torch.cuda.amp.autocast():
            loss = compute_loss(batch)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
```

## 📊 基准测试

### 1. 性能基准

```python
# benchmarks/performance_benchmark.py
import time
import numpy as np
from agentrl import Agent, Environment, InferenceEngine

def benchmark_inference_speed():
    """基准测试推理速度"""
    env = Environment("CartPole-v1")
    agent = Agent("PPO", env.observation_space, env.action_space)
    engine = InferenceEngine(agent, batch_size=32)
    
    # 准备测试数据
    observations = np.random.randn(1000, 4)
    
    # 预热
    for _ in range(10):
        engine.predict(observations[:32])
    
    # 基准测试
    start_time = time.time()
    for i in range(0, len(observations), 32):
        batch = observations[i:i+32]
        actions = engine.predict(batch)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = len(observations) / total_time
    
    print(f"推理速度: {fps:.2f} FPS")
    print(f"平均延迟: {total_time / len(observations) * 1000:.3f} ms")

def benchmark_training_speed():
    """基准测试训练速度"""
    env = Environment("CartPole-v1")
    agent = Agent("PPO", env.observation_space, env.action_space)
    trainer = PPOTrainer(agent, env)
    
    start_time = time.time()
    trainer.train(max_episodes=100)
    end_time = time.time()
    
    print(f"训练100回合耗时: {end_time - start_time:.2f} 秒")
```

## 🔄 持续集成

### 1. GitHub Actions 配置

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88
    
    - name: Test with pytest
      run: |
        pip install pytest pytest-cov
        pytest --cov=agentrl --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. 代码质量检查

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --ignore=E203,W503]

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.942
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## 📋 贡献流程

### 1. 分支管理

- `main`: 稳定版本分支
- `develop`: 开发分支
- `feature/*`: 功能开发分支
- `bugfix/*`: 错误修复分支
- `release/*`: 发布准备分支

### 2. 提交规范

使用约定式提交（Conventional Commits）：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

示例：
```
feat(algorithm): add SAC algorithm implementation

- Implement Soft Actor-Critic algorithm
- Add automatic entropy tuning
- Support continuous action spaces

Closes #123
```

类型说明：
- `feat`: 新功能
- `fix`: 错误修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具变动

### 3. Pull Request 流程

1. Fork 仓库
2. 创建功能分支
3. 编写代码和测试
4. 确保测试通过
5. 提交 Pull Request
6. 代码审查
7. 合并到主分支

## 🎯 最佳实践

1. **编写测试**: 所有新功能都应该有对应的测试
2. **文档完整**: 提供清晰的文档和示例
3. **性能考虑**: 注意代码性能，避免不必要的计算
4. **向后兼容**: 尽量保持API的向后兼容性
5. **模块化设计**: 保持代码的模块化和可复用性

通过遵循这些开发指南，您可以有效地为 AgentRL 框架贡献代码，扩展功能，并维护高质量的代码库。