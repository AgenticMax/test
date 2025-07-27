"""基础测试"""

import pytest
import numpy as np
import gymnasium as gym

def test_environment_creation():
    """测试环境创建"""
    from agentrl.core.environment import Environment
    
    env = Environment("CartPole-v1")
    assert env.env_id == "CartPole-v1"
    assert env.observation_space is not None
    assert env.action_space is not None
    env.close()

def test_agent_creation():
    """测试智能体创建"""
    from agentrl import Agent, Environment
    
    env = Environment("CartPole-v1")
    agent = Agent("PPO", env.observation_space, env.action_space)
    
    assert agent.algorithm == "PPO"
    assert hasattr(agent, 'agent')
    env.close()

def test_network_creation():
    """测试网络创建"""
    from agentrl.core.network import Network
    
    network = Network(input_shape=4, output_dim=2, hidden_dims=[64, 64])
    assert network.input_shape == 4
    assert network.output_dim == 2

def test_memory_creation():
    """测试内存缓冲区创建"""
    from agentrl.core.memory import ReplayBuffer
    
    buffer = ReplayBuffer(capacity=1000, state_shape=(4,), action_shape=(1,))
    assert buffer.capacity == 1000
    assert len(buffer) == 0

def test_config():
    """测试配置管理"""
    from agentrl.utils.config import Config
    
    config = Config({"test": {"value": 42}})
    assert config.get("test.value") == 42
    assert config.get("nonexistent", "default") == "default"

if __name__ == "__main__":
    pytest.main([__file__])