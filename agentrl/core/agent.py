"""智能体核心类"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .network import Network
from .memory import ReplayBuffer


class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, 
                 observation_space: Any,
                 action_space: Any,
                 config: Dict[str, Any]):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def act(self, observation: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """选择动作"""
        pass
        
    @abstractmethod
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """学习更新"""
        pass
        
    @abstractmethod
    def save(self, filepath: str) -> None:
        """保存模型"""
        pass
        
    @abstractmethod
    def load(self, filepath: str) -> None:
        """加载模型"""
        pass


class Agent:
    """智能体工厂类"""
    
    def __init__(self, 
                 algorithm: str,
                 observation_space: Any,
                 action_space: Any,
                 config: Optional[Dict[str, Any]] = None):
        
        self.algorithm = algorithm.upper()
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}
        
        # 创建对应算法的智能体
        self.agent = self._create_agent()
        
    def _create_agent(self) -> BaseAgent:
        """根据算法类型创建智能体"""
        if self.algorithm == "PPO":
            from ..algorithms.ppo import PPOAgent
            return PPOAgent(self.observation_space, self.action_space, self.config)
        elif self.algorithm == "DQN":
            from ..algorithms.dqn import DQNAgent
            return DQNAgent(self.observation_space, self.action_space, self.config)
        elif self.algorithm == "SAC":
            from ..algorithms.sac import SACAgent
            return SACAgent(self.observation_space, self.action_space, self.config)
        elif self.algorithm == "A3C":
            from ..algorithms.a3c import A3CAgent
            return A3CAgent(self.observation_space, self.action_space, self.config)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def act(self, observation: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """选择动作"""
        return self.agent.act(observation, training)
    
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """学习更新"""
        return self.agent.learn(batch)
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        self.agent.save(filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        self.agent.load(filepath)
    
    def get_network_parameters(self) -> Dict[str, torch.Tensor]:
        """获取网络参数"""
        return {name: param for name, param in self.agent.named_parameters()}
    
    def set_training_mode(self, training: bool = True) -> None:
        """设置训练模式"""
        if hasattr(self.agent, 'train'):
            self.agent.train(training)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        if hasattr(self.agent, 'memory'):
            return {
                "memory_size": len(self.agent.memory),
                "memory_capacity": self.agent.memory.capacity if hasattr(self.agent.memory, 'capacity') else 0
            }
        return {}