"""DQN算法实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union, Tuple
import random

from ..core.agent import BaseAgent
from ..core.network import Network
from ..core.memory import ReplayBuffer, PrioritizedReplayBuffer


class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    
    def __init__(self, input_shape: Union[int, Tuple[int, ...]], action_dim: int, hidden_dims: list):
        super().__init__()
        
        # 共享特征提取器
        self.feature_extractor = Network(
            input_shape=input_shape,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1]
        )
        
        # 值函数流
        self.value_stream = nn.Linear(hidden_dims[-1], 1)
        
        # 优势函数流
        self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class DQNAgent(BaseAgent):
    """DQN智能体"""
    
    def __init__(self, 
                 observation_space: Any,
                 action_space: Any,
                 config: Dict[str, Any]):
        
        super().__init__(observation_space, action_space, config)
        
        # 配置参数
        self.lr = config.get("lr", 1e-4)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.batch_size = config.get("batch_size", 32)
        self.buffer_size = config.get("buffer_size", 100000)
        self.learning_starts = config.get("learning_starts", 1000)
        self.train_freq = config.get("train_freq", 4)
        
        # 算法选择
        self.double_dqn = config.get("double_dqn", True)
        self.dueling_dqn = config.get("dueling_dqn", True)
        self.prioritized_replay = config.get("prioritized_replay", False)
        
        # 动作空间
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
        else:
            raise ValueError("DQN只支持离散动作空间")
        
        # 观测空间
        if hasattr(observation_space, 'shape'):
            self.obs_shape = observation_space.shape
        else:
            self.obs_shape = (observation_space,)
        
        # 创建网络
        hidden_dims = config.get("hidden_dims", [512, 512])
        
        if self.dueling_dqn:
            self.q_network = DuelingDQN(
                input_shape=self.obs_shape,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.target_q_network = DuelingDQN(
                input_shape=self.obs_shape,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
        else:
            self.q_network = Network(
                input_shape=self.obs_shape,
                output_dim=self.action_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            self.target_q_network = Network(
                input_shape=self.obs_shape,
                output_dim=self.action_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
        
        # 复制参数到目标网络
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # 经验回放缓冲区
        if self.prioritized_replay:
            # 假设动作是标量
            action_shape = (1,) if np.isscalar(0) else (self.action_dim,)
            self.memory = PrioritizedReplayBuffer(
                capacity=self.buffer_size,
                state_shape=self.obs_shape,
                action_shape=action_shape,
                device=str(self.device)
            )
        else:
            action_shape = (1,)
            self.memory = ReplayBuffer(
                capacity=self.buffer_size,
                state_shape=self.obs_shape,
                action_shape=action_shape,
                device=str(self.device)
            )
        
        # 统计信息
        self.total_steps = 0
        self.episode_count = 0
        
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        # ε-贪婪策略
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def step(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool):
        """存储经验并可能进行学习"""
        # 存储经验
        self.memory.add(
            state=state,
            action=np.array([action]),
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        self.total_steps += 1
        
        # 开始学习
        if (self.total_steps >= self.learning_starts and 
            self.total_steps % self.train_freq == 0 and
            self.memory.is_ready(self.batch_size)):
            
            self.learn()
        
        # 更新目标网络
        if self.total_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减ε
        if training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if done:
            self.episode_count += 1
    
    def learn(self, batch: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """DQN学习更新"""
        if batch is None:
            if self.prioritized_replay:
                batch, weights, indices = self.memory.sample(self.batch_size)
            else:
                batch = self.memory.sample(self.batch_size)
                weights = torch.ones(self.batch_size).to(self.device)
                indices = None
        else:
            weights = torch.ones(batch['states'].size(0)).to(self.device)
            indices = None
        
        states = batch['states']
        actions = batch['actions'].long()
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 用主网络选择动作，目标网络计算Q值
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_q_network(next_states).gather(1, next_actions)
            else:
                # 标准DQN
                next_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0]
            
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones).unsqueeze(1)
        
        # 计算TD误差
        td_errors = current_q_values - target_q_values
        
        # 损失函数
        if self.prioritized_replay:
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        # 更新优先级
        if self.prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_np)
        
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item(),
            "epsilon": self.epsilon,
            "td_error": td_errors.abs().mean().item()
        }
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
    
    def get_q_values(self, observation: np.ndarray) -> torch.Tensor:
        """获取Q值"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state)
        
        return q_values
    
    def set_training_mode(self, training: bool = True):
        """设置训练模式"""
        self.q_network.train(training)
        self.target_q_network.train(training)
    
    def named_parameters(self):
        """获取命名参数"""
        return self.q_network.named_parameters()