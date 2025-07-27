"""SAC算法实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union, Tuple

from ..core.agent import BaseAgent
from ..core.network import Network
from ..core.memory import ReplayBuffer


class SACAgent(BaseAgent):
    """SAC智能体"""
    
    def __init__(self, 
                 observation_space: Any,
                 action_space: Any,
                 config: Dict[str, Any]):
        
        super().__init__(observation_space, action_space, config)
        
        # 配置参数
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.alpha = config.get("alpha", 0.2)
        self.auto_alpha = config.get("auto_alpha", True)
        self.target_entropy = config.get("target_entropy", None)
        self.batch_size = config.get("batch_size", 256)
        self.buffer_size = config.get("buffer_size", 1000000)
        self.learning_starts = config.get("learning_starts", 1000)
        self.train_freq = config.get("train_freq", 1)
        
        # 动作空间
        if hasattr(action_space, 'shape'):
            self.action_dim = action_space.shape[0]
            self.action_high = torch.FloatTensor(action_space.high).to(self.device)
            self.action_low = torch.FloatTensor(action_space.low).to(self.device)
        else:
            raise ValueError("SAC只支持连续动作空间")
        
        # 观测空间
        if hasattr(observation_space, 'shape'):
            self.obs_shape = observation_space.shape
        else:
            self.obs_shape = (observation_space,)
        
        # 网络参数
        hidden_dims = config.get("hidden_dims", [256, 256])
        
        # Actor网络
        self.actor = SACActorNetwork(
            self.obs_shape, 
            self.action_dim, 
            hidden_dims
        ).to(self.device)
        
        # Critic网络（两个Q网络）
        self.critic1 = Network(
            input_shape=self.obs_shape[0] + self.action_dim if len(self.obs_shape) == 1 else 
                       (self.obs_shape[0] * np.prod(self.obs_shape[1:]) + self.action_dim,),
            output_dim=1,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.critic2 = Network(
            input_shape=self.obs_shape[0] + self.action_dim if len(self.obs_shape) == 1 else 
                       (self.obs_shape[0] * np.prod(self.obs_shape[1:]) + self.action_dim,),
            output_dim=1,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # 目标网络
        self.target_critic1 = Network(
            input_shape=self.obs_shape[0] + self.action_dim if len(self.obs_shape) == 1 else 
                       (self.obs_shape[0] * np.prod(self.obs_shape[1:]) + self.action_dim,),
            output_dim=1,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_critic2 = Network(
            input_shape=self.obs_shape[0] + self.action_dim if len(self.obs_shape) == 1 else 
                       (self.obs_shape[0] * np.prod(self.obs_shape[1:]) + self.action_dim,),
            output_dim=1,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        
        # 自动调节α
        if self.auto_alpha:
            if self.target_entropy is None:
                self.target_entropy = -self.action_dim
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        
        # 经验回放缓冲区
        action_shape = (self.action_dim,)
        self.memory = ReplayBuffer(
            capacity=self.buffer_size,
            state_shape=self.obs_shape,
            action_shape=action_shape,
            device=str(self.device)
        )
        
        # 统计信息
        self.total_steps = 0
        self.episode_count = 0
    
    def act(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action, _ = self.actor.sample(state)
            else:
                action, _ = self.actor.sample(state, deterministic=True)
        
        action = action.cpu().numpy().squeeze()
        return np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
    
    def step(self, 
             state: np.ndarray, 
             action: np.ndarray, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool):
        """存储经验并可能进行学习"""
        self.memory.add(
            state=state,
            action=action,
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
        
        if done:
            self.episode_count += 1
    
    def learn(self, batch: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """SAC学习更新"""
        if batch is None:
            batch = self.memory.sample(self.batch_size)
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 计算目标Q值
            next_state_actions = torch.cat([next_states, next_actions], dim=1)
            target_q1 = self.target_critic1(next_state_actions)
            target_q2 = self.target_critic2(next_state_actions)
            
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
        
        # 当前Q值
        state_actions = torch.cat([states, actions], dim=1)
        current_q1 = self.critic1(state_actions)
        current_q2 = self.critic2(state_actions)
        
        # Critic损失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新Critic1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # 更新Critic2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        new_state_actions = torch.cat([states, new_actions], dim=1)
        
        q1_new = self.critic1(new_state_actions)
        q2_new = self.critic2(new_state_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新α
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha,
            "q_value": current_q1.mean().item(),
        }
    
    def _soft_update(self, target_net, net):
        """软更新目标网络"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        if self.auto_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.auto_alpha and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
    
    def set_training_mode(self, training: bool = True):
        """设置训练模式"""
        self.actor.train(training)
        self.critic1.train(training)
        self.critic2.train(training)
    
    def named_parameters(self):
        """获取命名参数"""
        for name, param in self.actor.named_parameters():
            yield f"actor.{name}", param
        for name, param in self.critic1.named_parameters():
            yield f"critic1.{name}", param
        for name, param in self.critic2.named_parameters():
            yield f"critic2.{name}", param


class SACActorNetwork(nn.Module):
    """SAC Actor网络"""
    
    def __init__(self, obs_shape, action_dim, hidden_dims, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        if len(obs_shape) == 1:
            input_dim = obs_shape[0]
        else:
            input_dim = np.prod(obs_shape)
        
        # 共享网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # 均值和标准差分支
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        if len(state.shape) > 2:
            state = state.view(state.size(0), -1)
        
        features = self.shared_net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action).sum(dim=-1, keepdim=True)
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化采样
            action = torch.tanh(x_t)
            
            # 计算log概率
            log_prob = normal.log_prob(x_t)
            # 补偿tanh变换的雅可比行列式
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob