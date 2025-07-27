"""PPO算法实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union, Tuple

from ..core.agent import BaseAgent
from ..core.network import ActorCriticNetwork
from ..core.memory import EpisodeBuffer


class PPOAgent(BaseAgent):
    """PPO智能体"""
    
    def __init__(self, 
                 observation_space: Any,
                 action_space: Any,
                 config: Dict[str, Any]):
        
        super().__init__(observation_space, action_space, config)
        
        # 配置参数
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.mini_batch_size = config.get("mini_batch_size", 64)
        
        # 确定动作空间类型
        if hasattr(action_space, 'n'):
            # 离散动作空间
            self.action_dim = action_space.n
            self.continuous_actions = False
        else:
            # 连续动作空间
            self.action_dim = action_space.shape[0]
            self.continuous_actions = True
        
        # 获取观测空间维度
        if hasattr(observation_space, 'shape'):
            self.obs_shape = observation_space.shape
        else:
            self.obs_shape = (observation_space,)
        
        # 创建网络
        hidden_dims = config.get("hidden_dims", [256, 256])
        self.network = ActorCriticNetwork(
            input_shape=self.obs_shape,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            continuous_actions=self.continuous_actions,
            shared_network=config.get("shared_network", True)
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.lr,
            eps=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.get("total_steps", 1000000) // config.get("update_interval", 2048)
        )
        
        # 经验缓冲区
        self.memory = EpisodeBuffer(capacity=config.get("buffer_capacity", 10000))
        
        # 统计信息
        self.total_steps = 0
        self.episode_count = 0
        
    def act(self, observation: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """选择动作"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.network.get_action(state, deterministic=not training)
            value = self.network.get_value(state)
        
        action_np = action.cpu().numpy().squeeze()
        
        if training:
            # 存储用于学习的信息
            self._last_state = observation
            self._last_action = action_np
            self._last_value = value.cpu().numpy().item()
            self._last_log_prob = log_prob.cpu().numpy().item()
        
        if self.continuous_actions:
            return np.clip(action_np, self.action_space.low, self.action_space.high)
        else:
            return int(action_np)
    
    def step(self, reward: float, next_observation: np.ndarray, done: bool):
        """存储经验"""
        if hasattr(self, '_last_state'):
            self.memory.add(
                state=self._last_state,
                action=self._last_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                value=self._last_value,
                log_prob=self._last_log_prob
            )
            
            self.total_steps += 1
            
            if done:
                # 回合结束，计算最终值
                if not done:
                    next_state = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
                    final_value = self.network.get_value(next_state).cpu().numpy().item()
                else:
                    final_value = 0.0
                
                self.memory.finish_episode(final_value, self.gamma)
                self.episode_count += 1
    
    def learn(self, batch: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """PPO学习更新"""
        # 获取所有数据
        data = self.memory.get_all_data()
        if not data or len(data['states']) == 0:
            return {}
        
        # 转换为torch张量
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        
        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取数据集大小
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        update_count = 0
        
        # PPO多轮更新
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            np.random.shuffle(indices)
            
            # 小批次更新
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 前向传播
                if self.continuous_actions:
                    action_mean, action_std, values = self.network(batch_states)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    action_logits, values = self.network(batch_states)
                    action_probs = F.softmax(action_logits, dim=-1)
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions.long().squeeze())
                    entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 总损失
                loss = (policy_loss + 
                       self.value_loss_coef * value_loss - 
                       self.entropy_coef * entropy)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                update_count += 1
        
        # 更新学习率
        self.scheduler.step()
        
        # 清空缓冲区
        self.memory.clear()
        
        return {
            "policy_loss": total_policy_loss / update_count if update_count > 0 else 0,
            "value_loss": total_value_loss / update_count if update_count > 0 else 0,
            "entropy": total_entropy_loss / update_count if update_count > 0 else 0,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "updates": update_count
        }
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
    
    def get_action_distribution(self, observation: np.ndarray) -> torch.distributions.Distribution:
        """获取动作分布"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous_actions:
                action_mean, action_std, _ = self.network(state)
                return torch.distributions.Normal(action_mean, action_std)
            else:
                action_logits, _ = self.network(state)
                action_probs = F.softmax(action_logits, dim=-1)
                return torch.distributions.Categorical(action_probs)
    
    def set_training_mode(self, training: bool = True):
        """设置训练模式"""
        self.network.train(training)
    
    def named_parameters(self):
        """获取命名参数"""
        return self.network.named_parameters()