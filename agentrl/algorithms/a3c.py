"""A3C算法实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union, Tuple
import threading
import multiprocessing as mp

from ..core.agent import BaseAgent
from ..core.network import ActorCriticNetwork


class A3CAgent(BaseAgent):
    """A3C智能体"""
    
    def __init__(self, 
                 observation_space: Any,
                 action_space: Any,
                 config: Dict[str, Any]):
        
        super().__init__(observation_space, action_space, config)
        
        # 配置参数
        self.lr = config.get("lr", 7e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 1.0)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.n_steps = config.get("n_steps", 5)
        self.num_workers = config.get("num_workers", mp.cpu_count())
        
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
        
        # 全局网络
        hidden_dims = config.get("hidden_dims", [256, 256])
        self.global_network = ActorCriticNetwork(
            input_shape=self.obs_shape,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            continuous_actions=self.continuous_actions,
            shared_network=config.get("shared_network", True)
        ).to(self.device)
        
        # 本地网络（用于单独训练时）
        self.local_network = ActorCriticNetwork(
            input_shape=self.obs_shape,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            continuous_actions=self.continuous_actions,
            shared_network=config.get("shared_network", True)
        ).to(self.device)
        
        # 全局优化器
        self.global_optimizer = torch.optim.Adam(
            self.global_network.parameters(), 
            lr=self.lr
        )
        
        # 共享状态（用于多进程）
        self.global_network.share_memory()
        
        # 统计信息
        self.total_steps = 0
        self.episode_count = 0
        
        # 临时存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def act(self, observation: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """选择动作"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.local_network.get_action(state, deterministic=not training)
            value = self.local_network.get_value(state)
        
        action_np = action.cpu().numpy().squeeze()
        
        if training:
            # 存储用于学习的信息
            self.states.append(observation)
            self.actions.append(action_np)
            self.values.append(value.cpu().numpy().item())
            self.log_probs.append(log_prob.cpu().numpy().item())
        
        if self.continuous_actions:
            return np.clip(action_np, self.action_space.low, self.action_space.high)
        else:
            return int(action_np)
    
    def step(self, reward: float, next_observation: np.ndarray, done: bool):
        """存储步骤信息"""
        if len(self.states) > 0:
            self.rewards.append(reward)
            self.dones.append(done)
            self.total_steps += 1
            
            # 当收集到足够步骤或回合结束时，进行学习
            if len(self.states) >= self.n_steps or done:
                self.learn(next_observation, done)
                self._clear_buffers()
            
            if done:
                self.episode_count += 1
    
    def learn(self, next_observation: np.ndarray = None, done: bool = False) -> Dict[str, float]:
        """A3C学习更新"""
        if len(self.states) == 0:
            return {}
        
        # 计算最终值
        if done:
            final_value = 0.0
        else:
            next_state = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                final_value = self.local_network.get_value(next_state).cpu().numpy().item()
        
        # 计算returns和advantages
        returns = self._compute_returns(final_value)
        advantages = self._compute_advantages(returns)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # 标准化advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 前向传播
        if self.continuous_actions:
            action_mean, action_std, values = self.local_network(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            action_logits, values = self.local_network(states)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions.long())
            entropy = dist.entropy().mean()
        
        # 策略损失
        policy_loss = -(new_log_probs * advantages_tensor).mean()
        
        # 值函数损失
        value_loss = F.mse_loss(values.squeeze(), returns_tensor)
        
        # 总损失
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss - 
                     self.entropy_coef * entropy)
        
        # 反向传播（使用全局网络的梯度）
        self.global_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.max_grad_norm)
        
        # 将本地网络的梯度复制到全局网络
        for local_param, global_param in zip(self.local_network.parameters(), 
                                           self.global_network.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad
        
        # 更新全局网络
        self.global_optimizer.step()
        
        # 同步本地网络参数
        self.local_network.load_state_dict(self.global_network.state_dict())
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }
    
    def _compute_returns(self, final_value: float) -> np.ndarray:
        """计算returns"""
        returns = []
        running_return = final_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            running_return = reward + self.gamma * running_return * (1 - done)
            returns.insert(0, running_return)
        
        return np.array(returns)
    
    def _compute_advantages(self, returns: np.ndarray) -> np.ndarray:
        """计算advantages (GAE)"""
        if self.gae_lambda == 1.0:
            # 如果lambda=1，advantages就是returns - values
            return returns - np.array(self.values)
        
        advantages = []
        gae = 0.0
        
        # 添加最终值
        values_with_final = self.values + [returns[-1] if len(returns) > 0 else 0]
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = returns[-1] if len(returns) > 0 else 0
            else:
                next_value = values_with_final[i + 1]
            
            delta = (self.rewards[i] + 
                    self.gamma * next_value * (1 - self.dones[i]) - 
                    self.values[i])
            
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - self.dones[i])
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def _clear_buffers(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def sync_with_global(self):
        """与全局网络同步"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'global_optimizer_state_dict': self.global_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_network.load_state_dict(checkpoint['global_network_state_dict'])
        self.local_network.load_state_dict(checkpoint['global_network_state_dict'])
        self.global_optimizer.load_state_dict(checkpoint['global_optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
    
    def set_training_mode(self, training: bool = True):
        """设置训练模式"""
        self.global_network.train(training)
        self.local_network.train(training)
    
    def named_parameters(self):
        """获取命名参数"""
        return self.global_network.named_parameters()


class A3CWorker:
    """A3C工作进程"""
    
    def __init__(self, 
                 worker_id: int,
                 global_network: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 env_config: Dict[str, Any],
                 config: Dict[str, Any]):
        
        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.env_config = env_config
        self.config = config
        
        # 创建本地智能体
        from ..core.environment import Environment
        self.env = Environment(**env_config)
        
        self.agent = A3CAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=config
        )
        
        # 与全局网络同步
        self.agent.sync_with_global()
    
    def run(self, max_episodes: int = 1000):
        """运行工作进程"""
        for episode in range(max_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.act(obs, training=True)
                next_obs, reward, done, _ = self.env.step(action)
                
                self.agent.step(reward, next_obs, done)
                
                obs = next_obs
                episode_reward += reward
            
            # 定期同步
            if episode % 10 == 0:
                self.agent.sync_with_global()
            
            print(f"Worker {self.worker_id}, Episode {episode}, Reward: {episode_reward}")


def train_a3c_parallel(env_config: Dict[str, Any], 
                      agent_config: Dict[str, Any],
                      num_workers: int = None,
                      max_episodes: int = 1000):
    """并行训练A3C"""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # 创建全局网络和优化器
    from ..core.environment import Environment
    temp_env = Environment(**env_config)
    
    global_agent = A3CAgent(
        observation_space=temp_env.observation_space,
        action_space=temp_env.action_space,
        config=agent_config
    )
    
    temp_env.close()
    
    # 创建工作进程
    processes = []
    for worker_id in range(num_workers):
        worker = A3CWorker(
            worker_id=worker_id,
            global_network=global_agent.global_network,
            optimizer=global_agent.global_optimizer,
            env_config=env_config,
            config=agent_config
        )
        
        process = mp.Process(target=worker.run, args=(max_episodes,))
        processes.append(process)
        process.start()
    
    # 等待所有进程完成
    for process in processes:
        process.join()
    
    return global_agent