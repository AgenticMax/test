"""经验回放缓冲区"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque, namedtuple
import threading


Transition = namedtuple('Transition', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int,
                 state_shape: Tuple[int, ...],
                 action_shape: Tuple[int, ...],
                 device: str = "cpu"):
        
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = torch.device(device)
        
        # 预分配内存
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        
        # 线程锁
        self._lock = threading.Lock()
    
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """添加经验"""
        with self._lock:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        if self.size < batch_size:
            batch_size = self.size
        
        with self._lock:
            indices = np.random.choice(self.size, batch_size, replace=False)
            
            batch = {
                'states': torch.FloatTensor(self.states[indices]).to(self.device),
                'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
                'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
                'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
                'dones': torch.BoolTensor(self.dones[indices]).to(self.device)
            }
        
        return batch
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够的经验进行采样"""
        return self.size >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放缓冲区"""
    
    def __init__(self,
                 capacity: int,
                 state_shape: Tuple[int, ...],
                 action_shape: Tuple[int, ...],
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 1e-6,
                 device: str = "cpu"):
        
        super().__init__(capacity, state_shape, action_shape, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # 优先级存储
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        # 构建sum tree
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        
        self.tree_capacity = tree_capacity
        self.sum_tree = np.zeros(2 * tree_capacity - 1)
        self.min_tree = np.full(2 * tree_capacity - 1, float('inf'))
    
    def _update_tree(self, idx: int, priority: float):
        """更新sum tree"""
        tree_idx = idx + self.tree_capacity - 1
        
        # 更新sum tree
        delta = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        
        # 向上传播
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += delta
        
        # 更新min tree
        tree_idx = idx + self.tree_capacity - 1
        self.min_tree[tree_idx] = priority
        
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx + 1],
                self.min_tree[2 * tree_idx + 2]
            )
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """添加经验"""
        super().add(state, action, reward, next_state, done)
        
        # 设置最大优先级
        self.priorities[self.ptr - 1] = self.max_priority
        self._update_tree(self.ptr - 1, self.max_priority)
    
    def _sample_idx(self, segment_start: float, segment_end: float) -> int:
        """在segment中采样索引"""
        target = random.uniform(segment_start, segment_end)
        tree_idx = 0
        
        while tree_idx < self.tree_capacity - 1:
            left_child = 2 * tree_idx + 1
            right_child = 2 * tree_idx + 2
            
            if target <= self.sum_tree[left_child]:
                tree_idx = left_child
            else:
                target -= self.sum_tree[left_child]
                tree_idx = right_child
        
        return tree_idx - (self.tree_capacity - 1)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """采样批次"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = []
        weights = []
        segment_size = self.sum_tree[0] / batch_size
        
        with self._lock:
            for i in range(batch_size):
                segment_start = segment_size * i
                segment_end = segment_size * (i + 1)
                idx = self._sample_idx(segment_start, segment_end)
                indices.append(idx)
                
                # 计算重要性采样权重
                priority = self.priorities[idx]
                prob = priority / self.sum_tree[0]
                weight = (self.size * prob) ** (-self.beta)
                weights.append(weight)
            
            # 归一化权重
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
            
            batch = {
                'states': torch.FloatTensor(self.states[indices]).to(self.device),
                'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
                'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
                'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
                'dones': torch.BoolTensor(self.dones[indices]).to(self.device)
            }
            
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            
            # 更新beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, weights_tensor, np.array(indices)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + 1e-6) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
                self._update_tree(idx, priority)


class EpisodeBuffer:
    """回合缓冲区，用于on-policy算法"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_data = []
    
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            value: float = 0.0,
            log_prob: float = 0.0):
        """添加一步经验"""
        self.episode_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value,
            'log_prob': log_prob
        })
    
    def finish_episode(self, final_value: float = 0.0, gamma: float = 0.99):
        """结束回合，计算returns和advantages"""
        if not self.episode_data:
            return
        
        # 计算returns
        returns = []
        running_return = final_value
        
        for step in reversed(self.episode_data):
            running_return = step['reward'] + gamma * running_return * (1 - step['done'])
            returns.insert(0, running_return)
        
        # 计算advantages (GAE)
        advantages = []
        gae = 0.0
        lambda_gae = 0.95
        
        for i in reversed(range(len(self.episode_data))):
            step = self.episode_data[i]
            
            if i == len(self.episode_data) - 1:
                next_value = final_value
            else:
                next_value = self.episode_data[i + 1]['value']
            
            delta = step['reward'] + gamma * next_value * (1 - step['done']) - step['value']
            gae = delta + gamma * lambda_gae * gae * (1 - step['done'])
            advantages.insert(0, gae)
        
        # 添加到缓冲区
        episode = {
            'states': np.array([step['state'] for step in self.episode_data]),
            'actions': np.array([step['action'] for step in self.episode_data]),
            'rewards': np.array([step['reward'] for step in self.episode_data]),
            'values': np.array([step['value'] for step in self.episode_data]),
            'log_probs': np.array([step['log_prob'] for step in self.episode_data]),
            'returns': np.array(returns),
            'advantages': np.array(advantages)
        }
        
        self.buffer.append(episode)
        self.episode_data = []
    
    def get_all_data(self) -> Dict[str, np.ndarray]:
        """获取所有数据"""
        if not self.buffer:
            return {}
        
        # 合并所有回合数据
        all_data = {
            'states': np.concatenate([ep['states'] for ep in self.buffer]),
            'actions': np.concatenate([ep['actions'] for ep in self.buffer]),
            'rewards': np.concatenate([ep['rewards'] for ep in self.buffer]),
            'values': np.concatenate([ep['values'] for ep in self.buffer]),
            'log_probs': np.concatenate([ep['log_probs'] for ep in self.buffer]),
            'returns': np.concatenate([ep['returns'] for ep in self.buffer]),
            'advantages': np.concatenate([ep['advantages'] for ep in self.buffer])
        }
        
        return all_data
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.episode_data = []
    
    def __len__(self) -> int:
        return len(self.buffer)