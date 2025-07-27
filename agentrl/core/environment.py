"""环境封装类"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
from concurrent.futures import ThreadPoolExecutor
import threading


class VectorizedEnvironment:
    """向量化环境，支持并行执行"""
    
    def __init__(self, env_id: str, num_envs: int = 4, **kwargs):
        self.env_id = env_id
        self.num_envs = num_envs
        self.envs = [gym.make(env_id, **kwargs) for _ in range(num_envs)]
        
        # 获取环境信息
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        self._states = None
        self._dones = [False] * num_envs
        
    def reset(self) -> np.ndarray:
        """重置所有环境"""
        observations = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            observations.append(obs)
            self._dones[i] = False
        
        self._states = np.array(observations)
        return self._states
    
    def step(self, actions: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """并行执行动作"""
        observations, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self._dones[i]:
                # 如果环境已结束，重置环境
                obs, _ = env.reset()
                reward = 0.0
                done = False
                info = {}
                self._dones[i] = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self._dones[i] = done
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        self._states = np.array(observations)
        return self._states, np.array(rewards), np.array(dones), infos
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()


class Environment:
    """环境封装类，提供额外功能"""
    
    def __init__(self, 
                 env_id: str,
                 vectorized: bool = False,
                 num_envs: int = 4,
                 frame_stack: int = 1,
                 frame_skip: int = 1,
                 normalize_observations: bool = False,
                 normalize_rewards: bool = False,
                 **kwargs):
        
        self.env_id = env_id
        self.vectorized = vectorized
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        
        # 创建环境
        if vectorized:
            self.env = VectorizedEnvironment(env_id, num_envs, **kwargs)
        else:
            self.env = gym.make(env_id, **kwargs)
        
        # 环境信息
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # 观测和奖励归一化统计
        if self.normalize_observations:
            self.obs_mean = np.zeros(self.observation_space.shape)
            self.obs_var = np.ones(self.observation_space.shape)
            self.obs_count = 0
        
        if self.normalize_rewards:
            self.reward_mean = 0.0
            self.reward_var = 1.0
            self.reward_count = 0
        
        # 帧堆叠缓存
        if self.frame_stack > 1:
            self.frame_buffer = None
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        if self.vectorized:
            obs = self.env.reset()
        else:
            obs, _ = self.env.reset()
        
        # 初始化帧缓存
        if self.frame_stack > 1:
            if self.vectorized:
                self.frame_buffer = np.repeat(obs[np.newaxis], self.frame_stack, axis=0)
                return self.frame_buffer.transpose(1, 0, 2).reshape(len(obs), -1)
            else:
                self.frame_buffer = np.repeat(obs[np.newaxis], self.frame_stack, axis=0)
                return self.frame_buffer.flatten()
        
        return self._normalize_observation(obs)
    
    def step(self, action) -> Tuple[np.ndarray, Union[float, np.ndarray], Union[bool, np.ndarray], Union[Dict, List[Dict]]]:
        """执行动作"""
        total_reward = 0
        done = False
        info = {}
        
        # 帧跳跃
        for _ in range(self.frame_skip):
            if self.vectorized:
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if np.any(done):
                    break
            else:
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done:
                    break
        
        # 更新帧缓存
        if self.frame_stack > 1:
            if self.vectorized:
                self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
                self.frame_buffer[-1] = obs
                obs = self.frame_buffer.transpose(1, 0, 2).reshape(len(obs), -1)
            else:
                self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
                self.frame_buffer[-1] = obs
                obs = self.frame_buffer.flatten()
        
        obs = self._normalize_observation(obs)
        total_reward = self._normalize_reward(total_reward)
        
        return obs, total_reward, done, info
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测"""
        if not self.normalize_observations:
            return obs
        
        # 更新统计信息
        if self.vectorized:
            batch_mean = np.mean(obs, axis=0)
            batch_var = np.var(obs, axis=0)
            batch_count = len(obs)
        else:
            batch_mean = obs
            batch_var = np.zeros_like(obs)
            batch_count = 1
        
        # 在线更新均值和方差
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        
        self.obs_mean += delta * batch_count / total_count
        self.obs_var = (self.obs_count * self.obs_var + 
                       batch_count * batch_var + 
                       np.square(delta) * self.obs_count * batch_count / total_count) / total_count
        self.obs_count = total_count
        
        # 归一化
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)
    
    def _normalize_reward(self, reward: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """归一化奖励"""
        if not self.normalize_rewards:
            return reward
        
        # 更新奖励统计
        if self.vectorized:
            batch_mean = np.mean(reward)
            batch_var = np.var(reward)
            batch_count = len(reward)
        else:
            batch_mean = reward
            batch_var = 0.0
            batch_count = 1
        
        # 在线更新
        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count
        
        self.reward_mean += delta * batch_count / total_count
        self.reward_var = (self.reward_count * self.reward_var + 
                          batch_count * batch_var + 
                          np.square(delta) * self.reward_count * batch_count / total_count) / total_count
        self.reward_count = total_count
        
        # 归一化
        return reward / (np.sqrt(self.reward_var) + 1e-8)
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def render(self, mode: str = "human"):
        """渲染环境"""
        if hasattr(self.env, 'render'):
            return self.env.render()
        elif hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            return self.env.envs[0].render()
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """获取回合统计信息"""
        stats = {
            "observation_space": str(self.observation_space),
            "action_space": str(self.action_space),
            "vectorized": self.vectorized,
            "frame_stack": self.frame_stack,
            "frame_skip": self.frame_skip,
        }
        
        if self.normalize_observations:
            stats.update({
                "obs_mean": self.obs_mean.tolist() if hasattr(self.obs_mean, 'tolist') else self.obs_mean,
                "obs_var": self.obs_var.tolist() if hasattr(self.obs_var, 'tolist') else self.obs_var,
                "obs_count": self.obs_count,
            })
        
        if self.normalize_rewards:
            stats.update({
                "reward_mean": self.reward_mean,
                "reward_var": self.reward_var,
                "reward_count": self.reward_count,
            })
        
        return stats