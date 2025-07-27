"""DQN训练器"""

import numpy as np
from typing import Dict, Any
from .trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    """DQN专用训练器"""
    
    def __init__(self, agent, env, config=None):
        super().__init__(agent, env, config)
        
        # DQN特定参数
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        
    def train_step(self, episode: int) -> Dict[str, Any]:
        """DQN训练步骤"""
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        training_loss = {}
        
        while not done and episode_length < self.max_steps_per_episode:
            action = self.agent.act(obs, training=True)
            next_obs, reward, done, info = self.env.step(action)
            
            # DQN使用step方法存储经验并可能学习
            self.agent.step(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # 获取最近的训练损失（如果有的话）
        if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'total_steps'):
            # 检查是否刚刚进行了学习
            if (self.agent.agent.total_steps >= self.agent.agent.learning_starts and 
                self.agent.agent.total_steps % self.agent.agent.train_freq == 0):
                
                # 手动调用learn获取损失信息
                if self.agent.agent.memory.is_ready(self.agent.agent.batch_size):
                    training_loss = self.agent.agent.learn()
                    
                    # 记录训练损失
                    if training_loss and "loss" in training_loss:
                        self.training_losses.append(training_loss["loss"])
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "training_loss": training_loss
        }