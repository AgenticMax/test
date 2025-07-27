"""PPO训练器"""

import numpy as np
from typing import Dict, Any
from .trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """PPO专用训练器"""
    
    def __init__(self, agent, env, config=None):
        super().__init__(agent, env, config)
        
        # PPO特定参数
        self.update_interval = self.config.get("update_interval", 2048)
        self.collect_steps = 0
        
    def train_step(self, episode: int) -> Dict[str, Any]:
        """PPO训练步骤"""
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        training_loss = {}
        
        while not done and episode_length < self.max_steps_per_episode:
            action = self.agent.act(obs, training=True)
            next_obs, reward, done, info = self.env.step(action)
            
            # PPO使用step方法存储经验
            self.agent.step(reward, next_obs, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.collect_steps += 1
            
            # 定期更新
            if self.collect_steps >= self.update_interval:
                training_loss = self.agent.learn()
                self.collect_steps = 0
                
                # 记录训练损失
                if training_loss:
                    for key, value in training_loss.items():
                        if isinstance(value, (int, float)):
                            self.training_losses.append(value)
                            break  # 只记录第一个数值损失
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "training_loss": training_loss
        }