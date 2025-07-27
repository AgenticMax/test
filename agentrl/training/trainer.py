"""基础训练器"""

import os
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..utils.logger import Logger
from ..utils.config import Config


class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(self, 
                 agent: Any,
                 env: Any,
                 config: Optional[Dict[str, Any]] = None):
        
        self.agent = agent
        self.env = env
        self.config = Config(config or {})
        
        # 训练参数
        self.max_episodes = self.config.get("max_episodes", 1000)
        self.max_steps_per_episode = self.config.get("max_steps_per_episode", 1000)
        self.eval_freq = self.config.get("eval_freq", 100)
        self.save_freq = self.config.get("save_freq", 100)
        self.log_freq = self.config.get("log_freq", 10)
        
        # 日志和保存路径
        self.log_dir = self.config.get("log_dir", "./logs")
        self.save_dir = self.config.get("save_dir", "./models")
        self.experiment_name = self.config.get("experiment_name", f"experiment_{int(time.time())}")
        
        # 创建目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 日志记录器
        self.logger = Logger(
            log_dir=os.path.join(self.log_dir, self.experiment_name),
            use_tensorboard=self.config.get("use_tensorboard", True),
            use_wandb=self.config.get("use_wandb", False),
            wandb_config=self.config.get("wandb_config", {})
        )
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.eval_rewards = []
        self.best_eval_reward = -float('inf')
        
        # 回调函数
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        """添加回调函数"""
        self.callbacks.append(callback)
    
    def _call_callbacks(self, event: str, **kwargs):
        """调用回调函数"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)
    
    @abstractmethod
    def train_step(self, episode: int) -> Dict[str, Any]:
        """训练一个回合"""
        pass
    
    def train(self, 
              max_episodes: Optional[int] = None,
              early_stopping_patience: int = 100,
              target_reward: Optional[float] = None) -> Dict[str, List]:
        """开始训练"""
        
        if max_episodes is not None:
            self.max_episodes = max_episodes
        
        self.logger.info(f"开始训练，最大回合数: {self.max_episodes}")
        self._call_callbacks("on_training_begin", trainer=self)
        
        start_time = time.time()
        patience_counter = 0
        
        try:
            with tqdm(range(self.max_episodes), desc="训练进度") as pbar:
                for episode in pbar:
                    self._call_callbacks("on_episode_begin", episode=episode, trainer=self)
                    
                    # 训练一个回合
                    episode_stats = self.train_step(episode)
                    
                    # 记录统计信息
                    episode_reward = episode_stats.get("episode_reward", 0)
                    episode_length = episode_stats.get("episode_length", 0)
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        "回合奖励": f"{episode_reward:.2f}",
                        "平均奖励": f"{np.mean(self.episode_rewards[-100:]):.2f}",
                        "回合长度": episode_length
                    })
                    
                    # 记录日志
                    if episode % self.log_freq == 0:
                        self._log_training_stats(episode, episode_stats)
                    
                    # 评估
                    if episode % self.eval_freq == 0 and episode > 0:
                        eval_stats = self.evaluate()
                        self._log_eval_stats(episode, eval_stats)
                        
                        # 早停检查
                        current_eval_reward = eval_stats.get("mean_reward", 0)
                        if current_eval_reward > self.best_eval_reward:
                            self.best_eval_reward = current_eval_reward
                            patience_counter = 0
                            # 保存最佳模型
                            self.save_model(os.path.join(self.save_dir, f"{self.experiment_name}_best.pt"))
                        else:
                            patience_counter += 1
                        
                        # 目标奖励检查
                        if target_reward is not None and current_eval_reward >= target_reward:
                            self.logger.info(f"达到目标奖励 {target_reward}，提前停止训练")
                            break
                        
                        # 早停检查
                        if patience_counter >= early_stopping_patience:
                            self.logger.info(f"评估奖励连续 {early_stopping_patience} 次未改善，提前停止训练")
                            break
                    
                    # 保存模型
                    if episode % self.save_freq == 0 and episode > 0:
                        self.save_model(os.path.join(self.save_dir, f"{self.experiment_name}_episode_{episode}.pt"))
                    
                    self._call_callbacks("on_episode_end", episode=episode, trainer=self, stats=episode_stats)
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        
        finally:
            total_time = time.time() - start_time
            self.logger.info(f"训练完成，总时间: {total_time:.2f}秒")
            
            # 保存最终模型
            self.save_model(os.path.join(self.save_dir, f"{self.experiment_name}_final.pt"))
            
            # 绘制训练曲线
            self.plot_training_curves()
            
            self._call_callbacks("on_training_end", trainer=self)
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "training_losses": self.training_losses
        }
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """评估智能体"""
        self.logger.info(f"开始评估，评估回合数: {num_episodes}")
        
        eval_rewards = []
        eval_lengths = []
        
        # 设置为评估模式
        self.agent.set_training_mode(False)
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < self.max_steps_per_episode:
                if render:
                    self.env.render()
                
                action = self.agent.act(obs, training=False)
                obs, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        # 恢复训练模式
        self.agent.set_training_mode(True)
        
        eval_stats = {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "mean_length": np.mean(eval_lengths),
        }
        
        self.eval_rewards.append(eval_stats["mean_reward"])
        
        return eval_stats
    
    def _log_training_stats(self, episode: int, stats: Dict[str, Any]):
        """记录训练统计信息"""
        # 计算移动平均
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]
        
        log_data = {
            "episode": episode,
            "episode_reward": stats.get("episode_reward", 0),
            "mean_reward_100": np.mean(recent_rewards) if recent_rewards else 0,
            "episode_length": stats.get("episode_length", 0),
            "mean_length_100": np.mean(recent_lengths) if recent_lengths else 0,
        }
        
        # 添加训练损失
        if "training_loss" in stats:
            log_data.update(stats["training_loss"])
        
        self.logger.log_scalars("train", log_data, episode)
    
    def _log_eval_stats(self, episode: int, stats: Dict[str, float]):
        """记录评估统计信息"""
        self.logger.log_scalars("eval", stats, episode)
        
        self.logger.info(
            f"Episode {episode} - 评估结果: "
            f"平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}, "
            f"范围: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]"
        )
    
    def save_model(self, filepath: str):
        """保存模型"""
        self.agent.save(filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.agent.load(filepath)
        self.logger.info(f"模型已从以下路径加载: {filepath}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 回合奖励
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label="Episode Reward")
        if len(self.episode_rewards) > 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(self.episode_rewards)), moving_avg, 
                          label="Moving Average (100)", linewidth=2)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 回合长度
        axes[0, 1].plot(self.episode_lengths, alpha=0.6, label="Episode Length")
        if len(self.episode_lengths) > 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(self.episode_lengths)), moving_avg, 
                          label="Moving Average (100)", linewidth=2)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 评估奖励
        if self.eval_rewards:
            eval_episodes = list(range(0, len(self.eval_rewards) * self.eval_freq, self.eval_freq))
            axes[1, 0].plot(eval_episodes, self.eval_rewards, 'o-', label="Eval Reward")
            axes[1, 0].set_title("Evaluation Rewards")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Reward")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 训练损失
        if self.training_losses:
            axes[1, 1].plot(self.training_losses, label="Training Loss")
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.log_dir, self.experiment_name, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线已保存到: {plot_path}")


class Trainer(BaseTrainer):
    """通用训练器"""
    
    def train_step(self, episode: int) -> Dict[str, Any]:
        """训练一个回合"""
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        training_loss = {}
        
        while not done and episode_length < self.max_steps_per_episode:
            action = self.agent.act(obs, training=True)
            next_obs, reward, done, info = self.env.step(action)
            
            # 存储经验（如果智能体支持）
            if hasattr(self.agent, 'step'):
                self.agent.step(obs, action, reward, next_obs, done)
            
            # 学习（如果智能体支持）
            if hasattr(self.agent, 'learn') and hasattr(self.agent, 'memory'):
                if (hasattr(self.agent.memory, 'is_ready') and 
                    self.agent.memory.is_ready(getattr(self.agent, 'batch_size', 32))):
                    loss_info = self.agent.learn()
                    if loss_info:
                        training_loss = loss_info
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "training_loss": training_loss
        }