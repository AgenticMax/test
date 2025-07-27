"""配置管理工具"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union


class Config:
    """配置管理类"""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], str]] = None):
        """
        初始化配置
        
        Args:
            config: 配置字典或配置文件路径
        """
        self._config = {}
        
        if isinstance(config, dict):
            self._config = config.copy()
        elif isinstance(config, str):
            self.load_from_file(config)
        elif config is None:
            pass
        else:
            raise ValueError("config必须是字典、文件路径或None")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other: Union[Dict[str, Any], 'Config']) -> None:
        """更新配置"""
        if isinstance(other, Config):
            other_config = other._config
        else:
            other_config = other
        
        self._deep_update(self._config, other_config)
    
    def _deep_update(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                self._deep_update(d1[key], value)
            else:
                d1[key] = value
    
    def load_from_file(self, filepath: str) -> None:
        """从文件加载配置"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            if ext == '.json':
                config = json.load(f)
            elif ext in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
        
        self._config = config
    
    def save_to_file(self, filepath: str) -> None:
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        ext = os.path.splitext(filepath)[1].lower()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if ext == '.json':
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            elif ext in ['.yml', '.yaml']:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
    
    def keys(self):
        """获取所有键"""
        return self._config.keys()
    
    def values(self):
        """获取所有值"""
        return self._config.values()
    
    def items(self):
        """获取所有键值对"""
        return self._config.items()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式赋值"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持in操作"""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def create_default_config() -> Config:
    """创建默认配置"""
    default_config = {
        # 通用配置
        "experiment_name": "agentrl_experiment",
        "seed": 42,
        "device": "auto",  # auto, cpu, cuda
        
        # 环境配置
        "env": {
            "env_id": "CartPole-v1",
            "vectorized": False,
            "num_envs": 4,
            "frame_stack": 1,
            "frame_skip": 1,
            "normalize_observations": False,
            "normalize_rewards": False
        },
        
        # 智能体配置
        "agent": {
            "algorithm": "PPO",
            "hidden_dims": [256, 256],
            "lr": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 100000
        },
        
        # PPO特定配置
        "ppo": {
            "clip_ratio": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 4,
            "update_interval": 2048,
            "gae_lambda": 0.95
        },
        
        # DQN特定配置
        "dqn": {
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "target_update_freq": 1000,
            "learning_starts": 1000,
            "train_freq": 4,
            "double_dqn": True,
            "dueling_dqn": True,
            "prioritized_replay": False
        },
        
        # SAC特定配置
        "sac": {
            "tau": 0.005,
            "alpha": 0.2,
            "auto_alpha": True,
            "target_entropy": None,
            "learning_starts": 1000,
            "train_freq": 1
        },
        
        # A3C特定配置
        "a3c": {
            "num_workers": None,  # None表示使用CPU核心数
            "n_steps": 5,
            "gae_lambda": 1.0
        },
        
        # 训练配置
        "training": {
            "max_episodes": 1000,
            "max_steps_per_episode": 1000,
            "eval_freq": 100,
            "save_freq": 100,
            "log_freq": 10,
            "early_stopping_patience": 100,
            "target_reward": None
        },
        
        # 日志配置
        "logging": {
            "log_dir": "./logs",
            "save_dir": "./models",
            "use_tensorboard": True,
            "use_wandb": False,
            "wandb_config": {
                "project": "agentrl",
                "name": None
            }
        }
    }
    
    return Config(default_config)