"""
AgentRL - 高效强化学习智能体框架
==================================

一个专为提升推理和训练效率而设计的现代化强化学习智能体框架。
"""

__version__ = "0.1.0"
__author__ = "AgentRL Team"

from .core.agent import Agent
from .core.environment import Environment
from .training.trainer import Trainer
from .training.ppo_trainer import PPOTrainer
from .training.dqn_trainer import DQNTrainer
from .inference.engine import InferenceEngine
from .utils.logger import Logger
from .utils.config import Config

__all__ = [
    "Agent",
    "Environment", 
    "Trainer",
    "PPOTrainer",
    "DQNTrainer",
    "InferenceEngine",
    "Logger",
    "Config",
]