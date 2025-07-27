"""日志记录工具"""

import os
import logging
from typing import Dict, Any, Optional, List
import json


class Logger:
    """训练日志记录器"""
    
    def __init__(self, 
                 log_dir: str,
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 wandb_config: Dict[str, Any] = None):
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置Python日志
        self.logger = logging.getLogger('AgentRL')
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir)
                self.info("TensorBoard日志记录已启用")
            except ImportError:
                self.warning("无法导入TensorBoard，已禁用TensorBoard日志记录")
        
        # Wandb
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb_config = wandb_config or {}
                self.wandb.init(
                    project=wandb_config.get("project", "agentrl"),
                    name=wandb_config.get("name", None),
                    config=wandb_config.get("config", {}),
                    dir=log_dir
                )
                self.info("Wandb日志记录已启用")
            except ImportError:
                self.warning("无法导入Wandb，已禁用Wandb日志记录")
        
        # 保存配置信息
        self.metrics_history = {}
    
    def info(self, message: str):
        """记录信息级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误级别日志"""
        self.logger.error(message)
    
    def log_scalars(self, tag: str, scalars: Dict[str, float], step: int):
        """记录标量指标"""
        # 保存到历史记录
        if tag not in self.metrics_history:
            self.metrics_history[tag] = {}
        
        for key, value in scalars.items():
            if key not in self.metrics_history[tag]:
                self.metrics_history[tag][key] = []
            self.metrics_history[tag][key].append({"step": step, "value": value})
        
        # TensorBoard
        if self.tensorboard_writer:
            for key, value in scalars.items():
                self.tensorboard_writer.add_scalar(f"{tag}/{key}", value, step)
            self.tensorboard_writer.flush()
        
        # Wandb
        if self.wandb:
            wandb_data = {f"{tag}/{key}": value for key, value in scalars.items()}
            wandb_data["step"] = step
            self.wandb.log(wandb_data)
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, step)
        
        if self.wandb:
            self.wandb.log({tag: self.wandb.Histogram(values), "step": step})
    
    def log_image(self, tag: str, image, step: int):
        """记录图像"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(tag, image, step)
        
        if self.wandb:
            self.wandb.log({tag: self.wandb.Image(image), "step": step})
    
    def log_video(self, tag: str, video, step: int, fps: int = 30):
        """记录视频"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_video(tag, video, step, fps=fps)
        
        if self.wandb:
            self.wandb.log({tag: self.wandb.Video(video, fps=fps), "step": step})
    
    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """记录超参数"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)
        
        if self.wandb:
            self.wandb.config.update(hparams)
    
    def save_metrics(self, filename: str = "metrics.json"):
        """保存指标历史到文件"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        self.info(f"指标历史已保存到: {filepath}")
    
    def load_metrics(self, filename: str = "metrics.json"):
        """从文件加载指标历史"""
        filepath = os.path.join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.metrics_history = json.load(f)
            self.info(f"指标历史已从以下路径加载: {filepath}")
        else:
            self.warning(f"指标文件不存在: {filepath}")
    
    def get_metric_history(self, tag: str, key: str) -> List[Dict[str, float]]:
        """获取特定指标的历史记录"""
        return self.metrics_history.get(tag, {}).get(key, [])
    
    def close(self):
        """关闭日志记录器"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb:
            self.wandb.finish()
        
        # 保存最终指标
        self.save_metrics()
        
        self.info("日志记录器已关闭")