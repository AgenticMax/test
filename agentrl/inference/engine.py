"""推理引擎"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
import time
import logging

class InferenceEngine:
    """高效推理引擎"""
    
    def __init__(self, 
                 agent: Any,
                 device: Optional[str] = None,
                 batch_size: int = 1,
                 optimize: bool = True):
        """
        初始化推理引擎
        
        Args:
            agent: 训练好的智能体
            device: 推理设备
            batch_size: 批量推理大小
            optimize: 是否优化模型
        """
        self.agent = agent
        self.batch_size = batch_size
        
        # 设备选择
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 设置评估模式
        self.agent.set_training_mode(False)
        
        # 模型优化
        if optimize:
            self._optimize_model()
        
        # 统计信息
        self.inference_times = []
        self.logger = logging.getLogger("InferenceEngine")
    
    def _optimize_model(self):
        """优化模型以提高推理速度"""
        try:
            # 获取网络
            if hasattr(self.agent, 'agent'):
                network = self.agent.agent.network if hasattr(self.agent.agent, 'network') else None
            elif hasattr(self.agent, 'network'):
                network = self.agent.network
            else:
                network = None
            
            if network is not None:
                # 使用TorchScript优化
                network.eval()
                
                # 创建示例输入
                if hasattr(self.agent, 'obs_shape'):
                    sample_input = torch.randn(1, *self.agent.obs_shape).to(self.device)
                elif hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'obs_shape'):
                    sample_input = torch.randn(1, *self.agent.agent.obs_shape).to(self.device)
                else:
                    # 尝试默认形状
                    sample_input = torch.randn(1, 4).to(self.device)
                
                try:
                    # 尝试转换为TorchScript
                    traced_model = torch.jit.trace(network, sample_input)
                    traced_model.eval()
                    self.optimized_network = traced_model
                    self.logger.info("模型已优化为TorchScript")
                except Exception as e:
                    self.logger.warning(f"TorchScript优化失败: {e}")
                    self.optimized_network = None
            else:
                self.optimized_network = None
                
        except Exception as e:
            self.logger.warning(f"模型优化失败: {e}")
            self.optimized_network = None
    
    def predict(self, observations: Union[np.ndarray, List[np.ndarray]], 
                deterministic: bool = True) -> Union[np.ndarray, List]:
        """
        单次推理
        
        Args:
            observations: 观测数据
            deterministic: 是否使用确定性策略
            
        Returns:
            动作
        """
        start_time = time.time()
        
        # 处理输入
        if isinstance(observations, list):
            batch_obs = np.array(observations)
        else:
            batch_obs = observations if len(observations.shape) > 1 else observations[np.newaxis]
        
        # 转换为tensor
        obs_tensor = torch.FloatTensor(batch_obs).to(self.device)
        
        with torch.no_grad():
            if self.optimized_network is not None:
                # 使用优化后的网络
                try:
                    if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'continuous_actions'):
                        continuous_actions = self.agent.agent.continuous_actions
                    else:
                        continuous_actions = False
                    
                    if continuous_actions:
                        # 连续动作空间
                        action_mean, action_std, _ = self.optimized_network(obs_tensor)
                        if deterministic:
                            actions = torch.tanh(action_mean)
                        else:
                            dist = torch.distributions.Normal(action_mean, action_std)
                            actions = torch.tanh(dist.sample())
                    else:
                        # 离散动作空间
                        action_logits, _ = self.optimized_network(obs_tensor)
                        if deterministic:
                            actions = torch.argmax(action_logits, dim=-1)
                        else:
                            action_probs = torch.softmax(action_logits, dim=-1)
                            dist = torch.distributions.Categorical(action_probs)
                            actions = dist.sample()
                    
                    actions = actions.cpu().numpy()
                except Exception as e:
                    self.logger.warning(f"优化网络推理失败，回退到原始方法: {e}")
                    actions = self._fallback_predict(obs_tensor, deterministic)
            else:
                # 使用原始智能体
                actions = self._fallback_predict(obs_tensor, deterministic)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 返回结果
        if len(batch_obs) == 1:
            return actions[0] if len(actions.shape) > 0 else actions.item()
        else:
            return actions
    
    def _fallback_predict(self, obs_tensor: torch.Tensor, deterministic: bool) -> np.ndarray:
        """回退的推理方法"""
        batch_size = obs_tensor.shape[0]
        actions = []
        
        for i in range(batch_size):
            obs = obs_tensor[i].cpu().numpy()
            action = self.agent.act(obs, training=not deterministic)
            actions.append(action)
        
        return np.array(actions)
    
    def batch_predict(self, 
                     observations: List[np.ndarray],
                     deterministic: bool = True) -> List:
        """
        批量推理
        
        Args:
            observations: 观测数据列表
            deterministic: 是否使用确定性策略
            
        Returns:
            动作列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(observations), self.batch_size):
            batch_obs = observations[i:i + self.batch_size]
            batch_actions = self.predict(batch_obs, deterministic)
            
            if isinstance(batch_actions, np.ndarray) and len(batch_actions.shape) > 0:
                results.extend(batch_actions.tolist())
            else:
                results.append(batch_actions)
        
        return results
    
    def benchmark(self, 
                 num_samples: int = 1000,
                 obs_shape: Optional[tuple] = None) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            num_samples: 测试样本数
            obs_shape: 观测形状
            
        Returns:
            性能指标
        """
        if obs_shape is None:
            if hasattr(self.agent, 'obs_shape'):
                obs_shape = self.agent.obs_shape
            elif hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'obs_shape'):
                obs_shape = self.agent.agent.obs_shape
            else:
                obs_shape = (4,)  # 默认形状
        
        # 生成随机观测
        test_observations = [np.random.randn(*obs_shape) for _ in range(num_samples)]
        
        # 清空之前的时间记录
        self.inference_times = []
        
        # 进行推理
        start_time = time.time()
        _ = self.batch_predict(test_observations, deterministic=True)
        total_time = time.time() - start_time
        
        # 计算指标
        avg_inference_time = np.mean(self.inference_times) * 1000  # 转换为毫秒
        std_inference_time = np.std(self.inference_times) * 1000
        throughput = num_samples / total_time  # 每秒推理数
        
        results = {
            "avg_inference_time_ms": avg_inference_time,
            "std_inference_time_ms": std_inference_time,
            "throughput_per_sec": throughput,
            "total_time_sec": total_time,
            "num_samples": num_samples
        }
        
        self.logger.info(f"推理性能基准测试结果:")
        self.logger.info(f"  平均推理时间: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        self.logger.info(f"  吞吐量: {throughput:.2f} samples/sec")
        self.logger.info(f"  总时间: {total_time:.2f} sec")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "device": str(self.device),
            "batch_size": self.batch_size,
            "optimized": self.optimized_network is not None,
        }
        
        # 模型参数数量
        total_params = 0
        if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'named_parameters'):
            for name, param in self.agent.agent.named_parameters():
                total_params += param.numel()
        elif hasattr(self.agent, 'named_parameters'):
            for name, param in self.agent.named_parameters():
                total_params += param.numel()
        
        info["total_parameters"] = total_params
        
        # 内存使用
        if torch.cuda.is_available() and self.device.type == 'cuda':
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated(self.device)
            info["gpu_memory_cached"] = torch.cuda.memory_reserved(self.device)
        
        return info
    
    def save_optimized_model(self, filepath: str):
        """保存优化后的模型"""
        if self.optimized_network is not None:
            torch.jit.save(self.optimized_network, filepath)
            self.logger.info(f"优化模型已保存到: {filepath}")
        else:
            self.logger.warning("没有优化模型可保存")
    
    def load_optimized_model(self, filepath: str):
        """加载优化后的模型"""
        try:
            self.optimized_network = torch.jit.load(filepath, map_location=self.device)
            self.optimized_network.eval()
            self.logger.info(f"优化模型已从以下路径加载: {filepath}")
        except Exception as e:
            self.logger.error(f"加载优化模型失败: {e}")
            self.optimized_network = None