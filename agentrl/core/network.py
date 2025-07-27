"""神经网络组件"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN(nn.Module):
    """卷积神经网络"""
    
    def __init__(self,
                 input_channels: int,
                 input_height: int,
                 input_width: int,
                 conv_layers: List[Dict],
                 fc_layers: List[int],
                 output_dim: int):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # 卷积层
        conv_modules = []
        in_channels = input_channels
        
        for layer_config in conv_layers:
            conv_modules.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config.get("stride", 1),
                padding=layer_config.get("padding", 0)
            ))
            conv_modules.append(nn.ReLU())
            
            if layer_config.get("pool"):
                conv_modules.append(nn.MaxPool2d(layer_config["pool"]))
            
            in_channels = layer_config["out_channels"]
        
        self.conv_network = nn.Sequential(*conv_modules)
        
        # 计算卷积输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output = self.conv_network(dummy_input)
            self.conv_output_dim = conv_output.numel()
        
        # 全连接层
        self.fc_network = MLP(
            input_dim=self.conv_output_dim,
            hidden_dims=fc_layers,
            output_dim=output_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        conv_out = self.conv_network(x)
        conv_out = conv_out.view(batch_size, -1)
        return self.fc_network(conv_out)


class Network(nn.Module):
    """通用网络类"""
    
    def __init__(self, 
                 input_shape: Union[int, Tuple[int, ...]],
                 output_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 network_type: str = "mlp",
                 **kwargs):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network_type = network_type
        
        if network_type == "mlp":
            if isinstance(input_shape, (tuple, list)):
                input_dim = np.prod(input_shape)
            else:
                input_dim = input_shape
            
            self.network = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                **kwargs
            )
        
        elif network_type == "cnn":
            if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
                raise ValueError("CNN需要3D输入形状 (channels, height, width)")
            
            channels, height, width = input_shape
            self.network = CNN(
                input_channels=channels,
                input_height=height,
                input_width=width,
                conv_layers=kwargs.get("conv_layers", [
                    {"out_channels": 32, "kernel_size": 8, "stride": 4},
                    {"out_channels": 64, "kernel_size": 4, "stride": 2},
                    {"out_channels": 64, "kernel_size": 3, "stride": 1}
                ]),
                fc_layers=hidden_dims,
                output_dim=output_dim
            )
        
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.network_type == "mlp" and len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        return self.network(x)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 shared_network: bool = True,
                 continuous_actions: bool = False,
                 **kwargs):
        super().__init__()
        
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.shared_network = shared_network
        self.continuous_actions = continuous_actions
        
        if shared_network:
            # 共享特征提取器
            self.shared_net = Network(
                input_shape=input_shape,
                output_dim=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                **kwargs
            )
            
            # Actor头
            if continuous_actions:
                self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
                self.actor_std = nn.Linear(hidden_dims[-1], action_dim)
            else:
                self.actor = nn.Linear(hidden_dims[-1], action_dim)
            
            # Critic头
            self.critic = nn.Linear(hidden_dims[-1], 1)
        
        else:
            # 独立的Actor和Critic网络
            self.actor_net = Network(
                input_shape=input_shape,
                output_dim=action_dim if not continuous_actions else action_dim * 2,
                hidden_dims=hidden_dims,
                **kwargs
            )
            
            self.critic_net = Network(
                input_shape=input_shape,
                output_dim=1,
                hidden_dims=hidden_dims,
                **kwargs
            )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 1:  # Critic输出层
                    nn.init.orthogonal_(module.weight, gain=1)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """前向传播"""
        if self.shared_network:
            features = self.shared_net(x)
            
            if self.continuous_actions:
                action_mean = self.actor_mean(features)
                action_std = F.softplus(self.actor_std(features)) + 1e-5
                value = self.critic(features)
                return action_mean, action_std, value
            else:
                action_logits = self.actor(features)
                value = self.critic(features)
                return action_logits, value
        
        else:
            if self.continuous_actions:
                actor_out = self.actor_net(x)
                action_mean = actor_out[:, :self.action_dim]
                action_std = F.softplus(actor_out[:, self.action_dim:]) + 1e-5
                value = self.critic_net(x)
                return action_mean, action_std, value
            else:
                action_logits = self.actor_net(x)
                value = self.critic_net(x)
                return action_logits, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作和动作概率"""
        with torch.no_grad():
            if self.continuous_actions:
                action_mean, action_std, _ = self.forward(x)
                
                if deterministic:
                    action = action_mean
                    log_prob = torch.zeros_like(action_mean)
                else:
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                
                return action, log_prob
            
            else:
                action_logits, _ = self.forward(x)
                action_probs = F.softmax(action_logits, dim=-1)
                
                if deterministic:
                    action = torch.argmax(action_probs, dim=-1, keepdim=True)
                else:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().unsqueeze(-1)
                
                log_prob = torch.log(action_probs.gather(1, action) + 1e-8)
                return action, log_prob
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """获取状态值"""
        with torch.no_grad():
            if self.shared_network:
                features = self.shared_net(x)
                return self.critic(features)
            else:
                return self.critic_net(x)