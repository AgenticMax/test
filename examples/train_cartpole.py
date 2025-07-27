"""
CartPole环境训练示例
使用PPO算法训练智能体解决CartPole-v1任务
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentrl import Agent, Environment, PPOTrainer
from agentrl.utils.config import create_default_config


def main():
    """主训练函数"""
    
    # 创建配置
    config = create_default_config()
    
    # 修改配置
    config.set("env.env_id", "CartPole-v1")
    config.set("agent.algorithm", "PPO")
    config.set("agent.lr", 3e-4)
    config.set("training.max_episodes", 500)
    config.set("training.eval_freq", 50)
    config.set("logging.experiment_name", "cartpole_ppo")
    
    print("=== AgentRL CartPole训练示例 ===")
    print(f"环境: {config.get('env.env_id')}")
    print(f"算法: {config.get('agent.algorithm')}")
    print(f"最大回合数: {config.get('training.max_episodes')}")
    print(f"学习率: {config.get('agent.lr')}")
    print()
    
    # 创建环境
    env = Environment(config.get("env.env_id"))
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print()
    
    # 创建智能体
    agent_config = {
        "lr": config.get("agent.lr"),
        "gamma": config.get("agent.gamma"),
        "clip_ratio": config.get("ppo.clip_ratio"),
        "value_loss_coef": config.get("ppo.value_loss_coef"),
        "entropy_coef": config.get("ppo.entropy_coef"),
        "ppo_epochs": config.get("ppo.ppo_epochs"),
        "update_interval": config.get("ppo.update_interval"),
        "hidden_dims": config.get("agent.hidden_dims")
    }
    
    agent = Agent(
        algorithm="PPO",
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=agent_config
    )
    
    print(f"智能体创建成功，网络参数数量: {sum(p.numel() for p in agent.agent.network.parameters())}")
    print()
    
    # 创建训练器
    trainer_config = {
        "max_episodes": config.get("training.max_episodes"),
        "eval_freq": config.get("training.eval_freq"),
        "save_freq": config.get("training.save_freq"),
        "log_freq": config.get("training.log_freq"),
        "log_dir": config.get("logging.log_dir"),
        "save_dir": config.get("logging.save_dir"),
        "experiment_name": config.get("logging.experiment_name"),
        "use_tensorboard": config.get("logging.use_tensorboard"),
        "update_interval": config.get("ppo.update_interval")
    }
    
    trainer = PPOTrainer(agent, env, trainer_config)
    
    # 开始训练
    print("开始训练...")
    training_results = trainer.train(
        target_reward=195.0,  # CartPole的解决标准
        early_stopping_patience=50
    )
    
    print("\n=== 训练完成 ===")
    print(f"训练回合数: {len(training_results['episode_rewards'])}")
    print(f"最终平均奖励: {sum(training_results['episode_rewards'][-10:]) / 10:.2f}")
    print(f"最高奖励: {max(training_results['episode_rewards']):.2f}")
    
    # 评估训练好的智能体
    print("\n=== 最终评估 ===")
    eval_results = trainer.evaluate(num_episodes=20, render=False)
    print(f"评估奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"奖励范围: [{eval_results['min_reward']:.2f}, {eval_results['max_reward']:.2f}]")
    
    # 保存最终模型
    model_path = os.path.join(trainer.save_dir, f"{trainer.experiment_name}_final.pt")
    print(f"\n模型已保存到: {model_path}")
    
    # 创建推理引擎进行性能测试
    from agentrl.inference import InferenceEngine
    
    print("\n=== 推理性能测试 ===")
    inference_engine = InferenceEngine(agent, optimize=True)
    
    # 性能基准测试
    benchmark_results = inference_engine.benchmark(num_samples=1000)
    
    # 模型信息
    model_info = inference_engine.get_model_info()
    print(f"模型参数数量: {model_info['total_parameters']}")
    print(f"推理设备: {model_info['device']}")
    print(f"模型已优化: {model_info['optimized']}")
    
    print("\n训练和评估完成！")
    print("查看TensorBoard日志:")
    print(f"  tensorboard --logdir {trainer.log_dir}")


if __name__ == "__main__":
    main()