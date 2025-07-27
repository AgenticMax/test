# AgentRL 框架文档中心

欢迎来到 AgentRL 强化学习框架的文档中心！这里提供了完整的使用指南、API 参考和开发文档。

## 📚 文档导航

### 🚀 快速开始
- [README.md](../README.md) - 项目介绍和快速开始
- [FRAMEWORK_SUMMARY.md](../FRAMEWORK_SUMMARY.md) - 框架功能总结

### 🏗️ 架构与设计
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 详细的系统架构文档
  - 系统架构总览
  - 核心模块详解
  - 数据流架构
  - 性能优化架构
  - 扩展机制

### 📖 API 参考
- [API_REFERENCE.md](./API_REFERENCE.md) - 完整的 API 参考文档
  - 核心类 API
  - 工具类 API
  - 算法特定配置
  - 环境配置选项
  - 监控和日志配置

### 🛠️ 开发指南
- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - 开发者指南
  - 开发环境设置
  - 框架扩展指南
  - 测试指南
  - 代码规范
  - 性能优化
  - 持续集成

## 🎯 按用途分类

### 👥 用户文档

#### 初学者
1. **[项目介绍](../README.md)** - 了解 AgentRL 是什么
2. **[快速开始](../README.md#-快速开始)** - 5分钟上手指南
3. **[基础示例](../examples/)** - 从简单例子开始学习

#### 进阶用户
1. **[系统架构](./ARCHITECTURE.md)** - 深入理解框架设计
2. **[API 参考](./API_REFERENCE.md)** - 详细的接口文档
3. **[配置指南](./API_REFERENCE.md#-算法特定配置)** - 参数调优指南

#### 专业用户
1. **[性能优化](./ARCHITECTURE.md#-性能优化架构)** - 提升训练和推理效率
2. **[扩展机制](./ARCHITECTURE.md#-扩展机制)** - 添加自定义功能
3. **[部署指南](./API_REFERENCE.md#-inferenceengine-类)** - 生产环境部署

### 🔧 开发者文档

#### 贡献者
1. **[开发环境](./DEVELOPER_GUIDE.md#-开发环境设置)** - 搭建开发环境
2. **[代码规范](./DEVELOPER_GUIDE.md#-代码规范)** - 编码标准和最佳实践
3. **[测试指南](./DEVELOPER_GUIDE.md#-测试指南)** - 编写和运行测试

#### 扩展开发
1. **[添加新算法](./DEVELOPER_GUIDE.md#1-添加新算法)** - 实现自定义算法
2. **[添加新网络](./DEVELOPER_GUIDE.md#2-添加新网络架构)** - 创建自定义网络架构
3. **[环境扩展](./DEVELOPER_GUIDE.md#3-添加新环境支持)** - 支持新的强化学习环境

## 🔍 按主题分类

### 算法相关
- [支持的算法](../FRAMEWORK_SUMMARY.md#-支持的算法)
- [PPO 配置](./API_REFERENCE.md#ppo-配置参数)
- [DQN 配置](./API_REFERENCE.md#dqn-配置参数)
- [SAC 配置](./API_REFERENCE.md#sac-配置参数)
- [A3C 配置](./API_REFERENCE.md#a3c-配置参数)

### 训练相关
- [训练器架构](./ARCHITECTURE.md#3-训练层-training-layer)
- [训练流程](./ARCHITECTURE.md#32-训练流程设计)
- [回调系统](./API_REFERENCE.md#-回调系统)
- [早停机制](./API_REFERENCE.md#basetrainer)

### 推理相关
- [推理引擎](./ARCHITECTURE.md#4-推理层-inference-layer)
- [性能优化](./ARCHITECTURE.md#42-优化策略)
- [批量推理](./API_REFERENCE.md#predict)
- [模型部署](./API_REFERENCE.md#-inferenceengine-类)

### 环境相关
- [环境系统](./ARCHITECTURE.md#14-环境系统-agentrlcoreenvironmentpy)
- [向量化环境](./ARCHITECTURE.md#环境管理架构)
- [环境配置](./API_REFERENCE.md#-环境配置选项)
- [预处理管道](./API_REFERENCE.md#基础环境配置)

### 监控相关
- [日志系统](./ARCHITECTURE.md#52-日志系统架构)
- [TensorBoard](./API_REFERENCE.md#tensorboard-配置)
- [Wandb 集成](./API_REFERENCE.md#wandb-配置)
- [性能监控](./ARCHITECTURE.md#1-性能监控)

## 📊 示例代码库

### 基础示例
```python
# 最简单的使用示例
from agentrl import Agent, Environment, PPOTrainer

env = Environment("CartPole-v1")
agent = Agent("PPO", env.observation_space, env.action_space)
trainer = PPOTrainer(agent, env)
trainer.train(episodes=1000)
```

### 完整示例
- [CartPole 训练](../examples/train_cartpole.py) - PPO 算法完整训练流程
- [DQN 训练](../examples/train_dqn.py) - DQN 算法训练示例

### 高级示例
```python
# 自定义配置的完整示例
from agentrl import Agent, Environment, PPOTrainer
from agentrl.utils.config import create_default_config

# 创建自定义配置
config = create_default_config()
config.set("agent.lr", 1e-4)
config.set("training.max_episodes", 2000)
config.set("logging.use_wandb", True)

# 创建组件
env = Environment("CartPole-v1", {"normalize_obs": True})
agent = Agent("PPO", env.observation_space, env.action_space, config.get("agent"))
trainer = PPOTrainer(agent, env, config.get("training"))

# 训练和评估
history = trainer.train()
results = trainer.evaluate(episodes=100)
```

## 🔧 常用操作指南

### 安装和设置
```bash
# 安装框架
pip install -r requirements.txt

# 运行示例
python examples/train_cartpole.py

# 启动 TensorBoard
tensorboard --logdir ./logs
```

### 配置管理
```python
# 加载配置文件
config = Config("config.yaml")

# 动态修改配置
config.set("agent.lr", 3e-4)
config.set("training.max_episodes", 1000)

# 保存配置
config.save_to_file("my_config.json")
```

### 模型管理
```python
# 保存模型
agent.save("models/my_model.pt")

# 加载模型
agent.load("models/my_model.pt")

# 模型推理
engine = InferenceEngine(agent)
actions = engine.predict(observations)
```

## 📋 FAQ 常见问题

### Q: 如何选择合适的算法？
A: 参考[算法特性对比表](./ARCHITECTURE.md#22-算法特性对比)，根据您的具体需求选择：
- **PPO**: 稳定性高，适合大多数场景
- **DQN**: 样本效率高，适合离散动作空间
- **SAC**: 适合连续控制任务
- **A3C**: 适合需要高并行度的场景

### Q: 如何提高训练效率？
A: 参考[性能优化指南](./ARCHITECTURE.md#-性能优化架构)：
- 使用向量化环境
- 启用GPU加速
- 调整批量大小
- 使用混合精度训练

### Q: 如何添加自定义算法？
A: 参考[添加新算法指南](./DEVELOPER_GUIDE.md#1-添加新算法)，按步骤实现：
1. 创建算法类
2. 注册到工厂
3. 添加默认配置
4. 编写测试

### Q: 如何进行超参数调优？
A: 建议使用以下策略：
- 从默认配置开始
- 使用网格搜索或贝叶斯优化
- 监控训练曲线
- 参考论文中的推荐值

## 🤝 社区和支持

### 获取帮助
- **GitHub Issues**: 报告bug和请求功能
- **Discussions**: 技术讨论和经验分享
- **Wiki**: 社区维护的知识库

### 贡献方式
1. **代码贡献**: 提交 Pull Request
2. **文档改进**: 修正错误，增加示例
3. **bug 报告**: 提交详细的问题描述
4. **功能建议**: 提出新功能需求

### 开发路线图
- [ ] 支持更多强化学习算法
- [ ] 集成分布式训练
- [ ] 添加模型量化支持
- [ ] 开发可视化界面
- [ ] 支持更多部署平台

---

**最后更新**: 2024年1月

**文档版本**: v0.1.0

如有任何问题或建议，欢迎通过 GitHub Issues 联系我们！