# 冰湖强化学习项目

## 概述

本项目探讨了多种强化学习（RL）算法，特别是价值迭代、策略迭代、Q-Learning 和 SARSA，用于解决 OpenAI Gym 的“冰湖”环境。冰湖是一个网格世界导航任务，代理需要从起始点到达目标，同时避免掉入洞中。项目的目标是评估和比较不同 RL 方法在奖励和步数上的有效性。

## 实现的算法

1. **价值迭代**：一种经典的 RL 算法，通过迭代更新状态值来计算最优策略。
2. **策略迭代**：通过评估一个策略并逐步改进，直到收敛。
3. **Q-Learning**：一种无模型的算法，通过学习在状态中采取的动作的价值，使代理能够根据采样的经验改进其策略。
4. **SARSA**：与 Q-Learning 类似，但使用实际采取的动作更新 Q 值，这可能导致不同的收敛特性。

## 环境要求

建议使用 pip 安装所需的包：

```bash
conda create frozenlake python=3.10 -y
conda activate frozenlake
pip install numpy pandas gymnasium[toy_text] seaborn matplotlib tqdm
```

## 如何运行项目

1. 克隆代码库：

   ```bash
   git clone https://github.com/BrowserOrientedProgramer/FrozenLake-experiment.git
   cd FrozenLake-experment
   ```

2. 运行主脚本：

   ```bash
   # mkdir img # 可更改save_path将结果保存到指定位置
   python main.py
   ```

这将对不同大小的随机地图执行强化学习算法，并将学习到的策略和性能指标的可视化结果保存在 `img` 目录中。

## 结果

项目生成了各种图表，展示了：

- 每种方法学习到的 Q 值和策略。
- Q-Learning 和 SARSA 的累积奖励和每集平均步数。

这些结果有助于理解每种算法在不同场景中的收敛情况。

## 可视化

项目包括以下可视化内容：

1. **Q 值和策略**：热图显示学习到的 Q 值以及相应的动作，以箭头的形式在冰湖上表示。
2. **性能指标**：折线图展示 Q-Learning 和 SARSA 在每集中的累积奖励和平均步数。

## 结论

本项目作为经典 RL 算法的实现指南，展示了它们在导航冰湖环境中的表现。不同方法之间的比较提供了对它们在解决强化学习任务中的优缺点的洞察。

## 参考资料

课程的B站链接：[【强化学习的数学原理】课程：从零开始到透彻理解（完结）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1sd4y167NS)

课程的Github项目：[MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning: This is the homepage of a new book entitled "Mathematical Foundations of Reinforcement Learning." (github.com)](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
