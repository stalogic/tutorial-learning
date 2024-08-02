

## 2023-Yichen Lu-RL-CCD
### Abstract
* 背景: 
  * 并发时钟和数据（CCD）优化是现代商业工具中一种被广泛采用的方法，它使用时钟倾斜和延迟固定策略的混合物来解决定时违规。
  * Concurrent Clock and Data (CCD) optimization is a well adopted approach in modern commercial tools that resolves timing violations using a mixture of clock skewing and delay fixing strategies.
* 问题：
  * 然而，现有的CCD算法存在缺陷。特别是，它们没有正确地对不同优化策略的违反端点进行优先排序，导致了全局次优结果。 
  * However, existing CCD algorithms are flawed. Particularly, they fail to prioritize violating endpoints for different optimization strategies
correctly, leading to flow-wise globally sub-optimal results.
* 目标：
  * 我们通过介绍RL-CCD来克服这个问题，这是一种强化学习（RL）代理，它选择端点进行有用的倾斜优先级
  * we overcome this issue by presenting RL-CCD, a Reinforcement Learning (RL) agent that selects endpoints for useful skew prioritization
* 途径：
  * 利用所提出的EP-GNN，一个面向端点的图神经网络（GNN）模型，以及一个基于变压器的自监督注意机制。
  * using the proposed EP-GNN, an endpoint-oriented Graph Neural Network (GNN) model, and a Transformer-based self-supervised attention mechanism.
* 结果：
  * 在5−12nm技术的19种工业设计上的实验结果表明，与商业工具相比，RL-CCD实现了高达64%的总负松弛（TNS）减少和66.5%的违反端点（NVE）改进。
  * Experimental results on 19 industrial designs in 5−12nm technologies
demonstrate that RL-CCD achieves up to 64% Total Negative Slack
(TNS) reduction and 66.5% number of violating endpoints (NVE)
improvement over the native implementation of a commercial tool.

---
    "Clock skewing" 和 "delay (logic) fixing" 是数字电路设计中使用的术语，它们与电路的时序特性有关。
    
    1. Clock Skewing (时钟偏斜):
    Clock skew 是指在数字电路中，不同部分接收到时钟信号的时间差异。在一个同步数字系统中，理想情况下所有触发器都应该在同一个时钟边沿触发，但由于信号传播延迟的不同，实际上时钟信号到达各个触发器的时间会有细微的差别，这就是时钟偏斜。
    时钟偏斜可以是自然的，也可以是故意引入的。时钟偏斜的调整，即 clock skewing，是一种技术，用于通过改变时钟路径的延迟来调整时钟信号到达各个触发器的时间，目的是为了减少时钟偏斜的影响，提高电路的同步性和性能。正面的时钟偏斜可以用来平衡不同的信号路径，确保数据在正确的时钟周期内稳定。
    2. Delay (Logic) Fixing (延迟(逻辑)修复):
    Delay fixing 是指在数字电路设计中，对逻辑路径的延迟进行调整，以确保所有的信号都能在正确的时钟周期内到达目的地。这通常涉及到以下几种操作：
    - 增加延迟（Inserting Delays）：如果某个信号路径比预期更快，可能会在该路径上插入额外的延迟元件（如缓冲器、延迟线等），以使其与其他路径同步。
    - 减少延迟（Reducing Delays）：如果某个信号路径比预期更慢，可能需要优化该路径的设计，减少延迟，以便信号能够及时到达。
    逻辑延迟修复的目的是为了满足电路的时序要求，确保数据在时钟周期内的稳定性和可靠性，避免时序违例（timing violations）的发生，从而提高电路的整体性能和可靠性。这通常涉及到使用时序分析工具来识别和修复时序问题。
---
#### 执行流程
在每个RL时间步（训练迭代）中，智能体选择一个端点，并根据重叠计算屏蔽其他端点。当所有违规端点被屏蔽或选中后，选择过程完成。接着，对RL选中的端点应用时序余量以设计最差负slack（WNS），为后续的时钟偏斜优化做准备。
![RL-CCD 运行流程](img/rl_ccd_fig_2.png)
在剩余的布局优化步骤之前，移除应用的余量，并使用缓冲、调整尺寸、逻辑重构和合法化等技术进行优化。最后，实现的TNS值作为当前轨迹的RL奖励，用于通过名为REINFORCE的政策梯度算法更新框架参数。

