# A COMPREHENSIVE SURVEY OF LLM ALIGNMENT TECHNIQUES: RLHF, RLAIF, PPO, DPO AND MORE

### Abstract 
随着自监督学习、预训练语料库、指令微调和大型Transformer的发展，LLMs能够生成符合人类期望的响应。然而，由于训练数据质量不一，生成不期望响应的问题依然存在。过去两年，提出了多种方法来改善LLMs，特别是与人类期望的对齐。

### Introduction
过去几十年中，通过自监督学习对LLMs进行预训练取得了显著进展，这得益于更大的解码器Transformer、数万亿标记的使用和多GPU计算的并行化。预训练后，指令微调被用于指导LLMs对人类查询做出响应。尽管有这些进步，但一个关键问题仍未解决：LLMs可能会生成不期望的响应。为了减轻这种风险，将LLMs与人类价值观对齐至关重要。RLHF已成为对齐LLMs的一种突破性技术，推动了GPT-4、Claude和Gemini等强大模型的发展。

![The 13 categorical directions for xPO to align an LLM with human preference](img/zhichao2024-fig1.png)

### 2 Categorical Outline

#### 2.1 Reward Model
在本节中，我们将讨论四种类型的奖励模型：
1. 显式奖励模型与隐式奖励模型：显式奖励模型通过微调预训练的LLM来直接为提示和响应分配分数，而隐式奖励模型则不需要显式训练，而是通过其他方式（如DPO中的映射）来对齐LLM。
2. 点奖励模型与偏好模型：点奖励模型为每个提示和响应分配一个分数，而偏好模型则基于用户偏好来对响应进行排序。
3. 标记级奖励模型与响应级奖励模型：标记级奖励模型为每个标记分配分数，而响应级奖励模型为整个响应分配一个分数。
4. 仅使用负面偏好训练奖励模型：这种模型只关注不期望的响应，并据此训练奖励模型。
这些不同的奖励模型在图2中有所展示。

![The four subtopics of reward model](img/zhichao2024-fig2.png)

##### 2.1.1 Explicit Reward Model vs. Implicit Reward Model
在RLHF中，研究人员收集了一个包含提示x、期望响应yw和不期望响应yl的三元组数据集。基于这个数据集，通过微调预训练的LLM，导出了显式奖励模型rϕ(x, y)，用于在RL环境中对齐LLM策略。隐式奖励模型rθ(x, y)则跳过了显式奖励模型的训练过程，例如在DPO中，通过建立最优奖励模型和最优策略之间的映射来实现对齐，无需直接推导奖励模型。

##### 2.1.2 Pointwise Reward Model vs. Preferencewise Model
在RLHF中，原始方法通过点奖励模型和Bradley-Terry模型估计期望响应优于不期望响应的概率。但这种方法不能直接获得成对偏好，且难以处理人类标注的不一致性。为此，提出了Nash学习，它直接建模期望响应优于其他响应的概率，以更好地处理这些问题。

##### 2.1.3 Response-Level Reward vs. Token-Level Reward
在RLHF和DPO中，奖励是基于Response-level的，而在马尔可夫决策过程中，奖励是在每个动作后给出的，这促使了Token-level奖励模型的引入。

##### 2.1.4 Negative Preference Optimization
在RLHF数据集中，人类既标记了期望响应也标记了不期望响应。最近，随着大型语言模型（LLM）能力的提升，一些研究人员认为LLM可以生成比人类标记者产生的高质量响应。因此，他们选择仅使用收集的数据集中的提示和不期望响应，使用LLM生成期望响应。

#### 2.2 Feedback
在本节中，我们将讨论三种类型的反馈：1. 偏好反馈与二元反馈；2. 成对反馈与列表反馈；3. 人类反馈与AI反馈。这些反馈的图表可以在图3中找到。

![The four subtopics of feedback](img/zhichao2024-fig3.png)

##### 2.2.1 Preference Feedback vs. Binary Feedback
在RLHF论文中，收集了偏好反馈，即yw > yl。然而，后续工作如KTO和DRO表明，偏好反馈更难收集，收集二元反馈（如“点赞”或“点踩”）可能更有优势。

##### 2.2.2 Pairwise Feedback vs. Listwise Feedback
在RLHF中，列表反馈被收集，即将多个响应视为成对比较。但后续研究，如LiPO，建议将列表偏好视为排序问题，而不是多个成对比较，这可能是更有优势的方法。


##### 2.2.3 Human Feedback vs. AI Feedback
RLHF通常需要人类比较多个响应，这既耗时又成本高。随着LLM的发展，现在可以使用AI来提供反馈，从而降低成本和时间。

#### 2.3  Reinforcement Learning (RL)
RL的目标是最大化响应奖励并最小化策略模型与初始参考模型之间的差异。讨论包括基于参考与无参考RL、长度控制RL、不同差异RL和在策略与离策略RL。
$$
\begin{align}
\pi^*_{\theta}(y|x) &= \max_{\pi_\theta} \mathbb{E}_{x \sim D} \left[\mathbb{E}_{y \sim \pi_\theta(y|x)} r(x, y) - \beta D_{KL} \large(\pi_\theta (y|x) || \pi_{ref} (y|x) \large) \right] \\

&=\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta (y|x)} \left[r(x,y) - \beta \log \frac{\pi_\theta (y|x)}{\pi_{ref} (y|x)} \right] \\

\end{align}
$$

##### 2.3.1 Reference-Based RL vs. Reference-Free RL
RLHF的目标是最小化当前策略和参考策略之间的距离。大多数方法集中在基于参考策略的方法上，但引入参考策略增加了内存负担。为了解决这个问题，提出了避免使用参考策略的方法，如SimPO。


##### 2.3.2  Length-Control RL
使用LLM评估时，它们倾向于偏好冗长的回答，即使没有提供额外信息。这种偏见可能会影响LLM的对齐。此外，LLM回答的冗长性可能会增加人类阅读和理解所需的时间。为了解决这个问题，后续工作如R-DPO和SimPO包含了长度控制的考虑。


##### 2.3.3 Different Divergences in RL
在RLHF中，反向KL散度用于评估策略差异，但会降低响应多样性。研究正在探索其他散度度量以改善这一问题。

##### 2.3.4 On-policy or Off-policy Learning
在RL中，on-policy学习从最新策略中采样响应，而off-policy使用较早的响应，这可能导致响应与当前策略不匹配。


#### 2.4 Optimization
在LLMs的对齐过程中，优化策略包括两种主要方法：迭代/在线偏好优化和非线性/离线偏好优化，以及分离SFT和Alignment与合并SFT和Alignment。迭代优化通过实时反馈调整模型，而非线性优化在训练后不再调整。分离SFT和Alignment先预训练后对齐，合并则同时进行，各有利弊。图4展示了这些策略的对比。

![The two subtopics of optimization](img/zhichao2024-fig4.png)

##### 2.4.1 Iterative/Online Preference Optimization vs. Non-Iterative/Offline Preference Optimization
使用收集的数据集进行对齐称为非迭代/离线优化。而人类标记新数据或LLMs同时生成和评估响应时，可实现迭代/在线优化。

##### 2.4.2 Separating SFT and Alignment vs. Merging SFT and Alignment
在RLHF中，传统方法是将SFT和alignment分开进行，这可能导致效率低下和遗忘问题。ORPO和PAFT等研究尝试通过将两者结合或同时进行微调来解决这些问题。
