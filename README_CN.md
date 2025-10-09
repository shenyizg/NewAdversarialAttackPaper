# Latest Adversarial Attack Papers
**update at 2025-10-09 08:24:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. When Should Selfish Miners Double-Spend?**

自私的矿工何时应该加倍花钱？ cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2501.03227v2) [paper-pdf](http://arxiv.org/pdf/2501.03227v2)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.

摘要: 传统的双重支出攻击模型忽略了孤儿区块带来的收入损失。另一方面，自私的采矿文献通常忽视攻击者在每个攻击周期中免费重复支出的机会。本文中，我们对攻击进行了严格的随机分析，其中对手的目标是在自私地挖掘的同时进行双重支出。为此，我们首先结合顽固和自私的采矿攻击，即构建一个策略，让攻击者表现得顽固，直到其私人分支达到一定长度，然后转向自私。我们为每个参数制度提供最佳的确定性。接下来，我们提供了仍然比诚实采矿更有利可图的最大顽固度，并论证了顽固度水平与$k$-确认规则之间的联系。我们表明，在每个攻击周期中，如果顽固程度高于$k$，对手就可以获得双重支出的免费机会。在每个周期中，对于给定的顽固度水平，我们严格制定双重消费的可能性有多大。我们进一步修改顽固政权中的攻击，以隐藏攻击并增加双重消费的概率。



## **2. Sparse Representations Improve Adversarial Robustness of Neural Network Classifiers**

稀疏表示提高神经网络分类器的对抗鲁棒性 cs.LG

Killian Steunou is the main contributor and corresponding author of  this work

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2509.21130v2) [paper-pdf](http://arxiv.org/pdf/2509.21130v2)

**Authors**: Killian Steunou, Théo Druilhe, Sigurd Saue

**Abstract**: Deep neural networks perform remarkably well on image classification tasks but remain vulnerable to carefully crafted adversarial perturbations. This work revisits linear dimensionality reduction as a simple, data-adapted defense. We empirically compare standard Principal Component Analysis (PCA) with its sparse variant (SPCA) as front-end feature extractors for downstream classifiers, and we complement these experiments with a theoretical analysis. On the theory side, we derive exact robustness certificates for linear heads applied to SPCA features: for both $\ell_\infty$ and $\ell_2$ threat models (binary and multiclass), the certified radius grows as the dual norms of $W^\top u$ shrink, where $W$ is the projection and $u$ the head weights. We further show that for general (non-linear) heads, sparsity reduces operator-norm bounds through a Lipschitz composition argument, predicting lower input sensitivity. Empirically, with a small non-linear network after the projection, SPCA consistently degrades more gracefully than PCA under strong white-box and black-box attacks while maintaining competitive clean accuracy. Taken together, the theory identifies the mechanism (sparser projections reduce adversarial leverage) and the experiments verify that this benefit persists beyond the linear setting. Our code is available at https://github.com/killian31/SPCARobustness.

摘要: 深度神经网络在图像分类任务中表现出色，但仍然容易受到精心设计的对抗性扰动的影响。这项工作重新审视了线性降维作为一种简单的、适应数据的防御。我们根据经验比较了标准主成分分析（PCA）与其稀疏变体（SPCA）作为下游分类器的前端特征提取器，并通过理论分析补充这些实验。在理论方面，我们为应用于SPCA特征的线性头部推导出精确的鲁棒性证书：对于$\ell_\infty$和$\ell_2 $威胁模型（二元和多类），认证半径随着$W &\top u$的双重规范缩小而增加，其中$W$是投影，$u$是头部重量。我们进一步表明，对于一般（非线性）头部，稀疏性通过Lipschitz合成论点减少了操作符规范界限，从而预测了较低的输入敏感性。从经验上看，投影后存在一个小型非线性网络，在强白盒和黑匣子攻击下，SPCA始终比PCA降级得更优雅，同时保持有竞争力的干净准确性。总而言之，该理论确定了机制（稀疏的预测减少了对抗杠杆），实验验证了这种好处在线性环境之外仍然存在。我们的代码可在https://github.com/killian31/SPCARobustness上获取。



## **3. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

更安全：通过高效的前前推理推进安全一致 cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严峻的安全挑战。现有的对齐方法通常难以覆盖不同的安全场景，并且仍然容易受到对抗攻击。在这项工作中，我们提出了SAGER，这是一个通过eFficient Ex-Ante Reasoning进行安全调整的框架。我们的方法通过初始评估、规则验证和路径校准来实例化结构化的Ex-Ante推理，并嵌入预定义的安全规则以提供透明且可验证的安全判断。具体来说，我们的方法由两个训练阶段组成：（1）使用合成轨迹进行监督微调，以教授多阶段Ex-Ante推理，以及（2）分步推理偏好优化，以共同增强安全性、实用性和效率。对多个开源LLM的实验表明，SAGER显着增强了安全性能，同时保持了帮助性和响应效率。



## **4. DP-SNP-TIHMM: Differentially Private, Time-Inhomogeneous Hidden Markov Models for Synthesizing Genome-Wide Association Datasets**

DP-SNP-TIHM：用于合成全基因组关联数据集的差异私密、时间不均匀隐马尔科夫模型 cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05777v1) [paper-pdf](http://arxiv.org/pdf/2510.05777v1)

**Authors**: Shadi Rahimian, Mario Fritz

**Abstract**: Single nucleotide polymorphism (SNP) datasets are fundamental to genetic studies but pose significant privacy risks when shared. The correlation of SNPs with each other makes strong adversarial attacks such as masked-value reconstruction, kin, and membership inference attacks possible. Existing privacy-preserving approaches either apply differential privacy to statistical summaries of these datasets or offer complex methods that require post-processing and the usage of a publicly available dataset to suppress or selectively share SNPs.   In this study, we introduce an innovative framework for generating synthetic SNP sequence datasets using samples derived from time-inhomogeneous hidden Markov models (TIHMMs). To preserve the privacy of the training data, we ensure that each SNP sequence contributes only a bounded influence during training, enabling strong differential privacy guarantees. Crucially, by operating on full SNP sequences and bounding their gradient contributions, our method directly addresses the privacy risks introduced by their inherent correlations.   Through experiments conducted on the real-world 1000 Genomes dataset, we demonstrate the efficacy of our method using privacy budgets of $\varepsilon \in [1, 10]$ at $\delta=10^{-4}$. Notably, by allowing the transition models of the HMM to be dependent on the location in the sequence, we significantly enhance performance, enabling the synthetic datasets to closely replicate the statistical properties of non-private datasets. This framework facilitates the private sharing of genomic data while offering researchers exceptional flexibility and utility.

摘要: 单核苷酸多态性（SNP）数据集是遗传研究的基础，但在共享时会带来重大的隐私风险。SNP之间的相关性使得强大的对抗性攻击，如掩蔽值重建、亲属和成员推断攻击成为可能。现有的隐私保护方法要么将差异隐私应用于这些数据集的统计摘要，要么提供复杂的方法，这些方法需要后处理和使用公开可用的数据集来抑制或选择性地共享SNP。   在这项研究中，我们引入了一个创新的框架，用于生成合成SNP序列数据集，使用来自时间非齐次隐马尔可夫模型（TIHALGOT）的样本。为了保护训练数据的隐私，我们确保每个SNP序列在训练期间仅贡献有限的影响，从而实现强大的差异隐私保证。至关重要的是，通过对完整的SNP序列进行操作并限制其梯度贡献，我们的方法直接解决了其固有相关性带来的隐私风险。   通过在现实世界的1000个基因组数据集上进行的实验，我们使用隐私预算$\varepattack\in [1，10]$ at $\delta=10 '''' s来证明我们的方法的有效性。值得注意的是，通过允许Markov的过渡模型依赖于序列中的位置，我们显着增强了性能，使合成数据集能够紧密复制非私有数据集的统计属性。该框架促进了基因组数据的私人共享，同时为研究人员提供了卓越的灵活性和实用性。



## **5. Evidence of Cognitive Biases in Capture-the-Flag Cybersecurity Competitions**

捕获旗帜网络安全竞赛中认知偏见的证据 cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05771v1) [paper-pdf](http://arxiv.org/pdf/2510.05771v1)

**Authors**: Carolina Carreira, Anu Aggarwal, Alejandro Cuevas, Maria José Ferreira, Hanan Hibshi, Cleotilde Gonzalez

**Abstract**: Understanding how cognitive biases influence adversarial decision-making is essential for developing effective cyber defenses. Capture-the-Flag (CTF) competitions provide an ecologically valid testbed to study attacker behavior at scale, simulating real-world intrusion scenarios under pressure. We analyze over 500,000 submission logs from picoCTF, a large educational CTF platform, to identify behavioral signatures of cognitive biases with defensive implications. Focusing on availability bias and the sunk cost fallacy, we employ a mixed-methods approach combining qualitative coding, descriptive statistics, and generalized linear modeling. Our findings show that participants often submitted flags with correct content but incorrect formatting (availability bias), and persisted in attempting challenges despite repeated failures and declining success probabilities (sunk cost fallacy). These patterns reveal that biases naturally shape attacker behavior in adversarial contexts. Building on these insights, we outline a framework for bias-informed adaptive defenses that anticipate, rather than simply react to, adversarial actions.

摘要: 了解认知偏见如何影响对抗决策对于开发有效的网络防御至关重要。捕获旗帜（CTF）比赛提供了一个生态有效的测试平台，可以大规模研究攻击者的行为，模拟压力下的现实世界入侵场景。我们分析了来自大型教育CTF平台picoCTF的500，000多个提交日志，以识别具有防御意义的认知偏见的行为特征。我们专注于可用性偏差和沉没成本谬误，采用结合定性编码、描述性统计和广义线性建模的混合方法。我们的研究结果表明，参与者经常提交内容正确但格式不正确的标志（可用性偏差），并且尽管一再失败和成功概率下降（沉没成本谬误），但仍坚持尝试挑战。这些模式表明，偏见自然地塑造了敌对背景下的攻击者行为。在这些见解的基础上，我们概述了一个基于偏见的适应性防御框架，该框架可以预测而不是简单地对对抗行为做出反应。



## **6. Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks**

无处不在的捷径：探索多触发后门攻击 cs.LG

13 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2401.15295v4) [paper-pdf](http://arxiv.org/pdf/2401.15295v4)

**Authors**: Yige Li, Jiabo He, Hanxun Huang, Jun Sun, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). Although numerous methods for detecting and mitigating backdoor attacks have been proposed, most rely on identifying and eliminating the ``shortcut" created by the backdoor, which links a specific source class to a target class. However, these approaches can be easily circumvented by designing multiple backdoor triggers that create shortcuts everywhere and therefore nowhere specific. In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks including \textit{parallel}, \textit{sequential}, and \textit{hybrid} attacks, we demonstrate that 1) multiple triggers can coexist, overwrite, or cross-activate one another, and 2) MTBAs easily break the prevalent shortcut assumption underlying most existing backdoor detection/removal methods, rendering them ineffective. Given the security risk posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to facilitate future research on detecting and mitigating these attacks, and we also discuss potential defense strategies against MTBAs. Our code is available at https://github.com/bboylyg/Multi-Trigger-Backdoor-Attacks.

摘要: 后门攻击已成为深度神经网络（DNN）预训练和部署的重大威胁。尽管已经提出了许多检测和减轻后门攻击的方法，但大多数方法依赖于识别和消除后门创建的“快捷方式”，该“快捷方式”将特定的源类链接到目标类。然而，通过设计多个后门触发器可以轻松规避这些方法，这些触发器可以在任何地方创建快捷方式，因此没有具体的地方。在这项研究中，我们探讨了多触发后门攻击（MTBA）的概念，即多个对手利用不同类型的触发器来毒害同一数据集。通过提出和研究三种类型的多触发器攻击，包括\textit{parallel}、\textit{serial}和\textit{hybrid}攻击，我们证明了1）多个触发器可以共存、覆盖或交叉激活彼此，2）MTA很容易打破大多数现有后门检测/删除方法背后的普遍捷径假设，使其无效。鉴于MTBA构成的安全风险，我们创建了一个多触发后门中毒数据集，以促进未来检测和减轻这些攻击的研究，我们还讨论了针对MTBA的潜在防御策略。我们的代码可在https://github.com/bboylyg/Multi-Trigger-Backdoor-Attacks上获取。



## **7. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

通过弯曲和局部固有维度的几何引导对抗提示检测 cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.

摘要: 对抗性提示能够越狱前沿大型语言模型（LLM）并引发不良行为，对其安全部署构成重大障碍。当前的缓解策略主要依赖于激活内置防御机制或微调LLM，这两者计算成本很高，并且可能会牺牲模型效用。相比之下，基于检测的方法对于在现实世界应用程序中部署更有效和实用。然而，对抗性提示和良性提示之间的根本区别仍然知之甚少。在这项工作中，我们引入了CurvaLID，这是一种新型防御框架，可以通过利用其几何属性来有效检测对抗提示。它与LLM类型无关，提供跨不同对抗提示和LLM架构的统一检测框架。CurvaLID基于文本提示的几何分析，以揭示其潜在差异。从理论上讲，我们通过Whewell方程将弯曲的概念扩展到$n维单词嵌入空间，使我们能够量化局部几何属性，包括底层流中的语义移动和弯曲。为了进一步增强我们的解决方案，我们利用局部本质模糊性（LID）来捕获对抗子空间中文本提示的补充几何特征。我们的研究结果表明，对抗性提示表现出与良性提示不同的几何特征，使CurvaLID能够实现近乎完美的分类，并在对抗性提示检测方面优于最先进的检测器。CurvaLID作为一种模型不可知的方法，可在多个LLM和攻击系列中推广，提供可靠且高效的防范恶意查询的保护措施。



## **8. Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms**

对平衡系统的稳健性进行基准测试以应对不利引起的伤害 cs.LG

54 Pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2508.16481v2) [paper-pdf](http://arxiv.org/pdf/2508.16481v2)

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic

**Abstract**: Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at github.com/JNoether/BAD-ACTS

摘要: 确保代理系统的安全使用需要彻底了解这些系统在受到攻击时可能表现出的恶意行为范围。在本文中，我们评估了基于LLM的代理系统针对旨在引发代理有害行为的攻击的稳健性。为此，我们提出了一种新型的代理系统危害分类法和一种新型基准BAD-SYS，用于研究代理系统针对广泛有害行为的安全性。BAD-SYS由不同应用环境中的4个代理系统实现以及包含188个有害行为高质量示例的数据集组成。这使得能够对各种有害行为、可用工具和代理间通信结构的代理系统的稳健性进行全面研究。使用这个基准测试，我们分析的鲁棒性的代理系统对攻击者，控制系统中的代理之一，目的是操纵其他代理执行有害的目标行动。我们的研究结果表明，攻击具有很高的成功率，表明即使是系统中的单个对抗代理也会对安全性产生重大影响。即使代理使用简单的基于预算的防御策略，此攻击仍然有效。然而，我们还提出了一种基于消息监控的更有效的防御。我们相信该基准为代理系统的安全研究提供了多样化的测试平台。基准测试可以在github.com/JNoether/BAD-ACTS上找到



## **9. Adversarial Reinforcement Learning for Large Language Model Agent Safety**

用于大语言模型代理安全的对抗强化学习 cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05442v1) [paper-pdf](http://arxiv.org/pdf/2510.05442v1)

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.

摘要: 大型语言模型（LLM）代理可以利用Google Search等工具来完成复杂的任务。然而，这种工具的使用引入了间接提示注入的风险，其中隐藏在工具输出中的恶意指令可以操纵代理，从而带来数据泄露等安全风险。当前的防御策略通常依赖于对已知攻击数据集进行微调LLM代理。然而，这些数据集的生成依赖于手动设计的攻击模式，这限制了它们的多样性，并使代理容易受到新型提示注入的影响。为了解决这一局限性，我们提出了针对代理安全的对抗强化学习（ARLAS），这是一个新颖的框架，通过将问题表述为两人零和游戏来利用对抗强化学习（RL）。ARLAS联合培训了两名LLM：攻击者学会自主生成各种提示注入，而代理则学会在完成分配的任务的同时防御它们。为了确保针对广泛攻击的鲁棒性并防止循环学习，我们采用了基于群体的学习框架，该框架训练代理抵御所有之前的攻击者检查点。在BrowserGym和AgentDojo上进行评估，使用ARLAS微调的代理比原始模型实现了显着降低的攻击成功率，同时也提高了任务成功率。我们的分析进一步证实，对抗过程会产生一系列多样化且具有挑战性的攻击，从而导致与基本模型相比更强大的代理。



## **10. Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail**

通过峰值神经网络梯度稀疏度追踪实现准确性与稳健性权衡 cs.NE

Work under peer-review

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2509.23762v2) [paper-pdf](http://arxiv.org/pdf/2509.23762v2)

**Authors**: Nhan T. Luu

**Abstract**: Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, particularly for vision-related tasks, remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks.

摘要: 尖峰神经网络（SNN）引起了计算神经科学和人工智能日益增长的兴趣，主要是由于其固有的能源效率和紧凑的内存占用。然而，在SNN中实现对抗稳健性，特别是对于视觉相关任务，仍然是一个新生且未充分探索的挑战。最近的研究提出利用稀疏梯度作为一种正规化形式，以增强针对对抗性扰动的鲁棒性。在这项工作中，我们提出了一个令人惊讶的发现：在特定的架构配置下，SNN表现出自然的梯度稀疏性，并且可以在不需要任何显式正规化的情况下实现最先进的对抗防御性能。进一步的分析揭示了鲁棒性和概括性之间的权衡：虽然稀疏梯度有助于提高对抗弹性，但它们可能会损害模型的概括能力;相反，更密集的梯度支持更好的概括性，但会增加对攻击的脆弱性。



## **11. RegMix: Adversarial Mutual and Generalization Regularization for Enhancing DNN Robustness**

RegMix：对抗性相互和广义正规化以增强DNN稳健性 cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05317v1) [paper-pdf](http://arxiv.org/pdf/2510.05317v1)

**Authors**: Zhenyu Liu, Varun Ojha

**Abstract**: Adversarial training is the most effective defense against adversarial attacks. The effectiveness of the adversarial attacks has been on the design of its loss function and regularization term. The most widely used loss function in adversarial training is cross-entropy and mean squared error (MSE) as its regularization objective. However, MSE enforces overly uniform optimization between two output distributions during training, which limits its robustness in adversarial training scenarios. To address this issue, we revisit the idea of mutual learning (originally designed for knowledge distillation) and propose two novel regularization strategies tailored for adversarial training: (i) weighted adversarial mutual regularization and (ii) adversarial generalization regularization. In the former, we formulate a decomposed adversarial mutual Kullback-Leibler divergence (KL-divergence) loss, which allows flexible control over the optimization process by assigning unequal weights to the main and auxiliary objectives. In the latter, we introduce an additional clean target distribution into the adversarial training objective, improving generalization and enhancing model robustness. Extensive experiments demonstrate that our proposed methods significantly improve adversarial robustness compared to existing regularization-based approaches.

摘要: 对抗训练是对抗攻击最有效的防御。对抗性攻击的有效性在于其损失函数和正规化项的设计。对抗性训练中最广泛使用的损失函数是作为其正规化目标的交叉熵和均方误差（SSE）。然而，SSE在训练期间在两个输出分布之间强制执行过于均匀的优化，这限制了其在对抗训练场景中的鲁棒性。为了解决这个问题，我们重新审视了相互学习的想法（最初是为了知识蒸馏而设计的），并提出了两种专为对抗性训练量身定制的新型正规化策略：（i）加权对抗性相互正规化和（ii）对抗性概括正规化。在前者中，我们制定了分解的对抗相互Kullback-Leibler分歧（KL-分歧）损失，它允许通过为主要目标和辅助目标分配不相等的权重来灵活控制优化过程。在后者中，我们在对抗训练目标中引入了额外的干净目标分布，从而提高了概括性并增强了模型稳健性。大量实验表明，与现有的基于正规化的方法相比，我们提出的方法显着提高了对抗鲁棒性。



## **12. Proactive defense against LLM Jailbreak**

针对LLM越狱的积极防御 cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05052v1) [paper-pdf](http://arxiv.org/pdf/2510.05052v1)

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.

摘要: 强大的大型语言模型（LLM）的激增需要强大的安全对齐，但这些模型仍然容易受到不断发展的对抗攻击，包括迭代搜索成功查询的多回合越狱。当前的防御措施主要是反应性和静态的，通常无法抵御这些基于搜索的攻击。在本文中，我们介绍了ProAct，这是一种新颖的主动防御框架，旨在扰乱和误导自主越狱过程。我们的核心想法是故意向对手提供“虚假回应”，这些回应似乎是成功越狱攻击的结果，但不包含实际的有害内容。这些误导性响应为攻击者的内部优化循环提供了错误信号，导致对抗性搜索提前终止并有效地越狱。通过在最先进的LLM、越狱框架和安全基准上进行广泛的实验，我们的方法持续且显着地将攻击成功率降低高达92%。与其他防御框架结合使用时，它进一步将最新攻击策略的成功率降低至0%。ProAct代表了一种垂直防御策略，可以作为额外的护栏，以增强LLM的安全性，抵御最有效的越狱攻击。



## **13. Rethinking Exact Unlearning under Exposure: Extracting Forgotten Data under Exact Unlearning in Large Language Model**

重新思考暴露下的精确取消学习：大型语言模型中精确取消学习下的被遗忘数据 cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2505.24379v2) [paper-pdf](http://arxiv.org/pdf/2505.24379v2)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.

摘要: 大型语言模型通常在从网络收集的数据集上进行训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确的取消学习（在没有目标数据的情况下从头开始重新训练模型）被广泛认为是减轻部署中隐私风险的黄金标准。在本文中，我们在实际部署环境中重新审视了这一假设，其中暴露了取消学习前和取消学习后的日志API，例如在开放重量场景中。针对此设置，我们引入了一种新颖的数据提取攻击，该攻击利用来自取消学习前模型的信号来指导取消学习后模型，从而发现反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。我们的研究结果表明，取消学习可能会以一种矛盾的方式增加现实世界部署期间隐私泄露的风险，鉴于此，我们主张评估取消学习方法，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。代码可在https://github.com/Nicholas0228/unlearned_data_extraction_llm上公开获取。



## **14. Imperceptible Jailbreaking against Large Language Models**

针对大型语言模型的无形越狱 cs.CL

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05025v1) [paper-pdf](http://arxiv.org/pdf/2510.05025v1)

**Authors**: Kuofeng Gao, Yiming Li, Chao Du, Xin Wang, Xingjun Ma, Shu-Tao Xia, Tianyu Pang

**Abstract**: Jailbreaking attacks on the vision modality typically rely on imperceptible adversarial perturbations, whereas attacks on the textual modality are generally assumed to require visible modifications (e.g., non-semantic suffixes). In this paper, we introduce imperceptible jailbreaks that exploit a class of Unicode characters called variation selectors. By appending invisible variation selectors to malicious questions, the jailbreak prompts appear visually identical to original malicious questions on screen, while their tokenization is "secretly" altered. We propose a chain-of-search pipeline to generate such adversarial suffixes to induce harmful responses. Our experiments show that our imperceptible jailbreaks achieve high attack success rates against four aligned LLMs and generalize to prompt injection attacks, all without producing any visible modifications in the written prompt. Our code is available at https://github.com/sail-sg/imperceptible-jailbreaks.

摘要: 对视觉形态的越狱攻击通常依赖于难以察觉的对抗性扰动，而对文本形态的攻击通常被认为需要可见的修改（例如，非语义后缀）。在本文中，我们引入了难以察觉的越狱，这些越狱利用了一类称为变体选择器的Unicode字符。通过在恶意问题中添加隐形变异选择器，越狱提示在视觉上与屏幕上原始恶意问题相同，而它们的标记化被“秘密”更改。我们提出了一个搜索链管道来生成此类对抗性后缀以引发有害反应。我们的实验表明，我们的不可感知的越狱对四个对齐的LLM实现了高攻击成功率，并推广到提示注入攻击，所有这些都不会在书面提示中产生任何可见的修改。我们的代码可在https://github.com/sail-sg/imperceptible-jailbreaks上获取。



## **15. Cooperative Decentralized Backdoor Attacks on Vertical Federated Learning**

垂直联邦学习的合作去中心后门攻击 cs.LG

This paper is currently under review in the IEEE/ACM Transactions on  Networking Special Issue on AI and Networking

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2501.09320v2) [paper-pdf](http://arxiv.org/pdf/2501.09320v2)

**Authors**: Seohyun Lee, Wenzhi Fang, Anindya Bijoy Das, Seyyedali Hosseinalipour, David J. Love, Christopher G. Brinton

**Abstract**: Federated learning (FL) is vulnerable to backdoor attacks, where adversaries alter model behavior on target classification labels by embedding triggers into data samples. While these attacks have received considerable attention in horizontal FL, they are less understood for vertical FL (VFL), where devices hold different features of the samples, and only the server holds the labels. In this work, we propose a novel backdoor attack on VFL which (i) does not rely on gradient information from the server and (ii) considers potential collusion among multiple adversaries for sample selection and trigger embedding. Our label inference model augments variational autoencoders with metric learning, which adversaries can train locally. A consensus process over the adversary graph topology determines which datapoints to poison. We further propose methods for trigger splitting across the adversaries, with an intensity-based implantation scheme skewing the server towards the trigger. Our convergence analysis reveals the impact of backdoor perturbations on VFL indicated by a stationarity gap for the trained model, which we verify empirically as well. We conduct experiments comparing our attack with recent backdoor VFL approaches, finding that ours obtains significantly higher success rates for the same main task performance despite not using server information. Additionally, our results verify the impact of collusion on attack performance.

摘要: 联邦学习（FL）容易受到后门攻击，其中对手通过将触发器嵌入数据样本来改变目标分类标签上的模型行为。虽然这些攻击在水平FL中受到了相当大的关注，但对于垂直FL（VFL），它们的理解较少，其中设备持有样本的不同特征，并且只有服务器持有标签。在这项工作中，我们提出了一种新的后门攻击VFL（i）不依赖于梯度信息从服务器和（ii）考虑样本选择和触发器嵌入多个对手之间的潜在勾结。我们的标签推理模型通过度量学习增强了变分自编码器，对手可以在本地进行训练。针对对手图布局的共识过程确定要毒害哪些数据点。我们进一步提出了在对手之间进行触发分离的方法，其中基于强度的植入方案将服务器倾斜到触发器。我们的收敛分析揭示了后门扰动对VFL的影响，这由训练模型的平稳性差距所表明，我们也通过经验验证了这一点。我们进行了实验，将我们的攻击与最近的后门VFL方法进行了比较，发现尽管没有使用服务器信息，但我们的攻击在相同的主要任务性能下获得了显着更高的成功率。此外，我们的结果验证了共谋对攻击性能的影响。



## **16. NatGVD: Natural Adversarial Example Attack towards Graph-based Vulnerability Detection**

NatGVD：针对基于图的漏洞检测的自然对抗示例攻击 cs.CR

10 pages, 2 figures (2 additional figures in Appendices)

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04987v1) [paper-pdf](http://arxiv.org/pdf/2510.04987v1)

**Authors**: Avilash Rath, Weiliang Qi, Youpeng Li, Xinda Wang

**Abstract**: Graph-based models learn rich code graph structural information and present superior performance on various code analysis tasks. However, the robustness of these models against adversarial example attacks in the context of vulnerability detection remains an open question. This paper proposes NatGVD, a novel attack methodology that generates natural adversarial vulnerable code to circumvent GNN-based and graph-aware transformer-based vulnerability detectors. NatGVD employs a set of code transformations that modify graph structure while preserving code semantics. Instead of injecting dead or unrelated code like previous works, NatGVD considers naturalness requirements: generated examples should not be easily recognized by humans or program analysis tools. With extensive evaluation of NatGVD on state-of-the-art vulnerability detection systems, the results reveal up to 53.04% evasion rate across GNN-based detectors and graph-aware transformer-based detectors. We also explore potential defense strategies to enhance the robustness of these systems against NatGVD.

摘要: 基于图的模型学习丰富的代码图结构信息，并在各种代码分析任务中呈现卓越的性能。然而，这些模型在漏洞检测的背景下对抗示例攻击的稳健性仍然是一个悬而未决的问题。本文提出了NatGVD，这是一种新型攻击方法，可以生成自然对抗性漏洞代码来规避基于GNN和基于图形感知转换器的漏洞检测器。NatGVD采用一组代码转换，可以修改图结构，同时保留代码语义。NatGVD没有像之前的作品那样注入死或不相关的代码，而是考虑自然性要求：生成的示例不应轻易被人类或程序分析工具识别。通过对NatGVD在最先进漏洞检测系统上的广泛评估，结果显示基于GNN的检测器和基于图形感知的基于变压器的检测器的规避率高达53.04%。我们还探索潜在的防御策略，以增强这些系统对NatGVD的稳健性。



## **17. Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning**

数据集属性对深度迁移学习的成员推断漏洞的影响 cs.CR

Accepted to NeurIPS 2025; 47 pages, 13 figures

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2402.06674v5) [paper-pdf](http://arxiv.org/pdf/2402.06674v5)

**Authors**: Marlon Tobaben, Hibiki Ito, Joonas Jälkö, Yuan He, Antti Honkela

**Abstract**: Membership inference attacks (MIAs) are used to test practical privacy of machine learning models. MIAs complement formal guarantees from differential privacy (DP) under a more realistic adversary model. We analyse MIA vulnerability of fine-tuned neural networks both empirically and theoretically, the latter using a simplified model of fine-tuning. We show that the vulnerability of non-DP models when measured as the attacker advantage at a fixed false positive rate reduces according to a simple power law as the number of examples per class increases. A similar power-law applies even for the most vulnerable points, but the dataset size needed for adequate protection of the most vulnerable points is very large.

摘要: 成员资格推理攻击（MIA）用于测试机器学习模型的实际隐私。MIA补充了更现实的对手模型下的差异隐私（DP）的形式保证。我们从经验和理论上分析了微调神经网络的MIA脆弱性，后者使用简化的微调模型。我们表明，当以固定误报率下的攻击者优势来衡量时，非DP模型的脆弱性随着每个类别示例数量的增加根据简单的乘势定律而减少。类似的乘势定律甚至适用于最脆弱的点，但充分保护最脆弱的点所需的数据集大小非常大。



## **18. Sampling-aware Adversarial Attacks Against Large Language Models**

针对大型语言模型的采样感知对抗攻击 cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2507.04446v3) [paper-pdf](http://arxiv.org/pdf/2507.04446v3)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point greedy generations, overlooking the inherently stochastic nature of LLMs and overestimating robustness. We show that for the goal of eliciting harmful responses, repeated sampling of model outputs during the attack complements prompt optimization and serves as a strong and efficient attack vector. By casting attacks as a resource allocation problem between optimization and sampling, we determine compute-optimal trade-offs and show that integrating sampling into existing attacks boosts success rates by up to 37\% and improves efficiency by up to two orders of magnitude. We further analyze how distributions of output harmfulness evolve during an adversarial attack, discovering that many common optimization strategies have little effect on output harmfulness. Finally, we introduce a label-free proof-of-concept objective based on entropy maximization, demonstrating how our sampling-aware perspective enables new optimization targets. Overall, our findings establish the importance of sampling in attacks to accurately assess and strengthen LLM safety at scale.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代中的有害响应，忽视了LLM固有的随机性并高估了稳健性。我们表明，为了引发有害响应，攻击期间对模型输出进行重复采样可以补充即时优化，并充当强大而有效的攻击载体。通过将攻击视为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可以将成功率提高高达37%，并将效率提高高达两个数量级。我们进一步分析了在对抗性攻击过程中输出危害性的分布如何演变，发现许多常见的优化策略对输出危害性几乎没有影响。最后，我们介绍了一个基于熵最大化的无标签概念验证目标，展示了我们的采样感知视角如何实现新的优化目标。总的来说，我们的研究结果确立了在攻击中采样的重要性，以准确评估和加强LLM的大规模安全性。



## **19. Unified Threat Detection and Mitigation Framework (UTDMF): Combating Prompt Injection, Deception, and Bias in Enterprise-Scale Transformers**

统一威胁检测和缓解框架（UTDMF）：对抗规模变形金刚中的即时注入、欺骗和偏见 cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04528v1) [paper-pdf](http://arxiv.org/pdf/2510.04528v1)

**Authors**: Santhosh KumarRavindran

**Abstract**: The rapid adoption of large language models (LLMs) in enterprise systems exposes vulnerabilities to prompt injection attacks, strategic deception, and biased outputs, threatening security, trust, and fairness. Extending our adversarial activation patching framework (arXiv:2507.09406), which induced deception in toy networks at a 23.9% rate, we introduce the Unified Threat Detection and Mitigation Framework (UTDMF), a scalable, real-time pipeline for enterprise-grade models like Llama-3.1 (405B), GPT-4o, and Claude-3.5. Through 700+ experiments per model, UTDMF achieves: (1) 92% detection accuracy for prompt injection (e.g., jailbreaking); (2) 65% reduction in deceptive outputs via enhanced patching; and (3) 78% improvement in fairness metrics (e.g., demographic bias). Novel contributions include a generalized patching algorithm for multi-threat detection, three groundbreaking hypotheses on threat interactions (e.g., threat chaining in enterprise workflows), and a deployment-ready toolkit with APIs for enterprise integration.

摘要: 企业系统中大型语言模型（LLM）的快速采用暴露了即时注入攻击、战略欺骗和有偏见的输出的漏洞，威胁到安全性、信任和公平性。我们扩展了我们的对抗性激活补丁框架（arXiv：2507.09406），该框架在玩具网络中以23.9%的比例引发了欺骗，我们引入了统一威胁检测和缓解框架（UTDMF），这是一个可扩展的实时管道，用于Llama-3.1（405 B）、GPT-4 o和Claude-3.5等企业级模型。通过每个型号700+次实验，UTDMF实现了：（1）92%的即时注射检测准确率（例如，越狱）;（2）通过增强的补丁将欺骗性输出减少65%;以及（3）公平性指标提高78%（例如，人口偏见）。新颖的贡献包括用于多威胁检测的通用补丁算法、关于威胁相互作用的三个开创性假设（例如，企业工作流程中的威胁链），以及一个具有企业集成API的部署就绪工具包。



## **20. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

通过在线自玩强化学习来追逐移动目标，以实现更安全的语言模型 cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2506.07468v3) [paper-pdf](http://arxiv.org/pdf/2506.07468v3)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).

摘要: 传统语言模型（LM）安全对齐依赖于反应性、不相交的过程：攻击者利用静态模型，然后进行防御性微调以修补暴露的漏洞。这种顺序方法造成了不匹配--攻击者过度适应过时的防御，而防御者则永远落后于新兴威胁。为了解决这个问题，我们提出了Self-RedTeam，这是一种在线自玩强化学习算法，攻击者和防御者代理通过持续的交互共同进化。我们将安全调整视为一个两人零和游戏，其中单一模型在攻击者和防御者角色之间交替--生成对抗性提示并防范它们--而奖励LM则判定结果。这实现了动态协同适应。我们以零和游戏的博弈论框架为基础，建立了一个理论安全保证，这激励了我们的方法的设计：如果自我游戏收敛于纳什均衡，防御者将可靠地对任何对抗输入产生安全反应。从经验上看，与针对静态防御者训练的攻击者相比，Self-RedTeam发现了更多样化的攻击（+21.8%SBERT），并在安全基准上实现了更高的稳健性（例如，WildJailBreak上+65.5%）比防守者训练对抗静态攻击者。我们进一步提出隐藏的思想链，允许代理人私下计划，这可以增强对抗多样性并减少过度拒绝。我们的结果促使LM安全培训从反应性修补转向主动协同进化，通过多代理强化学习（MARL）实现LM的可扩展、自主和稳健的自我改进。



## **21. SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations**

SECA：引发LLM幻觉的语义等效和一致攻击 cs.CL

Accepted at NeurIPS 2025. Code is available at  https://github.com/Buyun-Liang/SECA

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04398v1) [paper-pdf](http://arxiv.org/pdf/2510.04398v1)

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.

摘要: 大型语言模型（LLM）越来越多地部署在高风险领域。然而，最先进的LLM经常会产生幻觉，从而引发人们对其可靠性的严重担忧。之前的工作探索了LLM中幻觉引发的对抗攻击，但它经常会产生不切实际的提示，要么通过插入胡言乱语的标记，要么通过改变原来的含义。因此，这些方法对幻觉在实践中如何发生的了解有限。虽然计算机视觉中的对抗攻击通常涉及对输入图像的真实修改，但寻找真实的对抗提示以引发LLM幻觉的问题在很大程度上仍然没有得到充分的研究。为了解决这一差距，我们提出了语义等效和连贯攻击（SECA），通过对提示进行现实修改来引发幻觉，以保留其含义，同时保持语义连贯性。我们的贡献有三重：（i）我们将寻找对幻觉引发的现实攻击制定为在语义等效和一致性约束下输入提示空间上的受约束优化问题;（ii）我们引入了一种约束保持零阶方法来有效地搜索对抗但可行的提示;和（iii）我们通过开放式多-的实验来证明与现有方法相比，SECA的选择问题回答任务实现了更高的攻击成功率，同时几乎不会违反约束。SECA强调了开源和商业梯度不可访问的LLM对现实且合理的提示变化的敏感性。代码可在https://github.com/Buyun-Liang/SECA上获取。



## **22. Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models**

揭开后门：通过预训练语言模型的学生注意力异常评分进行可解释的防御 cs.CL

15 pages total (9 pages main text + 4 pages appendix + references),  12 figures, preprint version. The final version may differ

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04347v1) [paper-pdf](http://arxiv.org/pdf/2510.04347v1)

**Authors**: Anindya Sundar Das, Kangjie Chen, Monowar Bhuyan

**Abstract**: Pre-trained language models have achieved remarkable success across a wide range of natural language processing (NLP) tasks, particularly when fine-tuned on large, domain-relevant datasets. However, they remain vulnerable to backdoor attacks, where adversaries embed malicious behaviors using trigger patterns in the training data. These triggers remain dormant during normal usage, but, when activated, can cause targeted misclassifications. In this work, we investigate the internal behavior of backdoored pre-trained encoder-based language models, focusing on the consistent shift in attention and gradient attribution when processing poisoned inputs; where the trigger token dominates both attention and gradient signals, overriding the surrounding context. We propose an inference-time defense that constructs anomaly scores by combining token-level attention and gradient information. Extensive experiments on text classification tasks across diverse backdoor attack scenarios demonstrate that our method significantly reduces attack success rates compared to existing baselines. Furthermore, we provide an interpretability-driven analysis of the scoring mechanism, shedding light on trigger localization and the robustness of the proposed defense.

摘要: 预训练的语言模型在广泛的自然语言处理（NLP）任务中取得了显着的成功，特别是在对大型领域相关数据集进行微调时。然而，它们仍然容易受到后门攻击，对手使用触发模式在训练数据中嵌入恶意行为。这些触发器在正常使用期间保持休眠状态，但当激活时，可能会导致有针对性的错误分类。在这项工作中，我们研究了后门预训练的基于编码器的语言模型的内部行为，重点关注处理有毒输入时注意力和梯度归因的一致转变;其中触发标记主导了注意力和梯度信号，凌驾于周围的上下文。我们提出了一种推理时防御，通过结合标记级注意力和梯度信息来构建异常分数。针对各种后门攻击场景中的文本分类任务进行的广泛实验表明，与现有基线相比，我们的方法显着降低了攻击成功率。此外，我们还对评分机制提供了可解释性驱动的分析，揭示了触发定位和拟议防御的鲁棒性。



## **23. VortexPIA: Indirect Prompt Injection Attack against LLMs for Efficient Extraction of User Privacy**

VortexPIA：针对LLM的间接提示注入攻击，以有效提取用户隐私 cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04261v1) [paper-pdf](http://arxiv.org/pdf/2510.04261v1)

**Authors**: Yu Cui, Sicheng Pan, Yifei Liu, Haibin Zhang, Cong Zuo

**Abstract**: Large language models (LLMs) have been widely deployed in Conversational AIs (CAIs), while exposing privacy and security threats. Recent research shows that LLM-based CAIs can be manipulated to extract private information from human users, posing serious security threats. However, the methods proposed in that study rely on a white-box setting that adversaries can directly modify the system prompt. This condition is unlikely to hold in real-world deployments. The limitation raises a critical question: can unprivileged attackers still induce such privacy risks in practical LLM-integrated applications? To address this question, we propose \textsc{VortexPIA}, a novel indirect prompt injection attack that induces privacy extraction in LLM-integrated applications under black-box settings. By injecting token-efficient data containing false memories, \textsc{VortexPIA} misleads LLMs to actively request private information in batches. Unlike prior methods, \textsc{VortexPIA} allows attackers to flexibly define multiple categories of sensitive data. We evaluate \textsc{VortexPIA} on six LLMs, covering both traditional and reasoning LLMs, across four benchmark datasets. The results show that \textsc{VortexPIA} significantly outperforms baselines and achieves state-of-the-art (SOTA) performance. It also demonstrates efficient privacy requests, reduced token consumption, and enhanced robustness against defense mechanisms. We further validate \textsc{VortexPIA} on multiple realistic open-source LLM-integrated applications, demonstrating its practical effectiveness.

摘要: 大型语言模型（LLM）已广泛部署在对话式人工智能（CAIs）中，同时暴露了隐私和安全威胁。最近的研究表明，基于LLM的CAE可能会被操纵以提取人类用户的私人信息，从而构成严重的安全威胁。然而，该研究中提出的方法依赖于白盒设置，对手可以直接修改系统提示。这种情况在现实世界的部署中不太可能成立。这一限制提出了一个关键问题：无特权攻击者仍然能在实际的LLM集成应用程序中引发此类隐私风险吗？为了解决这个问题，我们提出了\textsk {VortexPIA}，这是一种新型的间接提示注入攻击，可以在黑匣子设置下在LLM集成应用程序中诱导隐私提取。通过注入包含错误记忆的代币高效数据，\textsk {VortexPIA}误导LLM批量主动请求私人信息。与之前的方法不同，\textsk {VortexPIA}允许攻击者灵活定义多个类别的敏感数据。我们在四个基准数据集的六个LLM上评估\textsk {VortexPIA}，涵盖传统和推理LLM。结果表明，\textsk {VortexPIA}的性能显着优于基线，并实现了最先进的（SOTA）性能。它还展示了高效的隐私请求、减少的令牌消耗以及增强的防御机制稳健性。我们在多个现实的开源LLM集成应用程序上进一步验证了\textsk {VortexPIA}，证明了其实际有效性。



## **24. Machine Unlearning in Speech Emotion Recognition via Forget Set Alone**

通过Forget Set Alone实现语音情感识别的机器去学习 cs.SD

Submitted to ICASSP 2026

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04251v1) [paper-pdf](http://arxiv.org/pdf/2510.04251v1)

**Authors**: Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz

**Abstract**: Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.

摘要: 语音情感识别旨在从语音信号中识别出情感状态，已广泛应用于人机交互、教育、医疗保健等许多领域。然而，由于语音数据包含丰富的敏感信息，出于隐私考虑，发言人可能会要求删除部分数据。当前的机器去学习方法很大程度上依赖于样本以外的数据。然而，当数据重新分配受到限制并需要大数据环境下的大量计算资源时，这种依赖会带来挑战。我们提出了一种新颖的基于对抗攻击的方法，该方法仅使用被遗忘的数据来微调预训练的语音情感识别模型。实验结果表明，所提出的方法可以有效地从模型中去除需要遗忘的数据的知识，同时在情感识别测试集中保持较高的模型性能。



## **25. Concept-Based Masking: A Patch-Agnostic Defense Against Adversarial Patch Attacks**

基于概念的屏蔽：对抗补丁攻击的补丁无关防御 cs.CV

neurips workshop

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04245v1) [paper-pdf](http://arxiv.org/pdf/2510.04245v1)

**Authors**: Ayushi Mehrotra, Derek Peng, Dipkamal Bhusal, Nidhi Rastogi

**Abstract**: Adversarial patch attacks pose a practical threat to deep learning models by forcing targeted misclassifications through localized perturbations, often realized in the physical world. Existing defenses typically assume prior knowledge of patch size or location, limiting their applicability. In this work, we propose a patch-agnostic defense that leverages concept-based explanations to identify and suppress the most influential concept activation vectors, thereby neutralizing patch effects without explicit detection. Evaluated on Imagenette with a ResNet-50, our method achieves higher robust and clean accuracy than the state-of-the-art PatchCleanser, while maintaining strong performance across varying patch sizes and locations. Our results highlight the promise of combining interpretability with robustness and suggest concept-driven defenses as a scalable strategy for securing machine learning models against adversarial patch attacks.

摘要: 对抗性补丁攻击通过局部扰动（通常在物理世界中实现）强制进行有针对性的错误分类，对深度学习模型构成了实际威胁。现有的防御通常假设事先了解补丁大小或位置，从而限制了其适用性。在这项工作中，我们提出了一种补丁不可知的防御，该防御利用基于概念的解释来识别和抑制最有影响力的概念激活载体，从而在没有明确检测的情况下中和补丁效应。通过ResNet-50在Imagenette上进行评估，我们的方法比最先进的PatchCleanser实现了更高的稳健性和清洁准确性，同时在不同的贴片尺寸和位置上保持强劲的性能。我们的结果强调了将可解释性与稳健性相结合的前景，并建议概念驱动的防御作为一种可扩展的策略，用于保护机器学习模型免受对抗性补丁攻击。



## **26. Blending adversarial training and representation-conditional purification via aggregation improves adversarial robustness**

通过聚合混合对抗训练和代表条件纯化提高对抗鲁棒性 cs.CV

Published in Transactions on Machine Learning Research (09/2025). 25  pages, 1 figure, 19 tables

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2306.06081v6) [paper-pdf](http://arxiv.org/pdf/2306.06081v6)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and a carefully chosen aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for Cifar-10, Cifar-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at: https://github.com/emaballarin/CARSO .

摘要: 在这项工作中，我们提出了一种新型的图像分类对抗防御机制-- CARSO --以协同鲁棒性增强的方式融合了对抗训练和对抗净化的范式。该方法建立在经过对抗训练的分类器之上，并学会将其与潜在干扰的输入相关的内部表示映射到试探性干净重建的分布上。来自这种分布的多个样本由相同的反向训练模型进行分类，精心选择的输出聚合最终构成感兴趣的稳健预测。通过跨不同图像数据集的强自适应攻击成熟基准进行的实验评估表明，CARSO能够抵御为随机防御而设计的自适应端到端白盒攻击。我们的方法虽然付出了适度的精确度，但大幅提高了Cifar-10、Cifar-100和TinyImageNet-200 $\ell_\infty$的最新水平，针对AutoAttack的稳健分类准确性。获取预训练模型的代码和说明可访问：https://github.com/emaballarin/CARSO。



## **27. Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System**

基于说话人嵌入的零镜头文本到语音系统中的对抗攻击和鲁棒防御 eess.AS

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2410.04017v2) [paper-pdf](http://arxiv.org/pdf/2410.04017v2)

**Authors**: Ze Li, Yao Shi, Yunfei Xu, Ming Li

**Abstract**: Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker's audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model's robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.

摘要: 基于说话者嵌入的零镜头文本到语音（TTC）系统能够使用最少的数据为未见说话者提供高质量的语音合成。然而，这些系统很容易受到对抗性攻击，攻击者向原始说话者的音频波形引入难以察觉的扰动，导致合成语音声音像另一个人一样。该漏洞带来了重大的安全风险，包括说话者身份欺骗和未经授权的语音操纵。本文研究了解决这些威胁的两种主要防御策略：对抗性训练和对抗性净化。对抗性训练通过在训练过程中集成对抗性示例来增强模型的稳健性，从而提高对此类攻击的抵抗力。另一方面，对抗性净化采用扩散概率模型来将对抗性干扰的音频恢复为干净的形式。实验结果表明，这些防御机制可以显着降低对抗性扰动的影响，增强对抗性环境中基于说话人嵌入的零发射TTC系统的安全性和可靠性。



## **28. Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data**

桌面上的边界：针对结构化数据的高效基于决策的黑匣子攻击 cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2509.22850v2) [paper-pdf](http://arxiv.org/pdf/2509.22850v2)

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems.

摘要: 与视觉和语言领域相比，结构化数据中的对抗稳健性仍然是一个未充分探索的前沿。在这项工作中，我们引入了一种针对表格数据量身定制的新型黑匣子、基于决策的对抗攻击。我们的方法将无梯度方向估计与迭代边界搜索相结合，能够在最小的Oracle访问下高效导航离散和连续特征空间。大量实验表明，我们的方法成功地妥协了不同模型的几乎整个测试集，范围包括经典机器学习分类器和基于大型语言模型（LLM）的管道。值得注意的是，该攻击的成功率始终高于90%，而每个实例只需要少量查询。这些结果凸显了表格模型对对抗性扰动的严重脆弱性，凸显了现实世界决策系统中迫切需要更强的防御。



## **29. AttackSeqBench: Benchmarking Large Language Models in Analyzing Attack Sequences within Cyber Threat Intelligence**

AttackSeqBench：在分析网络威胁情报中的攻击序列时对大型语言模型进行基准测试 cs.CR

36 pages, 9 figures

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2503.03170v2) [paper-pdf](http://arxiv.org/pdf/2503.03170v2)

**Authors**: Haokai Ma, Javier Yong, Yunshan Ma, Kuei Chen, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: Cyber Threat Intelligence (CTI) reports document observations of cyber threats, synthesizing evidence about adversaries' actions and intent into actionable knowledge that informs detection, response, and defense planning. However, the unstructured and verbose nature of CTI reports poses significant challenges for security practitioners to manually extract and analyze such sequences. Although large language models (LLMs) exhibit promise in cybersecurity tasks such as entity extraction and knowledge graph construction, their understanding and reasoning capabilities towards behavioral sequences remains underexplored. To address this, we introduce AttackSeqBench, a benchmark designed to systematically evaluate LLMs' reasoning abilities across the tactical, technical, and procedural dimensions of adversarial behaviors, while satisfying Extensibility, Reasoning Scalability, and Domain-dpecific Epistemic Expandability. We further benchmark 7 LLMs, 5 LRMs and 4 post-training strategies across the proposed 3 benchmark settings and 3 benchmark tasks within our AttackSeqBench to identify their advantages and limitations in such specific domain. Our findings contribute to a deeper understanding of LLM-driven CTI report understanding and foster its application in cybersecurity operations.

摘要: 网络威胁情报（RTI）报告记录了对网络威胁的观察，将有关对手行为和意图的证据合成为可操作的知识，为检测、响应和防御规划提供信息。然而，RTI报告的非结构化和冗长性给安全从业者手动提取和分析此类序列带来了巨大挑战。尽管大型语言模型（LLM）在实体提取和知识图构建等网络安全任务中表现出希望，但其对行为序列的理解和推理能力仍然未得到充分探索。为了解决这个问题，我们引入了AttackSeqBench，这是一个基准，旨在系统地评估LLM在对抗行为的战术、技术和程序方面的推理能力，同时满足可扩展性、推理可扩展性和特定领域的认识扩展性。我们在AttackSeqBench内提议的3个基准设置和3个基准任务中进一步对7个LLM、5个LRM和4个训练后策略进行基准测试，以确定它们在此类特定领域的优势和局限性。我们的研究结果有助于更深入地理解法学硕士驱动的RTI报告理解，并促进其在网络安全运营中的应用。



## **30. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

SafeGuider：针对文本到图像模型的稳健且实用的内容安全控制 cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.05173v2) [paper-pdf](http://arxiv.org/pdf/2510.05173v2)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.

摘要: 文本到图像模型在从自然语言描述生成高质量图像方面表现出了非凡的能力。然而，这些模型非常容易受到对抗提示的影响，这可能会绕过安全措施并产生有害内容。尽管有各种防御策略，但在现实世界应用程序中保持实用性的同时实现针对攻击的鲁棒性仍然是一个重大挑战。为了解决这个问题，我们首先对稳定扩散（SD）模型中的文本编码器进行了实证研究，该模型是一种广泛使用且具有代表性的文本到图像模型。我们的研究结果表明，[EOS]令牌充当语义聚合器，在其嵌入空间中的良性提示和对抗提示之间表现出明显的分布模式。基于这一见解，我们引入了\textBF{SafeGuider}，这是一个两步框架，旨在在不影响发电质量的情况下进行稳健的安全控制。SafeGuider将嵌入级识别模型与安全意识特征擦除束搜索算法相结合。此集成使该框架能够为良性提示维持高质量图像生成，同时确保针对域内和域外攻击的强大防御。SafeGuider在最大限度地降低攻击成功率方面表现出出色的有效性，在各种攻击场景中实现的最高攻击成功率仅为5.48%。此外，\textBF{SafeGuider}不会拒绝为不安全提示生成或产生黑色图像，而是生成安全且有意义的图像，增强了其实际实用性。此外，SafeGuider不限于SD模型，可以有效应用于其他文本到图像模型，例如Flux模型，展示了其在不同架构中的通用性和适应性。我们希望SafeGuider能够为安全文本到图像系统的实际部署提供一些线索。



## **31. Cyber Warfare During Operation Sindoor: Malware Campaign Analysis and Detection Framework**

斯诺登行动期间的网络战：恶意软件活动分析和检测框架 cs.CR

Accepted for presentation at the 21st International Conference on  Information Systems Security (ICISS 2025)

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04118v1) [paper-pdf](http://arxiv.org/pdf/2510.04118v1)

**Authors**: Prakhar Paliwal, Atul Kabra, Manjesh Kumar Hanawal

**Abstract**: Rapid digitization of critical infrastructure has made cyberwarfare one of the important dimensions of modern conflicts. Attacking the critical infrastructure is an attractive pre-emptive proposition for adversaries as it can be done remotely without crossing borders. Such attacks disturb the support systems of the opponents to launch any offensive activities, crippling their fighting capabilities. Cyberattacks during cyberwarfare can not only be used to steal information, but also to spread disinformation to bring down the morale of the opponents. Recent wars in Europe, Africa, and Asia have demonstrated the scale and sophistication that the warring nations have deployed to take the early upper hand. In this work, we focus on the military action launched by India, code-named Operation Sindoor, to dismantle terror infrastructure emanating from Pakistan and the cyberattacks launched by Pakistan. In particular, we study the malware used by Pakistan APT groups to deploy Remote Access Trojans in Indian systems. We provide details of the tactics and techniques used in the RAT deployment and develop a telemetry framework to collect necessary event logs using Osquery with a custom extension. Finally, we develop a detection rule that can be readily deployed to detect the presence of the RAT or any exploitation performed by the malware.

摘要: 关键基础设施的快速数字化使网络战成为现代冲突的重要方面之一。对于对手来说，攻击关键基础设施是一个有吸引力的先发制人的提议，因为它可以在不跨境的情况下远程完成。此类攻击会扰乱对手的支持系统发动任何进攻活动，削弱他们的战斗能力。网络战中的网络攻击不仅可以用于窃取信息，还可以用于传播虚假信息，打击对手的士气。最近在欧洲、非洲和亚洲发生的战争表明，交战国家为了尽早占据上风而部署的规模和复杂性。在这项工作中，我们重点关注印度发起的代号为“斯诺登行动”的军事行动，旨在摧毁源自巴基斯坦的恐怖基础设施以及巴基斯坦发起的网络攻击。特别是，我们研究了巴基斯坦APT团体用来在印度系统中部署远程访问特洛伊木马的恶意软件。我们提供了RAT部署中使用的策略和技术的详细信息，并开发了一个遥感框架，以使用Osquery和自定义扩展来收集必要的事件日志。最后，我们开发了一个检测规则，可以轻松部署该规则来检测RAT的存在或恶意软件执行的任何利用。



## **32. Universal Adversarial Perturbation Attacks On Modern Behavior Cloning Policies**

对现代行为克隆政策的普遍对抗性扰动攻击 cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2502.03698v2) [paper-pdf](http://arxiv.org/pdf/2502.03698v2)

**Authors**: Akansha Kalra, Basavasagar Patil, Guanhong Tao, Daniel S. Brown

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to offline universal perturbation attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and Vector-Quantizied Behavior Transformer (VQ-BET). We study the vulnerability of these methods to universal adversarial perturbations. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are often transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities to black-box attacks. To the best of our knowledge, we are the first to present a systematic study of the vulnerabilities of different LfD algorithms to both white-box and black-box attacks. Our findings highlight the vulnerabilities of modern BC algorithms, paving the way for future work in addressing such limitations.

摘要: 从演示中学习（LfD）算法在机器人操纵任务中显示出有希望的结果，但其对离线普遍扰动攻击的脆弱性仍然没有得到充分的研究。本文全面研究了对经典算法和最近提出的算法的对抗攻击，包括行为克隆（BC）、LSTM-GMM、隐式行为克隆（IBC）、扩散策略（DP）和Vector-Quanized Behavior Transformer（VQ-BET）。我们研究这些方法对普遍对抗性扰动的脆弱性。我们对几个模拟机器人操纵任务的实验表明，当前的大多数方法都极易受到对抗性扰动的影响。我们还表明，这些攻击通常可以跨算法、架构和任务转移，从而引发了黑匣子攻击的安全漏洞。据我们所知，我们是第一个对不同LfD算法对白盒和黑盒攻击的脆弱性进行系统研究的人。我们的研究结果凸显了现代BC算法的漏洞，为未来解决此类限制的工作铺平了道路。



## **33. Quantifying Distributional Robustness of Agentic Tool-Selection**

量化统计刀具选择的分布稳健性 cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.03992v1) [paper-pdf](http://arxiv.org/pdf/2510.03992v1)

**Authors**: Jehyeok Yeon, Isha Chaudhary, Gagandeep Singh

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic systems where they map user intents to relevant external tools to fulfill a task. A critical step in this process is tool selection, where a retriever first surfaces candidate tools from a larger pool, after which the LLM selects the most appropriate one. This pipeline presents an underexplored attack surface where errors in selection can lead to severe outcomes like unauthorized data access or denial of service, all without modifying the agent's model or code. While existing evaluations measure task performance in benign settings, they overlook the specific vulnerabilities of the tool selection mechanism under adversarial conditions. To address this gap, we introduce ToolCert, the first statistical framework that formally certifies tool selection robustness. ToolCert models tool selection as a Bernoulli success process and evaluates it against a strong, adaptive attacker who introduces adversarial tools with misleading metadata, and are iteratively refined based on the agent's previous choices. By sampling these adversarial interactions, ToolCert produces a high-confidence lower bound on accuracy, formally quantifying the agent's worst-case performance. Our evaluation with ToolCert uncovers the severe fragility: under attacks injecting deceptive tools or saturating retrieval, the certified accuracy bound drops near zero, an average performance drop of over 60% compared to non-adversarial settings. For attacks targeting the retrieval and selection stages, the certified accuracy bound plummets to less than 20% after just a single round of adversarial adaptation. ToolCert thus reveals previously unexamined security threats inherent to tool selection and provides a principled method to quantify an agent's robustness to such threats, a necessary step for the safe deployment of agentic systems.

摘要: 大型语言模型（LLM）越来越多地部署在代理系统中，它们将用户意图映射到相关的外部工具来完成任务。此过程中的一个关键步骤是工具选择，检索器首先从更大的池中寻找候选工具，然后LLM选择最合适的工具。该管道呈现了一个未充分探索的攻击表面，其中选择错误可能会导致严重的结果，例如未经授权的数据访问或拒绝服务，而所有这些都无需修改代理的模型或代码。虽然现有的评估在良性环境中衡量任务性能，但它们忽视了对抗条件下工具选择机制的特定漏洞。为了解决这一差距，我们引入了Tools Cert，这是第一个正式认证工具选择稳健性的统计框架。Tools Cert将工具选择建模为伯努里成功过程，并针对强大的、自适应的攻击者进行评估，攻击者引入具有误导性元数据的对抗性工具，并根据代理之前的选择进行迭代改进。通过对这些对抗性相互作用进行抽样，Tools Cert产生了准确性的高置信下限，正式量化代理的最坏情况表现。我们对Tools Cert的评估揭示了严重的脆弱性：在注入欺骗性工具或饱和检索的攻击下，认证的准确性界限下降到接近零，与非对抗性设置相比，平均性能下降超过60%。对于针对检索和选择阶段的攻击，仅经过一轮对抗性适应后，认证的准确性界限就会暴跌至不到20%。因此，Tools Cert揭示了工具选择固有的先前未经审查的安全威胁，并提供了一种原则性的方法来量化代理对此类威胁的稳健性，这是安全部署代理系统的必要步骤。



## **34. Cascading Adversarial Bias from Injection to Distillation in Language Models**

语言模型中从注入到蒸馏的对抗偏差级联 cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2505.24842v2) [paper-pdf](http://arxiv.org/pdf/2505.24842v2)

**Authors**: Harsh Chaudhari, Jamie Hayes, Matthew Jagielski, Ilia Shumailov, Milad Nasr, Alina Oprea

**Abstract**: Model distillation has become essential for creating smaller, deployable language models that retain larger system capabilities. However, widespread deployment raises concerns about resilience to adversarial manipulation. This paper investigates vulnerability of distilled models to adversarial injection of biased content during training. We demonstrate that adversaries can inject subtle biases into teacher models through minimal data poisoning, which propagates to student models and becomes significantly amplified. We propose two propagation modes: Untargeted Propagation, where bias affects multiple tasks, and Targeted Propagation, focusing on specific tasks while maintaining normal behavior elsewhere. With only 25 poisoned samples (0.25% poisoning rate), student models generate biased responses 76.9% of the time in targeted scenarios - higher than 69.4% in teacher models. For untargeted propagation, adversarial bias appears 6x-29x more frequently in student models on unseen tasks. We validate findings across six bias types (targeted advertisements, phishing links, narrative manipulations, insecure coding practices), various distillation methods, and different modalities spanning text and code generation. Our evaluation reveals shortcomings in current defenses - perplexity filtering, bias detection systems, and LLM-based autorater frameworks - against these attacks. Results expose significant security vulnerabilities in distilled models, highlighting need for specialized safeguards. We propose practical design principles for building effective adversarial bias mitigation strategies.

摘要: 模型提炼对于创建更小的、可部署的语言模型以保留更大的系统能力至关重要。然而，广泛部署引发了人们对对抗性操纵弹性的担忧。本文研究了提炼模型在训练期间对偏见内容的对抗性注入的脆弱性。我们证明，对手可以通过最少的数据中毒将微妙的偏见注入教师模型，这些偏见传播到学生模型并被显着放大。我们提出了两种传播模式：非目标传播（偏差影响多个任务）和目标传播（专注于特定任务，同时在其他地方保持正常行为）。由于只有25个中毒样本（中毒率为0.25%），学生模型在目标场景中有76.9%的时间产生偏见反应，高于教师模型的69.4%。对于无针对性传播，在学生模型中，在看不见的任务中，对抗性偏见的出现频率要高出6倍-29倍。我们验证了六种偏见类型（有针对性的广告、网络钓鱼链接、叙事操纵、不安全的编码实践）、各种提炼方法以及跨越文本和代码生成的不同模式的调查结果。我们的评估揭示了当前针对这些攻击的防御措施（困惑过滤、偏差检测系统和基于LLM的自动生成器框架）的缺陷。结果暴露了提炼模型中的重大安全漏洞，凸显了对专门保护措施的需求。我们提出了实用的设计原则来构建有效的对抗偏见缓解策略。



## **35. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

没有对抗防御的对抗防御：通过实例级主成分去除增强语言模型稳健性 cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.21750v2) [paper-pdf](http://arxiv.org/pdf/2507.21750v2)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.

摘要: 预训练的语言模型（PLM）推动了自然语言处理的重大进展，但仍然容易受到对抗攻击，引发了对其在现实世界应用程序中稳健性的担忧。之前的研究试图通过隐式或显式地在训练过程中引入对抗性扰动来减轻对抗性攻击的影响。虽然这两种策略都增强了稳健性，但它们通常会产生很高的计算成本。在这项工作中，我们提出了一个简单而有效的附加模块，该模块通过删除实例级主成分来增强PLM的对抗鲁棒性，而不依赖于传统的对抗防御或干扰原始训练数据。我们的方法将嵌入空间转换为逼近高斯属性，从而降低其对对抗性扰动的敏感性，同时保留语义关系。这种转换以一种最小化对抗性噪音对决策边界的影响的方式对齐嵌入分布，增强稳健性，而无需对抗性示例或昂贵的训练时间扩展。对八个基准数据集的评估表明，我们的方法提高了对抗稳健性，同时保持了与基线相当的攻击前准确性，实现了稳健性和概括性之间的平衡。



## **36. Deep Learning-Based Multi-Factor Authentication: A Survey of Biometric and Smart Card Integration Approaches**

基于深度学习的多因素认证：生物识别和智能卡集成方法的调查 cs.CR

14 pages, 3 figures, 6 tables

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.05163v1) [paper-pdf](http://arxiv.org/pdf/2510.05163v1)

**Authors**: Abdelilah Ganmati, Karim Afdel, Lahcen Koutti

**Abstract**: In the era of pervasive cyber threats and exponential growth in digital services, the inadequacy of single-factor authentication has become increasingly evident. Multi-Factor Authentication (MFA), which combines knowledge-based factors (passwords, PINs), possession-based factors (smart cards, tokens), and inherence-based factors (biometric traits), has emerged as a robust defense mechanism. Recent breakthroughs in deep learning have transformed the capabilities of biometric systems, enabling higher accuracy, resilience to spoofing, and seamless integration with hardware-based solutions. At the same time, smart card technologies have evolved to include on-chip biometric verification, cryptographic processing, and secure storage, thereby enabling compact and secure multi-factor devices. This survey presents a comprehensive synthesis of recent work (2019-2025) at the intersection of deep learning, biometrics, and smart card technologies for MFA. We analyze biometric modalities (face, fingerprint, iris, voice), review hardware-based approaches (smart cards, NFC, TPMs, secure enclaves), and highlight integration strategies for real-world applications such as digital banking, healthcare IoT, and critical infrastructure. Furthermore, we discuss the major challenges that remain open, including usability-security tradeoffs, adversarial attacks on deep learning models, privacy concerns surrounding biometric data, and the need for standardization in MFA deployment. By consolidating current advancements, limitations, and research opportunities, this survey provides a roadmap for designing secure, scalable, and user-friendly authentication frameworks.

摘要: 在网络威胁无处不在和数字服务呈指数级增长的时代，单因素身份验证的不足日益明显。多因素认证（MFA）结合了基于知识的因素（密码、个人识别码）、基于占有的因素（智能卡、代币）和基于固有的因素（生物识别特征），已成为一种强大的防御机制。深度学习的最新突破改变了生物识别系统的能力，实现了更高的准确性、对欺骗的恢复能力，并与基于硬件的解决方案无缝集成。与此同时，智能卡技术已经发展到包括片上生物识别验证、加密处理和安全存储，从而实现紧凑且安全的多因素设备。这项调查全面综合了MFA深度学习、生物识别和智能卡技术交叉领域的近期工作（2019-2025年）。我们分析生物识别模式（面部、指纹、虹膜、语音），审查基于硬件的方法（智能卡、NFC、TPS、安全飞地），并重点介绍数字银行、医疗保健物联网等现实世界应用程序的集成策略和关键基础设施。此外，我们还讨论了仍然存在的主要挑战，包括可用性与安全性的权衡、对深度学习模型的对抗性攻击、围绕生物识别数据的隐私问题以及MFA部署标准化的需要。通过整合当前的进步、限制和研究机会，本调查为设计安全、可扩展和用户友好的身份验证框架提供了路线图。



## **37. Thought Purity: A Defense Framework For Chain-of-Thought Attack**

思想纯洁性：思想链攻击的防御框架 cs.LG

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.12314v2) [paper-pdf](http://arxiv.org/pdf/2507.12314v2)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense framework that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.

摘要: 而强化学习训练的大型推理模型（LRM，例如，Deepseek-R1）在不断发展的大型语言模型（LLM）领域展示了高级推理能力，但它们对安全威胁的敏感性仍然是一个关键漏洞。这种弱点在思想链（CoT）生成过程中尤其明显，其中后门提示攻击等对抗方法可以系统性地颠覆模型的核心推理机制。新兴的思想链攻击（CoTA）通过利用即时可控性来揭示了这一漏洞，同时通过低成本干预降低CoT的安全性和任务性能。为了解决这种复杂的安全性能漏洞，我们提出了Thought Purity（TP）：一种防御框架，可以系统性地增强对恶意内容的抵抗力，同时保持运营效率。我们的解决方案通过三个协同组件实现这一目标：（1）安全优化的数据处理管道（2）强化学习增强的规则约束（3）自适应监控指标。我们的方法在强化学习一致的推理系统中建立了第一个针对CoTA漏洞的全面防御机制，显着推进了下一代人工智能架构的安全功能平衡。



## **38. From Theory to Practice: Evaluating Data Poisoning Attacks and Defenses in In-Context Learning on Social Media Health Discourse**

从理论到实践：评估社交媒体健康话语的上下文学习中的数据中毒攻击和防御 cs.LG

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03636v1) [paper-pdf](http://arxiv.org/pdf/2510.03636v1)

**Authors**: Rabeya Amin Jhuma, Mostafa Mohaimen Akand Faisal

**Abstract**: This study explored how in-context learning (ICL) in large language models can be disrupted by data poisoning attacks in the setting of public health sentiment analysis. Using tweets of Human Metapneumovirus (HMPV), small adversarial perturbations such as synonym replacement, negation insertion, and randomized perturbation were introduced into the support examples. Even these minor manipulations caused major disruptions, with sentiment labels flipping in up to 67% of cases. To address this, a Spectral Signature Defense was applied, which filtered out poisoned examples while keeping the data's meaning and sentiment intact. After defense, ICL accuracy remained steady at around 46.7%, and logistic regression validation reached 100% accuracy, showing that the defense successfully preserved the dataset's integrity. Overall, the findings extend prior theoretical studies of ICL poisoning to a practical, high-stakes setting in public health discourse analysis, highlighting both the risks and potential defenses for robust LLM deployment. This study also highlights the fragility of ICL under attack and the value of spectral defenses in making AI systems more reliable for health-related social media monitoring.

摘要: 这项研究探讨了大型语言模型中的上下文学习（ICL）如何被公共卫生情绪分析环境中的数据中毒攻击所破坏。使用人类偏肺病毒（HPPV）的推文，在支持示例中引入了同义词替换、否定插入和随机干扰等小的对抗性干扰。即使是这些微小的操纵也会造成重大破坏，情绪标签翻转的案例高达67%。为了解决这个问题，应用了光谱签名防御，该防御可以过滤掉有毒示例，同时保持数据的意义和情感完好无损。经过防御后，ICL的准确率稳定在46.7%左右，逻辑回归验证的准确率达到100%，表明防御成功地保留了数据集的完整性。总体而言，这些研究结果将先前对ICL中毒的理论研究扩展到公共卫生话语分析中的实用、高风险环境，强调了强大LLM部署的风险和潜在防御措施。这项研究还强调了ICL在攻击下的脆弱性，以及光谱防御在使人工智能系统在与健康相关的社交媒体监控中更加可靠方面的价值。



## **39. Cyber Resilience of Three-phase Unbalanced Distribution System Restoration under Sparse Adversarial Attack on Load Forecasting**

负载预测稀疏对抗攻击下的三期不平衡配电系统恢复的网络韧性 eess.SY

10 pages, 7 figures

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03635v1) [paper-pdf](http://arxiv.org/pdf/2510.03635v1)

**Authors**: Chen Chao, Zixiao Ma, Ziang Zhang

**Abstract**: System restoration is critical for power system resilience, nonetheless, its growing reliance on artificial intelligence (AI)-based load forecasting introduces significant cybersecurity risks. Inaccurate forecasts can lead to infeasible planning, voltage and frequency violations, and unsuccessful recovery of de-energized segments, yet the resilience of restoration processes to such attacks remains largely unexplored. This paper addresses this gap by quantifying how adversarially manipulated forecasts impact restoration feasibility and grid security. We develop a gradient-based sparse adversarial attack that strategically perturbs the most influential spatiotemporal inputs, exposing vulnerabilities in forecasting models while maintaining stealth. We further create a restoration-aware validation framework that embeds these compromised forecasts into a sequential restoration model and evaluates operational feasibility using an unbalanced three-phase optimal power flow formulation. Simulation results show that the proposed approach is more efficient and stealthier than baseline attacks. It reveals system-level failures, such as voltage and power ramping violations that prevent the restoration of critical loads. These findings provide actionable insights for designing cybersecurity-aware restoration planning frameworks.

摘要: 系统恢复对于电力系统的弹性至关重要，尽管如此，其对基于人工智能（AI）的负荷预测的日益依赖带来了重大的网络安全风险。不准确的预测可能会导致不可行的规划、电压和频率违规以及断电段的不成功恢复，但恢复过程对此类攻击的弹性在很大程度上仍然没有被探索。本文通过量化不利操纵的预测如何影响恢复可行性和电网安全来解决这一差距。我们开发了一种基于梯度的稀疏对抗攻击，可以战略性地扰乱最有影响力的时空输入，暴露预测模型中的漏洞，同时保持隐形性。我们进一步创建了一个评估感知验证框架，将这些受损的预测嵌入到顺序恢复模型中，并使用不平衡的三期最优潮流公式评估运营可行性。仿真结果表明，该方法比基线攻击更高效、更隐蔽。它揭示了系统级故障，例如阻碍关键负载恢复的电压和功率斜坡违规。这些发现为设计网络安全意识的恢复规划框架提供了可操作的见解。



## **40. Explainable but Vulnerable: Adversarial Attacks on XAI Explanation in Cybersecurity Applications**

可解释但脆弱：网络安全应用中XAI解释的对抗攻击 cs.CR

10 pages, 9 figures, 4 tables

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03623v1) [paper-pdf](http://arxiv.org/pdf/2510.03623v1)

**Authors**: Maraz Mia, Mir Mehedi A. Pritom

**Abstract**: Explainable Artificial Intelligence (XAI) has aided machine learning (ML) researchers with the power of scrutinizing the decisions of the black-box models. XAI methods enable looking deep inside the models' behavior, eventually generating explanations along with a perceived trust and transparency. However, depending on any specific XAI method, the level of trust can vary. It is evident that XAI methods can themselves be a victim of post-adversarial attacks that manipulate the expected outcome from the explanation module. Among such attack tactics, fairwashing explanation (FE), manipulation explanation (ME), and backdoor-enabled manipulation attacks (BD) are the notable ones. In this paper, we try to understand these adversarial attack techniques, tactics, and procedures (TTPs) on explanation alteration and thus the effect on the model's decisions. We have explored a total of six different individual attack procedures on post-hoc explanation methods such as SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanation), and IG (Integrated Gradients), and investigated those adversarial attacks in cybersecurity applications scenarios such as phishing, malware, intrusion, and fraudulent website detection. Our experimental study reveals the actual effectiveness of these attacks, thus providing an urgency for immediate attention to enhance the resiliency of XAI methods and their applications.

摘要: 可解释人工智能（XAI）帮助机器学习（ML）研究人员审查黑匣子模型的决策。XAI方法能够深入研究模型的行为，最终生成解释以及感知的信任和透明度。然而，根据任何特定的XAI方法，信任程度可能会有所不同。显然，XAI方法本身也可能成为操纵解释模块预期结果的后对抗攻击的受害者。在这些攻击策略中，公平解释（FE）、操纵解释（ME）和后门操纵攻击（BD）是值得注意的。在本文中，我们试图了解这些对抗攻击技术、策略和程序（TTP）对解释变更的影响，从而了解对模型决策的影响。我们探索了针对事后解释方法的总共六种不同的单独攻击程序，例如SHAP（SHapley Additive exPlanations）、LIME（本地可解释模型不可知解释）和IG（集成入侵），并调查了网络安全应用场景中的这些对抗攻击，例如网络钓鱼、恶意软件、入侵和欺诈网站检测。我们的实验研究揭示了这些攻击的实际有效性，因此迫切需要立即关注增强XAI方法及其应用程序的弹性。



## **41. Cross-Modal Content Optimization for Steering Web Agent Preferences**

用于引导Web代理偏好的跨模式内容优化 cs.AI

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03612v1) [paper-pdf](http://arxiv.org/pdf/2510.03612v1)

**Authors**: Tanqiu Jiang, Min Bai, Nikolaos Pappas, Yanjun Qi, Sandesh Swamy

**Abstract**: Vision-language model (VLM)-based web agents increasingly power high-stakes selection tasks like content recommendation or product ranking by combining multimodal perception with preference reasoning. Recent studies reveal that these agents are vulnerable against attackers who can bias selection outcomes through preference manipulations using adversarial pop-ups, image perturbations, or content tweaks. Existing work, however, either assumes strong white-box access, with limited single-modal perturbations, or uses impractical settings. In this paper, we demonstrate, for the first time, that joint exploitation of visual and textual channels yields significantly more powerful preference manipulations under realistic attacker capabilities. We introduce Cross-Modal Preference Steering (CPS) that jointly optimizes imperceptible modifications to an item's visual and natural language descriptions, exploiting CLIP-transferable image perturbations and RLHF-induced linguistic biases to steer agent decisions. In contrast to prior studies that assume gradient access, or control over webpages, or agent memory, we adopt a realistic black-box threat setup: a non-privileged adversary can edit only their own listing's images and textual metadata, with no insight into the agent's model internals. We evaluate CPS on agents powered by state-of-the-art proprietary and open source VLMs including GPT-4.1, Qwen-2.5VL and Pixtral-Large on both movie selection and e-commerce tasks. Our results show that CPS is significantly more effective than leading baseline methods. For instance, our results show that CPS consistently outperforms baselines across all models while maintaining 70% lower detection rates, demonstrating both effectiveness and stealth. These findings highlight an urgent need for robust defenses as agentic systems play an increasingly consequential role in society.

摘要: 基于视觉语言模型（VLM）的网络代理通过将多模式感知与偏好推理相结合，越来越多地支持内容推荐或产品排名等高风险选择任务。最近的研究表明，这些代理很容易受到攻击者的攻击，攻击者可以通过使用对抗性弹出窗口、图像扰动或内容调整的偏好操纵来偏差选择结果。然而，现有的工作要么假设强白盒访问，单模式扰动有限，要么使用不切实际的设置。在本文中，我们首次证明，在现实的攻击者能力下，视觉和文本渠道的联合利用会产生显着更强大的偏好操纵。我们引入了跨模式偏好引导（CPS），它联合优化对物品的视觉和自然语言描述的不可感知的修改，利用CLIP可转移的图像扰动和RLHF引起的语言偏差来引导代理决策。与假设梯度访问或控制网页或代理内存的先前研究相反，我们采用了现实的黑匣子威胁设置：非特权对手只能编辑自己列表的图像和文本元数据，而不深入了解代理的模型内部。我们在电影选择和电子商务任务方面评估由最先进的专有和开源VLM（包括GPT-4.1、Qwen-2.5BL和Pixtral-Large）提供支持的代理的CPS。我们的结果表明，CPS比领先的基线方法明显更有效。例如，我们的结果表明，CPS在所有型号中始终优于基线，同时保持了70%的检测率，证明了有效性和隐蔽性。这些发现凸显了对强大防御的迫切需要，因为代理系统在社会中发挥着越来越重要的作用。



## **42. NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning**

NanoFlux：针对多领域推理的对抗性双重LLM评估和蒸馏 cs.LG

preprint version

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2509.23252v2) [paper-pdf](http://arxiv.org/pdf/2509.23252v2)

**Authors**: Raviteja Anantha, Soheil Hor, Teodor Nicola Antoniu, Layne C. Price

**Abstract**: We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.

摘要: 我们提出了NanoFlux，这是一种新型的对抗框架，用于生成有针对性的训练数据以改进LLM推理，其中包含少于200个示例的对抗生成的数据集优于传统的微调方法。该框架采用了交替作为攻击者和防御者的模型之间的竞争动态，由工具增强的法官监督，合成多步骤问题以及针对特定推理能力的解释性注释。与全基准微调相比，在NanoFlox生成的数据上微调4 B参数模型可在不同领域获得性能提升：数学推理（GSMHard）+5.9%，科学推理（GenomeBench）+3.6%，医学推理（MultiMedQA）+16.6%，同时将计算要求降低3- 14倍。消融研究揭示了数据集特征和模型性能之间的非单调关系，揭示了问题复杂性和推理质量的特定领域的最佳点。NanoFlux通过基于嵌入的新颖性过滤、工具增强评估和多跳推理来自动生成训练数据，这表明未来的模型改进可能在于智能合成小型精确目标训练数据集。



## **43. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

NEXUS：在多轮LLM越狱中利用不安全序列的网络探索 cs.CR

Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03417v1) [paper-pdf](http://arxiv.org/pdf/2510.03417v1)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但仍然容易受到越狱攻击，特别是在良性交换中散布恶意意图并绕过对齐机制的多回合越狱。现有的方法常常无法很好地探索对抗空间，依赖于手工制作的启发式方法，或者缺乏系统性的查询细化。我们介绍了NEXUS（用于eXploiting Unsafe Sequences的网络探索），这是一个用于构建、细化和执行优化多回合攻击的模块化框架。NEXUS包括：（1）IncreghtNet，它将有害意图分层扩展到主题、实体和查询链的结构化语义网络中;（2）反馈驱动的模拟器，通过攻击者-受害者-法官LLM协作使用危害性和语义相似性基准来迭代细化和修剪这些链;（3）网络穿越器，自适应地导航细化查询空间以进行实时攻击。该管道揭示了LLC之间隐秘、高成功的对抗路径。在几种闭源和开源LLM上，NEXUS将攻击成功率比之前的方法提高了2.1%至19.4%。代码：https://github.com/inspire-lab/NEXUS



## **44. Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles**

通过潜在合奏的随机共振进行测试时防御对抗攻击 cs.CV

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03224v1) [paper-pdf](http://arxiv.org/pdf/2510.03224v1)

**Authors**: Dong Lao, Yuxiang Zhang, Haniyeh Ehsani Oskouie, Yangchao Wu, Alex Wong, Stefano Soatto

**Abstract**: We propose a test-time defense mechanism against adversarial attacks: imperceptible image perturbations that significantly alter the predictions of a model. Unlike existing methods that rely on feature filtering or smoothing, which can lead to information loss, we propose to "combat noise with noise" by leveraging stochastic resonance to enhance robustness while minimizing information loss. Our approach introduces small translational perturbations to the input image, aligns the transformed feature embeddings, and aggregates them before mapping back to the original reference image. This can be expressed in a closed-form formula, which can be deployed on diverse existing network architectures without introducing additional network modules or fine-tuning for specific attack types. The resulting method is entirely training-free, architecture-agnostic, and attack-agnostic. Empirical results show state-of-the-art robustness on image classification and, for the first time, establish a generic test-time defense for dense prediction tasks, including stereo matching and optical flow, highlighting the method's versatility and practicality. Specifically, relative to clean (unperturbed) performance, our method recovers up to 68.1% of the accuracy loss on image classification, 71.9% on stereo matching, and 29.2% on optical flow under various types of adversarial attacks.

摘要: 我们提出了一种针对对抗攻击的测试时防御机制：显着改变模型预测的不可感知的图像扰动。与依赖于特征过滤或平滑的现有方法可能导致信息丢失不同，我们建议通过利用随机共振来“以噪音对抗噪音”，以增强鲁棒性，同时最大限度地减少信息丢失。我们的方法向输入图像引入小的平移扰动，对齐转换后的特征嵌入，并在映射回原始参考图像之前将它们聚合。这可以用封闭式公式来表达，该公式可以部署在各种现有网络架构上，而无需引入额外的网络模块或针对特定攻击类型进行微调。由此产生的方法完全免训练、架构不可知、攻击不可知。经验结果显示了图像分类的最新鲁棒性，并首次为密集预测任务（包括立体匹配和光学流）建立了通用测试时防御，凸显了该方法的通用性和实用性。具体来说，相对于干净（未受干扰）的性能，我们的方法在各种类型的对抗性攻击下恢复了高达68.1%的图像分类准确性损失、71.9%的立体匹配准确性损失和29.2%的光通量准确性损失。



## **45. Latent Diffusion Unlearning: Protecting Against Unauthorized Personalization Through Trajectory Shifted Perturbations**

潜在扩散消除学习：通过轨迹转移扰动防止未经授权的个性化 cs.CV

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03089v1) [paper-pdf](http://arxiv.org/pdf/2510.03089v1)

**Authors**: Naresh Kumar Devulapally, Shruti Agarwal, Tejas Gokhale, Vishnu Suresh Lokhande

**Abstract**: Text-to-image diffusion models have demonstrated remarkable effectiveness in rapid and high-fidelity personalization, even when provided with only a few user images. However, the effectiveness of personalization techniques has lead to concerns regarding data privacy, intellectual property protection, and unauthorized usage. To mitigate such unauthorized usage and model replication, the idea of generating ``unlearnable'' training samples utilizing image poisoning techniques has emerged. Existing methods for this have limited imperceptibility as they operate in the pixel space which results in images with noise and artifacts. In this work, we propose a novel model-based perturbation strategy that operates within the latent space of diffusion models. Our method alternates between denoising and inversion while modifying the starting point of the denoising trajectory: of diffusion models. This trajectory-shifted sampling ensures that the perturbed images maintain high visual fidelity to the original inputs while being resistant to inversion and personalization by downstream generative models. This approach integrates unlearnability into the framework of Latent Diffusion Models (LDMs), enabling a practical and imperceptible defense against unauthorized model adaptation. We validate our approach on four benchmark datasets to demonstrate robustness against state-of-the-art inversion attacks. Results demonstrate that our method achieves significant improvements in imperceptibility ($\sim 8 \% -10\%$ on perceptual metrics including PSNR, SSIM, and FID) and robustness ( $\sim 10\%$ on average across five adversarial settings), highlighting its effectiveness in safeguarding sensitive data.

摘要: 文本到图像的扩散模型已经在快速和高保真的个性化中表现出显著的效果，即使在仅提供少数用户图像的情况下。然而，个性化技术的有效性导致了对数据隐私、知识产权保护和未经授权使用的担忧。为了减少这种未经授权的使用和模型复制，出现了利用图像中毒技术生成“不可学习”训练样本的想法。用于此的现有方法具有有限的不可感知性，因为它们在像素空间中操作，这导致具有噪声和伪影的图像。在这项工作中，我们提出了一种新型的基于模型的扰动策略，该策略在扩散模型的潜在空间内运行。我们的方法在去噪和逆之间交替，同时修改去噪轨迹的起点：扩散模型。这种椭圆形移动采样确保受干扰的图像对原始输入保持高视觉保真度，同时抵抗下游生成模型的倒置和个性化。这种方法将不可学习性集成到潜在扩散模型（LDM）的框架中，从而能够针对未经授权的模型适应进行实用且不可感知的防御。我们在四个基准数据集上验证了我们的方法，以展示针对最先进的倒置攻击的鲁棒性。结果表明，我们的方法在不可感知性（$\sim 8 \%-10\%$，感知指标包括PSNR、SSIM和DID）和稳健性（$\sim 10\%$在五个对抗性设置中平均）方面实现了显着改进，凸显了其在保护敏感数据方面的有效性。



## **46. Unveiling Unicode's Unseen Underpinnings in Undermining Authorship Attribution**

在破坏作者归属中揭开Unicode不可见的基础 cs.CR

31 pages, 7 figures, 3 tables

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2508.15840v2) [paper-pdf](http://arxiv.org/pdf/2508.15840v2)

**Authors**: Robert Dilworth

**Abstract**: When using a public communication channel -- whether formal or informal, such as commenting or posting on social media -- end users have no expectation of privacy: they compose a message and broadcast it for the world to see. Even if an end user takes utmost precautions to anonymize their online presence -- using an alias or pseudonym; masking their IP address; spoofing their geolocation; concealing their operating system and user agent; deploying encryption; registering with a disposable phone number or email; disabling non-essential settings; revoking permissions; and blocking cookies and fingerprinting -- one obvious element still lingers: the message itself. Assuming they avoid lapses in judgment or accidental self-exposure, there should be little evidence to validate their actual identity, right? Wrong. The content of their message -- necessarily open for public consumption -- exposes an attack vector: stylometric analysis, or author profiling. In this paper, we dissect the technique of stylometry, discuss an antithetical counter-strategy in adversarial stylometry, and devise enhancements through Unicode steganography.

摘要: 当使用公共沟通渠道时--无论是正式还是非正式的，例如在社交媒体上评论或发帖--最终用户对隐私没有期望：他们撰写一条消息并将其广播给全世界观看。即使最终用户采取最大的预防措施来匿名他们的在线存在--使用别名或假名;掩盖他们的IP地址;欺骗他们的地理位置;隐藏他们的操作系统和用户代理;部署加密;使用一次性电话号码或电子邮件注册;禁用非必要设置;撤销权限;阻止cookie和指纹识别--一个明显的因素仍然挥之不去：消息本身。假设他们避免判断失误或意外的自我暴露，那么应该没有什么证据来验证他们的实际身份，对吧？错了他们的信息内容--必须向公众开放--暴露了攻击载体：文体分析或作者分析。在本文中，我们剖析了风格测量技术，讨论了对抗性风格测量中的对立反策略，并通过Unicode隐写术设计增强功能。



## **47. Privacy-Aware Design of Distributed MIMO ISAC Systems**

分布式多输入多输入多 eess.SP

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2409.12874v3) [paper-pdf](http://arxiv.org/pdf/2409.12874v3)

**Authors**: Henrik Åkesson, Marco Gomes, Diana Pamela Moya Osorio

**Abstract**: Integrated Sensing and Communication (ISAC) systems raise unprecedented challenges regarding security and privacy since related applications involve the gathering of sensitive, identifiable information about people and the environment, which can lead to privacy leakage. Privacy-aware measures can steer the design of ISAC systems to prevent privacy violations. Thus, we explore this perspective for the design of distributed massive multiple-input multiple-output ISAC systems. For this purpose, we introduce an adversarial model where a malicious user exploits the interference from ISAC signals to extract sensing information. To mitigate this threat, we propose an iterative privacy-aware framework of two blocks: precoder design and access point selection. The precoder design aims to minimize the mutual information between the sensing and communication signals by imposing constraints on sensing and communication performance and maximum transmit power. The access point selection also aims to minimize the mutual information between communication and sensing signals by strategically selecting access points that transmit ISAC signals, and sensing receivers. Results show a reduction in the effectiveness of the attack measured by the probability of detection of the attacker.

摘要: 集成传感与通信（ISAC）系统在安全和隐私方面提出了前所未有的挑战，因为相关应用涉及收集有关人类和环境的敏感、可识别信息，这可能会导致隐私泄露。隐私意识措施可以指导ISAC系统的设计，以防止隐私侵犯。因此，我们探索了分布式大规模多输入多输出ISAC系统的设计的这一视角。为此，我们引入了一种对抗模型，其中恶意用户利用ISAC信号的干扰来提取感知信息。为了减轻这种威胁，我们提出了一个由两个部分组成的迭代隐私感知框架：预编码器设计和接入点选择。预编码器设计旨在通过对传感和通信性能以及最大发射功率施加限制来最大限度地减少传感和通信信号之间的互信息。接入点选择还旨在通过策略性地选择传输ISAC信号的接入点和感测接收机来最大限度地减少通信和感测信号之间的互信息。结果显示，通过检测攻击者的概率来衡量，攻击的有效性有所降低。



## **48. Rethinking the Vulnerability of Concept Erasure and a New Method**

重新思考概念擦除的脆弱性和新方法 cs.LG

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2502.17537v3) [paper-pdf](http://arxiv.org/pdf/2502.17537v3)

**Authors**: Alex D. Richardson, Kaicheng Zhang, Lucas Beerens, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. In response, concept erasure (defense) methods have been developed to "unlearn" specific concepts through post-hoc finetuning. However, recent concept restoration (attack) methods have demonstrated that these supposedly erased concepts can be recovered using adversarially crafted prompts, revealing a critical vulnerability in current defense mechanisms. In this work, we first investigate the fundamental sources of adversarial vulnerability and reveal that vulnerabilities are pervasive in the prompt embedding space of concept-erased models, a characteristic inherited from the original pre-unlearned model. Furthermore, we introduce **RECORD**, a novel coordinate-descent-based restoration algorithm that consistently outperforms existing restoration methods by up to 17.8 times. We conduct extensive experiments to assess its compute-performance tradeoff and propose acceleration strategies.

摘要: 文本到图像传播模型的激增引发了严重的隐私和安全问题，特别是在受版权保护或有害图像的生成方面。作为回应，概念擦除（防御）方法被开发出来，通过事后微调来“忘记”特定概念。然而，最近的概念恢复（攻击）方法表明，这些所谓已删除的概念可以使用敌对制作的提示来恢复，这揭示了当前防御机制中的一个关键漏洞。在这项工作中，我们首先调查了对抗性漏洞的基本来源，并揭示了漏洞在概念擦除模型的即时嵌入空间中普遍存在，这是从原始未学习模型继承的特征。此外，我们还引入了 **Record*，这是一种新型的基于坐标下降的恢复算法，其性能始终比现有恢复方法高出17.8倍。我们进行了广泛的实验来评估其计算性能权衡并提出加速策略。



## **49. Untargeted Jailbreak Attack**

无目标越狱攻击 cs.CR

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.02999v1) [paper-pdf](http://arxiv.org/pdf/2510.02999v1)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that \textsc{UJA} can achieve over 80\% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%.

摘要: 现有的对大型语言模型（LLM）的基于梯度的越狱攻击，例如贪婪协调梯度（GCG）和COLD-Attack，通常会优化对抗性后缀，以将LLM输出与预定义的目标响应保持一致。然而，通过将优化目标限制为诱导预定义的目标，这些方法本质上限制了对抗搜索空间，从而限制了其总体攻击功效。此外，现有方法通常需要大量的优化迭代来满足固定目标和原始模型响应之间的大差距，导致攻击效率低。   为了克服定向越狱攻击的局限性，我们提出了第一个基于梯度的非定向越狱攻击（UJA），旨在在不强制执行任何预定义模式的情况下引发不安全的响应。具体来说，我们制定了一个无针对性的攻击目标，以最大化LLM响应的不安全概率，该概率可以使用判断模型进行量化。由于目标是不可微的，因此我们进一步将其分解为两个可微的子目标，用于优化最佳有害反应和相应的对抗提示，并通过理论分析来验证分解。与有针对性的越狱攻击相比，UJA的无限制目标显着扩大了搜索空间，从而能够更灵活、更有效地探索LLM漏洞。广泛的评估表明，\textsk {UJA}只需100次优化迭代即可针对最近的安全性一致的LLM实现超过80%的攻击成功率，比I-GCG和COLD-Attack等最先进的基于梯度的攻击性能高出20%以上。



## **50. Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain**

Agentland的恶意：深入人工智能供应链后门的兔子洞 cs.CR

27 pages

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.05159v1) [paper-pdf](http://arxiv.org/pdf/2510.05159v1)

**Authors**: Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin

**Abstract**: The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.

摘要: 根据人工智能代理自身交互中的数据（例如网络浏览或工具使用）进行微调的做法虽然是提高代理能力的强大通用配方，但也在人工智能供应链中引入了一个关键的安全漏洞。在这项工作中，我们表明，对手可以很容易地毒害数据收集管道，以嵌入被特定目标短语触发的难以检测的后门，这样当代理遇到这些触发器时，就会执行不安全或恶意的操作。我们形式化并验证了三种针对供应链不同层的现实威胁模型：1）微调数据的直接中毒，其中攻击者控制了一小部分训练痕迹; 2）环境中毒，其中恶意指令被注入到创建训练数据时抓取的网页或调用的工具中; 3）供应链中毒，即根据干净的数据对预先后门的基础模型进行微调，以提高其代理能力。我们的结果很明显：攻击者可以通过毒害仅2%的收集痕迹，嵌入后门，导致代理在存在特定触发器时泄露机密用户信息，成功率超过80%。该漏洞适用于所有三种威胁模型。此外，我们证明，包括两种护栏模型和一种基于重量的防御在内的主要防护措施无法检测或防止恶意行为。这些发现凸显了代理人工智能开发面临的紧迫威胁，并强调了对数据收集流程和端到端模型供应链进行严格安全审查的迫切需要。



