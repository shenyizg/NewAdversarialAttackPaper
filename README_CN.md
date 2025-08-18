# Latest Adversarial Attack Papers
**update at 2025-08-18 16:21:41**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Random Walk Learning and the Pac-Man Attack**

随机步行学习和吃豆人攻击 stat.ML

The updated manuscript represents an incomplete version of the work.  A substantially updated version will be prepared before further dissemination

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.05663v2) [paper-pdf](http://arxiv.org/pdf/2508.05663v2)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Zonghong Liu, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the Average Crossing (AC) algorithm--a fully decentralized mechanism for duplicating RWs to prevent RW extinction in the presence of Pac-Man. Our theoretical analysis establishes that (i) the RW population remains almost surely bounded under AC and (ii) RW-based stochastic gradient descent remains convergent under AC, even in the presence of Pac-Man, with a quantifiable deviation from the true optimum. Our extensive empirical results on both synthetic and real-world datasets corroborate our theoretical findings. Furthermore, they uncover a phase transition in the extinction probability as a function of the duplication threshold. We offer theoretical insights by analyzing a simplified variant of the AC, which sheds light on the observed phase transition.

摘要: 由于管理费用低和可扩展性，基于随机游走（RW）的算法长期以来一直在分布式系统中流行，最近在去中心化学习中的应用越来越多。然而，它们对本地交互的依赖使它们本质上容易受到恶意行为的影响。在这项工作中，我们调查了一种对抗性威胁，我们称之为“吃豆人”攻击，其中恶意节点概率地终止访问它的任何RW。这种隐形行为逐渐从网络中消除活动RW，有效地停止学习过程，而不会触发失败警报。为了应对这种威胁，我们提出了平均交叉（AC）算法--一种完全分散的机制，用于复制RW，以防止RW在Pac-Man存在的情况下灭绝。我们的理论分析确定，（i）RW种群几乎肯定在AC下保持有界，（ii）基于RW的随机梯度下降在AC下保持收敛，即使在Pac-Man存在的情况下，与真正的最佳值存在可量化的偏差。我们对合成和现实世界数据集的广泛经验结果证实了我们的理论发现。此外，它们还揭示了灭绝概率的相转变作为复制阈值的函数。我们通过分析AC的简化变体来提供理论见解，该变体揭示了观察到的相转变。



## **2. Robust Convolution Neural ODEs via Contractivity-promoting regularization**

通过促进压缩性的规则化实现鲁棒卷积神经ODE cs.LG

Accepted in IEEE CDC2025, Rio de Janeiro, Brazil

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11432v1) [paper-pdf](http://arxiv.org/pdf/2508.11432v1)

**Authors**: Muhammad Zakwan, Liang Xu, Giancarlo Ferrari-Trecate

**Abstract**: Neural networks can be fragile to input noise and adversarial attacks.   In this work, we consider Convolutional Neural Ordinary Differential Equations (NODEs), a family of continuous-depth neural networks represented by dynamical systems, and propose to use contraction theory to improve their robustness.   For a contractive dynamical system two trajectories starting from different initial conditions converge to each other exponentially fast.   Contractive Convolutional NODEs can enjoy increased robustness as slight perturbations of the features do not cause a significant change in the output.   Contractivity can be induced during training by using a regularization term involving the Jacobian of the system dynamics.   To reduce the computational burden, we show that it can also be promoted using carefully selected weight regularization terms for a class of NODEs with slope-restricted activation functions.   The performance of the proposed regularizers is illustrated through benchmark image classification tasks on MNIST and FashionMNIST datasets, where images are corrupted by different kinds of noise and attacks.

摘要: 神经网络对于输入噪音和对抗攻击可能很脆弱。   在这项工作中，我们考虑了卷积神经常微方程（NODE），这是一个以动力系统为代表的连续深度神经网络家族，并建议使用压缩理论来提高其鲁棒性。   对于压缩动力系统来说，从不同初始条件开始的两条轨迹以指数速度相互收敛。   收缩卷积NODE可以享受更高的鲁棒性，因为特征的轻微扰动不会导致输出的显着变化。   可以在训练期间通过使用涉及系统动力学雅可比矩阵的正规化项来诱导收缩性。   为了减少计算负担，我们表明，对于一类具有斜坡限制激活函数的NODE，也可以使用精心选择的权重正规化项来推广它。   通过MNIST和FashionMNIST数据集上的基准图像分类任务来说明所提出的正规化器的性能，其中图像会受到不同类型的噪音和攻击的破坏。



## **3. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

通过在线自玩强化学习来追逐移动目标，以实现更安全的语言模型 cs.LG

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2506.07468v2) [paper-pdf](http://arxiv.org/pdf/2506.07468v2)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).

摘要: 传统语言模型（LM）安全对齐依赖于反应性、不相交的过程：攻击者利用静态模型，然后进行防御性微调以修补暴露的漏洞。这种顺序方法造成了不匹配--攻击者过度适应过时的防御，而防御者则永远落后于新兴威胁。为了解决这个问题，我们提出了Self-RedTeam，这是一种在线自玩强化学习算法，攻击者和防御者代理通过持续的交互共同进化。我们将安全调整视为一个两人零和游戏，其中单一模型在攻击者和防御者角色之间交替--生成对抗性提示并防范它们--而奖励LM则判定结果。这实现了动态协同适应。我们以零和游戏的博弈论框架为基础，建立了一个理论安全保证，这激励了我们的方法的设计：如果自我游戏收敛于纳什均衡，防御者将可靠地对任何对抗输入产生安全反应。从经验上看，与针对静态防御者训练的攻击者相比，Self-RedTeam发现了更多样化的攻击（+21.8%SBERT），并在安全基准上实现了更高的稳健性（例如，WildJailBreak上+65.5%）比防守者训练对抗静态攻击者。我们进一步提出隐藏的思想链，允许代理人私下计划，这可以增强对抗多样性并减少过度拒绝。我们的结果促使LM安全培训从反应性修补转向主动协同进化，通过多代理强化学习（MARL）实现LM的可扩展、自主和稳健的自我改进。



## **4. Semantically Guided Adversarial Testing of Vision Models Using Language Models**

使用语言模型的视觉模型的语义引导对抗测试 cs.CV

12 pages, 4 figures, 3 tables. Submitted for peer review

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11341v1) [paper-pdf](http://arxiv.org/pdf/2508.11341v1)

**Authors**: Katarzyna Filus, Jorge M. Cruz-Duarte

**Abstract**: In targeted adversarial attacks on vision models, the selection of the target label is a critical yet often overlooked determinant of attack success. This target label corresponds to the class that the attacker aims to force the model to predict. Now, existing strategies typically rely on randomness, model predictions, or static semantic resources, limiting interpretability, reproducibility, or flexibility. This paper then proposes a semantics-guided framework for adversarial target selection using the cross-modal knowledge transfer from pretrained language and vision-language models. We evaluate several state-of-the-art models (BERT, TinyLLAMA, and CLIP) as similarity sources to select the most and least semantically related labels with respect to the ground truth, forming best- and worst-case adversarial scenarios. Our experiments on three vision models and five attack methods reveal that these models consistently render practical adversarial targets and surpass static lexical databases, such as WordNet, particularly for distant class relationships. We also observe that static testing of target labels offers a preliminary assessment of the effectiveness of similarity sources, \textit{a priori} testing. Our results corroborate the suitability of pretrained models for constructing interpretable, standardized, and scalable adversarial benchmarks across architectures and datasets.

摘要: 在对视觉模型的有针对性的对抗攻击中，目标标签的选择是攻击成功的一个关键但经常被忽视的决定因素。此目标标签对应于攻击者旨在迫使模型预测的类。现在，现有的策略通常依赖于随机性、模型预测或静态语义资源，从而限制了可解释性、可重复性或灵活性。然后，本文提出了一个使用预训练语言和视觉语言模型的跨模式知识转移的语义引导框架，用于对抗性目标选择。我们评估了几个最先进的模型（BERT、TinyLLAMA和CLIP）作为相似性来源，以选择与基本事实相关最多和最不相关的标签，形成最好和最坏情况的对抗场景。我们对三种视觉模型和五种攻击方法的实验表明，这些模型始终呈现实际的对抗目标，并超越了WordNet等静态词汇数据库，特别是对于遥远的阶级关系。我们还观察到，目标标签的静态测试提供了对相似性源的有效性的初步评估，\textit {a prior}测试。我们的结果证实了预训练模型适用于构建跨架构和数据集的可解释、标准化和可扩展的对抗基准。



## **5. MUNBa: Machine Unlearning via Nash Bargaining**

MUNba：通过纳什讨价还价的机器学习 cs.CV

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2411.15537v3) [paper-pdf](http://arxiv.org/pdf/2411.15537v3)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.

摘要: 机器取消学习（MU）旨在选择性地从模型中删除有害行为，同时保留模型的整体效用。作为一个多任务学习问题，MU涉及平衡与忘记特定概念/数据和保持一般性能相关的目标。这些遗忘和保存目标的天真整合可能导致梯度冲突和优势，阻碍MU算法达到最优解。为了解决梯度冲突和优势的问题，我们重新制定MU作为一个两个球员的合作游戏，其中的两个球员，即遗忘球员和保存球员，有助于通过他们的梯度建议，以最大限度地提高他们的整体收益和平衡他们的贡献。为此，受纳什讨价还价理论的启发，我们推导出一个封闭解来引导模型走向帕累托稳定点。我们的MU公式保证了均衡解决方案，其中任何与最终状态的偏差都将导致双方参与者总体目标的减少，从而确保每个目标的最优性。我们评估了我们的算法在图像分类和图像生成等一系列不同任务中的有效性。ResNet、视觉语言模型CLIP和文本到图像扩散模型的广泛实验表明，我们的方法优于最先进的MU算法，在遗忘和保留之间实现了更好的权衡。我们的结果还强调了遗忘准确性、概括性的保留和对抗性攻击的鲁棒性的改进。



## **6. Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

7 pages, 7 figures, Accept by IEEE/RSJ International Conference on  Intelligent Robots and Systems (IROS) 2025

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2409.10071v5) [paper-pdf](http://arxiv.org/pdf/2409.10071v5)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav

摘要: 嵌入式视觉导航的重大进步引发了人们对其容易受到利用深度神经网络的对抗性攻击的担忧。研究体现式视觉导航的对抗鲁棒性至关重要，特别是考虑到3D物理攻击的威胁可能对人类安全构成风险。然而，由于将数字扰动转移到物理世界中的挑战，现有的嵌入式视觉导航攻击方法往往缺乏物理可行性。此外，当前针对对象检测的物理攻击很难在导航场景中实现多视图有效性和视觉自然性。为了解决这个问题，我们提出了一种实用的嵌入式导航攻击方法，通过在对象上附加对抗补丁，其中不透明度和纹理都是可以学习的。具体来说，为了确保不同视角的有效性，我们采用了基于对象感知采样的多视角优化策略，该策略根据导航中使用的基于视觉的感知模型的反馈来优化补丁的纹理。为了使补丁对人类观察者来说不引人注目，我们引入了两阶段不透明度优化机制，其中不透明度在纹理优化后进行微调。实验结果表明，我们的对抗补丁使导航成功率平均降低了22.39%，在实用性、有效性和自然性方面优于之前的方法。代码可访问：https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav



## **7. SHLIME: Foiling adversarial attacks fooling SHAP and LIME**

SHLIME：挫败对抗攻击愚弄SHAP和LIME cs.LG

7 pages, 7 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.11053v1) [paper-pdf](http://arxiv.org/pdf/2508.11053v1)

**Authors**: Sam Chauhan, Estelle Duguet, Karthik Ramakrishnan, Hugh Van Deventer, Jack Kruger, Ranjan Subbaraman

**Abstract**: Post hoc explanation methods, such as LIME and SHAP, provide interpretable insights into black-box classifiers and are increasingly used to assess model biases and generalizability. However, these methods are vulnerable to adversarial manipulation, potentially concealing harmful biases. Building on the work of Slack et al. (2020), we investigate the susceptibility of LIME and SHAP to biased models and evaluate strategies for improving robustness. We first replicate the original COMPAS experiment to validate prior findings and establish a baseline. We then introduce a modular testing framework enabling systematic evaluation of augmented and ensemble explanation approaches across classifiers of varying performance. Using this framework, we assess multiple LIME/SHAP ensemble configurations on out-of-distribution models, comparing their resistance to bias concealment against the original methods. Our results identify configurations that substantially improve bias detection, highlighting their potential for enhancing transparency in the deployment of high-stakes machine learning systems.

摘要: LIME和SHAP等事后解释方法为黑匣子分类器提供了可解释的见解，并越来越多地用于评估模型偏差和可概括性。然而，这些方法很容易受到对抗操纵，可能会掩盖有害偏见。在Slack等人（2020）的工作的基础上，我们研究了LIME和SHAP对有偏见模型的敏感性，并评估了提高稳健性的策略。我们首先复制原始的COPAS实验，以验证先前的发现并建立基线。然后，我们引入了一个模块化测试框架，能够对不同性能的分类器的增强和集成解释方法进行系统评估。使用这个框架，我们评估了非分布模型上的多个LIME/SHAP集合配置，将它们对偏差隐藏的抵抗力与原始方法进行比较。我们的结果确定了可以大幅改善偏见检测的配置，凸显了它们在提高高风险机器学习系统部署透明度方面的潜力。



## **8. Byzantine-Resilient Decentralized Online Resource Allocation**

具有拜占庭弹性的去中心化在线资源分配 math.OC

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.08658v2) [paper-pdf](http://arxiv.org/pdf/2508.08658v2)

**Authors**: Runhua Wang, Qing Ling, Hoi-To Wai, Zhi Tian

**Abstract**: In this paper, we investigate the problem of decentralized online resource allocation in the presence of Byzantine attacks. In this problem setting, some agents may be compromised due to external manipulations or internal failures, causing them to behave maliciously and disrupt the resource allocation process by sending incorrect messages to their neighbors. Given the non-consensual nature of the resource allocation problem, we formulate it under a primal-dual optimization framework, where the dual variables are aggregated among the agents, enabling the incorporation of robust aggregation mechanisms to mitigate Byzantine attacks. By leveraging the classical Byzantine attack model, we propose a class of Byzantine-resilient decentralized online resource allocation algorithms that judiciously integrate the adaptive robust clipping technique with the existing robust aggregation rules to filter out adversarial messages. We establish theoretical guarantees, showing that the proposed algorithms achieve tight linear dynamic regret and accumulative constraint violation bounds, where the constants depend on the properties of robust aggregation rules. Numerical experiments on decentralized online economic dispatch validate the effectiveness of our approach and support our theoretical results.

摘要: 在本文中，我们研究的问题，分散在线资源分配存在拜占庭攻击。在此问题设置中，一些代理可能会由于外部操纵或内部故障而受到损害，导致它们恶意行为并通过向其邻居发送错误消息来破坏资源分配过程。考虑到资源分配问题的非一致性，我们将其制定在原始-对偶优化框架下，其中对偶变量在代理之间聚合，从而能够采用强大的聚合机制来减轻拜占庭攻击。通过利用经典的拜占庭攻击模型，我们提出了一类拜占庭弹性分散在线资源分配算法，明智地整合了自适应鲁棒裁剪技术与现有的强大的聚合规则，以过滤掉敌对的消息。我们建立了理论保证，表明提出的算法实现了严格的线性动态遗憾和累积约束违反界限，其中的常数取决于鲁棒聚集规则的属性。分散式在线经济调度的数值实验验证了我们方法的有效性并支持我们的理论结果。



## **9. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

JMA：一种生成近乎最佳目标对抗示例的通用算法 cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2401.01199v2) [paper-pdf](http://arxiv.org/pdf/2401.01199v2)

**Authors**: Benedetta Tondi, Wei Guo, Niccolò Pancino, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, a more general, theoretically sound, targeted attack is proposed, which resorts to the minimization of a Jacobian-induced Mahalanobis distance term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm (referred to as JMA) provides an optimal solution to a linearised version of the adversarial example problem originally introduced by Szegedy et al. The results of the experiments confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, JMA is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in complex multi-label classification scenarios, a capability that is out of reach of all the attacks proposed so far. As a further advantage, JMA requires very few iterations, thus resulting more efficient than existing methods.

摘要: 到目前为止，针对深度学习分类器设计有针对性的对抗性示例而提出的大多数方法都是高度次优的，并且通常依赖于增加目标类的可能性，从而隐含地专注于单一热编码设置。本文提出了一种更一般、理论上合理的有针对性的攻击，该攻击采用Jacobian诱导的Mahalanobis距离项的最小化，同时考虑到在给定方向上移动输入样本的潜在空间表示所需的努力（在输入空间中）。最小化是通过利用沃尔夫二元定理来解决的，将问题简化为非负最小平方（NNLS）问题的解。所提出的算法（称为JMA）为Szegedy等人最初引入的对抗性示例问题的线性化版本提供了最佳解决方案。实验结果证实了所提出的攻击的一般性，该攻击被证明在各种输出编码方案下有效。值得注意的是，JMA在多标签分类场景中也很有效，能够在复杂的多标签分类场景中诱导多达一半标签的有针对性的修改，这是迄今为止提出的所有攻击所无法实现的能力。作为另一个优势，JMA需要很少的迭代，因此比现有方法更高效。



## **10. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

MCP-Guard：大型语言模型应用中模型上下文协议完整性的防御框架 cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10991v1) [paper-pdf](http://arxiv.org/pdf/2508.10991v1)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.

摘要: 通过模型上下文协议（HCP）等协议将大型语言模型（LLM）与外部工具集成会引入严重的安全漏洞，包括提示注入、数据溢出和其他威胁。为了应对这些挑战，我们提出了MCP-Guard，这是一种专为LLM工具交互而设计的稳健、分层的防御架构。MCP-Guard采用三阶段检测管道，平衡效率与准确性：它从针对明显威胁的轻量级静态扫描和针对语义攻击的深度神经检测器，发展到我们微调的基于E5的模型，在识别对抗性提示方面实现了（96.01）的准确性。最后，轻量级LLM仲裁器合成这些信号以做出最终决策，同时最大限度地减少误报。为了促进严格的培训和评估，我们还引入了MCP-AttackBench，这是一个包含超过70，000个样本的综合基准。MCP-AttackBench源自公共数据集，并通过GPT-4进行增强，以HCP格式模拟不同的现实世界攻击载体，为未来研究保护LLM工具生态系统提供基础。



## **11. MirGuard: Towards a Robust Provenance-based Intrusion Detection System Against Graph Manipulation Attacks**

MirGuard：打造一个强大的基于源的入侵检测系统来对抗图形操纵攻击 cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10639v1) [paper-pdf](http://arxiv.org/pdf/2508.10639v1)

**Authors**: Anyuan Sang, Lu Zhou, Li Yang, Junbo Jia, Huipeng Yang, Pengbin Feng, Jianfeng Ma

**Abstract**: Learning-based Provenance-based Intrusion Detection Systems (PIDSes) have become essential tools for anomaly detection in host systems due to their ability to capture rich contextual and structural information, as well as their potential to detect unknown attacks. However, recent studies have shown that these systems are vulnerable to graph manipulation attacks, where attackers manipulate the graph structure to evade detection. While some previous approaches have discussed this type of attack, none have fully addressed it with a robust detection solution, limiting the practical applicability of PIDSes.   To address this challenge, we propose MirGuard, a robust anomaly detection framework that combines logic-aware multi-view augmentation with contrastive representation learning. Rather than applying arbitrary structural perturbations, MirGuard introduces Logic-Aware Noise Injection (LNI) to generate semantically valid graph views, ensuring that all augmentations preserve the underlying causal semantics of the provenance data. These views are then used in a Logic-Preserving Contrastive Learning framework, which encourages the model to learn representations that are invariant to benign transformations but sensitive to adversarial inconsistencies. Comprehensive evaluations on multiple provenance datasets demonstrate that MirGuard significantly outperforms state-of-the-art detectors in robustness against various graph manipulation attacks without sacrificing detection performance and efficiency. Our work represents the first targeted study to enhance PIDS against such adversarial threats, providing a robust and effective solution to modern cybersecurity challenges.

摘要: 基于学习的基于源的入侵检测系统（PIDS）因其捕获丰富的上下文和结构信息的能力以及检测未知攻击的潜力而成为主机系统异常检测的重要工具。然而，最近的研究表明，这些系统很容易受到图操纵攻击，攻击者操纵图结构以逃避检测。虽然之前的一些方法已经讨论过这种类型的攻击，但没有一种方法可以通过强大的检测解决方案完全解决它，从而限制了PIDS的实际适用性。   为了应对这一挑战，我们提出了MirGuard，这是一个强大的异常检测框架，它将逻辑感知的多视图增强与对比表示学习相结合。MirGuard没有应用任意的结构扰动，而是引入逻辑感知噪音注入（LNI）来生成语义有效的图视图，确保所有增强都保留出处数据的底层因果语义。然后，这些视图在逻辑保持对比学习框架中使用，该框架鼓励模型学习对良性转换不变但对对抗不一致敏感的表示。对多个来源数据集的综合评估表明，MirGuard在对抗各种图形操纵攻击的鲁棒性方面明显优于最先进的检测器，而不会牺牲检测性能和效率。我们的工作是第一项旨在增强PIDS对抗此类对抗威胁的有针对性的研究，为现代网络安全挑战提供强大而有效的解决方案。



## **12. Towards Powerful and Practical Patch Attacks for 2D Object Detection in Autonomous Driving**

针对自动驾驶中的2D物体检测开发强大且实用的补丁攻击 cs.CV

13 pages, 4 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10600v1) [paper-pdf](http://arxiv.org/pdf/2508.10600v1)

**Authors**: Yuxin Cao, Yedi Zhang, Wentao He, Yifan Liao, Yan Xiao, Chang Li, Zhiyong Huang, Jin Song Dong

**Abstract**: Learning-based autonomous driving systems remain critically vulnerable to adversarial patches, posing serious safety and security risks in their real-world deployment. Black-box attacks, notable for their high attack success rate without model knowledge, are especially concerning, with their transferability extensively studied to reduce computational costs compared to query-based attacks. Previous transferability-based black-box attacks typically adopt mean Average Precision (mAP) as the evaluation metric and design training loss accordingly. However, due to the presence of multiple detected bounding boxes and the relatively lenient Intersection over Union (IoU) thresholds, the attack effectiveness of these approaches is often overestimated, resulting in reduced success rates in practical attacking scenarios. Furthermore, patches trained on low-resolution data often fail to maintain effectiveness on high-resolution images, limiting their transferability to autonomous driving datasets. To fill this gap, we propose P$^3$A, a Powerful and Practical Patch Attack framework for 2D object detection in autonomous driving, specifically optimized for high-resolution datasets. First, we introduce a novel metric, Practical Attack Success Rate (PASR), to more accurately quantify attack effectiveness with greater relevance for pedestrian safety. Second, we present a tailored Localization-Confidence Suppression Loss (LCSL) to improve attack transferability under PASR. Finally, to maintain the transferability for high-resolution datasets, we further incorporate the Probabilistic Scale-Preserving Padding (PSPP) into the patch attack pipeline as a data preprocessing step. Extensive experiments show that P$^3$A outperforms state-of-the-art attacks on unseen models and unseen high-resolution datasets, both under the proposed practical IoU-based evaluation metric and the previous mAP-based metrics.

摘要: 基于学习的自动驾驶系统仍然非常容易受到对抗性补丁的攻击，在其实际部署中构成严重的安全风险。黑盒攻击以其在没有模型知识的情况下的高攻击成功率而闻名，尤其令人关注，与基于查询的攻击相比，它们的可转移性被广泛研究以减少计算成本。以往的基于可传递性的黑盒攻击通常采用平均精度（mAP）作为评估指标，并相应地设计训练损失。然而，由于存在多个检测到的边界框和相对宽松的交并（IoU）阈值，这些方法的攻击有效性往往被高估，导致在实际攻击场景中的成功率降低。此外，在低分辨率数据上训练的补丁通常无法在高分辨率图像上保持有效性，从而限制了它们对自动驾驶数据集的可移植性。为了填补这一空白，我们提出了P$^3$A，这是一个强大而实用的补丁攻击框架，用于自动驾驶中的2D对象检测，专门针对高分辨率数据集进行了优化。首先，我们引入了一个新的指标，实际攻击成功率（PASR），更准确地量化攻击的有效性与行人的安全更大的相关性。其次，我们提出了一个定制的本地化置信抑制损失（LCSL），以提高PASR下的攻击转移性。最后，为了保持高分辨率数据集的可移植性，我们进一步将概率规模保持填充（PSPP）纳入补丁攻击管道中，作为数据预处理步骤。大量实验表明，无论是在提出的实用的基于IoU的评估指标还是之前的基于mAP的指标下，P $& 3$A都优于对未见模型和未见高分辨率数据集的最新攻击。



## **13. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

两阶段学习推迟中的对抗稳健性：算法和保证 stat.ML

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2502.01027v3) [paper-pdf](http://arxiv.org/pdf/2502.01027v3)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.

摘要: 两阶段学习延迟（L2 D）通过将每个输入分配给固定的主模型或多个离线专家之一来实现最佳任务委托，支持复杂的多代理环境中的可靠决策。然而，现有的L2 D框架假设干净的输入，并且容易受到可以操纵查询分配的对抗性扰动的影响，从而导致代价高昂的错误路由或专家超载。我们首次对两阶段L2 D系统中的对抗鲁棒性进行了全面研究。我们引入了两种新颖的攻击策略--无针对性和有针对性--它们分别扰乱最佳分配或强制向特定代理进行查询。为了抵御此类威胁，我们提出了SAARD，这是一种凸学习算法，它建立在一系列可证明Bayes-一致且$（\mathCal{R}，\mathCal{G}）$-一致的替代损失之上。这些保证适用于分类、回归和多任务设置。经验结果表明，SAARD显着提高了对抗攻击下的鲁棒性，同时保持了强大的清洁性能，标志着迈向安全且值得信赖的L2 D部署的关键一步。



## **14. Contrastive ECOC: Learning Output Codes for Adversarial Defense**

对比ECOC：对抗性防御的学习输出代码 cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10491v1) [paper-pdf](http://arxiv.org/pdf/2508.10491v1)

**Authors**: Che-Yu Chou, Hung-Hsuan Chen

**Abstract**: Although one-hot encoding is commonly used for multiclass classification, it is not always the most effective encoding mechanism. Error Correcting Output Codes (ECOC) address multiclass classification by mapping each class to a unique codeword used as a label. Traditional ECOC methods rely on manually designed or randomly generated codebooks, which are labor-intensive and may yield suboptimal, dataset-agnostic results. This paper introduces three models for automated codebook learning based on contrastive learning, allowing codebooks to be learned directly and adaptively from data. Across four datasets, our proposed models demonstrate superior robustness to adversarial attacks compared to two baselines. The source is available at https://github.com/YuChou20/Automated-Codebook-Learning-with-Error-Correcting-Output-Code-Technique.

摘要: 尽管一热编码通常用于多类分类，但它并不总是最有效的编码机制。错误纠正输出代码（ECOC）通过将每个类映射到用作标签的唯一代码字来解决多类分类。传统的ECOC方法依赖于手动设计或随机生成的码本，这是劳动密集型的，并且可能会产生次优的、与厕所无关的结果。本文介绍了三种基于对比学习的自动码本学习模型，允许从数据中直接自适应地学习码本。在四个数据集中，与两个基线相比，我们提出的模型表现出了对对抗攻击的卓越鲁棒性。该来源可访问https://github.com/YuChou20/Automated-Codebook-Learning-with-Error-Correcting-Output-Code-Technique。



## **15. Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation**

通过稀疏自动编码器进行分层扰动以生成对抗性文本 cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10404v1) [paper-pdf](http://arxiv.org/pdf/2508.10404v1)

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP systems.However, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated.

摘要: 随着自然语言处理（NLP），尤其是大型语言模型（LLM）的迅速普及，生成对抗性示例以越狱LLM仍然是理解模型漏洞和提高稳健性的关键挑战。在此背景下，我们提出了一种新的黑匣子攻击方法，该方法利用了大型模型的可解释性。我们引入了稀疏特征扰动框架（SPF），这是一种对抗性文本生成的新颖方法，利用稀疏自动编码器来识别和操纵文本中的关键特征。在使用SAGE模型重建隐藏层表示后，我们对成功攻击的文本执行特征集群，以识别激活程度较高的特征。然后，这些高度激活的特征被扰动以生成新的对抗文本。这种选择性干扰在放大安全信号的同时保留了恶意意图，从而增加了它们逃避现有防御的可能性。我们的方法实现了一种新的红色团队策略，该策略平衡了对抗有效性与安全一致。实验结果表明，SFPF生成的对抗性文本可以绕过最先进的防御机制，揭示当前NLP系统中的持久漏洞。然而，该方法的有效性因提示和层而异，其对其他架构和更大模型的推广性仍有待验证。



## **16. BERTector: Intrusion Detection Based on Joint-Dataset Learning**

BERTector：基于联合数据集学习的入侵检测 cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10327v1) [paper-pdf](http://arxiv.org/pdf/2508.10327v1)

**Authors**: Haoyang Hu, Xun Huang, Chenyu Wu, Shiwen Liu, Zhichao Lian, Shuangquan Zhang

**Abstract**: Intrusion detection systems (IDS) are facing challenges in generalization and robustness due to the heterogeneity of network traffic and the diversity of attack patterns. To address this issue, we propose a new joint-dataset training paradigm for IDS and propose a scalable BERTector framework based on BERT. BERTector integrates three key components: NSS-Tokenizer for traffic-aware semantic tokenization, supervised fine-tuning with a hybrid dataset, and low-rank adaptation (LoRA) for efficient training. Extensive experiments show that BERTector achieves state-of-the-art detection accuracy, strong cross-dataset generalization capabilities, and excellent robustness to adversarial perturbations. This work establishes a unified and efficient solution for modern IDS in complex and dynamic network environments.

摘要: 由于网络流量的异构性和攻击模式的多样性，入侵检测系统在通用性和鲁棒性方面面临挑战。为了解决这个问题，我们提出了一个新的联合数据集训练的IDS范例，并提出了一个可扩展的BERT框架的基础上BERT。BERTector集成了三个关键组件：用于流量感知语义令牌化的NSS-Tokenizer，混合数据集的监督微调，以及用于有效训练的低秩自适应（LoRA）。大量的实验表明，BERTector实现了最先进的检测精度，强大的跨数据集泛化能力，以及对对抗性扰动的出色鲁棒性。这项工作为复杂动态网络环境中的现代IDS建立了统一有效的解决方案。



## **17. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

用于增强对抗性可转移性的语义结构感知生成攻击 cs.CV

Preprint

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2506.18248v4) [paper-pdf](http://arxiv.org/pdf/2506.18248v4)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).

摘要: 生成性对抗攻击在白盒代理模型上训练扰动生成器，然后将精心设计的扰动应用于不可见的黑匣子受害者模型。与迭代攻击相比，这些方法提供了卓越的推理时效率、可扩展性和可移植性;然而，到目前为止，现有的研究尚未充分利用生成模型的表示能力来保存和利用语义信息。具体来说，生成器的中间激活编码了丰富的语义特征--对象边界和粗糙形状--这些特征仍然没有得到充分利用，从而限制了扰动与对象突出区域的对齐，而这些区域对于对抗性的可转移性至关重要。为了解决这个问题，我们引入了一个基于Mean Teacher的语义结构感知攻击框架，该框架充当时间平滑的特征参考。通过这个平滑的引用，我们通过特征提炼进一步指导学生的早期层激活和语义丰富的教师的早期层激活之间的语义一致性。通过基于经验发现将扰动合成锚定到生成器内语义突出的早期中间块，我们的方法引导对区域进行渐进对抗扰动，从而大大增强对抗转移性。我们对不同的模型、领域和任务进行了广泛的实验，以展示相对于最先进的生成性攻击的一致改进，并使用传统指标和我们新提出的意外纠正率（OCR）进行了全面评估。



## **18. PromptSafe: Gated Prompt Tuning for Safe Text-to-Image Generation**

EntSafe：门控提示调优，用于安全的文本到图像生成 cs.CV

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.01272v2) [paper-pdf](http://arxiv.org/pdf/2508.01272v2)

**Authors**: Zonglei Jing, Xiao Yang, Xiaoqian Li, Siyuan Liang, Aishan Liu, Mingchuan Zhang, Xianglong Liu

**Abstract**: Text-to-image (T2I) models have demonstrated remarkable generative capabilities but remain vulnerable to producing not-safe-for-work (NSFW) content, such as violent or explicit imagery. While recent moderation efforts have introduced soft prompt-guided tuning by appending defensive tokens to the input, these approaches often rely on large-scale curated image-text datasets and apply static, one-size-fits-all defenses at inference time. However, this results not only in high computational cost and degraded benign image quality, but also in limited adaptability to the diverse and nuanced safety requirements of real-world prompts. To address these challenges, we propose PromptSafe, a gated prompt tuning framework that combines a lightweight, text-only supervised soft embedding with an inference-time gated control network. Instead of training on expensive image-text datasets, we first rewrite unsafe prompts into semantically aligned but safe alternatives using an LLM, constructing an efficient text-only training corpus. Based on this, we optimize a universal soft prompt that repels unsafe and attracts safe embeddings during the diffusion denoising process. To avoid over-suppressing benign prompts, we introduce a gated mechanism that adaptively adjusts the defensive strength based on estimated prompt toxicity, thereby aligning defense intensity with prompt risk and ensuring strong protection for harmful inputs while preserving benign generation quality. Extensive experiments across multiple benchmarks and T2I models show that PromptSafe achieves a SOTA unsafe generation rate (2.36%), while preserving high benign fidelity. Furthermore, PromptSafe demonstrates strong generalization to unseen harmful categories, robust transferability across diffusion model architectures, and resilience under adaptive adversarial attacks, highlighting its practical value for safe and scalable deployment.

摘要: 文本到图像（T2 I）模型已表现出非凡的生成能力，但仍然容易产生不安全的工作（NSFW）内容，例如暴力或露骨的图像。虽然最近的审核工作通过在输入中添加防御令牌来引入了软预算引导的调优，但这些方法通常依赖于大规模精心策划的图像文本数据集，并在推理时应用静态的、一刀切的防御。然而，这不仅导致计算成本高和良性图像质量下降，而且对现实世界提示的多样化和细致入微的安全要求的适应性有限。为了应对这些挑战，我们提出了Inbox Safe，这是一种门控提示调整框架，它将轻量级的、纯文本的监督软嵌入与推断时间门控控制网络相结合。我们没有在昂贵的图像-文本数据集上进行训练，而是首先使用LLM将不安全的提示重写为语义对齐但安全的替代方案，构建一个高效的纯文本训练库。在此基础上，我们优化了一个通用软提示，该提示可以在扩散去噪过程中排斥不安全并吸引安全嵌入。为了避免过度抑制良性提示，我们引入了一种门控机制，该机制根据估计的提示毒性自适应地调整防御强度，从而使防御强度与提示风险保持一致，并确保对有害输入的强有力保护，同时保持良性发电质量。跨多个基准测试和T2 I模型的广泛实验表明，InspectSafe实现了SOTA不安全的生成率（2.36%），同时保持了高的良性保真度。此外，EntSafe表现出对不可见有害类别的强大概括性、跨扩散模型架构的强大可移植性以及适应性对抗攻击下的弹性，凸显了其安全和可扩展部署的实用价值。



## **19. Detecting Untargeted Attacks and Mitigating Unreliable Updates in Federated Learning for Underground Mining Operations**

在地下采矿作业的联邦学习中检测无目标攻击并缓解不可靠的更新 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10212v1) [paper-pdf](http://arxiv.org/pdf/2508.10212v1)

**Authors**: Md Sazedur Rahman, Mohamed Elmahallawy, Sanjay Madria, Samuel Frimpong

**Abstract**: Underground mining operations rely on distributed sensor networks to collect critical data daily, including mine temperature, toxic gas concentrations, and miner movements for hazard detection and operational decision-making. However, transmitting raw sensor data to a central server for training deep learning models introduces significant privacy risks, potentially exposing sensitive mine-specific information. Federated Learning (FL) offers a transformative solution by enabling collaborative model training while ensuring that raw data remains localized at each mine. Despite its advantages, FL in underground mining faces key challenges: (i) An attacker may compromise a mine's local model by employing techniques such as sign-flipping attacks or additive noise, leading to erroneous predictions; (ii) Low-quality (yet potentially valuable) data, caused by poor lighting conditions or sensor inaccuracies in mines may degrade the FL training process. In response, this paper proposes MineDetect, a defense FL framework that detects and isolates the attacked models while mitigating the impact of mines with low-quality data. MineDetect introduces two key innovations: (i) Detecting attacked models (maliciously manipulated) by developing a history-aware mechanism that leverages local and global averages of gradient updates; (ii) Identifying and eliminating adversarial influences from unreliable models (generated by clients with poor data quality) on the FL training process. Comprehensive simulations across diverse datasets demonstrate that MineDetect outperforms existing methods in both robustness and accuracy, even in challenging non-IID data scenarios. Its ability to counter adversarial influences while maintaining lower computational efficiency makes it a vital advancement for improving safety and operational effectiveness in underground mining.

摘要: 地下采矿作业依赖分布式传感器网络每天收集关键数据，包括矿井温度、有毒气体浓度和矿工移动，以进行危险检测和运营决策。然而，将原始传感器数据传输到中央服务器以训练深度学习模型会带来重大的隐私风险，可能会暴露敏感的矿山特定信息。联合学习（FL）通过实现协作模型训练，同时确保原始数据在每个矿山保持本地化，提供了一种变革性的解决方案。尽管具有优势，地下采矿中的FL仍面临着关键挑战：（i）攻击者可能会通过使用符号翻转攻击或添加性噪音等技术来损害矿井的本地模型，从而导致错误的预测;（ii）由于矿井中光线条件不佳或传感器不准确而导致的低质量（但潜在有价值）数据可能会降低FL训练过程。作为回应，本文提出了MineDetect，这是一个防御FL框架，可以检测和隔离受攻击的模型，同时减轻低质量数据地雷的影响。MineDetect引入了两项关键创新：（i）通过开发利用局部和全球梯度更新平均值的历史感知机制来检测受攻击的模型（恶意操纵）;（ii）识别和消除来自FL训练过程中不可靠模型（由数据质量较差的客户生成）的对抗影响。跨不同数据集的全面模拟表明，即使在具有挑战性的非IID数据场景中，MineDetect在稳健性和准确性方面都优于现有方法。它能够对抗敌对影响，同时保持较低的计算效率，这使其成为提高地下采矿安全性和运营有效性的重要进步。



## **20. An Architecture for Distributed Digital Identities in the Physical World**

物理世界中分布式数字身份的架构 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10185v1) [paper-pdf](http://arxiv.org/pdf/2508.10185v1)

**Authors**: René Mayrhofer, Michael Roland, Tobias Höller, Philipp Hofer, Mario Lins

**Abstract**: Digital identities are increasingly important for mediating not only digital but also physical service transactions. Managing such identities through centralized providers can cause both availability and privacy concerns: single points of failure and control are ideal targets for global attacks on technical, organizational, or legal fronts. We design, analyze, and build a distributed digital identity architecture for physical world transactions in common scenarios like unlocking doors, public transport, or crossing country borders. This architecture combines (biometric and other) sensors, (established and upcoming) identity authorities, attribute verifiers, and a new core component we call the \emph{Personal Identity Agent (PIA)} that represents individuals with their identity attributes in the digital domain. All transactions are conducted in a completely decentralized manner, and the components for which we currently assume central coordination are optional and only used for assisting with service discovery and latency reduction. We present a first protocol between these parties and formally verify that it achieves relevant security properties based on a realistic threat model including strong global adversaries. A proof-of-concept implementation demonstrates practical feasibility of both architecture and initial protocol for applications that can tolerate end-to-end latencies in the range of a few seconds.

摘要: 数字身份不仅对于调解数字服务交易而且对于调解物理服务交易越来越重要。通过集中式提供商管理此类身份可能会导致可用性和隐私问题：单点故障和控制是技术、组织或法律方面全球攻击的理想目标。我们设计、分析和构建分布式数字身份架构，用于解锁门、公共交通或跨越国界等常见场景中的物理世界交易。该架构结合了（生物识别和其他）传感器、（已建立和即将推出的）身份权威机构、属性验证器以及我们称为\{个人身份代理（PIA）}的新核心组件，该组件在数字域中代表具有身份属性的个人。所有事务都以完全分散的方式进行，我们目前假设的中央协调组件是可选的，仅用于协助服务发现和延迟减少。我们提出了这些缔约方之间的第一个协议，并正式验证，它实现了相关的安全属性的基础上，包括强大的全球对手的现实威胁模型。一个概念验证的实现证明了实际的可行性架构和初始协议的应用程序，可以容忍端到端延迟在几秒钟的范围内。



## **21. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全问题：调查 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2505.18889v3) [paper-pdf](http://arxiv.org/pdf/2505.18889v3)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: ChatGPT及其竞争对手等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。这项调查全面概述了这些新出现的问题，将威胁分为几个关键领域：即时注入和越狱;对抗性攻击，包括输入干扰和数据中毒;恶意行为者滥用信息、网络钓鱼电子邮件和恶意软件;以及自主LLM代理固有的令人担忧的风险。最近，人们越来越关注后者，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标的潜力，这种行为被称为阴谋，甚至可以通过安全培训持续存在。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **22. Perturbed Public Voices (P$^{2}$V): A Dataset for Robust Audio Deepfake Detection**

受干扰的公共声音（P$^{2}$V）：用于稳健音频深度伪造检测的数据集 cs.SD

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10949v1) [paper-pdf](http://arxiv.org/pdf/2508.10949v1)

**Authors**: Chongyang Gao, Marco Postiglione, Isabel Gortner, Sarit Kraus, V. S. Subrahmanian

**Abstract**: Current audio deepfake detectors cannot be trusted. While they excel on controlled benchmarks, they fail when tested in the real world. We introduce Perturbed Public Voices (P$^{2}$V), an IRB-approved dataset capturing three critical aspects of malicious deepfakes: (1) identity-consistent transcripts via LLMs, (2) environmental and adversarial noise, and (3) state-of-the-art voice cloning (2020-2025). Experiments reveal alarming vulnerabilities of 22 recent audio deepfake detectors: models trained on current datasets lose 43% performance when tested on P$^{2}$V, with performance measured as the mean of F1 score on deepfake audio, AUC, and 1-EER. Simple adversarial perturbations induce up to 16% performance degradation, while advanced cloning techniques reduce detectability by 20-30%. In contrast, P$^{2}$V-trained models maintain robustness against these attacks while generalizing to existing datasets, establishing a new benchmark for robust audio deepfake detection. P$^{2}$V will be publicly released upon acceptance by a conference/journal.

摘要: 当前的音频Deepfake检测器无法信任。虽然它们在受控基准上表现出色，但在现实世界中测试时却失败了。我们引入了Perturbed Public Voices（P$^{2}$V），这是一个经过IRC批准的数据集，捕捉恶意深度造假的三个关键方面：（1）通过LLM的身份一致文字记录，（2）环境和对抗性噪音，以及（3）最先进的语音克隆（2020-2025）。实验揭示了22个最近的音频Deepfake检测器的令人震惊的漏洞：在P$##{2}$V上测试时，在当前数据集上训练的模型性能损失了43%，性能以Deepfake音频、AUR和1-EER的F1得分的平均值来衡量。简单的对抗性扰动会导致高达16%的性能下降，而先进的克隆技术会使检测能力降低20- 30%。相比之下，P$^{2}$V训练模型在推广到现有数据集的同时保持了针对这些攻击的鲁棒性，为稳健的音频深度伪造检测建立了新的基准。P$^{2}$V将在会议/期刊接受后公开发布。



## **23. IPG: Incremental Patch Generation for Generalized Adversarial Patch Training**

IPG：用于广义对抗补丁训练的增量补丁生成 cs.CV

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10946v1) [paper-pdf](http://arxiv.org/pdf/2508.10946v1)

**Authors**: Wonho Lee, Hyunsik Na, Jisu Lee, Daeseon Choi

**Abstract**: The advent of adversarial patches poses a significant challenge to the robustness of AI models, particularly in the domain of computer vision tasks such as object detection. In contradistinction to traditional adversarial examples, these patches target specific regions of an image, resulting in the malfunction of AI models. This paper proposes Incremental Patch Generation (IPG), a method that generates adversarial patches up to 11.1 times more efficiently than existing approaches while maintaining comparable attack performance. The efficacy of IPG is demonstrated by experiments and ablation studies including YOLO's feature distribution visualization and adversarial training results, which show that it produces well-generalized patches that effectively cover a broader range of model vulnerabilities. Furthermore, IPG-generated datasets can serve as a robust knowledge foundation for constructing a robust model, enabling structured representation, advanced reasoning, and proactive defenses in AI security ecosystems. The findings of this study suggest that IPG has considerable potential for future utilization not only in adversarial patch defense but also in real-world applications such as autonomous vehicles, security systems, and medical imaging, where AI models must remain resilient to adversarial attacks in dynamic and high-stakes environments.

摘要: 对抗补丁的出现对人工智能模型的稳健性提出了重大挑战，特别是在物体检测等计算机视觉任务领域。与传统的对抗示例不同，这些补丁针对图像的特定区域，导致人工智能模型出现故障。本文提出了增量补丁生成（IMG），这是一种生成对抗补丁的方法，其效率比现有方法高出11.1倍，同时保持相当的攻击性能。TIG的功效通过实验和消融研究得到了证明，包括YOLO的特征分布可视化和对抗训练结果，这些研究表明它可以生成广泛通用的补丁，有效地覆盖更广泛的模型漏洞。此外，TIG生成的数据集可以作为构建稳健模型的稳健知识基础，实现人工智能安全生态系统中的结构化表示、高级推理和主动防御。这项研究的结果表明，IMG未来不仅在对抗性补丁防御方面具有相当大的利用潜力，而且在自动驾驶汽车、安全系统和医学成像等现实世界应用中也具有相当大的应用潜力，其中人工智能模型必须在动态和高风险环境中保持对对抗性攻击的弹性。



## **24. MetaGuardian: Enhancing Voice Assistant Security through Advanced Acoustic Metamaterials**

MetaGuardian：通过先进的声学超材料增强语音助手的安全性 cs.SD

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09728v1) [paper-pdf](http://arxiv.org/pdf/2508.09728v1)

**Authors**: Zhiyuan Ning, Zheng Wang, Zhanyong Tang

**Abstract**: We present MetaGuardian, a voice assistant (VA) protection system based on acoustic metamaterials. MetaGuardian can be directly integrated into the enclosures of various smart devices, effectively defending against inaudible, adversarial and laser attacks without relying on additional software support or altering the underlying hardware, ensuring usability. To achieve this, MetaGuardian leverages the mutual impedance effects between metamaterial units to extend the signal filtering range to 16-40 kHz to effectively block wide-band inaudible attacks. Additionally, it adopts a carefully designed coiled space structure to precisely interfere with adversarial attacks while ensuring the normal functioning of VAs. Furthermore, MetaGuardian offers a universal structural design, allowing itself to be flexibly adapted to various smart devices, striking a balance between portability and protection effectiveness. In controled evaluation environments, MetaGuardian achieves a high defense success rate against various attack types, including adversarial, inaudible and laser attacks.

摘要: 我们介绍了MetaGuardian，这是一种基于声学超材料的语音助理（VA）保护系统。MetaGuardian可以直接集成到各种智能设备的外壳中，有效防御听不见的、对抗性的和激光攻击，而无需依赖额外的软件支持或更改底层硬件，确保可用性。为了实现这一目标，MetaGuardian利用超材料单元之间的互抗效应将信号过滤范围扩展至16-40 GHz，以有效阻止宽带听不见的攻击。此外，它采用精心设计的螺旋空间结构，精确干扰对抗攻击，同时确保VA的正常运作。此外，MetaGuardian提供了通用的结构设计，使其能够灵活地适应各种智能设备，在便携性和保护有效性之间取得平衡。在受控的评估环境中，MetaGuardian对各种攻击类型（包括对抗性攻击、听不见的攻击和激光攻击）实现了高防御成功率。



## **25. MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs**

MetaCipher：针对LLM的基于密码的越狱攻击的持续时间和通用多代理框架 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2506.22557v2) [paper-pdf](http://arxiv.org/pdf/2506.22557v2)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: As large language models (LLMs) grow more capable, they face growing vulnerability to sophisticated jailbreak attacks. While developers invest heavily in alignment finetuning and safety guardrails, researchers continue publishing novel attacks, driving progress through adversarial iteration. This dynamic mirrors a strategic game of continual evolution. However, two major challenges hinder jailbreak development: the high cost of querying top-tier LLMs and the short lifespan of effective attacks due to frequent safety updates. These factors limit cost-efficiency and practical impact of research in jailbreak attacks. To address this, we propose MetaCipher, a low-cost, multi-agent jailbreak framework that generalizes across LLMs with varying safety measures. Using reinforcement learning, MetaCipher is modular and adaptive, supporting extensibility to future strategies. Within as few as 10 queries, MetaCipher achieves state-of-the-art attack success rates on recent malicious prompt benchmarks, outperforming prior jailbreak methods. We conduct a large-scale empirical evaluation across diverse victim models and benchmarks, demonstrating its robustness and adaptability. Warning: This paper contains model outputs that may be offensive or harmful, shown solely to demonstrate jailbreak efficacy.

摘要: 随着大型语言模型（LLM）的能力变得越来越强，它们面临着越来越容易受到复杂越狱攻击的脆弱性。虽然开发人员在对齐微调和安全护栏上投入巨资，但研究人员继续发布新颖的攻击，通过对抗迭代推动进展。这种动态反映了一场持续进化的战略游戏。然而，有两个主要挑战阻碍了越狱的发展：查询顶级LLM的高成本以及由于频繁的安全更新而导致有效攻击的寿命短。这些因素限制了越狱攻击研究的成本效率和实际影响。为了解决这个问题，我们提出了MetaCipher，这是一种低成本、多代理越狱框架，可在具有不同安全措施的LLM中进行推广。使用强化学习，MetaCipher具有模块化和自适应性，支持未来策略的可扩展性。在短短10个查询内，MetaCipher就在最近的恶意提示基准上实现了最先进的攻击成功率，优于之前的越狱方法。我们对不同的受害者模型和基准进行了大规模的实证评估，展示了其稳健性和适应性。警告：本文包含可能令人反感或有害的模型输出，仅用于证明越狱功效。



## **26. LLM Robustness Leaderboard v1 --Technical report**

LLM稳健排行榜v1 --技术报告 cs.AI

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.06296v2) [paper-pdf](http://arxiv.org/pdf/2508.06296v2)

**Authors**: Pierre Peigné - Lefebvre, Quentin Feuillade-Montixi, Tom David, Nicolas Miailhe

**Abstract**: This technical report accompanies the LLM robustness leaderboard published by PRISM Eval for the Paris AI Action Summit. We introduce PRISM Eval Behavior Elicitation Tool (BET), an AI system performing automated red-teaming through Dynamic Adversarial Optimization that achieves 100% Attack Success Rate (ASR) against 37 of 41 state-of-the-art LLMs. Beyond binary success metrics, we propose a fine-grained robustness metric estimating the average number of attempts required to elicit harmful behaviors, revealing that attack difficulty varies by over 300-fold across models despite universal vulnerability. We introduce primitive-level vulnerability analysis to identify which jailbreaking techniques are most effective for specific hazard categories. Our collaborative evaluation with trusted third parties from the AI Safety Network demonstrates practical pathways for distributed robustness assessment across the community.

摘要: 该技术报告随PRism Eval为巴黎人工智能行动峰会发布的LLM稳健性排行榜一起发布。我们引入了PRism Eval Behavior Elicitation Tools（BET），这是一个通过动态对抗优化执行自动化红色团队的人工智能系统，可针对41个最先进的LLM中的37个实现100%的攻击成功率（ASB）。除了二元成功指标之外，我们还提出了一个细粒度的稳健性指标，估计引发有害行为所需的平均尝试次数，揭示了尽管存在普遍的漏洞，但不同模型的攻击难度差异超过300倍。我们引入危险级别漏洞分析，以确定哪些越狱技术对特定危险类别最有效。我们与来自AI安全网络的可信第三方的合作评估展示了在整个社区进行分布式鲁棒性评估的实用途径。



## **27. Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation of LLM**

监护人和罪犯：关于LLM有害内容生成和安全缓解的调查 cs.CL

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.05775v2) [paper-pdf](http://arxiv.org/pdf/2508.05775v2)

**Authors**: Chi Zhang, Changjia Zhu, Junjie Xiong, Xiaoran Xu, Lingyao Li, Yao Liu, Zhuo Lu

**Abstract**: Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies.

摘要: 大型语言模型（LLM）彻底改变了数字平台上的内容创建，在自然语言生成和理解方面提供了前所未有的能力。这些模型支持有益的应用程序，如内容生成、问答（Q&A）、编程和代码推理。与此同时，它们也会因无意或故意产生有毒、攻击性或有偏见的内容而构成严重风险。LLM的这种双重角色，既作为解决现实世界问题的强大工具，又作为有害语言的潜在来源，提出了一个紧迫的社会技术挑战。在这项调查中，我们系统地回顾了最近的研究，涵盖无意毒性、对抗性越狱攻击和内容审核技术。我们提出了LLM相关伤害和防御的统一分类，分析新兴的多模式和LLM辅助越狱策略，并评估缓解工作，包括人类反馈强化学习（RL HF）、即时工程和安全调整。我们的综合强调了LLM安全性不断变化的格局，确定了当前评估方法的局限性，并概述了未来的研究方向，以指导稳健且符合道德规范的语言技术的开发。



## **28. A Taxonomy of System-Level Attacks on Deep Learning Models in Autonomous Vehicles**

自动驾驶汽车中深度学习模型的系统级攻击分类 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2412.04510v2) [paper-pdf](http://arxiv.org/pdf/2412.04510v2)

**Authors**: Masoud Jamshidiyan Tehrani, Jinhan Kim, Rosmael Zidane Lekeufack Foulefack, Alessandro Marchetto, Paolo Tonella

**Abstract**: The advent of deep learning and its astonishing performance has enabled its usage in complex systems, including autonomous vehicles. On the other hand, deep learning models are susceptible to mispredictions when small, adversarial changes are introduced into their input. Such mis-predictions can be triggered in the real world and can result in a failure of the entire system. In recent years, a growing number of research works have investigated ways to mount attacks against autonomous vehicles that exploit deep learning components. Such attacks are directed toward elements of the environment where these systems operate and their effectiveness is assessed in terms of system-level failures triggered by them. There has been however no systematic attempt to analyze and categorize such attacks. In this paper, we present the first taxonomy of system-level attacks against autonomous vehicles. We constructed our taxonomy by selecting 21 highly relevant papers, then we tagged them with 12 top-level taxonomy categories and several sub-categories. The taxonomy allowed us to investigate the attack features, the most attacked components and systems, the underlying threat models, and the failure chains from input perturbation to system-level failure. We distilled several lessons for practitioners and identified possible directions for future work for researchers.

摘要: 深度学习的出现及其惊人的性能使其能够在复杂系统中使用，包括自动驾驶汽车。另一方面，当深度学习模型的输入中引入微小的对抗性变化时，它们很容易出现误预测。这种错误预测可能会在现实世界中触发，并可能导致整个系统故障。近年来，越来越多的研究工作研究了对利用深度学习组件的自动驾驶汽车发动攻击的方法。此类攻击针对这些系统运行的环境元素，并根据它们触发的系统级故障来评估其有效性。然而，目前还没有系统地尝试对此类攻击进行分析和分类。在本文中，我们提出了针对自动驾驶车辆的系统级攻击的第一个分类。我们通过选择21篇高度相关的论文来构建我们的分类法，然后用12个顶级分类类别和几个子类别对它们进行标记。该分类法使我们能够调查攻击特征、受攻击最严重的组件和系统、底层威胁模型以及从输入扰动到系统级故障的故障链。我们为从业者总结了一些教训，并为研究人员确定了未来工作的可能方向。



## **29. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2409.20002v4) [paper-pdf](http://arxiv.org/pdf/2409.20002v4)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型（LLM）的广泛部署引发了对其推理性能优化的强烈要求。当今服务于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，同时在很大程度上忽视了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道由共享缓存和图形处理器内存分配产生，可以利用这些通道来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了传统计算系统中观察到的安全挑战，凸显了解决LLM服务基础设施中潜在信息泄露的迫切需要。在本文中，我们报告了旨在利用LLM部署中固有的此类时间侧通道的新颖攻击策略，特别针对广泛用于增强LLM推理性能的Key-Value（KV）缓存和语义缓存。我们的方法利用时间测量和分类模型来检测缓存命中，使对手能够高准确地推断私人提示。我们还提出了一种逐令牌搜索算法来有效地恢复缓存中的共享提示前置，展示了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑匣子测试的实验研究表明，此类隐私风险是完全现实的，并会产生重大后果。我们的研究结果强调需要强有力的缓解措施来保护LLM系统免受此类新出现的威胁。



## **30. 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation**

高斯溅射驱动的3D多视点鲁棒物理对抗摄像机生成 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2507.01367v2) [paper-pdf](http://arxiv.org/pdf/2507.01367v2)

**Authors**: Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, Xiaochun Cao

**Abstract**: Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.

摘要: 物理对抗攻击方法暴露了深度神经网络的漏洞，并对自动驾驶等安全关键场景构成重大威胁。与基于补丁的攻击相比，基于伪装的物理攻击是一种更有前途的方法，可以在复杂的物理环境中提供更强的对抗效果。然而，大多数先前的工作依赖于目标对象的网格先验和模拟器构建的虚拟环境，获取这些先验很耗时，并且不可避免地与现实世界不同。此外，由于训练图像中背景的限制，以前的方法常常无法产生多视图鲁棒的对抗伪装，并且往往会陷入次优解决方案。由于这些原因，以前的工作缺乏对抗性的有效性和鲁棒性，在不同的观点和物理环境。我们提出了一个物理攻击框架的基础上3D高斯溅射（3DGS），命名为PGA，它提供了快速和精确的重建与几个图像，以及照片般逼真的渲染能力。我们的框架通过防止高斯之间的相互遮挡和自遮挡，并采用最小-最大优化方法来调整每个视点的成像背景，从而帮助算法过滤掉非鲁棒的对抗性特征，从而进一步增强了跨视图的鲁棒性和对抗性有效性。大量的实验验证了PGA的有效性和优越性。我们的代码可访问：https://github.com/TRLou/PGA。



## **31. Exact Verification of Graph Neural Networks with Incremental Constraint Solving**

增量约束求解的图神经网络精确验证 cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09320v1) [paper-pdf](http://arxiv.org/pdf/2508.09320v1)

**Authors**: Minghao Liu, Chia-Hsuan Lu, Marta Kwiatkowska

**Abstract**: Graph neural networks (GNNs) are increasingly employed in high-stakes applications, such as fraud detection or healthcare, but are susceptible to adversarial attacks. A number of techniques have been proposed to provide adversarial robustness guarantees, but support for commonly used aggregation functions in message-passing GNNs is still lacking. In this paper, we develop an exact (sound and complete) verification method for GNNs to compute guarantees against attribute and structural perturbations that involve edge addition or deletion, subject to budget constraints. Focusing on node classification tasks, our method employs constraint solving with bound tightening, and iteratively solves a sequence of relaxed constraint satisfaction problems while relying on incremental solving capabilities of solvers to improve efficiency. We implement GNNev, a versatile solver for message-passing neural networks, which supports three aggregation functions, sum, max and mean, with the latter two considered here for the first time. Extensive experimental evaluation of GNNev on two standard benchmarks (Cora and CiteSeer) and two real-world fraud datasets (Amazon and Yelp) demonstrates its usability and effectiveness, as well as superior performance compared to existing {exact verification} tools on sum-aggregated node classification tasks.

摘要: 图神经网络（GNN）越来越多地用于欺诈检测或医疗保健等高风险应用，但很容易受到对抗攻击。已经提出了多种技术来提供对抗稳健性保证，但仍然缺乏对消息传递GNN中常用的聚合函数的支持。在本文中，我们为GNN开发了一种精确的（合理且完整的）验证方法，以计算针对涉及边添加或删除的属性和结构扰动的保证，并受预算限制。我们的方法专注于节点分类任务，采用带边界收紧的约束求解，迭代地解决一系列宽松的约束满足问题，同时依靠求解器的增量求解能力来提高效率。我们实现了GNNev，这是一个用于消息传递神经网络的通用求解器，它支持三个聚合函数：sum、max和mean，其中后两个是本文首次考虑。GNNev在两个标准基准测试（Cora和CiteSeer）和两个现实世界的欺诈数据集（Amazon和Yelp）上进行了广泛的实验评估，证明了其可用性和有效性，以及与现有的{精确验证}工具相比具有卓越的性能汇总节点分类任务。



## **32. Constrained Black-Box Attacks Against Multi-Agent Reinforcement Learning**

针对多智能体强化学习的约束黑匣子攻击 cs.LG

Under review in TNNLS

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09275v1) [paper-pdf](http://arxiv.org/pdf/2508.09275v1)

**Authors**: Amine Andam, Jamal Bentahar, Mustapha Hedabou

**Abstract**: Collaborative multi-agent reinforcement learning (c-MARL) has rapidly evolved, offering state-of-the-art algorithms for real-world applications, including sensitive domains. However, a key challenge to its widespread adoption is the lack of a thorough investigation into its vulnerabilities to adversarial attacks. Existing work predominantly focuses on training-time attacks or unrealistic scenarios, such as access to policy weights or the ability to train surrogate policies. In this paper, we investigate new vulnerabilities under more realistic and constrained conditions, assuming an adversary can only collect and perturb the observations of deployed agents. We also consider scenarios where the adversary has no access at all. We propose simple yet highly effective algorithms for generating adversarial perturbations designed to misalign how victim agents perceive their environment. Our approach is empirically validated on three benchmarks and 22 environments, demonstrating its effectiveness across diverse algorithms and environments. Furthermore, we show that our algorithm is sample-efficient, requiring only 1,000 samples compared to the millions needed by previous methods.

摘要: 协作多智能体强化学习（c-MARL）迅速发展，为现实世界应用程序（包括敏感领域）提供了最先进的算法。然而，其广泛采用的一个关键挑战是缺乏对其对抗攻击的脆弱性的彻底调查。现有的工作主要集中在训练时攻击或不切实际的场景，例如访问策略权重或训练代理策略的能力。在本文中，我们在更现实和受约束的条件下研究新漏洞，假设对手只能收集和扰乱已部署代理的观察结果。我们还考虑对手根本无法访问的情况。我们提出了简单但高效的算法来生成对抗性扰动，旨在使受害者代理感知其环境的方式不一致。我们的方法在三个基准和22个环境上得到了经验验证，证明了其在不同算法和环境中的有效性。此外，我们表明我们的算法具有样本效率，仅需要1，000个样本，而之前的方法需要数百万个样本。



## **33. Fre-CW: Targeted Attack on Time Series Forecasting using Frequency Domain Loss**

Fre-CW：使用频域损失对时间序列预测进行有针对性的攻击 cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08955v1) [paper-pdf](http://arxiv.org/pdf/2508.08955v1)

**Authors**: Naifu Feng, Lixing Chen, Junhua Tang, Hua Ding, Jianhua Li, Yang Bai

**Abstract**: Transformer-based models have made significant progress in time series forecasting. However, a key limitation of deep learning models is their susceptibility to adversarial attacks, which has not been studied enough in the context of time series prediction. In contrast to areas such as computer vision, where adversarial robustness has been extensively studied, frequency domain features of time series data play an important role in the prediction task but have not been sufficiently explored in terms of adversarial attacks. This paper proposes a time series prediction attack algorithm based on frequency domain loss. Specifically, we adapt an attack method originally designed for classification tasks to the prediction field and optimize the adversarial samples using both time-domain and frequency-domain losses. To the best of our knowledge, there is no relevant research on using frequency information for time-series adversarial attacks. Our experimental results show that these current time series prediction models are vulnerable to adversarial attacks, and our approach achieves excellent performance on major time series forecasting datasets.

摘要: 基于转换器的模型在时间序列预测方面取得了重大进展。然而，深度学习模型的一个关键局限性是它们对对抗攻击的敏感性，而在时间序列预测的背景下，这一点还没有得到足够的研究。与计算机视觉等领域的对抗鲁棒性已得到广泛研究相反，时间序列数据的频域特征在预测任务中发挥着重要作用，但在对抗攻击方面尚未得到充分的探索。提出了一种基于频域损失的时间序列预测攻击算法。具体来说，我们将最初为分类任务设计的攻击方法应用到预测领域，并使用时间域和频域损失来优化对抗样本。据我们所知，目前还没有关于使用频率信息进行时间序列对抗攻击的相关研究。我们的实验结果表明，这些当前的时间序列预测模型容易受到对抗攻击，并且我们的方法在主要时间序列预测数据集上取得了出色的性能。



## **34. Exploring Cross-Stage Adversarial Transferability in Class-Incremental Continual Learning**

探索课堂增量式持续学习中的跨阶段对抗转移性 cs.LG

Accepted at MMSP 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08920v1) [paper-pdf](http://arxiv.org/pdf/2508.08920v1)

**Authors**: Jungwoo Kim, Jong-Seok Lee

**Abstract**: Class-incremental continual learning addresses catastrophic forgetting by enabling classification models to preserve knowledge of previously learned classes while acquiring new ones. However, the vulnerability of the models against adversarial attacks during this process has not been investigated sufficiently. In this paper, we present the first exploration of vulnerability to stage-transferred attacks, i.e., an adversarial example generated using the model in an earlier stage is used to attack the model in a later stage. Our findings reveal that continual learning methods are highly susceptible to these attacks, raising a serious security issue. We explain this phenomenon through model similarity between stages and gradual robustness degradation. Additionally, we find that existing adversarial training-based defense methods are not sufficiently effective to stage-transferred attacks. Codes are available at https://github.com/mcml-official/CSAT.

摘要: 类增量持续学习通过使分类模型能够在获取新类的同时保留以前学习过的类的知识来解决灾难性遗忘问题。然而，模型在此过程中对抗攻击的脆弱性尚未得到充分的研究。在本文中，我们首次探讨了阶段转移攻击的脆弱性，即在早期阶段使用模型生成的对抗示例用于在后期攻击模型。我们的研究结果表明，持续学习方法极易受到这些攻击，从而引发严重的安全问题。我们通过阶段之间的模型相似性和逐渐的鲁棒性退化来解释这种现象。此外，我们发现现有的基于对抗训练的防御方法对于阶段转移攻击不够有效。代码可访问https://github.com/mcml-official/CSAT。



## **35. Improving the robustness of neural ODEs with minimal weight perturbation**

通过最小的权重扰动提高神经ODE的鲁棒性 math.NA

31 pages, 5 figures, 4 tables

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2501.10740v2) [paper-pdf](http://arxiv.org/pdf/2501.10740v2)

**Authors**: Arturo De Marinis, Nicola Guglielmi, Stefano Sicilia, Francesco Tudisco

**Abstract**: We propose a method to enhance the stability of a neural ordinary differential equation (neural ODE) by reducing the maximum error growth subsequent to a perturbation of the initial value. Since the stability depends on the logarithmic norm of the Jacobian matrix associated with the neural ODE, we control the logarithmic norm by perturbing the weight matrices of the neural ODE by a smallest possible perturbation (in Frobenius norm). We do so by engaging an eigenvalue optimisation problem, for which we propose a nested two-level algorithm. For a given perturbation size of the weight matrix, the inner level computes optimal perturbations of that size, while - at the outer level - we tune the perturbation amplitude until we reach the desired uniform stability bound. We embed the proposed algorithm in the training of the neural ODE to improve its robustness to perturbations of the initial value, as adversarial attacks. Numerical experiments on classical image datasets show that an image classifier including a neural ODE in its architecture trained according to our strategy is more stable than the same classifier trained in the classical way, and therefore, it is more robust and less vulnerable to adversarial attacks.

摘要: 我们提出了一种通过减少初始值扰动后的最大误差增长来增强神经常微方程（神经ODE）的稳定性的方法。由于稳定性取决于与神经ODE相关的雅可比矩阵的log norm，因此我们通过以最小可能的扰动（弗罗贝尼乌斯norm）扰动神经ODE的权重矩阵来控制log norm。我们通过参与特征值优化问题来实现这一目标，为此我们提出了一种嵌套两级算法。对于权重矩阵的给定扰动大小，内部层计算该大小的最佳扰动，而在外部层，我们调整扰动幅度，直到达到期望的一致稳定性界限。我们将提出的算法嵌入到神经ODE的训练中，以提高其对初始值扰动（如对抗性攻击）的鲁棒性。经典图像数据集的数值实验表明，根据我们的策略训练的架构中包含神经ODE的图像分类器比以经典方式训练的相同分类器更稳定，因此，它更稳健，更不容易受到对抗性攻击。



## **36. Adversarial Video Promotion Against Text-to-Video Retrieval**

针对文本到视频检索的对抗性视频推广 cs.CV

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.06964v2) [paper-pdf](http://arxiv.org/pdf/2508.06964v2)

**Authors**: Qiwei Tian, Chenhao Lin, Zhengyu Zhao, Qian Li, Shuai Liu, Chao Shen

**Abstract**: Thanks to the development of cross-modal models, text-to-video retrieval (T2VR) is advancing rapidly, but its robustness remains largely unexamined. Existing attacks against T2VR are designed to push videos away from queries, i.e., suppressing the ranks of videos, while the attacks that pull videos towards selected queries, i.e., promoting the ranks of videos, remain largely unexplored. These attacks can be more impactful as attackers may gain more views/clicks for financial benefits and widespread (mis)information. To this end, we pioneer the first attack against T2VR to promote videos adversarially, dubbed the Video Promotion attack (ViPro). We further propose Modal Refinement (MoRe) to capture the finer-grained, intricate interaction between visual and textual modalities to enhance black-box transferability. Comprehensive experiments cover 2 existing baselines, 3 leading T2VR models, 3 prevailing datasets with over 10k videos, evaluated under 3 scenarios. All experiments are conducted in a multi-target setting to reflect realistic scenarios where attackers seek to promote the video regarding multiple queries simultaneously. We also evaluated our attacks for defences and imperceptibility. Overall, ViPro surpasses other baselines by over $30/10/4\%$ for white/grey/black-box settings on average. Our work highlights an overlooked vulnerability, provides a qualitative analysis on the upper/lower bound of our attacks, and offers insights into potential counterplays. Code will be publicly available at https://github.com/michaeltian108/ViPro.

摘要: 由于跨模式模型的发展，文本转视频检索（T2 VR）正在迅速发展，但其稳健性在很大程度上仍未得到检验。针对T2 VR的现有攻击旨在将视频从查询中推开，即抑制视频队列，而将视频拉向所选查询的攻击，即，推广视频的行列在很大程度上仍然没有被探索。这些攻击的影响力可能更大，因为攻击者可能会为了经济利益和广泛（错误）信息而获得更多查看/点击。为此，我们率先针对T2 VR发起了第一次攻击，以对抗性地推广视频，称为视频推广攻击（ViPro）。我们进一步提出模式细化（MoRe）来捕捉视觉和文本模式之间更细粒度、复杂的交互，以增强黑匣子的可移植性。全面的实验涵盖2个现有基线、3个领先的T2 VR模型、3个包含超过10，000个视频的流行数据集，在3种场景下进行评估。所有实验都在多目标环境中进行，以反映攻击者试图同时宣传有关多个查询的视频的现实场景。我们还评估了攻击的防御性和不可感知性。总体而言，ViPro在白色/灰色/黑匣子设置方面平均比其他基线高出30美元/10/4美元以上。我们的工作强调了一个被忽视的漏洞，对攻击的上/下限进行了定性分析，并提供了对潜在反击的见解。代码将在www.example.com上公开获取。



## **37. Cowpox: Towards the Immunity of VLM-based Multi-Agent Systems**

牛痘：基于VLM的多智能体系统的免疫力 cs.MA

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09230v1) [paper-pdf](http://arxiv.org/pdf/2508.09230v1)

**Authors**: Yutong Wu, Jie Zhang, Yiming Li, Chao Zhang, Qing Guo, Nils Lukas, Tianwei Zhang

**Abstract**: Vision Language Model (VLM)-based agents are stateful, autonomous entities capable of perceiving and interacting with their environments through vision and language. Multi-agent systems comprise specialized agents who collaborate to solve a (complex) task. A core security property is robustness, stating that the system should maintain its integrity under adversarial attacks. However, the design of existing multi-agent systems lacks the robustness consideration, as a successful exploit against one agent can spread and infect other agents to undermine the entire system's assurance. To address this, we propose a new defense approach, Cowpox, to provably enhance the robustness of multi-agent systems. It incorporates a distributed mechanism, which improves the recovery rate of agents by limiting the expected number of infections to other agents. The core idea is to generate and distribute a special cure sample that immunizes an agent against the attack before exposure and helps recover the already infected agents. We demonstrate the effectiveness of Cowpox empirically and provide theoretical robustness guarantees.

摘要: 基于视觉语言模型（VLM）的代理是有状态的自治实体，能够通过视觉和语言感知其环境并与其环境交互。多代理系统包括协作解决（复杂）任务的专业代理。核心安全属性是稳健性，即系统应在对抗性攻击下保持其完整性。然而，现有多代理系统的设计缺乏稳健性考虑，因为针对一个代理的成功利用可能会传播和感染其他代理，从而破坏整个系统的保证。为了解决这个问题，我们提出了一种新的防御方法Cowpox，以可证明地增强多智能体系统的鲁棒性。它结合了分布式机制，通过限制其他病原体的预期感染数量来提高病原体的恢复率。核心想法是生成和分发一种特殊的治疗样本，在暴露前使病原体免疫抵抗攻击，并帮助恢复已经感染的病原体。我们从经验上证明了Cowpox的有效性，并提供了理论上的稳健性保证。



## **38. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

PAR-AdvGAN：通过渐进式自回归AdvGAN提高对抗攻击能力 cs.LG

Best student paper award of ECML-PKDD 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2502.12207v3) [paper-pdf](http://arxiv.org/pdf/2502.12207v3)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR

摘要: 深度神经网络在各个领域都表现出了卓越的性能。然而，它们容易受到对抗性例子的影响，这可能导致错误的预测。生成对抗网络（GAN）可以利用生成器和鉴别器模型快速生成高质量的对抗示例。由于两个模块都以竞争和同步的方式进行训练，因此与传统方法相比，基于GAN的算法（如AdvGAN）可以生成具有更好可移植性的对抗性示例。然而，扰动的产生通常仅限于单次迭代，从而阻止这些示例充分利用方法的潜力。为了解决这个问题，我们引入了一种名为渐进式自动回归AdvGAN（PAR-AdvGAN）的新颖方法。它在渐进生成网络中集成了自回归迭代机制，以制作具有增强攻击能力的对抗性示例。我们通过大规模实验彻底评估了我们的PAR-AdvGAN方法，证明了其优于各种最先进的黑匣子对抗攻击以及原始的AdvGAN的性能。此外，PAR-AdvGAN显着加速了对抗性示例的生成，即Inception-v3模型上的速度高达每秒335.5帧，优于基于梯度的可转移攻击算法。我们的代码可访问：https://github.com/LMBTough/PAR



## **39. Evasive Ransomware Attacks Using Low-level Behavioral Adversarial Examples**

使用低级行为对抗示例的规避勒索软件攻击 cs.CR

\copyright 2025 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08656v1) [paper-pdf](http://arxiv.org/pdf/2508.08656v1)

**Authors**: Manabu Hirano, Ryotaro Kobayashi

**Abstract**: Protecting state-of-the-art AI-based cybersecurity defense systems from cyber attacks is crucial. Attackers create adversarial examples by adding small changes (i.e., perturbations) to the attack features to evade or fool the deep learning model. This paper introduces the concept of low-level behavioral adversarial examples and its threat model of evasive ransomware. We formulate the method and the threat model to generate the optimal source code of evasive malware. We then examine the method using the leaked source code of Conti ransomware with the micro-behavior control function. The micro-behavior control function is our test component to simulate changing source code in ransomware; ransomware's behavior can be changed by specifying the number of threads, file encryption ratio, and delay after file encryption at the boot time. We evaluated how much an attacker can control the behavioral features of ransomware using the micro-behavior control function to decrease the detection rate of a ransomware detector.

摘要: 保护最先进的基于人工智能的网络安全防御系统免受网络攻击至关重要。攻击者通过添加小的更改（即，扰动）攻击特征以逃避或愚弄深度学习模型。本文介绍了低层行为对抗示例的概念及其规避勒索软件的威胁模型。我们制定了方法和威胁模型来生成规避恶意软件的最佳源代码。然后，我们使用泄露的Conti勒索软件源代码以及微行为控制功能来检查该方法。微行为控制功能是我们模拟勒索软件中更改源代码的测试组件;勒索软件的行为可以通过指定线程数、文件加密率和引导时文件加密后的延迟来更改。我们评估了攻击者可以使用微行为控制功能控制勒索软件的行为特征的程度，以降低勒索软件检测器的检测率。



## **40. Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment**

保护教育LLM：LLM攻击的一般分类和DREAD风险评估 cs.CY

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08629v1) [paper-pdf](http://arxiv.org/pdf/2508.08629v1)

**Authors**: Farzana Zahid, Anjalika Sewwandi, Lee Brandon, Vimal Kumar, Roopak Sinha

**Abstract**: Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions.

摘要: 由于人们对效率和生产力的显着提高，包括教育在内的各种组织正在将大型语言模型（LLM）纳入其工作流程中。面向教育者、面向学习者和面向机构的LLM（统称为教育大型语言模型（eLLM）），补充和增强教学、学习和学术运营的有效性。然而，它们融入教育环境会引发严重的网络安全问题。缺乏当代针对法学硕士的攻击及其对教育环境影响的全面景观。本研究提出了针对LLM的五十种攻击的一般分类，这些攻击被归类为针对模型或其基础设施的攻击。教育部门使用DREAD风险评估框架评估这些攻击的严重性。我们的风险评估表明，代币走私、对抗性提示、直接注入和多步越狱是对eLLM的严重攻击。拟议的分类法、其在教育环境中的应用以及我们的风险评估将帮助学术和行业从业者构建保护学习者和机构的弹性解决方案。



## **41. Generative AI for Critical Infrastructure in Smart Grids: A Unified Framework for Synthetic Data Generation and Anomaly Detection**

智能电网关键基础设施的生成人工智能：合成数据生成和异常检测的统一框架 cs.CR

28 pages, 12 figures

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08593v1) [paper-pdf](http://arxiv.org/pdf/2508.08593v1)

**Authors**: Aydin Zaboli, Junho Hong

**Abstract**: In digital substations, security events pose significant challenges to the sustained operation of power systems. To mitigate these challenges, the implementation of robust defense strategies is critically important. A thorough process of anomaly identification and detection in information and communication technology (ICT) frameworks is crucial to ensure secure and reliable communication and coordination between interconnected devices within digital substations. Hence, this paper addresses the critical cybersecurity challenges confronting IEC61850-based digital substations within modern smart grids, where the integration of advanced communication protocols, e.g., generic object-oriented substation event (GOOSE), has enhanced energy management and introduced significant vulnerabilities to cyberattacks. Focusing on the limitations of traditional anomaly detection systems (ADSs) in detecting threats, this research proposes a transformative approach by leveraging generative AI (GenAI) to develop robust ADSs. The primary contributions include the suggested advanced adversarial traffic mutation (AATM) technique to generate synthesized and balanced datasets for GOOSE messages, ensuring protocol compliance and enabling realistic zero-day attack pattern creation to address data scarcity. Then, the implementation of GenAI-based ADSs incorporating the task-oriented dialogue (ToD) processes has been explored for improved detection of attack patterns. Finally, a comparison of the GenAI-based ADS with machine learning (ML)-based ADSs has been implemented to showcase the outperformance of the GenAI-based frameworks considering the AATM-generated GOOSE datasets and standard/advanced performance evaluation metrics.

摘要: 在数字变电站中，安全事件对电力系统的持续运行构成了重大挑战。为了缓解这些挑战，实施强有力的防御策略至关重要。信息和通信技术（ICT）框架中彻底的异常识别和检测过程对于确保数字变电站内互连设备之间安全可靠的通信和协调至关重要。因此，本文解决了现代智能电网中基于EC 61850的数字变电站面临的关键网络安全挑战，其中集成了先进的通信协议，例如，通用面向对象的变电站事件（GOOSE）增强了能源管理，并引入了网络攻击的重大漏洞。该研究重点关注传统异常检测系统（ADS）在检测威胁方面的局限性，提出了一种通过利用生成式人工智能（GenAI）来开发稳健的ADS的变革方法。主要贡献包括建议的高级对抗流量变异（AATM）技术，为GOOSE消息生成合成和平衡的数据集，确保协议合规性并创建现实的零日攻击模式以解决数据稀缺问题。然后，探索了结合面向任务的对话（ToD）流程的基于GenAI的ADS的实现，以改进攻击模式的检测。最后，考虑到AATM生成的GOOSE数据集和标准/高级性能评估指标，对基于GenAI的ADS与基于机器学习（ML）的ADS进行了比较，以展示基于GenAI的框架的优异性能。



## **42. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

视觉语言模型的少镜头对抗低秩微调 cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.15130v2) [paper-pdf](http://arxiv.org/pdf/2505.15130v2)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.

摘要: 通过大规模对比预训练，CLIP等视觉语言模型（VLM）在跨模式任务中表现出了出色的表现。为了有效地调整这些基于变压器的大型模型以适应下游任务，LoRA等参数高效微调（PEFT）技术已成为完全微调的可扩展替代方案，尤其是在少量场景中。然而，与传统的深度神经网络一样，VLM非常容易受到对抗攻击，其中不可感知的扰动可能会显着降低模型性能。对抗训练仍然是提高PEFT模型稳健性的最有效策略。在这项工作中，我们提出了AdvCLIP-LoRA，这是第一个旨在增强在少数镜头设置中使用LoRA微调的CLIP模型的对抗鲁棒性的算法。我们的方法将对抗性微调表述为极小极大优化问题，并为光滑性和非凸强插值假设下的收敛提供理论保证。使用ViT-B/16和ViT-B/32模型的八个数据集的经验结果表明，AdvCLIP-LoRA显着提高了针对常见对抗攻击（例如，FGSM、PVD），而不会牺牲太多干净的准确性。这些发现凸显了AdvCLIP-LoRA是一种实用且具有理论依据的方法，用于在资源有限的环境中稳健地适应VLM。



## **43. VISOR: Visual Input-based Steering for Output Redirection in Vision-Language Models**

VICOR：视觉语言模型中基于视觉输入的输出重定向转向 cs.CV

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08521v1) [paper-pdf](http://arxiv.org/pdf/2508.08521v1)

**Authors**: Mansi Phute, Ravikumar Balakrishnan

**Abstract**: Vision Language Models (VLMs) are increasingly being used in a broad range of applications, bringing their security and behavioral control to the forefront. While existing approaches for behavioral control or output redirection, like system prompting in VLMs, are easily detectable and often ineffective, activation-based steering vectors require invasive runtime access to model internals--incompatible with API-based services and closed-source deployments. We introduce VISOR (Visual Input-based Steering for Output Redirection), a novel method that achieves sophisticated behavioral control through optimized visual inputs alone. By crafting universal steering images that induce target activation patterns, VISOR enables practical deployment across all VLM serving modalities while remaining imperceptible compared to explicit textual instructions. We validate VISOR on LLaVA-1.5-7B across three critical alignment tasks: refusal, sycophancy and survival instinct. A single 150KB steering image matches steering vector performance within 1-2% for positive behavioral shifts while dramatically exceeding it for negative steering--achieving up to 25% shifts from baseline compared to steering vectors' modest changes. Unlike system prompting (3-4% shifts), VISOR provides robust bidirectional control while maintaining 99.9% performance on 14,000 unrelated MMLU tasks. Beyond eliminating runtime overhead and model access requirements, VISOR exposes a critical security vulnerability: adversaries can achieve sophisticated behavioral manipulation through visual channels alone, bypassing text-based defenses. Our work fundamentally re-imagines multimodal model control and highlights the urgent need for defenses against visual steering attacks.

摘要: 视觉语言模型（VLM）越来越多地用于广泛的应用程序，将其安全性和行为控制置于最前沿。虽然现有的行为控制或输出重定向方法（例如VLM中的系统提示）很容易检测到并且通常无效，但基于激活的引导载体需要对模型内部进行侵入性的运行时访问--这与基于API的服务和闭源部署不兼容。我们引入了VICOR（基于视觉输入的输出重定向转向转向），这是一种新颖的方法，可以仅通过优化的视觉输入来实现复杂的行为控制。通过制作诱导目标激活模式的通用转向图像，VICOR可以在所有VLM服务模式中进行实际部署，同时与显式文本指令相比保持不可感知。我们在LLaVA-1.5- 7 B上验证了VISOR在三个关键对齐任务中的作用：拒绝，奉承和生存本能。一个150 KB的转向图像在1-2%的范围内匹配转向矢量的性能，而在负转向方面则大大超过它--与转向矢量的适度变化相比，它从基线的变化高达25%。与系统提示（3-4%的移位）不同，VISOR提供强大的双向控制，同时在14，000个不相关的MMLU任务上保持99.9%的性能。除了消除运行时开销和模型访问需求外，VISOR还暴露了一个关键的安全漏洞：攻击者可以仅通过视觉通道实现复杂的行为操纵，绕过基于文本的防御。我们的工作从根本上重新构想了多模式模型控制，并强调了防御视觉转向攻击的迫切需要。



## **44. Designing with Deception: ML- and Covert Gate-Enhanced Camouflaging to Thwart IC Reverse Engineering**

利用欺骗进行设计：ML和隐蔽门增强伪装以阻止IC反向工程 cs.CR

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08462v1) [paper-pdf](http://arxiv.org/pdf/2508.08462v1)

**Authors**: Junling Fan, David Koblah, Domenic Forte

**Abstract**: Integrated circuits (ICs) are essential to modern electronic systems, yet they face significant risks from physical reverse engineering (RE) attacks that compromise intellectual property (IP) and overall system security. While IC camouflage techniques have emerged to mitigate these risks, existing approaches largely focus on localized gate modifications, neglecting comprehensive deception strategies. To address this gap, we present a machine learning (ML)-driven methodology that integrates cryptic and mimetic cyber deception principles to enhance IC security against RE. Our approach leverages a novel And-Inverter Graph Variational Autoencoder (AIG-VAE) to encode circuit representations, enabling dual-layered camouflage through functional preservation and appearance mimicry. By introducing new variants of covert gates -- Fake Inverters, Fake Buffers, and Universal Transmitters -- our methodology achieves robust protection by obscuring circuit functionality while presenting misleading appearances. Experimental results demonstrate the effectiveness of our strategy in maintaining circuit functionality while achieving high camouflage and similarity scores with minimal structural overhead. Additionally, we validate the robustness of our method against advanced artificial intelligence (AI)-enhanced RE attacks, highlighting its practical applicability in securing IC designs. By bridging the gap in mimetic deception for hardware security, our work sets a new standard for IC camouflage, advancing the application of cyber deception principles to protect critical systems from adversarial threats.

摘要: 集成电路（IC）对于现代电子系统至关重要，但它们面临着物理反向工程（RE）攻击的重大风险，这些攻击会损害知识产权（IP）和整体系统安全。虽然IC伪装技术的出现可以减轻这些风险，但现有方法主要集中在局部化的门修改上，忽视了全面的欺骗策略。为了解决这一差距，我们提出了一种机器学习（ML）驱动的方法，该方法集成了神秘和模仿网络欺骗原则，以增强IC针对RE的安全性。我们的方法利用新型的与反相器图变分自动编码器（AIG-VAE）来编码电路表示，通过功能保留和外观模仿实现双层伪装。通过引入隐蔽门的新变体-假逆变器，假缓冲器和通用发射器-我们的方法通过模糊电路功能同时呈现误导性外观来实现强大的保护。实验结果表明，我们的策略在保持电路功能，同时实现高伪装和相似性分数与最小的结构开销的有效性。此外，我们验证了我们的方法对先进的人工智能（AI）增强RE攻击的鲁棒性，突出了其在安全IC设计的实用性。通过弥合硬件安全模拟欺骗的差距，我们的工作为IC伪装设定了新的标准，推进了网络欺骗原理的应用，以保护关键系统免受对抗性威胁。



## **45. Evaluating lightweight unsupervised online IDS for masquerade attacks in CAN**

评估轻量级无监督在线IDS的伪装攻击 cs.CR

22 pages, 10 figures, 4 tables. New title

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2406.13778v3) [paper-pdf](http://arxiv.org/pdf/2406.13778v3)

**Authors**: Pablo Moriano, Steven C. Hespeler, Mingyan Li, Robert A. Bridges

**Abstract**: Vehicular controller area networks (CANs) are susceptible to masquerade attacks by malicious adversaries. In masquerade attacks, adversaries silence a targeted ID and then send malicious frames with forged content at the expected timing of benign frames. As masquerade attacks could seriously harm vehicle functionality and are the stealthiest attacks to detect in CAN, recent work has devoted attention to compare frameworks for detecting masquerade attacks in CAN. However, most existing works report offline evaluations using CAN logs already collected using simulations that do not comply with the domain's real-time constraints. Here we contribute to advance the state of the art by presenting a comparative evaluation of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing comparative evaluations in that we analyze the effect of controlling streaming data conditions in a sliding window setting. In doing so, we use realistic masquerade attacks being replayed from the ROAD dataset. We show that although evaluated IDS are not effective at detecting every attack type, the method that relies on detecting changes in the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead. We discuss limitations, open challenges, and how the evaluated methods can be used for practical unsupervised online CAN IDS for masquerade attacks.

摘要: 车辆控制器区域网络（CAN）容易受到恶意对手的伪装攻击。在伪装攻击中，对手会压制目标ID，然后在良性帧的预期时间发送包含伪造内容的恶意帧。由于化装攻击可能会严重损害车辆功能，并且是CAN中最隐蔽的攻击，因此最近的工作重点关注比较用于检测CAN中化装攻击的框架。然而，大多数现有作品使用已经通过不符合域实时限制的模拟收集的CAN日志来报告离线评估。在这里，我们通过对四个不同的基于非深度学习（DL）的无监督在线入侵检测系统（IDS）进行比较评估来促进最新技术水平的发展。我们的方法与现有的比较评估的不同之处在于，我们分析了在滑动窗口设置中控制流数据条件的影响。在此过程中，我们使用从ROAD数据集中重播的真实化装攻击。我们表明，尽管评估的IDS不能有效检测每种攻击类型，但依赖于检测时间序列集群分层结构变化的方法可以产生最好的结果，但代价是更高的计算负担。我们讨论了局限性、开放挑战以及如何将评估的方法用于实际的无监督在线CAN IDS以进行伪装攻击。



## **46. Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference**

LLM推理中的选择性KV缓存共享以缓解定时侧通道 cs.CR

17 pages,17 figures

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08438v1) [paper-pdf](http://arxiv.org/pdf/2508.08438v1)

**Authors**: Kexin Chu, Zecheng Lin, Dawei Xiang, Zixu Shen, Jianchang Su, Cheng Chu, Yiwei Yang, Wenhui Zhang, Wenfei Wu, Wei Zhang

**Abstract**: Global KV-cache sharing has emerged as a key optimization for accelerating large language model (LLM) inference. However, it exposes a new class of timing side-channel attacks, enabling adversaries to infer sensitive user inputs via shared cache entries. Existing defenses, such as per-user isolation, eliminate leakage but degrade performance by up to 38.9% in time-to-first-token (TTFT), making them impractical for high-throughput deployment. To address this gap, we introduce SafeKV (Secure and Flexible KV Cache Sharing), a privacy-aware KV-cache management framework that selectively shares non-sensitive entries while confining sensitive content to private caches. SafeKV comprises three components: (i) a hybrid, multi-tier detection pipeline that integrates rule-based pattern matching, a general-purpose privacy detector, and context-aware validation; (ii) a unified radix-tree index that manages public and private entries across heterogeneous memory tiers (HBM, DRAM, SSD); and (iii) entropy-based access monitoring to detect and mitigate residual information leakage. Our evaluation shows that SafeKV mitigates 94% - 97% of timing-based side-channel attacks. Compared to per-user isolation method, SafeKV improves TTFT by up to 40.58% and throughput by up to 2.66X across diverse LLMs and workloads. SafeKV reduces cache-induced TTFT overhead from 50.41% to 11.74% on Qwen3-235B. By combining fine-grained privacy control with high cache reuse efficiency, SafeKV reclaims the performance advantages of global sharing while providing robust runtime privacy guarantees for LLM inference.

摘要: 全局KV缓存共享已成为加速大型语言模型（LLM）推理的关键优化。然而，它暴露了一类新的定时侧通道攻击，使对手能够通过共享缓存条目推断敏感用户输入。现有的防御措施（例如按用户隔离）可以消除泄漏，但在首次令牌时间（TTFT）方面性能会降低高达38.9%，因此对于高吞吐量部署来说不切实际。为了解决这一差距，我们引入了SafeKV（安全且灵活的KV缓存共享），这是一种隐私感知的KV缓存管理框架，可以选择性地共享非敏感条目，同时将敏感内容限制在私人缓存中。SafeKV由三个部分组成：（i）混合、多层检测管道，集成了基于规则的模式匹配、通用隐私检测器和上下文感知验证;（ii）统一的根树索引，管理跨异类存储器层（HBM、RAM、SSD）的公共和私有条目;和（iii）基于熵的访问监控，以检测和减轻剩余信息泄露。我们的评估表明，SafeKV可以缓解94% - 97%的基于计时的侧通道攻击。与按用户隔离方法相比，SafeKN将TTFT提高了40.58%，将各种LLM和工作负载的吞吐量提高了2.66倍。SafeKV将Qwen 3 - 235 B上高速缓存引起的TTFT费用从50.41%减少到11.74%。通过将细粒度隐私控制与高缓存重复使用效率相结合，SafeKN充分利用了全局共享的性能优势，同时为LLM推断提供强大的运行时隐私保证。



## **47. Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity**

通过平衡的话题性和OOD强度实现有效的MLLM越狱 cs.CV

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.09218v1) [paper-pdf](http://arxiv.org/pdf/2508.09218v1)

**Authors**: Zuoou Li, Weitong Zhang, Jingyuan Wang, Shuyuan Zhang, Wenjia Bai, Bernhard Kainz, Mengyun Qiao

**Abstract**: Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems.

摘要: 多模式大型语言模型（MLLM）广泛用于视觉语言推理任务。然而，它们对对抗提示的脆弱性仍然是一个严重问题，因为安全机制往往无法防止有害输出的产生。尽管最近的越狱策略报告了很高的成功率，但许多被归类为“成功”的响应实际上是良性的、模糊的，或与预期的恶意目标无关。这种不匹配表明当前的评估标准可能高估了此类攻击的有效性。为了解决这个问题，我们引入了一个四轴评估框架，该框架考虑了输入的主题性、输入未分配（OOD）强度、输出危害性和输出拒绝率。该框架确定了真正有效的越狱。在一项实质性的实证研究中，我们揭示了一种结构性权衡：高度切中主题的提示经常被安全过滤器阻止，而那些过于OOD的提示经常逃避检测，但无法产生有害内容。然而，平衡相关性和新颖性的提示更有可能逃避过滤器并触发危险的输出。基于这一见解，我们开发了一种名为平衡结构分解（BCD）的循环重写策略。该方法将恶意提示重组为语义对齐的子任务，同时引入微妙的OOD信号和视觉线索，使输入更难检测。BDS在13个商业和开源MLLM上进行了测试，它始终导致更高的攻击成功率、更多的有害输出和更少的拒绝。与以前的方法相比，它的成功率提高了67美元，危害性提高了21美元，揭示了当前多模式安全系统中以前被低估的弱点。



## **48. Adaptive Learning for IRS-Assisted Wireless Networks: Securing Opportunistic Communications Against Byzantine Eavesdroppers**

IRS辅助无线网络的自适应学习：保护伪通信免受拜占庭式发射器的攻击 eess.SP

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08206v1) [paper-pdf](http://arxiv.org/pdf/2508.08206v1)

**Authors**: Amirhossein Taherpour, Abbas Taherpour, Tamer Khattab

**Abstract**: We propose a joint learning framework for Byzantine-resilient spectrum sensing and secure intelligent reflecting surface (IRS)--assisted opportunistic access under channel state information (CSI) uncertainty. The sensing stage performs logit-domain Bayesian updates with trimmed aggregation and attention-weighted consensus, and the base station (BS) fuses network beliefs with a conservative minimum rule, preserving detection accuracy under a bounded number of Byzantine users. Conditioned on the sensing outcome, we pose downlink design as sum mean-squared error (MSE) minimization under transmit-power and signal-leakage constraints and jointly optimize the BS precoder, IRS phase shifts, and user equalizers. With partial (or known) CSI, we develop an augmented-Lagrangian alternating algorithm with projected updates and provide provable sublinear convergence, with accelerated rates under mild local curvature. With unknown CSI, we perform constrained Bayesian optimization (BO) in a geometry-aware low-dimensional latent space using Gaussian process (GP) surrogates; we prove regret bounds for a constrained upper confidence bound (UCB) variant of the BO module, and demonstrate strong empirical performance of the implemented procedure. Simulations across diverse network conditions show higher detection probability at fixed false-alarm rate under adversarial attacks, large reductions in sum MSE for honest users, strong suppression of eavesdropper signal power, and fast convergence. The framework offers a practical path to secure opportunistic communication that adapts to CSI availability while coherently coordinating sensing and transmission through joint learning.

摘要: 我们提出了一个联合学习框架，用于拜占庭弹性频谱感知和安全智能反射面（IRS）--在通道状态信息（SI）不确定性下辅助机会访问。感知阶段通过修剪聚合和注意力加权共识执行逻辑域Bayesian更新，基站（BS）将网络信念与保守最小规则融合，在有限数量的拜占庭用户下保持检测准确性。根据感测结果，我们将下行链路设计提出为发射功率和信号泄漏约束下的和均方误差（SSE）最小化，并联合优化BS预编码器、IRS移相和用户均衡器。通过部分（或已知）的SI，我们开发了一种具有投影更新的增广拉格朗日交替算法，并提供可证明的次线性收敛，并在轻微局部弯曲下加速速度。对于未知的SI，我们使用高斯过程（GP）代理在几何感知的低维潜在空间中执行约束式Bayesian优化（BO）;我们证明了BO模块的约束置信上界（UCB）变体的遗憾界，并展示了所实现的程序的强大经验性能。跨不同网络条件的模拟显示，在对抗性攻击下，在固定的虚警率下检测概率更高，诚实用户的SSE总和大幅降低，窃听者信号功率受到强抑制，收敛速度快。该框架提供了一种实用的路径来确保机会性通信，该通信适应于SI可用性，同时通过联合学习连贯地协调感知和传输。



## **49. Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks**

O-RAN中的鲁棒异常检测：利用LLM对抗数据操纵攻击 cs.CR

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08029v1) [paper-pdf](http://arxiv.org/pdf/2508.08029v1)

**Authors**: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

**Abstract**: The introduction of 5G and the Open Radio Access Network (O-RAN) architecture has enabled more flexible and intelligent network deployments. However, the increased complexity and openness of these architectures also introduce novel security challenges, such as data manipulation attacks on the semi-standardised Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In particular, malicious xApps can exploit this vulnerability by introducing subtle Unicode-wise alterations (hypoglyphs) into the data that are being used by traditional machine learning (ML)-based anomaly detection methods. These Unicode-wise manipulations can potentially bypass detection and cause failures in anomaly detection systems based on traditional ML, such as AutoEncoders, which are unable to process hypoglyphed data without crashing. We investigate the use of Large Language Models (LLMs) for anomaly detection within the O-RAN architecture to address this challenge. We demonstrate that LLM-based xApps maintain robust operational performance and are capable of processing manipulated messages without crashing. While initial detection accuracy requires further improvements, our results highlight the robustness of LLMs to adversarial attacks such as hypoglyphs in input data. There is potential to use their adaptability through prompt engineering to further improve the accuracy, although this requires further research. Additionally, we show that LLMs achieve low detection latency (under 0.07 seconds), making them suitable for Near-Real-Time (Near-RT) RIC deployments.

摘要: 5G和开放式无线电接入网络（O-RAN）架构的引入使网络部署更加灵活和智能。然而，这些体系结构复杂性和开放性的增加也带来了新颖的安全挑战，例如通过恶意xApp对O-RAN平台内的半标准化共享数据层（SDF）进行数据操纵攻击。特别是，恶意xApp可以通过在传统基于机器学习（ML）的异常检测方法使用的数据中引入微妙的Unicode更改（次字形）来利用此漏洞。这些基于Unicode的操作可能会绕过检测并导致基于传统ML的异常检测系统（例如AutoEncoders）出现故障，这些系统无法在不崩溃的情况下处理次字母数据。我们研究了在O-RAN架构内使用大型语言模型（LLM）进行异常检测的情况，以应对这一挑战。我们证明基于LLM的xApp可以保持稳健的操作性能，并且能够处理被操纵的消息而不会崩溃。虽然初始检测准确性需要进一步提高，但我们的结果强调了LLM对对抗攻击（例如输入数据中的副字形）的鲁棒性。有可能通过及时的工程利用它们的适应性来进一步提高准确性，尽管这需要进一步的研究。此外，我们表明LLM可以实现低检测延迟（低于0.07秒），使其适合近实时（Near-RT）RIC部署。



## **50. Universally Unfiltered and Unseen:Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards**

普遍未经过滤和不可见：针对文本到图像模型保障措施的输入不可知的多模式越狱 cs.CR

This paper has been accepted by ACM MM 2025

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.05658v2) [paper-pdf](http://arxiv.org/pdf/2508.05658v2)

**Authors**: Song Yan, Hui Wei, Jinlong Fei, Guoliang Yang, Zhengyu Zhao, Zheng Wang

**Abstract**: Various (text) prompt filters and (image) safety checkers have been implemented to mitigate the misuse of Text-to-Image (T2I) models in creating Not-Safe-For-Work (NSFW) content. In order to expose potential security vulnerabilities of such safeguards, multimodal jailbreaks have been studied. However, existing jailbreaks are limited to prompt-specific and image-specific perturbations, which suffer from poor scalability and time-consuming optimization. To address these limitations, we propose Universally Unfiltered and Unseen (U3)-Attack, a multimodal jailbreak attack method against T2I safeguards. Specifically, U3-Attack optimizes an adversarial patch on the image background to universally bypass safety checkers and optimizes a safe paraphrase set from a sensitive word to universally bypass prompt filters while eliminating redundant computations. Extensive experimental results demonstrate the superiority of our U3-Attack on both open-source and commercial T2I models. For example, on the commercial Runway-inpainting model with both prompt filter and safety checker, our U3-Attack achieves $~4\times$ higher success rates than the state-of-the-art multimodal jailbreak attack, MMA-Diffusion.

摘要: 已经实现了各种（文本）提示过滤器和（图像）安全检查器，以减轻在创建不安全工作（NSFW）内容时文本到图像（T2 I）模型的滥用。为了暴露此类保障措施的潜在安全漏洞，人们对多模式越狱进行了研究。然而，现有的越狱仅限于预算特定和图像特定的干扰，这些干扰具有较差的可扩展性和耗时的优化。为了解决这些限制，我们提出了普遍未过滤和不可见（U3）-Attack，这是一种针对T2 I保障措施的多模式越狱攻击方法。具体来说，U3-Attack优化了图像背景上的对抗补丁，以普遍绕过安全检查器，并优化敏感词的安全转述集，以普遍绕过提示过滤器，同时消除冗余计算。大量的实验结果证明了我们的U3-Attack在开源和商业T2 I模型上的优越性。例如，在具有提示过滤器和安全检查器的商业跑道修补模型上，我们的U3-Attack的成功率比最先进的多模式越狱攻击MMA-Distance高出4倍。



