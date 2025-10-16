# Latest Adversarial Attack Papers
**update at 2025-10-16 16:22:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Provably Invincible Adversarial Attacks on Reinforcement Learning Systems: A Rate-Distortion Information-Theoretic Approach**

对强化学习系统的可证明无敌的对抗攻击：一种速率失真信息理论方法 cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13792v1) [paper-pdf](http://arxiv.org/pdf/2510.13792v1)

**Authors**: Ziqing Lu, Lifeng Lai, Weiyu Xu

**Abstract**: Reinforcement learning (RL) for the Markov Decision Process (MDP) has emerged in many security-related applications, such as autonomous driving, financial decisions, and drone/robot algorithms. In order to improve the robustness/defense of RL systems against adversaries, studying various adversarial attacks on RL systems is very important. Most previous work considered deterministic adversarial attack strategies in MDP, which the recipient (victim) agent can defeat by reversing the deterministic attacks. In this paper, we propose a provably ``invincible'' or ``uncounterable'' type of adversarial attack on RL. The attackers apply a rate-distortion information-theoretic approach to randomly change agents' observations of the transition kernel (or other properties) so that the agent gains zero or very limited information about the ground-truth kernel (or other properties) during the training. We derive an information-theoretic lower bound on the recipient agent's reward regret and show the impact of rate-distortion attacks on state-of-the-art model-based and model-free algorithms. We also extend this notion of an information-theoretic approach to other types of adversarial attack, such as state observation attacks.

摘要: 马尔科夫决策过程（MDP）的强化学习（RL）已出现在许多与安全相关的应用中，例如自动驾驶、金融决策和无人机/机器人算法。为了提高RL系统对对手的鲁棒性/防御性，研究对RL系统的各种对抗攻击非常重要。之前的大多数工作都考虑了MDP中的确定性对抗攻击策略，接收者（受害者）代理可以通过逆转确定性攻击来击败这些策略。在本文中，我们提出了一种可证明的对RL的“无敌”或“不可对抗”类型的对抗性攻击。攻击者应用率失真信息论方法来随机改变代理对过渡核（或其他属性）的观察，以便代理在训练期间获得零或非常有限的有关地面真值核（或其他属性）的信息。我们推导出接收代理奖励后悔的信息理论下限，并展示了速率失真攻击对最先进的基于模型和无模型算法的影响。我们还将信息论方法的概念扩展到其他类型的对抗性攻击，例如状态观察攻击。



## **2. Towards Adversarial Robustness and Uncertainty Quantification in DINOv2-based Few-Shot Anomaly Detection**

基于DINOv2的少镜头异常检测中的对抗鲁棒性和不确定性量化 cs.CV

10 pages, 5 figures, 3 tables

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13643v1) [paper-pdf](http://arxiv.org/pdf/2510.13643v1)

**Authors**: Akib Mohammed Khan, Bartosz Krawczyk

**Abstract**: Foundation models such as DINOv2 have shown strong performance in few-shot anomaly detection, yet two key questions remain unexamined: (i) how susceptible are these detectors to adversarial perturbations; and (ii) how well do their anomaly scores reflect calibrated uncertainty? Building on AnomalyDINO, a training-free deep nearest-neighbor detector over DINOv2 features, we present one of the first systematic studies of adversarial attacks and uncertainty estimation in this setting. To enable white-box gradient attacks while preserving test-time behavior, we attach a lightweight linear head to frozen DINOv2 features only for crafting perturbations. Using this heuristic, we evaluate the impact of FGSM across the MVTec-AD and VisA datasets and observe consistent drops in F1, AUROC, AP, and G-mean, indicating that imperceptible perturbations can flip nearest-neighbor relations in feature space to induce confident misclassification. Complementing robustness, we probe reliability and find that raw anomaly scores are poorly calibrated, revealing a gap between confidence and correctness that limits safety-critical use. As a simple, strong baseline toward trustworthiness, we apply post-hoc Platt scaling to the anomaly scores for uncertainty estimation. The resulting calibrated posteriors yield significantly higher predictive entropy on adversarially perturbed inputs than on clean ones, enabling a practical flagging mechanism for attack detection while reducing calibration error (ECE). Our findings surface concrete vulnerabilities in DINOv2-based few-shot anomaly detectors and establish an evaluation protocol and baseline for robust, uncertainty-aware anomaly detection. We argue that adversarial robustness and principled uncertainty quantification are not optional add-ons but essential capabilities if anomaly detection systems are to be trustworthy and ready for real-world deployment.

摘要: DINOv 2等基础模型在少量异常检测中表现出了强劲的性能，但有两个关键问题仍未得到研究：（i）这些检测器对对抗性扰动的敏感性有多大;（ii）它们的异常分数反映了校准的不确定性？基于AnomalyDINO（一种针对DINOv 2特征的免训练深度近邻检测器），我们在这种环境下首次对对抗攻击和不确定性估计进行了系统性研究之一。为了启用白盒梯度攻击，同时保留测试时行为，我们将轻量级线性头附加到冻结的DINOv 2特征上，仅用于制造扰动。使用这种启发式，我们评估了FGSM对MMVTec-AD和VisA数据集的影响，并观察到F1、AUROC、AP和G-mean的一致下降，这表明不可感知的扰动可以翻转特征空间中的最近邻关系，从而引发可信的错误分类。除了鲁棒性之外，我们还调查了可靠性，发现原始异常分数校准不良，揭示了信心和正确性之间的差距，从而限制了安全关键使用。作为可信度的简单、强基线，我们将事后普拉特缩放应用于异常分数以进行不确定性估计。由此产生的校准后验在对抗干扰的输入上产生的预测信息比在干净的输入上明显更高，从而实现了实用的攻击检测标记机制，同时减少了校准误差（ECA）。我们的研究结果揭示了基于DINOv 2的几次异常检测器中的具体漏洞，并为稳健的、不确定性感知的异常检测建立了评估协议和基线。我们认为，如果异常检测系统要值得信赖并为现实世界的部署做好准备，对抗稳健性和原则性的不确定性量化不是可选的附加组件，而是必不可少的能力。



## **3. Selective Adversarial Attacks on LLM Benchmarks**

LLM基准的选择性对抗攻击 cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13570v1) [paper-pdf](http://arxiv.org/pdf/2510.13570v1)

**Authors**: Ivan Dubrovsky, Anastasia Orlova, Illarion Iov, Nina Gubina, Irena Gureeva, Alexey Zaytsev

**Abstract**: Benchmarking outcomes increasingly govern trust, selection, and deployment of LLMs, yet these evaluations remain vulnerable to semantically equivalent adversarial perturbations. Prior work on adversarial robustness in NLP has emphasized text attacks that affect many models equally, leaving open the question of whether it is possible to selectively degrade or enhance performance while minimally affecting other models. We formalize this problem and study selective adversarial attacks on MMLU - a widely used benchmark designed to measure a language model's broad general knowledge and reasoning ability across different subjects. Using canonical attacks integrated into TextAttack framework, we introduce a protocol for selectivity assessment, develop a custom constraint to increase selectivity of attacks and propose a surrogate-LLM pipeline that generates selective perturbations. Empirically, we find that selective adversarial attacks exist and can materially alter relative rankings, challenging the fairness, reproducibility, and transparency of leaderboard-driven evaluation. Our results motivate perturbation-aware reporting and robustness diagnostics for LLM evaluation and demonstrate that even subtle edits can shift comparative judgments.

摘要: 基准结果越来越多地影响着LLM的信任、选择和部署，但这些评估仍然容易受到语义等效的对抗性扰动的影响。之前关于NLP对抗鲁棒性的工作强调了对许多模型同等影响的文本攻击，这留下了一个问题：是否可以选择性地降低或增强性能，同时对其他模型的影响最小。我们将这个问题形式化，并研究对MMLU的选择性对抗攻击，MMLU是一个广泛使用的基准，旨在衡量语言模型在不同学科中广泛的常识和推理能力。使用集成到文本攻击框架中的规范攻击，我们引入了选择性评估协议，开发自定义约束以增加攻击的选择性，并提出了生成选择性扰动的代理LLM管道。从经验上讲，我们发现存在选择性对抗性攻击，并且可以实质性地改变相对排名，挑战排行榜驱动的评估的公平性，可重复性和透明度。我们的研究结果激发了LLM评估的扰动感知报告和鲁棒性诊断，并表明即使是细微的编辑也可以改变比较判断。



## **4. Systematic Literature Review on Vehicular Collaborative Perception - A Computer Vision Perspective**

车辆协作感知的系统文献综述--计算机视觉视角 cs.CV

38 pages, 8 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2504.04631v2) [paper-pdf](http://arxiv.org/pdf/2504.04631v2)

**Authors**: Lei Wan, Jianxin Zhao, Andreas Wiedholz, Manuel Bied, Mateus Martinez de Lucena, Abhishek Dinkar Jagtap, Andreas Festag, Antônio Augusto Fröhlich, Hannan Ejaz Keen, Alexey Vinel

**Abstract**: The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.

摘要: 自动驾驶汽车的有效性依赖于可靠的感知能力。尽管人工智能和传感器融合技术取得了重大进步，但当前的单车感知系统继续遇到局限性，特别是视觉遮挡和有限的远程检测能力。由车对车（V2 V）和车对基础设施（V2 I）通信实现的协作感知（CP）已成为缓解这些问题并提高自主系统可靠性的一种有前途的解决方案。除了通信领域的进步之外，计算机视觉界越来越关注通过协作方法改善车辆感知。然而，仍然缺乏彻底审查现有工作并减少主观偏见的系统性文献审查。这种系统性方法有助于识别研究差距、识别研究中的共同趋势，并为未来的研究方向提供信息。作为回应，这项研究遵循PRISMA 2020指南，包括106篇同行评审的文章。这些出版物是根据模式、协作方案和关键感知任务进行分析的。通过比较分析，本综述说明了不同的方法如何解决实际问题，例如姿势错误、时间延迟、通信约束、域转移、异类和对抗性攻击。此外，它还批判性地审查了评估方法，强调了当前指标与CP基本目标之间的不一致。通过深入研究所有相关主题，本评论对挑战、机遇和风险提供了宝贵的见解，为推进车辆协作感知研究提供参考。



## **5. Towards Quantum Enhanced Adversarial Robustness with Rydberg Reservoir Learnin**

利用Rydberg水库学习实现量子增强对抗鲁棒性 quant-ph

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13473v1) [paper-pdf](http://arxiv.org/pdf/2510.13473v1)

**Authors**: Shehbaz Tariq, Muhammad Talha, Symeon Chatzinotas, Hyundong Shin

**Abstract**: Quantum reservoir computing (QRC) leverages the high-dimensional, nonlinear dynamics inherent in quantum many-body systems for extracting spatiotemporal patterns in sequential and time-series data with minimal training overhead. Although QRC inherits the expressive capabilities associated with quantum encodings, recent studies indicate that quantum classifiers based on variational circuits remain susceptible to adversarial perturbations. In this perspective, we investigate the first systematic evaluation of adversarial robustness in a QRC based learning model. Our reservoir comprises an array of strongly interacting Rydberg atoms governed by a fixed Hamiltonian, which naturally evolves under complex quantum dynamics, producing high-dimensional embeddings. A lightweight multilayer perceptron serves as the trainable readout layer. We utilize the balanced datasets, namely MNIST, Fashion-MNIST, and Kuzushiji-MNIST, as a benchmark for rigorously evaluating the impact of augmenting the quantum reservoir with a Multilayer perceptron (MLP) in white-box adversarial attacks to assess its robustness. We demonstrate that this approach yields significantly higher accuracy than purely classical models across all perturbation strengths tested. This hybrid approach reveals a new source of quantum advantage and

摘要: 量子储层计算（QRC）利用量子多体系统固有的多维、非线性动力学，以最小的训练负担提取顺序和时间序列数据中的时空模式。尽管QRC继承了与量子编码相关的表达能力，但最近的研究表明，基于变分电路的量子分类器仍然容易受到对抗性扰动的影响。从这个角度来看，我们研究了基于QRC的学习模型中对对抗稳健性的首次系统评估。我们的水库由一系列强相互作用的里德堡原子组成，这些原子由固定的汉密尔顿量控制，该Hamilton量在复杂的量子动力学下自然进化，产生多维嵌入。轻量级的多层感知器充当可训练的读出层。我们利用平衡数据集（即MNIST、Fashion-MNIST和Kuzushiji-MNIST）作为基准，严格评估在白盒对抗攻击中使用多层感知器（MLP）增强量子库的影响，以评估其稳健性。我们证明，在所有测试的扰动强度下，这种方法比纯经典模型产生了显着更高的准确性。这种混合方法揭示了量子优势的新来源，



## **6. Generalist++: A Meta-learning Framework for Mitigating Trade-off in Adversarial Training**

通才++：一个用于缓解对抗性培训中权衡的元学习框架 cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13361v1) [paper-pdf](http://arxiv.org/pdf/2510.13361v1)

**Authors**: Yisen Wang, Yichuan Mo, Hongjun Wang, Junyi Li, Zhouchen Lin

**Abstract**: Despite the rapid progress of neural networks, they remain highly vulnerable to adversarial examples, for which adversarial training (AT) is currently the most effective defense. While AT has been extensively studied, its practical applications expose two major limitations: natural accuracy tends to degrade significantly compared with standard training, and robustness does not transfer well across attacks crafted under different norm constraints. Unlike prior works that attempt to address only one issue within a single network, we propose to partition the overall generalization goal into multiple sub-tasks, each assigned to a dedicated base learner. By specializing in its designated objective, each base learner quickly becomes an expert in its field. In the later stages of training, we interpolate their parameters to form a knowledgeable global learner, while periodically redistributing the global parameters back to the base learners to prevent their optimization trajectories from drifting too far from the shared target. We term this framework Generalist and introduce three variants tailored to different application scenarios. Both theoretical analysis and extensive experiments demonstrate that Generalist achieves lower generalization error and significantly alleviates the trade-off problems compared with baseline methods. Our results suggest that Generalist provides a promising step toward developing fully robust classifiers in the future.

摘要: 尽管神经网络进步迅速，但它们仍然极易受到对抗性例子的影响，而对抗性训练（AT）是目前最有效的防御。虽然AT已经得到了广泛的研究，但其实际应用暴露了两个主要局限性：与标准训练相比，自然准确性往往会显着下降，并且鲁棒性在不同规范约束下设计的攻击中无法很好地转移。与之前试图在单个网络中仅解决一个问题的作品不同，我们建议将总体概括目标划分为多个子任务，每个子任务分配给一个专用的基本学习器。通过专注于其指定的目标，每个基础学习者都会很快成为其领域的专家。在训练的后期阶段，我们对它们的参数进行插值，以形成知识渊博的全局学习器，同时定期将全局参数重新分配回基本学习器，以防止它们的优化轨迹偏离共享目标太远。我们将此框架命名为Generalist，并引入了针对不同应用场景量身定制的三个变体。理论分析和大量实验都表明，与基线方法相比，Generalist实现了更低的概括误差，并显着减轻了权衡问题。我们的结果表明，Generalist为未来开发完全稳健的分类器迈出了有希望的一步。



## **7. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

SafeGuider：针对文本到图像模型的稳健且实用的内容安全控制 cs.CR

Accepted by ACM CCS 2025, Code is available at [this https  URL](https://github.com/pgqihere/safeguider)

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.05173v3) [paper-pdf](http://arxiv.org/pdf/2510.05173v3)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce SafeGuider, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, SafeGuider generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.

摘要: 文本到图像模型在从自然语言描述生成高质量图像方面表现出了非凡的能力。然而，这些模型非常容易受到对抗提示的影响，这可能会绕过安全措施并产生有害内容。尽管有各种防御策略，但在现实世界应用程序中保持实用性的同时实现针对攻击的鲁棒性仍然是一个重大挑战。为了解决这个问题，我们首先对稳定扩散（SD）模型中的文本编码器进行了实证研究，该模型是一种广泛使用且具有代表性的文本到图像模型。我们的研究结果表明，[EOS]令牌充当语义聚合器，在其嵌入空间中的良性提示和对抗提示之间表现出明显的分布模式。基于这一见解，我们引入了SafeGuider，这是一个两步框架，旨在在不影响发电质量的情况下进行稳健的安全控制。SafeGuider将嵌入级识别模型与安全意识特征擦除束搜索算法相结合。此集成使该框架能够为良性提示维持高质量图像生成，同时确保针对域内和域外攻击的强大防御。SafeGuider在最大限度地降低攻击成功率方面表现出出色的有效性，在各种攻击场景中实现的最高攻击成功率仅为5.48%。此外，SafeGuider不会拒绝为不安全提示生成或产生黑色图像，而是生成安全且有意义的图像，增强了其实际实用性。此外，SafeGuider不限于SD模型，可以有效应用于其他文本到图像模型，例如Flux模型，展示了其在不同架构中的通用性和适应性。我们希望SafeGuider能够为安全文本到图像系统的实际部署提供一些线索。



## **8. SAJA: A State-Action Joint Attack Framework on Multi-Agent Deep Reinforcement Learning**

SAJA：一个基于多智能体深度强化学习的状态-动作联合攻击框架 cs.AI

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13262v1) [paper-pdf](http://arxiv.org/pdf/2510.13262v1)

**Authors**: Weiqi Guo, Guanjun Liu, Ziyuan Zhou

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) has shown potential for cooperative and competitive tasks such as autonomous driving and strategic gaming. However, models trained by MADRL are vulnerable to adversarial perturbations on states and actions. Therefore, it is essential to investigate the robustness of MADRL models from an attack perspective. Existing studies focus on either state-only attacks or action-only attacks, but do not consider how to effectively joint them. Simply combining state and action perturbations such as randomly perturbing states and actions does not exploit their potential synergistic effects. In this paper, we propose the State-Action Joint Attack (SAJA) framework that has a good synergistic effects. SAJA consists of two important phases: (1) In the state attack phase, a multi-step gradient ascent method utilizes both the actor network and the critic network to compute an adversarial state, and (2) in the action attack phase, based on the perturbed state, a second gradient ascent uses the critic network to craft the final adversarial action. Additionally, a heuristic regularizer measuring the distance between the perturbed actions and the original clean ones is added into the loss function to enhance the effectiveness of the critic's guidance. We evaluate SAJA in the Multi-Agent Particle Environment (MPE), demonstrating that (1) it outperforms and is more stealthy than state-only or action-only attacks, and (2) existing state or action defense methods cannot defend its attacks.

摘要: 多智能体深度强化学习（MADRL）已显示出自动驾驶和战略游戏等合作和竞争任务的潜力。然而，MADRL训练的模型很容易受到状态和动作的对抗性扰动的影响。因此，从攻击的角度研究MADRL模型的稳健性至关重要。现有的研究要么关注仅针对状态的攻击，要么关注仅针对动作的攻击，但没有考虑如何有效地将它们联合起来。简单地组合状态和动作扰动（例如随机扰动状态和动作）并不能利用其潜在的协同效应。本文提出了具有良好协同效应的状态行动联合攻击（SAJA）框架。SAJA由两个重要阶段组成：（1）在状态攻击阶段，多步梯度上升方法利用行动者网络和评论者网络来计算对抗状态，（2）在动作攻击阶段，基于受干扰的状态，第二次梯度上升使用评论者网络来制作最终的对抗动作。此外，在损失函数中添加了一个启发式正规化器，用于测量受干扰的动作与原始干净的动作之间的距离，以增强评论家指导的有效性。我们在多智能体粒子环境（MBE）中评估了SAJA，证明（1）它优于仅状态或仅动作攻击，并且比仅状态攻击更隐蔽，并且（2）现有的状态或动作防御方法无法防御其攻击。



## **9. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

个人可以操纵多主体的集体决策吗？ cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2509.16494v2) [paper-pdf](http://arxiv.org/pdf/2509.16494v2)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.

摘要: 个体大型语言模型（LLM）已在医疗保健和法律等各个领域展现出强大的能力。最近的研究还表明，协调的多智能体系统通过协作表现出增强的决策和推理能力。然而，由于单个LLM的脆弱性以及访问多代理系统中所有代理的困难，出现了一个关键问题：如果攻击者只知道一个代理，他们还能生成能够误导集体决策的对抗样本吗？为了探索这个问题，我们将其描述为一个信息不完整的游戏，其中攻击者只知道一个目标代理，并且缺乏对系统中其他代理的了解。通过这个公式，我们提出了M-Spoiler，这是一个模拟多智能体系统内的智能体交互以生成对抗样本的框架。然后使用这些样本来操纵目标系统中的目标代理，误导系统的协作决策过程。更具体地说，M-Spoiler引入了一种顽固代理，它通过模拟目标系统中代理的潜在顽固反应来积极帮助优化对抗样本。这增强了生成的对抗样本误导系统的有效性。通过针对各种任务的广泛实验，我们的研究结果证实了多代理系统中单个代理的知识所带来的风险，并证明了我们框架的有效性。我们还探索了几种防御机制，表明我们提出的攻击框架仍然比基线更有效，强调了进一步研究防御策略的必要性。



## **10. Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models**

视觉-语言-动作模型的模型不可知的对抗攻击和防御 cs.CV

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13237v1) [paper-pdf](http://arxiv.org/pdf/2510.13237v1)

**Authors**: Haochuan Xu, Yun Sing Koh, Shuhuai Huang, Zirun Zhou, Di Wang, Jun Sakuma, Jingfeng Zhang

**Abstract**: Vision-Language-Action (VLA) models have achieved revolutionary progress in robot learning, enabling robots to execute complex physical robot tasks from natural language instructions. Despite this progress, their adversarial robustness remains underexplored. In this work, we propose both adversarial patch attack and corresponding defense strategies for VLA models. We first introduce the Embedding Disruption Patch Attack (EDPA), a model-agnostic adversarial attack that generates patches directly placeable within the camera's view. In comparison to prior methods, EDPA can be readily applied to different VLA models without requiring prior knowledge of the model architecture, or the controlled robotic manipulator. EDPA constructs these patches by (i) disrupting the semantic alignment between visual and textual latent representations, and (ii) maximizing the discrepancy of latent representations between adversarial and corresponding clean visual inputs. Through the optimization of these objectives, EDPA distorts the VLA's interpretation of visual information, causing the model to repeatedly generate incorrect actions and ultimately result in failure to complete the given robotic task. To counter this, we propose an adversarial fine-tuning scheme for the visual encoder, in which the encoder is optimized to produce similar latent representations for both clean and adversarially perturbed visual inputs. Extensive evaluations on the widely recognized LIBERO robotic simulation benchmark demonstrate that EDPA substantially increases the task failure rate of cutting-edge VLA models, while our proposed defense effectively mitigates this degradation. The codebase is accessible via the homepage at https://edpa-attack.github.io/.

摘要: 视觉-语言-动作（VLA）模型在机器人学习方面取得了革命性的进展，使机器人能够从自然语言指令执行复杂的物理机器人任务。尽管取得了这一进展，但它们的对抗鲁棒性仍然没有得到充分的研究。在这项工作中，我们提出了对抗补丁攻击和相应的防御策略VLA模型。我们首先介绍嵌入中断补丁攻击（EDPA），这是一种与模型无关的对抗性攻击，可以生成可直接放置在相机视图中的补丁。与现有方法相比，EDPA可以很容易地应用于不同的VLA模型，而不需要模型架构或受控机器人操纵器的先验知识。EDPA通过（i）破坏视觉和文本潜在表示之间的语义对齐，以及（ii）最大化对抗性和相应的干净视觉输入之间潜在表示的差异来构建这些补丁。通过优化这些目标，EDPA扭曲了VLA对视觉信息的解释，导致模型反复生成错误的动作，最终导致无法完成给定的机器人任务。为了解决这个问题，我们提出了一种针对视觉编码器的对抗性微调方案，其中编码器经过优化，以为干净和对抗干扰的视觉输入产生类似的潜在表示。对广泛认可的LIBERO机器人仿真基准的广泛评估表明，EDPA大大增加了尖端VLA模型的任务失败率，而我们提出的防御有效地减轻了这种退化。代码库可通过主页https://edpa-attack.github.io/访问。



## **11. SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs**

SHIELD：分类器引导的预算，实现更强大、更安全的LVLM cs.CL

Preprint

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13190v1) [paper-pdf](http://arxiv.org/pdf/2510.13190v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs.

摘要: 大型视觉语言模型（LVLM）解锁了强大的多模式推理，但也扩大了攻击面，特别是通过在良性提示中隐藏有害目标的对抗性输入。我们提出SHIELD，这是一个轻量级的、模型不可知的预处理框架，它将细粒度的安全分类与特定类别的指导和显式动作（Block、Reframe、Forward）结合起来。与二元版主不同，SHIELD编写了量身定制的安全提示，无需再培训即可强制执行细致入微的拒绝或安全重定向。在五个基准和五个有代表性的LVLM中，SHIELD持续降低越狱和不跟随率，同时保持实用性。我们的方法是即插即用的，所产生的负担可以忽略不计，并且可以轻松扩展到新的攻击类型--作为弱对齐和强对齐LVLM的实用安全补丁。



## **12. Improving Transferability of Adversarial Examples via Bayesian Attacks**

通过Bayesian攻击提高对抗性示例的可移植性 cs.LG

Accepted by TCSVT

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2307.11334v2) [paper-pdf](http://arxiv.org/pdf/2307.11334v2)

**Authors**: Qizhang Li, Yiwen Guo, Xiaochen Yang, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples allows for the attack on unknown deep neural networks (DNNs), posing a serious threat to many applications and attracting great attention. In this paper, we improve the transferability of adversarial examples by incorporating the Bayesian formulation into both the model parameters and model input, enabling their joint diversification. We demonstrate that combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability. By introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Additionally, we propose a principled approach to fine-tune model parameters within this Bayesian framework. Extensive experiments demonstrate that our method achieves a new state-of-the-art in transfer-based attacks, significantly improving the average success rate on ImageNet and CIFAR-10. Code at: https://github.com/qizhangli/MoreBayesian-jrnl.

摘要: 对抗性示例的可移植性允许对未知深度神经网络（DNN）进行攻击，对许多应用构成严重威胁并引起了极大关注。在本文中，我们通过将Bayesian公式融入模型参数和模型输入中来提高对抗性示例的可移植性，从而实现它们的联合多样化。我们证明，模型输入和模型参数的Bayesian公式的组合可以显着提高可移植性。通过在模型输入上引入后验分布的高级逼近，对抗可转移性实现了进一步的增强，在无需模型微调的情况下进行攻击时超越了所有最新技术水平。此外，我们还提出了一种有原则的方法来在此Bayesian框架内微调模型参数。大量实验表明，我们的方法在基于传输的攻击方面实现了新的最新水平，显着提高了ImageNet和CIFAR-10上的平均成功率。代码：www.example.com。



## **13. Privacy-Aware Framework of Robust Malware Detection in Indoor Robots: Hybrid Quantum Computing and Deep Neural Networks**

室内机器人鲁棒恶意软件检测的隐私感知框架：混合量子计算和深度神经网络 cs.CR

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13136v1) [paper-pdf](http://arxiv.org/pdf/2510.13136v1)

**Authors**: Tan Le, Van Le, Sachin Shetty

**Abstract**: Indoor robotic systems within Cyber-Physical Systems (CPS) are increasingly exposed to Denial of Service (DoS) attacks that compromise localization, control and telemetry integrity. We propose a privacy-aware malware detection framework for indoor robotic systems, which leverages hybrid quantum computing and deep neural networks to counter DoS threats in CPS, while preserving privacy information. By integrating quantum-enhanced feature encoding with dropout-optimized deep learning, our architecture achieves up to 95.2% detection accuracy under privacy-constrained conditions. The system operates without handcrafted thresholds or persistent beacon data, enabling scalable deployment in adversarial environments. Benchmarking reveals robust generalization, interpretability and resilience against training instability through modular circuit design. This work advances trustworthy AI for secure, autonomous CPS operations.

摘要: 网络物理系统（CPS）内的室内机器人系统越来越容易受到拒绝服务（DPS）攻击，从而损害定位、控制和遥感完整性。我们为室内机器人系统提出了一种隐私感知恶意软件检测框架，该框架利用混合量子计算和深度神经网络来对抗CPS中的DPS威胁，同时保护隐私信息。通过将量子增强特征编码与辍学优化深度学习集成，我们的架构在隐私受限的条件下实现了高达95.2%的检测准确率。该系统无需手工制作的阈值或持久信标数据即可运行，从而能够在对抗环境中进行可扩展部署。基准测试通过模块化电路设计揭示了鲁棒的概括性、可解释性和针对训练不稳定性的弹性。这项工作推动了值得信赖的人工智能，以实现安全、自主的CPS操作。



## **14. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

RedTeamCUA：混合Web-OS环境中计算机使用代理的真实对抗测试 cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2505.21936v3) [paper-pdf](http://arxiv.org/pdf/2505.21936v3)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Yuting Ning, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning high ASRs in realistic end-to-end settings, with the strongest-to-date Claude 4.5 Sonnet | CUA exhibiting the highest ASR of 60%, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.

摘要: 计算机使用代理（CUA）承诺在操作系统（OS）和网络上自动执行复杂任务，但仍然容易受到间接提示注入的影响。当前对该威胁的评估要么缺乏对现实但受控的环境的支持，要么忽视了涉及两个接口的混合Web操作系统攻击场景。为了解决这个问题，我们提出了RedTeamCUA，这是一个对抗性测试框架，具有新颖的混合沙盒，该沙盒将基于虚拟机的操作系统环境与基于Docker的Web平台集成在一起。我们的沙箱支持为红色团队定制的关键功能，例如灵活的对抗场景配置，以及通过在对抗注入时直接初始化测试来将对抗评估与CUA的导航限制分开的设置。使用RedTeamCUA，我们开发RTC-Bench，这是一个包含864个示例的综合基准测试，可以调查现实的混合Web操作系统攻击场景和基本安全漏洞。对当前前沿CUA进行基准测试发现重大漏洞：Claude 3.7十四行诗|CUA的ASB为42.9%，而受评估的最安全的CUA Operator仍为7.6%。值得注意的是，CUA经常尝试执行尝试率高达92.5%的对抗任务，尽管由于能力限制而未能完成这些任务。尽管如此，我们在现实的端到端环境中观察到了很高的ASB，其中包括迄今为止最强的Claude 4.5十四行诗|CUA表现出最高的ASB为60%，这表明CUA威胁已经可能给用户和计算机系统带来切实的风险。总体而言，RedTeamCUA为推进对CUA漏洞的现实、受控和系统性分析提供了一个重要框架，强调了在现实世界部署之前对间接提示注入进行强有力的防御的迫切需要。



## **15. SoundnessBench: A Soundness Benchmark for Neural Network Verifiers**

SoundnessBench：神经网络验证者的健全基准 cs.LG

Preprint

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2412.03154v2) [paper-pdf](http://arxiv.org/pdf/2412.03154v2)

**Authors**: Xingjian Zhou, Keyi Shen, Andy Xu, Hongji Xu, Cho-Jui Hsieh, Huan Zhang, Zhouxing Shi

**Abstract**: Neural network (NN) verification aims to formally verify properties of NNs, which is crucial for ensuring the behavior of NN-based models in safety-critical applications. In recent years, the community has developed many NN verifiers and benchmarks to evaluate them. However, existing benchmarks typically lack ground-truth for hard instances where no current verifier can verify the property and no counterexample can be found. This makes it difficult to validate the soundness of a verifier, when it claims verification on such challenging instances that no other verifier can handle. In this work, we develop a new benchmark for NN verification, named "SoundnessBench", specifically for testing the soundness of NN verifiers. SoundnessBench consists of instances with deliberately inserted counterexamples that are hidden from adversarial attacks commonly used to find counterexamples. Thereby, it can identify false verification claims when hidden counterexamples are known to exist. We design a training method to produce NNs with hidden counterexamples and systematically construct our SoundnessBench with instances across various model architectures, activation functions, and input data. We demonstrate that our training effectively produces hidden counterexamples and our SoundnessBench successfully identifies bugs in state-of-the-art NN verifiers. Our code is available at https://github.com/MVP-Harry/SoundnessBench and our benchmark is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.

摘要: 神经网络（NN）验证旨在正式验证NN的属性，这对于确保基于NN的模型在安全关键应用中的行为至关重要。近年来，社区开发了许多NN验证器和基准来评估它们。然而，现有的基准测试通常缺乏针对当前验证者可以验证属性且找不到反例的硬实例的基本事实。这使得当验证者声称对没有其他验证者可以处理的具有挑战性的实例进行验证时，很难验证者的合理性。在这项工作中，我们开发了一个新的NN验证基准，名为“SoundnessBench”，专门用于测试NN验证器的可靠性。SoundnessBench由带有故意插入反例的实例组成，这些反例隐藏在通常用于寻找反例的对抗性攻击中。因此，当已知存在隐藏反例时，它可以识别错误的验证声明。我们设计了一种训练方法来生成具有隐藏反例的NN，并使用跨各种模型架构、激活函数和输入数据的实例系统地构建我们的SoundnessBench。我们证明我们的训练有效地产生了隐藏的反例，并且我们的SoundnessBench成功地识别了最先进的NN验证器中的错误。我们的代码可在https://github.com/MVP-Harry/SoundnessBench上获取，我们的基准可在https://huggingface.co/datasets/SoundnessBench/SoundnessBench上获取。



## **16. A Survey of Graph Unlearning**

图形遗忘研究综述 cs.LG

15 page review paper on graph unlearning

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2310.02164v4) [paper-pdf](http://arxiv.org/pdf/2310.02164v4)

**Authors**: Anwar Said, Ngoc N. Tran, Yuying Zhao, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the \textit{right to be forgotten}. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, recommender systems, and resource-constrained environments like the Internet of Things, illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.

摘要: 图形取消学习是追求负责任的人工智能的一项关键进步，它提供了从训练模型中删除敏感数据痕迹的方法，从而维护了\textit{被遗忘权}。显然，图机器学习对数据隐私和对抗攻击表现出敏感性，因此需要应用图去学习技术来有效解决这些问题。在这篇全面的调查论文中，我们首次对图形学习方法进行了系统性回顾，涵盖了多种方法，并提供了详细的分类学和最新的文献概述，以促进对该领域新接触的研究人员的理解。为了确保清晰度，我们对图形取消学习中使用的基本概念和评估指标进行了清晰的解释，以迎合具有不同专业知识水平的更广泛受众。通过深入研究潜在的应用，我们探索了图去学习在各个领域的多功能性，包括但不限于社交网络、对抗性环境、推荐系统和物联网等资源受限环境，说明了其在保护数据隐私和增强人工智能系统稳健性方面的潜在影响。最后，我们揭示了有前途的研究方向，鼓励在图去学习领域取得进一步的进步和创新。通过奠定坚实的基础和促进持续进步，这项调查旨在激励研究人员进一步推进图学习领域，从而为人工智能系统的道德发展注入信心，并加强机器学习技术在各个领域的负责任应用。



## **17. KoALA: KL-L0 Adversarial Detector via Label Agreement**

KoALA：通过标签协议的KL-L0对抗检测器 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12752v1) [paper-pdf](http://arxiv.org/pdf/2510.12752v1)

**Authors**: Siqi Li, Yasser Shoukry

**Abstract**: Deep neural networks are highly susceptible to adversarial attacks, which pose significant risks to security- and safety-critical applications. We present KoALA (KL-L0 Adversarial detection via Label Agreement), a novel, semantics-free adversarial detector that requires no architectural changes or adversarial retraining. KoALA operates on a simple principle: it detects an adversarial attack when class predictions from two complementary similarity metrics disagree. These metrics-KL divergence and an L0-based similarity-are specifically chosen to detect different types of perturbations. The KL divergence metric is sensitive to dense, low-amplitude shifts, while the L0-based similarity is designed for sparse, high-impact changes. We provide a formal proof of correctness for our approach. The only training required is a simple fine-tuning step on a pre-trained image encoder using clean images to ensure the embeddings align well with both metrics. This makes KOALA a lightweight, plug-and-play solution for existing models and various data modalities. Our extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet confirm our theoretical claims. When the theorem's conditions are met, KoALA consistently and effectively detects adversarial examples. On the full test sets, KoALA achieves a precision of 0.94 and a recall of 0.81 on ResNet/CIFAR-10, and a precision of 0.66 and a recall of 0.85 on CLIP/Tiny-ImageNet.

摘要: 深度神经网络极易受到对抗攻击，这对安全和安全关键应用程序构成重大风险。我们提出了KoALA（通过标签协议进行KL-L0对抗性检测），这是一种新颖的、无语义的对抗性检测器，不需要架构更改或对抗性再培训。KoALA的工作原理很简单：当来自两个互补相似性指标的类预测不一致时，它会检测到对抗攻击。这些指标（KL偏差和基于L0的相似性）是专门选择的，以检测不同类型的扰动。KL背离指标对密集、低幅度的漂移敏感，而基于L0的相似性则针对稀疏、高影响的变化而设计。我们为我们的方法提供了正确性的正式证明。唯一需要的训练是使用干净图像对预训练的图像编码器进行简单的微调步骤，以确保嵌入与两个指标良好一致。这使得KOALA成为现有模型和各种数据模式的轻量级、即插即用解决方案。我们对ResNet/CIFAR-10和CLIP/Tiny-ImageNet的广泛实验证实了我们的理论主张。当满足该定理的条件时，KoALA会一致有效地检测对抗性示例。在完整测试集上，KoALA在ResNet/CIFAR-10上的精确度为0.94，召回率为0.81，在CLIP/Tiny-ImageNet上的精确度为0.66，召回率为0.85。



## **18. Towards Robust Artificial Intelligence: Self-Supervised Learning Approach for Out-of-Distribution Detection**

迈向稳健的人工智能：用于分布外检测的自我监督学习方法 cs.AI

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12713v1) [paper-pdf](http://arxiv.org/pdf/2510.12713v1)

**Authors**: Wissam Salhab, Darine Ameyed, Hamid Mcheick, Fehmi Jaafar

**Abstract**: Robustness in AI systems refers to their ability to maintain reliable and accurate performance under various conditions, including out-of-distribution (OOD) samples, adversarial attacks, and environmental changes. This is crucial in safety-critical systems, such as autonomous vehicles, transportation, or healthcare, where malfunctions could have severe consequences. This paper proposes an approach to improve OOD detection without the need of labeled data, thereby increasing the AI systems' robustness. The proposed approach leverages the principles of self-supervised learning, allowing the model to learn useful representations from unlabeled data. Combined with graph-theoretical techniques, this enables the more efficient identification and categorization of OOD samples. Compared to existing state-of-the-art methods, this approach achieved an Area Under the Receiver Operating Characteristic Curve (AUROC) = 0.99.

摘要: 人工智能系统的鲁棒性是指它们在各种条件下保持可靠和准确性能的能力，包括分发外（OOD）样本、对抗性攻击和环境变化。这对于自动驾驶汽车、交通或医疗保健等安全关键系统至关重要，这些系统的故障可能会造成严重后果。本文提出了一种在不需要标记数据的情况下改进OOD检测的方法，从而提高人工智能系统的鲁棒性。所提出的方法利用了自我监督学习的原则，使模型能够从未标记的数据中学习有用的表示。结合图论技术，这使得更有效的识别和分类的OOD样本。与现有的最先进的方法相比，该方法实现了受试者工作特征曲线下面积（AUROC）= 0.99。



## **19. Keep Calm and Avoid Harmful Content: Concept Alignment and Latent Manipulation Towards Safer Answers**

保持冷静并避免有害内容：概念一致和潜在操纵以获得更安全的答案 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12672v1) [paper-pdf](http://arxiv.org/pdf/2510.12672v1)

**Authors**: Ruben Belo, Claudia Soares, Marta Guimaraes

**Abstract**: Large Language Models are susceptible to jailbreak attacks that bypass built-in safety guardrails (e.g., by tricking the model with adversarial prompts). We propose Concept Alignment and Concept Manipulation \textbf{CALM}, an inference-time method that suppresses harmful concepts by modifying latent representations of the last layer of the model, without retraining. Leveraging \gls*{cw} technique from Computer Vision combined with orthogonal projection, CALM removes unwanted latent directions associated with harmful content while preserving model performance. Experiments show that CALM reduces harmful outputs and outperforms baseline methods in most metrics, offering a lightweight approach to AI safety with no additional training data or model fine-tuning, while incurring only a small computational overhead at inference.

摘要: 大型语言模型容易受到绕过内置安全护栏的越狱攻击（例如，通过用对抗性提示欺骗模型）。我们提出概念对齐和概念操纵\textBF{CALM}，这是一种推理时方法，通过修改模型最后一层的潜在表示来抑制有害概念，无需重新训练。利用计算机视觉中的\gls*{cw}技术与垂直投影相结合，CALM可以删除与有害内容相关的不需要的潜在方向，同时保留模型性能。实验表明，CALM减少了有害输出，并在大多数指标上优于基线方法，为人工智能安全提供了一种轻量级方法，无需额外的训练数据或模型微调，同时在推理时只产生很小的计算负担。



## **20. Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain**

Agentland的恶意：深入人工智能供应链后门的兔子洞 cs.CR

27 pages

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.05159v2) [paper-pdf](http://arxiv.org/pdf/2510.05159v2)

**Authors**: Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin

**Abstract**: The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.

摘要: 根据人工智能代理自身交互中的数据（例如网络浏览或工具使用）进行微调的做法虽然是提高代理能力的强大通用配方，但也在人工智能供应链中引入了一个关键的安全漏洞。在这项工作中，我们表明，对手可以很容易地毒害数据收集管道，以嵌入被特定目标短语触发的难以检测的后门，这样当代理遇到这些触发器时，就会执行不安全或恶意的操作。我们形式化并验证了三种针对供应链不同层的现实威胁模型：1）微调数据的直接中毒，其中攻击者控制了一小部分训练痕迹; 2）环境中毒，其中恶意指令被注入到创建训练数据时抓取的网页或调用的工具中; 3）供应链中毒，即根据干净的数据对预先后门的基础模型进行微调，以提高其代理能力。我们的结果很明显：攻击者可以通过毒害仅2%的收集痕迹，嵌入后门，导致代理在存在特定触发器时泄露机密用户信息，成功率超过80%。该漏洞适用于所有三种威胁模型。此外，我们证明，包括两种护栏模型和一种基于重量的防御在内的主要防护措施无法检测或防止恶意行为。这些发现凸显了代理人工智能开发面临的紧迫威胁，并强调了对数据收集流程和端到端模型供应链进行严格安全审查的迫切需要。



## **21. PEAR: Planner-Executor Agent Robustness Benchmark**

PEAR：规划者-执行者代理稳健性基准 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07505v2) [paper-pdf](http://arxiv.org/pdf/2510.07505v2)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）已成为处理跨不同领域复杂、多步骤任务的强大范式。然而，尽管MAS的能力令人印象深刻，但仍然容易受到对抗操纵。现有的研究通常会检查孤立的攻击表面或特定场景，从而缺乏对MAS漏洞的全面了解。为了弥合这一差距，我们引入了PEAR，这是一个用于系统评估规划者-执行者MAS的实用性和脆弱性的基准。虽然兼容各种MAS体系结构，我们的基准集中在规划者-执行器结构，这是一个实用的和广泛采用的设计。通过大量的实验，我们发现：（1）弱规划器比弱执行器更严重地降低了清洁任务的整体性能;（2）虽然规划器的内存模块是必不可少的，但执行器的内存模块并不影响清洁任务的性能;（3）任务性能和鲁棒性之间存在权衡;以及（4）针对计划者的攻击在误导系统方面特别有效。这些发现提供了可操作的见解，提高MAS的鲁棒性，并奠定了基础，在多智能体设置的原则性防御。



## **22. Proof of Cloud: Data Center Execution Assurance for Confidential VMs**

云证明：机密虚拟机的数据中心执行保证 cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12469v1) [paper-pdf](http://arxiv.org/pdf/2510.12469v1)

**Authors**: Filip Rezabek, Moe Mahhouk, Andrew Miller, Stefan Genchev, Quintus Kilbourn, Georg Carle, Jonathan Passerat-Palmbach

**Abstract**: Confidential Virtual Machines (CVMs) protect data in use by running workloads inside hardware-isolated environments. In doing so, they also inherit the limitations of the underlying hardware. Trusted Execution Environments (TEEs), which enforce this isolation, explicitly exclude adversaries with physical access from their threat model. Commercial TEEs, e.g., Intel TDX, thus assume infrastructure providers do not physically exploit hardware and serve as safeguards instead. This creates a tension: tenants must trust provider integrity at the hardware layer, yet existing remote attestation offers no way to verify that CVMs actually run on physically trusted platforms, leaving today's CVM deployments unable to demonstrate that their guarantees align with the TEE vendor's threat model.   We bridge this confidence gap with Data Center Execution Assurance (DCEA), a design generating "Proofs of Cloud". DCEA binds a CVM to its underlying platform using vTPM-anchored measurements, ensuring CVM launch evidence and TPM quotes refer to the same physical chassis.   This takes advantage of the fact that data centers are often identifiable via TPMs. Our approach applies to CVMs accessing vTPMs and running on top of software stacks fully controlled by the cloud provider, as well as single-tenant bare-metal deployments with discrete TPMs. We trust providers for integrity (certificate issuance), but not for the confidentiality of CVM-visible state. DCEA enables remote verification of a CVM's platform origin and integrity, mitigating attacks like replay and attestation proxying. We include a candidate implementation on Google Cloud and Intel TDX that leverages Intel TXT for trusted launch. Our design refines CVMs' threat model and provides a practical path for deploying high-assurance, confidential workloads in minimally trusted environments.

摘要: 机密虚拟机（CVM）通过在硬件隔离环境中运行工作负载来保护使用中的数据。在这样做时，它们也继承了底层硬件的限制。可信执行环境（TEE）强制执行这种隔离，明确地将具有物理访问权限的对手排除在其威胁模型之外。商业TEE，例如，因此，英特尔TDX假设基础设施提供商不会实际利用硬件，而是作为安全措施。这就造成了一种紧张：租户必须在硬件层信任提供商的完整性，但现有的远程证明无法验证云服务器是否实际运行在物理上受信任的平台上，这使得目前的云服务器部署无法证明其保证与TEE供应商的威胁模型一致。   我们通过数据中心执行保证（DSCA）来弥合这一信心差距，这是一种生成“云证明”的设计。DCAE使用vTPS锚定的测量将CGM绑定到其底层平台，确保CGM启动证据和BPM引用引用相同的物理底盘。   这利用了数据中心通常可以通过TPS识别的事实。我们的方法适用于访问vTPS并在完全由云提供商控制的软件堆栈上运行的CGM，以及具有离散TPS的单租户纯金部署。我们信任提供商的完整性（证书发布），但不信任CVS可见状态的机密性。DCAE支持远程验证CGM的平台起源和完整性，减轻重播和证明虚拟化等攻击。我们在Google Cloud和英特尔RDX上提供了一个候选实施，该实施利用英特尔XT进行可信的发布。我们的设计完善了云服务器的威胁模型，并提供了在最低信任环境中部署高保证、保密的工作负载的实用路径。



## **23. MS-GAGA: Metric-Selective Guided Adversarial Generation Attack**

MS-GAGA：度量选择引导的对抗生成攻击 cs.CV

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12468v1) [paper-pdf](http://arxiv.org/pdf/2510.12468v1)

**Authors**: Dion J. X. Ho, Gabriel Lee Jun Rong, Niharika Shrivastava, Harshavardhan Abichandani, Pai Chet Ng, Xiaoxiao Miao

**Abstract**: We present MS-GAGA (Metric-Selective Guided Adversarial Generation Attack), a two-stage framework for crafting transferable and visually imperceptible adversarial examples against deepfake detectors in black-box settings. In Stage 1, a dual-stream attack module generates adversarial candidates: MNTD-PGD applies enhanced gradient calculations optimized for small perturbation budgets, while SG-PGD focuses perturbations on visually salient regions. This complementary design expands the adversarial search space and improves transferability across unseen models. In Stage 2, a metric-aware selection module evaluates candidates based on both their success against black-box models and their structural similarity (SSIM) to the original image. By jointly optimizing transferability and imperceptibility, MS-GAGA achieves up to 27% higher misclassification rates on unseen detectors compared to state-of-the-art attacks.

摘要: 我们提出了MS-GAGA（度量选择性引导对抗生成攻击），这是一个两阶段框架，用于针对黑匣子环境中的Deepfake检测器制作可转移且视觉上不可感知的对抗示例。在第一阶段，双流攻击模块生成对抗候选者：MNTD-PVD应用针对小扰动预算进行优化的增强的梯度计算，而SG-PVD将扰动集中在视觉上的突出区域。这种补充设计扩大了对抗搜索空间，并提高了跨未见过模型的可移植性。在第二阶段，指标感知的选择模块根据候选者对抗黑匣子模型的成功及其与原始图像的结构相似性（SSIM）来评估候选者。通过联合优化可转移性和不可感知性，与最先进的攻击相比，MS-GAGA在不可见检测器上实现了高达27%的错误分类率。



## **24. IP-Augmented Multi-Modal Malicious URL Detection Via Token-Contrastive Representation Enhancement and Multi-Granularity Fusion**

通过令牌对比表示增强和多粒度融合进行IP增强多模式恶意URL检测 cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12395v1) [paper-pdf](http://arxiv.org/pdf/2510.12395v1)

**Authors**: Ye Tian, Yanqiu Yu, Liangliang Song, Zhiquan Liu, Yanbin Wang, Jianguo Sun

**Abstract**: Malicious URL detection remains a critical cybersecurity challenge as adversaries increasingly employ sophisticated evasion techniques including obfuscation, character-level perturbations, and adversarial attacks. Although pre-trained language models (PLMs) like BERT have shown potential for URL analysis tasks, three limitations persist in current implementations: (1) inability to effectively model the non-natural hierarchical structure of URLs, (2) insufficient sensitivity to character-level obfuscation, and (3) lack of mechanisms to incorporate auxiliary network-level signals such as IP addresses-all essential for robust detection. To address these challenges, we propose CURL-IP, an advanced multi-modal detection framework incorporating three key innovations: (1) Token-Contrastive Representation Enhancer, which enhances subword token representations through token-aware contrastive learning to produce more discriminative and isotropic embeddings; (2) Cross-Layer Multi-Scale Aggregator, employing hierarchical aggregation of Transformer outputs via convolutional operations and gated MLPs to capture both local and global semantic patterns across layers; and (3) Blockwise Multi-Modal Coupler that decomposes URL-IP features into localized block units and computes cross-modal attention weights at the block level, enabling fine-grained inter-modal interaction. This architecture enables simultaneous preservation of fine-grained lexical cues, contextual semantics, and integration of network-level signals. Our evaluation on large-scale real-world datasets shows the framework significantly outperforms state-of-the-art baselines across binary and multi-class classification tasks.

摘要: 恶意URL检测仍然是一个关键的网络安全挑战，因为对手越来越多地使用复杂的规避技术，包括混淆、字符级扰动和对抗性攻击。尽管BERT等预训练语言模型（PLM）已显示出URL分析任务的潜力，但当前的实现中仍然存在三个局限性：（1）无法有效地建模URL的非自然分层结构，（2）对字符级混淆的敏感性不足，（3）缺乏纳入辅助网络级信号（例如IP地址）的机制--所有这些对于鲁棒检测来说都至关重要。为了应对这些挑战，我们提出了CROL-IP，这是一种先进的多模式检测框架，融合了三项关键创新：（1）令牌对比表示增强器，它通过令牌感知的对比学习来增强子词令牌表示，以产生更具区分性和各向同性的嵌入;（2）跨层多尺度聚合器，通过卷积运算和门控MLP采用Transformer输出的分层聚合来跨层捕获局部和全局语义模式;和（3）绑定多模式耦合器，将URL-IP特征分解为局部块单元，并计算块级别的跨模式注意力权重，从而实现细粒度的模式间交互。该架构能够同时保存细粒度的词汇线索、上下文语义和网络级信号的集成。我们对大规模现实世界数据集的评估表明，该框架在二元和多类分类任务中的表现显着优于最先进的基线。



## **25. DeepTrust: Multi-Step Classification through Dissimilar Adversarial Representations for Robust Android Malware Detection**

DeepTrust：通过不同的对抗表示进行多步骤分类，以实现稳健的Android恶意软件检测 cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12310v1) [paper-pdf](http://arxiv.org/pdf/2510.12310v1)

**Authors**: Daniel Pulido-Cortázar, Daniel Gibert, Felip Manyà

**Abstract**: Over the last decade, machine learning has been extensively applied to identify malicious Android applications. However, such approaches remain vulnerable against adversarial examples, i.e., examples that are subtly manipulated to fool a machine learning model into making incorrect predictions. This research presents DeepTrust, a novel metaheuristic that arranges flexible classifiers, like deep neural networks, into an ordered sequence where the final decision is made by a single internal model based on conditions activated in cascade. In the Robust Android Malware Detection competition at the 2025 IEEE Conference SaTML, DeepTrust secured the first place and achieved state-of-the-art results, outperforming the next-best competitor by up to 266% under feature-space evasion attacks. This is accomplished while maintaining the highest detection rate on non-adversarial malware and a false positive rate below 1%. The method's efficacy stems from maximizing the divergence of the learned representations among the internal models. By using classifiers inducing fundamentally dissimilar embeddings of the data, the decision space becomes unpredictable for an attacker. This frustrates the iterative perturbation process inherent to evasion attacks, enhancing system robustness without compromising accuracy on clean examples.

摘要: 在过去的十年中，机器学习已被广泛应用于识别恶意Android应用程序。然而，此类方法仍然容易受到对抗性例子的影响，即这些例子被巧妙操纵以欺骗机器学习模型做出错误的预测。这项研究提出了DeepTrust，这是一种新型的元启发式方法，它将深度神经网络等灵活分类器安排成有序序列，最终决策由单个内部模型根据级联激活的条件做出。在2025年IEEE会议SaTML的稳健Android恶意软件检测竞赛中，DeepTrust获得了第一名，并取得了最先进的成绩，在功能空间规避攻击下，表现优于次佳竞争对手高达266%。实现这一目标的同时保持对非对抗性恶意软件的最高检测率和低于1%的假阳性率。该方法的功效源于最大化内部模型之间所学习的表示的分歧。通过使用分类器引发根本不同的数据嵌入，决策空间对于攻击者来说变得不可预测。这会挫败规避攻击固有的迭代扰动过程，增强系统稳健性，而不会损害干净示例的准确性。



## **26. Train Stochastic Non Linear Coupled ODEs to Classify and Generate**

训练随机非线性耦合ODE进行分类和生成 cond-mat.dis-nn

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12286v1) [paper-pdf](http://arxiv.org/pdf/2510.12286v1)

**Authors**: Stefano Gagliani, Feliciano Giuseppe Pacifico, Lorenzo Chicchi, Duccio Fanelli, Diego Febbe, Lorenzo Buffoni, Raffaele Marino

**Abstract**: A general class of dynamical systems which can be trained to operate in classification and generation modes are introduced. A procedure is proposed to plant asymptotic stationary attractors of the deterministic model. Optimizing the dynamical system amounts to shaping the architecture of inter-nodes connection to steer the evolution towards the assigned equilibrium, as a function of the class to which the item - supplied as an initial condition - belongs to. Under the stochastic perspective, point attractors are turned into probability distributions, made analytically accessible via the linear noise approximation. The addition of noise proves beneficial to oppose adversarial attacks, a property that gets engraved into the trained adjacency matrix and therefore also inherited by the deterministic counterpart of the optimized stochastic model. By providing samples from the target distribution as an input to a feedforward neural network (or even to a dynamical model of the same typology of the adopted for classification purposes), yields a fully generative scheme. Conditional generation is also possible by merging classification and generation modalities. Automatic disentanglement of isolated key features is finally proven.

摘要: 介绍了一类一般的动态系统，可以训练它们以分类和生成模式运行。提出了一种方法来种植确定性模型的渐进平稳吸引子。优化动态系统相当于塑造节点间连接的架构，以引导演变向指定平衡，作为物品（作为初始条件提供）所属类别的函数。在随机的角度下，点吸引子被转化为概率分布，通过线性噪音逼近进行分析访问。噪声的添加被证明有利于对抗对抗性攻击，这是一种刻在训练的邻接矩阵中的属性，因此也被优化的随机模型的确定性对应物继承。通过将来自目标分布的样本作为输入提供给前馈神经网络（或者甚至提供给用于分类目的的相同类型的动态模型），产生完全生成的方案。条件生成也可以通过合并分类和生成模态来实现。孤立的关键特征的自动解纠缠最终得到证明。



## **27. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

L2 M-AID：通过融合大型语言模型的语义推理与多智能体强化学习来自主网络物理防御（预印本） cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07363v2) [paper-pdf](http://arxiv.org/pdf/2510.07363v2)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.

摘要: 工业物联网（IIoT）的日益集成使关键的网络物理系统面临复杂的多阶段攻击，这些攻击无法逃避缺乏上下文感知的传统防御。本文介绍了L2 M-AID，这是一种新型的自主工业防御框架，使用LLM授权的多智能体强化学习。L2 M-AID组织了一个协作代理团队，每个代理都由大型语言模型（LLM）驱动，以实现自适应和弹性的安全性。核心创新在于两种人工智能范式的深度融合：我们利用LLM作为语义桥梁，将庞大的非结构化遥感数据转化为丰富的上下文状态表示，使代理能够推理对手意图，而不仅仅是匹配模式。这种语义感知状态使多智能体强化学习（MARL）算法MAPPO能够学习复杂的合作策略。MARL奖励功能经过独特设计，旨在平衡安全目标（威胁消除）与运营必要性，明确惩罚破坏物理过程稳定性的行为。为了验证我们的方法，我们对基准SWaT数据集和基于MITRE ATA & CK for ICS框架生成的新型合成数据集进行了广泛的实验。结果表明，L2 M-AID在关键指标上的表现显着优于传统IDS、深度学习异常检测器和单代理RL基线，实现了97.2%的检测率，同时将误报率降低了80%以上，并将响应时间提高了四倍。至关重要的是，它在维持物理过程稳定性方面表现出色，为保护关键国家基础设施提供了强大的新范式。



## **28. Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Adversarial Attack on TAGs**

揭开Graph-LLM的漏洞：对TAG的可解释多维对抗攻击 cs.LG

12 pages, 4 figures

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12233v1) [paper-pdf](http://arxiv.org/pdf/2510.12233v1)

**Authors**: Bowen Fan, Zhilin Guo, Xunkai Li, Yihan Zhou, Bing Zhou, Zhenjun Li, Rong-Hua Li, Guoren Wang

**Abstract**: Graph Neural Networks (GNNs) have become a pivotal framework for modeling graph-structured data, enabling a wide range of applications from social network analysis to molecular chemistry. By integrating large language models (LLMs), text-attributed graphs (TAGs) enhance node representations with rich textual semantics, significantly boosting the expressive power of graph-based learning. However, this sophisticated synergy introduces critical vulnerabilities, as Graph-LLMs are susceptible to adversarial attacks on both their structural topology and textual attributes. Although specialized attack methods have been designed for each of these aspects, no work has yet unified them into a comprehensive approach. In this work, we propose the Interpretable Multi-Dimensional Graph Attack (IMDGA), a novel human-centric adversarial attack framework designed to orchestrate multi-level perturbations across both graph structure and textual features. IMDGA utilizes three tightly integrated modules to craft attacks that balance interpretability and impact, enabling a deeper understanding of Graph-LLM vulnerabilities. Through rigorous theoretical analysis and comprehensive empirical evaluations on diverse datasets and architectures, IMDGA demonstrates superior interpretability, attack effectiveness, stealthiness, and robustness compared to existing methods. By exposing critical weaknesses in TAG representation learning, this work uncovers a previously underexplored semantic dimension of vulnerability in Graph-LLMs, offering valuable insights for improving their resilience. Our code and resources are publicly available at https://anonymous.4open.science/r/IMDGA-7289.

摘要: 图神经网络（GNN）已经成为对图结构数据进行建模的关键框架，能够实现从社交网络分析到分子化学的广泛应用。通过集成大型语言模型（LLM），文本属性图（TAG）增强了具有丰富文本语义的节点表示，显着提高了基于图的学习的表达能力。然而，这种复杂的协同作用引入了关键的漏洞，因为Graph-LLM容易受到对其结构拓扑和文本属性的对抗性攻击。虽然专门的攻击方法已被设计用于这些方面的每一个，还没有工作将它们统一成一个全面的方法。在这项工作中，我们提出了可解释多维图攻击（IMDGA），这是一种新型的以人为中心的对抗攻击框架，旨在协调跨图结构和文本特征的多层扰动。IMDGA利用三个紧密集成的模块来设计平衡可解释性和影响的攻击，从而能够更深入地了解Graph-LLM漏洞。通过对不同数据集和架构进行严格的理论分析和全面的实证评估，IMDGA展示了与现有方法相比更出色的可解释性、攻击有效性、隐蔽性和鲁棒性。通过揭露TAG表示学习中的关键弱点，这项工作揭示了Graph-LLM中先前未充分探索的漏洞语义维度，为提高其弹性提供了宝贵的见解。我们的代码和资源可在https://anonymous.4open.science/r/IMDGA-7289上公开获取。



## **29. How Vulnerable Is My Learned Policy? Universal Adversarial Perturbation Attacks On Modern Behavior Cloning Policies**

我的习得政策有多脆弱？对现代行为克隆政策的普遍对抗性扰动攻击 cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2502.03698v3) [paper-pdf](http://arxiv.org/pdf/2502.03698v3)

**Authors**: Akansha Kalra, Basavasagar Patil, Guanhong Tao, Daniel S. Brown

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to offline universal perturbation attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and Vector-Quantizied Behavior Transformer (VQ-BET). We study the vulnerability of these methods to universal adversarial perturbations. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are often transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities to black-box attacks. To the best of our knowledge, we are the first to present a systematic study of the vulnerabilities of different LfD algorithms to both white-box and black-box attacks. Our findings highlight the vulnerabilities of modern BC algorithms, paving the way for future work in addressing such limitations.

摘要: 从演示中学习（LfD）算法在机器人操纵任务中显示出有希望的结果，但其对离线普遍扰动攻击的脆弱性仍然没有得到充分的研究。本文全面研究了对经典算法和最近提出的算法的对抗攻击，包括行为克隆（BC）、LSTM-GMM、隐式行为克隆（IBC）、扩散策略（DP）和Vector-Quanized Behavior Transformer（VQ-BET）。我们研究这些方法对普遍对抗性扰动的脆弱性。我们对几个模拟机器人操纵任务的实验表明，当前的大多数方法都极易受到对抗性扰动的影响。我们还表明，这些攻击通常可以跨算法、架构和任务转移，从而引发了黑匣子攻击的安全漏洞。据我们所知，我们是第一个对不同LfD算法对白盒和黑盒攻击的脆弱性进行系统研究的人。我们的研究结果凸显了现代BC算法的漏洞，为未来解决此类限制的工作铺平了道路。



## **30. Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing**

使用高级文本预处理对传统、LLM生成和对抗性网络钓鱼电子邮件进行稳健的基于ML的检测 cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11915v1) [paper-pdf](http://arxiv.org/pdf/2510.11915v1)

**Authors**: Deeksha Hareesha Kulal, Chidozie Princewill Arannonu, Afsah Anwar, Nidhi Rastogi, Quamar Niyaz

**Abstract**: Phishing remains a critical cybersecurity threat, especially with the advent of large language models (LLMs) capable of generating highly convincing malicious content. Unlike earlier phishing attempts which are identifiable by grammatical errors, misspellings, incorrect phrasing, and inconsistent formatting, LLM generated emails are grammatically sound, contextually relevant, and linguistically natural. These advancements make phishing emails increasingly difficult to distinguish from legitimate ones, challenging traditional detection mechanisms. Conventional phishing detection systems often fail when faced with emails crafted by LLMs or manipulated using adversarial perturbation techniques. To address this challenge, we propose a robust phishing email detection system featuring an enhanced text preprocessing pipeline. This pipeline includes spelling correction and word splitting to counteract adversarial modifications and improve detection accuracy. Our approach integrates widely adopted natural language processing (NLP) feature extraction techniques and machine learning algorithms. We evaluate our models on publicly available datasets comprising both phishing and legitimate emails, achieving a detection accuracy of 94.26% and F1-score of 84.39% in model deployment setting. To assess robustness, we further evaluate our models using adversarial phishing samples generated by four attack methods in Python TextAttack framework. Additionally, we evaluate models' performance against phishing emails generated by LLMs including ChatGPT and Llama. Results highlight the resilience of models against evolving AI-powered phishing threats.

摘要: 网络钓鱼仍然是一个严重的网络安全威胁，特别是随着能够生成高度令人信服的恶意内容的大型语言模型（LLM）的出现。与早期的网络钓鱼尝试（可通过语法错误、拼写错误、措辞不正确和格式不一致）不同，LLM生成的电子邮件语法健全、上下文相关且语言自然。这些进步使得网络钓鱼电子邮件越来越难以与合法电子邮件区分开来，从而挑战了传统的检测机制。当面对由LLM制作或使用对抗性干扰技术操纵的电子邮件时，传统的网络钓鱼检测系统通常会失败。为了应对这一挑战，我们提出了一种强大的网络钓鱼电子邮件检测系统，具有增强的文本预处理管道。该管道包括拼写纠正和单词拆分，以抵消对抗性修改并提高检测准确性。我们的方法集成了广泛采用的自然语言处理（NLP）特征提取技术和机器学习算法。我们在包括网络钓鱼和合法电子邮件的公开数据集上评估了我们的模型，在模型部署设置中实现了94.26%的检测准确率和84.39%的F1评分。为了评估稳健性，我们使用Python文本攻击框架中四种攻击方法生成的对抗性网络钓鱼样本进一步评估我们的模型。此外，我们还评估模型针对ChatGPT和Llama等LLM生成的网络钓鱼电子邮件的性能。结果凸显了模型对不断变化的人工智能驱动网络钓鱼威胁的弹性。



## **31. A Comprehensive Survey of Website Fingerprinting Attacks and Defenses in Tor: Advances and Open Challenges**

Tor网站指纹攻击和防御的全面调查：进展和开放挑战 cs.CR

43 pages

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11804v1) [paper-pdf](http://arxiv.org/pdf/2510.11804v1)

**Authors**: Yuwen Cui, Guangjing Wang, Khanh Vu, Kai Wei, Kehan Shen, Zhengyuan Jiang, Xiao Han, Ning Wang, Zhuo Lu, Yao Liu

**Abstract**: The Tor network provides users with strong anonymity by routing their internet traffic through multiple relays. While Tor encrypts traffic and hides IP addresses, it remains vulnerable to traffic analysis attacks such as the website fingerprinting (WF) attack, achieving increasingly high fingerprinting accuracy even under open-world conditions. In response, researchers have proposed a variety of defenses, ranging from adaptive padding, traffic regularization, and traffic morphing to adversarial perturbation, that seek to obfuscate or reshape traffic traces. However, these defenses often entail trade-offs between privacy, usability, and system performance. Despite extensive research, a comprehensive survey unifying WF datasets, attack methodologies, and defense strategies remains absent. This paper fills that gap by systematically categorizing existing WF research into three key domains: datasets, attack models, and defense mechanisms. We provide an in-depth comparative analysis of techniques, highlight their strengths and limitations under diverse threat models, and discuss emerging challenges such as multi-tab browsing and coarse-grained traffic features. By consolidating prior work and identifying open research directions, this survey serves as a foundation for advancing stronger privacy protection in Tor.

摘要: Tor网络通过多个中继路由用户的互联网流量，为用户提供了强大的匿名性。虽然Tor加密流量并隐藏IP地址，但它仍然容易受到网站指纹识别（WF）攻击等流量分析攻击，即使在开放世界条件下也能实现越来越高的指纹识别准确性。作为回应，研究人员提出了各种防御措施，从自适应填充、流量规则化、流量变形到对抗性扰动，旨在模糊或重塑流量轨迹。然而，这些防御通常需要在隐私、可用性和系统性能之间进行权衡。尽管进行了广泛的研究，但仍然缺乏统一WF数据集、攻击方法和防御策略的全面调查。本文通过将现有的WF研究系统地分类为三个关键领域：数据集、攻击模型和防御机制来填补这一空白。我们对技术进行深入的比较分析，强调它们在不同威胁模型下的优势和局限性，并讨论多选项卡浏览和粗粒度流量功能等新出现的挑战。通过整合之前的工作并确定开放的研究方向，这项调查为在Tor中推进更强有力的隐私保护奠定了基础。



## **32. Adversarial Attacks Leverage Interference Between Features in Superposition**

对抗性攻击利用叠加特征之间的干扰 cs.LG

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11709v1) [paper-pdf](http://arxiv.org/pdf/2510.11709v1)

**Authors**: Edward Stevinson, Lucas Prieto, Melih Barsbey, Tolga Birdal

**Abstract**: Fundamental questions remain about when and why adversarial examples arise in neural networks, with competing views characterising them either as artifacts of the irregularities in the decision landscape or as products of sensitivity to non-robust input features. In this paper, we instead argue that adversarial vulnerability can stem from efficient information encoding in neural networks. Specifically, we show how superposition - where networks represent more features than they have dimensions - creates arrangements of latent representations that adversaries can exploit. We demonstrate that adversarial perturbations leverage interference between superposed features, making attack patterns predictable from feature arrangements. Our framework provides a mechanistic explanation for two known phenomena: adversarial attack transferability between models with similar training regimes and class-specific vulnerability patterns. In synthetic settings with precisely controlled superposition, we establish that superposition suffices to create adversarial vulnerability. We then demonstrate that these findings persist in a ViT trained on CIFAR-10. These findings reveal adversarial vulnerability can be a byproduct of networks' representational compression, rather than flaws in the learning process or non-robust inputs.

摘要: 基本问题仍然是关于神经网络中何时以及为何出现对抗性示例，相互竞争的观点将它们描述为决策环境中不规则性的产物，或者是对非稳健输入特征敏感性的产物。在本文中，我们认为对抗性脆弱性可能源于神经网络中的高效信息编码。具体来说，我们展示了叠加（网络代表的特征比维度更多）如何创建对手可以利用的潜在表示的安排。我们证明，对抗性扰动利用了叠加特征之间的干扰，使攻击模式可以从特征排列中预测。我们的框架为两种已知现象提供了机械解释：具有相似训练机制的模型之间的对抗攻击转移性和特定类别的漏洞模式。在精确控制叠加的合成环境中，我们确定叠加足以创造对抗性的脆弱性。然后我们证明这些发现在接受CIFAR-10训练的ViT中仍然存在。这些发现揭示了对抗性脆弱性可能是网络代表性压缩的副产品，而不是学习过程中的缺陷或非稳健输入。



## **33. LLMAtKGE: Large Language Models as Explainable Attackers against Knowledge Graph Embeddings**

LLMAtKGE：大型语言模型作为知识图嵌入的可解释攻击者 cs.CL

13 pages

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11584v1) [paper-pdf](http://arxiv.org/pdf/2510.11584v1)

**Authors**: Ting Li, Yang Yang, Yipeng Yu, Liang Yao, Guoqing Chao, Ruifeng Xu

**Abstract**: Adversarial attacks on knowledge graph embeddings (KGE) aim to disrupt the model's ability of link prediction by removing or inserting triples. A recent black-box method has attempted to incorporate textual and structural information to enhance attack performance. However, it is unable to generate human-readable explanations, and exhibits poor generalizability. In the past few years, large language models (LLMs) have demonstrated powerful capabilities in text comprehension, generation, and reasoning. In this paper, we propose LLMAtKGE, a novel LLM-based framework that selects attack targets and generates human-readable explanations. To provide the LLM with sufficient factual context under limited input constraints, we design a structured prompting scheme that explicitly formulates the attack as multiple-choice questions while incorporating KG factual evidence. To address the context-window limitation and hesitation issues, we introduce semantics-based and centrality-based filters, which compress the candidate set while preserving high recall of attack-relevant information. Furthermore, to efficiently integrate both semantic and structural information into the filter, we precompute high-order adjacency and fine-tune the LLM with a triple classification task to enhance filtering performance. Experiments on two widely used knowledge graph datasets demonstrate that our attack outperforms the strongest black-box baselines and provides explanations via reasoning, and showing competitive performance compared with white-box methods. Comprehensive ablation and case studies further validate its capability to generate explanations.

摘要: 对知识图嵌入（KGE）的对抗攻击旨在通过删除或插入三重组来破坏模型的链接预测能力。最近的一种黑匣子方法试图合并文本和结构信息以增强攻击性能。然而，它无法生成人类可读的解释，并且表现出较差的概括性。在过去的几年里，大型语言模型（LLM）在文本理解、生成和推理方面展示了强大的能力。在本文中，我们提出了LLMAtKGE，这是一个基于LLM的新型框架，可以选择攻击目标并生成人类可读的解释。为了在有限的输入限制下为LLM提供足够的事实背景，我们设计了一个结构化的提示方案，该方案将攻击明确地制定为多项选择题，同时纳入KG事实证据。为了解决上下文窗口限制和犹豫问题，我们引入了基于语义和基于中心性的过滤器，它们压缩候选集，同时保留攻击相关信息的高召回率。此外，为了有效地将语义和结构信息集成到过滤器中，我们预先计算了高位邻近并通过三重分类任务微调LLM，以增强过滤性能。在两个广泛使用的知识图谱数据集上的实验表明，我们的攻击优于最强的黑盒基线，并通过推理提供解释，与白盒方法相比，表现出有竞争力的性能。全面的消融和病例研究进一步验证了其产生解释的能力。



## **34. TBRD: TESLA Authenticated UAS Broadcast Remote ID**

TESLA认证的UAS广播远程ID cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11343v1) [paper-pdf](http://arxiv.org/pdf/2510.11343v1)

**Authors**: Jason Veara, Manav Jain, Kyle Moy, Aanjhan Ranganathan

**Abstract**: Mysterious sightings of Unmanned Aircraft Systems (UAS) over U.S. military facilities, suburban neighborhoods, and commercial airports have intensified scrutiny of drone activity. To increase accountability, the Federal Aviation Administration (FAA) introduced a Remote ID mandate, requiring unmanned aircraft to broadcast their location, operator's location, and identity in real-time. However, current standards leave authentication mechanisms underspecified, enabling spoofing, relay, and replay attacks that can undermine surveillance efforts and potentially disrupt UAS-to-UAS coordination in future deployments. In this paper, we propose TBRD, a practical system for authenticating Remote ID messages in a manner that aligns with existing standards and UAS capabilities. TBRD leverages the TESLA protocol and mobile device TEEs, and introduces a verification mechanism to build a lightweight, mission-scoped authentication system that is both computationally efficient and requires a low communication footprint. We evaluate the performance of TBRD using both an FAA-requirements compatible proof-of-concept implementation for performance metrics and a simulated 4-drone swarm mission scenario to demonstrate its security guarantees under adversarial conditions. Our system provides a 50\% reduction in authentication overhead compared to digital signatures and a 100x reduction in computation time. Our results demonstrate that TBRD can be integrated into current Remote ID infrastructures to provide a scalable, standards-compliant message authentication for both regulatory and operational use cases.

摘要: 在美国军事设施、郊区和商业机场上空神秘发现无人机系统（UAS），这加剧了对无人机活动的审查。为了加强问责制，美国联邦航空管理局（FAA）引入了远程ID强制令，要求无人驾驶飞机实时广播其位置、操作员位置和身份。然而，当前的标准对身份验证机制的规定不足，从而导致欺骗、中继和重播攻击，这可能会破坏监视工作，并可能会破坏未来部署中的UAS到UAS协调。在本文中，我们提出了TBRD，这是一个用于以符合现有标准和UAS功能的方式认证远程ID消息的实用系统。TBRD利用TESLA协议和移动终端TEEs，并引入验证机制来构建轻量级、任务范围的身份验证系统，该系统既计算高效，又需要低通信占用空间。我们使用FAA要求兼容的性能指标概念验证实现和模拟的4无人机群任务场景来评估TBRD的性能，以证明其在对抗条件下的安全保证。与数字签名相比，我们的系统提供了50%的认证开销减少和100倍的计算时间减少。我们的研究结果表明，TBRD可以集成到当前的远程ID基础设施，为监管和运营用例提供可扩展的，符合标准的消息身份验证。



## **35. Attacks by Content: Automated Fact-checking is an AI Security Issue**

内容攻击：自动事实核查是人工智能安全问题 cs.CL

Accepted to EMNLP 2025

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11238v1) [paper-pdf](http://arxiv.org/pdf/2510.11238v1)

**Authors**: Michael Schlichtkrull

**Abstract**: When AI agents retrieve and reason over external documents, adversaries can manipulate the data they receive to subvert their behaviour. Previous research has studied indirect prompt injection, where the attacker injects malicious instructions. We argue that injection of instructions is not necessary to manipulate agents - attackers could instead supply biased, misleading, or false information. We term this an attack by content. Existing defenses, which focus on detecting hidden commands, are ineffective against attacks by content. To defend themselves and their users, agents must critically evaluate retrieved information, corroborating claims with external evidence and evaluating source trustworthiness. We argue that this is analogous to an existing NLP task, automated fact-checking, which we propose to repurpose as a cognitive self-defense tool for agents.

摘要: 当人工智能代理检索并推理外部文档时，对手可以操纵他们收到的数据来颠覆他们的行为。之前的研究研究了间接提示注入，即攻击者注入恶意指令。我们认为，指令的注入对于操纵代理来说不是必要的--攻击者可能会提供有偏见、误导性或虚假的信息。我们将这称为内容攻击。现有的防御措施专注于检测隐藏命令，对内容攻击无效。为了保护自己及其用户，代理必须批判性地评估检索到的信息，用外部证据证实主张并评估来源的可信度。我们认为，这类似于现有的NLP任务，即自动事实检查，我们建议将其重新用作代理人的认知自卫工具。



## **36. Navigating the Dual-Use Nature and Security Implications of Reconfigurable Intelligent Surfaces in Next-Generation Wireless Systems**

应对下一代无线系统中可重构智能表面的双重用途性质和安全影响 eess.SP

This manuscript has been accepted for publication in IEEE  Communications Surveys and Tutorials. It was received on January 17, 2025,  and revised on July 1 and September 16, 2025. This version was accepted on  October 10, 2025

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11113v1) [paper-pdf](http://arxiv.org/pdf/2510.11113v1)

**Authors**: Hetong Wang, Tiejun Lv, Yashuai Cao, Weicai Li, Jie Zeng, Pingmu Huang, Muhammad Khurram Khan

**Abstract**: Reconfigurable intelligent surface (RIS) technology offers significant promise in enhancing wireless communication systems, but its dual-use potential also introduces substantial security risks. This survey explores the security implications of RIS in next-generation wireless networks. We first highlight the dual-use nature of RIS, demonstrating how its communication-enhancing capabilities can be exploited by adversaries to compromise legitimate users. We identify a new class of security vulnerabilities termed ``passive-active hybrid attacks,'' where RIS, despite passively handling signals, can be reconfigured to actively engage in malicious activities, enabling various RIS-assisted attacks, such as eavesdropping, man-in-the-middle (MITM), replay, reflection jamming, and side-channel attacks. Furthermore, we reveal how adversaries can exploit the openness of wireless channels to introduce adversarial perturbations in artificial intelligence-driven RIS networks, disrupting communication terminals and causing misclassifications or errors in RIS reflection predictions. Despite these risks, RIS technology also plays a critical role in enhancing security and privacy across radio frequency (RF) and visible light communication (VLC) systems. By synthesizing current insights and highlighting emerging threats, we provide actionable insights into cross-layer collaboration, advanced adversarial defenses, and the balance between security and cost. This survey provides a comprehensive overview of RIS technology's security landscape and underscores the urgent need for robust security frameworks in the development of future wireless systems.

摘要: 可重新配置智能表面（RIS）技术在增强无线通信系统方面提供了巨大的前景，但其双重用途潜力也带来了巨大的安全风险。本调查探讨了RIS在下一代无线网络中的安全影响。我们首先强调RIS的双重用途性质，展示对手如何利用其通信增强功能来危害合法用户。我们发现了一类新的安全漏洞，称为“被动-主动混合攻击”，其中RIS尽管被动处理信号，但可以重新配置为主动参与恶意活动，从而实现各种RIS辅助攻击，例如窃听、中间人（MTM）、重播、反射干扰和侧通道攻击。此外，我们还揭示了对手如何利用无线通道的开放性在人工智能驱动的RIS网络中引入对抗性扰动，扰乱通信终端并导致RIS反射预测中的分类错误或错误。尽管存在这些风险，RIS技术在增强射频（RF）和可见光通信（SLC）系统的安全性和隐私方面也发挥着关键作用。通过综合当前见解并强调新出现的威胁，我们为跨层协作、高级对抗性防御以及安全与成本之间的平衡提供可操作的见解。这项调查全面概述了RIS技术的安全格局，并强调了未来无线系统开发中对强大的安全框架的迫切需求。



## **37. CoDefend: Cross-Modal Collaborative Defense via Diffusion Purification and Prompt Optimization**

CoDefend：通过扩散净化和即时优化的跨模式协同防御 cs.CV

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11096v1) [paper-pdf](http://arxiv.org/pdf/2510.11096v1)

**Authors**: Fengling Zhu, Boshi Liu, Jingyu Hua, Sheng Zhong

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable success in tasks such as image captioning, visual question answering, and cross-modal reasoning by integrating visual and textual modalities. However, their multimodal nature also exposes them to adversarial threats, where attackers can perturb either modality or both jointly to induce harmful, misleading, or policy violating outputs. Existing defense strategies, such as adversarial training and input purification, face notable limitations: adversarial training typically improves robustness only against known attacks while incurring high computational costs, whereas conventional purification approaches often suffer from degraded image quality and insufficient generalization to complex multimodal tasks.   In this work, we focus on defending the visual modality, which frequently serves as the primary entry point for adversarial manipulation. We propose a supervised diffusion based denoising framework that leverages paired adversarial clean image datasets to fine-tune diffusion models with directional, task specific guidance. Unlike prior unsupervised purification methods such as DiffPure, our approach achieves higher quality reconstructions while significantly improving defense robustness in multimodal tasks. Furthermore, we incorporate prompt optimization as a complementary defense mechanism, enhancing resistance against diverse and unseen attack strategies.   Extensive experiments on image captioning and visual question answering demonstrate that our method not only substantially improves robustness but also exhibits strong transferability to unknown adversarial attacks. These results highlight the effectiveness of supervised diffusion based denoising for multimodal defense, paving the way for more reliable and secure deployment of MLLMs in real world applications.

摘要: 多模式大型语言模型（MLLM）通过集成视觉和文本模式，在图像字幕、视觉问答和跨模式推理等任务中取得了显着的成功。然而，它们的多模式性质也使它们面临对抗威胁，攻击者可以共同扰乱其中一种模式或两者，以引发有害、误导或违反政策的输出。现有的防御策略，例如对抗性训练和输入净化，面临着显着的局限性：对抗性训练通常只提高针对已知攻击的鲁棒性，同时会产生很高的计算成本，而传统的净化方法往往会出现图像质量下降和对复杂多模式任务的概括不足的问题。   在这项工作中，我们专注于捍卫视觉形态，它经常作为对抗性操纵的主要切入点。我们提出了一种基于监督扩散的去噪框架，该框架利用成对的对抗性干净图像数据集，通过定向、特定任务的指导来微调扩散模型。与之前的无监督纯化方法（例如DistPure）不同，我们的方法实现了更高质量的重建，同时显着提高了多模式任务中的防御鲁棒性。此外，我们还将即时优化作为补充防御机制，增强对多样化和不可见攻击策略的抵抗力。   关于图像字幕和视觉问答的大量实验表明，我们的方法不仅大大提高了鲁棒性，而且还表现出对未知对抗攻击的强大可移植性。这些结果凸显了基于监督扩散的去噪对多模式防御的有效性，为在现实世界应用中更可靠、更安全地部署MLLM铺平了道路。



## **38. TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data**

TabAttackBench：表格数据对抗性攻击的基准 cs.LG

71 pages, 21 figures, 11 tables

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2505.21027v2) [paper-pdf](http://arxiv.org/pdf/2505.21027v2)

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks are well studied in unstructured domains such as images, their behaviour on tabular data remains underexplored due to mixed feature types and complex inter-feature dependencies. This study introduces a comprehensive benchmark that evaluates adversarial attacks on tabular datasets with respect to both effectiveness and imperceptibility. We assess five white-box attack algorithms (FGSM, BIM, PGD, DeepFool, and C\&W) across four representative models (LR, MLP, TabTransformer and FT-Transformer) using eleven datasets spanning finance, energy, and healthcare domains. The benchmark employs four quantitative imperceptibility metrics (proximity, sparsity, deviation, and sensitivity) to characterise perturbation realism. The analysis quantifies the trade-off between these two aspects and reveals consistent differences between attack types, with $\ell_\infty$-based attacks achieving higher success but lower subtlety, and $\ell_2$-based attacks offering more realistic perturbations. The benchmark findings offer actionable insights for designing more imperceptible adversarial attacks, advancing the understanding of adversarial vulnerability in tabular machine learning.

摘要: 对抗性攻击通过对输入数据的不可感知的扰动引发错误预测，对机器学习模型构成重大威胁。虽然这些攻击在图像等非结构化领域中得到了很好的研究，但由于混合的特征类型和复杂的特征间依赖关系，它们在表格数据上的行为仍然没有得到充分的研究。这项研究引入了一个全面的基准，从有效性和不可感知性方面评估对表格数据集的对抗攻击。我们使用涵盖金融、能源和医疗保健领域的十一个数据集，评估了四个代表性模型（LR、MLP、TabTransformer和FT-Transformer）的五种白盒攻击算法（FGSM、BMI、PVD、DeepFool和C\& W）。该基准采用四种量化不可感知性指标（接近性、稀疏性、偏差和敏感性）来验证扰动真实性。该分析量化了这两个方面之间的权衡，并揭示了攻击类型之间的一致差异，基于$\ell_\infty$的攻击取得了更高的成功，但微妙性较低，而基于$\ell_2 $的攻击提供了更现实的干扰。基准调查结果为设计更难以察觉的对抗攻击提供了可行的见解，从而促进了对表格机器学习中对抗漏洞的理解。



## **39. Adversarial Robustness in One-Stage Learning-to-Defer**

一阶段学习推迟中的对抗稳健性 stat.ML

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.10988v1) [paper-pdf](http://arxiv.org/pdf/2510.10988v1)

**Authors**: Yannis Montreuil, Letian Yu, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Learning-to-Defer (L2D) enables hybrid decision-making by routing inputs either to a predictor or to external experts. While promising, L2D is highly vulnerable to adversarial perturbations, which can not only flip predictions but also manipulate deferral decisions. Prior robustness analyses focus solely on two-stage settings, leaving open the end-to-end (one-stage) case where predictor and allocation are trained jointly. We introduce the first framework for adversarial robustness in one-stage L2D, covering both classification and regression. Our approach formalizes attacks, proposes cost-sensitive adversarial surrogate losses, and establishes theoretical guarantees including $\mathcal{H}$, $(\mathcal{R }, \mathcal{F})$, and Bayes consistency. Experiments on benchmark datasets confirm that our methods improve robustness against untargeted and targeted attacks while preserving clean performance.

摘要: 学习延迟（L2 D）通过将输入路由到预测器或外部专家来实现混合决策。L2 D虽然前景光明，但很容易受到对抗性扰动的影响，这种扰动不仅会翻转预测，还会操纵延期决定。先前的稳健性分析仅关注两阶段设置，留下了联合训练预测器和分配的端到端（一阶段）情况。我们在单阶段L2 D中引入了第一个对抗鲁棒性框架，涵盖分类和回归。我们的方法形式化攻击，提出成本敏感的对抗性替代损失，并建立理论保证，包括$\mathcal{H}$、$（\mathcal{R }，\mathcal{F}）$和Bayes一致性。对基准数据集的实验证实，我们的方法可以提高针对非目标和目标攻击的鲁棒性，同时保持干净的性能。



## **40. Neutral Agent-based Adversarial Policy Learning against Deep Reinforcement Learning in Multi-party Open Systems**

多方开放系统中基于中立代理的对抗性政策学习与深度强化学习 cs.LG

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.10937v1) [paper-pdf](http://arxiv.org/pdf/2510.10937v1)

**Authors**: Qizhou Peng, Yang Zheng, Yu Wen, Yanna Wu, Yingying Du

**Abstract**: Reinforcement learning (RL) has been an important machine learning paradigm for solving long-horizon sequential decision-making problems under uncertainty. By integrating deep neural networks (DNNs) into the RL framework, deep reinforcement learning (DRL) has emerged, which achieved significant success in various domains. However, the integration of DNNs also makes it vulnerable to adversarial attacks. Existing adversarial attack techniques mainly focus on either directly manipulating the environment with which a victim agent interacts or deploying an adversarial agent that interacts with the victim agent to induce abnormal behaviors. While these techniques achieve promising results, their adoption in multi-party open systems remains limited due to two major reasons: impractical assumption of full control over the environment and dependent on interactions with victim agents.   To enable adversarial attacks in multi-party open systems, in this paper, we redesigned an adversarial policy learning approach that can mislead well-trained victim agents without requiring direct interactions with these agents or full control over their environments. Particularly, we propose a neutral agent-based approach across various task scenarios in multi-party open systems. While the neutral agents seemingly are detached from the victim agents, indirectly influence them through the shared environment. We evaluate our proposed method on the SMAC platform based on Starcraft II and the autonomous driving simulation platform Highway-env. The experimental results demonstrate that our method can launch general and effective adversarial attacks in multi-party open systems.

摘要: 强化学习（RL）一直是解决不确定性下的长期顺序决策问题的重要机器学习范式。通过将深度神经网络（DNN）集成到RL框架中，深度强化学习（DRL）应运而生，并在各个领域取得了显着成功。然而，DNN的集成也使其容易受到对抗攻击。现有的对抗性攻击技术主要集中在直接操纵受害者代理交互的环境或部署与受害者代理交互的对抗性代理以诱导异常行为。虽然这些技术取得了可喜的成果，但由于两个主要原因，它们在多方开放系统中的应用仍然有限：对环境的完全控制的不切实际的假设和依赖于与受害者代理的交互。   为了在多方开放系统中实现对抗性攻击，在本文中，我们重新设计了一种对抗性策略学习方法，该方法可以误导训练有素的受害者代理，而无需与这些代理直接交互或完全控制其环境。特别是，我们提出了一个中立的基于代理的方法在多方开放系统中的各种任务场景。而中立主体表面上与受害主体是分离的，通过共享环境间接影响受害主体。我们在基于星际争霸II的SMAC平台和自动驾驶模拟平台Highway-dev上评估了我们提出的方法。实验结果表明，我们的方法可以在多方开放系统中发起普遍有效的对抗攻击。



## **41. TabVLA: Targeted Backdoor Attacks on Vision-Language-Action Models**

TabVLA：对视觉-语言-动作模型的有针对性的后门攻击 cs.CR

8 pages, 8 tables, 1 figure. Under review

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.10932v1) [paper-pdf](http://arxiv.org/pdf/2510.10932v1)

**Authors**: Zonghuan Xu, Xiang Zheng, Xingjun Ma, Yu-Gang Jiang

**Abstract**: With the growing deployment of Vision-Language-Action (VLA) models in real-world embodied AI systems, their increasing vulnerability to backdoor attacks poses a serious safety threat. A backdoored VLA agent can be covertly triggered by a pre-injected backdoor to execute adversarial actions, potentially causing system failures or even physical harm. Although backdoor attacks on VLA models have been explored, prior work has focused only on untargeted attacks, leaving the more practically threatening scenario of targeted manipulation unexamined. In this paper, we study targeted backdoor attacks on VLA models and introduce TabVLA, a novel framework that enables such attacks via black-box fine-tuning. TabVLA explores two deployment-relevant inference-time threat models: input-stream editing and in-scene triggering. It formulates poisoned data generation as an optimization problem to improve attack effectivess. Experiments with OpenVLA-7B on the LIBERO benchmark reveal that the vision channel is the principal attack surface: targeted backdoors succeed with minimal poisoning, remain robust across variations in trigger design, and are degraded only by positional mismatches between fine-tuning and inference triggers. We also investigate a potential detection-based defense against TabVLA, which reconstructs latent visual triggers from the input stream to flag activation-conditioned backdoor samples. Our work highlights the vulnerability of VLA models to targeted backdoor manipulation and underscores the need for more advanced defenses.

摘要: 随着视觉-语言-动作（VLA）模型在现实世界的嵌入式人工智能系统中的越来越多的部署，它们对后门攻击的脆弱性越来越大，构成了严重的安全威胁。后门的VLA代理可能会被预先注入的后门秘密触发，以执行对抗动作，可能导致系统故障甚至身体伤害。尽管已经探索了对VLA模型的后门攻击，但之前的工作仅专注于无针对性的攻击，而没有审查针对性操纵的更具实际威胁的情况。本文中，我们研究了对VLA模型的有针对性的后门攻击，并介绍了TabVLA，这是一种通过黑匣子微调支持此类攻击的新型框架。TabVLA探索了两种与部署相关的推断时威胁模型：输入流编辑和场景内触发。它将有毒数据生成制定为一个优化问题，以提高攻击效果。在LIBERO基准上使用OpenVLA-7 B进行的实验表明，视觉通道是主要的攻击表面：有针对性的后门以最小的中毒成功，在触发器设计的变化中保持稳健，并且仅因微调和推理触发器之间的位置不匹配而降级。我们还研究了一种针对TabVLA的潜在基于检测的防御，该防御从输入流中重建潜在视觉触发器，以标记受激活条件的后门样本。我们的工作强调了VLA模型对有针对性的后门操纵的脆弱性，并强调了对更先进防御的需求。



## **42. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

没有对抗防御的对抗防御：通过实例级主成分去除增强语言模型稳健性 cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2507.21750v3) [paper-pdf](http://arxiv.org/pdf/2507.21750v3)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.

摘要: 预训练的语言模型（PLM）推动了自然语言处理的重大进展，但仍然容易受到对抗攻击，引发了对其在现实世界应用程序中稳健性的担忧。之前的研究试图通过隐式或显式地在训练过程中引入对抗性扰动来减轻对抗性攻击的影响。虽然这两种策略都增强了稳健性，但它们通常会产生很高的计算成本。在这项工作中，我们提出了一个简单而有效的附加模块，该模块通过删除实例级主成分来增强PLM的对抗鲁棒性，而不依赖于传统的对抗防御或干扰原始训练数据。我们的方法将嵌入空间转换为逼近高斯属性，从而降低其对对抗性扰动的敏感性，同时保留语义关系。这种转换以一种最小化对抗性噪音对决策边界的影响的方式对齐嵌入分布，增强稳健性，而无需对抗性示例或昂贵的训练时间扩展。对八个基准数据集的评估表明，我们的方法提高了对抗稳健性，同时保持了与基线相当的攻击前准确性，实现了稳健性和概括性之间的平衡。



## **43. On Surjectivity of Neural Networks: Can you elicit any behavior from your model?**

关于神经网络的满摄性：你能从你的模型中引出任何行为吗？ cs.LG

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2508.19445v2) [paper-pdf](http://arxiv.org/pdf/2508.19445v2)

**Authors**: Haozhe Jiang, Nika Haghtalab

**Abstract**: Given a trained neural network, can any specified output be generated by some input? Equivalently, does the network correspond to a function that is surjective? In generative models, surjectivity implies that any output, including harmful or undesirable content, can in principle be generated by the networks, raising concerns about model safety and jailbreak vulnerabilities. In this paper, we prove that many fundamental building blocks of modern neural architectures, such as networks with pre-layer normalization and linear-attention modules, are almost always surjective. As corollaries, widely used generative frameworks, including GPT-style transformers and diffusion models with deterministic ODE solvers, admit inverse mappings for arbitrary outputs. By studying surjectivity of these modern and commonly used neural architectures, we contribute a formalism that sheds light on their unavoidable vulnerability to a broad class of adversarial attacks.

摘要: 给定一个经过训练的神经网络，任何指定的输出都可以由某些输入生成吗？同样，网络是否对应于满射函数？在生成模型中，主观性意味着任何输出，包括有害或不受欢迎的内容，原则上都可以由网络生成，这引发了对模型安全性和越狱漏洞的担忧。在本文中，我们证明了现代神经架构的许多基本构建块，例如具有预层归一化和线性注意模块的网络，几乎总是满射的。作为推论，广泛使用的生成式框架（包括GPT式转换器和具有确定性ODE解算器的扩散模型）允许任意输出的逆映射。通过研究这些现代和常用的神经架构的主观性，我们提出了一种形式主义，揭示了它们不可避免地容易受到一类对抗攻击的脆弱性。



## **44. Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations**

概念掌舵者：利用K稀疏自动编码器实现测试时间可控生成 cs.CV

23 pages, 18 figures

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2501.19066v3) [paper-pdf](http://arxiv.org/pdf/2501.19066v3)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) -- all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers

摘要: 尽管文本到图像生成模型取得了显着的进步，但它们容易受到对抗性攻击，并无意中生成不安全、不道德的内容。现有的方法通常依赖于微调模型来删除特定概念，这在计算上昂贵、缺乏可扩展性和/或损害发电质量。在这项工作中，我们提出了一种利用k-稀疏自动编码器（k-SAEs）的新型框架，以在扩散模型中实现高效且可解释的概念操纵。具体来说，我们首先在文本嵌入的潜在空间中识别可解释的单语义概念，并利用它们来精确地引导生成远离或转向给定概念（例如，裸体）或引入新概念（例如，摄影风格）--所有在测试时间。通过大量实验，我们证明我们的方法非常简单，不需要重新训练基本模型或LoRA适配器，不损害生成质量，并且对对抗提示操作具有鲁棒性。我们的方法在不安全概念删除方面提高了$\mathBF{20.01\%}$，在风格操纵方面有效，并且比当前最先进技术快$\mathBF{\sim 5}$x。代码可访问：https://github.com/kim-dahye/steerers



## **45. SASER: Stego attacks on open-source LLMs**

SABER：Stego攻击开源LLM cs.CR

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2510.10486v1) [paper-pdf](http://arxiv.org/pdf/2510.10486v1)

**Authors**: Ming Tan, Wei Li, Hu Tao, Hailong Ma, Aodi Liu, Qian Chen, Zilong Wang

**Abstract**: Open-source large language models (LLMs) have demonstrated considerable dominance over proprietary LLMs in resolving neural processing tasks, thanks to the collaborative and sharing nature. Although full access to source codes, model parameters, and training data lays the groundwork for transparency, we argue that such a full-access manner is vulnerable to stego attacks, and their ill-effects are not fully understood. In this paper, we conduct a systematic formalization for stego attacks on open-source LLMs by enumerating all possible threat models associated with adversary objectives, knowledge, and capabilities. Therein, the threat posed by adversaries with internal knowledge, who inject payloads and triggers during the model sharing phase, is of practical interest. We go even further and propose the first stego attack on open-source LLMs, dubbed SASER, which wields impacts through identifying targeted parameters, embedding payloads, injecting triggers, and executing payloads sequentially. Particularly, SASER enhances the attack robustness against quantization-based local deployment by de-quantizing the embedded payloads. In addition, to achieve stealthiness, SASER devises the performance-aware importance metric to identify targeted parameters with the least degradation of model performance. Extensive experiments on LlaMA2-7B and ChatGLM3-6B, without quantization, show that the stealth rate of SASER outperforms existing stego attacks (for general DNNs) by up to 98.1%, while achieving the same attack success rate (ASR) of 100%. More importantly, SASER improves ASR on quantized models from 0 to 100% in all settings. We appeal for investigations on countermeasures against SASER in view of the significant attack effectiveness.

摘要: 由于协作和共享的性质，开源大型语言模型（LLM）在解决神经处理任务方面表现出了相对于专有LLM的相当大的主导地位。尽管对源代码、模型参数和训练数据的完全访问为透明度奠定了基础，但我们认为，这种完全访问方式很容易受到隐纹攻击，并且其不良影响尚未被完全理解。在本文中，我们通过列举与对手目标、知识和能力相关的所有可能威胁模型，对开源LLM的隐刻攻击进行了系统的形式化。其中，具有内部知识的对手在模型共享阶段注入有效负载和触发器所构成的威胁具有实际意义。我们更进一步，提出了对开源LLM的第一个隐写攻击，称为SASER，它通过识别目标参数，嵌入有效载荷，注入触发器和顺序执行有效载荷来产生影响。特别地，SASER通过对嵌入的有效载荷进行去量化来增强对基于量化的本地部署的攻击鲁棒性。此外，为了实现隐蔽性，SASER设计了性能感知的重要性度量，以识别目标参数，同时最大限度地降低模型性能。在LlaMA 2 - 7 B和ChatGLM 3 - 6 B上进行的大量实验表明，SASER的隐身率比现有的隐写攻击（对于一般DNN）高出98.1%，同时达到100%的攻击成功率（ASR）。更重要的是，SABER在所有设置中将量化模型的ASB从0%提高到100%。鉴于SABER的攻击效果显着，我们呼吁对针对SABER的反制措施进行调查。



## **46. Boosting Adversarial Transferability via Commonality-Oriented Gradient Optimization**

通过面向公共性的梯度优化提高对抗性可移植性 cs.CV

23 pages

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2506.06992v2) [paper-pdf](http://arxiv.org/pdf/2506.06992v2)

**Authors**: Yanting Gao, Yepeng Liu, Junming Liu, Qi Zhang, Hongyun Zhang, Duoqian Miao, Cairong Zhao

**Abstract**: Exploring effective and transferable adversarial examples is vital for understanding the characteristics and mechanisms of Vision Transformers (ViTs). However, adversarial examples generated from surrogate models often exhibit weak transferability in black-box settings due to overfitting. Existing methods improve transferability by diversifying perturbation inputs or applying uniform gradient regularization within surrogate models, yet they have not fully leveraged the shared and unique features of surrogate models trained on the same task, leading to suboptimal transfer performance. Therefore, enhancing perturbations of common information shared by surrogate models and suppressing those tied to individual characteristics offers an effective way to improve transferability. Accordingly, we propose a commonality-oriented gradient optimization strategy (COGO) consisting of two components: Commonality Enhancement (CE) and Individuality Suppression (IS). CE perturbs the mid-to-low frequency regions, leveraging the fact that ViTs trained on the same dataset tend to rely more on mid-to-low frequency information for classification. IS employs adaptive thresholds to evaluate the correlation between backpropagated gradients and model individuality, assigning weights to gradients accordingly. Extensive experiments demonstrate that COGO significantly improves the transfer success rates of adversarial attacks, outperforming current state-of-the-art methods.

摘要: 探索有效且可转移的对抗示例对于理解视觉变形者（ViT）的特征和机制至关重要。然而，由于过度匹配，从代理模型生成的对抗性示例在黑匣子环境中通常表现出较弱的可移植性。现有的方法通过多样化扰动输入或在代理模型内应用均匀的梯度正规化来提高可移植性，但它们没有充分利用在同一任务上训练的代理模型的共享和独特特征，导致次优的传输性能。因此，增强对代理模型共享的公共信息的扰动并抑制与个体特征相关的扰动提供了提高可移植性的有效方法。因此，我们提出了一种面向共性的梯度优化策略（COGO），该策略由两个部分组成：共性增强（CE）和个性抑制（IS）。CE扰乱了中低频区域，利用了在相同数据集上训练的ViT往往更多地依赖中低频信息进行分类这一事实。IS采用自适应阈值来评估反向传播的梯度与模型个性之间的相关性，并相应地为梯度分配权重。大量实验表明，COGO显着提高了对抗性攻击的传输成功率，优于当前最先进的方法。



## **47. Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction**

下游天气预报模型的对抗攻击：在热带气旋轨迹预测中的应用 cs.LG

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2510.10140v1) [paper-pdf](http://arxiv.org/pdf/2510.10140v1)

**Authors**: Yue Deng, Francisco Santos, Pang-Ning Tan, Lifeng Luo

**Abstract**: Deep learning based weather forecasting (DLWF) models leverage past weather observations to generate future forecasts, supporting a wide range of downstream tasks, including tropical cyclone (TC) trajectory prediction. In this paper, we investigate their vulnerability to adversarial attacks, where subtle perturbations to the upstream weather forecasts can alter the downstream TC trajectory predictions. Although research on adversarial attacks in DLWF models has grown recently, generating perturbed upstream forecasts that reliably steer downstream output toward attacker-specified trajectories remains a challenge. First, conventional TC detection systems are opaque, non-differentiable black boxes, making standard gradient-based attacks infeasible. Second, the extreme rarity of TC events leads to severe class imbalance problem, making it difficult to develop efficient attack methods that will produce the attacker's target trajectories. Furthermore, maintaining physical consistency in adversarially generated forecasts presents another significant challenge. To overcome these limitations, we propose Cyc-Attack, a novel method that perturbs the upstream forecasts of DLWF models to generate adversarial trajectories. First, we pre-train a differentiable surrogate model to approximate the TC detector's output, enabling the construction of gradient-based attacks. Cyc-Attack also employs skewness-aware loss function with kernel dilation strategy to address the imbalance problem. Finally, a distance-based gradient weighting scheme and regularization are used to constrain the perturbations and eliminate spurious trajectories to ensure the adversarial forecasts are realistic and not easily detectable.

摘要: 基于深度学习的天气预报（DLWF）模型利用过去的天气观测来生成未来预报，支持广泛的下游任务，包括热带气旋（TC）轨迹预测。在本文中，我们研究了它们对对抗攻击的脆弱性，其中对上游天气预报的微妙扰动可能会改变下游TC轨迹预测。尽管对DLWF模型中对抗性攻击的研究最近有所增加，但生成受干扰的上游预测以可靠地引导下游产出转向攻击者指定的轨迹仍然是一个挑战。首先，传统的TC检测系统是不透明的、不可区分的黑匣子，使得标准的基于梯度的攻击不可行。其次，TC事件的极端罕见导致严重的阶级失衡问题，使得很难开发出产生攻击者目标轨迹的高效攻击方法。此外，在对抗生成的预测中保持物理一致性也是另一个重大挑战。为了克服这些限制，我们提出了周期攻击，这是一种新颖的方法，可以扰乱DLWF模型的上游预测以生成对抗轨迹。首先，我们预训练一个可微代理模型来逼近TC检测器的输出，从而能够构建基于梯度的攻击。循环攻击还采用了偏度感知损失函数和核膨胀策略来解决不平衡问题。最后，使用基于距离的梯度加权方案和正则化来约束扰动并消除虚假轨迹，以确保对抗性预测是现实的并且不容易检测。



## **48. A-IPO: Adaptive Intent-driven Preference Optimization**

A-IPO：自适应意图驱动的偏好优化 cs.CL

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2510.10077v1) [paper-pdf](http://arxiv.org/pdf/2510.10077v1)

**Authors**: Wenqing Wang, Muhammad Asif Ali, Ali Shoker, Ruohan Yang, Junyang Chen, Ying Sha, Huan Wang

**Abstract**: Human preferences are diverse and dynamic, shaped by regional, cultural, and social factors. Existing alignment methods like Direct Preference Optimization (DPO) and its variants often default to majority views, overlooking minority opinions and failing to capture latent user intentions in prompts.   To address these limitations, we introduce \underline{\textbf{A}}daptive \textbf{\underline{I}}ntent-driven \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization (\textbf{A-IPO}). Specifically,A-IPO introduces an intention module that infers the latent intent behind each user prompt and explicitly incorporates this inferred intent into the reward function, encouraging stronger alignment between the preferred model's responses and the user's underlying intentions. We demonstrate, both theoretically and empirically, that incorporating an intention--response similarity term increases the preference margin (by a positive shift of $\lambda\,\Delta\mathrm{sim}$ in the log-odds), resulting in clearer separation between preferred and dispreferred responses compared to DPO.   For evaluation, we introduce two new benchmarks, Real-pref, Attack-pref along with an extended version of an existing dataset, GlobalOpinionQA-Ext, to assess real-world and adversarial preference alignment.   Through explicit modeling of diverse user intents,A-IPO facilitates pluralistic preference optimization while simultaneously enhancing adversarial robustness in preference alignment. Comprehensive empirical evaluation demonstrates that A-IPO consistently surpasses existing baselines, yielding substantial improvements across key metrics: up to +24.8 win-rate and +45.6 Response-Intention Consistency on Real-pref; up to +38.6 Response Similarity and +52.2 Defense Success Rate on Attack-pref; and up to +54.6 Intention Consistency Score on GlobalOpinionQA-Ext.

摘要: Human preferences are diverse and dynamic, shaped by regional, cultural, and social factors. Existing alignment methods like Direct Preference Optimization (DPO) and its variants often default to majority views, overlooking minority opinions and failing to capture latent user intentions in prompts.   To address these limitations, we introduce \underline{\textbf{A}}daptive \textbf{\underline{I}}ntent-driven \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization (\textbf{A-IPO}). Specifically,A-IPO introduces an intention module that infers the latent intent behind each user prompt and explicitly incorporates this inferred intent into the reward function, encouraging stronger alignment between the preferred model's responses and the user's underlying intentions. We demonstrate, both theoretically and empirically, that incorporating an intention--response similarity term increases the preference margin (by a positive shift of $\lambda\,\Delta\mathrm{sim}$ in the log-odds), resulting in clearer separation between preferred and dispreferred responses compared to DPO.   For evaluation, we introduce two new benchmarks, Real-pref, Attack-pref along with an extended version of an existing dataset, GlobalOpinionQA-Ext, to assess real-world and adversarial preference alignment.   Through explicit modeling of diverse user intents,A-IPO facilitates pluralistic preference optimization while simultaneously enhancing adversarial robustness in preference alignment. Comprehensive empirical evaluation demonstrates that A-IPO consistently surpasses existing baselines, yielding substantial improvements across key metrics: up to +24.8 win-rate and +45.6 Response-Intention Consistency on Real-pref; up to +38.6 Response Similarity and +52.2 Defense Success Rate on Attack-pref; and up to +54.6 Intention Consistency Score on GlobalOpinionQA-Ext.



## **49. SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-based Web Agents**

SecureWebArena：基于LVLM的Web代理的整体安全评估基准 cs.CR

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2510.10073v1) [paper-pdf](http://arxiv.org/pdf/2510.10073v1)

**Authors**: Zonghao Ying, Yangguang Shao, Jianle Gan, Gan Xu, Junjie Shen, Wenxin Zhang, Quanchen Zou, Junzheng Shi, Zhenfei Yin, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: Large vision-language model (LVLM)-based web agents are emerging as powerful tools for automating complex online tasks. However, when deployed in real-world environments, they face serious security risks, motivating the design of security evaluation benchmarks. Existing benchmarks provide only partial coverage, typically restricted to narrow scenarios such as user-level prompt manipulation, and thus fail to capture the broad range of agent vulnerabilities. To address this gap, we present \tool{}, the first holistic benchmark for evaluating the security of LVLM-based web agents. \tool{} first introduces a unified evaluation suite comprising six simulated but realistic web environments (\eg, e-commerce platforms, community forums) and includes 2,970 high-quality trajectories spanning diverse tasks and attack settings. The suite defines a structured taxonomy of six attack vectors spanning both user-level and environment-level manipulations. In addition, we introduce a multi-layered evaluation protocol that analyzes agent failures across three critical dimensions: internal reasoning, behavioral trajectory, and task outcome, facilitating a fine-grained risk analysis that goes far beyond simple success metrics. Using this benchmark, we conduct large-scale experiments on 9 representative LVLMs, which fall into three categories: general-purpose, agent-specialized, and GUI-grounded. Our results show that all tested agents are consistently vulnerable to subtle adversarial manipulations and reveal critical trade-offs between model specialization and security. By providing (1) a comprehensive benchmark suite with diverse environments and a multi-layered evaluation pipeline, and (2) empirical insights into the security challenges of modern LVLM-based web agents, \tool{} establishes a foundation for advancing trustworthy web agent deployment.

摘要: 基于大型视觉语言模型（LVLM）的Web代理正在成为自动化复杂在线任务的强大工具。然而，当部署在现实世界环境中时，它们面临着严重的安全风险，从而推动了安全评估基准的设计。现有的基准仅提供部分覆盖，通常仅限于用户级提示操纵等狭窄场景，因此无法捕获广泛的代理漏洞。为了解决这一差距，我们提出了\tool{}，这是第一个用于评估基于LVLM的Web代理安全性的整体基准。\tool{}首先引入了一个统一的评估套件，该套件由六个模拟但真实的Web环境（例如，电子商务平台、社区论坛）组成，并包括跨越不同任务和攻击设置的2，970个高质量轨迹。该套件定义了涵盖用户级和环境级操作的六种攻击载体的结构化分类法。此外，我们还引入了一种多层评估协议，该协议在三个关键维度上分析代理失败：内部推理、行为轨迹和任务结果，从而促进了远超出简单成功指标的细粒度风险分析。使用这个基准，我们对9个有代表性的LVLM进行了大规模实验，这些LVLM分为三类：通用、代理专用和基于图形界面的。我们的结果表明，所有测试的代理都始终容易受到微妙的对抗操纵的影响，并揭示了模型专业化和安全性之间的关键权衡。通过提供（1）具有多样化环境和多层评估管道的全面基准套件，以及（2）对现代基于LVLM的Web代理的安全挑战的经验见解，\tool{}为推进值得信赖的Web代理部署奠定了基础。



## **50. GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models**

GenoArmory：基因组基础模型对抗攻击的统一评估框架 cs.LG

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2505.10983v2) [paper-pdf](http://arxiv.org/pdf/2505.10983v2)

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Zhenyu Pan, Weian Mao, Haoyang Fang, Hao Xu, Han Liu, Binghui Wang, Yan Chen

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.

摘要: 我们为基因组基础模型（GFM）提出了第一个统一的对抗攻击基准，名为GenoArmory。与现有的GFM基准不同，GenoArmory提供了第一个全面的评估框架来系统地评估GFM对对抗攻击的脆弱性。在方法上，我们使用四种广泛采用的攻击算法和三种防御策略评估了五种最先进的GFM的对抗稳健性。重要的是，我们的基准提供了一个易于访问且全面的框架来分析GFM在模型架构、量化方案和训练数据集方面的漏洞。此外，我们还引入了GenoAdv，这是一个新的对抗性样本数据集，旨在提高GFM安全性。从经验上看，与生成模型相比，分类模型对对抗性扰动表现出更大的鲁棒性，凸显了任务类型对模型脆弱性的影响。此外，对抗性攻击经常针对具有生物学意义的基因组区域，这表明这些模型有效地捕获了有意义的序列特征。



