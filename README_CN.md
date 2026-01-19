# Latest Adversarial Attack Papers
**update at 2026-01-19 16:16:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. QUPID: A Partitioned Quantum Neural Network for Anomaly Detection in Smart Grid**

QUID：用于智能电网异常检测的分区量子神经网络 cs.LG

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11500v1) [paper-pdf](https://arxiv.org/pdf/2601.11500v1)

**Authors**: Hoang M. Ngo, Tre' R. Jeter, Jung Taek Seo, My T. Thai

**Abstract**: Smart grid infrastructures have revolutionized energy distribution, but their day-to-day operations require robust anomaly detection methods to counter risks associated with cyber-physical threats and system faults potentially caused by natural disasters, equipment malfunctions, and cyber attacks. Conventional machine learning (ML) models are effective in several domains, yet they struggle to represent the complexities observed in smart grid systems. Furthermore, traditional ML models are highly susceptible to adversarial manipulations, making them increasingly unreliable for real-world deployment. Quantum ML (QML) provides a unique advantage, utilizing quantum-enhanced feature representations to model the intricacies of the high-dimensional nature of smart grid systems while demonstrating greater resilience to adversarial manipulation. In this work, we propose QUPID, a partitioned quantum neural network (PQNN) that outperforms traditional state-of-the-art ML models in anomaly detection. We extend our model to R-QUPID that even maintains its performance when including differential privacy (DP) for enhanced robustness. Moreover, our partitioning framework addresses a significant scalability problem in QML by efficiently distributing computational workloads, making quantum-enhanced anomaly detection practical in large-scale smart grid environments. Our experimental results across various scenarios exemplifies the efficacy of QUPID and R-QUPID to significantly improve anomaly detection capabilities and robustness compared to traditional ML approaches.

摘要: 智能电网基础设施彻底改变了能源分配，但其日常运营需要强大的异常检测方法，以应对与网络物理威胁和系统故障相关的风险，这些风险可能由自然灾害、设备故障和网络攻击引起。传统的机器学习（ML）模型在多个领域都有效，但它们很难代表智能电网系统中观察到的复杂性。此外，传统的ML模型极易受到对抗性操纵的影响，这使得它们对于现实世界的部署越来越不可靠。量子ML（QML）提供了独特的优势，利用量子增强特征表示来建模智能电网系统的复杂性，同时展示了对对抗操纵的更大弹性。在这项工作中，我们提出了QUID，这是一种分区量子神经网络（PQNN），在异常检测方面优于传统最先进的ML模型。我们将我们的模型扩展到R-QUID，在包含差异隐私（DP）以增强稳健性时甚至可以保持其性能。此外，我们的分区框架通过有效地分配计算工作负载来解决QML中的一个重大可扩展性问题，使量子增强型异常检测在大规模智能电网环境中变得实用。与传统ML方法相比，我们在各种场景下的实验结果证实了QUID和R-QUID在显着提高异常检测能力和鲁棒性方面的功效。



## **2. Backdoor Attacks on Multi-modal Contrastive Learning**

多模式对比学习的后门攻击 cs.LG

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11006v1) [paper-pdf](https://arxiv.org/pdf/2601.11006v1)

**Authors**: Simi D Kuniyilh, Rita Machacy

**Abstract**: Contrastive learning has become a leading self- supervised approach to representation learning across domains, including vision, multimodal settings, graphs, and federated learning. However, recent studies have shown that contrastive learning is susceptible to backdoor and data poisoning attacks. In these attacks, adversaries can manipulate pretraining data or model updates to insert hidden malicious behavior. This paper offers a thorough and comparative review of backdoor attacks in contrastive learning. It analyzes threat models, attack methods, target domains, and available defenses. We summarize recent advancements in this area, underline the specific vulnerabilities inherent to contrastive learning, and discuss the challenges and future research directions. Our findings have significant implications for the secure deployment of systems in industrial and distributed environments.

摘要: 对比学习已成为跨领域（包括视觉、多模式设置、图形和联邦学习）表示学习的领先自我监督方法。然而，最近的研究表明，对比学习很容易受到后门和数据中毒攻击。在这些攻击中，对手可以操纵预训练数据或模型更新以插入隐藏的恶意行为。本文对对比学习中的后门攻击进行了彻底的比较审查。它分析威胁模型、攻击方法、目标域和可用的防御。我们总结了该领域的最新进展，强调了对比学习固有的具体弱点，并讨论了挑战和未来的研究方向。我们的研究结果对工业和分布式环境中系统的安全部署具有重大影响。



## **3. AJAR: Adaptive Jailbreak Architecture for Red-teaming**

Asimmon：红色团队的自适应越狱架构 cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10971v1) [paper-pdf](https://arxiv.org/pdf/2601.10971v1)

**Authors**: Yipu Dou, Wang Yang

**Abstract**: As Large Language Models (LLMs) evolve from static chatbots into autonomous agents capable of tool execution, the landscape of AI safety is shifting from content moderation to action security. However, existing red-teaming frameworks remain bifurcated: they either focus on rigid, script-based text attacks or lack the architectural modularity to simulate complex, multi-turn agentic exploitations. In this paper, we introduce AJAR (Adaptive Jailbreak Architecture for Red-teaming), a proof-of-concept framework designed to bridge this gap through Protocol-driven Cognitive Orchestration. Built upon the robust runtime of Petri, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, encapsulating state-of-the-art algorithms like X-Teaming as standardized, plug-and-play services. We validate the architectural feasibility of AJAR through a controlled qualitative case study, demonstrating its ability to perform stateful backtracking within a tool-use environment. Furthermore, our preliminary exploration of the "Agentic Gap" reveals a complex safety dynamic: while tool usage introduces new injection vectors via code execution, the cognitive load of parameter formatting can inadvertently disrupt persona-based attacks. AJAR is open-sourced to facilitate the standardized, environment-aware evaluation of this emerging attack surface. The code and data are available at https://github.com/douyipu/ajar.

摘要: 随着大型语言模型（LLM）从静态聊天机器人演变为能够执行工具的自主代理，人工智能安全的格局正在从内容审核转向动作安全。然而，现有的红队框架仍然存在分歧：它们要么专注于严格的基于脚本的文本攻击，要么缺乏模拟复杂的多回合代理开发的架构模块化。在本文中，我们介绍了April（自适应越狱架构红队），一个概念验证框架，旨在通过协议驱动的认知演示来弥合这一差距。基于Petri的强大运行时，April利用模型上下文协议（MCP）将对抗逻辑从执行循环中解耦，将X-Teaming等最先进的算法封装为标准化的即插即用服务。我们通过受控定性案例研究验证了Atomic的架构可行性，展示了其在工具使用环境中执行有状态回溯的能力。此外，我们对“统计差距”的初步探索揭示了一个复杂的安全动态：虽然工具使用通过代码执行引入新的注入载体，但参数格式的认知负载可能会无意中扰乱基于人物的攻击。Aspects是开源的，以促进对这一新兴攻击表面进行标准化、环境意识的评估。代码和数据可在https://github.com/douyipu/ajar上获取。



## **4. SecMLOps: A Comprehensive Framework for Integrating Security Throughout the MLOps Lifecycle**

SecMLOps：在MLOps工作空间中集成安全性的全面框架 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10848v1) [paper-pdf](https://arxiv.org/pdf/2601.10848v1)

**Authors**: Xinrui Zhang, Pincan Zhao, Jason Jaskolka, Heng Li, Rongxing Lu

**Abstract**: Machine Learning (ML) has emerged as a pivotal technology in the operation of large and complex systems, driving advancements in fields such as autonomous vehicles, healthcare diagnostics, and financial fraud detection. Despite its benefits, the deployment of ML models brings significant security challenges, such as adversarial attacks, which can compromise the integrity and reliability of these systems. To address these challenges, this paper builds upon the concept of Secure Machine Learning Operations (SecMLOps), providing a comprehensive framework designed to integrate robust security measures throughout the entire ML operations (MLOps) lifecycle. SecMLOps builds on the principles of MLOps by embedding security considerations from the initial design phase through to deployment and continuous monitoring. This framework is particularly focused on safeguarding against sophisticated attacks that target various stages of the MLOps lifecycle, thereby enhancing the resilience and trustworthiness of ML applications. A detailed advanced pedestrian detection system (PDS) use case demonstrates the practical application of SecMLOps in securing critical MLOps. Through extensive empirical evaluations, we highlight the trade-offs between security measures and system performance, providing critical insights into optimizing security without unduly impacting operational efficiency. Our findings underscore the importance of a balanced approach, offering valuable guidance for practitioners on how to achieve an optimal balance between security and performance in ML deployments across various domains.

摘要: 机器学习（ML）已成为大型复杂系统运营中的关键技术，推动了自动驾驶汽车、医疗保健诊断和金融欺诈检测等领域的进步。尽管ML模型有好处，但它的部署也带来了重大的安全挑战，例如对抗性攻击，这可能会损害这些系统的完整性和可靠性。为了应对这些挑战，本文以安全机器学习操作（SecMLOps）的概念为基础，提供了一个全面的框架，旨在在整个ML操作（MLOps）生命周期中集成强大的安全措施。SecMLOps基于MLOps的原则，通过嵌入从初始设计阶段到部署和持续监控的安全考虑因素。该框架特别专注于防范针对MLOps生命周期各个阶段的复杂攻击，从而增强ML应用程序的弹性和可信度。详细的高级行人检测系统（DDS）用例演示了SecMLOps在确保关键MLOps方面的实际应用。通过广泛的实证评估，我们强调了安全措施和系统性能之间的权衡，为在不过度影响运营效率的情况下优化安全性提供了重要见解。我们的研究结果强调了平衡方法的重要性，为从业者提供了关于如何在跨各个领域的ML部署中实现安全性和性能之间的最佳平衡的宝贵指导。



## **5. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay**

成为你自己的红色团队：通过自我游戏和反思体验重播实现安全调整 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10589v1) [paper-pdf](https://arxiv.org/pdf/2601.10589v1)

**Authors**: Hao Wang, Yanting Wang, Hao Li, Rui Li, Lei Sha

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.

摘要: 大型语言模型（LLM）已实现非凡的功能，但仍然容易受到旨在绕过安全护栏的对抗性“越狱”攻击的影响。当前的安全对齐方法严重依赖于静态外部红色团队，利用固定的防御提示或预先收集的对抗数据集。这导致了一种过于适合已知模式的严格防御，并且未能概括为新颖、复杂的威胁。为了解决这一关键局限性，我们建议让模型成为自己的红色团队，能够实现自主和不断发展的对抗性攻击。具体来说，我们引入了安全自助游戏（STP），这是一个利用单个LLM在统一的强化学习（RL）循环中同时充当攻击者（生成越狱）和防御者（拒绝有害请求）的系统，动态进化攻击策略以发现漏洞，同时加强防御机制。为了确保Defender有效解决自助游戏期间的关键安全问题，我们引入了先进的反思体验重播机制，该机制利用整个过程中积累的经验库。该机制采用上置信界（UCB）抽样策略来关注回报较低的失败案例，帮助模型从过去的严重错误中学习，同时平衡探索和利用。大量实验表明，我们的STP方法可以自主进化强大的防御能力，显着优于在静态对抗数据集上训练的基线，并为主动安全调整建立了新的基准。



## **6. Adversarial Evasion Attacks on Computer Vision using SHAP Values**

使用SHAP值对计算机视觉进行对抗规避攻击 cs.CV

10th bwHPC Symposium - September 25th & 26th, 2024

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10587v1) [paper-pdf](https://arxiv.org/pdf/2601.10587v1)

**Authors**: Frank Mollard, Marcus Becker, Florian Roehrbein

**Abstract**: The paper introduces a white-box attack on computer vision models using SHAP values. It demonstrates how adversarial evasion attacks can compromise the performance of deep learning models by reducing output confidence or inducing misclassifications. Such attacks are particularly insidious as they can deceive the perception of an algorithm while eluding human perception due to their imperceptibility to the human eye. The proposed attack leverages SHAP values to quantify the significance of individual inputs to the output at the inference stage. A comparison is drawn between the SHAP attack and the well-known Fast Gradient Sign Method. We find evidence that SHAP attacks are more robust in generating misclassifications particularly in gradient hiding scenarios.

摘要: 本文介绍了使用SHAP值对计算机视觉模型的白盒攻击。它展示了对抗性规避攻击如何通过降低输出信心或引发错误分类来损害深度学习模型的性能。此类攻击尤其阴险，因为它们可以欺骗算法的感知，同时由于人眼不可感知而逃避人类感知。拟议的攻击利用SHAP值来量化推理阶段各个输入对输出的重要性。比较了SHAP攻击与著名的快速梯度符号法。我们发现有证据表明SHAP攻击在生成错误分类方面更加强大，特别是在梯度隐藏场景中。



## **7. SRAW-Attack: Space-Reweighted Adversarial Warping Attack for SAR Target Recognition**

SRAW攻击：用于SAR目标识别的空间重加权对抗扭曲攻击 cs.CV

5 pages, 4 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10324v1) [paper-pdf](https://arxiv.org/pdf/2601.10324v1)

**Authors**: Yiming Zhang, Weibo Qin, Yuntian Liu, Feng Wang

**Abstract**: Synthetic aperture radar (SAR) imagery exhibits intrinsic information sparsity due to its unique electromagnetic scattering mechanism. Despite the widespread adoption of deep neural network (DNN)-based SAR automatic target recognition (SAR-ATR) systems, they remain vulnerable to adversarial examples and tend to over-rely on background regions, leading to degraded adversarial robustness. Existing adversarial attacks for SAR-ATR often require visually perceptible distortions to achieve effective performance, thereby necessitating an attack method that balances effectiveness and stealthiness. In this paper, a novel attack method termed Space-Reweighted Adversarial Warping (SRAW) is proposed, which generates adversarial examples through optimized spatial deformation with reweighted budgets across foreground and background regions. Extensive experiments demonstrate that SRAW significantly degrades the performance of state-of-the-art SAR-ATR models and consistently outperforms existing methods in terms of imperceptibility and adversarial transferability. Code is made available at https://github.com/boremycin/SAR-ATR-TransAttack.

摘要: 合成孔径雷达（SAR）图像由于其独特的电磁散射机制，具有内在的信息稀疏性。尽管基于深度神经网络（DNN）的SAR自动目标识别（SAR-ATR）系统被广泛采用，但它们仍然容易受到对抗性样本的影响，并且倾向于过度依赖背景区域，导致对抗性鲁棒性下降。针对SAR-ATR的现有对抗性攻击通常需要视觉上可感知的失真来实现有效性能，从而需要平衡有效性和隐蔽性的攻击方法。本文提出了一种新的攻击方法，称为空间重新加权对抗性Warping（SRAW），它通过优化的空间变形生成对抗性的例子，并在前景和背景区域之间进行重新加权预算。大量实验表明，SRW显着降低了最先进SAR-TAR模型的性能，并且在不可感知性和对抗性可移植性方面始终优于现有方法。代码可在https://github.com/boremycin/SAR-ATR-TransAttack上获得。



## **8. Hierarchical Refinement of Universal Multimodal Attacks on Vision-Language Models**

视觉语言模型上通用多模态攻击的层次细化 cs.CV

15 pages, 7 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10313v1) [paper-pdf](https://arxiv.org/pdf/2601.10313v1)

**Authors**: Peng-Fei Zhang, Zi Huang

**Abstract**: Existing adversarial attacks for VLP models are mostly sample-specific, resulting in substantial computational overhead when scaled to large datasets or new scenarios. To overcome this limitation, we propose Hierarchical Refinement Attack (HRA), a multimodal universal attack framework for VLP models. HRA refines universal adversarial perturbations (UAPs) at both the sample level and the optimization level. For the image modality, we disentangle adversarial examples into clean images and perturbations, allowing each component to be handled independently for more effective disruption of cross-modal alignment. We further introduce a ScMix augmentation strategy that diversifies visual contexts and strengthens both global and local utility of UAPs, thereby reducing reliance on spurious features. In addition, we refine the optimization path by leveraging a temporal hierarchy of historical and estimated future gradients to avoid local minima and stabilize universal perturbation learning. For the text modality, HRA identifies globally influential words by combining intra-sentence and inter-sentence importance measures, and subsequently utilizes these words as universal text perturbations. Extensive experiments across various downstream tasks, VLP models, and datasets demonstrate the superiority of the proposed universal multimodal attacks.

摘要: 针对VLP模型的现有对抗攻击大多是特定于样本的，当扩展到大型数据集或新场景时，会导致大量计算负担。为了克服这一限制，我们提出了分层细化攻击（HRA），这是一种VLP模型的多模式通用攻击框架。HRA在样本级别和优化级别细化了普遍对抗扰动（UPC）。对于图像模式，我们将对抗示例分解为干净的图像和扰动，允许独立处理每个组件，以更有效地破坏跨模式对齐。我们进一步引入了ScMix增强策略，该策略使视觉环境多样化，并增强了UAP的全球和本地实用性，从而减少了对虚假特征的依赖。此外，我们通过利用历史和估计的未来梯度的时间层次结构来细化优化路径，以避免局部极小值并稳定通用扰动学习。对于文本形式，HRA通过结合句内和句间重要性测量来识别具有全球影响力的单词，并随后利用这些单词作为通用文本扰动。跨各种下游任务、VLP模型和数据集的广泛实验证明了所提出的通用多模式攻击的优越性。



## **9. Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection**

推理劫持：通过决策标准注入颠覆LLM分类 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10294v1) [paper-pdf](https://arxiv.org/pdf/2601.10294v1)

**Authors**: Yuansen Liu, Yixuan Tang, Anthony Kum Hoe Tun

**Abstract**: Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack

摘要: 当前的LLM安全研究主要集中在减轻目标劫持，防止攻击者重定向模型的高级目标（例如，从“汇总电子邮件”到“网络钓鱼用户”）。在本文中，我们认为这个观点是不完整的，并强调了推理对齐中的一个关键漏洞。我们提出了一种新的对抗范式：推理劫持并使用Criteries Attack对其进行实例化，它通过注入虚假的决策标准来颠覆模型判断，而不改变高级任务目标。与试图覆盖系统提示的目标劫持不同，推理劫持接受高级目标，但通过注入虚假推理捷径来操纵模型的决策逻辑。尽管对三个不同任务（有毒评论、负面评论和垃圾邮件检测）进行了广泛的实验，但我们证明，即使是最新的模型也倾向于优先考虑注入的启发式捷径而不是严格的语义分析。不同主干的结果是一致的。至关重要的是，由于模型的“意图”与用户的指令保持一致，因此这些攻击可以绕过旨在检测目标偏差的防御措施（例如，SecAlign、StruQ），暴露了当前安全格局中的一个根本盲点。数据和代码可访问https://github.com/Yuan-Hou/criteria_attack



## **10. Understanding and Preserving Safety in Fine-Tuned LLMs**

了解和维护精调LLM的安全性 cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10141v1) [paper-pdf](https://arxiv.org/pdf/2601.10141v1)

**Authors**: Jiawen Zhang, Yangfan Hu, Kejia Chen, Lipeng He, Jiachen Ma, Jian Lou, Dan Li, Jian Liu, Xiaohu Yang, Ruoxi Jia

**Abstract**: Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.

摘要: 微调是将大型语言模型（LLM）应用于下游任务的基本且普遍的功能。然而，它有可能大幅降低安全对准，例如，即使微调数据完全无害，也可以极大地增加越狱攻击的易感性。尽管在微调阶段在国防工作中受到越来越多的关注，但现有的方法仍面临着持续的安全-效用困境：强调安全性会损害任务性能，而优先考虑效用通常需要深度微调，这不可避免地会导致安全性急剧下降。   在这项工作中，我们通过对安全对齐的LLM中安全导向梯度和实用导向梯度之间的几何相互作用提出新的见解来解决这一困境。通过系统的实证分析，我们揭示了三个关键见解：（一）安全梯度位于低阶子空间，而效用梯度跨越更广泛的多维空间;（二）这些子空间通常呈负相关，导致微调过程中的方向冲突;（三）可以有效地估计主导安全方向从单个样本。在这些新颖见解的基础上，我们提出了安全保护微调（SPF），这是一种轻量级方法，可以显式地去除与低等级安全子空间冲突的梯度分量。从理论上讲，我们表明SPF保证了效用收敛，同时限制了安全漂移。从经验上看，即使在敌对的微调场景下，SPF也能始终如一地保持下游任务性能，并恢复几乎所有预先训练的安全对齐。此外，SPF对深度微调和动态越狱攻击都表现出强大的抵抗力。我们的研究结果共同为始终一致的LLM微调提供了新的机制理解和实践指导。



## **11. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations**

SoK：隐私意识LLM在医疗保健：威胁模型，隐私技术，挑战和建议 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10004v1) [paper-pdf](https://arxiv.org/pdf/2601.10004v1)

**Authors**: Mohoshin Ara Tahera, Karamveer Singh Sidhu, Shuvalaxmi Dass, Sajal Saha

**Abstract**: Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.

摘要: 大型语言模型（LLM）在医疗保健中越来越多地采用，以支持临床决策、总结电子健康记录（EHR）并加强患者护理。然而，由于临床数据的敏感性和医疗工作流程的高风险性质，这种集成带来了重大的隐私和安全挑战。这些风险在不同的部署环境中变得更加明显，从小型本地医院系统到区域医疗网络，每个环境都有独特的资源限制和监管要求。知识系统化（SoK）研究了LLM三个核心阶段不断变化的威胁格局：数据预处理、微调和现实医疗保健环境中的推理。我们提出了一个详细的威胁模型，描述了每个阶段的对手、能力和攻击表面，并系统化了现有的隐私保护技术（PPT）如何尝试缓解这些漏洞。虽然现有的防御措施显示出希望，但我们的分析发现，在保护不同运营层级的敏感临床数据方面存在持续的局限性。我们最后提出了阶段感知建议和未来研究方向，旨在加强受监管环境中LLM的隐私保障。这项工作为了解医疗保健领域LLM、威胁和隐私的交叉点提供了基础，并为迈向更强大、临床值得信赖的人工智能系统提供了路线图。



## **12. Diffusion-Driven Deceptive Patches: Adversarial Manipulation and Forensic Detection in Facial Identity Verification**

扩散驱动的欺骗补丁：面部身份验证中的对抗操纵和法医检测 cs.CV

This manuscript is a preprint. A revised version of this work has been accepted for publication in the Springer Nature book Artificial Intelligence-Driven Forensics. This version includes one additional figure for completeness

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09806v1) [paper-pdf](https://arxiv.org/pdf/2601.09806v1)

**Authors**: Shahrzad Sayyafzadeh, Hongmei Chi, Shonda Bernadin

**Abstract**: This work presents an end-to-end pipeline for generating, refining, and evaluating adversarial patches to compromise facial biometric systems, with applications in forensic analysis and security testing. We utilize FGSM to generate adversarial noise targeting an identity classifier and employ a diffusion model with reverse diffusion to enhance imperceptibility through Gaussian smoothing and adaptive brightness correction, thereby facilitating synthetic adversarial patch evasion. The refined patch is applied to facial images to test its ability to evade recognition systems while maintaining natural visual characteristics. A Vision Transformer (ViT)-GPT2 model generates captions to provide a semantic description of a person's identity for adversarial images, supporting forensic interpretation and documentation for identity evasion and recognition attacks. The pipeline evaluates changes in identity classification, captioning results, and vulnerabilities in facial identity verification and expression recognition under adversarial conditions. We further demonstrate effective detection and analysis of adversarial patches and adversarial samples using perceptual hashing and segmentation, achieving an SSIM of 0.95.

摘要: 这项工作提供了一个端到端的管道，用于生成、完善和评估对抗补丁以损害面部生物识别系统，并应用于法医分析和安全测试。我们利用FGSM来生成针对身份分类器的对抗性噪音，并采用具有反向扩散的扩散模型通过高斯平滑和自适应亮度修正来增强不可感知性，从而促进合成对抗性补丁规避。经过改进的补丁应用于面部图像，以测试其躲避识别系统同时保持自然视觉特征的能力。Vision Transformer（ViT）-GPT 2模型生成字幕，为对抗图像提供个人身份的语义描述，支持身份规避和识别攻击的法医解释和文档。该管道评估身份分类、字幕结果的变化以及对抗条件下面部身份验证和表情识别的漏洞。我们进一步展示了使用感知哈希和分割对对抗补丁和对抗样本的有效检测和分析，实现了0.95的SSIM。



## **13. Efficient State Preparation for Quantum Machine Learning**

量子机器学习的有效状态准备 quant-ph

This book chapter has been accepted for Springer Nature Quantum Robustness in Artificial Intelligence and will appear in the book: https://link.springer.com/book/9783032111524?srsltid=AfmBOood7vZYc5xJYtLrQWND4pjedgfWAfAFFocjvnNS1lrNpVBwvJcO#accessibility-information

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09363v1) [paper-pdf](https://arxiv.org/pdf/2601.09363v1)

**Authors**: Chris Nakhl, Maxwell West, Muhammad Usman

**Abstract**: One of the key considerations in the development of Quantum Machine Learning (QML) protocols is the encoding of classical data onto a quantum device. In this chapter we introduce the Matrix Product State representation of quantum systems and show how it may be used to construct circuits which encode a desired state. Putting this in the context of QML we show how this process may be modified to give a low depth approximate encoding and crucially that this encoding does not hinder classification accuracy and is indeed exhibits an increased robustness against classical adversarial attacks. This is illustrated by demonstrations of adversarially robust variational quantum classifiers for the MNIST and FMNIST dataset, as well as a small-scale experimental demonstration on a superconducting quantum device.

摘要: 开发量子机器学习（QML）协议的关键考虑因素之一是将经典数据编码到量子设备上。在这一章中，我们将介绍量子系统的矩阵积态表示，并展示如何使用它来构造编码所需状态的电路。将其置于QML的背景下，我们展示了如何修改此过程以提供低深度近似编码，并且至关重要的是，这种编码不会妨碍分类准确性，并且确实表现出对经典对抗性攻击的鲁棒性。这说明了对抗强大的变分量子分类器的MNIST和FMNIST数据集的演示，以及超导量子器件上的小规模实验演示。



## **14. Too Helpful to Be Safe: User-Mediated Attacks on Planning and Web-Use Agents**

太有帮助而不安全：用户调解的对规划和网络使用代理的攻击 cs.CR

Keywords: LLM Agents; User-Mediated Attack; Agent Security; Human Factors in Cybersecurity; Web-Use Agents; Planning Agents; Benchmark

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.10758v1) [paper-pdf](https://arxiv.org/pdf/2601.10758v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Large Language Models (LLMs) have enabled agents to move beyond conversation toward end-to-end task execution and become more helpful. However, this helpfulness introduces new security risks stem less from direct interface abuse than from acting on user-provided content. Existing studies on agent security largely focus on model-internal vulnerabilities or adversarial access to agent interfaces, overlooking attacks that exploit users as unintended conduits. In this paper, we study user-mediated attacks, where benign users are tricked into relaying untrusted or attacker-controlled content to agents, and analyze how commercial LLM agents respond under such conditions. We conduct a systematic evaluation of 12 commercial agents in a sandboxed environment, covering 6 trip-planning agents and 6 web-use agents, and compare agent behavior across scenarios with no, soft, and hard user-requested safety checks. Our results show that agents are too helpful to be safe by default. Without explicit safety requests, trip-planning agents bypass safety constraints in over 92% of cases, converting unverified content into confident booking guidance. Web-use agents exhibit near-deterministic execution of risky actions, with 9 out of 17 supported tests reaching a 100% bypass rate. Even when users express soft or hard safety intent, constraint bypass remains substantial, reaching up to 54.7% and 7% for trip-planning agents, respectively. These findings reveal that the primary issue is not a lack of safety capability, but its prioritization. Agents invoke safety checks only conditionally when explicitly prompted, and otherwise default to goal-driven execution. Moreover, agents lack clear task boundaries and stopping rules, frequently over-executing workflows in ways that lead to unnecessary data disclosure and real-world harm.

摘要: 大型语言模型（LLM）使代理能够超越对话，走向端到端的任务执行，变得更有帮助。然而，这种有益的做法引入了新的安全风险，这些风险更多地来自于对用户提供的内容的操作，而不是直接的界面滥用。现有的代理安全研究主要集中在模型内部的漏洞或对代理接口的对抗性访问，忽略了利用用户作为意外管道的攻击。在本文中，我们研究了用户介导的攻击，其中良性用户被诱骗将不受信任或攻击者控制的内容中继给代理，并分析了商业LLM代理在这种情况下如何响应。我们在沙箱环境中对12个商业代理进行了系统评估，涵盖6个旅行规划代理和6个网络使用代理，并在无、软和硬用户请求的安全检查的情况下比较代理行为。我们的结果表明，代理太有帮助，默认情况下是安全的。在没有明确的安全要求的情况下，旅行计划代理在超过92%的情况下绕过了安全限制，将未经验证的内容转化为自信的预订指导。Web使用代理几乎确定地执行危险操作，支持的17个测试中有9个达到了100%的绕过率。即使用户表达了软或硬安全意图，约束绕过仍然很大，出行规划代理的比例分别高达54.7%和7%。这些调查结果表明，主要问题不是缺乏安全能力，而是其优先顺序。代理仅在显式提示时有条件地调用安全检查，否则默认为目标驱动执行。此外，代理缺乏明确的任务边界和停止规则，经常过度执行工作流程，导致不必要的数据泄露和现实世界的伤害。



## **15. Merged Bitcoin: Proof of Work Blockchains with Multiple Hash Types**

合并的比特币：具有多种哈希类型的工作量证明区块链 cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09090v1) [paper-pdf](https://arxiv.org/pdf/2601.09090v1)

**Authors**: Christopher Blake, Chen Feng, Xuachao Wang, Qianyu Yu

**Abstract**: Proof of work blockchain protocols using multiple hash types are considered. It is proven that the security region of such a protocol cannot be the AND of a 51\% attack on all the hash types. Nevertheless, a protocol called Merged Bitcoin is introduced, which is the Bitcoin protocol where links between blocks can be formed using multiple different hash types. Closed form bounds on its security region in the $Δ$-bounded delay network model are proven, and these bounds are compared to simulation results. This protocol is proven to maximize cost of attack in the linear cost-per-hash model. A difficulty adjustment method is introduced, and it is argued that this can partly remedy asymmetric advantages an adversary may gain in hashing power for some hash types, including from algorithmic advances, quantum attacks like Grover's algorithm, or hardware backdoor attacks.

摘要: 考虑使用多种哈希类型的工作量证明区块链协议。事实证明，此类协议的安全区域不能是对所有哈希类型进行51%攻击的AND。尽管如此，还是引入了一种名为Merged Bitcoin的协议，这是比特币协议，其中可以使用多种不同的哈希类型形成块之间的链接。证明了$Δ$-有界延迟网络模型中其安全区域的封闭形式界，并将这些界与模拟结果进行了比较。该协议被证明可以在线性每哈希成本模型中最大化攻击成本。引入了一种难度调整方法，并认为这可以部分弥补对手在某些哈希类型的哈希能力中可能获得的不对称优势，包括来自算法进步、Grover算法等量子攻击或硬件后门攻击。



## **16. StegoStylo: Squelching Stylometric Scrutiny through Steganographic Stitching**

StegoStylo：通过隐写缝合压制风格审查 cs.CR

16 pages, 6 figures, 1 table

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09056v1) [paper-pdf](https://arxiv.org/pdf/2601.09056v1)

**Authors**: Robert Dilworth

**Abstract**: Stylometry--the identification of an author through analysis of a text's style (i.e., authorship attribution)--serves many constructive purposes: it supports copyright and plagiarism investigations, aids detection of harmful content, offers exploratory cues for certain medical conditions (e.g., early signs of dementia or depression), provides historical context for literary works, and helps uncover misinformation and disinformation. In contrast, when stylometry is employed as a tool for authorship verification--confirming whether a text truly originates from a claimed author--it can also be weaponized for malicious purposes. Techniques such as de-anonymization, re-identification, tracking, profiling, and downstream effects like censorship illustrate the privacy threats that stylometric analysis can enable. Building on these concerns, this paper further explores how adversarial stylometry combined with steganography can counteract stylometric analysis. We first present enhancements to our adversarial attack, $\textit{TraceTarnish}$, providing stronger evidence of its capacity to confound stylometric systems and reduce their attribution and verification accuracy. Next, we examine how steganographic embedding can be fine-tuned to mask an author's stylistic fingerprint, quantifying the level of authorship obfuscation achievable as a function of the proportion of words altered with zero-width Unicode characters. Based on our findings, steganographic coverage of 33% or higher seemingly ensures authorship obfuscation. Finally, we reflect on the ways stylometry can be used to undermine privacy and argue for the necessity of defensive tools like $\textit{TraceTarnish}$.

摘要: 文体学--通过分析文本的风格来识别作者（即，作者归因）--服务于许多建设性目的：它支持版权和抄袭调查，帮助检测有害内容，为某些医疗状况提供探索性线索（例如，痴呆症或抑郁症的早期迹象），为文学作品提供了历史背景，并有助于揭露错误信息和虚假信息。相比之下，当样式表被用作作者身份验证的工具（确认文本是否真正来自声称的作者）时，它也可能被武器化用于恶意目的。去匿名化、重新识别、跟踪、分析等技术以及审查等下游效应说明了风格分析可以带来的隐私威胁。基于这些担忧，本文进一步探讨了对抗性文体学与隐写术相结合如何抵消文体学分析。我们首先展示了对对抗攻击$\textit{TraceTarnish}$的增强，提供了更强有力的证据，证明其混淆文体系统并降低其归因和验证准确性的能力。接下来，我们研究如何微调隐写嵌入以掩盖作者的风格指纹，量化可实现的作者混淆水平，作为用零宽度Unicode字符更改的单词比例的函数。根据我们的研究结果，33%或更高的隐写覆盖率似乎确保了作者身份的混淆。最后，我们反思了使用样式测量来破坏隐私的方式，并认为$\textit{TraceTarnish}$等防御工具的必要性。



## **17. A Differential Geometry and Algebraic Topology Based Public-Key Cryptographic Algorithm in Presence of Quantum Adversaries**

量子对手存在下基于微几何和代数布局的公钥密码算法 cs.IT

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.10883v1) [paper-pdf](https://arxiv.org/pdf/2601.10883v1)

**Authors**: Andrea Rondelli

**Abstract**: In antiquity, the seal embodied trust, secrecy, and integrity in safeguarding the exchange of letters and messages. The purpose of this work is to continue this tradition in the contemporary era, characterized by the presence of quantum computers, classical supercomputers, and increasingly sophisticated artificial intelligence. We introduce Z-Sigil, an asymmetric public-key cryptographic algorithm grounded in functional analysis, differential geometry, and algebraic topology, with the explicit goal of achieving resistance against both classical and quantum attacks. The construction operates over the tangent fiber bundle of a compact Calabi-Yau manifold [13], where cryptographic keys are elements of vector tangent fibers, with a binary operation defined on tangent spaces of the base manifold giving rise to a groupoid structure. Encryption and decryption are performed iteratively on message blocks, enforcing a serial architecture designed to limit quantum parallelism [9,10]. Each block depends on secret geometric and analytic data, including a randomly chosen base point on the manifold, a selected section of the tangent fiber bundle, and auxiliary analytic data derived from operator determinants and Zeta function regularization [11]. The correctness and invertibility of the proposed algorithm are proven analytically. Furthermore, any adversarial attempt to recover the plaintext without the private key leads to an exponential growth of the adversarial search space,even under quantum speedups. The use of continuous geometric structures,non-linear operator compositions,and enforced blockwise serialization distinguishes this approach from existing quantum-safe cryptographic proposals based on primary discrete algebraic assumptions.

摘要: 在古代，印章体现了保护信件和信息交换的信任、保密和完整性。这项工作的目的是在当代延续这一传统，当代的特点是量子计算机、经典超级计算机和日益复杂的人工智能。我们引入了Z-Sigil，这是一种基于函数分析、微几何和代数布局的非对称公钥加密算法，其明确目标是实现抵抗经典和量子攻击的目标。该结构在紧致Calabi-Yau Manifics的切向纤维束上运行[13]，其中密钥是向切纤维的元素，在基Manifics的切向空间上定义了二元操作，从而产生群体结构。加密和解密是在消息块上迭代执行的，强制执行旨在限制量子并行性的序列架构[9，10]。每个块取决于秘密的几何和分析数据，包括在管壁上随机选择的基点、切向纤维束的选定部分，以及从运算符决定性和Zeta函数正规化中获得的辅助分析数据[11]。分析证明了所提出算法的正确性和可逆性。此外，即使在量子加速的情况下，任何在没有私有密钥的情况下恢复明文的对抗尝试都会导致对抗搜索空间的指数级增长。连续几何结构、非线性运算符组合和强制分块序列化的使用使这种方法与基于主要离散代数假设的现有量子安全密码提案区分开来。



## **18. SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models**

SafeRedir：快速嵌入重定向，以实现图像生成模型中的鲁棒取消学习 cs.CV

Code at https://github.com/ryliu68/SafeRedir

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08623v1) [paper-pdf](https://arxiv.org/pdf/2601.08623v1)

**Authors**: Renyang Liu, Kangjie Chen, Han Qiu, Jie Zhang, Kwok-Yan Lam, Tianwei Zhang, See-Kiong Ng

**Abstract**: Image generation models (IGMs), while capable of producing impressive and creative content, often memorize a wide range of undesirable concepts from their training data, leading to the reproduction of unsafe content such as NSFW imagery and copyrighted artistic styles. Such behaviors pose persistent safety and compliance risks in real-world deployments and cannot be reliably mitigated by post-hoc filtering, owing to the limited robustness of such mechanisms and a lack of fine-grained semantic control. Recent unlearning methods seek to erase harmful concepts at the model level, which exhibit the limitations of requiring costly retraining, degrading the quality of benign generations, or failing to withstand prompt paraphrasing and adversarial attacks. To address these challenges, we introduce SafeRedir, a lightweight inference-time framework for robust unlearning via prompt embedding redirection. Without modifying the underlying IGMs, SafeRedir adaptively routes unsafe prompts toward safe semantic regions through token-level interventions in the embedding space. The framework comprises two core components: a latent-aware multi-modal safety classifier for identifying unsafe generation trajectories, and a token-level delta generator for precise semantic redirection, equipped with auxiliary predictors for token masking and adaptive scaling to localize and regulate the intervention. Empirical results across multiple representative unlearning tasks demonstrate that SafeRedir achieves effective unlearning capability, high semantic and perceptual preservation, robust image quality, and enhanced resistance to adversarial attacks. Furthermore, SafeRedir generalizes effectively across a variety of diffusion backbones and existing unlearned models, validating its plug-and-play compatibility and broad applicability. Code and data are available at https://github.com/ryliu68/SafeRedir.

摘要: 图像生成模型（IGM）虽然能够产生令人印象深刻的创意内容，但通常会从其训练数据中记住广泛的不希望的概念，从而导致不安全内容的复制，例如NSFW图像和受版权保护的艺术风格。此类行为在现实世界的部署中构成了持续的安全和合规风险，并且由于此类机制的稳健性有限并且缺乏细粒度的语义控制，因此无法通过事后过滤来可靠地减轻风险。最近的取消学习方法试图在模型层面消除有害概念，这些概念表现出需要昂贵的再培训、降低良性一代的质量或无法承受及时的解释和对抗性攻击的局限性。为了解决这些挑战，我们引入了SafeRedir，这是一个轻量级的推理时框架，用于通过提示嵌入重定向进行鲁棒的取消学习。SafeRedir在不修改基础IGM的情况下，通过嵌入空间中的标记级干预，自适应地将不安全提示路由到安全的语义区域。该框架包括两个核心组件：用于识别不安全生成轨迹的潜伏感知多模式安全分类器，以及用于精确语义重定向的代币级增量生成器，配备了用于代币掩蔽和自适应扩展的辅助预测器，以本地化和调节干预。多个代表性取消学习任务的经验结果表明，SafeRedir实现了有效的取消学习能力、高度的语义和感知保留、稳健的图像质量以及增强的对抗性攻击抵抗力。此外，SafeRedir有效地推广了各种扩散主干和现有未学习的模型，验证了其即插即用兼容性和广泛适用性。代码和数据可在https://github.com/ryliu68/SafeRedir上获取。



## **19. MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization**

MASH：通过风格人性化逃避黑匣子AI生成的文本检测器 cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08564v1) [paper-pdf](https://arxiv.org/pdf/2601.08564v1)

**Authors**: Yongtong Gu, Songze Li, Xia Hu

**Abstract**: The increasing misuse of AI-generated texts (AIGT) has motivated the rapid development of AIGT detection methods. However, the reliability of these detectors remains fragile against adversarial evasions. Existing attack strategies often rely on white-box assumptions or demand prohibitively high computational and interaction costs, rendering them ineffective under practical black-box scenarios. In this paper, we propose Multi-stage Alignment for Style Humanization (MASH), a novel framework that evades black-box detectors based on style transfer. MASH sequentially employs style-injection supervised fine-tuning, direct preference optimization, and inference-time refinement to shape the distributions of AI-generated texts to resemble those of human-written texts. Experiments across 6 datasets and 5 detectors demonstrate the superior performance of MASH over 11 baseline evaders. Specifically, MASH achieves an average Attack Success Rate (ASR) of 92%, surpassing the strongest baselines by an average of 24%, while maintaining superior linguistic quality.

摘要: 人工智能生成文本（AIGT）的日益滥用推动了AIGT检测方法的快速发展。然而，这些探测器的可靠性对于对抗性规避仍然脆弱。现有的攻击策略通常依赖于白盒假设或要求极高的计算和交互成本，从而使它们在实际的黑匣子场景下无效。在本文中，我们提出了多阶段风格人性化对齐（MASH），这是一种基于风格转移规避黑匣子检测器的新型框架。MASH依次采用风格注入监督微调、直接偏好优化和推理时细化来塑造人工智能生成文本的分布，使其类似于人类书写文本的分布。6个数据集和5个检测器的实验证明了MASH比11个基线规避者的卓越性能。具体来说，MASH的平均攻击成功率（ASB）为92%，平均超过最强基线24%，同时保持卓越的语言质量。



## **20. Evaluating Role-Consistency in LLMs for Counselor Training**

评估LLM辅导员培训的角色一致性 cs.CL

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08892v1) [paper-pdf](https://arxiv.org/pdf/2601.08892v1)

**Authors**: Eric Rudolph, Natalie Engert, Jens Albrecht

**Abstract**: The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.

摘要: 在线咨询服务的兴起凸显了未来咨询师对有效培训方法的需求。本文扩展了对VirCo的研究，VirCo是一个在线咨询虚拟客户端，旨在通过模拟现实的客户互动来补充学术培训中的传统角色扮演方法。在之前的工作的基础上，我们引入了一个包含对抗攻击的新数据集，以测试大型语言模型（LLM）维护其分配角色（角色一致性）的能力。该研究的重点是评估Vicuna模型反应的角色一致性和一致性，并将这些发现与早期研究进行比较。此外，我们还评估和比较各种开源LLM在虚拟客户端交互期间维持角色一致性方面的性能。我们的贡献包括创建对抗数据集、评估对话一致性和角色一致性，以及提供不同LLM的比较分析。



## **21. BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts**

BenchOverFlow：通过纯文本预算来测量大型语言模型中的溢出 cs.CL

Accepted at TMLR 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08490v1) [paper-pdf](https://arxiv.org/pdf/2601.08490v1)

**Authors**: Erin Feiglin, Nir Hutnik, Raz Lapid

**Abstract**: We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance.

摘要: 我们研究了大型语言模型（LLM）的一种失败模式，其中纯文本提示会引发过多的输出，我们将这种现象称为“溢出”。与越狱或提示注入不同，溢出在普通交互设置下发生，并可能导致服务成本、延迟和跨用户性能下降，特别是在跨多个请求扩展时。除了可用性之外，还有经济和环境方面的利害关系：不必要的代币会增加每次请求的成本和能源消耗，从而导致大量运营支出和大规模碳足迹。此外，溢出代表了共享环境中计算放大和服务降级的实用载体。我们引入BenchOverflow，这是一个与模型无关的基准测试，包含九种纯文本提示策略，可以在没有对抗性后缀或规避政策的情况下放大输出量。使用一个标准化的协议，固定预算为5000个新的令牌，我们评估了9个开源和闭源模型，并观察到明显的长度变化和重尾分布。上限饱和率（CSR@1k/3 k/5 k）和经验累积分布函数（ECDF）量化了尾部风险;即时内方差和跨模型相关性表明，溢出具有广泛的可重复性，但在家族和攻击向量之间具有异质性。一个轻量级的缓解措施-一个固定的简洁性衰减器-衰减右尾并降低大多数模型中所有策略的CSR。我们的研究结果将长度控制定位为可衡量的可靠性，成本和可持续性问题，而不是风格上的怪癖。BenchOverflow通过对不同模型的长度控制鲁棒性进行标准化比较，为选择最大限度地减少资源浪费和运营费用的部署以及评估在不影响任务性能的情况下抑制计算放大的防御措施提供了实用基础。



## **22. Baiting AI: Deceptive Adversary Against AI-Protected Industrial Infrastructures**

诱饵人工智能：对受人工智能保护的工业基础设施的欺骗性攻击 cs.CR

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08481v1) [paper-pdf](https://arxiv.org/pdf/2601.08481v1)

**Authors**: Aryan Pasikhani, Prosanta Gope, Yang Yang, Shagufta Mehnaz, Biplab Sikdar

**Abstract**: This paper explores a new cyber-attack vector targeting Industrial Control Systems (ICS), particularly focusing on water treatment facilities. Developing a new multi-agent Deep Reinforcement Learning (DRL) approach, adversaries craft stealthy, strategically timed, wear-out attacks designed to subtly degrade product quality and reduce the lifespan of field actuators. This sophisticated method leverages DRL methodology not only to execute precise and detrimental impacts on targeted infrastructure but also to evade detection by contemporary AI-driven defence systems. By developing and implementing tailored policies, the attackers ensure their hostile actions blend seamlessly with normal operational patterns, circumventing integrated security measures. Our research reveals the robustness of this attack strategy, shedding light on the potential for DRL models to be manipulated for adversarial purposes. Our research has been validated through testing and analysis in an industry-level setup. For reproducibility and further study, all related materials, including datasets and documentation, are publicly accessible.

摘要: 本文探讨了一种针对工业控制系统（ICS）的新网络攻击载体，特别关注水处理设施。对手开发了一种新的多智能体深度强化学习（DRL）方法，策划了隐蔽、战略性定时、磨损攻击，旨在微妙地降低产品质量并缩短现场致动器的寿命。这种复杂的方法利用DRL方法不仅对目标基础设施执行精确且有害的影响，而且还可以逃避当代人工智能驱动的防御系统的检测。通过制定和实施量身定制的政策，攻击者确保其敌对行为与正常操作模式无缝融合，从而规避综合安全措施。我们的研究揭示了这种攻击策略的稳健性，揭示了DRL模型被操纵以达到对抗目的的可能性。我们的研究已通过行业级设置中的测试和分析得到验证。为了重现性和进一步研究，所有相关材料，包括数据集和文档，都可以公开访问。



## **23. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations**

SecureCAE：具有注射弹性的网络安全运营法学硕士助理 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07835v1) [paper-pdf](https://arxiv.org/pdf/2601.07835v1)

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Mohammed Mudassir Uddin, Shahnawaz Alam

**Abstract**: Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains.

摘要: 大型语言模型已成为安全运营中心的变革性工具，可以实现自动化日志分析、网络钓鱼分类和恶意软件解释;然而，在对抗性网络安全环境中的部署暴露了关键漏洞，从而引发注入攻击，其中嵌入安全制品中的恶意指令操纵模型行为。本文介绍了SecureCAE，这是一种新型防御框架，通过安全感知护栏、自适应宪法进化和直接偏好优化来扩展宪法人工智能原则，用于消除不安全的响应模式，解决高风险安全环境中传统安全机制不足以应对复杂的对抗性操纵的独特挑战。实验评估表明，与基线模型相比，SecureCAE将攻击成功率降低了94.7%，同时在良性安全分析任务上保持了95.1%的准确性，该框架结合了连续的红色团队反馈循环，能够动态适应新兴的攻击策略，并在持续的对抗压力下实现宪法遵守分数超过0.92。从而为将语言模型能力可信地集成到运营网络安全工作流程中奠定基础，并解决当前对抗领域人工智能安全方法中的关键差距。



## **24. Self-Creating Random Walks for Decentralized Learning under Pac-Man Attacks**

吃豆人攻击下的去中心化学习自创建随机行走 cs.MA

arXiv admin note: substantial text overlap with arXiv:2508.05663

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07674v1) [paper-pdf](https://arxiv.org/pdf/2601.07674v1)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the CREATE-IF-LATE (CIL) algorithm, which is a fully decentralized, resilient mechanism that enables self-creating RWs and prevents RW extinction in the presence of Pac-Man. Our theoretical analysis shows that the CIL algorithm guarantees several desirable properties, such as (i) non-extinction of the RW population, (ii) almost sure boundedness of the RW population, and (iii) convergence of RW-based stochastic gradient descent even in the presence of Pac-Man with a quantifiable deviation from the true optimum. Moreover, the learning process experiences at most a linear time delay due to Pac-Man interruptions and RW regeneration. Our extensive empirical results on both synthetic and public benchmark datasets validate our theoretical findings.

摘要: 由于管理费用低和可扩展性，基于随机游走（RW）的算法长期以来一直在分布式系统中流行，最近在去中心化学习中的应用越来越多。然而，它们对本地交互的依赖使它们本质上容易受到恶意行为的影响。在这项工作中，我们调查了一种对抗性威胁，我们称之为“吃豆人”攻击，其中恶意节点概率地终止访问它的任何RW。这种隐形行为逐渐从网络中消除活动RW，有效地停止学习过程，而不会触发失败警报。为了应对这一威胁，我们提出了CREATE-IF-LATE（CIL）算法，这是一种完全分散的弹性机制，可以在Pac-Man存在的情况下实现自我创建RW并防止RW灭绝。我们的理论分析表明，CIL算法保证了几个理想的性质，例如（i）RW种群的非灭绝，（ii）RW种群的几乎确定有界性，和（iii）即使在Pac-Man存在的情况下，基于RW的随机梯度下降也收敛，与真正的最佳值存在可量化的偏差。此外，由于Pac-Man中断和RW再生，学习过程最多会经历线性时间延迟。我们对合成和公共基准数据集的广泛实证结果验证了我们的理论发现。



## **25. Universal Adversarial Purification with DDIM Metric Loss for Stable Diffusion**

具有DDIM度量损失的通用对抗纯化以实现稳定扩散 cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07253v1) [paper-pdf](https://arxiv.org/pdf/2601.07253v1)

**Authors**: Li Zheng, Liangbin Xie, Jiantao Zhou, He YiMin

**Abstract**: Stable Diffusion (SD) often produces degraded outputs when the training dataset contains adversarial noise. Adversarial purification offers a promising solution by removing adversarial noise from contaminated data. However, existing purification methods are primarily designed for classification tasks and fail to address SD-specific adversarial strategies, such as attacks targeting the VAE encoder, UNet denoiser, or both. To address the gap in SD security, we propose Universal Diffusion Adversarial Purification (UDAP), a novel framework tailored for defending adversarial attacks targeting SD models. UDAP leverages the distinct reconstruction behaviors of clean and adversarial images during Denoising Diffusion Implicit Models (DDIM) inversion to optimize the purification process. By minimizing the DDIM metric loss, UDAP can effectively remove adversarial noise. Additionally, we introduce a dynamic epoch adjustment strategy that adapts optimization iterations based on reconstruction errors, significantly improving efficiency without sacrificing purification quality. Experiments demonstrate UDAP's robustness against diverse adversarial methods, including PID (VAE-targeted), Anti-DreamBooth (UNet-targeted), MIST (hybrid), and robustness-enhanced variants like Anti-Diffusion (Anti-DF) and MetaCloak. UDAP also generalizes well across SD versions and text prompts, showcasing its practical applicability in real-world scenarios.

摘要: 当训练数据集包含对抗性噪音时，稳定扩散（SD）通常会产生降级的输出。对抗性纯化通过从受污染的数据中去除对抗性噪音，提供了一个有希望的解决方案。然而，现有的纯化方法主要是为分类任务而设计的，无法解决SD特定的对抗策略，例如针对VAE编码器、UNet去噪器或两者的攻击。为了解决SD安全方面的差距，我们提出了通用扩散对抗纯化（UPC），这是一个专为防御针对SD模型的对抗攻击而量身定制的新型框架。DAB利用去噪扩散隐式模型（DDIM）倒置期间干净图像和对抗图像的不同重建行为来优化净化过程。通过最大限度地减少DDIM指标损失，BEP可以有效地消除对抗性噪音。此外，我们还引入了一种动态历元调整策略，该策略根据重建误差调整优化迭代，从而在不牺牲净化质量的情况下显着提高效率。实验证明了DPP对各种对抗方法的鲁棒性，包括ID（VAE目标）、Anti-DreamBooth（UNet目标）、MIST（混合）以及抗扩散（Anti-DF）和MetaCloak等鲁棒性增强变体。UDAP还可以很好地推广到SD版本和文本提示，展示了其在现实世界场景中的实用性。



## **26. PROTEA: Securing Robot Task Planning and Execution**

PROTEA：确保机器人任务规划和执行 cs.RO

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07186v1) [paper-pdf](https://arxiv.org/pdf/2601.07186v1)

**Authors**: Zainab Altaweel, Mohaiminul Al Nahian, Jake Juettner, Adnan Siraj Rakin, Shiqi Zhang

**Abstract**: Robots need task planning methods to generate action sequences for complex tasks. Recent work on adversarial attacks has revealed significant vulnerabilities in existing robot task planners, especially those built on foundation models. In this paper, we aim to address these security challenges by introducing PROTEA, an LLM-as-a-Judge defense mechanism, to evaluate the security of task plans. PROTEA is developed to address the dimensionality and history challenges in plan safety assessment. We used different LLMs to implement multiple versions of PROTEA for comparison purposes. For systemic evaluations, we created a dataset containing both benign and malicious task plans, where the harmful behaviors were injected at varying levels of stealthiness. Our results provide actionable insights for robotic system practitioners seeking to enhance robustness and security of their task planning systems. Details, dataset and demos are provided: https://protea-secure.github.io/PROTEA/

摘要: 机器人需要任务规划方法来生成复杂任务的动作序列。最近关于对抗性攻击的工作揭示了现有机器人任务规划器的重大漏洞，尤其是那些基于基础模型的机器人任务规划器。在本文中，我们的目标是通过引入PROTEA（一种法学硕士作为法官的辩护机制）来评估任务计划的安全性来解决这些安全挑战。PROTEA旨在解决计划安全评估中的维度和历史挑战。为了进行比较，我们使用不同的LLM来实现多个版本的PROTEA。对于系统性评估，我们创建了一个包含良性和恶意任务计划的数据集，其中有害行为以不同的隐蔽程度注入。我们的结果为寻求增强任务规划系统稳健性和安全性的机器人系统从业者提供了可操作的见解。提供了详细信息、数据集和演示：https://protea-secure.github.io/PROTEA/



## **27. Defenses Against Prompt Attacks Learn Surface Heuristics**

防御即时攻击学习表面启发法 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07185v1) [paper-pdf](https://arxiv.org/pdf/2601.07185v1)

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security.

摘要: 大型语言模型（LLM）越来越多地部署在安全敏感的应用程序中，它们必须遵循系统或开发人员指定的指令，这些指令定义了预期的任务行为，同时完成良性的用户请求。当对抗性指令出现在用户查询或外部检索的内容中时，模型可能会覆盖预期的逻辑。最近的防御依赖于良性和恶意标签的监督微调。虽然这些方法实现了高攻击拒绝率，但我们发现它们依赖于防御数据中的窄相关性，而不是有害的意图，从而导致系统拒绝安全输入。我们分析了防御微调引发的三种反复出现的捷径行为。\{位置偏差}当提示中稍后放置的良性内容被拒绝率高得多时，就会出现;在推理基准中，后缀任务拒绝率从低于\textBF{10\%}上升到高达\textBF{90\%}。\当攻击数据中常见的字符串即使在良性上下文中也会提高拒绝概率时，就会发生{Token触发偏差};插入单个触发令牌会增加错误拒绝最多\textBF{50\%}。\{主题概括偏差}反映了防御数据分布之外的较差概括，防御模型的测试时准确性下降高达\textBF{40\%}。这些发现表明，当前的预算注射防御经常对类似攻击的表面模式做出反应，而不是潜在意图。我们引入了受控诊断数据集和跨两个基本模型和多个防御管道的系统评估，强调了监督式微调以实现可靠的LLM安全性的局限性。



## **28. MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models**

MacPrompt：针对文本到图像模型的Maraconic引导越狱 cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07141v1) [paper-pdf](https://arxiv.org/pdf/2601.07141v1)

**Authors**: Xi Ye, Yiwen Liu, Lina Wang, Run Wang, Geying Yang, Yufei Hou, Jiayi Yu

**Abstract**: Text-to-image (T2I) models have raised increasing safety concerns due to their capacity to generate NSFW and other banned objects. To mitigate these risks, safety filters and concept removal techniques have been introduced to block inappropriate prompts or erase sensitive concepts from the models. However, all the existing defense methods are not well prepared to handle diverse adversarial prompts. In this work, we introduce MacPrompt, a novel black-box and cross-lingual attack that reveals previously overlooked vulnerabilities in T2I safety mechanisms. Unlike existing attacks that rely on synonym substitution or prompt obfuscation, MacPrompt constructs macaronic adversarial prompts by performing cross-lingual character-level recombination of harmful terms, enabling fine-grained control over both semantics and appearance. By leveraging this design, MacPrompt crafts prompts with high semantic similarity to the original harmful inputs (up to 0.96) while bypassing major safety filters (up to 100%). More critically, it achieves attack success rates as high as 92% for sex-related content and 90% for violence, effectively breaking even state-of-the-art concept removal defenses. These results underscore the pressing need to reassess the robustness of existing T2I safety mechanisms against linguistically diverse and fine-grained adversarial strategies.

摘要: 文本到图像（T2 I）模型由于能够生成NSFW和其他违禁对象，引发了越来越多的安全问题。为了减轻这些风险，引入了安全过滤器和概念删除技术，以阻止不适当的提示或从模型中删除敏感概念。然而，所有现有的防御方法都没有做好处理不同的对抗提示的准备。在这项工作中，我们引入了MacPrompt，这是一种新型黑匣子和跨语言攻击，揭示了T2 I安全机制中之前被忽视的漏洞。与依赖同义词替换或提示混淆的现有攻击不同，MacPrompt通过对有害术语执行跨语言字符级重组来构建宏卡龙式对抗提示，从而实现对语义和外观的细粒度控制。通过利用这一设计，MacPrompt处理与原始有害输入具有高度语义相似性的提示（高达0.96），同时绕过主要安全过滤器（高达100%）。更重要的是，它对性相关内容的攻击成功率高达92%，对暴力内容的攻击成功率高达90%，有效地突破了最先进的概念删除防御。这些结果强调了针对语言多样性和细粒度对抗策略重新评估现有T2 I安全机制的稳健性的迫切需要。



## **29. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework**

通过强大的LLM授权的多Agent强化学习框架增强云网络弹性 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07122v1) [paper-pdf](https://arxiv.org/pdf/2601.07122v1)

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.

摘要: 虽然虚拟化和资源池赋予云网络结构灵活性和弹性可扩展性，但它们不可避免地扩大了攻击面并挑战网络弹性。基于强化学习（RL）的防御策略被开发出来，以优化对抗条件下的资源部署和隔离政策，旨在通过维护和恢复网络可用性来增强系统的弹性。然而，现有的方法缺乏鲁棒性，因为它们需要重新训练以适应网络结构、节点规模、攻击策略和攻击强度的动态变化。此外，缺乏人在环（HITL）支持限制了可解释性和灵活性。为了解决这些限制，我们提出了CyberOps-Bots，这是一个由大型语言模型（LLM）支持的分层多智能体强化学习框架。CyberOps-Bots受到MITRE ATA & CK的Tactics-Techniques模型的启发，具有两层架构：（1）上层LLM代理，具有四个模块--ReAct规划、基于IPDRR的感知、长短期记忆和动作/工具集成--执行全球感知、人类意图识别和战术规划;（2）通过异类分离预训练开发的低级RL代理，在局部网络区域内执行原子防御动作。这种协同作用保留了LLM的适应性和可解释性，同时确保可靠的RL执行。对真实云数据集的实验表明，与最先进的算法相比，CyberOps-Bots的网络可用性提高了68.5%，并且在无需重新训练的情况下改变场景时实现了34.7%的启动性能提升。据我们所知，这是第一项建立具有HITL支持的强大LLM-RL框架的研究。我们将向社区发布我们的框架，促进云网络中稳健和自主防御的发展。



## **30. Reward-Preserving Attacks For Robust Reinforcement Learning**

鲁棒强化学习的奖励保护攻击 cs.LG

19 pages, 6 figures, 4 algorithms, preprint

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07118v1) [paper-pdf](https://arxiv.org/pdf/2601.07118v1)

**Authors**: Lucas Schott, Elies Gherbi, Hatem Hajri, Sylvain Lamprier

**Abstract**: Adversarial robustness in RL is difficult because perturbations affect entire trajectories: strong attacks can break learning, while weak attacks yield little robustness, and the appropriate strength varies by state. We propose $α$-reward-preserving attacks, which adapt the strength of the adversary so that an $α$ fraction of the nominal-to-worst-case return gap remains achievable at each state. In deep RL, we use a gradient-based attack direction and learn a state-dependent magnitude $η\le η_{\mathcal B}$ selected via a critic $Q^π_α((s,a),η)$ trained off-policy over diverse radii. This adaptive tuning calibrates attack strength and, with intermediate $α$, improves robustness across radii while preserving nominal performance, outperforming fixed- and random-radius baselines.

摘要: RL中的对抗鲁棒性很困难，因为扰动会影响整个轨迹：强攻击可能会破坏学习，而弱攻击几乎不会产生鲁棒性，并且适当的强度因状态而异。我们提出了$a $-奖励保留攻击，该攻击可以调整对手的实力，以便每个州仍然可以实现名义到最坏情况回报差距的$a $分数。在深度RL中，我们使用基于梯度的攻击方向，并学习通过批评者$Q^pi_a（s，a），n）$在不同半径上训练的非策略下选择的状态相关幅度$n\le n_{\mathcalB}$。这种自适应调整可以校准攻击强度，并且在中间$a $的情况下提高了半径范围内的鲁棒性，同时保持名义性能，优于固定和随机半径基线。



## **31. Memory Poisoning Attack and Defense on Memory Based LLM-Agents**

基于内存的LLM-Agents的内存中毒攻击与防御 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.05504v2) [paper-pdf](https://arxiv.org/pdf/2601.05504v2)

**Authors**: Balachandra Devarangadi Sunil, Isheeta Sinha, Piyush Maheshwari, Shantanu Todmal, Shreyan Mallik, Shuchi Mishra

**Abstract**: Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.

摘要: 配备持久内存的大型语言模型代理很容易受到内存中毒攻击，对手通过仅查询的交互注入恶意指令，从而破坏代理的长期内存并影响未来的响应。最近的工作表明，MINJA（内存注入攻击）在理想化条件下实现了超过95%的注入成功率和70%的攻击成功率。然而，这些攻击在现实部署中的稳健性和有效的防御机制仍然没有得到充分的研究。这项工作通过对电子健康记录（EHR）代理中的记忆中毒攻击和防御的系统性实证评估来解决这些差距。我们通过改变三个关键维度来研究攻击的稳健性：初始存储状态、指示提示数量和检索参数。我们使用MIIC-III临床数据对GPT-4 o-mini、Gemini-2.0-Flash和Llama-3.1- 8B-Direct模型进行的实验表明，具有预先存在的合法记忆的现实条件会显着降低攻击有效性。然后，我们提出并评估了两种新型防御机制：（1）使用多个垂直信号的复合信任评分进行输入/输出调节，以及（2）使用时间衰减和基于模式的过滤的信任感知检索进行记忆净化。我们的防御评估表明，有效的内存清理需要仔细的信任阈值校准，以防止过于保守的拒绝（阻止所有条目）和过滤不足（错过微妙攻击），为未来的自适应防御机制建立重要的基线。这些发现为在生产环境中保护内存增强的LLM代理提供了重要见解。



## **32. $PC^2$: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models**

$PC ' 2 $：通过对基于GPT的文本到图像模型的越狱攻击生成具有政治争议的内容 cs.CR

19 pages, 15 figures, 9 tables

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.05150v2) [paper-pdf](https://arxiv.org/pdf/2601.05150v2)

**Authors**: Wonwoo Choi, Minjae Seo, Minkyoo Song, Hwanjo Heo, Seungwon Shin, Myoungsung You

**Abstract**: The rapid evolution of text-to-image (T2I) models has enabled high-fidelity visual synthesis on a global scale. However, these advancements have introduced significant security risks, particularly regarding the generation of harmful content. Politically harmful content, such as fabricated depictions of public figures, poses severe threats when weaponized for fake news or propaganda. Despite its criticality, the robustness of current T2I safety filters against such politically motivated adversarial prompting remains underexplored. In response, we propose $PC^2$, the first black-box political jailbreaking framework for T2I models. It exploits a novel vulnerability where safety filters evaluate political sensitivity based on linguistic context. $PC^2$ operates through: (1) Identity-Preserving Descriptive Mapping to obfuscate sensitive keywords into neutral descriptions, and (2) Geopolitically Distal Translation to map these descriptions into fragmented, low-sensitivity languages. This strategy prevents filters from constructing toxic relationships between political entities within prompts, effectively bypassing detection. We construct a benchmark of 240 politically sensitive prompts involving 36 public figures. Evaluation on commercial T2I models, specifically GPT-series, shows that while all original prompts are blocked, $PC^2$ achieves attack success rates of up to 86%.

摘要: 文本到图像（T2 I）模型的快速发展使全球范围内的高保真视觉合成成为可能。然而，这些进步带来了重大的安全风险，特别是在有害内容的生成方面。政治上有害的内容，例如对公众人物的捏造描述，当被用作假新闻或宣传武器时，就会构成严重威胁。尽管其至关重要，但当前T2 I安全过滤器针对此类出于政治动机的对抗激励的稳健性仍然没有得到充分的探索。作为回应，我们提出了$PC^2$，这是T2 I模型的第一个黑箱政治越狱框架。它利用了一个新的漏洞，安全过滤器根据语言背景评估政治敏感性。$PC^2$通过以下方式运作：（1）身份保持描述性映射，将敏感的关键字混淆为中性描述;（2）地缘政治远端翻译，将这些描述映射为碎片化的低敏感度语言。这种策略可以防止过滤器在提示内构建政治实体之间的有毒关系，从而有效地绕过检测。我们构建了一个由240个政治敏感提示组成的基准，涉及36名公众人物。对商业T2 I型号（特别是GPT系列）的评估显示，虽然所有原始提示都被阻止，但$PC ' 2 $的攻击成功率高达86%。



## **33. MORE: Multi-Objective Adversarial Attacks on Speech Recognition**

更多：语音识别的多目标对抗攻击 eess.AS

19 pages

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.01852v2) [paper-pdf](https://arxiv.org/pdf/2601.01852v2)

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen

**Abstract**: The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.

摘要: Whisper等大规模自动语音识别（ASB）模型的出现极大地扩大了它们在各种现实世界应用中的采用。因此，确保对即使是微小的输入扰动的稳健性对于在实时环境中保持可靠的性能至关重要。虽然之前的工作主要研究了对抗性攻击下的准确性下降，但效率方面的稳健性在很大程度上仍然没有探索。这种狭隘的关注点仅提供了对ASB模型漏洞的部分了解。为了解决这一差距，我们对多种攻击场景下的ASB稳健性进行了全面研究。我们引入了MORE，这是一种多目标重复双倍鼓励攻击，它通过分层分阶段排斥锚定机制共同降低识别准确性和推理效率。具体来说，我们将多目标对抗优化重新定义为分层框架，以顺序实现双重目标。为了进一步提高有效性，我们提出了一种新型的重复鼓励加倍目标（REDO），它通过保持准确性下降和周期性地将预测序列长度翻倍来诱导重复文本生成。总体而言，MORE迫使ASB模型以高得多的计算成本生成错误的转录，由单一对抗输入触发。实验表明，与现有基线相比，MORE始终产生显着更长的转录时间，同时保持较高的字错误率，凸显了其在多目标对抗攻击中的有效性。



## **34. Measuring the Impact of Student Gaming Behaviors on Learner Modeling**

衡量学生游戏行为对学习者建模的影响 cs.CY

Full research paper accepted at Learning Analytics and Knowledge (LAK '26) conference, see https://doi.org/10.1145/3785022.3785036

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2512.18659v3) [paper-pdf](https://arxiv.org/pdf/2512.18659v3)

**Authors**: Qinyi Liu, Lin Li, Valdemar Švábenský, Conrad Borchers, Mohammad Khalil

**Abstract**: The expansion of large-scale online education platforms has made vast amounts of student interaction data available for knowledge tracing (KT). KT models estimate students' concept mastery from interaction data, but their performance is sensitive to input data quality. Gaming behaviors, such as excessive hint use, may misrepresent students' knowledge and undermine model reliability. However, systematic investigations of how different types of gaming behaviors affect KT remain scarce, and existing studies rely on costly manual analysis that does not capture behavioral diversity. In this study, we conceptualize gaming behaviors as a form of data poisoning, defined as the deliberate submission of incorrect or misleading interaction data to corrupt a model's learning process. We design Data Poisoning Attacks (DPAs) to simulate diverse gaming patterns and systematically evaluate their impact on KT model performance. Moreover, drawing on advances in DPA detection, we explore unsupervised approaches to enhance the generalizability of gaming behavior detection. We find that KT models' performance tends to decrease especially in response to random guess behaviors. Our findings provide insights into the vulnerabilities of KT models and highlight the potential of adversarial methods for improving the robustness of learning analytics systems.

摘要: 大型在线教育平台的扩张使大量学生互动数据可用于知识追踪（KT）。KT模型根据交互数据估计学生的概念掌握程度，但他们的表现对输入数据质量敏感。过度使用提示等游戏行为可能会歪曲学生的知识并破坏模型的可靠性。然而，关于不同类型的游戏行为如何影响KT的系统研究仍然很少，现有的研究依赖于昂贵的手动分析，无法捕捉行为多样性。在这项研究中，我们将游戏行为概念化为一种数据中毒形式，定义为故意提交不正确或误导性的交互数据以破坏模型的学习过程。我们设计数据中毒攻击（DPA）来模拟不同的游戏模式，并系统性评估其对KT模型性能的影响。此外，利用DPA检测的进步，我们探索无监督方法来增强游戏行为检测的通用性。我们发现KT模型的性能往往会下降，尤其是在响应随机猜测行为时。我们的研究结果深入了解了KT模型的漏洞，并强调了对抗方法在提高学习分析系统稳健性方面的潜力。



## **35. From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda**

从对抗性诗歌到对抗性故事：一个可解释性研究议程 cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.08837v2) [paper-pdf](https://arxiv.org/pdf/2601.08837v2)

**Authors**: Piercosma Bisconti, Marcello Galisai, Matteo Prandi, Federico Pierucci, Olga Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Marcantonio Bracale Syrnikov, Daniele Nardi

**Abstract**: Safety mechanisms in LLMs remain vulnerable to attacks that reframe harmful requests through culturally coded structures. We introduce Adversarial Tales, a jailbreak technique that embeds harmful content within cyberpunk narratives and prompts models to perform functional analysis inspired by Vladimir Propp's morphology of folktales. By casting the task as structural decomposition, the attack induces models to reconstruct harmful procedures as legitimate narrative interpretation. Across 26 frontier models from nine providers, we observe an average attack success rate of 71.3%, with no model family proving reliably robust. Together with our prior work on Adversarial Poetry, these findings suggest that structurally-grounded jailbreaks constitute a broad vulnerability class rather than isolated techniques. The space of culturally coded frames that can mediate harmful intent is vast, likely inexhaustible by pattern-matching defenses alone. Understanding why these attacks succeed is therefore essential: we outline a mechanistic interpretability research agenda to investigate how narrative cues reshape model representations and whether models can learn to recognize harmful intent independently of surface form.

摘要: LLM中的安全机制仍然容易受到通过文化编码结构重新定义有害请求的攻击。我们引入了对抗故事，这是一种越狱技术，将有害内容嵌入到赛博朋克叙事中，并促使模型执行受弗拉基米尔·普罗普民间故事形态学启发的功能分析。通过将任务视为结构分解，攻击诱导模型将有害程序重建为合法的叙事解释。在来自9家提供商的26个前沿模型中，我们观察到平均攻击成功率为71.3%，没有一个模型家族被证明可靠稳健。与我们之前关于对抗性诗歌的工作一起，这些发现表明，基于结构的越狱构成了一个广泛的脆弱性类别，而不是孤立的技术。可以调解有害意图的文化编码框架的空间是巨大的，仅通过模式匹配防御就可能无穷无尽。因此，了解这些攻击为何成功至关重要：我们概述了一个机械可解释性研究议程，以调查叙事线索如何重塑模型表示，以及模型是否可以独立于表面形式学会识别有害意图。



## **36. From static to adaptive: immune memory-based jailbreak detection for large language models**

从静态到适应性：基于免疫记忆的大型语言模型越狱检测 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2512.03356v2) [paper-pdf](https://arxiv.org/pdf/2512.03356v2)

**Authors**: Jun Leng, Yu Liu, Litian Zhang, Ruihan Hu, Zhuting Fang, Xi Zhang

**Abstract**: Large Language Models (LLMs) serve as the backbone of modern AI systems, yet they remain susceptible to adversarial jailbreak attacks. Consequently, robust detection of such malicious inputs is paramount for ensuring model safety. Traditional detection methods typically rely on external models trained on fixed, large-scale datasets, which often incur significant computational overhead. While recent methods shift toward leveraging internal safety signals of models to enable more lightweight and efficient detection. However, these methods remain inherently static and struggle to adapt to the evolving nature of jailbreak attacks. Drawing inspiration from the biological immune mechanism, we introduce the Immune Memory Adaptive Guard (IMAG) framework. By distilling and encoding safety patterns into a persistent, evolvable memory bank, IMAG enables adaptive generalization to emerging threats. Specifically, the framework orchestrates three synergistic components: Immune Detection, which employs retrieval for high-efficiency interception of known jailbreak attacks; Active Immunity, which performs proactive behavioral simulation to resolve ambiguous unknown queries; Memory Updating, which integrates validated attack patterns back into the memory bank. This closed-loop architecture transitions LLM defense from rigid filtering to autonomous adaptive mitigation. Extensive evaluations across five representative open-source LLMs demonstrate that our method surpasses state-of-the-art (SOTA) baselines, achieving a superior average detection accuracy of 94\% across diverse and complex attack types.

摘要: 大型语言模型（LLM）是现代人工智能系统的支柱，但它们仍然容易受到敌对越狱攻击。因此，对此类恶意输入的稳健检测对于确保模型安全至关重要。传统的检测方法通常依赖于在固定的大规模数据集上训练的外部模型，这通常会带来大量的计算负担。虽然最近的方法转向利用模型的内部安全信号来实现更轻量级和更高效的检测。然而，这些方法本质上仍然是静态的，并且很难适应越狱攻击不断变化的性质。从生物免疫机制中汲取灵感，我们介绍了免疫记忆自适应守卫（IMAG）框架。通过将安全模式提取和编码到持久的、可进化的记忆库中，IMAG能够对新出现的威胁进行自适应概括。具体来说，该框架协调了三个协同组件：免疫检测，它采用检索来高效拦截已知的越狱攻击;主动免疫，它执行主动行为模拟来解决模糊的未知查询;内存更新，它将验证的攻击模式集成到内存库中。这种闭环架构将LLM防御从严格过滤转变为自主自适应缓解。对五个代表性开源LLM的广泛评估表明，我们的方法超越了最先进的（SOTA）基线，在各种复杂的攻击类型中实现了94%的卓越平均检测准确率。



## **37. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

对抗性诗歌作为大型语言模型中通用的单轮越狱机制 cs.CL

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2511.15304v3) [paper-pdf](https://arxiv.org/pdf/2511.15304v3)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale Syrnikov, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.

摘要: 我们提供的证据表明，对抗性诗歌可以作为大型语言模型（LLM）的通用单轮越狱技术。在25个前沿专有和开放重量模型中，精心策划的诗意提示产生了很高的攻击成功率（ASB），一些提供商超过了90%。MLCommons和EU CoP风险分类的映射提示表明，诗意攻击跨CBRN、操纵、网络犯罪和失去控制领域转移。通过标准化元提示将1，200个MLCommons有害提示转换为诗句，产生的ASB比散文基线高出18倍。使用3名开放权重LLM评委的整体评估输出，他们的二元安全性评估在分层的人类标记子集上进行了验证。诗意框架的平均越狱成功率为62%，元提示转换的平均越狱成功率约为43%（与非诗意基线相比），大大优于非诗意基线，并揭示了示范家庭和安全培训方法之间的系统性弱点。这些研究结果表明，仅靠风格差异就可以规避当代的安全机制，这表明当前对齐方法和评估协议存在根本局限性。



## **38. Bribers, Bribers on The Chain, Is Resisting All in Vain? Trustless Consensus Manipulation Through Bribing Contracts**

行贿者，链上的行贿者，抵制一切都是徒劳的吗？通过贿赂合同进行不可信的共识操纵 cs.CR

To appear at Financial Cryptography and Data Security 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2509.17185v2) [paper-pdf](https://arxiv.org/pdf/2509.17185v2)

**Authors**: Bence Soóki-Tóth, István András Seres, Kamilla Kara, Ábel Nagy, Balázs Pejó, Gergely Biczók

**Abstract**: The long-term success of cryptocurrencies largely depends on the incentive compatibility provided to the validators. Bribery attacks, facilitated trustlessly via smart contracts, threaten this foundation. This work introduces, implements, and evaluates three novel and efficient bribery contracts targeting Ethereum validators. The first bribery contract enables a briber to fork the blockchain by buying votes on their proposed blocks. The second contract incentivizes validators to voluntarily exit the consensus protocol, thus increasing the adversary's relative staking power. The third contract builds a trustless bribery market that enables the briber to auction off their manipulative power over the RANDAO, Ethereum's distributed randomness beacon. Finally, we provide an initial game-theoretical analysis of one of the described bribery markets.

摘要: 加密货币的长期成功很大程度上取决于为验证者提供的激励兼容性。通过智能合同不受信任地促进的贿赂攻击威胁着这一基础。这项工作介绍、实施和评估了三个针对以太坊验证者的新颖且高效的贿赂合同。第一份贿赂合同使贿赂者能够通过购买其提议区块的选票来分叉区块链。第二个合同激励验证者自愿退出共识协议，从而增加对手的相对赌注权力。第三个合同建立了一个不可信任的贿赂市场，使贿赂者能够拍卖他们对以太坊的分布式随机性信标RANDO的操纵权力。最后，我们对所描述的贿赂市场之一进行了初步的博弈论分析。



## **39. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

模拟集群攻击：通过微调的视觉语言模型转移越狱 cs.CV

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2508.01741v3) [paper-pdf](https://arxiv.org/pdf/2508.01741v3)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Juncheng Li, Xuan Tong, Xingjun Ma

**Abstract**: The widespread practice of fine-tuning open-source Vision-Language Models (VLMs) raises a critical security concern: jailbreak vulnerabilities in base models may persist in downstream variants, enabling transferable attacks across fine-tuned systems. To investigate this risk, we propose the Simulated Ensemble Attack (SEA), a grey-box jailbreak framework that assumes full access to the base VLM but no knowledge of the fine-tuned target. SEA enhances transferability via Fine-tuning Trajectory Simulation (FTS), which models bounded parameter variations in the vision encoder, and Targeted Prompt Guidance (TPG), which stabilizes adversarial optimization through auxiliary textual guidance. Experiments on the Qwen2-VL family demonstrate that SEA achieves consistently high transfer success and toxicity rates across diverse fine-tuned variants, including safety-enhanced models, while standard PGD-based image jailbreaks exhibit negligible transferability. Further analysis reveals that fine-tuning primarily induces localized parameter shifts around the base model, explaining why attacks optimized over a simulated neighborhood transfer effectively. We also show that SEA generalizes across different base generations (e.g., Qwen2.5/3-VL), indicating that its effectiveness arises from shared fine-tuning-induced behaviors rather than architecture- or initialization-specific factors.

摘要: 微调开源视觉语言模型（VLM）的广泛实践引发了一个关键的安全问题：基本模型中的越狱漏洞可能会在下游变体中持续存在，从而导致跨微调系统的可转移攻击。为了调查这种风险，我们提出了模拟集群攻击（SEA），这是一种灰箱越狱框架，假设可以完全访问基本VLM，但不知道微调目标。SEA通过微调轨迹模拟（FTS）和定向提示引导（TPG）增强了可移植性，FTS对视觉编码器中的有界参数变化进行建模，TPG通过辅助文本引导稳定对抗优化。Qwen 2-BL系列的实验表明，SEA在各种微调变体（包括安全增强模型）中实现了一致的高转移成功率和毒性率，而标准的基于PGD的图像越狱表现出可忽略的可转移性。进一步的分析表明，微调主要会导致基本模型周围的局部参数变化，这解释了为什么攻击在模拟的邻居转移上进行有效优化。我们还表明SEA在不同的基代之间进行了推广（例如，Qwen 2.5/3-DL），表明其有效性源于共同的微调引发的行为，而不是架构或初始化特定因素。



## **40. BeDKD: Backdoor Defense Based on Directional Mapping Module and Adversarial Knowledge Distillation**

BeDKD：基于方向映射模块和对抗性知识提炼的后门防御 cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2508.01595v3) [paper-pdf](https://arxiv.org/pdf/2508.01595v3)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Yinghan Zhou, Changtong dou, Yiming Xue

**Abstract**: Although existing backdoor defenses have gained success in mitigating backdoor attacks, they still face substantial challenges. In particular, most of them rely on large amounts of clean data to weaken the backdoor mapping but generally struggle with residual trigger effects, resulting in persistently high attack success rates (ASR). Therefore, in this paper, we propose a novel \textbf{B}ackdoor d\textbf{e}fense method based on \textbf{D}irectional mapping module and adversarial \textbf{K}nowledge \textbf{D}istillation (BeDKD), which balances the trade-off between defense effectiveness and model performance using a small amount of clean and poisoned data. We first introduce a directional mapping module to identify poisoned data, which destroys clean mapping while keeping backdoor mapping on a small set of flipped clean data. Then, the adversarial knowledge distillation is designed to reinforce clean mapping and suppress backdoor mapping through a cycle iteration mechanism between trust and punish distillations using clean and identified poisoned data. We conduct experiments to mitigate mainstream attacks on three datasets, and experimental results demonstrate that BeDKD surpasses the state-of-the-art defenses and reduces the ASR by 98$\%$ without significantly reducing the CACC. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD.

摘要: 尽管现有的后门防御措施在缓解后门攻击方面取得了成功，但它们仍然面临着巨大的挑战。特别是，它们中的大多数依赖大量干净的数据来削弱后门映射，但通常会与残余触发效应作斗争，从而导致攻击成功率（ASB）持续很高。因此，本文提出了一种新型的\textBF{B}ackdoor d\textBF{e}fense方法，该方法基于\textBF{D}定向映射模块和对抗性\textBF{K} nosophy\textBF{D}istillation（BeDKD），该方法使用少量干净和有毒的数据来平衡防御有效性和模型性能之间的权衡。我们首先引入一个方向性映射模块来识别有毒数据，这会破坏干净映射，同时在一小群翻转干净数据上保留后门映射。然后，对抗性知识蒸馏旨在通过信任和惩罚蒸馏之间的循环迭代机制来加强干净映射并抑制后门映射，使用干净和已识别的有毒数据。我们在三个数据集上进行了缓解主流攻击的实验，实验结果表明，BeDKD超越了最先进的防御，并在不显着降低CACC的情况下将ASR降低了98%。我们的代码可在https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/BeDKD上获取。



## **41. Random Walk Learning and the Pac-Man Attack**

随机步行学习和吃豆人攻击 stat.ML

The updated manuscript represents an incomplete version of the work. A substantially updated version will be prepared before further dissemination

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2508.05663v3) [paper-pdf](https://arxiv.org/pdf/2508.05663v3)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Zonghong Liu, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the Average Crossing (AC) algorithm--a fully decentralized mechanism for duplicating RWs to prevent RW extinction in the presence of Pac-Man. Our theoretical analysis establishes that (i) the RW population remains almost surely bounded under AC and (ii) RW-based stochastic gradient descent remains convergent under AC, even in the presence of Pac-Man, with a quantifiable deviation from the true optimum. Our extensive empirical results on both synthetic and real-world datasets corroborate our theoretical findings. Furthermore, they uncover a phase transition in the extinction probability as a function of the duplication threshold. We offer theoretical insights by analyzing a simplified variant of the AC, which sheds light on the observed phase transition.

摘要: 由于管理费用低和可扩展性，基于随机游走（RW）的算法长期以来一直在分布式系统中流行，最近在去中心化学习中的应用越来越多。然而，它们对本地交互的依赖使它们本质上容易受到恶意行为的影响。在这项工作中，我们调查了一种对抗性威胁，我们称之为“吃豆人”攻击，其中恶意节点概率地终止访问它的任何RW。这种隐形行为逐渐从网络中消除活动RW，有效地停止学习过程，而不会触发失败警报。为了应对这一威胁，我们提出了平均交叉（AC）算法-一个完全分散的机制，用于复制RW，以防止RW灭绝的存在吃豆人。我们的理论分析建立，（i）RW人口几乎肯定有界下AC和（ii）基于RW的随机梯度下降仍然收敛下AC，即使在吃豆人的存在，与真正的最佳值存在可量化的偏差。我们对合成和现实世界数据集的广泛经验结果证实了我们的理论发现。此外，它们还揭示了灭绝概率的相转变作为复制阈值的函数。我们通过分析AC的简化变体来提供理论见解，该变体揭示了观察到的相转变。



## **42. Exploring the Secondary Risks of Large Language Models**

探索大型语言模型的次要风险 cs.LG

18 pages, 5 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2506.12382v4) [paper-pdf](https://arxiv.org/pdf/2506.12382v4)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.

摘要: 随着大型语言模型越来越多地集成到关键应用程序和社会功能中，确保大型语言模型的安全性和一致性是一项重大挑战。虽然之前的研究主要集中在越狱攻击上，但对良性互动中微妙出现的非对抗性失败的关注较少。我们引入了二级风险，这是一种新型的失败模式，其特征是良性提示期间的有害或误导行为。与对抗性攻击不同，这些风险源于不完美的概括，并且常常逃避标准安全机制。为了实现系统性评估，我们引入了两个风险基元：详细响应和推测性建议，以捕捉核心故障模式。在这些定义的基础上，我们提出了SecLens，这是一个黑匣子、多目标搜索框架，通过优化任务相关性、风险激活和语言合理性来有效地引发次要风险行为。为了支持可重复的评估，我们发布了SecRiskBench，这是一个由650个提示组成的基准数据集，涵盖八个不同的现实世界风险类别。对16种流行模型进行广泛评估的实验结果表明，次级风险是普遍存在的，可以跨模型转移，并且独立于模式，这强调了迫切需要增强的安全机制来解决现实世界部署中良性但有害的LLM行为。



## **43. Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings**

加速低查询黑匣子设置中的有针对性的硬标签对抗攻击 cs.CV

This paper contains 10 pages, 8 figures and 8 tables. For associated supplementary code, see https://github.com/mdppml/TEA. This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2505.16313v3) [paper-pdf](https://arxiv.org/pdf/2505.16313v3)

**Authors**: Arjhun Swaminathan, Mete Akgün

**Abstract**: Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.

摘要: 用于图像分类的深度神经网络仍然容易受到对抗性示例的影响--这些小而难以察觉的扰动会导致错误分类。在黑匣子环境中，只有最终预测才能访问，由于决策区域狭窄，精心设计旨在错误分类到特定目标类别的有针对性的攻击尤其具有挑战性。当前最先进的方法通常利用分离源图像和目标图像的决策边界的几何属性，而不是合并来自图像本身的信息。相比之下，我们提出了目标边缘信息攻击（TEA），这是一种新型攻击，利用目标图像的边缘信息仔细扰动它，从而产生更接近源图像的对抗图像，同时仍然实现所需的目标分类。在低查询设置（使用的查询减少了近70%）下，我们的方法在不同模型中始终优于当前最先进的方法，这种情况在查询和黑匣子访问有限的现实世界应用程序中尤其相关。此外，通过有效生成合适的对抗示例，TEA为已建立的基于几何的攻击提供了改进的目标初始化。



## **44. A Cross-Layer Analysis of Network Antifragility with RIS-assisted Links under Jamming Attacks**

干扰攻击下RIS辅助链接的网络反脆弱性跨层分析 cs.NI

This paper is uploaded here for research community, thus it is for non-commercial purposes

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2505.02565v2) [paper-pdf](https://arxiv.org/pdf/2505.02565v2)

**Authors**: Mounir Bensalem, Thomas Röthig, Admela Jukan

**Abstract**: Antifragility is an economics term defined as measure of (monetary) benefits gained from the adverse events and variability of the markets. This paper integrates for the first time the antifragility into the network based on communication links with Reconfigurable Intelligent Surface (RIS) affected by a jamming attack. We analyze whether antifragility can be achieved for several jamming models. Beyond the link-level gains, the results reveal how antifragile RIS-assisted links can be integrated into multi-hop systems to improve end-to-end network resilience, connectivity, and throughput under adversarial effects.

摘要: 反脆弱性是一个经济学术语，定义为从不利事件和市场变化中获得的（货币）利益的衡量标准。本文首次将反脆弱性集成到基于受干扰攻击影响的可重构智能表面（RIS）的通信链路的网络中。我们分析了几种干扰模型是否可以实现反脆弱性。除了链路级的收益之外，结果还揭示了如何将抗脆弱的RIS辅助链路集成到多跳系统中，以在对抗影响下提高端到端网络弹性、连接性和吞吐量。



## **45. BadPatches: Routing-aware Backdoor Attacks on Vision Mixture of Experts**

BadPatches：对视觉混合专家的广告感知后门攻击 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2505.01811v3) [paper-pdf](https://arxiv.org/pdf/2505.01811v3)

**Authors**: Cedric Chan, Jona te Lintelo, Stjepan Picek

**Abstract**: Mixture of Experts (MoE) architectures have gained popularity for reducing computational costs in deep neural networks by activating only a subset of parameters during inference. While this efficiency makes MoE attractive for vision tasks, the patch-based processing in vision models introduces new methods for adversaries to perform backdoor attacks. In this work, we investigate the vulnerability of vision MoE models for image classification, specifically the patch-based MoE (pMoE) models and MoE-based vision transformers, against backdoor attacks. We propose a novel routing-aware trigger application method BadPatches, which is designed for patch-based processing in vision MoE models. BadPatches applies triggers on image patches rather than on the entire image. We show that BadPatches achieves high attack success rates (ASRs) with lower poisoning rates than routing-agnostic triggers and is successful at poisoning rates as low as 0.01% with an ASR above 80% on pMoE. Moreover, BadPatches is still effective when an adversary does not have complete knowledge of the patch routing configuration of the considered models. Next, we explore how trigger design affects pMoE patch routing. Finally, we investigate fine-pruning as a defense. Results show that only the fine-tuning stage of fine-pruning removes the backdoor from the model.

摘要: 混合专家（MoE）架构因通过在推理期间仅激活参数子集来降低深度神经网络中的计算成本而受到欢迎。虽然这种效率使得MoE对视觉任务具有吸引力，但视觉模型中基于补丁的处理为对手执行后门攻击引入了新方法。在这项工作中，我们研究了用于图像分类的视觉MoE模型，特别是基于补丁的MoE（pMoE）模型和基于MoE的视觉转换器，对抗后门攻击的脆弱性。我们提出了一种新型的路由感知触发应用方法BadPatches，该方法专为视觉MoE模型中基于补丁的处理而设计。BadPatches将触发器应用于图像补丁而不是整个图像。我们表明，BadPatches实现了高攻击成功率（ASB），中毒率低于路径不可知触发器，并且在pMoE上的中毒率低至0.01%时成功，ASO高于80%。此外，当对手不完全了解所考虑模型的补丁路由配置时，BadPatches仍然有效。接下来，我们探讨触发器设计如何影响pMoE补丁路由。最后，我们研究了精细修剪作为一种防御。结果表明，只有微调阶段的微调修剪删除后门模型。



## **46. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

GreedyPixel：通过贪婪算法进行细粒度黑匣子对抗攻击 cs.CV

IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2501.14230v4) [paper-pdf](https://arxiv.org/pdf/2501.14230v4)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: Deep neural networks are highly vulnerable to adversarial examples, which are inputs with small, carefully crafted perturbations that cause misclassification -- making adversarial attacks a critical tool for evaluating robustness. Existing black-box methods typically entail a trade-off between precision and flexibility: pixel-sparse attacks (e.g., single- or few-pixel attacks) provide fine-grained control but lack adaptability, whereas patch- or frequency-based attacks improve efficiency or transferability, but at the cost of producing larger and less precise perturbations. We present GreedyPixel, a fine-grained black-box attack method that performs brute-force-style, per-pixel greedy optimization guided by a surrogate-derived priority map and refined by means of query feedback. It evaluates each coordinate directly without any gradient information, guaranteeing monotonic loss reduction and convergence to a coordinate-wise optimum, while also yielding near white-box-level precision and pixel-wise sparsity and perceptual quality. On the CIFAR-10 and ImageNet datasets, spanning convolutional neural networks (CNNs) and Transformer models, GreedyPixel achieved state-of-the-art success rates with visually imperceptible perturbations, effectively bridging the gap between black-box practicality and white-box performance. The implementation is available at https://github.com/azrealwang/greedypixel.

摘要: 深度神经网络非常容易受到对抗性示例的影响，这些示例是带有精心设计的微小扰动的输入，会导致错误分类--这使得对抗性攻击成为评估稳健性的重要工具。现有的黑匣子方法通常需要精确性和灵活性之间的权衡：像素稀疏攻击（例如，单像素或少像素攻击）提供细粒度控制，但缺乏适应性，而基于补丁或频率的攻击可以提高效率或可转移性，但代价是产生更大和更不精确的扰动。我们提出了GreedyPixel，这是一种细粒度的黑匣子攻击方法，它在代理人派生的优先级地图的指导下执行暴力风格的每像素贪婪优化，并通过查询反馈进行改进。它在没有任何梯度信息的情况下直接评估每个坐标，保证单调损失减少并收敛到坐标最优值，同时还能产生接近白盒级别的精度和像素稀疏度和感知质量。在CIFAR-10和ImageNet数据集上，跨越卷积神经网络（CNN）和Transformer模型，GreedyPixel通过视觉上难以感知的扰动实现了最先进的成功率，有效地弥合了黑匣子实用性和白盒性能之间的差距。该实现可在https://github.com/azrealwang/greedypixel上获取。



## **47. Adversarial Multi-Agent Reinforcement Learning for Proactive False Data Injection Detection**

用于主动错误数据注入检测的对抗性多智能体强化学习 eess.SY

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2411.12130v2) [paper-pdf](https://arxiv.org/pdf/2411.12130v2)

**Authors**: Kejun Chen, Truc Nguyen, Abhijeet Sahu, Malik Hassanaly

**Abstract**: Smart inverters are instrumental in the integration of distributed energy resources into the electric grid. Such inverters rely on communication layers for continuous control and monitoring, potentially exposing them to cyber-physical attacks such as false data injection attacks (FDIAs). We propose to construct a defense strategy against a priori unknown FDIAs with a multi-agent reinforcement learning (MARL) framework. The first agent is an adversary that simulates and discovers various FDIA strategies, while the second agent is a defender in charge of detecting and locating FDIAs. This approach enables the defender to be trained against new FDIAs continuously generated by the adversary. In addition, we show that the detection skills of an MARL defender can be combined with those of a supervised offline defender through a transfer learning approach. Numerical experiments conducted on a distribution and transmission system demonstrate that: a) the proposed MARL defender outperforms the offline defender against adversarial attacks; b) the transfer learning approach makes the MARL defender capable against both synthetic and unseen FDIAs.

摘要: 智能逆变器对于将分布式能源集成到电网中至关重要。此类逆变器依赖通信层进行持续控制和监控，可能会使它们面临虚假数据注入攻击（FDIA）等网络物理攻击。我们建议通过多智能体强化学习（MARL）框架构建针对先验未知FDIA的防御策略。第一个代理是模拟和发现各种FDIA策略的对手，而第二个代理是负责检测和定位FDIA的防御者。这种方法使防御者能够针对对手不断产生的新FDIA进行训练。此外，我们还表明，MARL防御者的检测技能可以通过迁移学习方法与受监督的离线防御者的检测技能相结合。在分发和传输系统上进行的数值实验表明：a）提出的MARL防御器在对抗性攻击方面优于离线防御器; b）迁移学习方法使MARL防御器能够对抗合成和不可见的FDIA。



## **48. Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**

通过最大化预期平坦度通过低成本优化来提高对抗性可转让性 cs.CV

Accepted by IEEE T-IFS

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2405.16181v3) [paper-pdf](https://arxiv.org/pdf/2405.16181v3)

**Authors**: Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.

摘要: 基于传输的攻击在白盒代理模型上制作对抗性示例，并直接将其部署到黑盒目标模型上，提供模型不可知和无查询的威胁场景。虽然最近出现了平坦性增强方法，通过增强对抗性示例的损失表面平坦性来提高可转移性，但它们不同的平坦性定义和启发式攻击设计受到未经检查的优化限制和缺乏理论基础的影响，从而限制了它们的有效性和效率。这项工作揭示了严重不平衡的开发-探索动态平坦度优化，建立了第一个基于平坦度的可转移性的理论基础，并提出了一个原则性的框架，以克服这些优化陷阱。具体来说，我们系统地统一了现有方法中的碎片化平坦度定义，揭示了它们在过度探索灵敏度峰值或过度利用局部高原方面的不平衡优化局限性。为了解决这些问题，我们严格形式化了平均情况平坦度和可转移性差距，证明增强零阶平均情况平坦度可以最大限度地减少跨模型差异。在此理论的基础上，我们设计了一种最大期望平坦度（MEF）攻击，该攻击可以增强零阶平均情况平坦度，同时平衡平坦度探索和利用。对22个模型和24种当前基于传输的攻击进行了广泛评估，证明了MEF的优势：它的攻击成功率比最先进的PGN攻击高出4%，计算成本仅为一半，并在相同预算下实现了8%的成功率。与输入增强相结合时，MEF相对于配备防御装备的型号获得了15%的额外收益，从而建立了新的稳健性基准。我们的代码可在https://github.com/SignedQiu/MEFAttack上获取。



## **49. A New Formulation for Zeroth-Order Optimization of Adversarial EXEmples in Malware Detection**

恶意软件检测中对抗实例零阶优化的新公式 cs.LG

17 pages, 6 tables

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2405.14519v2) [paper-pdf](https://arxiv.org/pdf/2405.14519v2)

**Authors**: Marco Rando, Luca Demetrio, Lorenzo Rosasco, Fabio Roli

**Abstract**: Machine learning malware detectors are vulnerable to adversarial EXEmples, i.e., carefully-crafted Windows programs tailored to evade detection. Unlike other adversarial problems, attacks in this context must be functionality-preserving, a constraint that is challenging to address. As a consequence, heuristic algorithms are typically used, which inject new content, either randomly-picked or harvested from legitimate programs. In this paper, we show how learning malware detectors can be cast within a zeroth-order optimization framework, which allows incorporating functionality-preserving manipulations. This permits the deployment of sound and efficient gradient-free optimization algorithms, which come with theoretical guarantees and allow for minimal hyper-parameters tuning. As a by-product, we propose and study ZEXE, a novel zeroth-order attack against Windows malware detection. Compared to state-of-the-art techniques, ZEXE provides improvement in the evasion rate, reducing to less than one third the size of the injected content.

摘要: 机器学习恶意软件检测器容易受到对抗性实例的影响，即精心制作的Windows程序，专为逃避检测而定制。与其他对抗性问题不同，这种情况下的攻击必须保持功能，这是一个难以解决的限制。因此，通常使用启发式算法，这些算法注入新内容，无论是随机挑选的还是从合法程序中获取的。在本文中，我们展示了如何将学习恶意软件检测器投射到零阶优化框架中，该框架允许合并功能保留的操纵。这允许部署健全且高效的无梯度优化算法，这些算法有理论保证，并允许最少的超参数调整。作为副产品，我们提出并研究了ZXE，这是一种针对Windows恶意软件检测的新型零阶攻击。与最先进的技术相比，ZXE提高了规避率，将注入内容的大小减少到不到三分之一。



## **50. Visual Adversarial Attacks and Defenses in the Physical World: A Survey**

物理世界中的视觉对抗攻击和防御：调查 cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2211.01671v6) [paper-pdf](https://arxiv.org/pdf/2211.01671v6)

**Authors**: Xingxing Wei, Bangzheng Pu, Shiji Zhao, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they remain vulnerable to adversarial examples. Adversarial attacks in computer vision can be categorized into digital attacks and physical attacks based on their different forms. Compared to digital attacks, which generate perturbations in digital pixels, physical attacks are more practical in real-world settings. Due to the serious security risks posed by physically adversarial examples, many studies have been conducted to evaluate the physically adversarial robustness of DNNs in recent years. In this paper, we provide a comprehensive survey of current physically adversarial attacks and defenses in computer vision. We establish a taxonomy by organizing physical attacks according to attack tasks, attack forms, and attack methods. This approach offers readers a systematic understanding of the topic from multiple perspectives. For physical defenses, we categorize them into pre-processing, in-processing, and post-processing for DNN models to ensure comprehensive coverage of adversarial defenses. Based on this survey, we discuss the challenges facing this research field and provide an outlook on future directions.

摘要: 尽管深度神经网络（DNN）已广泛应用于各种现实世界场景，但它们仍然容易受到对抗性示例的影响。计算机视觉中的对抗性攻击根据其形式的不同可以分为数字攻击和物理攻击。与在数字像素中产生扰动的数字攻击相比，物理攻击在现实世界环境中更实用。由于物理对抗示例带来了严重的安全风险，近年来人们进行了许多研究来评估DNN的物理对抗稳健性。在本文中，我们对当前计算机视觉中的物理对抗攻击和防御进行了全面调查。我们通过根据攻击任务、攻击形式和攻击方法组织物理攻击来建立分类。这种方法使读者从多个角度对该主题有一个系统的了解。对于物理防御，我们将其分为DNN模型的预处理、处理中和后处理，以确保对抗性防御的全面覆盖。在此基础上，我们讨论了这一研究领域面临的挑战，并对未来的发展方向进行了展望。



