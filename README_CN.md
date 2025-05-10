# Latest Adversarial Attack Papers
**update at 2025-05-10 15:08:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2501.19040v2) [paper-pdf](http://arxiv.org/pdf/2501.19040v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型容易受到对抗攻击，对手会精心设计特定的输入序列来引发有害、暴力、私密或错误的输出。在这项工作中，我们研究了它们的最坏情况稳健性，即是否存在导致此类不良结果的对抗性例子。我们使用更强的白盒攻击来对最坏情况的稳健性进行上限，这表明当前大多数确定性防御实现了近0%的最坏情况的稳健性。我们提出了使用分数背包求解器或0-1背包求解器的随机平滑的一般紧下界，并使用它们来限制所有随机防御的最坏情况稳健性。基于这些求解器，我们为之前的几个经验防御提供了理论下限。例如，我们证明了特定情况的稳健性，使用统一核进行平滑，针对\texttit {任何可能的攻击}，平均$\ell_0 $扰动为2.02或平均后缀长度为6.41。



## **2. SUUM: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

SUUM：基于时间戳的中本风格区块链很脆弱 cs.CR

27 pages, 6 figures

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05328v1) [paper-pdf](http://arxiv.org/pdf/2505.05328v1)

**Authors**: Junjie Hu, Na Ruan

**Abstract**: We introduce two advanced attack strategies, the Unrestricted Uncle Maker (UUM) Attack and the Staircase-Unrestricted Uncle Maker (SUUM) Attack, which fundamentally threaten the security of timestamp-based Nakamoto-style blockchains by inflicting permanent systemic harm. Unlike prior work that merely enhances adversarial rewards, these attacks exploit vulnerabilities in timestamp manipulation and fork selection rules to irreversibly destabilize blockchain fairness and incentive mechanisms. Specifically, the SUUM attack enables adversaries to persistently launch attacks at zero cost, eliminating constraints on block withholding and risk-free conditions, while systematically maximizing rewards through coordinated timestamp adjustments and strategic block release.   Our analysis demonstrates that SUUM adversaries achieve disproportionate reward advantages over both UUM and the original Riskless Uncle Maker (RUM) Attack [CCS '23], with all three strategies surpassing honest mining. Crucially, SUUM's cost-free persistence allows adversaries to indefinitely drain rewards from honest participants by maintaining minimal difficulty risks through precise timestamp manipulation. This creates a self-reinforcing cycle: adversaries amplify their profits while suppressing honest returns, thereby permanently eroding the protocol's security assumptions. Through rigorous theoretical modeling and simulations, we validate how SUUM's combination of timestamp tampering, block withholding, and difficulty risk control enables unmitigated exploitation of consensus mechanisms. This work underscores the existential risks posed by timestamp-based Nakamoto-style protocols and advocates urgent countermeasures to ensure long-term stability.

摘要: 我们引入了两种高级攻击策略，无限制Uncle Maker（UUM）攻击和Staircase-Uncle Maker（SUUM）攻击，它们通过造成永久性系统性伤害，从根本上威胁基于时间戳的Nakamoto风格区块链的安全性。与之前仅增强对抗奖励的工作不同，这些攻击利用时间戳操纵和叉选择规则中的漏洞来不可逆转地破坏区块链公平性和激励机制的稳定。具体来说，SUUM攻击使对手能够持续以零成本发起攻击，消除对区块扣留和无风险条件的限制，同时通过协调的时间戳调整和战略区块释放系统性地最大化回报。   我们的分析表明，SUUM对手比UUM和最初的无风险Uncle Maker（RUM）Attack [CS ' 23]获得了不成比例的奖励优势，这三种策略都超过了诚实采矿。至关重要的是，SUUM的无成本持久性允许对手通过精确的时间戳操纵将困难风险保持在最低限度，从而无限期地消耗诚实参与者的回报。这造成了一个自我强化的循环：对手放大了他们的利润，同时压制了诚实的回报，从而永久地侵蚀了协议的安全假设。通过严格的理论建模和模拟，我们验证了SUUM的时间戳篡改、块扣留和难度风险控制的组合如何实现对共识机制的全面利用。这项工作强调了基于时间戳的中本式协议所带来的生存风险，并倡导采取紧急应对措施以确保长期稳定。



## **3. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

大型语言模型中的漏洞越狱和缓解 cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.15236v2) [paper-pdf](http://arxiv.org/pdf/2410.15236v2)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.

摘要: 大型语言模型（LLM）通过推进自然语言理解和生成，改变了人工智能，实现了医疗保健、软件工程和会话系统以外的应用。尽管过去几年取得了这些进步，但LLM仍表现出相当大的漏洞，特别是在引发注射和越狱攻击方面。本评论分析了这些漏洞的研究状况，并提出了可用的防御策略。我们大致将攻击方法分为基于模型的，基于模型的，多模式的和多语言的，涵盖了对抗性提示，后门注入和跨模式利用等技术。我们还回顾了各种防御机制，包括即时过滤、转换、对齐技术、多智能体防御和自我调节，评估它们的优点和缺点。我们还讨论了用于评估LLM安全性和稳健性的关键指标和基准，并指出了交互式环境中攻击成功的量化以及现有数据集中的偏差等挑战。通过识别当前的研究差距，我们提出了弹性对齐策略、针对不断发展的攻击的先进防御、越狱检测自动化以及道德和社会影响的未来方向。该审查强调了人工智能社区内持续研究与合作的必要性，以增强LLM安全性并确保其安全部署。



## **4. PointBA: Towards Backdoor Attacks in 3D Point Cloud**

PointBA：3D点云中的后门攻击 cs.LG

Accepted by ICCV 2021

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2103.16074v4) [paper-pdf](http://arxiv.org/pdf/2103.16074v4)

**Authors**: Xinke Li, Zhirui Chen, Yue Zhao, Zekun Tong, Yabang Zhao, Andrew Lim, Joey Tianyi Zhou

**Abstract**: 3D deep learning has been increasingly more popular for a variety of tasks including many safety-critical applications. However, recently several works raise the security issues of 3D deep models. Although most of them consider adversarial attacks, we identify that backdoor attack is indeed a more serious threat to 3D deep learning systems but remains unexplored. We present the backdoor attacks in 3D point cloud with a unified framework that exploits the unique properties of 3D data and networks. In particular, we design two attack approaches on point cloud: the poison-label backdoor attack (PointPBA) and the clean-label backdoor attack (PointCBA). The first one is straightforward and effective in practice, while the latter is more sophisticated assuming there are certain data inspections. The attack algorithms are mainly motivated and developed by 1) the recent discovery of 3D adversarial samples suggesting the vulnerability of deep models under spatial transformation; 2) the proposed feature disentanglement technique that manipulates the feature of the data through optimization methods and its potential to embed a new task. Extensive experiments show the efficacy of the PointPBA with over 95% success rate across various 3D datasets and models, and the more stealthy PointCBA with around 50% success rate. Our proposed backdoor attack in 3D point cloud is expected to perform as a baseline for improving the robustness of 3D deep models.

摘要: 3D深度学习对于各种任务（包括许多安全关键应用）越来越受欢迎。然而，最近有几部作品提出了3D深度模型的安全问题。尽管他们中的大多数人都考虑了对抗性攻击，但我们发现后门攻击确实是对3D深度学习系统更严重的威胁，但仍然没有被探索。我们通过一个统一的框架来展示3D点云中的后门攻击，该框架利用了3D数据和网络的独特属性。特别是，我们在点云上设计了两种攻击方法：中毒标签后门攻击（PointSBA）和干净标签后门攻击（PointCBA）。第一种方法在实践中简单有效，而假设进行某些数据检查，后者则更加复杂。攻击算法的动机和开发主要来自1）最近发现的3D对抗样本，表明深度模型在空间转换下的脆弱性; 2）提出的特征解纠缠技术，通过优化方法操纵数据的特征及其嵌入新任务的潜力。大量实验表明PointSBA的功效，在各种3D数据集和模型中成功率超过95%，而更隐蔽的PointCBA的成功率约为50%。我们在3D点云中提出的后门攻击预计将作为提高3D深度模型稳健性的基线。



## **5. DispBench: Benchmarking Disparity Estimation to Synthetic Corruptions**

DispBench：将差异估计与合成腐蚀进行基准 cs.CV

Accepted at CVPR 2025 Workshop on Synthetic Data for Computer Vision

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05091v1) [paper-pdf](http://arxiv.org/pdf/2505.05091v1)

**Authors**: Shashank Agnihotri, Amaan Ansari, Annika Dackermann, Fabian Rösch, Margret Keuper

**Abstract**: Deep learning (DL) has surpassed human performance on standard benchmarks, driving its widespread adoption in computer vision tasks. One such task is disparity estimation, estimating the disparity between matching pixels in stereo image pairs, which is crucial for safety-critical applications like medical surgeries and autonomous navigation. However, DL-based disparity estimation methods are highly susceptible to distribution shifts and adversarial attacks, raising concerns about their reliability and generalization. Despite these concerns, a standardized benchmark for evaluating the robustness of disparity estimation methods remains absent, hindering progress in the field.   To address this gap, we introduce DispBench, a comprehensive benchmarking tool for systematically assessing the reliability of disparity estimation methods. DispBench evaluates robustness against synthetic image corruptions such as adversarial attacks and out-of-distribution shifts caused by 2D Common Corruptions across multiple datasets and diverse corruption scenarios. We conduct the most extensive performance and robustness analysis of disparity estimation methods to date, uncovering key correlations between accuracy, reliability, and generalization. Open-source code for DispBench: https://github.com/shashankskagnihotri/benchmarking_robustness/tree/disparity_estimation/final/disparity_estimation

摘要: 深度学习（DL）在标准基准上的表现已经超越了人类，推动了其在计算机视觉任务中的广泛采用。其中一项任务是差异估计，即估计立体图像对中匹配像素之间的差异，这对于医疗手术和自主导航等安全关键应用至关重要。然而，基于DL的差异估计方法极易受到分布变化和对抗攻击的影响，从而引发了对其可靠性和概括性的担忧。尽管存在这些担忧，但仍然缺乏用于评估差异估计方法稳健性的标准化基准，从而阻碍了该领域的进展。   为了解决这一差距，我们引入了DispBench，这是一种全面的基准测试工具，用于系统评估差异估计方法的可靠性。DispBench评估针对合成图像损坏的稳健性，例如多个数据集和不同损坏场景中的2D常见损坏引起的对抗性攻击和分发外转移。我们对差异估计方法进行了迄今为止最广泛的性能和稳健性分析，揭示了准确性、可靠性和概括性之间的关键相关性。DispBench的开源代码：https://github.com/shashankskagnihotri/benchmarking_robustness/tree/disparity_estimation/final/disparity_estimation



## **6. Integrating Communication, Sensing, and Security: Progress and Prospects of PLS in ISAC Systems**

集成通信、传感和安全：ISAC系统中最大化（SCS）的进展和前景 cs.ET

IEEE COMST

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05090v1) [paper-pdf](http://arxiv.org/pdf/2505.05090v1)

**Authors**: Waqas Aman, El-Mehdi Illi, Marwa Qaraqe, Saif Al-Kuwari

**Abstract**: The sixth generation of wireless networks defined several key performance indicators (KPIs) for assessing its networks, mainly in terms of reliability, coverage, and sensing. In this regard, remarkable attention has been paid recently to the integrated sensing and communication (ISAC) paradigm as an enabler for efficiently and jointly performing communication and sensing using the same spectrum and hardware resources. On the other hand, ensuring communication and data security has been an imperative requirement for wireless networks throughout their evolution. The physical-layer security (PLS) concept paved the way to catering to the security needs in wireless networks in a sustainable way while guaranteeing theoretically secure transmissions, independently of the computational capacity of adversaries. Therefore, it is of paramount importance to consider a balanced trade-off between communication reliability, sensing, and security in future networks, such as the 5G and beyond, and the 6G. In this paper, we provide a comprehensive and system-wise review of designed secure ISAC systems from a PLS point of view. In particular, the impact of various physical-layer techniques, schemes, and wireless technologies to ensure the sensing-security trade-off is studied from the surveyed work. Furthermore, the amalgamation of PLS and ISAC is analyzed in a broader impact by considering attacks targeting data confidentiality, communication covertness, and sensing spoofing. The paper also serves as a tutorial by presenting several theoretical foundations on ISAC and PLS, which represent a practical guide for readers to develop novel secure ISAC network designs.

摘要: 第六代无线网络定义了几个关键性能指标（KPI）来评估其网络，主要是可靠性、覆盖范围和感知方面。在这方面，集成传感和通信（ISAC）范式最近引起了人们的极大关注，作为使用相同的频谱和硬件资源高效、联合执行通信和传感的推动者。另一方面，确保通信和数据安全一直是无线网络在整个发展过程中的迫切要求。物理层安全（SCS）概念为以可持续的方式满足无线网络的安全需求铺平了道路，同时保证理论上的安全传输，独立于对手的计算能力。因此，在未来网络（例如5G及更高版本和6G）中考虑通信可靠性、感知和安全性之间的平衡至关重要。在本文中，我们从最大限度的角度对所设计的安全ISAC系统进行了全面的、系统性的审查。特别是，从调查的工作中研究了各种物理层技术、方案和无线技术对确保传感安全权衡的影响。此外，通过考虑针对数据机密性、通信隐蔽性和传感欺骗的攻击，分析了SCS和ISAC的合并在更广泛的影响中。本文还作为一个教程，介绍了几个理论基础ISAC和PLS，这是一个实用的指南，为读者开发新的安全ISAC网络设计。



## **7. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

可靠地限制假阳性：通过多尺度保形预测的零镜头机器生成文本检测框架 cs.CL

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05084v1) [paper-pdf](http://arxiv.org/pdf/2505.05084v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.

摘要: 大型语言模型的快速发展引发了人们对其潜在被恶意行为者滥用的严重担忧。因此，开发有效的探测器来减轻这些风险已成为当务之急。然而，大多数现有的检测方法过度关注检测准确性，往往忽视了高假阳性率（FPR）带来的社会风险。本文通过利用保形预测（CP）来解决这个问题，该预测有效地限制了FPR的上界。虽然直接应用CP约束FPR，但也会导致检测性能显着降低。为了克服这种权衡，本文提出了一种通过多尺度保形预测（LCP）的零镜头机器生成文本检测框架，该框架既强制执行FPR约束又提高检测性能。本文还介绍了RealDet，这是一个跨越广泛领域的高质量数据集，可确保真实的校准并在与HCP结合时实现卓越的检测性能。经验评估表明，LCP有效地约束了FPR，显着增强了检测性能，并增强了针对多个检测器和数据集的对抗攻击的鲁棒性。



## **8. Uncovering the Limitations of Model Inversion Evaluation -- Benchmarks and Connection to Type-I Adversarial Attacks**

揭示模型倒置评估的局限性--基准和与I型对抗攻击的联系 cs.LG

Our dataset and code are available in the Supp

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.03519v2) [paper-pdf](http://arxiv.org/pdf/2505.03519v2)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information of private training data by exploiting access to machine learning models. The most common evaluation framework for MI attacks/defenses relies on an evaluation model that has been utilized to assess progress across almost all MI attacks and defenses proposed in recent years. In this paper, for the first time, we present an in-depth study of MI evaluation. Firstly, we construct the first comprehensive human-annotated dataset of MI attack samples, based on 28 setups of different MI attacks, defenses, private and public datasets. Secondly, using our dataset, we examine the accuracy of the MI evaluation framework and reveal that it suffers from a significant number of false positives. These findings raise questions about the previously reported success rates of SOTA MI attacks. Thirdly, we analyze the causes of these false positives, design controlled experiments, and discover the surprising effect of Type I adversarial features on MI evaluation, as well as adversarial transferability, highlighting a relationship between two previously distinct research areas. Our findings suggest that the performance of SOTA MI attacks has been overestimated, with the actual privacy leakage being significantly less than previously reported. In conclusion, we highlight critical limitations in the widely used MI evaluation framework and present our methods to mitigate false positive rates. We remark that prior research has shown that Type I adversarial attacks are very challenging, with no existing solution. Therefore, we urge to consider human evaluation as a primary MI evaluation framework rather than merely a supplement as in previous MI research. We also encourage further work on developing more robust and reliable automatic evaluation frameworks.

摘要: 模型倒置（MI）攻击旨在通过利用对机器学习模型的访问来重建私人训练数据的信息。MI攻击/防御最常见的评估框架依赖于一个评估模型，该模型已用于评估近年来提出的几乎所有MI攻击和防御的进展。本文首次对MI评估进行了深入的研究。首先，我们基于28个不同MI攻击、防御、私有和公共数据集的设置，构建了第一个全面的MI攻击样本的人类注释数据集。其次，使用我们的数据集，我们检查了MI评估框架的准确性，并发现它存在大量的误报。这些发现对先前报告的SOTA MI攻击成功率提出了质疑。第三，我们分析了这些假阳性的原因，设计了对照实验，并发现了I型对抗性特征对MI评估的惊人影响，以及对抗性可转移性，突出了两个以前不同的研究领域之间的关系。我们的研究结果表明，SOTA MI攻击的性能被高估了，实际的隐私泄露明显低于以前的报告。总之，我们强调了广泛使用的MI评估框架的关键局限性，并介绍了我们降低假阳性率的方法。我们指出，之前的研究表明，I型对抗性攻击非常具有挑战性，目前还没有解决方案。因此，我们敦促将人类评估视为主要的MI评估框架，而不仅仅是之前的MI研究的补充。我们还鼓励进一步开发更强大、更可靠的自动评估框架。



## **9. Economic Security of Multiple Shared Security Protocols**

多个共享安全协议的经济安全 cs.CR

21 pages, 6 figures

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.03843v2) [paper-pdf](http://arxiv.org/pdf/2505.03843v2)

**Authors**: Abhimanyu Nag, Dhruv Bodani, Abhishek Kumar

**Abstract**: As restaking protocols gain adoption across blockchain ecosystems, there is a need for Actively Validated Services (AVSs) to span multiple Shared Security Providers (SSPs). This leads to stake fragmentation which introduces new complications where an adversary may compromise an AVS by targeting its weakest SSP. In this paper, we formalize the Multiple SSP Problem and analyze two architectures : an isolated fragmented model called Model $\mathbb{M}$ and a shared unified model called Model $\mathbb{S}$, through a convex optimization and game-theoretic lens. We derive utility bounds, attack cost conditions, and market equilibrium that describes protocol security for both models. Our results show that while Model $\mathbb{M}$ offers deployment flexibility, it inherits lowest-cost attack vulnerabilities, whereas Model $\mathbb{S}$ achieves tighter security guarantees through single validator sets and aggregated slashing logic. We conclude with future directions of work including an incentive-compatible stake rebalancing allocation in restaking ecosystems.

摘要: 随着重新质押协议在区块链生态系统中的采用，需要主动验证服务（AVS）跨越多个共享安全提供商（SSP）。这导致股权碎片化，这引入了新的复杂性，其中对手可能通过瞄准其最弱的SSP来损害AVS。在本文中，我们形式化的多SSP问题和分析两个架构：一个孤立的碎片模型称为模型$\mathbb{M}$和一个共享的统一模型称为模型$\mathbb{S}$，通过凸优化和博弈论的镜头。我们推导出效用界，攻击成本条件，和市场均衡，描述了这两种模式的协议安全性。我们的结果表明，虽然模型$\mathbb{M}$提供了部署灵活性，但它继承了成本最低的攻击漏洞，而模型$\mathbb{S}$通过单个验证器集和聚合削减逻辑实现了更严格的安全保证。我们总结了未来的工作方向，包括激励兼容的股权重新平衡重新押注生态系统的分配。



## **10. Memory Under Siege: A Comprehensive Survey of Side-Channel Attacks on Memory**

围攻下的记忆：对记忆的侧通道攻击的全面调查 cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.04896v1) [paper-pdf](http://arxiv.org/pdf/2505.04896v1)

**Authors**: MD Mahady Hassan, Shanto Roy, Reza Rahaeimehr

**Abstract**: Side-channel attacks on memory (SCAM) exploit unintended data leaks from memory subsystems to infer sensitive information, posing significant threats to system security. These attacks exploit vulnerabilities in memory access patterns, cache behaviors, and other microarchitectural features to bypass traditional security measures. The purpose of this research is to examine SCAM, classify various attack techniques, and evaluate existing defense mechanisms. It guides researchers and industry professionals in improving memory security and mitigating emerging threats. We begin by identifying the major vulnerabilities in the memory system that are frequently exploited in SCAM, such as cache timing, speculative execution, \textit{Rowhammer}, and other sophisticated approaches. Next, we outline a comprehensive taxonomy that systematically classifies these attacks based on their types, target systems, attack vectors, and adversarial capabilities required to execute them. In addition, we review the current landscape of mitigation strategies, emphasizing their strengths and limitations. This work aims to provide a comprehensive overview of memory-based side-channel attacks with the goal of providing significant insights for researchers and practitioners to better understand, detect, and mitigate SCAM risks.

摘要: 对内存的侧通道攻击（SCAM）利用内存子系统的意外数据泄露来推断敏感信息，对系统安全构成重大威胁。这些攻击利用内存访问模式、缓存行为和其他微体系结构功能中的漏洞来绕过传统的安全措施。本研究的目的是检查SCAM、对各种攻击技术进行分类并评估现有的防御机制。它指导研究人员和行业专业人士提高内存安全性并缓解新出现的威胁。我们首先识别SCAM中经常利用的内存系统中的主要漏洞，例如缓存计时、推测执行、\textit{Rowhammer}和其他复杂方法。接下来，我们概述了一个全面的分类法，该分类法根据攻击的类型、目标系统、攻击载体和执行攻击所需的对抗能力对这些攻击进行系统分类。此外，我们还回顾了当前的缓解策略格局，强调了它们的优势和局限性。这项工作旨在全面概述基于内存的侧通道攻击，旨在为研究人员和从业者提供重要见解，以更好地理解、检测和减轻SCAM风险。



## **11. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

正规的鲁棒可靠的学习者和实例有针对性的攻击 cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.10572v4) [paper-pdf](http://arxiv.org/pdf/2410.10572v4)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.

摘要: 针对实例的数据中毒攻击（对手破坏训练集以在特定测试点上引发错误）引发了严重担忧。Balcan等人（2022）提出了一种解决这一挑战的方法，通过定义鲁棒可靠学习器的概念，即使存在数据中毒攻击，也可以在明确定义的假设下提供每个实例的正确性保证。然后，他们给出了一个通用的最佳（但计算效率低下）鲁棒可靠的学习器，以及一个计算高效的算法，用于线性分离器在线性分离器的情况。   在这项工作中，我们解决了Balcan等人（2022）留下的两个挑战。首先，Balcan et al（2022）中对鲁棒可靠学习者的定义对于高度灵活的假设类别来说变得空洞：如果H中有两个分类器h_0、h_1 \，两者在训练集上的误差为零，使得h_0（x）\neq h_1（x），那么鲁棒可靠学习者必须放弃x。我们通过定义一个修改的正规化鲁棒可靠学习器概念来解决这个问题，该概念允许在这种情况下的非平凡陈述。其次，Balcan等人（2022）的通用算法需要在每个测试点x上重新运行ERM Oracle（本质上是重新训练分类器），即使可以有效地实施ERM，这通常也是不切实际的。为了解决这个问题，我们表明，至少在某些有趣的情况下，我们可以通过使用动态算法设计的技术来设计可以在训练时间内产生次线性输出的算法。



## **12. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

Red联手机器思维：LLM中即时注射和越狱漏洞的系统评估 cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04806v1) [paper-pdf](http://arxiv.org/pdf/2505.04806v1)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.

摘要: 大型语言模型（LLM）越来越多地集成到消费者和企业应用程序中。尽管它们有能力，但它们仍然容易受到对抗攻击，例如超越对齐保障措施的立即注射和越狱。本文对针对各种最先进的法学硕士的越狱策略进行了系统调查。我们对1，400多个对抗提示进行了分类，分析了它们对GPT-4、Claude 2、Mistral 7 B和Vicuna的成功，并检查它们的概括性和构造逻辑。我们进一步提出分层缓解策略，并推荐混合红色团队和沙箱方法以实现强大的LLM安全性。



## **13. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

以毒攻毒：通过奖励中和防御恶意RL微调 cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.

摘要: 强化学习（RL）微调改变了大型语言模型，同时创建了我们实验验证的漏洞：我们的实验表明，恶意RL微调以显着的效率突破了安全护栏，只需要50个步骤和最少的对抗提示，有害的升级从0-2升级到7-9。这种攻击载体特别威胁具有参数级访问权限的开源模型。事实证明，针对监督式微调的现有防御措施对RL的动态反馈机制无效。我们引入了奖励中和，这是第一个专门针对RL微调攻击而设计的防御框架，建立了简洁的拒绝模式，使恶意奖励信号无效。我们的方法训练模型以产生攻击者无法利用的最小信息拒绝，系统性地抵消针对有害输出进行优化的尝试。实验验证了我们的方法在200次攻击步骤后保持较低的有害分数（不大于2），而标准模型迅速恶化。这项工作提供了第一个建设性的证据，证明可以实现针对日益容易获得的RL攻击的强大防御，解决了开权模型的关键安全差距。



## **14. REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM**

REVEAL：Vision LLM图像输入危害的多回合评估 cs.CL

13 pages (8 main), to be published in IJCAI 2025

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04673v1) [paper-pdf](http://arxiv.org/pdf/2505.04673v1)

**Authors**: Madhur Jindal, Saurabh Deshpande

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.   We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$).

摘要: Vision大型语言模型（VLLM）将图像处理能力与文本理解集成，从而增强用户交互并扩展应用程序领域，代表了人工智能的重大进步。然而，它们日益增加的复杂性带来了新的安全和道德挑战，特别是在多模式和多回合对话中。传统的安全评估框架专为基于文本的单轮交互而设计，不足以解决这些复杂性。为了弥合这一差距，我们引入了REVEAL（视觉启用的AI LLM负责任评估）框架，这是一个可扩展和自动化的管道，用于评估VLLM中的图像输入伤害。REVEAL包括自动图像挖掘、合成对抗数据生成、使用渐强攻击策略的多轮对话扩展，以及通过GPT-4 o等评估器进行的全面危害评估。   我们广泛评估了五种最先进的VLLM：GPT-4 o、Llama-3.2、Qwen 2-BL、Phi3.5V和Pixtral，涵盖三个重要的伤害类别：性伤害、暴力和错误信息。我们的研究结果表明，与单轮评估相比，多轮交互会导致缺陷率显着更高，凸显了VLLM中更深层次的漏洞。值得注意的是，根据我们的安全可用性指数（SUI）衡量，GPT-4 o表现出最平衡的性能，紧随其后的是Pixtral。此外，错误信息已成为一个需要加强上下文防御的关键领域。Llama-3.2表现出最高的MT缺陷率（16.55美元），而Qwen 2-BL表现出最高的MT拒绝率（19.1美元）。



## **15. Mitigating Many-Shot Jailbreaking**

减轻多枪越狱 cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.09604v2) [paper-pdf](http://arxiv.org/pdf/2504.09604v2)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.

摘要: 多镜头越狱（MSJ）是一种对抗性技术，它利用现代LLM的长上下文窗口来规避模型安全培训，方法是在提示中包含许多“假”助理在最终请求之前做出不当反应的示例。有了足够多的例子，该模型的上下文学习能力就会凌驾于其安全培训之上，并且它的反应就好像它是“假”助手一样。在这项工作中，我们探讨了不同的微调和输入清理方法单独和组合在减轻MSJ攻击方面的有效性。我们发现每种技术的增量缓解效果，并表明组合技术显着降低了MSJ攻击的有效性，同时保留了良性上下文学习和对话任务中的模型性能。我们认为，如果将我们的方法纳入模型安全培训后，可以有意义地改善这种脆弱性。



## **16. Machine Learning Cryptanalysis of a Quantum Random Number Generator**

量子随机数发生器的机器学习密码分析 cs.LG

Published article is at https://ieeexplore.ieee.org/document/8396276.  Related code is at  https://github.com/Nano-Neuro-Research-Lab/Machine-Learning-Cryptanalysis-of-a-Quantum-Random-Number-Generator

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/1905.02342v3) [paper-pdf](http://arxiv.org/pdf/1905.02342v3)

**Authors**: Nhan Duy Truong, Jing Yan Haw, Syed Muhamad Assad, Ping Koy Lam, Omid Kavehei

**Abstract**: Random number generators (RNGs) that are crucial for cryptographic applications have been the subject of adversarial attacks. These attacks exploit environmental information to predict generated random numbers that are supposed to be truly random and unpredictable. Though quantum random number generators (QRNGs) are based on the intrinsic indeterministic nature of quantum properties, the presence of classical noise in the measurement process compromises the integrity of a QRNG. In this paper, we develop a predictive machine learning (ML) analysis to investigate the impact of deterministic classical noise in different stages of an optical continuous variable QRNG. Our ML model successfully detects inherent correlations when the deterministic noise sources are prominent. After appropriate filtering and randomness extraction processes are introduced, our QRNG system, in turn, demonstrates its robustness against ML. We further demonstrate the robustness of our ML approach by applying it to uniformly distributed random numbers from the QRNG and a congruential RNG. Hence, our result shows that ML has potentials in benchmarking the quality of RNG devices.

摘要: 对加密应用至关重要的随机数生成器（RNG）一直是对抗攻击的对象。这些攻击利用环境信息来预测生成的随机数，这些随机数应该是真正随机且不可预测的。尽管量子随机数发生器（QRNG）基于量子性质的固有不确定性，但测量过程中经典噪音的存在会损害QRNG的完整性。本文中，我们开发了一种预测机器学习（ML）分析，以研究确定性经典噪音在光学连续变量QRNG不同阶段的影响。当确定性噪音源突出时，我们的ML模型成功检测到固有相关性。引入适当的过滤和随机性提取过程后，我们的QRNG系统反过来又展示了其对ML的鲁棒性。我们通过将ML方法应用于来自QRNG和全合RNG的均匀分布随机数，进一步证明了ML方法的鲁棒性。因此，我们的结果表明ML在对RNG设备的质量进行基准测试方面具有潜力。



## **17. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks**

可靠的解纠缠多视图学习对抗视图对抗攻击 cs.LG

11 pages, 11 figures, accepted by International Joint Conference on  Artificial Intelligence (IJCAI 2025)

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04046v1) [paper-pdf](http://arxiv.org/pdf/2505.04046v1)

**Authors**: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

**Abstract**: Recently, trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. In practice, however, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view learning models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that our RDML outperforms the state-of-the-art multi-view learning methods by a relatively large margin.

摘要: 最近，值得信赖的多视图学习引起了广泛关注，因为证据学习可以提供可靠的不确定性估计，以增强多视图预测的可信度。现有的可信多视图学习方法隐含地假设多视图数据是安全的。然而，在实践中，在自动驾驶和安全监控等安全敏感应用中，多视图数据经常面临来自对抗扰动的威胁，从而欺骗或扰乱多视图学习模型。这不可避免地会导致可信多视图学习中的对抗不可靠性问题（AUP）。为了克服这个棘手的问题，我们提出了一种新颖的多视图学习框架，即可靠解纠缠多视图学习（RDML）。具体来说，我们首先提出证据解纠缠学习，在相应证据的指导下将每个视图分解为干净且对抗的部分，这些证据由预先训练的证据提取器提取。然后，我们使用特征重新校准模块来减轻对抗性扰动的负面影响，并从中提取潜在的信息特征。最后，为了进一步忽略不可挽回的对抗干扰，设计了视角级证据关注机制。针对具有对抗性攻击的多视图分类任务的大量实验表明，我们的RDML以相对较大的优势优于最先进的多视图学习方法。



## **18. MergeGuard: Efficient Thwarting of Trojan Attacks in Machine Learning Models**

MergeGuard：有效阻止机器学习模型中的特洛伊木马攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.04015v1) [paper-pdf](http://arxiv.org/pdf/2505.04015v1)

**Authors**: Soheil Zibakhsh Shabgahi, Yaman Jandali, Farinaz Koushanfar

**Abstract**: This paper proposes MergeGuard, a novel methodology for mitigation of AI Trojan attacks. Trojan attacks on AI models cause inputs embedded with triggers to be misclassified to an adversary's target class, posing a significant threat to model usability trained by an untrusted third party. The core of MergeGuard is a new post-training methodology for linearizing and merging fully connected layers which we show simultaneously improves model generalizability and performance. Our Proof of Concept evaluation on Transformer models demonstrates that MergeGuard maintains model accuracy while decreasing trojan attack success rate, outperforming commonly used (post-training) Trojan mitigation by fine-tuning methodologies.

摘要: 本文提出了MergeGuard，这是一种缓解人工智能特洛伊攻击的新型方法。对人工智能模型的特洛伊木马攻击导致嵌入触发器的输入被错误分类到对手的目标类别，对不受信任的第三方训练的模型可用性构成重大威胁。MergeGuard的核心是一种新的训练后方法，用于线性化和合并完全连接的层，我们证明它可以同时提高模型的可概括性和性能。我们对Transformer模型的概念验证评估表明，MergeGuard保持了模型的准确性，同时降低了特洛伊木马攻击的成功率，通过微调方法优于常用的（训练后）特洛伊木马缓解。



## **19. Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA**

采用QROA对LLM进行通用和黑匣子仅查询响应攻击 cs.CL

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2406.02044v3) [paper-pdf](http://arxiv.org/pdf/2406.02044v3)

**Authors**: Hussein Jawad, Yassine Chenik, Nicolas J. -B. Brunel

**Abstract**: The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA

摘要: 大型语言模型（LLM）的迅速采用暴露了关键的安全和道德漏洞，特别是它们容易受到对抗性操纵的影响。本文介绍了QROA，这是一种新型黑匣子越狱方法，旨在识别对抗性后缀，这些后缀在附加到恶意指令时可以绕过LLM对齐保障措施。与现有的基于后缀的越狱方法不同，QROA不需要访问模型的logit或任何其他内部信息。它还消除了对人工模板的依赖，仅通过LLM的标准查询-响应界面操作。通过将攻击定义为优化强盗问题，QROA采用代理模型和令牌级优化来有效地探索后缀变体。此外，我们还提出了QROA-UNV，这是一种扩展，可以为各个模型识别通用的对抗性后缀，从而实现跨广泛指令的单查询越狱。对多个模型的测试表明攻击成功率（ASB）大于80%。这些发现凸显了关键漏洞，强调了对先进防御的需要，并有助于开发更强大的安全评估以实现安全的人工智能部署。该代码在以下链接上公开：https://github.com/qroa/QROA



## **20. Model-Targeted Data Poisoning Attacks against ITS Applications with Provable Convergence**

可证明收敛的面向模型的ITS应用数据中毒攻击 math.OC

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03966v1) [paper-pdf](http://arxiv.org/pdf/2505.03966v1)

**Authors**: Xin Wanga, Feilong Wang, Yuan Hong, R. Tyrrell Rockafellar, Xuegang, Ban

**Abstract**: The growing reliance of intelligent systems on data makes the systems vulnerable to data poisoning attacks. Such attacks could compromise machine learning or deep learning models by disrupting the input data. Previous studies on data poisoning attacks are subject to specific assumptions, and limited attention is given to learning models with general (equality and inequality) constraints or lacking differentiability. Such learning models are common in practice, especially in Intelligent Transportation Systems (ITS) that involve physical or domain knowledge as specific model constraints. Motivated by ITS applications, this paper formulates a model-target data poisoning attack as a bi-level optimization problem with a constrained lower-level problem, aiming to induce the model solution toward a target solution specified by the adversary by modifying the training data incrementally. As the gradient-based methods fail to solve this optimization problem, we propose to study the Lipschitz continuity property of the model solution, enabling us to calculate the semi-derivative, a one-sided directional derivative, of the solution over data. We leverage semi-derivative descent to solve the bi-level optimization problem, and establish the convergence conditions of the method to any attainable target model. The model and solution method are illustrated with a simulation of a poisoning attack on the lane change detection using SVM.

摘要: 智能系统对数据的日益依赖使得系统容易受到数据中毒攻击。这种攻击可能会破坏输入数据，从而危及机器学习或深度学习模型。以往关于数据中毒攻击的研究都受到特定假设的限制，对具有一般（等式和不等式）约束或缺乏可微性的学习模型的关注有限。这种学习模型在实践中很常见，特别是在涉及物理或领域知识作为特定模型约束的智能交通系统（ITS）中。受ITS应用的启发，本文将模型-目标数据中毒攻击描述为具有约束较低层问题的双层优化问题，旨在通过增量修改训练数据将模型解引导到对手指定的目标解。由于基于梯度的方法无法解决这个优化问题，我们建议研究模型解的Lipschitz连续性，使我们能够计算解对数据的半导（单边方向导）。我们利用半导下降来解决双层优化问题，并建立该方法对任何可达到的目标模型的收敛条件。通过对使用支持者对车道变更检测的中毒攻击进行模拟，说明了该模型和解决方法。



## **21. Sustainable Smart Farm Networks: Enhancing Resilience and Efficiency with Decision Theory-Guided Deep Reinforcement Learning**

可持续智能农场网络：通过决策理论指导的深度强化学习增强韧性和效率 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03721v1) [paper-pdf](http://arxiv.org/pdf/2505.03721v1)

**Authors**: Dian Chen, Zelin Wan, Dong Sam Ha, Jin-Hee Cho

**Abstract**: Solar sensor-based monitoring systems have become a crucial agricultural innovation, advancing farm management and animal welfare through integrating sensor technology, Internet-of-Things, and edge and cloud computing. However, the resilience of these systems to cyber-attacks and their adaptability to dynamic and constrained energy supplies remain largely unexplored. To address these challenges, we propose a sustainable smart farm network designed to maintain high-quality animal monitoring under various cyber and adversarial threats, as well as fluctuating energy conditions. Our approach utilizes deep reinforcement learning (DRL) to devise optimal policies that maximize both monitoring effectiveness and energy efficiency. To overcome DRL's inherent challenge of slow convergence, we integrate transfer learning (TL) and decision theory (DT) to accelerate the learning process. By incorporating DT-guided strategies, we optimize monitoring quality and energy sustainability, significantly reducing training time while achieving comparable performance rewards. Our experimental results prove that DT-guided DRL outperforms TL-enhanced DRL models, improving system performance and reducing training runtime by 47.5%.

摘要: 基于太阳能传感器的监测系统已成为一项重要的农业创新，通过集成传感器技术、物联网、边缘和云计算，促进了农场管理和动物福利。然而，这些系统对网络攻击的弹性及其对动态和受限能源供应的适应性在很大程度上仍未得到探索。为了应对这些挑战，我们提出了一个可持续的智能农场网络，旨在在各种网络和对抗性威胁以及波动的能源条件下保持高质量的动物监测。我们的方法利用深度强化学习（DRL）来设计最佳策略，以最大限度地提高监测效率和能源效率。为了克服DRL固有的收敛速度慢的挑战，我们集成了迁移学习（TL）和决策理论（DT）来加速学习过程。通过结合DT引导的策略，我们优化了质量和能源可持续性的监控，显着减少培训时间，同时实现相当的绩效奖励。我们的实验结果证明，DT引导的DRL优于TL增强的DRL模型，提高了系统性能，并将训练运行时间减少了47.5%。



## **22. Adversarial Robustness of Deep Learning Models for Inland Water Body Segmentation from SAR Images**

SAR图像内陆水体分割深度学习模型的对抗鲁棒性 eess.IV

21 pages, 15 figures, 2 tables

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.01884v2) [paper-pdf](http://arxiv.org/pdf/2505.01884v2)

**Authors**: Siddharth Kothari, Srinivasan Murali, Sankalp Kothari, Ujjwal Verma, Jaya Sreevalsan-Nair

**Abstract**: Inland water body segmentation from Synthetic Aperture Radar (SAR) images is an important task needed for several applications, such as flood mapping. While SAR sensors capture data in all-weather conditions as high-resolution images, differentiating water and water-like surfaces from SAR images is not straightforward. Inland water bodies, such as large river basins, have complex geometry, which adds to the challenge of segmentation. U-Net is a widely used deep learning model for land-water segmentation of SAR images. In practice, manual annotation is often used to generate the corresponding water masks as ground truth. Manual annotation of the images is prone to label noise owing to data poisoning attacks, especially due to complex geometry. In this work, we simulate manual errors in the form of adversarial attacks on the U-Net model and study the robustness of the model to human errors in annotation. Our results indicate that U-Net can tolerate a certain level of corruption before its performance drops significantly. This finding highlights the crucial role that the quality of manual annotations plays in determining the effectiveness of the segmentation model. The code and the new dataset, along with adversarial examples for robust training, are publicly available. (GitHub link - https://github.com/GVCL/IWSeg-SAR-Poison.git)

摘要: 从合成口径雷达（SAR）图像中分割内陆水体是洪水绘图等多种应用所需的重要任务。虽然SAR传感器将全天候条件下的数据捕获为高分辨率图像，但区分水和类水表面与SAR图像并不简单。内陆水体（例如大型流域）具有复杂的几何形状，这增加了分段的挑战。U-Net是一种广泛使用的深度学习模型，用于SAR图像的海陆分割。在实践中，通常使用手动注释来生成相应的水面具作为地面真相。由于数据中毒攻击，特别是由于复杂的几何形状，图像的手动注释容易出现标签噪音。在这项工作中，我们以对抗攻击的形式模拟了对U-Net模型的手动错误，并研究了该模型对注释中人为错误的鲁棒性。我们的结果表明，U-Net在性能显着下降之前可以容忍一定程度的腐败。这一发现凸显了手动注释的质量在决定分割模型的有效性方面所发挥的关键作用。代码和新数据集，以及用于稳健训练的对抗示例都已公开。（GitHub链接-https：//github.com/GVCL/IWSeg-SAR-Poison.git）



## **23. Data-Driven Falsification of Cyber-Physical Systems**

数据驱动的网络物理系统证伪 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03863v1) [paper-pdf](http://arxiv.org/pdf/2505.03863v1)

**Authors**: Atanu Kundu, Sauvik Gon, Rajarshi Ray

**Abstract**: Cyber-Physical Systems (CPS) are abundant in safety-critical domains such as healthcare, avionics, and autonomous vehicles. Formal verification of their operational safety is, therefore, of utmost importance. In this paper, we address the falsification problem, where the focus is on searching for an unsafe execution in the system instead of proving their absence. The contribution of this paper is a framework that (a) connects the falsification of CPS with the falsification of deep neural networks (DNNs) and (b) leverages the inherent interpretability of Decision Trees for faster falsification of CPS. This is achieved by: (1) building a surrogate model of the CPS under test, either as a DNN model or a Decision Tree, (2) application of various DNN falsification tools to falsify CPS, and (3) a novel falsification algorithm guided by the explanations of safety violations of the CPS model extracted from its Decision Tree surrogate. The proposed framework has the potential to exploit a repertoire of \emph{adversarial attack} algorithms designed to falsify robustness properties of DNNs, as well as state-of-the-art falsification algorithms for DNNs. Although the presented methodology is applicable to systems that can be executed/simulated in general, we demonstrate its effectiveness, particularly in CPS. We show that our framework, implemented as a tool \textsc{FlexiFal}, can detect hard-to-find counterexamples in CPS that have linear and non-linear dynamics. Decision tree-guided falsification shows promising results in efficiently finding multiple counterexamples in the ARCH-COMP 2024 falsification benchmarks~\cite{khandait2024arch}.

摘要: 网络物理系统（CPS）广泛应用于医疗保健、航空电子设备和自动驾驶汽车等安全关键领域。因此，对其运营安全性的正式验证至关重要。在本文中，我们解决了伪造问题，重点是搜索系统中不安全的执行，而不是证明它们的不存在。本文的贡献是一个框架，该框架（a）将CPS的伪造与深度神经网络（DNN）的伪造联系起来，并且（b）利用决策树的固有可解释性来更快地伪造CPS。这是通过以下方式实现的：（1）构建受测CPS的代理模型，无论是DNN模型还是决策树，（2）应用各种DNN伪造工具来伪造CPS，以及（3）以CPS模型的安全违规解释为指导的新型伪造算法从其决策树代理中提取。提出的框架有可能利用一系列旨在伪造DNN鲁棒性属性的\{对抗攻击}算法，以及DNN的最先进伪造算法。尽管所提出的方法适用于一般可以执行/模拟的系统，但我们证明了它的有效性，特别是在CPS中。我们表明，我们的框架，实现为一个工具，可以检测到很难找到的反例在CPS具有线性和非线性动态。决策树引导的证伪在有效地发现多个反例在ARCH-COMP 2024证伪基准测试中显示出有希望的结果。



## **24. ALMA: Aggregated Lipschitz Maximization Attack on Auto-encoders**

ALMA：对自动编码器的聚合Lipschitz最大化攻击 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03646v1) [paper-pdf](http://arxiv.org/pdf/2505.03646v1)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite the extensive use of deep autoencoders (AEs) in critical applications, their adversarial robustness remains relatively underexplored compared to classification models. AE robustness is characterized by the Lipschitz bounds of its components. Existing robustness evaluation frameworks based on white-box attacks do not fully exploit the vulnerabilities of intermediate ill-conditioned layers in AEs. In the context of optimizing imperceptible norm-bounded additive perturbations to maximize output damage, existing methods struggle to effectively propagate adversarial loss gradients throughout the network, often converging to less effective perturbations. To address this, we propose a novel layer-conditioning-based adversarial optimization objective that effectively guides the adversarial map toward regions of local Lipschitz bounds by enhancing loss gradient information propagation during attack optimization. We demonstrate through extensive experiments on state-of-the-art AEs that our adversarial objective results in stronger attacks, outperforming existing methods in both universal and sample-specific scenarios. As a defense method against this attack, we introduce an inference-time adversarially trained defense plugin that mitigates the effects of adversarial examples.

摘要: 尽管深度自动编码器（AE）在关键应用中广泛使用，但与分类模型相比，其对抗鲁棒性仍然相对未充分研究。AE鲁棒性由其成分的Lipschitz界来描述。现有的基于白盒攻击的稳健性评估框架并未充分利用AE中中间病态层的漏洞。在优化不可感知的规范有界添加性扰动以最大化输出损害的背景下，现有方法很难在整个网络中有效传播对抗损失梯度，通常会收敛到效率较低的扰动。为了解决这个问题，我们提出了一种新型的基于层条件的对抗性优化目标，该目标通过在攻击优化期间增强损失梯度信息传播，有效地将对抗性地图引导到局部Lipschitz界限区域。我们通过对最先进AE的广泛实验证明，我们的对抗目标会导致更强的攻击，在通用和特定样本场景中都优于现有方法。作为针对这种攻击的防御方法，我们引入了一个推理时对抗训练的防御插件，该插件可以减轻对抗示例的影响。



## **25. The Adaptive Arms Race: Redefining Robustness in AI Security**

自适应军备竞赛：重新定义人工智能安全的稳健性 cs.AI

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2312.13435v3) [paper-pdf](http://arxiv.org/pdf/2312.13435v3)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world AI-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. Canonical robustness evaluation relies on adaptive attacks, which leverage complete knowledge of the defense and are tailored to bypass it. This work broadens the notion of adaptivity, which we employ to enhance both attacks and defenses, showing how they can benefit from mutual learning through interaction. We introduce a framework for adaptively optimizing black-box attacks and defenses under the competitive game they form. To assess robustness reliably, it is essential to evaluate against realistic and worst-case attacks. We thus enhance attacks and their evasive arsenal together using RL, apply the same principle to defenses, and evaluate them first independently and then jointly under a multi-agent perspective. We find that active defenses, those that dynamically control system responses, are an essential complement to model hardening against decision-based attacks; that these defenses can be circumvented by adaptive attacks, something that elicits defenses being adaptive too. Our findings, supported by an extensive theoretical and empirical investigation, confirm that adaptive adversaries pose a serious threat to black-box AI-based systems, rekindling the proverbial arms race. Notably, our approach outperforms the state-of-the-art black-box attacks and defenses, while bringing them together to render effective insights into the robustness of real-world deployed ML-based systems.

摘要: 尽管做出了相当大的努力来使其稳健性，但现实世界中的基于人工智能的系统仍然容易受到基于决策的攻击，因为迄今为止证明其操作稳健性的明确证据很难解决。典型稳健性评估依赖于自适应攻击，这种攻击利用了对防御的完整知识，并经过量身定制以绕过它。这项工作扩展了自适应性的概念，我们使用它来增强攻击和防御，展示了它们如何通过交互从相互学习中受益。我们引入了一个框架，用于在黑匣子形成的竞争游戏下自适应地优化黑匣子攻击和防御。为了可靠地评估稳健性，必须针对现实和最坏情况的攻击进行评估。因此，我们使用RL共同增强攻击及其规避武器库，将相同的原则应用于防御，并首先独立评估它们，然后在多智能体的角度下联合评估它们。我们发现，主动防御（动态控制系统响应的防御）是针对基于决策的攻击的模型强化的重要补充;这些防御可以被自适应攻击规避，这使得防御也具有自适应性。我们的研究结果在广泛的理论和实证调查的支持下证实，适应性对手对基于黑匣子的人工智能系统构成了严重威胁，重新点燃了众所周知的军备竞赛。值得注意的是，我们的方法优于最先进的黑匣子攻击和防御，同时将它们结合在一起，以有效洞察现实世界部署的基于ML的系统的稳健性。



## **26. BadLingual: A Novel Lingual-Backdoor Attack against Large Language Models**

BadLingual：针对大型语言模型的新型语言后门攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03501v1) [paper-pdf](http://arxiv.org/pdf/2505.03501v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Wenbo Jiang, Kangjie Chen, Tianwei Zhang, Qingchuan Zhao, Guowen Xu

**Abstract**: In this paper, we present a new form of backdoor attack against Large Language Models (LLMs): lingual-backdoor attacks. The key novelty of lingual-backdoor attacks is that the language itself serves as the trigger to hijack the infected LLMs to generate inflammatory speech. They enable the precise targeting of a specific language-speaking group, exacerbating racial discrimination by malicious entities. We first implement a baseline lingual-backdoor attack, which is carried out by poisoning a set of training data for specific downstream tasks through translation into the trigger language. However, this baseline attack suffers from poor task generalization and is impractical in real-world settings. To address this challenge, we design BadLingual, a novel task-agnostic lingual-backdoor, capable of triggering any downstream tasks within the chat LLMs, regardless of the specific questions of these tasks. We design a new approach using PPL-constrained Greedy Coordinate Gradient-based Search (PGCG) based adversarial training to expand the decision boundary of lingual-backdoor, thereby enhancing the generalization ability of lingual-backdoor across various tasks. We perform extensive experiments to validate the effectiveness of our proposed attacks. Specifically, the baseline attack achieves an ASR of over 90% on the specified tasks. However, its ASR reaches only 37.61% across six tasks in the task-agnostic scenario. In contrast, BadLingual brings up to 37.35% improvement over the baseline. Our study sheds light on a new perspective of vulnerabilities in LLMs with multilingual capabilities and is expected to promote future research on the potential defenses to enhance the LLMs' robustness

摘要: 在本文中，我们提出了一种针对大型语言模型（LLM）的新形式后门攻击：语言后门攻击。语言后门攻击的关键新颖之处在于，语言本身充当了劫持受感染LLM以产生煽动性言语的触发器。它们能够准确瞄准特定语言群体，加剧恶意实体的种族歧视。我们首先实施基线语言后门攻击，通过翻译成触发语言来毒害特定下游任务的一组训练数据来执行该攻击。然而，这种基线攻击的任务概括性较差，并且在现实世界环境中不切实际。为了应对这一挑战，我们设计了BadLingual，这是一种新型的任务不可知语言后门，能够触发聊天LLM内的任何下游任务，无论这些任务的具体问题如何。我们设计了一种使用PPL约束的基于贪婪协调搜索（PGCG）的对抗训练的新方法，以扩大语言后门的决策边界，从而增强语言后门在各种任务中的概括能力。我们进行了广泛的实验来验证我们提出的攻击的有效性。具体来说，基线攻击在指定任务上实现了超过90%的ASB。然而，在任务不可知的场景中，其六项任务的ASB仅达到37.61%。相比之下，BadLingual较基线提高了37.35%。我们的研究揭示了具有多语言功能的LLM漏洞的新视角，并预计将促进未来对潜在防御措施的研究，以增强LLM的稳健性



## **27. Mitigating Backdoor Triggered and Targeted Data Poisoning Attacks in Voice Authentication Systems**

缓解语音认证系统中后门触发和有针对性的数据中毒攻击 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03455v1) [paper-pdf](http://arxiv.org/pdf/2505.03455v1)

**Authors**: Alireza Mohammadi, Keshav Sood, Dhananjay Thiruvady, Asef Nazari

**Abstract**: Voice authentication systems remain susceptible to two major threats: backdoor triggered attacks and targeted data poisoning attacks. This dual vulnerability is critical because conventional solutions typically address each threat type separately, leaving systems exposed to adversaries who can exploit both attacks simultaneously. We propose a unified defense framework that effectively addresses both BTA and TDPA. Our framework integrates a frequency focused detection mechanism that flags covert pitch boosting and sound masking backdoor attacks in near real time, followed by a convolutional neural network that addresses TDPA. This dual layered defense approach utilizes multidimensional acoustic features to isolate anomalous signals without requiring costly model retraining. In particular, our PBSM detection mechanism can seamlessly integrate into existing voice authentication pipelines and scale effectively for large scale deployments. Experimental results on benchmark datasets and their compression with the state of the art algorithm demonstrate that our PBSM detection mechanism outperforms the state of the art. Our framework reduces attack success rates to as low as five to fifteen percent while maintaining a recall rate of up to ninety five percent in recognizing TDPA.

摘要: 语音认证系统仍然容易受到两种主要威胁：后门触发攻击和有针对性的数据中毒攻击。这种双重漏洞至关重要，因为传统的解决方案通常会单独解决每种威胁类型，从而使系统暴露在可以同时利用这两种攻击的对手手中。我们提出了一个有效解决MTA和TDPA的统一防御框架。我们的框架集成了以频率为中心的检测机制，该机制近乎实时地标记隐蔽音调增强和声音掩蔽后门攻击，然后是解决TDPA的卷积神经网络。这种双层防御方法利用多维声学特征来隔离异常信号，而无需昂贵的模型再训练。特别是，我们的PBSM检测机制可以无缝集成到现有的语音认证管道中，并有效扩展以适应大规模部署。对基准数据集及其使用最先进算法进行压缩的实验结果表明，我们的PBSM检测机制优于最先进的技术。我们的框架将攻击成功率降低至低至百分之五到百分之十五，同时在识别TDPA时保持高达百分之九十五的召回率。



## **28. Robustness in AI-Generated Detection: Enhancing Resistance to Adversarial Attacks**

人工智能生成检测的鲁棒性：增强对抗性攻击的抵抗力 cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03435v1) [paper-pdf](http://arxiv.org/pdf/2505.03435v1)

**Authors**: Sun Haoxuan, Hong Yan, Zhan Jiahui, Chen Haoxing, Lan Jun, Zhu Huijia, Wang Weiqiang, Zhang Liqing, Zhang Jianfu

**Abstract**: The rapid advancement of generative image technology has introduced significant security concerns, particularly in the domain of face generation detection. This paper investigates the vulnerabilities of current AI-generated face detection systems. Our study reveals that while existing detection methods often achieve high accuracy under standard conditions, they exhibit limited robustness against adversarial attacks. To address these challenges, we propose an approach that integrates adversarial training to mitigate the impact of adversarial examples. Furthermore, we utilize diffusion inversion and reconstruction to further enhance detection robustness. Experimental results demonstrate that minor adversarial perturbations can easily bypass existing detection systems, but our method significantly improves the robustness of these systems. Additionally, we provide an in-depth analysis of adversarial and benign examples, offering insights into the intrinsic characteristics of AI-generated content. All associated code will be made publicly available in a dedicated repository to facilitate further research and verification.

摘要: 生成图像技术的快速发展带来了重大的安全问题，特别是在面部生成检测领域。本文研究了当前人工智能生成的人脸检测系统的漏洞。我们的研究表明，虽然现有的检测方法通常在标准条件下实现高准确性，但它们对对抗攻击的鲁棒性有限。为了应对这些挑战，我们提出了一种整合对抗性训练的方法，以减轻对抗性示例的影响。此外，我们利用扩散倒置和重建来进一步增强检测鲁棒性。实验结果表明，微小的对抗性扰动可以轻松绕过现有的检测系统，但我们的方法显着提高了这些系统的鲁棒性。此外，我们还对对抗性和良性示例进行深入分析，深入分析人工智能生成内容的内在特征。所有相关代码都将在专用存储库中公开，以促进进一步的研究和验证。



## **29. Attention-aggregated Attack for Boosting the Transferability of Facial Adversarial Examples**

提高面部对抗示例可移植性的注意力聚集攻击 cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03383v1) [paper-pdf](http://arxiv.org/pdf/2505.03383v1)

**Authors**: Jian-Wei Li, Wen-Ze Shao

**Abstract**: Adversarial examples have revealed the vulnerability of deep learning models and raised serious concerns about information security. The transfer-based attack is a hot topic in black-box attacks that are practical to real-world scenarios where the training datasets, parameters, and structure of the target model are unknown to the attacker. However, few methods consider the particularity of class-specific deep models for fine-grained vision tasks, such as face recognition (FR), giving rise to unsatisfactory attacking performance. In this work, we first investigate what in a face exactly contributes to the embedding learning of FR models and find that both decisive and auxiliary facial features are specific to each FR model, which is quite different from the biological mechanism of human visual system. Accordingly we then propose a novel attack method named Attention-aggregated Attack (AAA) to enhance the transferability of adversarial examples against FR, which is inspired by the attention divergence and aims to destroy the facial features that are critical for the decision-making of other FR models by imitating their attentions on the clean face images. Extensive experiments conducted on various FR models validate the superiority and robust effectiveness of the proposed method over existing methods.

摘要: 对抗性的例子揭示了深度学习模型的脆弱性，并引发了人们对信息安全的严重担忧。基于传输的攻击是黑匣子攻击中的热门话题，这种攻击对于攻击者未知目标模型的训练数据集、参数和结构的现实场景很实用。然而，很少有方法考虑针对细粒度视觉任务（例如人脸识别（FR））的特定类别深度模型的特殊性，从而导致攻击性能不令人满意。在这项工作中，我们首先研究了面部中的哪些因素对FR模型的嵌入学习做出了贡献，发现决定性和辅助面部特征都是每个FR模型特有的，这与人类视觉系统的生物学机制截然不同。因此，我们提出了一种名为注意力聚集攻击（AAA）的新型攻击方法，以增强对抗性示例针对FR的可移植性，该方法受到注意力分歧的启发，旨在通过模仿其他FR模型对干净面部图像的注意力来破坏对决策至关重要的面部特征。在各种FR模型上进行的大量实验验证了所提出的方法相对于现有方法的优越性和鲁棒性。



## **30. A Chaos Driven Metric for Backdoor Attack Detection**

一种基于混沌驱动的后门攻击检测方法 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03208v1) [paper-pdf](http://arxiv.org/pdf/2505.03208v1)

**Authors**: Hema Karnam Surendrababu, Nithin Nagaraj

**Abstract**: The advancement and adoption of Artificial Intelligence (AI) models across diverse domains have transformed the way we interact with technology. However, it is essential to recognize that while AI models have introduced remarkable advancements, they also present inherent challenges such as their vulnerability to adversarial attacks. The current work proposes a novel defense mechanism against one of the most significant attack vectors of AI models - the backdoor attack via data poisoning of training datasets. In this defense technique, an integrated approach that combines chaos theory with manifold learning is proposed. A novel metric - Precision Matrix Dependency Score (PDS) that is based on the conditional variance of Neurochaos features is formulated. The PDS metric has been successfully evaluated to distinguish poisoned samples from non-poisoned samples across diverse datasets.

摘要: 人工智能（AI）模型在不同领域的进步和采用改变了我们与技术互动的方式。然而，必须认识到，虽然人工智能模型带来了显着的进步，但它们也面临着固有的挑战，例如容易受到对抗攻击。当前的工作提出了一种新颖的防御机制，以对抗人工智能模型最重要的攻击载体之一--通过训练数据集的数据中毒进行的后门攻击。在这种防御技术中，提出了一种将混乱理论与多维学习相结合的集成方法。提出了一种基于Neurochaos特征的条件方差的新型指标--精确矩阵依赖性得分（DDS）。PDC指标已成功评估，可在不同数据集中区分有毒样本与非有毒样本。



## **31. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

使用机械可解释性来应对大型语言模型的对抗攻击 cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2503.06269v2) [paper-pdf](http://arxiv.org/pdf/2503.06269v2)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.

摘要: 用于针对LLM创建对抗性扰动的传统白盒方法通常仅依赖于目标模型的梯度计算，而忽略了负责攻击成功或失败的内部机制。相反，分析这些内部机制的可解释性研究缺乏运行时干预之外的实际应用。我们通过引入一种新颖的白盒方法来弥合这一差距，该方法利用机械解释性技术来制作实用的对抗性输入。具体来说，我们首先识别接受子空间--不会触发模型拒绝机制的特征载体集--然后使用基于梯度的优化将嵌入从拒绝子空间重新路由到接受子空间，有效地实现越狱。与经常失败或需要数小时计算的现有技术相比，这种有针对性的方法显着降低了计算成本，在几分钟甚至几秒钟内就实现了对Gemma 2、Llama3.2和Qwen 2.5等最先进模型80- 95%的攻击成功率。我们相信这种方法为攻击研究和防御开发开辟了新的方向。此外，它展示了机械解释性的实际应用，而其他方法效率较低，这凸显了它的实用性。代码和生成的数据集可在https://github.com/Sckathach/subspace-rerouting上获取。



## **32. PEEK: Phishing Evolution Framework for Phishing Generation and Evolving Pattern Analysis using Large Language Models**

TEK：使用大型语言模型进行网络钓鱼生成和演变模式分析的网络钓鱼进化框架 cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2411.11389v2) [paper-pdf](http://arxiv.org/pdf/2411.11389v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), in particular, deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains detection effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems and people vulnerable to an ever-growing array of attacks. We propose the first Phishing Evolution FramEworK (PEEK) for augmenting phishing email datasets with respect to quality and diversity, and analyzing changing phishing patterns for detection to adapt to updated phishing attacks. Specifically, we integrate large language models (LLMs) into the process of adversarial training to enhance the performance of the generated dataset and leverage persuasion principles in a recurrent framework to facilitate the understanding of changing phishing strategies. PEEK raises the proportion of usable phishing samples from 21.4% to 84.8%, surpassing existing works that rely on prompting and fine-tuning LLMs. The phishing datasets provided by PEEK, with evolving phishing patterns, outperform the other two available LLM-generated phishing email datasets in improving detection robustness. PEEK phishing boosts detectors' accuracy to over 88% and reduces adversarial sensitivity by up to 70%, still maintaining 70% detection accuracy against adversarial attacks.

摘要: 网络钓鱼仍然是一种普遍存在的网络威胁，因为攻击者制作了欺骗性电子邮件来引诱受害者泄露敏感信息。虽然人工智能（AI），特别是深度学习，已成为防御网络钓鱼攻击的关键组成部分，但这些方法面临着严重的局限性。主要由于隐私问题，公开可用的、多样化的和更新的数据的稀缺限制了检测有效性。随着网络钓鱼策略的迅速发展，在有限、过时的数据上训练的模型很难检测到新的、复杂的欺骗策略，从而使系统和人们容易受到越来越多的攻击。我们提出了第一个网络钓鱼Evolution FramEworK（TEK），用于增强网络钓鱼电子邮件数据集的质量和多样性，并分析不断变化的网络钓鱼模式进行检测，以适应更新的网络钓鱼攻击。具体来说，我们将大型语言模型（LLM）集成到对抗训练过程中，以增强生成的数据集的性能，并在循环框架中利用说服原则，以促进对不断变化的网络钓鱼策略的理解。TEK将可用网络钓鱼样本的比例从21.4%提高到84.8%，超过了依赖提示和微调LLM的现有作品。TEK提供的网络钓鱼数据集具有不断变化的网络钓鱼模式，在提高检测稳健性方面优于其他两个可用的LLM生成的网络钓鱼电子邮件数据集。TEK网络钓鱼将检测器的准确性提高至88%以上，并将对抗敏感性降低高达70%，针对对抗攻击仍保持70%的检测准确性。



## **33. Adversarial Sample Generation for Anomaly Detection in Industrial Control Systems**

用于工业控制系统异常检测的对抗样本生成 cs.CR

Accepted in the 1st Workshop on Modeling and Verification for Secure  and Performant Cyber-Physical Systems in conjunction with Cyber-Physical  Systems and Internet-of-Things Week, Irvine, USA, May 6-9, 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03120v1) [paper-pdf](http://arxiv.org/pdf/2505.03120v1)

**Authors**: Abdul Mustafa, Muhammad Talha Khan, Muhammad Azmi Umer, Zaki Masood, Chuadhry Mujeeb Ahmed

**Abstract**: Machine learning (ML)-based intrusion detection systems (IDS) are vulnerable to adversarial attacks. It is crucial for an IDS to learn to recognize adversarial examples before malicious entities exploit them. In this paper, we generated adversarial samples using the Jacobian Saliency Map Attack (JSMA). We validate the generalization and scalability of the adversarial samples to tackle a broad range of real attacks on Industrial Control Systems (ICS). We evaluated the impact by assessing multiple attacks generated using the proposed method. The model trained with adversarial samples detected attacks with 95% accuracy on real-world attack data not used during training. The study was conducted using an operational secure water treatment (SWaT) testbed.

摘要: 基于机器学习（ML）的入侵检测系统（IDS）容易受到对抗攻击。对于IDS来说，在恶意实体利用它们之前学会识别对抗性示例至关重要。在本文中，我们使用雅可比显着地图攻击（JSM）生成对抗样本。我们验证了对抗样本的一般性和可扩展性，以应对对工业控制系统（ICS）的广泛实际攻击。我们通过评估使用所提出的方法产生的多个攻击来评估影响。用对抗样本训练的模型在训练期间未使用的现实世界攻击数据上以95%的准确率检测到攻击。该研究使用可操作的安全水处理（SWaT）测试台进行。



## **34. Adversarial Attacks in Multimodal Systems: A Practitioner's Survey**

多模式系统中的对抗性攻击：从业者的调查 cs.LG

Accepted in IEEE COMPSAC 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03084v1) [paper-pdf](http://arxiv.org/pdf/2505.03084v1)

**Authors**: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj

**Abstract**: The introduction of multimodal models is a huge step forward in Artificial Intelligence. A single model is trained to understand multiple modalities: text, image, video, and audio. Open-source multimodal models have made these breakthroughs more accessible. However, considering the vast landscape of adversarial attacks across these modalities, these models also inherit vulnerabilities of all the modalities, and ultimately, the adversarial threat amplifies. While broad research is available on possible attacks within or across these modalities, a practitioner-focused view that outlines attack types remains absent in the multimodal world. As more Machine Learning Practitioners adopt, fine-tune, and deploy open-source models in real-world applications, it's crucial that they can view the threat landscape and take the preventive actions necessary. This paper addresses the gap by surveying adversarial attacks targeting all four modalities: text, image, video, and audio. This survey provides a view of the adversarial attack landscape and presents how multimodal adversarial threats have evolved. To the best of our knowledge, this survey is the first comprehensive summarization of the threat landscape in the multimodal world.

摘要: 多模式模型的引入是人工智能向前迈出的一大步。单个模型经过训练以理解多种模式：文本、图像、视频和音频。开源多模式模型使这些突破变得更容易实现。然而，考虑到这些模式中对抗性攻击的广阔格局，这些模型也继承了所有模式的脆弱性，最终，对抗性威胁会被放大。虽然对这些模式内部或跨这些模式的可能攻击进行了广泛的研究，但在多模式世界中仍然缺乏以攻击者为中心、概述攻击类型的观点。随着越来越多的机器学习实践者在现实世界的应用程序中采用、微调和部署开源模型，他们能够查看威胁格局并采取必要的预防措施至关重要。本文通过调查针对所有四种模式（文本、图像、视频和音频）的对抗攻击来解决这一差距。这项调查提供了对抗性攻击格局的视图，并展示了多模式对抗性威胁的演变方式。据我们所知，这项调查是对多模式世界威胁格局的首次全面总结。



## **35. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

大型语言模型作为软件分析中稳健的数据生成器：我们已经做到了吗？ cs.SE

Accepted to the AI Model/Data Track of the Evaluation and Assessment  in Software Engineering (EASE) 2025 Conference

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2411.10565v3) [paper-pdf](http://arxiv.org/pdf/2411.10565v3)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.

摘要: 大型语言模型（LLM）生成的数据越来越多地用于软件分析，但目前尚不清楚该数据与人类编写的数据相比如何，特别是当模型暴露于对抗场景时。对抗性攻击可能会损害软件系统的可靠性和安全性，因此，与作为模型性能基准的人类编写数据相比，了解LLM生成的数据在这些条件下的表现如何，可以为LLM生成的数据是否提供类似的稳健性和有效性提供有价值的见解。为了解决这一差距，我们系统地评估和比较人类编写的数据和LLM生成的数据的质量，以便在对抗性攻击的背景下微调稳健的预训练模型（Ptms）。我们评估了六种广泛使用的PtM的稳健性，这些PtM在对抗性攻击之前和之后根据人类编写和LLM生成的数据进行了微调。该评估在三个流行的软件分析任务中使用了九种最先进的（SOTA）对抗性攻击技术：克隆检测，代码摘要和代码审查讨论中的情感分析。此外，我们使用11个相似性度量来分析生成的对抗性示例的质量。我们的研究结果表明，虽然对LLM生成的数据进行微调的PTM与对人类编写的数据进行微调的PTM具有竞争力，但它们在软件分析任务中对对抗性攻击的鲁棒性较低。我们的研究强调了进一步探索提高LLM生成的训练数据质量的必要性，以开发高性能且能够抵御软件分析中的对抗性攻击的模型。



## **36. Adversarial Robustness Analysis of Vision-Language Models in Medical Image Segmentation**

医学图像分割中视觉语言模型的对抗鲁棒性分析 cs.CV

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02971v1) [paper-pdf](http://arxiv.org/pdf/2505.02971v1)

**Authors**: Anjila Budathoki, Manish Dhakal

**Abstract**: Adversarial attacks have been fairly explored for computer vision and vision-language models. However, the avenue of adversarial attack for the vision language segmentation models (VLSMs) is still under-explored, especially for medical image analysis.   Thus, we have investigated the robustness of VLSMs against adversarial attacks for 2D medical images with different modalities with radiology, photography, and endoscopy. The main idea of this project was to assess the robustness of the fine-tuned VLSMs specially in the medical domain setting to address the high risk scenario.   First, we have fine-tuned pre-trained VLSMs for medical image segmentation with adapters.   Then, we have employed adversarial attacks -- projected gradient descent (PGD) and fast gradient sign method (FGSM) -- on that fine-tuned model to determine its robustness against adversaries.   We have reported models' performance decline to analyze the adversaries' impact.   The results exhibit significant drops in the DSC and IoU scores after the introduction of these adversaries. Furthermore, we also explored universal perturbation but were not able to find for the medical images.   \footnote{https://github.com/anjilab/secure-private-ai}

摘要: 计算机视觉和视觉语言模型的对抗性攻击已经得到了充分的探索。然而，视觉语言分割模型（VLSM）的对抗攻击途径仍然没有得到充分的探索，尤其是对于医学图像分析。   因此，我们研究了VLSM对放射学、摄影和内窥镜检查等不同模式的2D医学图像对抗攻击的稳健性。该项目的主要想法是评估微调VLSM的稳健性，特别是在医疗领域环境中，以应对高风险场景。   首先，我们对预训练的VLSM进行了微调，用于使用适配器进行医学图像分割。   然后，我们对该微调模型采用了对抗攻击--投影梯度下降（PVD）和快速梯度符号法（FGSM）--以确定其对对手的鲁棒性。   我们报告了模型的性能下降，以分析对手的影响。   结果显示，引入这些对手后，DSA和IoU分数显着下降。此外，我们还探索了普遍扰动，但未能找到医学图像。   \脚注{https：//github.com/anjilab/secure-private-ai}



## **37. Constrained Adversarial Learning for Automated Software Testing: a literature review**

用于自动化软件测试的约束对抗学习：文献综述 cs.SE

36 pages, 4 tables, 2 figures, Discover Applied Sciences journal

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2303.07546v2) [paper-pdf](http://arxiv.org/pdf/2303.07546v2)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: It is imperative to safeguard computer applications and information systems against the growing number of cyber-attacks. Automated software testing tools can be developed to quickly analyze many lines of code and detect vulnerabilities by generating function-specific testing data. This process draws similarities to the constrained adversarial examples generated by adversarial machine learning methods, so there could be significant benefits to the integration of these methods in testing tools to identify possible attack vectors. Therefore, this literature review is focused on the current state-of-the-art of constrained data generation approaches applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance their software testing tools with adversarial testing methods and improve the resilience and robustness of their information systems. The found approaches were systematized, and the advantages and limitations of those specific for white-box, grey-box, and black-box testing were analyzed, identifying research gaps and opportunities to automate the testing tools with data generated by adversarial attacks.

摘要: 保护计算机应用程序和信息系统免受日益增多的网络攻击至关重要。可以开发自动化软件测试工具来快速分析多行代码并通过生成特定于功能的测试数据来检测漏洞。该过程与对抗性机器学习方法生成的受约束对抗示例具有相似之处，因此将这些方法集成到测试工具中以识别可能的攻击载体可能会带来显着的好处。因此，本次文献综述重点关注当前应用于对抗性学习和软件测试的约束数据生成方法的最新发展水平，旨在指导研究人员和开发人员通过对抗性测试方法增强其软件测试工具，并提高其信息系统的弹性和稳健性。对所发现的方法进行了系统化，并分析了白盒、灰盒和黑盒测试方法的优点和局限性，确定了研究差距和机会，以利用对抗性攻击生成的数据自动化测试工具。



## **38. Commitment Attacks on Ethereum's Reward Mechanism**

对以太坊奖励机制的承诺攻击 cs.CR

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2407.19479v2) [paper-pdf](http://arxiv.org/pdf/2407.19479v2)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains, such as Ethereum, are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization.   In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized, not only in the context of these attacks, but also practical for implementation in Ethereum.

摘要: 以太坊等无需许可的大型区块链中的验证者通常是回报最大化的理性参与者。以太坊依赖于协议内激励，例如对正确和及时投票的奖励，来诱导诚实行为并保护区块链。然而，外部激励，例如区块提议者捕获最大可提取值（MEV）的机会，可能会引诱验证者偏离诚实协议参与。   我们展示了对LMD Ghost（以太坊共识机制的核心部分）的一系列承诺攻击。我们展示了单个对抗性区块提案者如何通过操纵以太坊的及时投票奖励系统来策划长期连锁重组。这些攻击破坏了提议者和选民之间预期的权力平衡：通过利用可信的威胁，对抗性提议者可以强迫选民从之前的位置进入与诚实链冲突的支持区块，从而实现链重组。   作为回应，我们引入了一种新颖的奖励机制，恢复选民对提案人权力的制衡作用。我们提出的缓解措施更加公平、更加分散，不仅在这些攻击的背景下，而且对于在以太坊中的实施也是可行的。



## **39. Robustness questions the interpretability of graph neural networks: what to do?**

鲁棒性质疑图神经网络的可解释性：该怎么办？ cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02566v1) [paper-pdf](http://arxiv.org/pdf/2505.02566v1)

**Authors**: Kirill Lukyanov, Georgii Sazonov, Serafim Boyarsky, Ilya Makarov

**Abstract**: Graph Neural Networks (GNNs) have become a cornerstone in graph-based data analysis, with applications in diverse domains such as bioinformatics, social networks, and recommendation systems. However, the interplay between model interpretability and robustness remains poorly understood, especially under adversarial scenarios like poisoning and evasion attacks. This paper presents a comprehensive benchmark to systematically analyze the impact of various factors on the interpretability of GNNs, including the influence of robustness-enhancing defense mechanisms.   We evaluate six GNN architectures based on GCN, SAGE, GIN, and GAT across five datasets from two distinct domains, employing four interpretability metrics: Fidelity, Stability, Consistency, and Sparsity. Our study examines how defenses against poisoning and evasion attacks, applied before and during model training, affect interpretability and highlights critical trade-offs between robustness and interpretability. The framework will be published as open source.   The results reveal significant variations in interpretability depending on the chosen defense methods and model architecture characteristics. By establishing a standardized benchmark, this work provides a foundation for developing GNNs that are both robust to adversarial threats and interpretable, facilitating trust in their deployment in sensitive applications.

摘要: 图形神经网络（GNN）已成为基于图形的数据分析的基石，应用于生物信息学、社交网络和推荐系统等各个领域。然而，模型可解释性和稳健性之间的相互作用仍然知之甚少，尤其是在中毒和规避攻击等对抗场景下。本文提出了一个全面的基准来系统地分析各种因素对GNN可解释性的影响，包括鲁棒性增强防御机制的影响。   我们在来自两个不同领域的五个数据集上评估了基于GCN、SAGE、GIN和GAT的六种GNN架构，采用四种可解释性指标：富达性、稳定性、一致性和稀疏性。我们的研究考察了模型训练之前和期间应用的针对中毒和规避攻击的防御措施如何影响可解释性，并强调了稳健性和可解释性之间的关键权衡。该框架将以开源形式发布。   结果揭示了可解释性的显着差异，具体取决于所选择的防御方法和模型架构特征。通过建立标准化的基准，这项工作为开发既对对抗威胁稳健又可解释的GNN提供了基础，从而促进了对其在敏感应用程序中部署的信任。



## **40. Bayesian Robust Aggregation for Federated Learning**

用于联邦学习的Bayesian稳健聚集 cs.LG

14 pages, 4 figures, 8 tables

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02490v1) [paper-pdf](http://arxiv.org/pdf/2505.02490v1)

**Authors**: Aleksandr Karakulev, Usama Zafar, Salman Toor, Prashant Singh

**Abstract**: Federated Learning enables collaborative training of machine learning models on decentralized data. This scheme, however, is vulnerable to adversarial attacks, when some of the clients submit corrupted model updates. In real-world scenarios, the total number of compromised clients is typically unknown, with the extent of attacks potentially varying over time. To address these challenges, we propose an adaptive approach for robust aggregation of model updates based on Bayesian inference. The mean update is defined by the maximum of the likelihood marginalized over probabilities of each client to be `honest'. As a result, the method shares the simplicity of the classical average estimators (e.g., sample mean or geometric median), being independent of the number of compromised clients. At the same time, it is as effective against attacks as methods specifically tailored to Federated Learning, such as Krum. We compare our approach with other aggregation schemes in federated setting on three benchmark image classification data sets. The proposed method consistently achieves state-of-the-art performance across various attack types with static and varying number of malicious clients.

摘要: 联合学习能够在去中心化数据上对机器学习模型进行协作训练。然而，当一些客户端提交损坏的模型更新时，该计划很容易受到对抗攻击。在现实世界的场景中，受攻击客户端的总数通常是未知的，攻击的程度可能会随着时间的推移而变化。为了解决这些挑战，我们提出了一种基于Bayesian推理的模型更新稳健聚合的自适应方法。平均更新由每个客户“诚实”的可能性的最大值定义。因此，该方法具有经典平均估计量的简单性（例如，样本平均值或几何中位数），与受影响客户的数量无关。与此同时，它与专门为联邦学习量身定制的方法（例如Krum）一样有效。我们在三个基准图像分类数据集上将我们的方法与联邦环境中的其他聚合方案进行了比较。所提出的方法在具有静态和不同数量的恶意客户端的各种攻击类型中始终实现最先进的性能。



## **41. Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training**

灾难性的过度匹配、熵差和参与率：快速对抗训练的无声$l & p$ Norm解决方案 cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02360v1) [paper-pdf](http://arxiv.org/pdf/2505.02360v1)

**Authors**: Fares B. Mehouachi, Saif Eddin Jabari

**Abstract**: Adversarial training is a cornerstone of robust deep learning, but fast methods like the Fast Gradient Sign Method (FGSM) often suffer from Catastrophic Overfitting (CO), where models become robust to single-step attacks but fail against multi-step variants. While existing solutions rely on noise injection, regularization, or gradient clipping, we propose a novel solution that purely controls the $l^p$ training norm to mitigate CO.   Our study is motivated by the empirical observation that CO is more prevalent under the $l^{\infty}$ norm than the $l^2$ norm. Leveraging this insight, we develop a framework for generalized $l^p$ attack as a fixed point problem and craft $l^p$-FGSM attacks to understand the transition mechanics from $l^2$ to $l^{\infty}$. This leads to our core insight: CO emerges when highly concentrated gradients where information localizes in few dimensions interact with aggressive norm constraints. By quantifying gradient concentration through Participation Ratio and entropy measures, we develop an adaptive $l^p$-FGSM that automatically tunes the training norm based on gradient information. Extensive experiments demonstrate that this approach achieves strong robustness without requiring additional regularization or noise injection, providing a novel and theoretically-principled pathway to mitigate the CO problem.

摘要: 对抗性训练是稳健深度学习的基石，但像快速梯度符号法（FGSM）这样的快速方法经常受到灾难性过适应（CO）的影响，即模型对单步攻击变得稳健，但对多步变体却失败。虽然现有的解决方案依赖于噪音注入、正规化或梯度限幅，但我们提出了一种新颖的解决方案，该解决方案纯粹控制$l ' p$训练规范以减轻CO。   我们的研究的动机是经验观察，即CO在$l &{\infty}$规范下比在$l & 2 $规范下更普遍。利用这一见解，我们开发了一个将广义$l ' p$攻击作为定点问题的框架，并精心设计了$l ' p$-FGSM攻击，以了解从$l ' 2 $到$l '#'#'的转变机制。这引出了我们的核心见解：当信息局部化在少数维度上的高度集中的梯度与激进的规范约束相互作用时，CO就会出现。通过通过参与率和信息量量化梯度集中度，我们开发了一种自适应的$l ' p$-FGSM，它可以根据梯度信息自动调整训练规范。大量实验表明，这种方法在不需要额外的正规化或噪音注入的情况下实现了很强的鲁棒性，提供了一种新颖且有理论原则的途径来缓解CO问题。



## **42. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

针对会员推断攻击的Bayes-Nash生成隐私 cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2410.07414v4) [paper-pdf](http://arxiv.org/pdf/2410.07414v4)

**Authors**: Tao Zhang, Rajagopal Venkatesaramani, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) expose significant privacy risks by determining whether an individual's data is in a dataset. While differential privacy (DP) mitigates such risks, it has several limitations in achieving an optimal balance between utility and privacy, include limited resolution in expressing this tradeoff in only a few privacy parameters, and intractable sensitivity calculations that may be necessary to provide tight privacy guarantees. We propose a game-theoretic framework that models privacy protection from MIA as a Bayesian game between a defender and an attacker. In this game, a dataset is the defender's private information, with privacy loss to the defender (which is gain to the attacker) captured in terms of the attacker's ability to infer membership of individuals in the dataset. To address the strategic complexity of this game, we represent the mixed strategy of the defender as a neural network generator which maps a private dataset to its public representation (for example, noisy summary statistics), while the mixed strategy of the attacker is captured by a discriminator which makes membership inference claims. We refer to the resulting computational approach as a general-sum Generative Adversarial Network, which is trained iteratively by alternating generator and discriminator updates akin to conventional GANs. We call the defender's data sharing policy thereby obtained Bayes-Nash Generative Privacy (BNGP). The BNGP strategy avoids sensitivity calculations, supports compositions of correlated mechanisms, is robust to the attacker's heterogeneous preferences over true and false positives, and yields provable differential privacy guarantees, albeit in an idealized setting.

摘要: 成员资格推断攻击（MIA）通过确定个人的数据是否在数据集中暴露了重大的隐私风险。虽然差异隐私（DP）可以减轻此类风险，但它在实现效用和隐私之间的最佳平衡方面存在一些局限性，包括仅用少数隐私参数表达这种权衡的分辨率有限，以及提供严格隐私保证可能需要的棘手敏感性计算。我们提出了一个博弈论框架，将MIA的隐私保护建模为防御者和攻击者之间的Bayesian博弈。在这个游戏中，数据集是防御者的私人信息，防御者的隐私损失（这是攻击者的收益）根据攻击者推断数据集中个人成员资格的能力来捕捉。为了解决这个游戏的策略复杂性，我们将防御者的混合策略表示为神经网络生成器，该生成器将私人数据集映射到其公共表示（例如，有噪的摘要统计数据），而攻击者的混合策略则由发起成员资格推断的搜索器捕获。我们将由此产生的计算方法称为通用和生成对抗网络，它通过类似于传统GAN的交替生成器和RST更新进行迭代训练。我们将由此获得的防御者的数据共享政策称为Bayes-Nash生成隐私（BNGP）。BCGP策略避免了敏感性计算，支持相关机制的组合，对攻击者对真阳性和假阳性的不同偏好具有鲁棒性，并产生可证明的差异隐私保证（尽管是在理想化的环境中）。



## **43. Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents**

多代理安全面临的开放挑战：迈向交互式人工智能代理的安全系统 cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.02077v1) [paper-pdf](http://arxiv.org/pdf/2505.02077v1)

**Authors**: Christian Schroeder de Witt

**Abstract**: Decentralized AI agents will soon interact across internet platforms, creating security challenges beyond traditional cybersecurity and AI safety frameworks. Free-form protocols are essential for AI's task generalization but enable new threats like secret collusion and coordinated swarm attacks. Network effects can rapidly spread privacy breaches, disinformation, jailbreaks, and data poisoning, while multi-agent dispersion and stealth optimization help adversaries evade oversightcreating novel persistent threats at a systemic level. Despite their critical importance, these security challenges remain understudied, with research fragmented across disparate fields including AI security, multi-agent learning, complex systems, cybersecurity, game theory, distributed systems, and technical AI governance. We introduce \textbf{multi-agent security}, a new field dedicated to securing networks of decentralized AI agents against threats that emerge or amplify through their interactionswhether direct or indirect via shared environmentswith each other, humans, and institutions, and characterize fundamental security-performance trade-offs. Our preliminary work (1) taxonomizes the threat landscape arising from interacting AI agents, (2) surveys security-performance tradeoffs in decentralized AI systems, and (3) proposes a unified research agenda addressing open challenges in designing secure agent systems and interaction environments. By identifying these gaps, we aim to guide research in this critical area to unlock the socioeconomic potential of large-scale agent deployment on the internet, foster public trust, and mitigate national security risks in critical infrastructure and defense contexts.

摘要: 去中心化的人工智能代理很快将在互联网平台上互动，从而带来超越传统网络安全和人工智能安全框架的安全挑战。自由形式的协议对于人工智能的任务概括至关重要，但也会产生秘密共谋和协同群攻击等新威胁。网络效应可以迅速传播隐私泄露、虚假信息、越狱和数据中毒，而多代理分散和隐形优化帮助对手逃避疏忽，从而在系统层面上产生新颖的持续威胁。尽管这些安全挑战至关重要，但研究仍然不足，研究分散在不同领域，包括人工智能安全、多代理学习、复杂系统、网络安全、博弈论、分布式系统和技术人工智能治理。我们引入了\textBF{多代理安全}，这是一个新领域，致力于保护去中心化人工智能代理网络免受通过相互作用而出现或放大的威胁（无论是直接还是间接通过与彼此、人类和机构的共享环境），并描述了基本的安全性能权衡。我们的初步工作（1）对交互人工智能代理产生的威胁格局进行分类，（2）调查去中心化人工智能系统中的安全性能权衡，（3）提出了一个统一的研究议程，解决设计安全代理系统和交互环境中的开放挑战。通过识别这些差距，我们的目标是指导这一关键领域的研究，以释放互联网上大规模代理部署的社会经济潜力，促进公众信任，并减轻关键基础设施和国防环境中的国家安全风险。



## **44. Lightweight Defense Against Adversarial Attacks in Time Series Classification**

时间序列分类中对抗攻击的轻量级防御 cs.LG

13 pages, 8 figures. Accepted at RAFDA Workshop, PAKDD 2025  (Springer, EI & Scopus indexed). Code:  https://github.com/Yi126/Lightweight-Defence

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.02073v1) [paper-pdf](http://arxiv.org/pdf/2505.02073v1)

**Authors**: Yi Han

**Abstract**: As time series classification (TSC) gains prominence, ensuring robust TSC models against adversarial attacks is crucial. While adversarial defense is well-studied in Computer Vision (CV), the TSC field has primarily relied on adversarial training (AT), which is computationally expensive. In this paper, five data augmentation-based defense methods tailored for time series are developed, with the most computationally intensive method among them increasing the computational resources by only 14.07% compared to the original TSC model. Moreover, the deployment process for these methods is straightforward. By leveraging these advantages of our methods, we create two combined methods. One of these methods is an ensemble of all the proposed techniques, which not only provides better defense performance than PGD-based AT but also enhances the generalization ability of TSC models. Moreover, the computational resources required for our ensemble are less than one-third of those required for PGD-based AT. These methods advance robust TSC in data mining. Furthermore, as foundation models are increasingly explored for time series feature learning, our work provides insights into integrating data augmentation-based adversarial defense with large-scale pre-trained models in future research.

摘要: 随着时间序列分类（TSC）的日益突出，确保强大的TSC模型抵御对抗性攻击至关重要。虽然对抗性防御在计算机视觉（CV）中得到了很好的研究，但TSC领域主要依赖于对抗性训练（AT），这在计算上是昂贵的。本文提出了五种基于数据增强的时间序列防御方法，其中计算量最大的方法与原始TSC模型相比仅增加了14.07%的计算资源。此外，这些方法的部署过程很简单。通过利用我们方法的这些优势，我们创建了两种组合方法。其中一种方法是所有提出的技术的集成，它不仅提供比基于PGD的AT更好的防御性能，而且还增强了OSC模型的概括能力。此外，我们的集成所需的计算资源还不到基于PGD的AT所需的三分之一。这些方法在数据挖掘中推进了稳健的OSC。此外，随着基础模型被越来越多地探索用于时间序列特征学习，我们的工作为在未来的研究中将基于数据增强的对抗性防御与大规模预训练模型集成提供了见解。



## **45. A Comprehensive Analysis of Adversarial Attacks against Spam Filters**

针对垃圾邮件过滤器的对抗性攻击综合分析 cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.03831v1) [paper-pdf](http://arxiv.org/pdf/2505.03831v1)

**Authors**: Esra Hotoğlu, Sevil Sen, Burcu Can

**Abstract**: Deep learning has revolutionized email filtering, which is critical to protect users from cyber threats such as spam, malware, and phishing. However, the increasing sophistication of adversarial attacks poses a significant challenge to the effectiveness of these filters. This study investigates the impact of adversarial attacks on deep learning-based spam detection systems using real-world datasets. Six prominent deep learning models are evaluated on these datasets, analyzing attacks at the word, character sentence, and AI-generated paragraph-levels. Novel scoring functions, including spam weights and attention weights, are introduced to improve attack effectiveness. This comprehensive analysis sheds light on the vulnerabilities of spam filters and contributes to efforts to improve their security against evolving adversarial threats.

摘要: 深度学习彻底改变了电子邮件过滤，这对于保护用户免受垃圾邮件、恶意软件和网络钓鱼等网络威胁至关重要。然而，对抗攻击的日益复杂，对这些过滤器的有效性构成了重大挑战。本研究使用现实世界数据集调查了对抗攻击对基于深度学习的垃圾邮件检测系统的影响。在这些数据集上评估了六个著名的深度学习模型，分析单词、字符句和人工智能生成的段落级别的攻击。引入了新颖的评分功能，包括垃圾邮件权重和注意力权重，以提高攻击有效性。这项全面的分析揭示了垃圾邮件过滤器的漏洞，并有助于提高其安全性，以应对不断变化的对抗威胁。



## **46. CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation**

CAMOUFLAGE：通过LLM驱动的对抗性主张转换开发错误信息检测系统 cs.CL

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01900v1) [paper-pdf](http://arxiv.org/pdf/2505.01900v1)

**Authors**: Mazal Bethany, Nishant Vishwamitra, Cho-Yu Jason Chiang, Peyman Najafirad

**Abstract**: Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims.

摘要: 自动化的基于证据的错误信息检测系统根据证据评估简短主张的真实性，但缺乏对其对抗漏洞的全面分析。现有的基于黑匣子文本的对抗攻击不适合基于证据的错误信息检测系统，因为这些攻击主要集中在涉及梯度或基于逻辑的优化策略的标记级替换上，而这些策略无法愚弄这些检测系统的多组件性质。这些系统结合了检索和主张证据比较模块，这需要攻击破坏证据检索和/或比较模块，以便得出错误的推论。我们提出了CAMOUFLAGE，这是一种迭代的、LLM驱动的方法，它采用双代理系统、即时优化代理和攻击代理来创建对抗性主张重写，从而操纵证据检索并误导主张证据比较，有效地绕过系统而不改变主张的含义。Attacker Agent生成试图误导检测器的语义等效重写，而Prompt Optimization Agent分析失败的攻击尝试并改进Attacker的提示以指导后续重写。这使得文本能够进行更大的结构和风格转换，而不是标记级别的替换，从而根据先前的结果调整变化的幅度。与现有方法不同，CAMOUFLAGE仅基于二进制模型决策来优化其攻击，以指导其重写过程，从而消除了对分类器logits或广泛查询的需要。我们在四个系统上评估了CAMOUFLAGE，包括两个最近的学术系统和两个真实世界的API，平均攻击成功率为46.92%，同时保持了文本的连贯性和语义等价性。



## **47. PQS-BFL: A Post-Quantum Secure Blockchain-based Federated Learning Framework**

PQS-BFL：后量子安全的基于区块链的联邦学习框架 cs.CR

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01866v1) [paper-pdf](http://arxiv.org/pdf/2505.01866v1)

**Authors**: Daniel Commey, Garth V. Crosby

**Abstract**: Federated Learning (FL) enables collaborative model training while preserving data privacy, but its classical cryptographic underpinnings are vulnerable to quantum attacks. This vulnerability is particularly critical in sensitive domains like healthcare. This paper introduces PQS-BFL (Post-Quantum Secure Blockchain-based Federated Learning), a framework integrating post-quantum cryptography (PQC) with blockchain verification to secure FL against quantum adversaries. We employ ML-DSA-65 (a FIPS 204 standard candidate, formerly Dilithium) signatures to authenticate model updates and leverage optimized smart contracts for decentralized validation. Extensive evaluations on diverse datasets (MNIST, SVHN, HAR) demonstrate that PQS-BFL achieves efficient cryptographic operations (average PQC sign time: 0.65 ms, verify time: 0.53 ms) with a fixed signature size of 3309 Bytes. Blockchain integration incurs a manageable overhead, with average transaction times around 4.8 s and gas usage per update averaging 1.72 x 10^6 units for PQC configurations. Crucially, the cryptographic overhead relative to transaction time remains minimal (around 0.01-0.02% for PQC with blockchain), confirming that PQC performance is not the bottleneck in blockchain-based FL. The system maintains competitive model accuracy (e.g., over 98.8% for MNIST with PQC) and scales effectively, with round times showing sublinear growth with increasing client numbers. Our open-source implementation and reproducible benchmarks validate the feasibility of deploying long-term, quantum-resistant security in practical FL systems.

摘要: 联邦学习（FL）支持协作模型训练，同时保护数据隐私，但其经典的密码学基础容易受到量子攻击。这种漏洞在医疗保健等敏感领域尤为严重。本文介绍了PQS-BFL（后量子安全基于区块链的联合学习），这是一个将后量子密码学（PQC）与区块链验证相结合的框架，以保护FL免受量子对手的攻击。我们使用ML-DSA-65（FIPS 204标准候选者，以前称为Dilithium）签名来验证模型更新，并利用优化的智能合约进行分散验证。对不同数据集（MNIST、SVHN、HAR）的广泛评估表明，PQS-BFL实现了高效的加密操作（平均PQC签名时间：0.65 ms，验证时间：0.53 ms），固定签名大小为3309。区块链集成会产生可管理的费用，PQC配置的平均交易时间约为4.8秒，每次更新的气体使用量平均为1.72 x 106单位。至关重要的是，相对于交易时间的加密费用仍然最小（对于采用区块链的PQC，约为0.01-0.02%），这证实了PQC性能并不是基于区块链的FL的瓶颈。该系统保持有竞争力的模型准确性（例如，配备PQC的MNIST超过98.8%），并且有效扩展，随着客户数量的增加，整周时间显示出亚线性增长。我们的开源实施和可重复基准验证了在实际FL系统中部署长期、抗量子安全性的可行性。



## **48. Rogue Cell: Adversarial Attack and Defense in Untrusted O-RAN Setup Exploiting the Traffic Steering xApp**

Rogue Cell：利用流量引导xApp的不受信任O-RAN设置中的对抗攻击和防御 cs.CR

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01816v1) [paper-pdf](http://arxiv.org/pdf/2505.01816v1)

**Authors**: Eran Aizikovich, Dudu Mimran, Edita Grolman, Yuval Elovici, Asaf Shabtai

**Abstract**: The Open Radio Access Network (O-RAN) architecture is revolutionizing cellular networks with its open, multi-vendor design and AI-driven management, aiming to enhance flexibility and reduce costs. Although it has many advantages, O-RAN is not threat-free. While previous studies have mainly examined vulnerabilities arising from O-RAN's intelligent components, this paper is the first to focus on the security challenges and vulnerabilities introduced by transitioning from single-operator to multi-operator RAN architectures. This shift increases the risk of untrusted third-party operators managing different parts of the network. To explore these vulnerabilities and their potential mitigation, we developed an open-access testbed environment that integrates a wireless network simulator with the official O-RAN Software Community (OSC) RAN intelligent component (RIC) cluster. This environment enables realistic, live data collection and serves as a platform for demonstrating APATE (adversarial perturbation against traffic efficiency), an evasion attack in which a malicious cell manipulates its reported key performance indicators (KPIs) and deceives the O-RAN traffic steering to gain unfair allocations of user equipment (UE). To ensure that O-RAN's legitimate activity continues, we introduce MARRS (monitoring adversarial RAN reports), a detection framework based on a long-short term memory (LSTM) autoencoder (AE) that learns contextual features across the network to monitor malicious telemetry (also demonstrated in our testbed). Our evaluation showed that by executing APATE, an attacker can obtain a 248.5% greater UE allocation than it was supposed to in a benign scenario. In addition, the MARRS detection method was also shown to successfully classify malicious cell activity, achieving accuracy of 99.2% and an F1 score of 0.978.

摘要: 开放无线电接入网络（O-RAN）架构凭借其开放的多供应商设计和人工智能驱动的管理，正在彻底改变蜂窝网络，旨在增强灵活性并降低成本。尽管O-RAN有很多优势，但它并非没有威胁。虽然之前的研究主要研究了O-RAN智能组件产生的漏洞，但本文首次关注从单运营商RAN架构过渡到多运营商RAN架构所带来的安全挑战和漏洞。这种转变增加了不受信任的第三方运营商管理网络不同部分的风险。为了探索这些漏洞及其潜在的缓解措施，我们开发了一个开放访问测试台环境，该环境将无线网络模拟器与官方O-RAN软件社区（OSC）RAN智能组件（RIC）集群集成。该环境能够实现真实的实时数据收集，并充当演示APATE（针对流量效率的对抗性扰动）的平台，APATE是一种规避攻击，其中恶意蜂窝操纵其报告的关键性能指标（KPI）并欺骗O-RAN流量引导以获得用户设备（UE）的不公平分配。为了确保O-RAN的合法活动继续进行，我们引入了MARRS（监控对抗性RAN报告），这是一种基于长短期记忆（LSTM）自动编码器（AE）的检测框架，可以学习整个网络的上下文特征以监控恶意遥感（也在我们的测试床上演示）。我们的评估表明，通过执行APATE，攻击者可以获得比良性情况下预期多248.5%的UE分配。此外，MARRS检测方法还被证明可以成功分类恶意细胞活动，准确率为99.2%，F1评分为0.978。



## **49. LeapFrog: The Rowhammer Instruction Skip Attack**

LeapFrog：Rowhammer指令跳过攻击 cs.CR

Accepted at EuroS&P 2025 and Hardware.io 2024,

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2404.07878v3) [paper-pdf](http://arxiv.org/pdf/2404.07878v3)

**Authors**: Andrew Adiletta, M. Caner Tol, Kemal Derya, Berk Sunar, Saad Islam

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats compromising data integrity and the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The LeapFrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, repositions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify LeapFrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first show the attack on a decision tree algorithm to show the potential implications. Secondly, we employ the attack on OpenSSL to bypass the encryption and reveal the plaintext. We then use our tools to scan the Open Quantum Safe library and report on the number of LeapFrog gadgets in the code. Lastly, we demonstrate this new attack vector through a practical demonstration in a client/server TLS handshake scenario, successfully inducing an instruction skip in a client application. Our findings extend the impact of Rowhammer attacks on control flow and contribute to developing more robust defenses against these increasingly sophisticated threats.

摘要: 自成立以来，Rowhammer漏洞利用已迅速演变为日益复杂的威胁，损害了受害者流程的数据完整性和控制流完整性。然而，攻击者识别易受攻击的目标（即，Rowhammer小工具），了解尝试错误的结果，并制定产生有用结果的攻击。   在本文中，我们提出了一种新型Rowhammer小工具，称为LeapFrog小工具，当它出现在受害者代码中时，它允许对手颠覆代码执行以绕过关键代码段（例如，身份验证检查逻辑、加密回合、安全协议中的填充）。当受害者代码将程序计数器（PC）值存储在用户或内核堆栈中时，LeapFrog小工具就会显现（例如，函数调用期间的返回地址），当被篡改时，会将返回地址重新定位到绕过安全关键代码模式的位置。   这项研究还提出了一个识别LeapFrog小工具的系统过程。该方法能够自动检测易感目标并确定最佳攻击参数。我们首先展示对决策树算法的攻击，以展示潜在的影响。其次，我们利用对OpenSSL的攻击来绕过加密并揭示明文。然后，我们使用我们的工具扫描Open Quantum Safe库并报告代码中LeapFrog小工具的数量。最后，我们通过在客户端/服务器TLS握手场景中的实际演示来演示这种新的攻击向量，成功地在客户端应用程序中诱导指令跳过。我们的研究结果扩展了Rowhammer攻击对控制流的影响，并有助于开发更强大的防御措施来应对这些日益复杂的威胁。



## **50. Modeling Behavioral Preferences of Cyber Adversaries Using Inverse Reinforcement Learning**

使用反向强化学习建模网络对手的行为偏好 cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.03817v1) [paper-pdf](http://arxiv.org/pdf/2505.03817v1)

**Authors**: Aditya Shinde, Prashant Doshi

**Abstract**: This paper presents a holistic approach to attacker preference modeling from system-level audit logs using inverse reinforcement learning (IRL). Adversary modeling is an important capability in cybersecurity that lets defenders characterize behaviors of potential attackers, which enables attribution to known cyber adversary groups. Existing approaches rely on documenting an ever-evolving set of attacker tools and techniques to track known threat actors. Although attacks evolve constantly, attacker behavioral preferences are intrinsic and less volatile. Our approach learns the behavioral preferences of cyber adversaries from forensics data on their tools and techniques. We model the attacker as an expert decision-making agent with unknown behavioral preferences situated in a computer host. We leverage attack provenance graphs of audit logs to derive a state-action trajectory of the attack. We test our approach on open datasets of audit logs containing real attack data. Our results demonstrate for the first time that low-level forensics data can automatically reveal an adversary's subjective preferences, which serves as an additional dimension to modeling and documenting cyber adversaries. Attackers' preferences tend to be invariant despite their different tools and indicate predispositions that are inherent to the attacker. As such, these inferred preferences can potentially serve as unique behavioral signatures of attackers and improve threat attribution.

摘要: 本文提出了一种使用反向强化学习（IRL）从系统级审计日志中进行攻击者偏好建模的整体方法。敌对者建模是网络安全领域的一项重要功能，可以让防御者描述潜在攻击者的行为，从而能够归因于已知的网络对手群体。现有的方法依赖于记录一组不断发展的攻击者工具和技术来跟踪已知的威胁参与者。尽管攻击不断发展，但攻击者的行为偏好是固有的，波动性较小。我们的方法从网络对手工具和技术的取证数据中学习网络对手的行为偏好。我们将攻击者建模为位于计算机主机中具有未知行为偏好的专家决策代理。我们利用审计日志的攻击来源图来推导攻击的状态动作轨迹。我们在包含真实攻击数据的审计日志开放数据集上测试我们的方法。我们的结果首次证明，低级取证数据可以自动揭示对手的主观偏好，这是建模和记录网络对手的额外维度。尽管攻击者的工具不同，但其偏好往往是不变的，并表明攻击者固有的倾向。因此，这些推断的偏好可能会作为攻击者的独特行为签名并改善威胁归因。



