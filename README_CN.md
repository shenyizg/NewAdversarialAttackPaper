# Latest Adversarial Attack Papers
**update at 2025-12-25 10:11:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents**

CoTDeceptor：针对CoT增强LLM代码代理的对抗代码混淆 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21250v1) [paper-pdf](https://arxiv.org/pdf/2512.21250v1)

**Authors**: Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He

**Abstract**: LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.

摘要: 基于LLM的代码代理（例如，ChatGPT Codex）越来越多地被部署为代码审查和安全审计任务的检测器。尽管CoT增强型LLM漏洞检测器被认为可以针对混淆的恶意代码提供更好的鲁棒性，但我们发现它们的推理链和语义抽象过程表现出可利用的系统弱点。这使得攻击者能够秘密嵌入恶意逻辑、绕过代码审查并在整个现实世界的软件供应链中传播后门组件。为了研究这个问题，我们提出了CoTDeceptor，第一个针对CoT增强型LLM检测器的对抗代码混淆框架。CoTDeceptor自主构建不断发展的、难以逆转的多阶段混淆策略链，有效扰乱CoT驱动的检测逻辑。我们获得了安全企业提供的恶意代码，实验结果表明CoTDeceptor针对最先进的LLM和漏洞检测代理实现了稳定且可转移的规避性能。CoTDeceptor绕过了15个漏洞类别中的14个，而之前的方法只绕过了2个。我们的研究结果强调了现实世界软件供应链中的潜在风险，并强调了对更强大和可解释的LLM支持的安全分析系统的需求。



## **2. Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks**

提高搜索高效硬标签攻击的射线搜索优化的收敛率 cs.LG

Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21241v1) [paper-pdf](https://arxiv.org/pdf/2512.21241v1)

**Authors**: Xinjie Xu, Shuyu Cheng, Dongwei Xu, Qi Xuan, Chen Ma

**Abstract**: In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.

摘要: 在硬标签黑匣子对抗攻击中，只能访问前1名的预测标签，令人望而却步的查询复杂性对实际部署构成了主要障碍。在本文中，我们重点优化一类代表性攻击，这些攻击搜索最佳射线方向，产生将良性图像移动到对抗区域所需的最小$\ell_2 $-norm扰动。受Nesterov加速梯度（NAG）的启发，我们提出了一种基于动量的算法ARS-OPT，该算法主动估计相对于从累积动量推断的未来射线方向的梯度。我们对其收敛行为进行了理论分析，表明ARS-OPT能够实现更准确的方向更新，并实现更快、更稳定的优化。为了进一步加速收敛，我们将代理模型先验纳入ARS-OPT的梯度估计中，从而产生性能增强的PARS-OPT。我们方法的优越性得到了标准假设下的理论保证的支持。ImageNet和CIFAR-10上的大量实验表明，我们的方法在查询效率方面超过了13种最先进的方法。



## **3. Time-Bucketed Balance Records: Bounded-Storage Ephemeral Tokens for Resource-Constrained Systems**

分时段平衡记录：资源受限系统的有界存储短暂令牌 cs.DS

14 pages, 1 figure, 1 Algorithm, 3 Theorems

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20962v1) [paper-pdf](https://arxiv.org/pdf/2512.20962v1)

**Authors**: Shaun Scovil, Bhargav Chickmagalur Nanjundappa

**Abstract**: Fungible tokens with time-to-live (TTL) semantics require tracking individual expiration times for each deposited unit. A naive implementation creates a new balance record per deposit, leading to unbounded storage growth and vulnerability to denial-of-service attacks. We present time-bucketed balance records, a data structure that bounds storage to O(k) records per account while guaranteeing that tokens never expire before their configured TTL. Our approach discretizes time into k buckets, coalescing deposits within the same bucket to limit unique expiration timestamps. We prove three key properties: (1) storage is bounded by k+1 records regardless of deposit frequency, (2) actual expiration time is always at least the configured TTL, and (3) adversaries cannot increase a victim's operation cost beyond O(k)[amortized] worst case. We provide a reference implementation in Solidity with measured gas costs demonstrating practical efficiency.

摘要: 具有生存时间（TLR）语义的可替代代币需要跟踪每个存入单位的单独到期时间。天真的实施会为每次存款创建新的余额记录，从而导致存储无限增长并容易受到拒绝服务攻击。我们提供分时段的余额记录，这是一种数据结构，将每个帐户的存储限制为O（k）个记录，同时保证令牌不会在其配置的TLR之前到期。我们的方法将时间离散化到k个桶中，将存款合并在同一桶中以限制唯一的到期时间戳。我们证明了三个关键属性：（1）无论存款频率如何，存储都以k+1条记录为界限，（2）实际到期时间始终至少为配置的TLR，以及（3）对手不能将受害者的操作成本增加到O（k）以上[摊销]最坏情况。我们在Solidity中提供了一个参考实施，其中测量的天然气成本证明了实际效率。



## **4. Robustness Certificates for Neural Networks against Adversarial Attacks**

神经网络抗对抗性攻击的鲁棒性证明 cs.LG

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20865v1) [paper-pdf](https://arxiv.org/pdf/2512.20865v1)

**Authors**: Sara Taheri, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Majid Zamani

**Abstract**: The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level.

摘要: 机器学习在安全关键领域的使用越来越多，放大了对抗威胁的风险，特别是破坏训练数据以降低性能或引发不安全行为的数据中毒攻击。大多数现有的防御缺乏正式保证或依赖于有关模型类别、攻击类型、中毒程度或逐点认证的限制性假设，从而限制了其实际可靠性。本文介绍了一个有原则的正式鲁棒性认证框架，该框架将基于梯度的训练建模为离散时间动态系统（dt-DS），并将中毒鲁棒性制定为正式安全验证问题。通过改编来自控制理论的屏障证书（BC）概念，我们引入了充分条件来证明稳健半径，以确保终端模型在最坏情况下${\ell}_p$-norm基于中毒的情况下保持安全。为了实现这一点，我们将BC参数化为在有限组中毒轨迹上训练的神经网络。我们进一步通过求解场景凸规划（SCP）来推导出可能大致正确（PAC）界限，这会产生扩展到训练集之外的认证稳健性半径的置信下限。重要的是，我们的框架还扩展到针对测试时攻击的认证，使其成为第一个在训练和测试时攻击环境中提供正式保证的统一框架。MNIST、SVHN和CIFAR-10上的实验表明，我们的方法可以证明非平凡的扰动预算，同时是模型不可知的，并且不需要攻击或污染水平的先验知识。



## **5. Defending against adversarial attacks using mixture of experts**

使用混合专家抵御对抗攻击 cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20821v1) [paper-pdf](https://arxiv.org/pdf/2512.20821v1)

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far

**Abstract**: Machine learning is a powerful tool enabling full automation of a huge number of tasks without explicit programming. Despite recent progress of machine learning in different domains, these models have shown vulnerabilities when they are exposed to adversarial threats. Adversarial threats aim to hinder the machine learning models from satisfying their objectives. They can create adversarial perturbations, which are imperceptible to humans' eyes but have the ability to cause misclassification during inference. Moreover, they can poison the training data to harm the model's performance or they can query the model to steal its sensitive information. In this paper, we propose a defense system, which devises an adversarial training module within mixture-of-experts architecture to enhance its robustness against adversarial threats. In our proposed defense system, we use nine pre-trained experts with ResNet-18 as their backbone. During end-to-end training, the parameters of expert models and gating mechanism are jointly updated allowing further optimization of the experts. Our proposed defense system outperforms state-of-the-art defense systems and plain classifiers, which use a more complex architecture than our model's backbone.

摘要: 机器学习是一种强大的工具，无需显式编程即可实现大量任务的完全自动化。尽管机器学习最近在不同领域取得了进展，但这些模型在面临对抗威胁时仍表现出脆弱性。对抗性威胁旨在阻碍机器学习模型实现其目标。它们可以产生对抗性扰动，人类肉眼无法察觉，但有能力在推理过程中导致错误分类。此外，他们可以毒害训练数据以损害模型的性能，或者他们可以查询模型以窃取其敏感信息。在本文中，我们提出了一种防御系统，该系统在混合专家架构中设计了一个对抗训练模块，以增强其对对抗威胁的鲁棒性。在我们提出的防御系统中，我们使用九名经过预先培训的专家，以ResNet-18为骨干。在端到端训练过程中，专家模型和门控机制的参数联合更新，从而进一步优化专家。我们提出的防御系统优于最先进的防御系统和普通分类器，后者使用比我们模型的主干更复杂的架构。



## **6. Safety Alignment of LMs via Non-cooperative Games**

通过非合作博弈实现LM的安全调整 cs.AI

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20806v1) [paper-pdf](https://arxiv.org/pdf/2512.20806v1)

**Authors**: Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov

**Abstract**: Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.

摘要: 确保语言模型（LM）的安全性同时保持其有用性仍然是人工智能协调的一个关键挑战。当前的方法依赖于顺序对抗训练：生成对抗提示并微调LM以抵御它们。我们引入了一种不同的范式：将安全对齐框架为攻击者LM和防御者LM之间通过在线强化学习联合训练的非零和游戏。每个LM都不断适应对方不断发展的策略，推动迭代改进。我们的方法使用基于偏好的奖励信号，而不是逐点比较，提供更强大的监督，并可能减少奖励黑客。我们的RL配方AdvGame改变了安全性和实用性的帕累托边界，产生了一个防御者LM，同时对对抗性攻击更有帮助，更有弹性。此外，由此产生的攻击者LM收敛到一个强大的，通用的红队代理，可以直接部署到探测任意目标模型。



## **7. Real-World Adversarial Attacks on RF-Based Drone Detectors**

现实世界对基于射频的无人机探测器的对抗攻击 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20712v1) [paper-pdf](https://arxiv.org/pdf/2512.20712v1)

**Authors**: Omer Gazit, Yael Itzhakev, Yuval Elovici, Asaf Shabtai

**Abstract**: Radio frequency (RF) based systems are increasingly used to detect drones by analyzing their RF signal patterns, converting them into spectrogram images which are processed by object detection models. Existing RF attacks against image based models alter digital features, making over-the-air (OTA) implementation difficult due to the challenge of converting digital perturbations to transmittable waveforms that may introduce synchronization errors and interference, and encounter hardware limitations. We present the first physical attack on RF image based drone detectors, optimizing class-specific universal complex baseband (I/Q) perturbation waveforms that are transmitted alongside legitimate communications. We evaluated the attack using RF recordings and OTA experiments with four types of drones. Our results show that modest, structured I/Q perturbations are compatible with standard RF chains and reliably reduce target drone detection while preserving detection of legitimate drones.

摘要: 基于射频（RF）的系统越来越多地用于通过分析无人机的RF信号模式来检测无人机，将其转换为由对象检测模型处理的谱图图像。现有的针对基于图像的模型的RF攻击会改变数字特征，使空中（OTA）实施变得困难，因为将数字扰动转换为可传输的波型具有挑战性，这可能会引入同步错误和干扰，并遇到硬件限制。我们首次对基于RF图像的无人机检测器进行物理攻击，优化与合法通信一起传输的特定类别通用复基带（I/Q）扰动波。我们使用四种无人机的射频记录和OTA实验评估了这次攻击。我们的结果表明，适度的结构化I/Q扰动与标准RF链兼容，并可靠地减少目标无人机检测，同时保留对合法无人机的检测。



## **8. Evasion-Resilient Detection of DNS-over-HTTPS Data Exfiltration: A Practical Evaluation and Toolkit**

DNS over-HTTPS数据泄露的规避检测：实用评估和工具包 cs.CR

61 pages Advisor : Dr Darren Hurley-Smith

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20423v1) [paper-pdf](https://arxiv.org/pdf/2512.20423v1)

**Authors**: Adam Elaoumari

**Abstract**: The purpose of this project is to assess how well defenders can detect DNS-over-HTTPS (DoH) file exfiltration, and which evasion strategies can be used by attackers. While providing a reproducible toolkit to generate, intercept and analyze DoH exfiltration, and comparing Machine Learning vs threshold-based detection under adversarial scenarios. The originality of this project is the introduction of an end-to-end, containerized pipeline that generates configurable file exfiltration over DoH using several parameters (e.g., chunking, encoding, padding, resolver rotation). It allows for file reconstruction at the resolver side, while extracting flow-level features using a fork of DoHLyzer. The pipeline contains a prediction side, which allows the training of machine learning models based on public labelled datasets and then evaluates them side-by-side with threshold-based detection methods against malicious and evasive DNS-Over-HTTPS traffic. We train Random Forest, Gradient Boosting and Logistic Regression classifiers on a public DoH dataset and benchmark them against evasive DoH exfiltration scenarios. The toolkit orchestrates traffic generation, file capture, feature extraction, model training and analysis. The toolkit is then encapsulated into several Docker containers for easy setup and full reproducibility regardless of the platform it is run on. Future research regarding this project is directed at validating the results on mixed enterprise traffic, extending the protocol coverage to HTTP/3/QUIC request, adding a benign traffic generation, and working on real-time traffic evaluation. A key objective is to quantify when stealth constraints make DoH exfiltration uneconomical and unworthy for the attacker.

摘要: 该项目的目的是评估防御者如何检测DNS over HTTPS（DoH）文件泄露，以及攻击者可以使用哪些规避策略。同时提供一个可复制的工具包来生成，拦截和分析DoH渗出，并在对抗场景下比较机器学习与基于阈值的检测。该项目的独创性在于引入了一个端到端的容器化管道，该管道使用多个参数（例如，分块、编码、填充、解析器旋转）。它允许在解析器端进行文件重建，同时使用DoHLyzer的分叉提取流级特征。该管道包含一个预测端，它允许基于公共标签数据集训练机器学习模型，然后使用基于阈值的检测方法针对恶意和规避性DNS-Over-HTTPS流量并行评估它们。我们在公共DoH数据集上训练随机森林、梯度增强和逻辑回归分类器，并针对规避DoH外流场景对它们进行基准测试。该工具包协调流量生成、文件捕获、特征提取、模型训练和分析。然后，该工具包被封装到多个Docker容器中，无论其运行在什么平台上，都可以轻松设置和完全可重复性。有关该项目的未来研究旨在验证混合企业流量的结果，将协议覆盖范围扩展到HTTP/3/QUIC请求，添加良性流量生成，并进行实时流量评估。一个关键目标是量化何时隐形限制使DoH撤离对攻击者来说不经济且不值得。



## **9. Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms**

机器人群中无通信协作避碰的基于应急模型的控制（MC） math.OC

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20391v1) [paper-pdf](https://arxiv.org/pdf/2512.20391v1)

**Authors**: Georg Schildbach

**Abstract**: Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.

摘要: 群操作中机器人之间的协作避免碰撞仍然是一个悬而未决的挑战。假设采用分散式架构，每个机器人负责做出自己的控制决策，包括运动规划。为此，大多数现有的方法大多依赖于群体代理之间某种形式的（无线）通信。然而，事实上，沟通是脆弱的。它可能会受到延迟、进一步延迟和数据包丢失、传输故障的影响，并容易受到对抗攻击，例如干扰或欺骗。本文提出了基于应急模型的控制（SMC）作为一种无通信替代方案。它遵循隐性合作范式，在该范式下，机器人的设计基于共识（离线）规则，类似于交通规则。它们包括定义每个机器人的应急轨迹，以及构建相互碰撞避免约束的方法。该设置是为了保证递归的可行性和所有群体成员之间的碰撞避免在闭环操作。此外，CMC自然满足即插即用范例，即，新的机器人进入蜂群。两个数值例子表明，避免碰撞的保证是完整的，机器人群体下CMC政权顺利运作。



## **10. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

乐观的TEE-Rollops：区块链上可扩展和可验证的生成式人工智能推理的混合架构 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.

摘要: 大型语言模型（LLM）快速集成到去中心化物理基础设施网络（DePin）中目前受到可验证性三困境的限制，该困境认为去中心化推理系统无法同时实现高计算完整性、低延迟和低成本。现有的加密解决方案，例如零知识机器学习（ZKML），存在超线性证明费用（O（k NlogN））的问题，这使得它们对于十亿个参数模型来说不可行。相反，乐观方法（opML）施加了禁止性的争议窗口，阻止了实时交互，而最近的“质量证明”（PoQ）范式则牺牲了加密完整性来进行主观语义评估，使网络容易受到模型降级攻击和奖励黑客攻击。在本文中，我们介绍了乐观TEE-Rollup（OTR），这是一种协调这些约束的混合验证协议。OTR利用NVIDIA H100机密计算可信执行环境（TEE）提供亚秒级临时最终结果，并以乐观的防欺诈机制和随机零知识抽查为基础，以减轻硬件侧通道风险。我们正式定义了有效归因证明（PoEA），这是一种共识机制，通过加密方式将执行跟踪与硬件证明绑定，从而保证模型的真实性。广泛的模拟表明，OTR实现了集中式基线99%的吞吐量，每次查询的边际成本费用为0.07美元，即使存在暂时性硬件漏洞，也能对理性对手保持拜占庭式的耐药性。



## **11. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

Odysseus：通过双重隐写术破解商业多模式法学硕士集成系统 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.

摘要: 通过将语言理解与图像等感知模式集成起来，多模式大型语言模型（MLLM）构成了现代人工智能系统的重要基础，特别是在开放和交互环境中运行的智能代理。然而，它们的可访问性不断增加也增加了滥用风险，例如产生有害或不安全内容。为了减轻这些风险，通常应用对齐技术来将模型行为与人类价值观对齐。尽管做出了这些努力，最近的研究表明，越狱攻击可能会绕过对齐并引发不安全的输出。目前，大多数现有的越狱方法都是针对开源模型量身定制的，并且对于商业MLLM集成系统（通常使用额外的过滤器）的有效性有限。这些过滤器可以检测和防止恶意输入和输出内容，从而显着减少越狱威胁。在本文中，我们揭示了这些安全过滤器的成功在很大程度上依赖于一个关键假设，即恶意内容必须在输入或输出中显式可见。这种假设虽然通常适用于传统的LLM集成系统，但在MLLM集成系统中却出现了问题，攻击者可以利用多种模式来隐藏对抗意图，从而导致现有的MLLM集成系统中出现错误的安全感。为了挑战这一假设，我们提出了Odysseus，一种新的越狱范例，它引入了双重隐写术来秘密地将恶意查询和响应嵌入到看起来很好的图像中。在基准数据集上进行的大量实验表明，我们的Odysseus成功地越狱了几个开创性和现实的MLLM集成系统，攻击成功率高达99%。它暴露了现有防御中的一个基本盲点，并呼吁重新思考MLLM集成系统中的跨模式安全性。



## **12. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

超越核心领域的人工智能安全：简历筛选作为专业LLM应用中对抗漏洞的案例研究 cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.

摘要: 大型语言模型（LLM）擅长文本理解和生成，非常适合代码审查和内容审核等自动化任务。然而，我们的研究发现了一个漏洞：LLM可能会被隐藏在输入数据（例如简历或代码）中的“对抗指令”操纵，导致它们偏离预期任务。值得注意的是，虽然代码审查等成熟领域可能存在防御措施，但在简历筛选和同行审查等其他常见应用中通常不存在防御措施。本文引入了一个基准来评估简历筛选中的此漏洞，揭示了某些攻击类型的攻击成功率超过80%。我们评估了两种防御机制：基于预算的防御可以减少10.1%的攻击，错误拒绝增加12.5%，而我们提出的使用LoRA适应的FIDS（通过分离的外部指令检测）可以减少15.4%的攻击，错误拒绝增加10.4%。组合方法可减少26.3%的攻击，证明训练时防御在安全性和实用程序保存方面优于推理时缓解措施。



## **13. IoT-based Android Malware Detection Using Graph Neural Network With Adversarial Defense**

使用具有对抗性防御的图神经网络进行基于物联网的Android恶意软件检测 cs.CR

13 pages

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20004v1) [paper-pdf](https://arxiv.org/pdf/2512.20004v1)

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang

**Abstract**: Since the Internet of Things (IoT) is widely adopted using Android applications, detecting malicious Android apps is essential. In recent years, Android graph-based deep learning research has proposed many approaches to extract relationships from applications as graphs to generate graph embeddings. First, we demonstrate the effectiveness of graph-based classification using a Graph Neural Network (GNN)-based classifier to generate API graph embeddings. The graph embeddings are combined with Permission and Intent features to train multiple machine learning and deep learning models for Android malware detection. The proposed classification approach achieves an accuracy of 98.33 percent on the CICMaldroid dataset and 98.68 percent on the Drebin dataset. However, graph-based deep learning models are vulnerable, as attackers can add fake relationships to evade detection by the classifier. Second, we propose a Generative Adversarial Network (GAN)-based attack algorithm named VGAE-MalGAN targeting graph-based GNN Android malware classifiers. The VGAE-MalGAN generator produces adversarial malware API graphs, while the VGAE-MalGAN substitute detector attempts to mimic the target detector. Experimental results show that VGAE-MalGAN can significantly reduce the detection rate of GNN-based malware classifiers. Although the model initially fails to detect adversarial malware, retraining with generated adversarial samples improves robustness and helps mitigate adversarial attacks.

摘要: 由于物联网（IoT）广泛采用Android应用程序，因此检测恶意Android应用程序至关重要。近年来，基于Android图形的深度学习研究提出了许多方法，从应用程序中提取关系作为图形来生成图形嵌入。首先，我们使用基于图神经网络（GNN）的分类器来证明基于图的分类生成API图嵌入的有效性。图嵌入与权限和意图功能相结合，可以训练多个机器学习和深度学习模型来进行Android恶意软件检测。所提出的分类方法在CICMaldroid数据集上实现了98.33%的准确率，在Drebin数据集上实现了98.68%的准确率。然而，基于图的深度学习模型是脆弱的，因为攻击者可以添加虚假的关系来逃避分类器的检测。其次，我们提出了一种基于生成对抗网络（GAN）的攻击算法VGAE-MalGAN，目标是基于图的GNN Android恶意软件分类器。VGAE-MalGAN生成器生成对抗性恶意软件API图，而VGAE-MalGAN替代检测器尝试模仿目标检测器。实验结果表明，VGAE-MalGAN可以显着降低基于GNN的恶意软件分类器的检测率。尽管该模型最初未能检测到对抗性恶意软件，但使用生成的对抗性样本进行重新训练可以提高鲁棒性并有助于减轻对抗性攻击。



## **14. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

宏观经济压力下金融机器学习中的条件对抗脆弱性 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.

摘要: 金融决策系统中使用的机器学习模型在非平稳经济环境中运行，但对抗稳健性通常是在静态假设下评估的。这项工作引入了条件对抗脆弱性，这是一种依赖政权的现象，其中对抗脆弱性在宏观经济压力时期被系统性放大。我们提出了一个用于时间索引表格财务分类任务的制度意识评估框架，该框架以经济压力的外部指标为条件进行稳健性评估。使用基于波动性的制度分割作为宏观经济状况的代理，我们评估平静和压力时期的模型行为，同时保持模型架构、攻击方法和评估协议不变。不同制度之间的基线预测性能保持可比性，这表明经济压力本身不会导致固有的性能下降。然而，在对抗性扰动下，在压力制度下运行的模型在预测准确性、操作决策阈值和风险敏感结果方面表现出明显更大的退化。我们进一步证明，这种放大会传播到假阴性率增加，从而增加了在不利条件下错过高风险病例的风险。为了补充数字稳健性指标，我们引入了一个基于使用大型语言模型对模型解释进行语义审计的解释治理层。总而言之，这些结果表明，金融机器学习中的对抗稳健性是一种依赖于制度的属性，并激励压力感知方法对高风险金融部署中的风险评估进行建模。



## **15. Multi-Layer Confidence Scoring for Detection of Out-of-Distribution Samples, Adversarial Attacks, and In-Distribution Misclassifications**

用于检测分布外样本、对抗性攻击和分布内错误分类的多层置信度评分 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19472v1) [paper-pdf](https://arxiv.org/pdf/2512.19472v1)

**Authors**: Lorenzo Capelli, Leandro de Souza Rosa, Gianluca Setti, Mauro Mangia, Riccardo Rovatti

**Abstract**: The recent explosive growth in Deep Neural Networks applications raises concerns about the black-box usage of such models, with limited trasparency and trustworthiness in high-stakes domains, which have been crystallized as regulatory requirements such as the European Union Artificial Intelligence Act. While models with embedded confidence metrics have been proposed, such approaches cannot be applied to already existing models without retraining, limiting their broad application. On the other hand, post-hoc methods, which evaluate pre-trained models, focus on solving problems related to improving the confidence in the model's predictions, and detecting Out-Of-Distribution or Adversarial Attacks samples as independent applications. To tackle the limited applicability of already existing methods, we introduce Multi-Layer Analysis for Confidence Scoring (MACS), a unified post-hoc framework that analyzes intermediate activations to produce classification-maps. From the classification-maps, we derive a score applicable for confidence estimation, detecting distributional shifts and adversarial attacks, unifying the three problems in a common framework, and achieving performances that surpass the state-of-the-art approaches in our experiments with the VGG16 and ViTb16 models with a fraction of their computational overhead.

摘要: 深度神经网络应用程序最近的爆炸式增长引发了人们对此类模型黑匣子使用的担忧，因为在高风险领域的传输性和可信度有限，而这些已被具体化为欧盟人工智能法案等监管要求。虽然已经提出了具有嵌入置信指标的模型，但如果不进行重新培训，此类方法就无法应用于现有的模型，从而限制了其广泛应用。另一方面，评估预训练模型的事后方法专注于解决与提高模型预测的置信度相关的问题，并将分布外或对抗性攻击样本作为独立应用程序检测。为了解决现有方法的有限适用性问题，我们引入了置信度评分多层分析（MACS），这是一个统一的事后框架，可以分析中间激活以生成分类图。从分类图中，我们推导出适用于置信度估计、检测分布变化和对抗性攻击、将这三个问题统一在一个通用框架中，并在我们的实验中实现超越最先进方法的性能VGG 16和ViTb 16模型，其计算费用很小。



## **16. SafeMed-R1: Adversarial Reinforcement Learning for Generalizable and Robust Medical Reasoning in Vision-Language Models**

SafeMed-R1：对抗强化学习，用于视觉语言模型中的可概括和稳健的医学推理 cs.AI

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19317v1) [paper-pdf](https://arxiv.org/pdf/2512.19317v1)

**Authors**: A. A. Gde Yogi Pramana, Jason Ray, Anthony Jaya, Michael Wijaya

**Abstract**: Vision--Language Models (VLMs) show significant promise for Medical Visual Question Answering (VQA), yet their deployment in clinical settings is hindered by severe vulnerability to adversarial attacks. Standard adversarial training, while effective for simpler tasks, often degrades both generalization performance and the quality of generated clinical reasoning. We introduce SafeMed-R1, a hybrid defense framework that ensures robust performance while preserving high-quality, interpretable medical reasoning. SafeMed-R1 employs a two-stage approach: at training time, we integrate Adversarial Training with Group Relative Policy Optimization (AT-GRPO) to explicitly robustify the reasoning process against worst-case perturbations; at inference time, we augment the model with Randomized Smoothing to provide certified $L_2$-norm robustness guarantees. We evaluate SafeMed-R1 on the OmniMedVQA benchmark across eight medical imaging modalities comprising over 88,000 samples. Our experiments reveal that standard fine-tuned VLMs, despite achieving 95\% accuracy on clean inputs, collapse to approximately 25\% under PGD attacks. In contrast, SafeMed-R1 maintains 84.45\% accuracy under the same adversarial conditions, representing a 59 percentage point improvement in robustness. Furthermore, we demonstrate that models trained with explicit chain-of-thought reasoning exhibit superior adversarial robustness compared to instruction-only variants, suggesting a synergy between interpretability and security in medical AI systems.

摘要: 视觉-语言模型（VLM）在医学视觉问题解答（VQA）方面表现出了巨大的前景，但它们在临床环境中的部署因严重容易受到对抗性攻击而受到阻碍。标准对抗训练虽然对简单的任务有效，但通常会降低概括性能和生成的临床推理的质量。我们引入SafeMed-R1，这是一种混合防御框架，可确保稳健的性能，同时保留高质量、可解释的医学推理。SafeMed-R1采用两阶段方法：在训练时，我们将对抗性训练与群体相对政策优化（AT-GRPO）集成，以显式地针对最坏情况的扰动鲁棒性推理过程;在推理时，我们使用随机平滑来增强模型，以提供经过认证的$L_2$规范鲁棒性保证。我们在OmniMedVQA基准上评估SafeMed-R1，涵盖八种医学成像模式，包括超过88，000个样本。我们的实验表明，尽管标准微调的VLM在干净的输入上实现了95%的准确率，但在PVD攻击下却崩溃到约25%。相比之下，SafeMed-R1在相同对抗条件下保持84.45%的准确性，鲁棒性提高了59个百分点。此外，我们证明，与仅描述的变体相比，用显式思维链推理训练的模型表现出更好的对抗鲁棒性，这表明医疗人工智能系统中的可解释性和安全性之间存在协同作用。



## **17. GShield: Mitigating Poisoning Attacks in Federated Learning**

GShield：减轻联邦学习中的中毒攻击 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19286v1) [paper-pdf](https://arxiv.org/pdf/2512.19286v1)

**Authors**: Sameera K. M., Serena Nicolazzo, Antonino Nocera, Vinod P., Rafidha Rehiman K. A

**Abstract**: Federated Learning (FL) has recently emerged as a revolutionary approach to collaborative training Machine Learning models. In particular, it enables decentralized model training while preserving data privacy, but its distributed nature makes it highly vulnerable to a severe attack known as Data Poisoning. In such scenarios, malicious clients inject manipulated data into the training process, thereby degrading global model performance or causing targeted misclassification. In this paper, we present a novel defense mechanism called GShield, designed to detect and mitigate malicious and low-quality updates, especially under non-independent and identically distributed (non-IID) data scenarios. GShield operates by learning the distribution of benign gradients through clustering and Gaussian modeling during an initial round, enabling it to establish a reliable baseline of trusted client behavior. With this benign profile, GShield selectively aggregates only those updates that align with the expected gradient patterns, effectively isolating adversarial clients and preserving the integrity of the global model. An extensive experimental campaign demonstrates that our proposed defense significantly improves model robustness compared to the state-of-the-art methods while maintaining a high accuracy of performance across both tabular and image datasets. Furthermore, GShield improves the accuracy of the targeted class by 43\% to 65\% after detecting malicious and low-quality clients.

摘要: 联合学习（FL）最近成为协作训练机器学习模型的革命性方法。特别是，它能够实现去中心化模型训练，同时保护数据隐私，但其分布式性质使其极易受到称为数据中毒的严重攻击。在此类情况下，恶意客户端将操纵数据注入到训练过程中，从而降低全局模型性能或导致有针对性的错误分类。在本文中，我们提出了一种名为GShield的新型防御机制，旨在检测和减轻恶意和低质量更新，特别是在非独立和同分布（非IID）数据场景下。GShield通过在初始一轮期间通过集群和高斯建模学习良性梯度的分布来运作，使其能够建立可信客户行为的可靠基线。通过这种良性配置文件，GShield选择性地仅聚合那些与预期梯度模式一致的更新，从而有效地隔离敌对客户并保持全球模型的完整性。一项广泛的实验活动表明，与最先进的方法相比，我们提出的防御显着提高了模型的稳健性，同时在表格和图像数据集中保持了高准确性的性能。此外，GShield在检测到恶意和低质量客户端后将目标类的准确性提高了43%至65%。



## **18. Elevating Intrusion Detection and Security Fortification in Intelligent Networks through Cutting-Edge Machine Learning Paradigms**

通过尖端机器学习范式提升智能网络中的入侵检测和安全防御 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19037v1) [paper-pdf](https://arxiv.org/pdf/2512.19037v1)

**Authors**: Md Minhazul Islam Munna, Md Mahbubur Rahman, Jaroslav Frnda, Muhammad Shahid Anwar, Alpamis Kutlimuratov

**Abstract**: The proliferation of IoT devices and their reliance on Wi-Fi networks have introduced significant security vulnerabilities, particularly the KRACK and Kr00k attacks, which exploit weaknesses in WPA2 encryption to intercept and manipulate sensitive data. Traditional IDS using classifiers face challenges such as model overfitting, incomplete feature extraction, and high false positive rates, limiting their effectiveness in real-world deployments. To address these challenges, this study proposes a robust multiclass machine learning based intrusion detection framework. The methodology integrates advanced feature selection techniques to identify critical attributes, mitigating redundancy and enhancing detection accuracy. Two distinct ML architectures are implemented: a baseline classifier pipeline and a stacked ensemble model combining noise injection, Principal Component Analysis (PCA), and meta learning to improve generalization and reduce false positives. Evaluated on the AWID3 data set, the proposed ensemble architecture achieves superior performance, with an accuracy of 98%, precision of 98%, recall of 98%, and a false positive rate of just 2%, outperforming existing state-of-the-art methods. This work demonstrates the efficacy of combining preprocessing strategies with ensemble learning to fortify network security against sophisticated Wi-Fi attacks, offering a scalable and reliable solution for IoT environments. Future directions include real-time deployment and adversarial resilience testing to further enhance the model's adaptability.

摘要: 物联网设备的激增及其对Wi-Fi网络的依赖引入了重大的安全漏洞，特别是KRACK和Kr 00 k攻击，它们利用WPA 2加密的弱点来拦截和操纵敏感数据。使用分类器的传统IDS面临着模型过匹配、不完整的特征提取和高误报率等挑战，限制了它们在现实世界部署中的有效性。为了应对这些挑战，本研究提出了一个基于鲁棒的多类机器学习的入侵检测框架。该方法集成了先进的特征选择技术来识别关键属性，减少冗余并提高检测准确性。实现了两种不同的ML架构：基线分类器管道和堆叠集成模型，该模型结合了噪音注入、主成分分析（PCA）和Meta学习，以提高概括性并减少误报。在AWID 3数据集上进行评估后，提出的集成架构实现了卓越的性能，准确率为98%，精确度为98%，召回率为98%，误报率仅为2%，优于现有的最先进方法。这项工作展示了将预处理策略与集成学习相结合以增强网络安全性以抵御复杂Wi-Fi攻击的功效，为物联网环境提供可扩展且可靠的解决方案。未来的方向包括实时部署和对抗韧性测试，以进一步增强模型的适应性。



## **19. DREAM: Dynamic Red-teaming across Environments for AI Models**

DREAM：人工智能模型跨环境的动态红色团队 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19016v1) [paper-pdf](https://arxiv.org/pdf/2512.19016v1)

**Authors**: Liming Lu, Xiang Gu, Junyu Huang, Jiawei Du, Yunhuai Liu, Yongbin Zhou, Shuchao Pang

**Abstract**: Large Language Models (LLMs) are increasingly used in agentic systems, where their interactions with diverse tools and environments create complex, multi-stage safety challenges. However, existing benchmarks mostly rely on static, single-turn assessments that miss vulnerabilities from adaptive, long-chain attacks. To fill this gap, we introduce DREAM, a framework for systematic evaluation of LLM agents against dynamic, multi-stage attacks. At its core, DREAM uses a Cross-Environment Adversarial Knowledge Graph (CE-AKG) to maintain stateful, cross-domain understanding of vulnerabilities. This graph guides a Contextualized Guided Policy Search (C-GPS) algorithm that dynamically constructs attack chains from a knowledge base of 1,986 atomic actions across 349 distinct digital environments. Our evaluation of 12 leading LLM agents reveals a critical vulnerability: these attack chains succeed in over 70% of cases for most models, showing the power of stateful, cross-environment exploits. Through analysis of these failures, we identify two key weaknesses in current agents: contextual fragility, where safety behaviors fail to transfer across environments, and an inability to track long-term malicious intent. Our findings also show that traditional safety measures, such as initial defense prompts, are largely ineffective against attacks that build context over multiple interactions. To advance agent safety research, we release DREAM as a tool for evaluating vulnerabilities and developing more robust defenses.

摘要: 大型语言模型（LLM）越来越多地用于代理系统，它们与不同工具和环境的交互会带来复杂、多阶段的安全挑战。然而，现有的基准大多依赖于静态、单轮评估，这些评估会错过自适应性、长链攻击的漏洞。为了填补这一空白，我们引入了DREAM，这是一个针对动态、多阶段攻击系统评估LLM代理的框架。DREAM的核心是使用跨环境对抗知识图（CE-AKG）来维护对漏洞的有状态、跨领域理解。该图指导上下文引导政策搜索（C-GPS）算法，该算法根据349个不同数字环境中1，986个原子动作的知识库动态构建攻击链。我们对12个领先的LLM代理的评估揭示了一个关键漏洞：对于大多数模型来说，这些攻击链在超过70%的情况下都取得了成功，展示了有状态、跨环境漏洞利用的力量。通过对这些失败的分析，我们发现了当前代理的两个关键弱点：上下文脆弱性，即安全行为无法跨环境转移，以及无法跟踪长期恶意意图。我们的研究结果还表明，传统的安全措施（例如初始防御提示）对于在多重交互中建立上下文的攻击在很大程度上无效。为了推进代理安全研究，我们发布DREAM作为评估漏洞和开发更强大防御的工具。



## **20. Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System**

用于大型语言模型安全评估的自动化Red-Teaming框架：全面的攻击生成和检测系统 cs.CR

18 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.20677v1) [paper-pdf](https://arxiv.org/pdf/2512.20677v1)

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）越来越多地部署在高风险领域，确保它们的安全性和一致性已成为一项关键挑战。现有的红色团队实践严重依赖手动测试，这限制了可扩展性，并且无法全面覆盖潜在对抗行为的巨大空间。本文介绍了一个自动化红色团队框架，该框架系统地生成、执行和评估对抗提示，以发现LLC中的安全漏洞。我们的框架集成了基于元提示的攻击合成、多模式漏洞检测和跨越六个主要威胁类别的标准化评估协议-奖励黑客攻击、欺骗性对齐、数据泄露、沙袋、不当工具使用和思想链操纵。GPT-OSS-20 B模型上的实验揭示了47个不同的漏洞，包括21个高严重性和12个新型攻击模式，与手动专家测试相比，漏洞发现率提高了3.9倍，同时保持89%的检测准确率。这些结果证明了该框架在实现可扩展、系统和可重复的人工智能安全评估方面的有效性。通过为提高对齐稳健性提供可操作的见解，这项工作推进了LLM自动化红色团队的状态，并有助于构建安全且值得信赖的人工智能系统的更广泛目标。



## **21. ISADM: An Integrated STRIDE, ATT&CK, and D3FEND Model for Threat Modeling Against Real-world Adversaries**

ISADM：集成的WRIDE、ITT & CK和D3 FEND模型，用于针对现实世界对手的威胁建模 cs.CR

34 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18751v1) [paper-pdf](https://arxiv.org/pdf/2512.18751v1)

**Authors**: Khondokar Fida Hasan, Hasibul Hossain Shajeeb, Chathura Abeydeera, Benjamin Turnbull, Matthew Warren

**Abstract**: FinTechs increasing connectivity, rapid innovation, and reliance on global digital infrastructures present significant cybersecurity challenges. Traditional cybersecurity frameworks often struggle to identify and prioritize sector-specific vulnerabilities or adapt to evolving adversary tactics, particularly in highly targeted sectors such as FinTech. To address these gaps, we propose ISADM (Integrated STRIDE-ATTACK-D3FEND Threat Model), a novel hybrid methodology applied to FinTech security that integrates STRIDE's asset-centric threat classification with MITRE ATTACK's catalog of real-world adversary behaviors and D3FEND's structured knowledge of countermeasures. ISADM employs a frequency-based scoring mechanism to quantify the prevalence of adversarial Tactics, Techniques, and Procedures (TTPs), enabling a proactive, score-driven risk assessment and prioritization framework. This proactive approach contributes to shifting organizations from reactive defense strategies toward the strategic fortification of critical assets. We validate ISADM through industry-relevant case study analyses, demonstrating how the approach replicates actual attack patterns and strengthens proactive threat modeling, guiding risk prioritization and resource allocation to the most critical vulnerabilities. Overall, ISADM offers a comprehensive hybrid threat modeling methodology that bridges asset-centric and adversary-centric analysis, providing FinTech systems with stronger defenses. The emphasis on real-world validation highlights its practical significance in enhancing the sector's cybersecurity posture through a frequency-informed, impact-aware prioritization scheme that combines empirical attacker data with contextual risk analysis.

摘要: 金融科技不断增强的连通性、快速创新以及对全球数字基础设施的依赖带来了重大的网络安全挑战。传统的网络安全框架往往难以识别和优先考虑特定行业的漏洞，或适应不断变化的对手策略，特别是在金融科技等高度针对性的行业。为了解决这些差距，我们提出了ISADM（集成CLARDE-ATTACK-D3 FEND威胁模型），这是一种应用于金融科技安全的新型混合方法，它将WRIDE的以资产为中心的威胁分类与MITRE ATTACK的现实世界对手行为目录和D3 FEND的结构化对策知识集成在一起。ISADM采用基于频率的评分机制来量化对抗性战术、技术和程序（TTP）的流行程度，从而实现积极主动的、分数驱动的风险评估和优先排序框架。这种积极主动的方法有助于组织从被动防御战略转向关键资产的战略强化。我们通过与行业相关的案例研究分析来验证ISADM，展示该方法如何复制实际的攻击模式并加强主动威胁建模，指导风险优先级和资源分配到最关键的漏洞。总体而言，ISADM提供了一种全面的混合威胁建模方法，可以弥合以资产为中心的分析和以对手为中心的分析，为金融科技系统提供更强大的防御。对现实世界验证的强调凸显了其在通过将经验攻击者数据与上下文风险分析相结合的频率信息、影响感知优先级计划来增强该行业的网络安全态势方面的实际意义。



## **22. Measuring the Impact of Student Gaming Behaviors on Learner Modeling**

衡量学生游戏行为对学习者建模的影响 cs.CY

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18659v1) [paper-pdf](https://arxiv.org/pdf/2512.18659v1)

**Authors**: Qinyi Liu, Lin Li, Valdemar Švábenský, Conrad Borchers, Mohammad Khalil

**Abstract**: The expansion of large-scale online education platforms has made vast amounts of student interaction data available for knowledge tracing (KT). KT models estimate students' concept mastery from interaction data, but their performance is sensitive to input data quality. Gaming behaviors, such as excessive hint use, may misrepresent students' knowledge and undermine model reliability. However, systematic investigations of how different types of gaming behaviors affect KT remain scarce, and existing studies rely on costly manual analysis that does not capture behavioral diversity. In this study, we conceptualize gaming behaviors as a form of data poisoning, defined as the deliberate submission of incorrect or misleading interaction data to corrupt a model's learning process. We design Data Poisoning Attacks (DPAs) to simulate diverse gaming patterns and systematically evaluate their impact on KT model performance. Moreover, drawing on advances in DPA detection, we explore unsupervised approaches to enhance the generalizability of gaming behavior detection. We find that KT models' performance tends to decrease especially in response to random guess behaviors. Our findings provide insights into the vulnerabilities of KT models and highlight the potential of adversarial methods for improving the robustness of learning analytics systems.

摘要: 大型在线教育平台的扩张使大量学生互动数据可用于知识追踪（KT）。KT模型根据交互数据估计学生的概念掌握程度，但他们的表现对输入数据质量敏感。过度使用提示等游戏行为可能会歪曲学生的知识并破坏模型的可靠性。然而，关于不同类型的游戏行为如何影响KT的系统研究仍然很少，现有的研究依赖于昂贵的手动分析，无法捕捉行为多样性。在这项研究中，我们将游戏行为概念化为一种数据中毒形式，定义为故意提交不正确或误导性的交互数据以破坏模型的学习过程。我们设计数据中毒攻击（DPA）来模拟不同的游戏模式，并系统性评估其对KT模型性能的影响。此外，利用DPA检测的进步，我们探索无监督方法来增强游戏行为检测的通用性。我们发现KT模型的性能往往会下降，尤其是在响应随机猜测行为时。我们的研究结果深入了解了KT模型的漏洞，并强调了对抗方法在提高学习分析系统稳健性方面的潜力。



## **23. Adversarial Robustness in Zero-Shot Learning:An Empirical Study on Class and Concept-Level Vulnerabilities**

零镜头学习中的对抗鲁棒性：关于类和概念级脆弱性的实证研究 cs.CV

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18651v1) [paper-pdf](https://arxiv.org/pdf/2512.18651v1)

**Authors**: Zhiyuan Peng, Zihan Ye, Shreyank N Gowda, Yuping Yan, Haotian Xu, Ling Shao

**Abstract**: Zero-shot Learning (ZSL) aims to enable image classifiers to recognize images from unseen classes that were not included during training. Unlike traditional supervised classification, ZSL typically relies on learning a mapping from visual features to predefined, human-understandable class concepts. While ZSL models promise to improve generalization and interpretability, their robustness under systematic input perturbations remain unclear. In this study, we present an empirical analysis about the robustness of existing ZSL methods at both classlevel and concept-level. Specifically, we successfully disrupted their class prediction by the well-known non-target class attack (clsA). However, in the Generalized Zero-shot Learning (GZSL) setting, we observe that the success of clsA is only at the original best-calibrated point. After the attack, the optimal bestcalibration point shifts, and ZSL models maintain relatively strong performance at other calibration points, indicating that clsA results in a spurious attack success in the GZSL. To address this, we propose the Class-Bias Enhanced Attack (CBEA), which completely eliminates GZSL accuracy across all calibrated points by enhancing the gap between seen and unseen class probabilities.Next, at concept-level attack, we introduce two novel attack modes: Class-Preserving Concept Attack (CPconA) and NonClass-Preserving Concept Attack (NCPconA). Our extensive experiments evaluate three typical ZSL models across various architectures from the past three years and reveal that ZSL models are vulnerable not only to the traditional class attack but also to concept-based attacks. These attacks allow malicious actors to easily manipulate class predictions by erasing or introducing concepts. Our findings highlight a significant performance gap between existing approaches, emphasizing the need for improved adversarial robustness in current ZSL models.

摘要: 零镜头学习（SEARCH）旨在使图像分类器能够识别来自训练期间未包含的不可见类别的图像。与传统的监督分类不同，CLARL通常依赖于学习从视觉特征到预定义的、人类可理解的类概念的映射。虽然SEARCH模型有望提高概括性和可解释性，但它们在系统输入扰动下的鲁棒性仍不清楚。在这项研究中，我们对现有CLARL方法在类级和概念级的稳健性进行了实证分析。具体来说，我们通过著名的非目标类攻击（clsA）成功地破坏了他们的类预测。然而，在广义零触发学习（GSTRL）设置中，我们观察到clsA的成功仅在原始的最佳校准点。攻击后，最佳的bestcalibration点的移动，和GALML模型保持相对较强的性能在其他校准点，表明clsA的结果在GALML虚假攻击成功。为了解决这个问题，我们提出了类偏差增强攻击（CBEA），它通过增加可见类和不可见类概率之间的差距来完全消除所有校准点上的GALML准确性。接下来，在概念级攻击中，我们引入了两种新的攻击模式：类保持概念攻击（CPconA）和非类保持概念攻击（NCPconA）。我们进行了广泛的实验评估了过去三年各种架构上的三种典型的SEARCH模型，发现SEARCH模型不仅容易受到传统的类攻击，而且容易受到基于概念的攻击。这些攻击允许恶意行为者通过删除或引入概念来轻松操纵类预测。我们的研究结果强调了现有方法之间的显着性能差距，强调了当前SEARCH模型中需要改进对抗鲁棒性。



## **24. DASH: Deception-Augmented Shared Mental Model for a Human-Machine Teaming System**

DASH：人机协作系统的欺骗增强共享心理模型 cs.HC

17 pages, 16 figures

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18616v1) [paper-pdf](https://arxiv.org/pdf/2512.18616v1)

**Authors**: Zelin Wan, Han Jun Yoon, Nithin Alluru, Terrence J. Moore, Frederica F. Nelson, Seunghyun Yoon, Hyuk Lim, Dan Dongseong Kim, Jin-Hee Cho

**Abstract**: We present DASH (Deception-Augmented Shared mental model for Human-machine teaming), a novel framework that enhances mission resilience by embedding proactive deception into Shared Mental Models (SMM). Designed for mission-critical applications such as surveillance and rescue, DASH introduces "bait tasks" to detect insider threats, e.g., compromised Unmanned Ground Vehicles (UGVs), AI agents, or human analysts, before they degrade team performance. Upon detection, tailored recovery mechanisms are activated, including UGV system reinstallation, AI model retraining, or human analyst replacement. In contrast to existing SMM approaches that neglect insider risks, DASH improves both coordination and security. Empirical evaluations across four schemes (DASH, SMM-only, no-SMM, and baseline) show that DASH sustains approximately 80% mission success under high attack rates, eight times higher than the baseline. This work contributes a practical human-AI teaming framework grounded in shared mental models, a deception-based strategy for insider threat detection, and empirical evidence of enhanced robustness under adversarial conditions. DASH establishes a foundation for secure, adaptive human-machine teaming in contested environments.

摘要: 我们提出了DASH（用于人机协作的欺骗增强共享心理模型），这是一个新颖的框架，通过将主动欺骗嵌入共享心理模型（Sim）中来增强任务弹性。DASH专为监视和救援等关键任务应用而设计，引入了“诱饵任务”来检测内部威胁，例如，在无人地面车辆（UGV）、人工智能代理或人类分析师降低团队绩效之前，会受到损害。检测到后，会激活定制的恢复机制，包括UGV系统重新安装、人工智能模型再培训或人力分析师更换。与忽视内部风险的现有SM方法相比，DASH改善了协调性和安全性。对四种方案（DASH、仅支持SM、无支持SM和基线）的经验评估表明，DASH在高攻击率下保持了约80%的任务成功率，比基线高出8倍。这项工作提供了一个基于共享心理模型的实用人类与人工智能协作框架、基于欺骗的内部威胁检测策略以及对抗条件下鲁棒性增强的经验证据。DASH为在有争议的环境中安全、自适应的人机协作奠定了基础。



## **25. Detection of AI Generated Images Using Combined Uncertainty Measures and Particle Swarm Optimised Rejection Mechanism**

使用组合的不确定性指标和粒子群优化拒绝机制检测人工智能生成的图像 cs.CV

Scientific Reports (2025)

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18527v1) [paper-pdf](https://arxiv.org/pdf/2512.18527v1)

**Authors**: Rahul Yumlembam, Biju Issac, Nauman Aslam, Eaby Kollonoor Babu, Josh Collyer, Fraser Kennedy

**Abstract**: As AI-generated images become increasingly photorealistic, distinguishing them from natural images poses a growing challenge. This paper presents a robust detection framework that leverages multiple uncertainty measures to decide whether to trust or reject a model's predictions. We focus on three complementary techniques: Fisher Information, which captures the sensitivity of model parameters to input variations; entropy-based uncertainty from Monte Carlo Dropout, which reflects predictive variability; and predictive variance from a Deep Kernel Learning framework using a Gaussian Process classifier. To integrate these diverse uncertainty signals, Particle Swarm Optimisation is used to learn optimal weightings and determine an adaptive rejection threshold. The model is trained on Stable Diffusion-generated images and evaluated on GLIDE, VQDM, Midjourney, BigGAN, and StyleGAN3, each introducing significant distribution shifts. While standard metrics such as prediction probability and Fisher-based measures perform well in distribution, their effectiveness degrades under shift. In contrast, the Combined Uncertainty measure consistently achieves an incorrect rejection rate of approximately 70 percent on unseen generators, successfully filtering most misclassified AI samples. Although the system occasionally rejects correct predictions from newer generators, this conservative behaviour is acceptable, as rejected samples can support retraining. The framework maintains high acceptance of accurate predictions for natural images and in-domain AI data. Under adversarial attacks using FGSM and PGD, the Combined Uncertainty method rejects around 61 percent of successful attacks, while GP-based uncertainty alone achieves up to 80 percent. Overall, the results demonstrate that multi-source uncertainty fusion provides a resilient and adaptive solution for AI-generated image detection.

摘要: 随着人工智能生成的图像变得越来越真实，将它们与自然图像区分开来构成了越来越大的挑战。本文提出了一个强大的检测框架，该框架利用多种不确定性指标来决定是信任还是拒绝模型的预测。我们专注于三种补充技术：Fisher Info，它捕捉模型参数对输入变化的敏感性; Monte Carlo Dropout的基于信息的不确定性，它反映了预测变化性;以及使用高斯过程分类器的深度核学习框架的预测变化。为了集成这些不同的不确定性信号，粒子群优化用于学习最佳权重并确定自适应拒绝阈值。该模型在稳定扩散生成的图像上进行训练，并在GLIDE，VQDM，Midjourney，BigGAN和StyleGAN 3上进行评估，每个都引入了显著的分布变化。虽然标准度量（如预测概率和基于Fisher的度量）在分布中表现良好，但它们的有效性在偏移下会下降。相比之下，组合不确定性度量在看不见的生成器上始终实现约70%的错误拒绝率，成功过滤了大多数错误分类的AI样本。虽然系统偶尔会拒绝来自新生成器的正确预测，但这种保守行为是可以接受的，因为拒绝的样本可以支持重新训练。该框架保持了对自然图像和领域内人工智能数据的准确预测的高度接受度。在使用FGSM和PVD的对抗性攻击下，组合不确定性方法拒绝了大约61%的成功攻击，而仅基于GP的不确定性就能达到80%。总体而言，结果表明多源不确定性融合为人工智能生成的图像检测提供了弹性和自适应的解决方案。



## **26. SoK: Understanding (New) Security Issues Across AI4Code Use Cases**

SoK：了解整个AI 4 Code用例的（新）安全问题 cs.CR

39 pages, 19 figures

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18456v1) [paper-pdf](https://arxiv.org/pdf/2512.18456v1)

**Authors**: Qilong Wu, Taoran Li, Tianyang Zhou, Varun Chandrasekaran

**Abstract**: AI-for-Code (AI4Code) systems are reshaping software engineering, with tools like GitHub Copilot accelerating code generation, translation, and vulnerability detection. Alongside these advances, however, security risks remain pervasive: insecure outputs, biased benchmarks, and susceptibility to adversarial manipulation undermine their reliability. This SoK surveys the landscape of AI4Code security across three core applications, identifying recurring gaps: benchmark dominance by Python and toy problems, lack of standardized security datasets, data leakage in evaluation, and fragile adversarial robustness. A comparative study of six state-of-the-art models illustrates these challenges: insecure patterns persist in code generation, vulnerability detection is brittle to semantic-preserving attacks, fine-tuning often misaligns security objectives, and code translation yields uneven security benefits. From this analysis, we distill three forward paths: embedding secure-by-default practices in code generation, building robust and comprehensive detection benchmarks, and leveraging translation as a route to security-enhanced languages. We call for a shift toward security-first AI4Code, where vulnerability mitigation and robustness are embedded throughout the development life cycle.

摘要: 代码人工智能（AI 4Code）系统正在重塑软件工程，GitHub Copilot等工具加速代码生成、翻译和漏洞检测。然而，除了这些进步之外，安全风险仍然普遍存在：不安全的输出、有偏见的基准以及容易受到对抗操纵的影响削弱了它们的可靠性。该SoK调查了三个核心应用程序的AI 4 Code安全格局，找出了反复出现的差距：Python和玩具问题的基准主导地位、缺乏标准化的安全数据集、评估中的数据泄露以及脆弱的对抗稳健性。对六个最先进模型的比较研究说明了这些挑战：不安全模式在代码生成中持续存在，漏洞检测对语义保留攻击很脆弱，微调经常使安全目标不一致，代码转换产生不均衡的安全效益。从这一分析中，我们提炼出三条前进路径：在代码生成中嵌入默认安全实践、构建稳健且全面的检测基准，以及利用翻译作为通往安全增强语言的途径。我们呼吁向安全第一的AI 4 Code转变，将漏洞缓解和稳健性嵌入到整个开发生命周期中。



## **27. Who Can See Through You? Adversarial Shielding Against VLM-Based Attribute Inference Attacks**

谁能看穿你？针对基于VAR的属性推理攻击的对抗屏蔽 cs.CV

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18264v1) [paper-pdf](https://arxiv.org/pdf/2512.18264v1)

**Authors**: Yucheng Fan, Jiawei Chen, Yu Tian, Zhaoxia Yin

**Abstract**: As vision-language models (VLMs) become widely adopted, VLM-based attribute inference attacks have emerged as a serious privacy concern, enabling adversaries to infer private attributes from images shared on social media. This escalating threat calls for dedicated protection methods to safeguard user privacy. However, existing methods often degrade the visual quality of images or interfere with vision-based functions on social media, thereby failing to achieve a desirable balance between privacy protection and user experience. To address this challenge, we propose a novel protection method that jointly optimizes privacy suppression and utility preservation under a visual consistency constraint. While our method is conceptually effective, fair comparisons between methods remain challenging due to the lack of publicly available evaluation datasets. To fill this gap, we introduce VPI-COCO, a publicly available benchmark comprising 522 images with hierarchically structured privacy questions and corresponding non-private counterparts, enabling fine-grained and joint evaluation of protection methods in terms of privacy preservation and user experience. Building upon this benchmark, experiments on multiple VLMs demonstrate that our method effectively reduces PAR below 25%, keeps NPAR above 88%, maintains high visual consistency, and generalizes well to unseen and paraphrased privacy questions, demonstrating its strong practical applicability for real-world VLM deployments.

摘要: 随着视觉语言模型（VLM）的广泛采用，基于VLM的属性推断攻击已成为一个严重的隐私问题，使对手能够从社交媒体上共享的图像中推断私人属性。这种不断升级的威胁需要专门的保护方法来保护用户隐私。然而，现有的方法常常会降低图像的视觉质量或干扰社交媒体上基于视觉的功能，从而无法在隐私保护和用户体验之间实现理想的平衡。为了应对这一挑战，我们提出了一种新颖的保护方法，该方法在视觉一致性约束下联合优化隐私抑制和效用保护。虽然我们的方法在概念上是有效的，但由于缺乏公开可用的评估数据集，方法之间的公平比较仍然具有挑战性。为了填补这一空白，我们引入了VPI-COCO，这是一个公开的基准，由522个具有分层结构隐私问题的图像和相应的非私人对应图像组成，可以对隐私保护和用户体验方面的保护方法进行细粒度的联合评估。在此基准的基础上，在多个VLM上进行的实验表明，我们的方法有效地将VAR降低到25%以下，将NBAR保持在88%以上，保持高度的视觉一致性，并很好地推广到不可见和解释的隐私问题，证明了其对现实世界VLM部署的强大实际适用性。



## **28. Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation**

打破思维，打破系统：通过类人心理操纵越狱大型语言模型 cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18244v1) [paper-pdf](https://arxiv.org/pdf/2512.18244v1)

**Authors**: Zehao Liu, Xi Lin

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation.

摘要: 大型语言模型（LLM）已经相当受欢迎，并受到日益复杂的安全机制的保护。然而，越狱攻击通过诱导模型产生违反政策的行为，继续构成严重的安全威胁。当前的范式专注于输入级异常，忽视了模型的内部心理测量状态可以被系统性操纵。为了解决这个问题，我们引入了心理越狱，这是一种新的越狱攻击范式，它暴露了LLM中的状态心理攻击表面，攻击者利用交互中对模型心理状态的操纵。基于这一见解，我们提出了类人心理操纵（HPM），这是一种黑匣子越狱方法，可以动态地描述目标模型的潜在心理脆弱性并综合量身定制的多回合攻击策略。通过利用模型对拟人化一致性的优化，HPM创造了一种心理压力，社会合规性凌驾于安全约束之上。为了系统性地衡量心理安全，我们构建了一个纳入心理测量数据集和政策腐败评分（PCS）的评估框架。针对各种模型（例如，GPT-4 o、DeepSeek-V3、Gemini-2-Flash），HPM的平均攻击成功率（ASB）为88.1%，优于最先进的攻击基线。我们的实验证明了对高级防御的强大渗透，包括对抗性即时优化（例如，LPO）和认知干预（例如，自我提醒）。最终，PCS分析证实HPM会引发安全崩溃以满足操纵环境。我们的工作倡导从静态内容过滤到心理安全的根本范式转变，优先考虑开发针对深度认知操纵的心理防御机制。



## **29. Performance Guarantees for Data Freshness in Resource-Constrained Adversarial IoT Systems**

资源受限的对抗性物联网系统中数据新鲜度的性能保证 cs.NI

6 pages, 4 figures, conference paper

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18155v1) [paper-pdf](https://arxiv.org/pdf/2512.18155v1)

**Authors**: Aresh Dadlani, Muthukrishnan Senthil Kumar, Omid Ardakanian, Ioanis Nikolaidis

**Abstract**: Timely updates are critical for real-time monitoring and control applications powered by the Internet of Things (IoT). As these systems scale, they become increasingly vulnerable to adversarial attacks, where malicious agents interfere with legitimate transmissions to reduce data rates, thereby inflating the age of information (AoI). Existing adversarial AoI models often assume stationary channels and overlook queueing dynamics arising from compromised sensing sources operating under resource constraints. Motivated by the G-queue framework, this paper investigates a two-source M/G/1/1 system in which one source is adversarial and disrupts the update process by injecting negative arrivals according to a Poisson process and inducing i.i.d. service slowdowns, bounded in attack rate and duration. Using moment generating functions, we then derive closed-form expressions for average and peak AoI for an arbitrary number of sources. Moreover, we introduce a worst-case constrained attack model and employ stochastic dominance arguments to establish analytical AoI bounds. Numerical results validate the analysis and highlight the impact of resource-limited adversarial interference under general service time distributions.

摘要: 及时更新对于由物联网（IoT）支持的实时监控和控制应用程序至关重要。随着这些系统的扩展，它们变得越来越容易受到对抗攻击，恶意代理会干扰合法传输以降低数据速率，从而扩大信息时代（AoI）。现有的对抗性AoI模型通常假设静态通道，并忽略在资源限制下运行的受损害传感源所产生的排队动态。受G队列框架的启发，本文研究了一个双源M/G/1/1系统，其中一个源是对抗性的，并通过根据Poisson过程注入负到达并诱导i. i. d来扰乱更新过程。服务减慢，攻击率和持续时间有限。然后，使用矩生成函数，我们推导出任意数量来源的平均和峰值AoI的封闭形式表达。此外，我们引入了最坏情况的约束攻击模型，并采用随机优势参数来建立分析AoI界限。数值结果验证了分析，并强调了一般服务时间分布下资源有限的对抗干扰的影响。



## **30. COBRA: Catastrophic Bit-flip Reliability Analysis of State-Space Models**

COBRA：状态空间模型的灾难性位翻转可靠性分析 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.15778v2) [paper-pdf](https://arxiv.org/pdf/2512.15778v2)

**Authors**: Sanjay Das, Swastik Bhattacharya, Shamik Kundu, Arnab Raha, Souvik Kundu, Kanad Basu

**Abstract**: State-space models (SSMs), exemplified by the Mamba architecture, have recently emerged as state-of-the-art sequence-modeling frameworks, offering linear-time scalability together with strong performance in long-context settings. Owing to their unique combination of efficiency, scalability, and expressive capacity, SSMs have become compelling alternatives to transformer-based models, which suffer from the quadratic computational and memory costs of attention mechanisms. As SSMs are increasingly deployed in real-world applications, it is critical to assess their susceptibility to both software- and hardware-level threats to ensure secure and reliable operation. Among such threats, hardware-induced bit-flip attacks (BFAs) pose a particularly severe risk by corrupting model parameters through memory faults, thereby undermining model accuracy and functional integrity. To investigate this vulnerability, we introduce RAMBO, the first BFA framework specifically designed to target Mamba-based architectures. Through experiments on the Mamba-1.4b model with LAMBADA benchmark, a cloze-style word-prediction task, we demonstrate that flipping merely a single critical bit can catastrophically reduce accuracy from 74.64% to 0% and increase perplexity from 18.94 to 3.75 x 10^6. These results demonstrate the pronounced fragility of SSMs to adversarial perturbations.

摘要: 以Mamba架构为例的状态空间模型（ASM）最近成为最先进的序列建模框架，提供线性时间可扩展性以及长上下文环境中的强劲性能。由于其效率、可扩展性和表达能力的独特组合，ASM已成为基于转换器的模型的引人注目的替代方案，后者面临注意力机制的二次计算和存储成本。随着ESM越来越多地部署在现实世界的应用程序中，评估它们对软件和硬件级威胁的敏感性以确保安全可靠的操作至关重要。在此类威胁中，硬件引发的位翻转攻击（BFA）通过存储器故障损坏模型参数，从而破坏模型准确性和功能完整性，从而构成了特别严重的风险。为了调查此漏洞，我们引入了RAMBE，这是第一个专门针对基于Mamba的架构而设计的BFA框架。通过使用LAMBADA基准（一种完形词预测任务）对Mamba-1.4b模型进行实验，我们证明，仅翻转一个关键位就会灾难性地将准确性从74.64%降低到0%，并将困惑度从18.94增加到3.75 x 106。这些结果证明了ESM对对抗性扰动的明显脆弱性。



## **31. Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization**

三思而后行：本地化对抗模拟的理性代理框架 cs.CR

17 pages, 9 figures, 6 tables. Revised version with an updated author list, expanded experimental results and analysis

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.06713v2) [paper-pdf](https://arxiv.org/pdf/2512.06713v2)

**Authors**: Donghang Duan, Xu Zheng, Yuefeng He, Chong Mu, Leyi Cai, Lizong Zhang

**Abstract**: Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent privacy paradox: users must disclose data to untrusted third parties for guaranteed privacy preservation. Moreover, directly migrating current solutions to local small-scale models (LSMs) offers a suboptimal solution with severe utility collapse. Our work argues that this failure stems not merely from the capability deficits of LSMs, but significantly from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SOTA) methods. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer architecture. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies tend to drift into an irrational state. Instead, RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible privacy benefits. This mechanism promotes a rational early-stopping criterion, and structurally prevents utility collapse. Extensive experiments on different benchmarks demonstrate that RLAA achieves a superior privacy-utility trade-off compared to strong baselines.

摘要: 当前基于LLM的文本匿名化框架通常依赖于强大的LLM的远程API服务，这造成了一个固有的隐私悖论：用户必须向不受信任的第三方披露数据以保证隐私保护。此外，将当前解决方案直接迁移到本地小规模模型（LSM）提供了一个次优解决方案，但公用事业严重崩溃。我们的工作认为，这种失败不仅源于LSM的能力缺陷，而且还很大程度上源于当前最先进（SOTA）方法所采用的贪婪对抗策略固有的不合理性。为了解决这个问题，我们提出了理性本地化对抗模拟（RLAA），这是一个完全本地化且免训练的框架，具有攻击者-干扰者-干扰者架构。我们将匿名化过程建模为边缘隐私收益（MPG）和边缘公用事业成本（MUC）之间的权衡，并证明贪婪策略往往会陷入非理性状态。相反，RLAA引入了一个仲裁员，充当理性守门人，验证攻击者的推断，以过滤掉提供可忽略不计隐私利益的反馈。该机制促进了合理的提前停止标准，并从结构上防止公用事业崩溃。对不同基准的广泛实验表明，与强基准相比，RLAA实现了更好的隐私与公用事业权衡。



## **32. SEA: Spectral Edge Attack on Graph Neural Networks**

SEA：对图神经网络的频谱边缘攻击 cs.LG

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.08964v3) [paper-pdf](https://arxiv.org/pdf/2512.08964v3)

**Authors**: Yongyu Wang

**Abstract**: Graph neural networks (GNNs) have been widely applied in a variety of domains. However, the very ability of graphs to represent complex data structures is both the key strength of GNNs and a major source of their vulnerability. Recent studies have shown that attacking GNNs by maliciously perturbing the underlying graph can severely degrade their performance. For attack methods, the central challenge is to maintain attack effectiveness while remaining difficult to detect. Most existing attacks require modifying the graph structure, such as adding or deleting edges, which is relatively easy to notice. To address this problem, this paper proposes a new attack model that employs spectral adversarial robustness evaluation to quantitatively analyze the vulnerability of each edge in a graph. By precisely targeting the weakest links, our method can achieve effective attacks without changing the connectivity pattern of edges in the graph, for example by subtly adjusting the weights of a small subset of the most vulnerable edges. We apply the proposed method to attack several classical graph neural network architectures, and experimental results show that our attack is highly effective.

摘要: 图神经网络（GNN）已广泛应用于各个领域。然而，图形表示复杂数据结构的能力本身既是GNN的关键优势，也是其脆弱性的主要来源。最近的研究表明，通过恶意干扰底层图来攻击GNN可能会严重降低其性能。对于攻击方法来说，核心挑战是保持攻击有效性，同时保持难以检测。大多数现有的攻击都需要修改图结构，例如添加或删除边，这相对容易注意到。为了解决这个问题，本文提出了一种新的攻击模型，该模型采用谱对抗鲁棒性评估来定量分析图中每条边的脆弱性。通过精确地针对最弱的链接，我们的方法可以在不改变图中边的连接性模式的情况下实现有效的攻击，例如通过巧妙地调整最脆弱边的一小子集的权重。我们应用所提出的方法来攻击几种经典的图神经网络架构，实验结果表明我们的攻击非常有效。



## **33. Ensuring Calibration Robustness in Split Conformal Prediction Under Adversarial Attacks**

对抗性攻击下分裂共形预测中的校准鲁棒性 stat.ML

Submitted to AISTATS 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2511.18562v2) [paper-pdf](https://arxiv.org/pdf/2511.18562v2)

**Authors**: Xunlei Qian, Yue Xing

**Abstract**: Conformal prediction (CP) provides distribution-free, finite-sample coverage guarantees but critically relies on exchangeability, a condition often violated under distribution shift. We study the robustness of split conformal prediction under adversarial perturbations at test time, focusing on both coverage validity and the resulting prediction set size. Our theoretical analysis characterizes how the strength of adversarial perturbations during calibration affects coverage guarantees under adversarial test conditions. We further examine the impact of adversarial training at the model-training stage. Extensive experiments support our theory: (i) Prediction coverage varies monotonically with the calibration-time attack strength, enabling the use of nonzero calibration-time attack to predictably control coverage under adversarial tests; (ii) target coverage can hold over a range of test-time attacks: with a suitable calibration attack, coverage stays within any chosen tolerance band across a contiguous set of perturbation levels; and (iii) adversarial training at the training stage produces tighter prediction sets that retain high informativeness.

摘要: 保形预测（CP）提供无分布、有限样本覆盖保证，但严重依赖于交换性，这是分布转移下经常违反的条件。我们研究测试时对抗性扰动下分裂保形预测的鲁棒性，重点关注覆盖有效性和由此产生的预测集大小。我们的理论分析描述了校准期间对抗性扰动的强度如何影响对抗性测试条件下的覆盖保证。我们进一步研究了模型训练阶段对抗训练的影响。大量的实验支持我们的理论：（i）预测覆盖率随着校准时间攻击强度单调变化，使得使用非零校准时间攻击来在对抗测试下可预测地控制覆盖率;（ii）目标覆盖率可以在一系列测试时间攻击中保持：使用合适的校准攻击，覆盖率保持在任何选定的容差范围内，跨越连续的扰动水平集;以及（iii）训练阶段的对抗训练产生保持高信息量的更紧密的预测集。



## **34. Lifefin: Escaping Mempool Explosions in DAG-based BFT**

Lifefin：逃离基于DAB的BFT中的Mempool爆炸 cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2511.15936v2) [paper-pdf](https://arxiv.org/pdf/2511.15936v2)

**Authors**: Jianting Zhang, Sen Yang, Alberto Sonnino, Sebastián Loza, Aniket Kate

**Abstract**: Directed Acyclic Graph (DAG)-based Byzantine Fault-Tolerant (BFT) protocols have emerged as promising solutions for high-throughput blockchains. By decoupling data dissemination from transaction ordering and constructing a well-connected DAG in the mempool, these protocols enable zero-message ordering and implicit view changes. However, we identify a fundamental liveness vulnerability: an adversary can trigger mempool explosions to prevent transaction commitment, ultimately compromising the protocol's liveness.   In response, this work presents Lifefin, a generic and self-stabilizing protocol designed to integrate seamlessly with existing DAG-based BFT protocols and circumvent such vulnerabilities. Lifefin leverages the Agreement on Common Subset (ACS) mechanism, allowing nodes to escape mempool explosions by committing transactions with bounded resource usage even in adverse conditions. As a result, Lifefin imposes (almost) zero overhead in typical cases while effectively eliminating liveness vulnerabilities.   To demonstrate the effectiveness of Lifefin, we integrate it into two state-of-the-art DAG-based BFT protocols, Sailfish and Mysticeti, resulting in two enhanced variants: Sailfish-Lifefin and Mysticeti-Lifefin. We implement these variants and compare them with the original Sailfish and Mysticeti systems. Our evaluation demonstrates that Lifefin achieves comparable transaction throughput while introducing only minimal additional latency to resist similar attacks.

摘要: 基于有向无环图（DAB）的拜占庭故障容忍（BFT）协议已成为高吞吐量区块链的有前途的解决方案。通过将数据传播与事务排序脱钩并在内存池中构建连接良好的DAB，这些协议实现了零消息排序和隐式视图更改。然而，我们发现了一个根本的活跃性漏洞：对手可以触发成员池爆炸以阻止事务承诺，最终损害协议的活跃性。   作为回应，这项工作提出了Lifefin，这是一种通用的自稳定协议，旨在与现有的基于DAB的BFT协议无缝集成并规避此类漏洞。Lifefin利用公共子集协议（ACS）机制，允许节点即使在不利条件下也通过提交具有有限资源使用量的事务来避免成员池爆炸。因此，Lifefin在典型情况下（几乎）实行零管理，同时有效地消除了活力漏洞。   为了证明Lifefin的有效性，我们将其集成到两个最先进的基于DAB的BFT协议Sailfish和Mysticeti中，从而产生了两个增强的变体：Sailfish-Lifefin和Mysticeti-Lifefin。我们实现这些变体并将它们与原始的Sailfish和Mysticeti系统进行比较。我们的评估表明，Lifefin实现了相当的交易吞吐量，同时仅引入了最小的额外延迟来抵抗类似的攻击。



## **35. Potent but Stealthy: Rethink Profile Pollution against Sequential Recommendation via Bi-level Constrained Reinforcement Paradigm**

有效但隐蔽：通过双层约束强化范式重新思考针对顺序推荐的个人资料污染 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2511.09392v4) [paper-pdf](https://arxiv.org/pdf/2511.09392v4)

**Authors**: Jiajie Su, Zihan Nan, Yunshan Ma, Xiaobo Xia, Xiaohua Feng, Weiming Liu, Xiang Chen, Xiaolin Zheng, Chaochao Chen

**Abstract**: Sequential Recommenders, which exploit dynamic user intents through interaction sequences, is vulnerable to adversarial attacks. While existing attacks primarily rely on data poisoning, they require large-scale user access or fake profiles thus lacking practicality. In this paper, we focus on the Profile Pollution Attack that subtly contaminates partial user interactions to induce targeted mispredictions. Previous PPA methods suffer from two limitations, i.e., i) over-reliance on sequence horizon impact restricts fine-grained perturbations on item transitions, and ii) holistic modifications cause detectable distribution shifts. To address these challenges, we propose a constrained reinforcement driven attack CREAT that synergizes a bi-level optimization framework with multi-reward reinforcement learning to balance adversarial efficacy and stealthiness. We first develop a Pattern Balanced Rewarding Policy, which integrates pattern inversion rewards to invert critical patterns and distribution consistency rewards to minimize detectable shifts via unbalanced co-optimal transport. Then we employ a Constrained Group Relative Reinforcement Learning paradigm, enabling step-wise perturbations through dynamic barrier constraints and group-shared experience replay, achieving targeted pollution with minimal detectability. Extensive experiments demonstrate the effectiveness of CREAT.

摘要: 顺序推荐器通过交互序列利用动态用户意图，容易受到对抗攻击。虽然现有的攻击主要依赖于数据中毒，但它们需要大规模用户访问或虚假配置文件，因此缺乏实用性。在本文中，我们重点关注个人资料污染攻击，它微妙地污染部分用户交互以引发有针对性的错误预测。以前的PPA方法有两个局限性，即i）过度依赖序列范围影响限制了对项目转变的细粒度扰动，并且ii）整体修改会导致可检测的分布变化。为了应对这些挑战，我们提出了一种受约束的强化驱动攻击CREAT，该攻击将双层优化框架与多奖励强化学习相结合，以平衡对抗功效和隐蔽性。我们首先开发了模式平衡奖励政策，该政策集成了模式倒置奖励以倒置关键模式，并集成了分布一致性奖励以最大限度地减少通过不平衡协同优传输可检测到的转变。然后，我们采用受约束的群体相对强化学习范式，通过动态障碍约束和群体共享体验回放实现分步扰动，以最小的可检测性实现有针对性的污染。大量实验证明了CREAT的有效性。



## **36. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CL

Presented at NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2511.02376v3) [paper-pdf](https://arxiv.org/pdf/2511.02376v3)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，对抗性提示会引发有害输出。然而，大多数评估都集中在单轮交互上，而现实世界的攻击则通过自适应多轮对话展开。我们介绍了AutoAdv，这是一个用于自动多回合越狱的免训练框架，在六个回合内对Llama-3.1-8B的攻击成功率高达95%，比单回合基线提高了24%。AutoAdv独特地结合了三种自适应机制：从成功的攻击中学习以增强未来提示的模式管理器、根据失败模式动态调整采样参数的温度管理器以及掩盖有害请求然后迭代细化的两阶段重写策略。对商业和开源模型（Llama-3.1-8B、GPT-4 o mini、Qwen 3 - 235 B、Mistral-7 B）的广泛评估揭示了当前安全机制中存在的持续漏洞，多回合攻击的表现始终优于单回合方法。这些发现表明，针对单轮交互优化的对齐策略无法在扩展对话中保持稳健性，凸显了对多轮感知防御的迫切需求。



## **37. Machine Unlearning in Speech Emotion Recognition via Forget Set Alone**

通过Forget Set Alone实现语音情感识别的机器去学习 cs.SD

Submitted to ICASSP 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2510.04251v2) [paper-pdf](https://arxiv.org/pdf/2510.04251v2)

**Authors**: Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz

**Abstract**: Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.

摘要: 语音情感识别是从语音信号中识别情感状态的一种方法，在人机交互、教育、医疗等领域有着广泛的应用。然而，由于语音数据包含丰富的敏感信息，部分数据可能会被要求删除的发言者出于隐私问题。目前的机器学习方法在很大程度上依赖于被遗忘的样本之外的数据。然而，当数据再分配受到限制并且在大数据的背景下需要大量计算资源时，这种依赖带来了挑战。我们提出了一种新的基于对抗性攻击的方法，该方法只使用要忘记的数据来微调预训练的语音情感识别模型。实验结果表明，该方法能够有效地去除模型中待遗忘数据的知识，同时保持模型在情感识别测试集上的高性能.



## **38. Membership Inference Attack with Partial Features**

具有部分特征的成员推断攻击 cs.LG

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.06244v2) [paper-pdf](https://arxiv.org/pdf/2508.06244v2)

**Authors**: Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Zhongyun Hua, Weizhe Zhang

**Abstract**: Machine learning models are vulnerable to membership inference attack, which can be used to determine whether a given sample appears in the training data. Most existing methods assume the attacker has full access to the features of the target sample. This assumption, however, does not hold in many real-world scenarios where only partial features are available, thereby limiting the applicability of these methods. In this work, we introduce Partial Feature Membership Inference (PFMI), a scenario where the adversary observes only partial features of each sample and aims to infer whether this observed subset was present in the training set. To address this problem, we propose MRAD (Memory-guided Reconstruction and Anomaly Detection), a two-stage attack framework that works in both white-box and black-box settings. In the first stage, MRAD leverages the latent memory of the target model to reconstruct the unknown features of the sample. We observe that when the known features are absent from the training set, the reconstructed sample deviates significantly from the true data distribution. Consequently, in the second stage, we use anomaly detection algorithms to measure the deviation between the reconstructed sample and the training data distribution, thereby determining whether the known features belong to a member of the training set. Empirical results demonstrate that MRAD is effective across various datasets, and maintains compatibility with off-the-shelf anomaly detection techniques. For example, on STL-10, our attack exceeds an AUC of around 0.75 even with 60% of the missing features.

摘要: 机器学习模型容易受到隶属度推理攻击，该攻击可用于确定给定样本是否出现在训练数据中。大多数现有方法都假设攻击者可以完全访问目标样本的特征。然而，这一假设在许多只有部分特征可用的现实世界场景中并不成立，从而限制了这些方法的适用性。在这项工作中，我们引入了部分特征隶属推理（PFMI），在这种情况下，对手仅观察每个样本的部分特征，并旨在推断该观察到的子集是否存在于训练集中。为了解决这个问题，我们提出了MRAD（内存引导重建和异常检测），这是一种两阶段攻击框架，适用于白盒和黑盒设置。在第一阶段，MRAD利用目标模型的潜在记忆来重建样本的未知特征。我们观察到，当训练集中缺乏已知特征时，重建的样本会显着偏离真实数据分布。因此，在第二阶段，我们使用异常检测算法来测量重建样本和训练数据分布之间的偏差，从而确定已知特征是否属于训练集的成员。经验结果表明，MRAD在各种数据集中都有效，并保持与现成的异常检测技术的兼容性。例如，在SPL-10上，即使有60%的特征缺失，我们的攻击也超过了约0.75的AUR。



## **39. Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models**

模拟集群攻击：通过微调的视觉语言模型转移越狱 cs.CV

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2508.01741v2) [paper-pdf](https://arxiv.org/pdf/2508.01741v2)

**Authors**: Ruofan Wang, Xin Wang, Yang Yao, Xuan Tong, Xingjun Ma

**Abstract**: Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.

摘要: 微调开源视觉语言模型（VLM）创建了一个关键但未充分探索的攻击表面：基础VLM中的漏洞可能会保留在微调的变体中，使其容易受到可转移的越狱攻击。为了证明这种风险，我们引入了模拟集群攻击（SEA），这是一种新型的灰箱越狱方法，其中对手可以完全访问基本VLM，但不知道微调目标的权重或训练配置。为了提高微调VLM之间的越狱转移性，SEA结合了两项关键技术：微调弹道模拟（FTS）和定向即时引导（TPG）。FTS通过模拟视觉编码器的参数变化来生成可转移的对抗图像，而TPG是一种文本策略，可以引导语言解码器朝着对抗优化的输出方向发展。Qwen 2-BL家族（2B和7 B）的实验表明，SEA在各种微调变体中实现了超过86.5%的高转移攻击成功率和接近49.5%的毒性率，即使是那些专门微调以改善安全行为的变体。值得注意的是，虽然直接基于PGD的图像越狱很少通过微调的VLM传输，但SEA可靠地利用了从基本模型继承的漏洞，显着增强了可传输性。这些发现凸显了迫切需要保护微调的专有VLM免受从开源基金会继承的可转移漏洞的影响，从而激励整个模型生命周期中的整体防御开发。



## **40. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

通用越狱后缀是强烈的注意力劫持者 cs.CR

Accepted at TACL 2026

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2506.12880v2) [paper-pdf](https://arxiv.org/pdf/2506.12880v2)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack, we observe that suffixes vary in efficacy: some are markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that a shallow, critical mechanism drives GCG's effectiveness. This mechanism builds on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG's universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving the attack's success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.

摘要: 我们研究基于后缀的越狱$\unicode{x2013}$这是一个针对大型语言模型（LLM）的强大攻击家族，这些模型优化对抗性后缀以规避安全对齐。关注广泛使用的基础GCG攻击，我们观察到后缀的功效各不相同：有些后缀明显更通用$\unicode{x2013}$，一般化为许多不可见的有害指令$\unicode{x2013}$。我们首先表明，GCG的有效性是一种肤浅的、关键的机制。该机制建立在生成之前从对抗性后缀到最终聊天模板令牌的信息流之上。量化这种机制在生成过程中的主导地位，我们发现GCG不规则且积极地劫持了情境化过程。至关重要的是，我们将劫持与普遍现象联系起来，更普遍的后缀意味着更强大的劫持者。随后，我们证明了这些见解具有实际意义：GCG的普遍性可以在没有额外计算成本的情况下有效地增强（在某些情况下高达5美元），并且还可以通过手术来减轻，至少将攻击的成功减半，并将效用损失最小。我们在http://github.com/matanbt/interp-jailbreak上发布我们的代码和数据。



## **41. SoK: Are Watermarks in LLMs Ready for Deployment?**

SoK：LLM中的水印准备好部署了吗？ cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.

摘要: 大型语言模型（LLM）改变了自然语言处理，在不同任务中展示了令人印象深刻的能力。然而，部署这些模型会带来与知识产权侵犯和潜在滥用相关的关键风险，特别是因为对手可以模仿这些模型来窃取服务或产生误导性输出。我们特别关注模型窃取攻击，因为它们与专有LLM高度相关，并对其安全、收入和道德部署构成严重威胁。虽然已经出现了各种水印技术来减轻这些风险，但目前尚不清楚社区和行业在LLM中开发和部署水印方面取得了多大进展。   为了弥合这一差距，我们的目标是通过1）提供LLM中水印的详细分类，2）提出一种新型知识产权分类器来探索水印在攻击和无攻击环境下的有效性和影响，3）分析LLM中现有水印的局限性，4）讨论LLM中水印的实际挑战和潜在的未来方向。通过广泛的实验，我们表明，尽管研究成果令人鼓舞，领先公司和社区也对部署水印给予了极大的关注，但由于这些技术对LLM和下游任务的模型效用产生不利影响，这些技术尚未在现实世界应用中充分发挥潜力。我们的研究结果提供了对LLM中的水印的深刻理解，强调了针对LLM部署量身定制的实用水印解决方案的必要性。



## **42. Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs**

通过从LLM到SLM的对抗性即时蒸馏进行高效且隐蔽的越狱攻击 cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2506.17231v2) [paper-pdf](https://arxiv.org/pdf/2506.17231v2)

**Authors**: Xiang Li, Chong Zhang, Jia Wang, Fangyu Wu, Yushi Li, Xiaobo Jin

**Abstract**: As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.

摘要: 随着对大型语言模型（LLM）的越狱攻击规模和复杂性不断升级，其效率和实用性受到限制，对LLM安全提出了深刻的挑战。越狱技术已经从手动提示工程发展到自动化方法。最近的进展已经自动化了越狱方法，利用LLM来生成越狱指令和对抗性示例，并带来了令人鼓舞的结果。然而，这些方法普遍包括LLM生成阶段，由于LLM部署和推理的复杂性，这阻碍了有效实施和更广泛的采用。为了缓解这个问题，我们引入了\textBF{对抗提示蒸馏}，这是一个创新框架，集成了掩蔽语言建模、强化学习和动态温度控制，将LLM越狱能力提炼成更小的语言模型（SLC）。这种方法可以实现高效、稳健的越狱攻击，同时保持高成功率并适应更广泛的应用程序上下文。实证评估证实了该方法在攻击功效、资源优化和跨模型通用性方面的优势。我们的研究强调了将越狱功能转移到SLC的可行性，揭示了LLC中的固有漏洞，并为推进LLM安全调查提供了新颖的见解。我们的代码可访问：https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt。



## **43. Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models**

针对个性化文本到图像扩散模型的数据集版权规避攻击 cs.CV

Accepted by IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2505.02824v2) [paper-pdf](https://arxiv.org/pdf/2505.02824v2)

**Authors**: Kuofeng Gao, Yufei Zhu, Yiming Li, Jiawang Bai, Yong Yang, Zhifeng Li, Shu-Tao Xia

**Abstract**: Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.

摘要: 文本到图像（T2 I）扩散模型能够根据文本提示生成高质量图像。然而，微调这些预先训练的模型以实现个性化会引发人们对未经授权的数据集使用的担忧。为了解决这个问题，最近提出了数据集所有权验证（DOV），通过后门技术将水印嵌入到微调数据集中。这些水印在良性样本上保持休眠状态，但在触发时会产生所有者指定的输出。尽管DOV有着承诺，但其针对版权规避攻击（CAE）的稳健性仍然有待探索。在本文中，我们研究对手如何绕过这些机制，使在带水印数据集上训练的模型能够绕过所有权验证。我们首先分析后门删除所实现的潜在攻击的局限性，包括DPD和T2 IShield。在实践中，DPD由于随机性而导致有效性不一致，而T2 IShield则在水印作为本地图像补丁嵌入时失败。为此，我们引入了CEAT 2 I，这是T2 I扩散模型中第一个专门针对DOV的CAE。CEAT 2 I由三个阶段组成：（1）由于观察到T2 I模型在中间特征方面更快地收敛在带水印样本上，而不是训练损失，因此我们可靠地检测带水印样本;（2）我们从检测到的样本的提示中迭代地消融令牌并监控特征变化以识别触发令牌;以及（3）我们应用封闭形式的概念擦除方法来去除注入的水印。大量实验表明，CEAT 2 I有效地规避了最先进的DOV机制，同时保留了模型性能。该代码可在https://github.com/csyufei/CEAT2I上获取。



## **44. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.

摘要: 大型语言模型（LLM）继续表现出越狱攻击的漏洞：精心设计的恶意输入，旨在绕过安全护栏并引发有害响应。因此，我们提出了AutoAdv，这是一个新颖的框架，可以自动生成对抗提示，以系统地评估和暴露LLM安全机制中的漏洞。我们的方法利用参数攻击者LLM通过战略重写技术、专门的系统提示和优化的超参数配置来产生语义伪装的恶意提示。我们工作的主要贡献是一种动态、多回合攻击方法，该方法分析失败的越狱尝试，并利用角色扮演、误导和上下文操纵等技术迭代生成细化的后续提示。我们使用StrongRESYS（arXiv：2402.10260 [cs.CL]）框架在连续交互回合中量化评估攻击成功率（ASB）。通过对最先进模型（包括ChatGPT、Llama和DeepSeek）进行广泛的实证评估，我们揭示了重大漏洞，我们的自动攻击在有害内容生成方面实现了高达86%的越狱成功率。我们的研究结果表明，当前的安全机制仍然容易受到复杂的多回合攻击，这凸显了对更强大的防御策略的迫切需要。



## **45. To See or Not to See -- Fingerprinting Devices in Adversarial Environments Amid Advanced Machine Learning**

看还是不看--先进机器学习中对抗环境中的指纹设备 cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2504.08264v2) [paper-pdf](https://arxiv.org/pdf/2504.08264v2)

**Authors**: Justin Feng, Amirmohammad Haddad, Nader Sehatbakhsh

**Abstract**: The increasing use of the Internet of Things raises security concerns. To address this, device fingerprinting is often employed to authenticate devices, detect adversaries, and identify eavesdroppers in an environment. This requires the ability to discern between legitimate and malicious devices which is achieved by analyzing the unique physical and/or operational characteristics of IoT devices. In the era of the latest progress in machine learning, particularly generative models, it is crucial to methodically examine the current studies in device fingerprinting. This involves explaining their approaches and underscoring their limitations when faced with adversaries armed with these ML tools. To systematically analyze existing methods, we propose a generic, yet simplified, model for device fingerprinting. Additionally, we thoroughly investigate existing methods to authenticate devices and detect eavesdropping, using our proposed model. We further study trends and similarities between works in authentication and eavesdropping detection and present the existing threats and attacks in these domains. Finally, we discuss future directions in fingerprinting based on these trends to develop more secure IoT fingerprinting schemes.

摘要: 物联网的使用越来越多引发了安全担忧。为了解决这个问题，通常使用设备指纹识别来验证设备、检测对手和识别环境中的窃听者。这需要能够区分合法设备和恶意设备，这是通过分析物联网设备的独特物理和/或操作特征来实现的。在机器学习（特别是生成模型）取得最新进展的时代，系统地检查当前设备指纹识别的研究至关重要。这涉及解释他们的方法并强调他们在面对配备这些ML工具的对手时的局限性。为了系统地分析现有方法，我们提出了一种通用但简化的设备指纹识别模型。此外，我们还使用我们提出的模型彻底研究了现有的验证设备和检测窃听的方法。我们进一步研究身份验证和窃听检测领域工作之间的趋势和相似之处，并呈现这些领域中现有的威胁和攻击。最后，我们根据这些趋势讨论了指纹识别的未来方向，以开发更安全的物联网指纹识别方案。



## **46. Bleeding Pathways: Vanishing Discriminability in LLM Hidden States Fuels Jailbreak Attacks**

流血途径：LLM隐藏状态中歧视性的消失助长了越狱攻击 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2503.11185v2) [paper-pdf](https://arxiv.org/pdf/2503.11185v2)

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen

**Abstract**: LLMs remain vulnerable to jailbreak attacks that exploit adversarial prompts to circumvent safety measures. Current safety fine-tuning approaches face two critical limitations. First, they often fail to strike a balance between security and utility, where stronger safety measures tend to over-reject harmless user requests. Second, they frequently miss malicious intent concealed within seemingly benign tasks, leaving models exposed to exploitation. Our work identifies a fundamental cause of these issues: during response generation, an LLM's capacity to differentiate harmful from safe outputs deteriorates. Experimental evidence confirms this, revealing that the separability between hidden states for safe and harmful responses diminishes as generation progresses. This weakening discrimination forces models to make compliance judgments earlier in the generation process, restricting their ability to recognize developing harmful intent and contributing to both aforementioned failures. To mitigate this vulnerability, we introduce DEEPALIGN - an inherent defense framework that enhances the safety of LLMs. By applying contrastive hidden-state steering at the midpoint of response generation, DEEPALIGN amplifies the separation between harmful and benign hidden states, enabling continuous intrinsic toxicity detection and intervention throughout the generation process. Across diverse LLMs spanning varying architectures and scales, it reduced attack success rates of nine distinct jailbreak attacks to near-zero or minimal. Crucially, it preserved model capability while reducing over-refusal. Models equipped with DEEPALIGN exhibited up to 3.5% lower error rates in rejecting challenging benign queries and maintained standard task performance with less than 1% decline. This marks a substantial advance in the safety-utility Pareto frontier.

摘要: LLM仍然容易受到越狱攻击，这些攻击利用对抗提示来规避安全措施。当前的安全微调方法面临两个关键限制。首先，它们往往无法在安全性和实用性之间取得平衡，更强的安全措施往往会过度拒绝无害的用户请求。其次，它们经常错过看似良性的任务中隐藏的恶意意图，从而使模型容易受到利用。我们的工作确定了这些问题的根本原因：在响应生成过程中，LLM区分有害输出和安全输出的能力下降。实验证据证实了这一点，揭示了安全反应和有害反应的隐藏状态之间的可分离性随着世代的发展而减弱。这种削弱的歧视迫使模型在生成过程的早期做出合规判断，限制了它们识别有害意图的能力，并导致了上述两次失败。为了缓解此漏洞，我们引入了DEEPALIGN --一种增强LLM安全性的固有防御框架。通过在反应生成的中点应用对比隐藏状态引导，DEEPALIGN放大了有害和良性隐藏状态之间的分离，从而在整个生成过程中实现持续的固有毒性检测和干预。在跨越不同架构和规模的各种LLM中，它将九种不同越狱攻击的攻击成功率降低到接近零或最低。至关重要的是，它保留了模型的能力，同时减少了过度拒绝。配备DEEPALIGN的模型在拒绝具有挑战性的良性查询方面的错误率降低了3.5%，并保持了标准任务性能，下降幅度不到1%。这标志着安全效用帕累托前沿的重大进步。



## **47. When Should Selfish Miners Double-Spend?**

什么时候自私的矿工应该加倍花钱？ cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2501.03227v4) [paper-pdf](https://arxiv.org/pdf/2501.03227v4)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.

摘要: 传统的双重支出攻击模型忽略了孤儿区块带来的收入损失。另一方面，自私的采矿文献通常忽视攻击者在每个攻击周期中免费重复支出的机会。在本文中，我们给出了一个严格的随机分析的攻击对手的目标是双花，而自私地挖掘。为此，我们首先结合顽固和自私的采矿攻击，即构建一个策略，让攻击者表现得顽固，直到其私人分支达到一定长度，然后转向自私。我们为每个参数制度提供最佳的确定性。接下来，我们提供了仍然比诚实采矿更有利可图的最大顽固度，并论证了顽固度水平与$k$-确认规则之间的联系。我们表明，在每个攻击周期中，如果顽固程度高于$k$，对手就可以免费获得双重消费的机会。在每个周期中，对于给定的顽固度水平，我们严格制定双重消费的可能性有多大。我们进一步修改顽固政权中的攻击，以隐藏攻击并增加双重消费的概率。



## **48. ThreatPilot: Attack-Driven Threat Intelligence Extraction**

ThreatPilot：攻击驱动的威胁情报提取 cs.CR

17 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2412.10872v2) [paper-pdf](https://arxiv.org/pdf/2412.10872v2)

**Authors**: Ming Xu, Hongtai Wang, Jiahao Liu, Xinfeng Li, Zhengmin Yu, Weili Han, Hoon Wei Lim, Jin Song Dong, Jiaheng Zhang

**Abstract**: Efficient defense against dynamically evolving advanced persistent threats (APT) requires the structured threat intelligence feeds, such as techniques used. However, existing threat-intelligence extraction techniques predominantly focuses on individual pieces of intelligence-such as isolated techniques or atomic indicators-resulting in fragmented and incomplete representations of real-world attacks. This granularity inherently limits on both the depth and the contextual richness of the extracted intelligence, making it difficult for downstream security systems to reason about multi-step behaviors or to generate actionable detections. To address this gap, we propose to extract the layered Attack-driven Threat Intelligence (ATIs), a comprehensive representation that captures the full spectrum of adversarial behavior. We propose ThreatPilot, which can accurately identify the AITs including complete tactics, techniques, multi-step procedures, and their procedure variants, and integrate the threat intelligence to software security application scenarios: the detection rules (i.e., Sigma) and attack command can be generated automatically to a more accuracy level. Experimental results on 1,769 newly crawly reports and 16 manually calibrated reports show ThreatPilot's effectiveness in identifying accuracy techniques, outperforming state-of-the-art approaches of AttacKG by 1.34X in F1 score. Further studies upon 64,185 application logs via Honeypot show that our Sigma rule generator significantly outperforms several existing rules-set in detecting the real-world malicious events. Industry partners confirm that our Sigma rule generator can significantly help save time and costs of the rule generation process. In addition, our generated commands achieve an execution rate of 99.3%, compared to 50.3% without the extracted intelligence.

摘要: 有效防御动态演变的高级持久威胁（APT）需要结构化的威胁情报源，例如使用的技术。然而，现有的威胁情报提取技术主要集中在单个情报片段上--例如孤立的技术或原子指示器--导致现实世界攻击的碎片化和不完整的表示。这种粒度本质上限制了提取情报的深度和上下文丰富性，使得下游安全系统难以推理多步骤行为或生成可操作的检测。为了解决这一差距，我们建议提取分层的攻击驱动威胁情报（ATI），这是一种捕捉全方位对抗行为的全面表示。我们提出了ThreatPilot，它可以准确识别包括完整的战术、技术、多步骤程序及其程序变体在内的AIT，并将威胁情报集成到软件安全应用场景：检测规则（即Sigma）和攻击命令可以自动生成到更准确的水平。1，769份新爬行报告和16份手动校准报告的实验结果表明，ThreatPilot在识别准确性技术方面的有效性，F1评分比AttacKG的最先进方法高出1.34倍。通过Honeypot对64，185个应用程序日志的进一步研究表明，我们的Sigma规则生成器在检测现实世界恶意事件方面的表现显着优于几个现有的规则集。行业合作伙伴确认，我们的Sigma规则生成器可以显着帮助节省规则生成过程的时间和成本。此外，我们生成的命令的执行率为99.3%，而在没有提取情报的情况下，执行率为50.3%。



## **49. Scalable Dendritic Modeling Advances Expressive and Robust Deep Spiking Neural Networks**

可扩展的树突状建模提高了表达性和鲁棒性的深度尖峰神经网络 cs.NE

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2412.06355v2) [paper-pdf](https://arxiv.org/pdf/2412.06355v2)

**Authors**: Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian

**Abstract**: Dendritic computation endows biological neurons with rich nonlinear integration and high representational capacity, yet it is largely missing in existing deep spiking neural networks (SNNs). Although detailed multi-compartment models can capture dendritic computations, their high computational cost and limited flexibility make them impractical for deep learning. To combine the advantages of dendritic computation and deep network architectures for a powerful, flexible and efficient computational model, we propose the dendritic spiking neuron (DendSN). DendSN explicitly models dendritic morphology and nonlinear integration in a streamlined design, leading to substantially higher expressivity than point neurons and wide compatibility with modern deep SNN architectures. Leveraging the efficient formulation and high-performance Triton kernels, dendritic SNNs (DendSNNs) can be efficiently trained and easily scaled to deeper networks. Experiments show that DendSNNs consistently outperform conventional SNNs on classification tasks. Furthermore, inspired by dendritic modulation and synaptic clustering, we introduce the dendritic branch gating (DBG) algorithm for task-incremental learning, which effectively reduces inter-task interference. Additional evaluations show that DendSNNs exhibit superior robustness to noise and adversarial attacks, along with improved generalization in few-shot learning scenarios. Our work firstly demonstrates the possibility of training deep SNNs with multiple nonlinear dendritic branches, and comprehensively analyzes the impact of dendrite computation on representation learning across various machine learning settings, thereby offering a fresh perspective on advancing SNN design.

摘要: 树突计算赋予生物神经元丰富的非线性积分和高表示能力，但在现有的深脉冲神经网络（SNN）中，它在很大程度上是缺失的。虽然详细的多隔室模型可以捕获树突计算，但其高计算成本和有限的灵活性使其对于深度学习不切实际。为了结合树突计算和深度网络架构的优势，构建一个强大、灵活和高效的计算模型，我们提出了树突尖峰神经元（DendSN）。DendSN在流线型设计中明确建模树突状形态和非线性整合，导致比点神经元更高的表达能力以及与现代深度SNN架构的广泛兼容性。利用高效的公式和高性能的Triton内核，树突状SNN（DendSNN）可以有效地训练并轻松扩展到更深的网络。实验表明，DendSNN在分类任务上始终优于传统SNN。此外，受树突调制和突触聚类的启发，我们引入了用于任务增量学习的树突分支门控（DBG）算法，该算法有效地减少了任务间的干扰。额外的评估表明，DendSNNs对噪声和对抗性攻击具有出色的鲁棒性，并且在少量学习场景中具有更好的泛化能力。我们的工作首先展示了用多个非线性树枝状分支训练深度SNN的可能性，并全面分析了树枝状计算对各种机器学习环境中表示学习的影响，从而为推进SNN设计提供了新的视角。



## **50. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

Accepted by ICCV-2025

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2406.05491v4) [paper-pdf](https://arxiv.org/pdf/2406.05491v4)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.

摘要: 通过充分利用多模式对齐，视觉语言预训练（VLP）模型在许多应用中展现出前所未有的能力。然而，之前的研究表明，它们很容易受到恶意制作的对抗样本的影响。尽管最近取得了成功，但这些方法通常是特定于实例的，并且需要为每个输入样本生成扰动。在本文中，我们揭示了VLP模型也容易受到实例不可知的普遍对抗扰动（UAP）的影响。具体来说，我们设计了一种新颖的具有跨模式条件的对比训练扰动生成器（C-PGC）来实现攻击。鉴于关键的多模式对齐是通过先进的对比学习技术实现的，我们设计将这一强大的武器用来对抗自己，即使用恶意版本的对比学习来基于我们精心制作的正和负图像-文本对来训练C-PGC，以本质上破坏VLP模型学习的对齐关系。此外，C-PGC充分利用视觉与语言（V+L）场景的特点，整合单模式和跨模式信息作为有效指导。大量实验表明，C-PGC成功迫使对抗样本离开VLP模型特征空间中的原始区域，从而从本质上增强了对各种受害者模型和V+L任务的攻击。GitHub存储库可访问https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks。



