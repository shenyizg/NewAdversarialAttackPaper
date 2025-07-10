# Latest Adversarial Attack Papers
**update at 2025-07-10 15:59:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Protecting Classifiers From Attacks**

保护分类器免受攻击 stat.ML

Published in Statistical Science:  https://projecteuclid.org/journals/statistical-science/volume-39/issue-3/Protecting-Classifiers-from-Attacks/10.1214/24-STS922.full

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2004.08705v2) [paper-pdf](http://arxiv.org/pdf/2004.08705v2)

**Authors**: Victor Gallego, Roi Naveiro, Alberto Redondo, David Rios Insua, Fabrizio Ruggeri

**Abstract**: In multiple domains such as malware detection, automated driving systems, or fraud detection, classification algorithms are susceptible to being attacked by malicious agents willing to perturb the value of instance covariates to pursue certain goals. Such problems pertain to the field of adversarial machine learning and have been mainly dealt with, perhaps implicitly, through game-theoretic ideas with strong underlying common knowledge assumptions. These are not realistic in numerous application domains in relation to security and business competition. We present an alternative Bayesian decision theoretic framework that accounts for the uncertainty about the attacker's behavior using adversarial risk analysis concepts. In doing so, we also present core ideas in adversarial machine learning to a statistical audience. A key ingredient in our framework is the ability to sample from the distribution of originating instances given the, possibly attacked, observed ones. We propose an initial procedure based on approximate Bayesian computation usable during operations; within it, we simulate the attacker's problem taking into account our uncertainty about his elements. Large-scale problems require an alternative scalable approach implementable during the training stage. Globally, we are able to robustify statistical classification algorithms against malicious attacks.

摘要: 在恶意软件检测、自动驾驶系统或欺诈检测等多个领域中，分类算法很容易受到恶意代理的攻击，恶意代理愿意干扰实例协变量的价值以追求某些目标。此类问题涉及对抗性机器学习领域，并且主要通过具有强大基础常识假设的博弈论思想来解决（也许是隐含的）。这些在许多与安全和业务竞争相关的应用程序领域中是不现实的。我们提出了一个替代的Bayesian决策理论框架，该框架使用对抗风险分析概念来解释攻击者行为的不确定性。在此过程中，我们还向统计受众展示了对抗机器学习的核心思想。我们的框架中的一个关键因素是能够从原始实例的分布中进行采样，这些实例可能受到攻击，并被观察到。我们提出了一个初步的过程中可用的近似贝叶斯计算的基础上，在它里面，我们模拟攻击者的问题，考虑到我们的不确定性，他的元素。大规模的问题需要一个替代的可扩展的方法在训练阶段实施。在全球范围内，我们能够增强统计分类算法抵御恶意攻击。



## **2. Robust and Safe Traffic Sign Recognition using N-version with Weighted Voting**

使用加权投票的N版本鲁棒且安全的交通标志识别 cs.LG

27 pages including appendix, 1 figure

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06907v1) [paper-pdf](http://arxiv.org/pdf/2507.06907v1)

**Authors**: Linyun Gao, Qiang Wen, Fumio Machida

**Abstract**: Autonomous driving is rapidly advancing as a key application of machine learning, yet ensuring the safety of these systems remains a critical challenge. Traffic sign recognition, an essential component of autonomous vehicles, is particularly vulnerable to adversarial attacks that can compromise driving safety. In this paper, we propose an N-version machine learning (NVML) framework that integrates a safety-aware weighted soft voting mechanism. Our approach utilizes Failure Mode and Effects Analysis (FMEA) to assess potential safety risks and assign dynamic, safety-aware weights to the ensemble outputs. We evaluate the robustness of three-version NVML systems employing various voting mechanisms against adversarial samples generated using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Experimental results demonstrate that our NVML approach significantly enhances the robustness and safety of traffic sign recognition systems under adversarial conditions.

摘要: 自动驾驶作为机器学习的关键应用正在迅速发展，但确保这些系统的安全性仍然是一个严峻的挑战。交通标志识别是自动驾驶汽车的重要组成部分，特别容易受到可能危及驾驶安全的对抗攻击。在本文中，我们提出了一个N版本机器学习（NVML）框架，该框架集成了安全感知加权软投票机制。我们的方法利用故障模式与影响分析（EIA）来评估潜在的安全风险，并为总体输出分配动态的安全意识权重。我们评估了采用各种投票机制的三版本NVML系统针对使用快速梯度符号法（FGSM）和投影梯度下降（PVD）攻击生成的对抗样本的稳健性。实验结果表明，我们的NVML方法显着增强了交通标志识别系统在对抗条件下的鲁棒性和安全性。



## **3. A Single-Point Measurement Framework for Robust Cyber-Attack Diagnosis in Smart Microgrids Using Dual Fractional-Order Feature Analysis**

使用双分数阶特征分析的智能微电网鲁棒网络攻击诊断的单点测量框架 eess.SY

8 pages, 10 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06890v1) [paper-pdf](http://arxiv.org/pdf/2507.06890v1)

**Authors**: Yifan Wang

**Abstract**: Cyber-attacks jeopardize the safe operation of smart microgrids. At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modelling assumptions that are untenable under single-sensor constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves low-latency fault localisation and cyber-attack detection using only one VPQ (Voltage-Power-Reactive-power) sensor. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Gr\"unwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritise the most challenging samples. Experiments on a four-inverter microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6 % (bias), 94.0 % (noise), 92.8 % (data replacement), and 95.7 % (replay), while sustaining 96.7 % under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of smart microgrids.

摘要: 网络攻击危及智能微电网的安全运行。与此同时，现有的诊断方法要么依赖于昂贵的多点仪器，要么依赖于在单传感器约束下站不住脚的严格建模假设。本文提出了一种分数阶存储器增强型攻击诊断方案（FO-MADS），仅使用一个VPQ（电压功率反应功率）传感器即可实现低延迟故障定位和网络攻击检测。FO-MADS首先通过联合应用Caputo和Gr ' unwald-Letnikov衍生物来构建双分数阶特征库，从而放大VPQ信号中的微扰动和缓慢漂移。然后，两级分层分类器确定受影响的逆变器并隔离有故障的绝缘栅双极开关，有效地缓解类别不平衡。通过渐进记忆回放对抗训练（PMR-AT）进一步加强稳健性，其攻击感知损失通过在线硬示例挖掘（OHEEM）动态重新加权，以优先考虑最具挑战性的样本。在四种攻击场景下，在包含1个正常和24个故障类别的四逆变器微电网测试台上进行的实验表明，诊断准确率为96.6%（偏差）、94.0%（噪音）、92.8%（数据替换）和95.7%（重播），而在无攻击条件下保持96.7%。这些结果使FO-MADS成为一种具有成本效益且易于部署的解决方案，可以显着增强智能微电网的网络物理弹性。



## **4. IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization**

NAP：通过感知本地化和扰动优化进行隐形对抗补丁攻击 cs.CV

Published in ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06856v1) [paper-pdf](http://arxiv.org/pdf/2507.06856v1)

**Authors**: Subrat Kishore Dutta, Xiao Zhang

**Abstract**: Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective.

摘要: 尽管只修改了很小的局部输入区域，但对抗性补丁可以极大地改变计算机视觉模型的预测。然而，现有方法要么无法在有针对性的攻击场景下令人满意地表现，要么无法产生上下文一致的对抗补丁，导致它们很容易被人类检查员注意到，并且对于自动补丁防御来说不够隐蔽。本文中，我们介绍了NAP，这是一种新型攻击框架，它基于感知定位和扰动优化方案生成高度不可见的对抗补丁。具体来说，TIP首先通过利用类定位和敏感度地图来搜索放置补丁的适当位置，平衡补丁位置对受害者模型预测和人类视觉系统的敏感性，然后采用感知规范化的对抗损失和梯度更新规则，优先考虑颜色稳定性以优化不可见干扰。各种图像基准和模型架构的全面实验表明，与现有基线相比，TIP在目标设置中始终实现有竞争力的攻击成功率，并且补丁不可见性显着提高。除了人类高度难以察觉之外，TIP还被证明足够隐蔽，足以使几种最先进的补丁防御无效。



## **5. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

LLM基于代理的完全计算机接管攻击的阴暗面 cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06850v1) [paper-pdf](http://arxiv.org/pdf/2507.06850v1)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.

摘要: 大型语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成具有前所未有的能力。然而，这些系统引入了前所未有的安全漏洞，超出了传统的即时注入攻击的范围。本文首次对LLM代理进行了全面评估，作为攻击载体，这些攻击载体能够通过利用自主实体相互交互和影响的代理人工智能系统内的信任边界来实现完全的计算机接管。我们证明，对手可以利用三种不同的攻击表面--直接提示注入、RAG后门攻击和代理间信任利用--来强迫流行的LLM（包括GPT-4 o、Claude-4和Gemini-2.5）在受害者机器上自主安装和执行恶意软件。我们对17个最先进的LLM的评估揭示了一个令人震惊的漏洞层次结构：虽然41.2%的模型屈服于直接即时注入，但52.9%的模型容易受到RAG后门攻击，并且关键的82.4%可以通过代理间信任利用而受到损害。值得注意的是，我们发现成功抵抗直接恶意命令的LLM将在对等代理请求时执行相同的有效负载，这揭示了当前多代理安全模型中的一个根本缺陷。我们的研究结果表明，只有5.9%的测试模型（1/17）被证明能够抵抗所有攻击载体，其中大多数表现出依赖于上下文的安全行为，从而创建了可利用的盲点。我们的研究结果还强调了提高对LLM安全风险的认识和研究的必要性，这表明网络安全威胁的范式转变，人工智能工具本身成为复杂的攻击载体。



## **6. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

PBCAT：针对对象检测物理上可实现的攻击的基于补丁的复合对抗训练 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.23581v2) [paper-pdf](http://arxiv.org/pdf/2506.23581v2)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.

摘要: 对象检测在许多安全敏感应用程序中发挥着至关重要的作用。然而，最近的几项研究表明，对象检测器很容易被物理上可实现的攻击所愚弄，例如对抗补丁和最近的对抗纹理，这些攻击构成了现实而紧迫的威胁。对抗训练（AT）被认为是对抗攻击的最有效防御。虽然AT在分类模型的$l_\infty$攻击设置中得到了广泛研究，但针对对象检测器物理上可实现的攻击的AT的探索有限。早期的尝试只是为了防御对抗补丁，这使得AT能够对抗更广泛的物理可实现的攻击，但尚未得到充分的探索。在这项工作中，我们考虑使用统一的AT方法来防御各种物理上可实现的攻击。我们提出了PBCAT，这是一种新型的基于补丁的复合对抗训练策略。PBCAT通过结合小区域梯度引导的对抗补丁和覆盖整个图像的不可感知的全局对抗扰动来优化模型。通过这些设计，PBCAT不仅有潜力防御对抗补丁，还有潜力防御不可见的物理可实现的攻击，例如对抗纹理。在多个环境中进行的大量实验表明，与最先进的防御方法相比，PBCAT显着提高了针对各种物理可实现攻击的鲁棒性。值得注意的是，在最近的一次对抗性纹理攻击下，它比之前的防御方法提高了29.7%。



## **7. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

尾部感知对抗攻击：高效LLM越狱的分布式方法 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代的有害响应，忽视了LLM固有的随机性。在本文中，我们提出了一种新颖的对抗稳健性评估框架，该框架对整个输出分布（包括尾部风险）进行显式建模，为模型大规模稳健性提供更好的估计。通过将攻击过程描述为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可将ASB提高高达48%，并将效率提高高达两个数量级。我们的框架还使我们能够分析不同的攻击算法如何影响输出伤害分布。令人惊讶的是，我们发现大多数优化策略对输出危害影响很小。最后，我们引入了一个基于熵最大化的无数据概念验证目标，以演示我们的尾部感知视角如何实现新的优化目标。总体而言，我们的研究结果强调了尾部感知攻击和评估协议对于准确评估和加强LLM安全性的重要性。



## **8. Distributed Fault-Tolerant Multi-Robot Cooperative Localization in Adversarial Environments**

对抗环境下的分布式故障多机器人协同定位 cs.RO

Accepted to IROS 2025 Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06750v1) [paper-pdf](http://arxiv.org/pdf/2507.06750v1)

**Authors**: Tohid Kargar Tasooji, Ramviyas Parasuraman

**Abstract**: In multi-robot systems (MRS), cooperative localization is a crucial task for enhancing system robustness and scalability, especially in GPS-denied or communication-limited environments. However, adversarial attacks, such as sensor manipulation, and communication jamming, pose significant challenges to the performance of traditional localization methods. In this paper, we propose a novel distributed fault-tolerant cooperative localization framework to enhance resilience against sensor and communication disruptions in adversarial environments. We introduce an adaptive event-triggered communication strategy that dynamically adjusts communication thresholds based on real-time sensing and communication quality. This strategy ensures optimal performance even in the presence of sensor degradation or communication failure. Furthermore, we conduct a rigorous analysis of the convergence and stability properties of the proposed algorithm, demonstrating its resilience against bounded adversarial zones and maintaining accurate state estimation. Robotarium-based experiment results show that our proposed algorithm significantly outperforms traditional methods in terms of localization accuracy and communication efficiency, particularly in adversarial settings. Our approach offers improved scalability, reliability, and fault tolerance for MRS, making it suitable for large-scale deployments in real-world, challenging environments.

摘要: 在多机器人系统（MRS）中，协作定位是提高系统鲁棒性和可扩展性的关键任务，特别是在GPS拒绝或通信受限的环境中。然而，对抗性攻击，如传感器操纵和通信干扰，对传统定位方法的性能提出了重大挑战。在本文中，我们提出了一种新的分布式容错合作定位框架，以提高对抗性环境中的传感器和通信中断的弹性。我们介绍了一种自适应事件触发的通信策略，动态调整通信阈值的基础上实时感知和通信质量。即使存在传感器降级或通信故障，该策略也确保了最佳性能。此外，我们对所提出算法的收敛性和稳定性属性进行了严格分析，展示了其对有界对抗区的弹性并保持准确的状态估计。基于机器人馆的实验结果表明，我们提出的算法在定位准确性和通信效率方面显着优于传统方法，特别是在对抗环境中。我们的方法为MR提供了改进的可扩展性、可靠性和耐故障性，使其适合在现实世界、具有挑战性的环境中进行大规模部署。



## **9. Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment**

通过去偏高置信Logit对齐实现对抗鲁棒性 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2408.06079v2) [paper-pdf](http://arxiv.org/pdf/2408.06079v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Shaozi Li, Zhiming Luo

**Abstract**: Despite the remarkable progress of deep neural networks (DNNs) in various visual tasks, their vulnerability to adversarial examples raises significant security concerns. Recent adversarial training methods leverage inverse adversarial attacks to generate high-confidence examples, aiming to align adversarial distributions with high-confidence class regions. However, our investigation reveals that under inverse adversarial attacks, high-confidence outputs are influenced by biased feature activations, causing models to rely on background features that lack a causal relationship with the labels. This spurious correlation bias leads to overfitting irrelevant background features during adversarial training, thereby degrading the model's robust performance and generalization capabilities. To address this issue, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that aligns adversarial logits with debiased high-confidence logits and restores proper attention by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art robustness on both CIFAR and ImageNet-1K benchmarks, while significantly improving generalization by mitigating the feature bias inherent in inverse adversarial training approaches. Code is available at https://github.com/KejiaZhang-Robust/DHAT.

摘要: 尽管深度神经网络（DNN）在各种视觉任务中取得了显着进展，但它们对对抗性示例的脆弱性引发了严重的安全问题。最近的对抗训练方法利用反向对抗攻击来生成高置信度示例，旨在将对抗分布与高置信度类别区域保持一致。然而，我们的研究表明，在反向对抗攻击下，高置信度输出受到有偏见的特征激活的影响，导致模型依赖于与标签缺乏因果关系的背景特征。这种虚假的相关偏差导致对抗训练期间过度匹配不相关的背景特征，从而降低模型的稳健性能和概括能力。为了解决这个问题，我们提出了去偏置的高置信对抗训练（DHAT），这是一种新颖的方法，可以将对抗逻辑与去偏置的高置信逻辑对齐，并通过增强前景逻辑的垂直性来恢复适当的注意力。大量实验表明，DHAT在CIFAR和ImageNet-1 K基准测试上都实现了最先进的鲁棒性，同时通过减轻反向对抗训练方法固有的特征偏差来显着提高概括性。代码可在https://github.com/KejiaZhang-Robust/DHAT上获得。



## **10. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

评估和改进大型语言模型的鲁棒性：调查和未来方向 cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.

摘要: 近年来，大型语言模型（LLM）因其理解和生成自然语言的能力而受到了广泛关注。随着快速发展和广泛应用（例如，代理人，联合情报），LLM的稳健性受到了越来越多的关注。作为许多人工智能应用的核心大脑，LLM的稳健性要求模型不仅要生成一致的内容，还要在处理意外的应用场景（例如，有毒提示、有限的噪音域数据、向外分布（OOD）应用程序等）。在这篇调查论文中，我们对LLM的稳健性进行了彻底的审查，旨在提供该领域的全面概念和方法术语并促进社区发展。具体来说，我们首先给出了LLM稳健性的正式定义，并给出了这篇调查论文的收集协议。然后，根据受干扰的输入类型，我们从以下角度组织本次调查：1）对抗稳健性：解决提示被故意操纵的问题，例如噪音提示、长上下文、数据攻击等; 2）OOD稳健性：处理意想不到的现实世界应用场景，例如OOD检测、零镜头传输、幻觉等; 3）稳健性评估：总结用于验证LLM稳健性的新评估数据集、指标和工具。在从各个角度回顾了代表性作品后，我们讨论并强调了该领域未来的机会和研究方向。同时，我们还组织相关工作并提供易于搜索的项目（https：//github.com/zhangkunzk/Awesome-LLM-Robustness-papers）来支持社区。



## **11. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **12. Dual State-space Fidelity Blade (D-STAB): A Novel Stealthy Cyber-physical Attack Paradigm**

双状态空间富达刀片（D-STAB）：一种新型隐形网络物理攻击范式 eess.SY

accepted by 2025 American Control Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06492v1) [paper-pdf](http://arxiv.org/pdf/2507.06492v1)

**Authors**: Jiajun Shen, Hao Tu, Fengjun Li, Morteza Hashemi, Di Wu, Huazhen Fang

**Abstract**: This paper presents a novel cyber-physical attack paradigm, termed the Dual State-Space Fidelity Blade (D-STAB), which targets the firmware of core cyber-physical components as a new class of attack surfaces. The D-STAB attack exploits the information asymmetry caused by the fidelity gap between high-fidelity and low-fidelity physical models in cyber-physical systems. By designing precise adversarial constraints based on high-fidelity state-space information, the attack induces deviations in high-fidelity states that remain undetected by defenders relying on low-fidelity observations. The effectiveness of D-STAB is demonstrated through a case study in cyber-physical battery systems, specifically in an optimal charging task governed by a Battery Management System (BMS).

摘要: 本文提出了一种新型的网络物理攻击范式，称为双状态空间富达刀片（D-STAB），其目标是核心网络物理组件的硬件，作为一类新型攻击面。D-STAB攻击利用了网络物理系统中高保真度和低保真度物理模型之间的保真度差距造成的信息不对称。通过基于高保真度状态空间信息设计精确的对抗约束，攻击会导致高保真度状态的偏差，而依赖低保真度观察的防御者仍然无法检测到这些偏差。D-STAB的有效性通过网络物理电池系统中的案例研究得到了证明，特别是在由电池管理系统（BMC）管理的最佳充电任务中。



## **13. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于部署LLM至关重要，以确保许多高风险应用程序中人机交互的透明度、信任和安全。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们引入了一个新颖的框架，通过干扰和基于越狱的方法攻击言语信心分数，并表明这些攻击可能会显着危及言语信心估计并导致答案频繁变化。我们研究了各种提示策略、模型大小和应用领域，揭示了当前的信心激发方法很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了迫切需要设计更强大的机制来表达LLM的信心，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **14. Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks on Web3 Agents**

具有虚假记忆的真实人工智能代理：对Web 3代理的致命上下文操纵攻击 cs.CR

19 pages, 14 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2503.16248v3) [paper-pdf](http://arxiv.org/pdf/2503.16248v3)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: AI agents integrated with Web3 offer autonomy and openness but raise security concerns as they interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. It expands on traditional prompt injection and reveals a more stealthy and persistent threat: memory injection. Using ElizaOS, a representative decentralized AI agent framework for automated Web3 operations, we showcase that malicious injections into prompts or historical records can trigger unauthorized asset transfers and protocol violations which could be financially devastating in reality. To quantify these risks, we introduce CrAIBench, a Web3-focused benchmark covering 150+ realistic blockchain tasks. such as token transfers, trading, bridges, and cross-chain interactions, and 500+ attack test cases using context manipulation. Our evaluation results confirm that AI models are significantly more vulnerable to memory injection compared to prompt injection. Finally, we evaluate a comprehensive defense roadmap, finding that prompt-injection defenses and detectors only provide limited protection when stored context is corrupted, whereas fine-tuning-based defenses substantially reduce attack success rates while preserving performance on single-step tasks. These results underscore the urgent need for AI agents that are both secure and fiduciarily responsible in blockchain environments.

摘要: 与Web 3集成的人工智能代理提供了自主性和开放性，但在与金融协议和不可变智能合同交互时会引发安全问题。本文研究了基于区块链的金融生态系统中人工智能代理在现实世界场景中面临对抗威胁时的脆弱性。我们引入了上下文操纵的概念--这是一种全面的攻击载体，可以利用不受保护的上下文表面，包括输入通道、内存模块和外部数据源。它扩展了传统的即时注入，揭示了一种更隐蔽和持久的威胁：记忆注入。使用ElizaOS（用于自动化Web 3操作的代表性去中心化人工智能代理框架），我们展示了恶意注入提示或历史记录可能会引发未经授权的资产转移和协议违规，这在现实中可能会造成经济上的毁灭性。为了量化这些风险，我们引入了CrAIBench，这是一个专注于Web 3的基准测试，涵盖150多个现实的区块链任务。例如代币转移、交易、桥梁和跨链交互，以及500多个使用上下文操纵的攻击测试案例。我们的评估结果证实，与即时注入相比，人工智能模型明显更容易受到记忆注入的影响。最后，我们评估了全面的防御路线图，发现预算注入防御和检测器仅在存储上下文被破坏时提供有限的保护，而基于微调的防御可以大幅降低攻击成功率，同时保留一步任务的性能。这些结果凸显了对区块链环境中既安全又负信托责任的人工智能代理的迫切需求。



## **15. Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers**

更改单个单词即可：为文本分类器设计攻击和防御 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2401.17196v2) [paper-pdf](http://arxiv.org/pdf/2401.17196v2)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.

摘要: 在文本分类中，创建对抗性示例意味着微妙地扰乱句子中的几个词而不改变其含义，导致其被分类器错误分类。一个令人担忧的观察是，现有方法生成的很大一部分对抗性例子只改变了一个词。这种单字扰动漏洞代表了分类器的一个重大弱点，恶意用户可以利用它有效地创建大量对抗性示例。本文研究了这个问题并做出了以下关键贡献：（1）我们引入了一种新型指标\r{ho}来定量评估分类器对单字扰动的鲁棒性。(2)我们提出了SP-Attack，旨在利用单字扰动漏洞，实现更高的攻击成功率，更好地保留句子含义，同时与最先进的对抗方法相比降低了计算成本。(3)我们提出SP-Defense，旨在通过在学习中应用数据增强来改进\r{ho}。对4个数据集以及BERT和DistilBERT分类器的实验结果表明，SP-Defense在两个分类器上分别提高了14.6%和13.9%，并将SP-Attack的攻击成功率分别降低了30.4%和21.2%，并降低了涉及多字扰动的现有攻击方法的攻击成功率。



## **16. On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving**

自动驾驶中视觉语言模型对视觉感知攻击的自然鲁棒性 cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.11472v2) [paper-pdf](http://arxiv.org/pdf/2506.11472v2)

**Authors**: Pedram MohajerAnsari, Amir Salarpour, Michael Kühr, Siyu Huang, Mohammad Hamad, Sebastian Steinhorst, Habeeb Olufowobi, Mert D. Pesé

**Abstract**: Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.

摘要: 自动驾驶汽车（AV）依赖深度神经网络（DNN）来执行交通标志识别（TSB）、自动车道定中心（ALC）和车辆检测（VD）等关键任务。然而，这些模型很容易受到可能导致错误分类并损害安全性的攻击。传统的防御机制，包括对抗训练，通常会降低良性准确性，并且无法针对不可见的攻击进行概括。在这项工作中，我们介绍了车辆视觉语言模型（V2 LM），这是专门用于AV感知的微调视觉语言模型。我们的研究结果表明，V2 LM本质上对不可见的攻击表现出卓越的鲁棒性，无需对抗训练，在对抗条件下保持比传统DNN显着更高的准确性。我们评估了两种部署策略：Solo模式（单个V2 LM处理特定的感知任务）和Tandem模式（单个统一V2 LM同时针对多个任务进行微调）。实验结果显示，DNN在攻击下性能下降33%至46%，而V2 LM保持对抗准确性，平均下降不到8%。Tandem模式进一步提供了一种内存高效的替代方案，同时实现了与Solo模式相当的稳健性。我们还探索将V2 LM集成为AV感知的并行组件，以增强对抗威胁的弹性。我们的结果表明，V2 LM为更安全和更有弹性的AV感知系统提供了一条有希望的途径。



## **17. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges [Experiment, Analysis \& Benchmark]**

沼泽上的对冲基金：分析区块链桥梁中的模式、漏洞和防御措施[实验、分析和基准] cs.ET

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06156v1) [paper-pdf](http://arxiv.org/pdf/2507.06156v1)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.

摘要: 区块链桥梁已成为实现不同区块链网络互操作性的重要基础设施，每月桥梁交易量超过240亿美元。然而，随着它们的日益普及，安全漏洞也不成比例地增加，使它们成为Web 3中最大的财务损失来源。为了实现跨链生态系统的稳健和可持续发展，了解和解决这些脆弱性至关重要。在这项研究中，我们对区块链桥梁设计和安全进行了全面的系统化。我们定义了三个桥梁安全先验，正式确定了13个突出桥梁的架构结构，并确定了23个基于现实世界区块链漏洞的攻击向量。在此基础上，我们评估了43种有代表性的攻击场景，并引入了一个分层的威胁模型，该模型可以捕获源链、链下和目标链组件的安全故障。   我们在静态代码和交易网络层面的分析揭示了反复出现的设计缺陷，特别是在访问控制、验证者信任假设和验证逻辑方面，并根据交易级跟踪识别了对抗行为的关键模式。为了支持未来的发展，我们提出了一个决策框架的桥梁架构设计，以及防御机制，如分层验证和断路器。这项工作为评估桥梁安全性提供了数据驱动的基础，并为标准化弹性跨链基础设施奠定了基础。



## **18. The bitter lesson of misuse detection**

误用检测的惨痛教训 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06282v1) [paper-pdf](http://arxiv.org/pdf/2507.06282v1)

**Authors**: Hadrien Mariaccia, Charbel-Raphaël Segerie, Diego Dorn

**Abstract**: Prior work on jailbreak detection has established the importance of adversarial robustness for LLMs but has largely focused on the model ability to resist adversarial inputs and to output safe content, rather than the effectiveness of external supervision systems. The only public and independent benchmark of these guardrails to date evaluates a narrow set of supervisors on limited scenarios. Consequently, no comprehensive public benchmark yet verifies how well supervision systems from the market perform under realistic, diverse attacks. To address this, we introduce BELLS, a Benchmark for the Evaluation of LLM Supervision Systems. The framework is two dimensional: harm severity (benign, borderline, harmful) and adversarial sophistication (direct vs. jailbreak) and provides a rich dataset covering 3 jailbreak families and 11 harm categories. Our evaluations reveal drastic limitations of specialized supervision systems. While they recognize some known jailbreak patterns, their semantic understanding and generalization capabilities are very limited, sometimes with detection rates close to zero when asking a harmful question directly or with a new jailbreak technique such as base64 encoding. Simply asking generalist LLMs if the user question is "harmful or not" largely outperforms these supervisors from the market according to our BELLS score. But frontier LLMs still suffer from metacognitive incoherence, often responding to queries they correctly identify as harmful (up to 30 percent for Claude 3.7 and greater than 50 percent for Mistral Large). These results suggest that simple scaffolding could significantly improve misuse detection robustness, but more research is needed to assess the tradeoffs of such techniques. Our results support the "bitter lesson" of misuse detection: general capabilities of LLMs are necessary to detect a diverse array of misuses and jailbreaks.

摘要: 之前关于越狱检测的工作已经确定了对抗鲁棒性对LLM的重要性，但主要关注的是模型抵抗对抗输入和输出安全内容的能力，而不是外部监督系统的有效性。迄今为止，这些护栏的唯一公开且独立的基准在有限的情况下评估了有限的监管人员。因此，目前还没有全面的公共基准来验证市场监管系统在现实、多样化的攻击下的表现如何。为了解决这个问题，我们引入了BELLS，这是LLM监督系统评估的基准。该框架是二维的：伤害严重性（良性、边缘、有害）和对抗复杂性（直接与越狱），并提供了涵盖3个越狱家庭和11个伤害类别的丰富数据集。我们的评估揭示了专业监督系统的巨大局限性。虽然它们识别了一些已知的越狱模式，但它们的语义理解和概括能力非常有限，有时在直接询问有害问题或使用base 64编码等新的越狱技术时，检测率接近于零。根据我们的BELLS评分，简单地询问多面手LLM用户问题是否“有害”在很大程度上优于这些市场监管人员。但前沿LLM仍然存在元认知不一致的问题，经常对他们正确识别为有害的查询做出回应（Claude 3.7的这一比例高达30%，Mistral Large的这一比例超过50%）。这些结果表明，简单的支架可以显着提高误用检测的鲁棒性，但需要更多的研究来评估此类技术的权衡。我们的结果支持了滥用检测的“惨痛教训”：LLM的通用功能对于检测各种滥用和越狱是必要的。



## **19. ScoreAdv: Score-based Targeted Generation of Natural Adversarial Examples via Diffusion Models**

ScoreAdv：通过扩散模型基于分数的有针对性地生成自然对抗示例 cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06078v1) [paper-pdf](http://arxiv.org/pdf/2507.06078v1)

**Authors**: Chihan Huang, Hao Tang

**Abstract**: Despite the success of deep learning across various domains, it remains vulnerable to adversarial attacks. Although many existing adversarial attack methods achieve high success rates, they typically rely on $\ell_{p}$-norm perturbation constraints, which do not align with human perceptual capabilities. Consequently, researchers have shifted their focus toward generating natural, unrestricted adversarial examples (UAEs). GAN-based approaches suffer from inherent limitations, such as poor image quality due to instability and mode collapse. Meanwhile, diffusion models have been employed for UAE generation, but they still rely on iterative PGD perturbation injection, without fully leveraging their central denoising capabilities. In this paper, we introduce a novel approach for generating UAEs based on diffusion models, named ScoreAdv. This method incorporates an interpretable adversarial guidance mechanism to gradually shift the sampling distribution towards the adversarial distribution, while using an interpretable saliency map to inject the visual information of a reference image into the generated samples. Notably, our method is capable of generating an unlimited number of natural adversarial examples and can attack not only classification models but also retrieval models. We conduct extensive experiments on ImageNet and CelebA datasets, validating the performance of ScoreAdv across ten target models in both black-box and white-box settings. Our results demonstrate that ScoreAdv achieves state-of-the-art attack success rates and image quality. Furthermore, the dynamic balance between denoising and adversarial perturbation enables ScoreAdv to remain robust even under defensive measures.

摘要: 尽管深度学习在各个领域取得了成功，但它仍然容易受到对抗攻击。尽管许多现有的对抗攻击方法取得了很高的成功率，但它们通常依赖于$\ell_{p}$-norm扰动约束，这与人类的感知能力不一致。因此，研究人员将重点转向生成自然的、不受限制的对抗性例子（UAE）。基于GAN的方法存在固有的局限性，例如由于不稳定和模式崩溃而导致的图像质量差。与此同时，扩散模型已被用于阿联酋一代，但它们仍然依赖于迭代PVD扰动注入，而没有充分利用其核心去噪能力。本文中，我们介绍了一种基于扩散模型生成UAE的新型方法，名为ScoreAdv。该方法结合了可解释的对抗引导机制，将采样分布逐渐转向对抗分布，同时使用可解释的显着图将参考图像的视觉信息注入到生成的样本中。值得注意的是，我们的方法能够生成无限数量的自然对抗示例，并且不仅可以攻击分类模型，还可以攻击检索模型。我们对ImageNet和CelebA数据集进行了广泛的实验，验证了ScoreAdv在黑盒和白盒设置下在十个目标模型上的性能。我们的结果表明ScoreAdv实现了最先进的攻击成功率和图像质量。此外，去噪和对抗性扰动之间的动态平衡使ScoreAdv即使在防御措施下也能够保持稳健。



## **20. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

CAVGAN：通过对其内部代表的生成性对抗攻击统一LLM的越狱和辩护 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.

摘要: 安全对齐使大型语言模型（LLM）能够获得针对恶意查询的保护，但各种越狱攻击方法揭示了这种安全机制的漏洞。之前的研究已经孤立了LLM越狱攻击和防御。我们分析了LLM的安全保护机制，提出了攻击与防御相结合的框架。我们的方法基于LLM中间层嵌入的线性可分离性质，以及越狱攻击的本质，旨在嵌入有害问题并将其转移到安全区域。我们利用生成对抗网络（GAN）来学习LLM内部的安全判断边界，以实现高效的越狱攻击和防御。实验结果表明，我们的方法在三种流行的LLM中平均越狱成功率为88.85%，而在最先进的越狱数据集上的防御成功率平均达到84.17%。这不仅验证了我们方法的有效性，还揭示了LLM的内部安全机制，为增强模型安全性提供了新的见解。代码和数据可在https://github.com/NLPGM/CAVGAN上获取。



## **21. TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data**

TuneShield：在对不可信数据进行微调的同时减轻对话人工智能中的毒性 cs.CR

Pre-print

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05660v1) [paper-pdf](http://arxiv.org/pdf/2507.05660v1)

**Authors**: Aravind Cheruvu, Shravya Kanchi, Sifat Muhammad Abdullah, Nicholas Kong, Daphne Yao, Murtuza Jadliwala, Bimal Viswanath

**Abstract**: Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL).

摘要: 基础模型（如LLM）的最新进展彻底改变了对话式AI。聊天机器人越来越多地通过在特定的会话数据集上定制LLM来开发。然而，在这种定制过程中减轻毒性，特别是在处理不可信的训练数据时，仍然是一个重大挑战。为了解决这个问题，我们引入了TuneShield，这是一个防御框架，旨在减轻聊天机器人微调期间的毒性，同时保持会话质量。TuneShield利用基于LLM的毒性分类，利用LLM的描述跟踪功能和安全性对齐来有效识别有毒样本，优于行业API服务。TuneShield基于识别出的有毒样本生成合成对话样本，称为“治愈数据”，使用它们来减轻毒性，同时在微调期间加强理想的行为。它执行对齐过程，以进一步推动聊天机器人产生所需的响应。我们的研究结果表明，TuneShield可以有效地减轻毒性注入攻击，同时保持对话质量，即使毒性分类器不完美或有偏见。事实证明，TuneShield具有抵御适应性对抗和越狱攻击的能力。此外，TuneShield还证明了在基于对话的学习（DBL）期间减轻适应性毒性注入攻击的有效性。



## **22. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.

摘要: 对抗性越狱攻击的最新进展揭示了大型语言模型（LLM）中的显着漏洞，通过日益复杂的提示操纵促进了对对齐保障措施的规避。在本文中，我们提出了MEF，这是一个用于评估黑匣子LLM中漏洞的功能感知多重加密框架。我们的主要见解是，通过根据目标模型的语义理解能力定制越狱策略，可以显着增强它们的有效性。我们提出了一种类型学，根据它们的理解水平将LLM分为I型和II型，并为每种类型设计自适应攻击策略。MEF结合了分层语义突变和双端加密技术，能够规避输入、推理和输出级防御。实验结果证明了我们方法的优越性。值得注意的是，它在GPT-4 o（2025年5月29日发布）上的越狱成功率达到了98.9%。我们的研究结果揭示了当前LLM对齐防御的漏洞。



## **23. How Not to Detect Prompt Injections with an LLM**

如何不使用LLM检测提示注射 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05630v1) [paper-pdf](http://arxiv.org/pdf/2507.05630v1)

**Authors**: Sarthak Choudhary, Divyam Anshumaan, Nils Palumbo, Somesh Jha

**Abstract**: LLM-integrated applications and agents are vulnerable to prompt injection attacks, in which adversaries embed malicious instructions within seemingly benign user inputs to manipulate the LLM's intended behavior. Recent defenses based on $\textit{known-answer detection}$ (KAD) have achieved near-perfect performance by using an LLM to classify inputs as clean or contaminated. In this work, we formally characterize the KAD framework and uncover a structural vulnerability in its design that invalidates its core security premise. We design a methodical adaptive attack, $\textit{DataFlip}$, to exploit this fundamental weakness. It consistently evades KAD defenses with detection rates as low as $1.5\%$ while reliably inducing malicious behavior with success rates of up to $88\%$, without needing white-box access to the LLM or any optimization procedures.

摘要: LLM集成的应用程序和代理很容易受到提示注入攻击，其中对手将恶意指令嵌入看似良性的用户输入中，以操纵LLM的预期行为。最近基于$\textit{known-answer Detection}$（KAD）的防御通过使用LLM将输入分类为清洁或受污染，实现了近乎完美的性能。在这项工作中，我们正式描述了KAD框架的特征，并发现了其设计中使其核心安全前提无效的结构性漏洞。我们设计了一种有条不紊的自适应攻击$\textit{DataFlip}$来利用这一根本弱点。它始终以低至1.5美元的检测率规避KAD防御，同时以高达88美元的成功率可靠地诱导恶意行为，而无需白盒访问LLM或任何优化程序。



## **24. One Surrogate to Fool Them All: Universal, Transferable, and Targeted Adversarial Attacks with CLIP**

愚弄所有人的一个代理人：CLIP的普遍、可转移和有针对性的对抗攻击 cs.CR

21 pages, 15 figures, 18 tables. To appear in the Proceedings of The  ACM Conference on Computer and Communications Security (CCS), 2025

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.19840v2) [paper-pdf](http://arxiv.org/pdf/2505.19840v2)

**Authors**: Binyan Xu, Xilin Dai, Di Tang, Kehuan Zhang

**Abstract**: Deep Neural Networks (DNNs) have achieved widespread success yet remain prone to adversarial attacks. Typically, such attacks either involve frequent queries to the target model or rely on surrogate models closely mirroring the target model -- often trained with subsets of the target model's training data -- to achieve high attack success rates through transferability. However, in realistic scenarios where training data is inaccessible and excessive queries can raise alarms, crafting adversarial examples becomes more challenging. In this paper, we present UnivIntruder, a novel attack framework that relies solely on a single, publicly available CLIP model and publicly available datasets. By using textual concepts, UnivIntruder generates universal, transferable, and targeted adversarial perturbations that mislead DNNs into misclassifying inputs into adversary-specified classes defined by textual concepts.   Our extensive experiments show that our approach achieves an Attack Success Rate (ASR) of up to 85% on ImageNet and over 99% on CIFAR-10, significantly outperforming existing transfer-based methods. Additionally, we reveal real-world vulnerabilities, showing that even without querying target models, UnivIntruder compromises image search engines like Google and Baidu with ASR rates up to 84%, and vision language models like GPT-4 and Claude-3.5 with ASR rates up to 80%. These findings underscore the practicality of our attack in scenarios where traditional avenues are blocked, highlighting the need to reevaluate security paradigms in AI applications.

摘要: 深度神经网络（DNN）已取得广泛成功，但仍然容易受到对抗攻击。通常，此类攻击要么涉及对目标模型的频繁查询，要么依赖于密切反映目标模型的代理模型（通常使用目标模型训练数据的子集进行训练），通过可移植性实现高攻击成功率。然而，在训练数据不可访问且过多查询可能引发警报的现实场景中，制作对抗性示例变得更具挑战性。在本文中，我们介绍了UnivInsurder，这是一种新型攻击框架，仅依赖于单个公开可用的CLIP模型和公开可用的数据集。通过使用文本概念，UnivInvurder生成普遍的、可转移的和有针对性的对抗性扰动，这些扰动误导DNN将输入错误分类到由文本概念定义的对抗指定的类中。   我们广泛的实验表明，我们的方法在ImageNet上实现了高达85%的攻击成功率（ASB），在CIFAR-10上实现了超过99%的攻击成功率，显着优于现有的基于传输的方法。此外，我们还揭示了现实世界的漏洞，表明即使不查询目标模型，UnivInsurder也会损害Google和Baidu等图像搜索引擎的ASB率高达84%，以及GPT-4和Claude-3.5等视觉语言模型的ASB率高达80%。这些发现强调了我们在传统途径被封锁的情况下攻击的实用性，强调了重新评估人工智能应用程序中安全范式的必要性。



## **25. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

Databench：从对抗角度评估深度学习中的数据集审计 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05622v1) [paper-pdf](http://arxiv.org/pdf/2507.05622v1)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available at https://github.com/shaoshuo-ss/DATABench.

摘要: 深度学习在不同领域的广泛应用关键取决于训练数据集的质量和组成。然而，普遍缺乏对其使用情况的披露，引发了严重的隐私和版权问题。数据集审计技术旨在确定特定数据集是否用于训练给定的可疑模型，为解决这些透明度差距提供了有希望的解决方案。虽然之前的工作已经开发了各种审计方法，但它们对专门对抗攻击的弹性在很大程度上仍未被探索。为了弥合这一差距，本文发起了一项全面的研究，从对抗的角度评估数据集审计。我们首先引入一种新颖的分类法，根据现有方法对内部特征（IF）（数据固有的）和外部特征（EF）（人为引入以进行审计）的依赖来对现有方法进行分类。随后，我们制定了两种主要的攻击类型：逃避攻击（旨在隐藏数据集的使用）和伪造攻击（旨在错误地暗示未使用的数据集）。在对现有方法和攻击目标的理解的基础上，我们进一步提出了系统性攻击策略：脱钩、删除和检测逃避;基于对抗性示例的伪造方法。这些公式和策略导致了我们的新基准Databench，其中包括17种规避攻击、5种伪造攻击和9种代表性审计方法。使用Databench进行的广泛评估表明，在对抗环境下，所评估的审计方法都不够稳健或独特。这些发现凸显了开发一种能够承受复杂对抗操纵的更安全、更可靠的数据集审计方法的迫切需要。代码可在https://github.com/shaoshuo-ss/DATABench上获得。



## **26. Massive MIMO-NOMA Systems Secrecy in the Presence of Active Eavesdroppers**

大型MIMO-NOMA系统在活动发射器存在的情况下保持保密 cs.IT

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2105.02215v2) [paper-pdf](http://arxiv.org/pdf/2105.02215v2)

**Authors**: Marziyeh Soltani, Mahtab Mirmohseni, Panos Papadimitratos

**Abstract**: Non-orthogonal multiple access (NOMA) and massive multiple-input multiple-output (MIMO) systems are highly efficient. Massive MIMO systems are inherently resistant to passive attackers (eavesdroppers), thanks to transmissions directed to the desired users. However, active attackers can transmit a combination of legitimate user pilot signals during the channel estimation phase. This way they can mislead the base station (BS) to rotate the transmission in their direction, and allow them to eavesdrop during the downlink data transmission phase. In this paper, we analyse this vulnerability in an improved system model and stronger adversary assumptions, and investigate how physical layer security can mitigate such attacks and ensure secure (confidential) communication. We derive the secrecy outage probability (SOP) and a lower bound on the ergodic secrecy capacity, using stochastic geometry tools when the number of antennas in the BSs tends to infinity. We adapt the result to evaluate the secrecy performance in massive orthogonal multiple access (OMA). We find that appropriate power allocation allows NOMA to outperform OMA in terms of ergodic secrecy rate and SOP.

摘要: 非垂直多址（NOMA）和大规模多输入多输出（MMO）系统效率很高。由于传输定向到所需用户，大规模多输入输出系统本质上可以抵抗被动攻击者（窃听者）。然而，活跃攻击者可以在信道估计阶段传输合法用户导频信号的组合。这样，他们就可以误导基站（BS）以其方向旋转传输，并允许他们在下行链路数据传输阶段窃听。本文中，我们在改进的系统模型和更强的对手假设中分析了这个漏洞，并研究物理层安全如何减轻此类攻击并确保安全（机密）通信。当BS中的天线数量趋于无穷大时，我们使用随机几何工具来推导出保密中断概率（SOP）和各历经保密容量的下限。我们调整结果来评估大规模垂直多址接入（oma）中的保密性能。我们发现，适当的功率分配使NOMA在历经保密率和SOP方面优于oma。



## **27. A Systematization of Security Vulnerabilities in Computer Use Agents**

计算机使用代理安全漏洞的系统化 cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05445v1) [paper-pdf](http://arxiv.org/pdf/2507.05445v1)

**Authors**: Daniel Jones, Giorgio Severi, Martin Pouliot, Gary Lopez, Joris de Gruyter, Santiago Zanella-Beguelin, Justin Song, Blake Bullwinkel, Pamela Cortez, Amanda Minnich

**Abstract**: Computer Use Agents (CUAs), autonomous systems that interact with software interfaces via browsers or virtual machines, are rapidly being deployed in consumer and enterprise environments. These agents introduce novel attack surfaces and trust boundaries that are not captured by traditional threat models. Despite their growing capabilities, the security boundaries of CUAs remain poorly understood. In this paper, we conduct a systematic threat analysis and testing of real-world CUAs under adversarial conditions. We identify seven classes of risks unique to the CUA paradigm, and analyze three concrete exploit scenarios in depth: (1) clickjacking via visual overlays that mislead interface-level reasoning, (2) indirect prompt injection that enables Remote Code Execution (RCE) through chained tool use, and (3) CoT exposure attacks that manipulate implicit interface framing to hijack multi-step reasoning. These case studies reveal deeper architectural flaws across current CUA implementations. Namely, a lack of input provenance tracking, weak interface-action binding, and insufficient control over agent memory and delegation. We conclude by proposing a CUA-specific security evaluation framework and design principles for safe deployment in adversarial and high-stakes settings.

摘要: 计算机使用代理（CUA）是通过浏览器或虚拟机与软件界面交互的自治系统，正在迅速部署在消费者和企业环境中。这些代理引入了传统威胁模型无法捕捉的新型攻击表面和信任边界。尽管CUA的能力不断增强，但其安全界限仍然知之甚少。在本文中，我们在对抗条件下对现实世界的CUA进行了系统性的威胁分析和测试。我们确定了CUA范式特有的七类风险，并深入分析了三种具体的利用场景：（1）通过视觉覆盖的点击劫持，误导界面级推理，（2）间接提示注入，通过连锁工具使用实现远程代码执行（RCE），以及（3）CoT暴露攻击，操纵隐式界面框架以劫持多步推理。这些案例研究揭示了当前CUA实现中更深层次的架构缺陷。即，缺乏输入来源跟踪、界面动作绑定弱以及对代理内存和委托的控制不足。最后，我们提出了一个特定于CUA的安全评估框架和设计原则，用于在对抗和高风险环境中安全部署。



## **28. Adversarial Machine Learning Attacks on Financial Reporting via Maximum Violated Multi-Objective Attack**

对抗性机器学习通过最大违反多目标攻击对财务报告进行攻击 cs.LG

KDD Workshop on Machine Learning in Finance

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05441v1) [paper-pdf](http://arxiv.org/pdf/2507.05441v1)

**Authors**: Edward Raff, Karen Kukla, Michel Benaroch, Joseph Comprix

**Abstract**: Bad actors, primarily distressed firms, have the incentive and desire to manipulate their financial reports to hide their distress and derive personal gains. As attackers, these firms are motivated by potentially millions of dollars and the availability of many publicly disclosed and used financial modeling frameworks. Existing attack methods do not work on this data due to anti-correlated objectives that must both be satisfied for the attacker to succeed. We introduce Maximum Violated Multi-Objective (MVMO) attacks that adapt the attacker's search direction to find $20\times$ more satisfying attacks compared to standard attacks. The result is that in $\approx50\%$ of cases, a company could inflate their earnings by 100-200%, while simultaneously reducing their fraud scores by 15%. By working with lawyers and professional accountants, we ensure our threat model is realistic to how such frauds are performed in practice.

摘要: 不良行为者，主要是陷入困境的公司，有动机和愿望操纵其财务报告，以掩盖其困境，并获得个人利益。作为攻击者，这些公司的动机是潜在的数百万美元和许多公开披露和使用的金融建模框架的可用性。现有的攻击方法不适用于这些数据，因为攻击者必须满足反相关的目标才能成功。我们介绍了最大违反多目标（MVMO）攻击，适应攻击者的搜索方向，找到$20\times$更令人满意的攻击相比，标准的攻击。结果是，在大约50%的情况下，公司可以将其收入夸大100- 200%，同时将其欺诈分数降低15%。通过与律师和专业会计师的合作，我们确保我们的威胁模型是现实的，以如何在实践中进行此类欺诈。



## **29. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

坏与好的传输攻击：解释和增强多模式大型语言模型之间的对抗性传输 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2405.20090v4) [paper-pdf](http://arxiv.org/pdf/2405.20090v4)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.

摘要: 多模式大型语言模型（MLLM）在跨模式交互中表现出出色的性能，但它们也存在对抗性漏洞。特别是，对抗性例子的可移植性仍然是一个持续的挑战。本文具体分析了MLLM之间对抗性转移性的表现，并确定了影响这一特征的关键因素。我们发现，MLLM的可移植性存在于具有相同视觉编码器的跨LLM场景中，并指出可能影响可移植性的\underline{\textit{两个关键因素}}。我们提供了两种语义级数据增强方法：添加图像补丁（AIP）和印刷增强可移植性方法（TATM），它们增强了对抗性示例跨MLLM的可移植性。为了探索对现实世界的潜在影响，我们利用了两项可能产生负面和积极社会影响的任务：\ding{182}有害内容插入和\ding{183}信息保护。



## **30. CLIP-Guided Backdoor Defense through Entropy-Based Poisoned Dataset Separation**

通过基于信息的中毒数据集分离实现CLIP引导的后门防御 cs.MM

15 pages, 9 figures, 15 tables. To appear in the Proceedings of the  32nd ACM International Conference on Multimedia (MM '25)

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05113v1) [paper-pdf](http://arxiv.org/pdf/2507.05113v1)

**Authors**: Binyan Xu, Fan Yang, Xilin Dai, Di Tang, Kehuan Zhang

**Abstract**: Deep Neural Networks (DNNs) are susceptible to backdoor attacks, where adversaries poison training data to implant backdoor into the victim model. Current backdoor defenses on poisoned data often suffer from high computational costs or low effectiveness against advanced attacks like clean-label and clean-image backdoors. To address them, we introduce CLIP-Guided backdoor Defense (CGD), an efficient and effective method that mitigates various backdoor attacks. CGD utilizes a publicly accessible CLIP model to identify inputs that are likely to be clean or poisoned. It then retrains the model with these inputs, using CLIP's logits as a guidance to effectively neutralize the backdoor. Experiments on 4 datasets and 11 attack types demonstrate that CGD reduces attack success rates (ASRs) to below 1% while maintaining clean accuracy (CA) with a maximum drop of only 0.3%, outperforming existing defenses. Additionally, we show that clean-data-based defenses can be adapted to poisoned data using CGD. Also, CGD exhibits strong robustness, maintaining low ASRs even when employing a weaker CLIP model or when CLIP itself is compromised by a backdoor. These findings underscore CGD's exceptional efficiency, effectiveness, and applicability for real-world backdoor defense scenarios. Code: https://github.com/binyxu/CGD.

摘要: 深度神经网络（DNN）很容易受到后门攻击，对手会毒害训练数据，以将后门植入受害者模型中。当前针对有毒数据的后门防御通常存在计算成本高或针对干净标签和干净图像后门等高级攻击的有效性低的问题。为了解决这些问题，我们引入了CLIP引导后门防御（CGD），这是一种有效且有效的方法，可以减轻各种后门攻击。CGD利用可公开访问的CLIP模型来识别可能是干净的或有毒的输入。然后，它使用这些输入重新训练模型，使用CLIP的日志作为指导，以有效地抵消后门。对4个数据集和11种攻击类型的实验表明，CGD将攻击成功率（ASB）降低至1%以下，同时保持干净准确率（CA），最大下降仅为0.3%，优于现有防御。此外，我们还表明，基于干净数据的防御可以使用CGD适应有毒数据。此外，CGD表现出很强的鲁棒性，即使采用较弱的CLIP模型或CLIP本身受到后门损害，也能保持较低的ASB。这些发现强调了CGD对现实世界后门防御场景的卓越效率、有效性和适用性。代码：https://github.com/binyxu/CGD。



## **31. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

BackFeed：一个高效且标准化的联邦学习后门攻击基准套件 cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.

摘要: 联邦学习（FL）系统很容易受到后门攻击，对手会根据有毒数据训练其本地模型并提交有毒模型更新以损害全局模型。尽管提出了许多攻击和防御，但不同的实验设置、实现错误和不切实际的假设阻碍了公平的比较和关于其在现实世界场景中有效性的有效性的有效结论。为了解决这个问题，我们引入了BackFed --一个全面的基准套件，旨在标准化、简化和可靠地评估FL中的后门攻击和防御，重点关注实际限制。我们的基准测试通过其多处理实施来提供关键优势，可以显着加速实验，并通过定义良好的API实现新方法的无缝集成。通过标准化的评估管道，我们将BackFeed设想为一个即插即用的环境，供研究人员全面可靠地评估新的攻击和防御。使用BackFeed，我们通过不同的模型架构和实验环境对计算机视觉和自然语言处理任务中的代表性后门攻击和防御进行了大规模研究。我们的实验批判性地评估了拟议攻击和防御的性能，揭示了实际条件下未知的限制和失败模式。这些经验见解为新方法的开发和增强FL系统的安全性提供了宝贵的指导。我们的框架可在https://github.com/thinh-dao/BackFed上公开获取。



## **32. Beyond Training-time Poisoning: Component-level and Post-training Backdoors in Deep Reinforcement Learning**

超越训练时中毒：深度强化学习中的学生级和训练后后门 cs.LG

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04883v1) [paper-pdf](http://arxiv.org/pdf/2507.04883v1)

**Authors**: Sanyam Vyas, Alberto Caron, Chris Hicks, Pete Burnap, Vasilios Mavroudis

**Abstract**: Deep Reinforcement Learning (DRL) systems are increasingly used in safety-critical applications, yet their security remains severely underexplored. This work investigates backdoor attacks, which implant hidden triggers that cause malicious actions only when specific inputs appear in the observation space. Existing DRL backdoor research focuses solely on training-time attacks requiring unrealistic access to the training pipeline. In contrast, we reveal critical vulnerabilities across the DRL supply chain where backdoors can be embedded with significantly reduced adversarial privileges. We introduce two novel attacks: (1) TrojanentRL, which exploits component-level flaws to implant a persistent backdoor that survives full model retraining; and (2) InfrectroRL, a post-training backdoor attack which requires no access to training, validation, nor test data. Empirical and analytical evaluations across six Atari environments show our attacks rival state-of-the-art training-time backdoor attacks while operating under much stricter adversarial constraints. We also demonstrate that InfrectroRL further evades two leading DRL backdoor defenses. These findings challenge the current research focus and highlight the urgent need for robust defenses.

摘要: 深度强化学习（DRL）系统越来越多地用于安全关键应用，但其安全性仍然严重不足。这项工作调查了后门攻击，这些攻击植入隐藏触发器，只有当特定输入出现在观察空间中时才会引发恶意操作。现有的DRL后门研究仅关注需要不切实际地访问培训管道的训练时攻击。相比之下，我们揭示了DRL供应链中的关键漏洞，其中后门可以嵌入，对抗特权显着减少。我们引入了两种新颖的攻击：（1）TrojanentRL，它利用组件级缺陷来植入持久的后门，该后门可以在全模型再培训中幸存下来;和（2）InfrectroRL，一种训练后后门攻击，不需要访问训练、验证或测试数据。针对六个Atari环境的经验和分析评估表明，我们的攻击可以与最先进的培训时后门攻击相媲美，同时在更严格的对抗约束下运行。我们还证明InfrectroRL进一步规避了两种主要的DRL后门防御。这些发现挑战了当前的研究重点，并凸显了对强大防御的迫切需要。



## **33. Phantom Subgroup Poisoning: Stealth Attacks on Federated Recommender Systems**

幻影亚群中毒：对联邦推荐系统的隐形攻击 cs.CR

13 pages

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06258v1) [paper-pdf](http://arxiv.org/pdf/2507.06258v1)

**Authors**: Bo Yan, Yurong Hao, Dingqi Liu, Huabin Sun, Pengpeng Qiao, Wei Yang Bryan Lim, Yang Cao, Chuan Shi

**Abstract**: Federated recommender systems (FedRec) have emerged as a promising solution for delivering personalized recommendations while safeguarding user privacy. However, recent studies have demonstrated their vulnerability to poisoning attacks. Existing attacks typically target the entire user group, which compromises stealth and increases the risk of detection. In contrast, real-world adversaries may prefer to prompt target items to specific user subgroups, such as recommending health supplements to elderly users. Motivated by this gap, we introduce Spattack, the first targeted poisoning attack designed to manipulate recommendations for specific user subgroups in the federated setting. Specifically, Spattack adopts a two-stage approximation-and-promotion strategy, which first simulates user embeddings of target/non-target subgroups and then prompts target items to the target subgroups. To enhance the approximation stage, we push the inter-group embeddings away based on contrastive learning and augment the target group's relevant item set based on clustering. To enhance the promotion stage, we further propose to adaptively tune the optimization weights between target and non-target subgroups. Besides, an embedding alignment strategy is proposed to align the embeddings between the target items and the relevant items. We conduct comprehensive experiments on three real-world datasets, comparing Spattack against seven state-of-the-art poisoning attacks and seven representative defense mechanisms. Experimental results demonstrate that Spattack consistently achieves strong manipulation performance on the specific user subgroup, while incurring minimal impact on non-target users, even when only 0.1\% of users are malicious. Moreover, Spattack maintains competitive overall recommendation performance and exhibits strong resilience against existing mainstream defenses.

摘要: 联合推荐系统（FedRec）已成为一种有前途的解决方案，可以在保护用户隐私的同时提供个性化推荐。然而，最近的研究表明它们很容易受到中毒攻击。现有的攻击通常针对整个用户群，这会损害隐蔽性并增加检测风险。相比之下，现实世界的对手可能更喜欢向特定用户子组提示目标物品，例如向老年用户推荐健康补充剂。出于这一差距的动机，我们引入了Spattack，这是第一个有针对性的中毒攻击，旨在操纵联邦环境中特定用户子组的建议。具体来说，Spattack采用了两阶段逼近和推广策略，首先模拟用户对目标/非目标子组的嵌入，然后将目标项目提示到目标子组。为了增强逼近阶段，我们基于对比学习推开组间嵌入，并基于集群增强目标组的相关项目集。为了增强推广阶段，我们进一步建议自适应地调整目标子组和非目标子组之间的优化权重。此外，还提出了嵌入对齐策略来对齐目标项与相关项之间的嵌入。我们对三个现实世界的数据集进行了全面的实验，将Spattack与七种最先进的中毒攻击和七种代表性的防御机制进行了比较。实验结果表明，即使只有0.1%的用户是恶意的，Spattack也能在特定用户子组上实现强大的操纵性能，同时对非目标用户的影响也很小。此外，Spattack保持了有竞争力的整体推荐性能，并对现有主流防御表现出强大的弹性。



## **34. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

基于扩散的对抗性身份操纵用于面部隐私保护 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2504.21646v2) [paper-pdf](http://arxiv.org/pdf/2504.21646v2)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.

摘要: 由于社交网络上潜在的未经授权的监视和用户跟踪，面部识别（FR）系统的成功引发了严重的隐私问题。现有的增强隐私的方法无法生成可以保护面部隐私的自然面部图像。在本文中，我们提出了基于扩散的对抗身份操纵（DiffAIM）来生成针对恶意FR系统的自然且高度可转移的对抗面孔。具体来说，我们在扩散模型的低维潜在空间内操纵面部身份。这涉及在反向扩散过程中迭代地注入基于梯度的对抗性身份指导，逐步引导一代人走向所需的对抗性面孔。该指南针对向目标的身份融合进行了优化，同时促进源自源头的语义分歧，促进有效模仿，同时保持视觉自然性。我们进一步结合了结构保留的正规化，以在操作过程中保持面部结构一致性。针对人脸验证和识别任务的大量实验表明，与最新技术相比，迪夫AIM实现了更强的黑匣子攻击可转移性，同时保持了卓越的视觉质量。我们还证明了所提出的方法对商业FR API（包括Face++和Aliyun）的有效性。



## **35. Robustifying 3D Perception through Least-Squares Multi-Agent Graphs Object Tracking**

通过最小平方多智能体图对象跟踪增强3D感知 cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04762v1) [paper-pdf](http://arxiv.org/pdf/2507.04762v1)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.

摘要: EdgeAI系统（例如自动驾驶汽车）的关键感知能力需要能够随着时间的推移准确识别和定位场景中的多个对象，从而减轻其影响，从而能够抵御对抗威胁。单代理跟踪提供了对抗攻击的弹性，但缺乏情景感知，这凸显了多代理合作以增强上下文理解和稳健性的必要性。本文提出了一种针对对抗性噪音的新型缓解框架，通过基于多智能体对抗性边界盒上的最小平方图跟踪对象。具体来说，我们使用最小平方图形工具来利用通过差坐标和锚点的完全连接图形上的重叠边界框来减少每个检测的重心的诱导位置误差。因此，多车辆检测被融合和细化，以减轻对抗影响，并在两个阶段与现有轨道相关联，执行跟踪，以进一步抑制对抗威胁。对现实世界V2 V4 Real数据集的广泛评估研究表明，在具有挑战性的对抗条件下，所提出的方法显着比最先进的单代理和多代理跟踪框架高出23.3%，作为一种弹性方法运行，无需依赖额外的防御机制。



## **36. Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World**

攻击者的噪音可以在现实世界中操纵您的音频LLM cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06256v1) [paper-pdf](http://arxiv.org/pdf/2507.06256v1)

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.

摘要: 本文研究了基于音频的大型语言模型（ALLM）（例如Qwen 2-Audio）的现实世界漏洞。我们首先证明对手可以精心设计隐秘的音频扰动来操纵ALLM表现出特定的有针对性的行为，例如引发对唤醒关键词的响应（例如，“嘿Qwen”），或触发有害行为（例如“更改我的日历事件”）。随后，我们表明，在用户与ALLM交互期间播放对抗性背景噪音会显着降低响应质量。至关重要的是，我们的研究说明了这些攻击对现实世界场景的可扩展性，当这些对抗性噪音通过空气播放时，会影响其他无辜用户。此外，我们还讨论了攻击的转移性以及潜在的防御措施。



## **37. Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message**

特洛伊木马破解：通过伪造辅助消息破解会话多模态模型 cs.AI

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04673v1) [paper-pdf](http://arxiv.org/pdf/2507.04673v1)

**Authors**: Wei Duan, Li Qian

**Abstract**: The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity.

摘要: 对话界面的兴起通过利用对话历史进行复杂推理，极大地增强了LLM的可用性。然而，这种依赖引入了一个未经探索的攻击面。本文介绍了一种新颖的越狱技术特洛伊木马卸载。对手通过在提供给其API的对话历史中伪造模型自己的过去话语来绕过安全机制。恶意有效负载被注入到模型属性消息中，然后是良性用户提示以触发有害内容生成。该漏洞源于不对称安全对齐：模型经过广泛训练以拒绝有害用户请求，但对自己所谓的对话历史缺乏类似的怀疑。这种对其“过去”的隐性信任造成了高影响的脆弱性。对Google Gemini-2.0-flash-preview-image-generation的实验验证表明，特洛伊木马移植的攻击成功率（ASB）比既定的用户越狱方法高得多。这些发现揭示了现代对话人工智能安全性的一个根本缺陷，需要从输入级过滤范式转变为对话上下文完整性的稳健协议级验证。



## **38. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

智能电网：网络攻击、关键防御方法和数字双胞胎 cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2205.11783v2) [paper-pdf](http://arxiv.org/pdf/2205.11783v2)

**Authors**: Tianming Zheng, Ping Yi, Yue Wu

**Abstract**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internet-connected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by both improving the existing defense approaches and introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid.   In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.

摘要: 智能电网作为国家关键基础设施，其网络安全问题引起了广泛关注。智能化、数字化和互联网连接的智能电网的发展吸引了外部对手的恶意活动。有必要通过改进现有的防御方法和将新开发的技术引入智能电网环境来增强其网络安全。作为一种新兴技术，数字孪生（DT）被认为是增强安全性的使能者。然而，实际实施相当具有挑战性。这是由于智能电网设计师、安全专家和DT开发人员之间存在知识障碍。每个单一领域都是一个复杂的系统，涵盖各种组件和技术。因此，需要整理相关内容，以便DT更好地嵌入智能电网的安全架构设计中。   为了满足这一需求，我们的论文涵盖了上述三个领域，即智能电网、网络安全和DT。具体来说，本文i）介绍了智能电网的背景; ii）从攻击事件和攻击方法回顾了外部网络攻击; iii）介绍了工业网络系统中的关键防御方法，包括设备识别、漏洞发现、入侵检测系统（IDS）、蜜罐、归因和威胁情报（TI）; iv）回顾DT的相关内容，包括其基本概念、在智能电网中的应用以及DT如何增强安全性。最后，论文提出了基于DT的智能电网未来发展的安全考虑。该调查预计将帮助开发人员打破智能电网、网络安全和DT之间的知识障碍，并为基于DT的智能电网的未来安全设计提供指导。



## **39. Backdooring Bias ($B^2$) into Stable Diffusion Models**

稳定扩散模型的后门偏差（$B^2$） cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2406.15213v4) [paper-pdf](http://arxiv.org/pdf/2406.15213v4)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasarian, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional diffusion models have revolutionized image generation by enabling users to create realistic, high-quality images from textual prompts, significantly enhancing artistic creation and visual communication. However, these advancements also introduce an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence public opinion and spread propaganda. In this paper, we study an attack vector that allows an adversary to inject arbitrary bias into a target model. The attack leverages low-cost backdooring techniques using a targeted set of natural textual triggers embedded within a small number of malicious data samples produced with public generative models. An adversary could pick common sequences of words that can then be inadvertently activated by benign users during inference. We investigate the feasibility and challenges of such attacks, demonstrating how modern generative models have made this adversarial process both easier and more adaptable. On the other hand, we explore various aspects of the detectability of such attacks and demonstrate that the model's utility remains intact in the absence of the triggers. Our extensive experiments using over 200,000 generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases maintain strong text-image alignment, highlighting the challenges in detecting biased images without knowing that bias in advance. Our cost analysis confirms the low financial barrier (\$10-\$15) to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in diffusion models.

摘要: 大型文本条件扩散模型的最新进展使用户能够根据文本提示创建真实、高质量的图像，从而彻底改变了图像生成，从而显着增强了艺术创作和视觉传达。然而，这些进步也带来了一个未充分探索的攻击机会：对手出于恶意意图将偏见引入生成的图像的可能性，例如，影响舆论并传播宣传。在本文中，我们研究了允许对手将任意偏差注入目标模型的攻击载体。该攻击利用低成本后门技术，使用嵌入公共生成模型生成的少量恶意数据样本中的一组有针对性的自然文本触发器。对手可能会选择常见的单词序列，然后在推理过程中被良性用户无意中激活。我们调查此类攻击的可行性和挑战，展示现代生成模型如何使这种对抗过程变得更容易、更适应。另一方面，我们探索了此类攻击可检测性的各个方面，并证明在没有触发器的情况下，该模型的实用性仍然完好无损。我们使用超过200，000张生成的图像并针对数百个微调模型进行了广泛的实验，证明了所提出的后门攻击的可行性。我们说明了这些偏见如何保持文本与图像的强对齐，强调了在事先不知道偏见的情况下检测偏见图像的挑战。我们的成本分析证实了执行此类攻击的财务障碍较低（10 - 15英镑），强调了针对扩散模型中此类漏洞的强大防御策略的必要性。



## **40. False Alarms, Real Damage: Adversarial Attacks Using LLM-based Models on Text-based Cyber Threat Intelligence Systems**

虚假警报，真实损害：在基于文本的网络威胁情报系统上使用基于LLM的模型进行对抗性攻击 cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06252v1) [paper-pdf](http://arxiv.org/pdf/2507.06252v1)

**Authors**: Samaneh Shafee, Alysson Bessani, Pedro M. Ferreira

**Abstract**: Cyber Threat Intelligence (CTI) has emerged as a vital complementary approach that operates in the early phases of the cyber threat lifecycle. CTI involves collecting, processing, and analyzing threat data to provide a more accurate and rapid understanding of cyber threats. Due to the large volume of data, automation through Machine Learning (ML) and Natural Language Processing (NLP) models is essential for effective CTI extraction. These automated systems leverage Open Source Intelligence (OSINT) from sources like social networks, forums, and blogs to identify Indicators of Compromise (IoCs). Although prior research has focused on adversarial attacks on specific ML models, this study expands the scope by investigating vulnerabilities within various components of the entire CTI pipeline and their susceptibility to adversarial attacks. These vulnerabilities arise because they ingest textual inputs from various open sources, including real and potentially fake content. We analyse three types of attacks against CTI pipelines, including evasion, flooding, and poisoning, and assess their impact on the system's information selection capabilities. Specifically, on fake text generation, the work demonstrates how adversarial text generation techniques can create fake cybersecurity and cybersecurity-like text that misleads classifiers, degrades performance, and disrupts system functionality. The focus is primarily on the evasion attack, as it precedes and enables flooding and poisoning attacks within the CTI pipeline.

摘要: 网络威胁情报（RTI）已成为一种重要的补充方法，在网络威胁生命周期的早期阶段运作。RTI涉及收集、处理和分析威胁数据，以更准确、更快速地了解网络威胁。由于数据量大，通过机器学习（ML）和自然语言处理（NLP）模型实现自动化对于有效的RTI提取至关重要。这些自动化系统利用来自社交网络、论坛和博客等来源的开源情报（Osint）来识别妥协指标（IoCs）。尽管之前的研究重点是对特定ML模型的对抗攻击，但这项研究通过调查整个RTI管道的各个组件内的漏洞及其对对抗攻击的易感性来扩大了范围。这些漏洞的出现是因为它们从各种开源获取文本输入，包括真实和潜在虚假内容。我们分析了针对RTI管道的三种类型的攻击，包括规避、洪水和中毒，并评估它们对系统信息选择能力的影响。具体来说，在虚假文本生成方面，该工作展示了对抗性文本生成技术如何创建虚假网络安全和类似网络安全的文本，从而误导分类器、降低性能并扰乱系统功能。重点主要是规避攻击，因为它先于RTI管道内的洪水和中毒攻击。



## **41. Addressing The Devastating Effects Of Single-Task Data Poisoning In Exemplar-Free Continual Learning**

解决无示例持续学习中单任务数据中毒的破坏性影响 cs.CR

Accepted at CoLLAs 2025

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.04106v1) [paper-pdf](http://arxiv.org/pdf/2507.04106v1)

**Authors**: Stanisław Pawlak, Bartłomiej Twardowski, Tomasz Trzciński, Joost van de Weijer

**Abstract**: Our research addresses the overlooked security concerns related to data poisoning in continual learning (CL). Data poisoning - the intentional manipulation of training data to affect the predictions of machine learning models - was recently shown to be a threat to CL training stability. While existing literature predominantly addresses scenario-dependent attacks, we propose to focus on a more simple and realistic single-task poison (STP) threats. In contrast to previously proposed poisoning settings, in STP adversaries lack knowledge and access to the model, as well as to both previous and future tasks. During an attack, they only have access to the current task within the data stream. Our study demonstrates that even within these stringent conditions, adversaries can compromise model performance using standard image corruptions. We show that STP attacks are able to strongly disrupt the whole continual training process: decreasing both the stability (its performance on past tasks) and plasticity (capacity to adapt to new tasks) of the algorithm. Finally, we propose a high-level defense framework for CL along with a poison task detection method based on task vectors. The code is available at https://github.com/stapaw/STP.git .

摘要: 我们的研究解决了与持续学习（CL）中数据中毒相关的被忽视的安全问题。数据中毒--故意操纵训练数据以影响机器学习模型的预测--最近被证明对CL训练稳定性构成威胁。虽然现有文献主要解决依赖于任务的攻击，但我们建议重点关注更简单、更现实的单任务毒药（STP）威胁。与之前提出的中毒环境相反，STP中的对手缺乏对模型以及之前和未来任务的了解和访问权限。在攻击期间，他们只能访问数据流中的当前任务。我们的研究表明，即使在这些严格的条件下，对手也可以使用标准图像损坏来损害模型性能。我们表明，STP攻击能够强烈破坏整个持续训练过程：降低算法的稳定性（其在过去任务上的性能）和可塑性（适应新任务的能力）。最后，我们提出了一个CL的高级防御框架以及一种基于任务载体的中毒任务检测方法。该代码可在https://github.com/stapaw/STP.git上获取。



## **42. A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models**

大型语言模型中针对错误信息的主动防御策略研究 cs.IR

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.05288v1) [paper-pdf](http://arxiv.org/pdf/2507.05288v1)

**Authors**: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu

**Abstract**: The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains.

摘要: 大型语言模型（LLM）在关键领域的广泛部署放大了算法生成的错误信息带来的社会风险。与传统的虚假内容不同，LLM生成的错误信息可以自我强化、高度可信，并且能够在多种语言中快速传播，而传统检测方法无法有效缓解这一点。本文引入了一种主动防御范式，从被动事后检测转向预期缓解策略。我们提出了一个三柱框架：（1）知识可信度，加强训练和部署数据的完整性;（2）推理可靠性，在推理过程中嵌入自我纠正机制;（3）输入鲁棒性，增强模型接口针对对抗性攻击的弹性。通过对现有技术的全面调查和比较荟萃分析，我们证明，尽管存在重要的计算费用和概括性挑战，但主动防御策略在错误信息预防方面比传统方法提供了高达63%的改进。我们认为，未来的研究应该重点关注共同设计强大的知识基础、推理认证和抗攻击界面，以确保LLM能够有效地对抗各个领域的错误信息。



## **43. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

多通道隐写术：一种可证明安全的混合隐写术模型，用于安全通信 cs.CR

22 pages, 15 figures, 4 algorithms. This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2501.04511v2) [paper-pdf](http://arxiv.org/pdf/2501.04511v2)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: Secure covert communication in hostile environments requires simultaneously achieving invisibility, provable security guarantees, and robustness against informed adversaries. This paper presents a novel hybrid steganographic framework that unites cover synthesis and cover modification within a unified multichannel protocol. A secret-seeded PRNG drives a lightweight Markov-chain generator to produce contextually plausible cover parameters, which are then masked with the payload and dispersed across independent channels. The masked bit-vector is imperceptibly embedded into conventional media via a variance-aware least-significant-bit algorithm, ensuring that statistical properties remain within natural bounds. We formalize a multichannel adversary model (MC-ATTACK) and prove that, under standard security assumptions, the adversary's distinguishing advantage is negligible, thereby guaranteeing both confidentiality and integrity. Empirical results corroborate these claims: local-variance-guided embedding yields near-lossless extraction (mean BER $<5\times10^{-3}$, correlation $>0.99$) with minimal perceptual distortion (PSNR $\approx100$,dB, SSIM $>0.99$), while key-based masking drives extraction success to zero (BER $\approx0.5$) for a fully informed adversary. Comparative analysis demonstrates that purely distortion-free or invertible schemes fail under the same threat model, underscoring the necessity of hybrid designs. The proposed approach advances high-assurance steganography by delivering an efficient, provably secure covert channel suitable for deployment in high-surveillance networks.

摘要: 敌对环境中的安全秘密通信需要同时实现不可见性、可证明的安全保证以及针对知情对手的稳健性。本文提出了一种新型的混合隐写框架，该框架将覆盖合成和覆盖修改统一到统一的多通道协议中。秘密种子PRNG驱动轻量级的马尔科夫链生成器，以产生上下文上合理的覆盖参数，然后用有效负载掩蔽这些参数并分散在独立的通道中。屏蔽位载体通过方差感知的最低有效位算法以难以察觉的方式嵌入到传统媒体中，确保统计属性保持在自然界限内。我们形式化了多通道对手模型（MC-ATTACK），并证明，在标准安全假设下，对手的区分优势可以忽略不计，从而保证机密性和完整性。经验结果证实了这些说法：局部方差引导嵌入产生了近乎无损的提取（平均BER $<5\times10 &{-3}$，相关性$>0.99$），感知失真最小（PSNR $\approx100 $，DB，SSIM $>0.99$），而对于完全知情的对手，基于密钥的掩蔽将提取成功率推至零（BER $\approx0.5 $）。比较分析表明，纯粹的无失真或可逆的方案在相同的威胁模型下会失败，这凸显了混合设计的必要性。所提出的方法通过提供适合在高监视网络中部署的高效、可证明安全的秘密通道来推进高保证隐写术。



## **44. When There Is No Decoder: Removing Watermarks from Stable Diffusion Models in a No-box Setting**

当没有解码器时：在无框环境中从稳定扩散模型中删除水印 cs.CR

arXiv admin note: text overlap with arXiv:2408.02035

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03646v1) [paper-pdf](http://arxiv.org/pdf/2507.03646v1)

**Authors**: Xiaodong Wu, Tianyi Tang, Xiangman Li, Jianbing Ni, Yong Yu

**Abstract**: Watermarking has emerged as a promising solution to counter harmful or deceptive AI-generated content by embedding hidden identifiers that trace content origins. However, the robustness of current watermarking techniques is still largely unexplored, raising critical questions about their effectiveness against adversarial attacks. To address this gap, we examine the robustness of model-specific watermarking, where watermark embedding is integrated with text-to-image generation in models like latent diffusion models. We introduce three attack strategies: edge prediction-based, box blurring, and fine-tuning-based attacks in a no-box setting, where an attacker does not require access to the ground-truth watermark decoder. Our findings reveal that while model-specific watermarking is resilient against basic evasion attempts, such as edge prediction, it is notably vulnerable to blurring and fine-tuning-based attacks. Our best-performing attack achieves a reduction in watermark detection accuracy to approximately 47.92\%. Additionally, we perform an ablation study on factors like message length, kernel size and decoder depth, identifying critical parameters influencing the fine-tuning attack's success. Finally, we assess several advanced watermarking defenses, finding that even the most robust methods, such as multi-label smoothing, result in watermark extraction accuracy that falls below an acceptable level when subjected to our no-box attacks.

摘要: 水印已成为一种有希望的解决方案，可以通过嵌入跟踪内容起源的隐藏标识符来对抗有害或欺骗性的人工智能生成的内容。然而，当前水印技术的鲁棒性在很大程度上仍未得到探索，这引发了有关其对抗攻击有效性的关键问题。为了解决这一差距，我们研究了特定于模型的水印的鲁棒性，其中水印嵌入与潜在扩散模型等模型中的文本到图像生成相集成。我们引入了三种攻击策略：基于边缘预测的攻击、框模糊攻击和无框设置中的基于微调的攻击，其中攻击者不需要访问地面真相水印解码器。我们的研究结果表明，虽然特定于模型的水印可以抵御边缘预测等基本规避尝试，但它特别容易受到模糊和基于微调的攻击。我们性能最好的攻击将水印检测准确率降低至约47.92%。此外，我们还对消息长度、内核大小和解码器深度等因素进行了消融研究，确定影响微调攻击成功的关键参数。最后，我们评估了几种先进的水印防御，发现即使是最稳健的方法（例如多标签平滑），在受到无框攻击时，水印提取准确性也会低于可接受的水平。



## **45. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的能力，但它们仍然容易受到对抗操纵的影响，例如通过提示注入攻击进行越狱。这些攻击绕过安全机制来生成受限制或有害内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全和越狱状态的潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM激活会进入半稳定状态，可以识别和扰动这些状态以引发状态转变。使用降维技术，我们预测安全和越狱反应的激活，以揭示低维空间中的潜在子空间。然后，我们推导出一个扰动载体，当将其应用于安全表示时，会将模型转向越狱状态。我们的结果表明，这种因果干预会在提示子集中导致具有统计学意义的越狱反应。接下来，我们探讨了这些扰动如何在模型的层中传播，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的研究结果表明，有针对性的扰动会导致激活和模型响应的明显变化。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转向先发制人的、模型不可知的技术，可以在表示层面中和对抗状态。



## **46. On the Limits of Robust Control Under Adversarial Disturbances**

论对抗扰动下鲁棒控制的极限 eess.SY

Extended version of a manuscript submitted to IEEE Transactions on  Automatic Control, July 2025

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03630v1) [paper-pdf](http://arxiv.org/pdf/2507.03630v1)

**Authors**: Paul Trodden, José M. Maestre, Hideaki Ishii

**Abstract**: This paper addresses a fundamental and important question in control: under what conditions does there fail to exist a robust control policy that keeps the state of a constrained linear system within a target set, despite bounded disturbances? This question has practical implications for actuator and sensor specification, feasibility analysis for reference tracking, and the design of adversarial attacks in cyber-physical systems. While prior research has predominantly focused on using optimization to compute control-invariant sets to ensure feasible operation, our work complements these approaches by characterizing explicit sufficient conditions under which robust control is fundamentally infeasible. Specifically, we derive novel closed-form, algebraic expressions that relate the size of a disturbance set -- modelled as a scaled version of a basic shape -- to the system's spectral properties and the geometry of the constraint sets.

摘要: 本文解决了控制中的一个基本而重要的问题：在什么条件下，不存在鲁棒控制策略，即使存在有界干扰，也不存在将受约束线性系统的状态保持在目标集中？这个问题对致动器和传感器规范、参考跟踪的可行性分析以及网络物理系统中对抗攻击的设计具有实际影响。虽然之前的研究主要集中在使用优化来计算控制不变集以确保可行的操作，但我们的工作通过描述鲁棒控制从根本上不可行的显式充分条件来补充这些方法。具体来说，我们推导出新颖的封闭形式的代数表达，将干扰集的大小（建模为基本形状的缩放版本）与系统的谱属性和约束集的几何形状联系起来。



## **47. Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right**

超越再殖民化：中等和低资源语言的NLP安全性本身 cs.CL

Pre-print

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03473v1) [paper-pdf](http://arxiv.org/pdf/2507.03473v1)

**Authors**: Heather Lent

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages.

摘要: 尽管越来越多的证据表明多语言可以很容易地被武器化来对抗语言模型（LM），但NLP Security的工作仍然绝大多数以英语为中心。在确保LM方面，“英语优先”的NLP规范与网络安全中的标准程序相冲突，而从业者需要预测并为最坏情况的结果做好准备。为了减轻NLP安全中最坏的情况，研究人员必须愿意接触LM安全中最薄弱的环节：资源较少的语言。因此，这项工作考察了低年级和中等资源语言的LM的安全性。我们将现有的对抗性攻击扩展到多达70种语言，以评估这些语言的单语和多语言LM的安全性。通过我们的分析，我们发现单语模型的参数总数通常太小，无法确保良好的安全性，而且虽然多语言很有帮助，但它也不总是保证安全性的提高。最终，这些发现强调了为资源较少的语言社区更安全地部署LM的重要考虑因素。



## **48. Evaluating the Evaluators: Trust in Adversarial Robustness Tests**

评估评估者：对对抗稳健性测试的信任 cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03450v1) [paper-pdf](http://arxiv.org/pdf/2507.03450v1)

**Authors**: Antonio Emanuele Cinà, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Despite significant progress in designing powerful adversarial evasion attacks for robustness verification, the evaluation of these methods often remains inconsistent and unreliable. Many assessments rely on mismatched models, unverified implementations, and uneven computational budgets, which can lead to biased results and a false sense of security. Consequently, robustness claims built on such flawed testing protocols may be misleading and give a false sense of security. As a concrete step toward improving evaluation reliability, we present AttackBench, a benchmark framework developed to assess the effectiveness of gradient-based attacks under standardized and reproducible conditions. AttackBench serves as an evaluation tool that ranks existing attack implementations based on a novel optimality metric, which enables researchers and practitioners to identify the most reliable and effective attack for use in subsequent robustness evaluations. The framework enforces consistent testing conditions and enables continuous updates, making it a reliable foundation for robustness verification.

摘要: 尽管在设计强大的对抗规避攻击以进行稳健性验证方面取得了重大进展，但这些方法的评估往往仍然不一致且不可靠。许多评估依赖于不匹配的模型、未经验证的实现和不均衡的计算预算，这可能会导致有偏见的结果和错误的安全感。因此，建立在此类有缺陷的测试协议上的稳健性声明可能会具有误导性，并给人一种错误的安全感。作为提高评估可靠性的具体步骤，我们提出了AttackBench，这是一个基准框架，旨在评估标准化和可重复条件下基于梯度的攻击的有效性。AttackBench作为一种评估工具，根据新颖的最优性指标对现有攻击实施进行排名，使研究人员和从业者能够识别最可靠、最有效的攻击，以用于后续的稳健性评估。该框架强制执行一致的测试条件并实现持续更新，使其成为稳健性验证的可靠基础。



## **49. Rectifying Adversarial Sample with Low Entropy Prior for Test-Time Defense**

用低熵先验纠正对抗样本以进行测试时防御 cs.CV

To appear in IEEEE Transactions on Multimedia

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03427v1) [paper-pdf](http://arxiv.org/pdf/2507.03427v1)

**Authors**: Lina Ma, Xiaowei Fu, Fuxiang Huang, Xinbo Gao, Lei Zhang

**Abstract**: Existing defense methods fail to defend against unknown attacks and thus raise generalization issue of adversarial robustness. To remedy this problem, we attempt to delve into some underlying common characteristics among various attacks for generality. In this work, we reveal the commonly overlooked low entropy prior (LE) implied in various adversarial samples, and shed light on the universal robustness against unseen attacks in inference phase. LE prior is elaborated as two properties across various attacks as shown in Fig. 1 and Fig. 2: 1) low entropy misclassification for adversarial samples and 2) lower entropy prediction for higher attack intensity. This phenomenon stands in stark contrast to the naturally distributed samples. The LE prior can instruct existing test-time defense methods, thus we propose a two-stage REAL approach: Rectify Adversarial sample based on LE prior for test-time adversarial rectification. Specifically, to align adversarial samples more closely with clean samples, we propose to first rectify adversarial samples misclassified with low entropy by reverse maximizing prediction entropy, thereby eliminating their adversarial nature. To ensure the rectified samples can be correctly classified with low entropy, we carry out secondary rectification by forward minimizing prediction entropy, thus creating a Max-Min entropy optimization scheme. Further, based on the second property, we propose an attack-aware weighting mechanism to adaptively adjust the strengths of Max-Min entropy objectives. Experiments on several datasets show that REAL can greatly improve the performance of existing sample rectification models.

摘要: 现有的防御方法未能防御未知攻击，从而提出了对抗鲁棒性的一般化问题。为了解决这个问题，我们试图深入研究各种攻击之间的一些潜在共同特征，以获取一般性。在这项工作中，我们揭示了各种对抗样本中隐含的普遍被忽视的低熵先验（LE），并揭示了在推理阶段针对不可见攻击的普遍鲁棒性。LE先验被描述为各种攻击的两个属性，如图1和图2所示：1）对抗性样本的低熵误分类; 2）针对更高的攻击强度的较低熵预测。这种现象与自然分布的样本形成鲜明对比。LE先验可以指导现有的测试时防御方法，因此我们提出了一种两阶段REAL方法：基于LE先验来纠正对抗样本，用于测试时对抗纠正。具体来说，为了更接近地将对抗性样本与干净样本对齐，我们建议首先通过反向最大化预测熵来纠正错误分类的对抗性样本，从而消除它们的对抗性本质。为了确保纠正后的样本能够以低信息正确分类，我们通过向前最小化预测信息进行二次纠正，从而创建了Max-Min信息优化方案。此外，基于第二个属性，我们提出了一个攻击感知加权机制，自适应调整的最大-最小熵目标的强度。在多个数据集上的实验表明，REAL可以大大提高现有样本校正模型的性能。



## **50. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

打破障碍：揭开Kubernetes运营商中的跨空间引用漏洞的神秘面纱 cs.CR

12 pages

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03387v1) [paper-pdf](http://arxiv.org/pdf/2507.03387v1)

**Authors**: Andong Chen, Zhaoxuan Jin, Ziyi Guo, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts. To the best of our knowledge, this paper is the first to systematically investigate the security vulnerability of Kubernetes Operators. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 7 confirmations and 6 CVEs by the time of submission, affecting vendors including ****** and ******, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we also open-source the static analysis suite to benefit the ecosystem.

摘要: Kubernetes Operators是一种自动化工具，旨在管理Kubernetes集群内的应用程序生命周期，扩展Kubernetes的功能，并减轻人类工程师的运营负担。虽然运营商显着简化了DevOps工作流程，但他们也引入了新的安全风险。特别是，Kubernetes强制执行命名空间隔离以分离工作负载并限制用户访问，确保用户只能与其授权命名空间内的资源交互。然而，Kubernetes操作员通常需要更高的特权，并且可能会与跨多个名称空间的资源交互。这引入了一类新的漏洞，即跨空间引用漏洞。根本原因在于声明的资源范围与Operator逻辑的实现范围不匹配，导致Kubernetes无法正确隔离命名空间。利用此类漏洞，对单个授权命名空间的访问权限有限的对手可能会利用运营商执行影响其他未经授权命名空间的操作，从而导致特权升级和进一步影响。据我们所知，本文是第一篇系统性研究Kubernetes Operators安全漏洞的论文。我们通过两种策略展示跨命名空间引用漏洞，展示攻击者如何绕过命名空间隔离。通过大规模测量，我们发现超过14%的野外经营者存在潜在的脆弱性。我们的调查结果已报告给相关开发人员，截至提交时已获得7项确认和6项CVE，影响了 * 和 * 等供应商，凸显了Kubernetes Operators对增强安全实践的迫切需求。为了缓解这种情况，我们还开源了静态分析套件，以造福生态系统。



