# Latest Adversarial Attack Papers
**update at 2025-04-07 09:23:55**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SoK: Attacks on Modern Card Payments**

SoK：对现代卡支付的攻击 cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03363v1) [paper-pdf](http://arxiv.org/pdf/2504.03363v1)

**Authors**: Xenia Hofmeier, David Basin, Ralf Sasse, Jorge Toro-Pozo

**Abstract**: EMV is the global standard for smart card payments and its NFC-based version, EMV contactless, is widely used, also for mobile payments. In this systematization of knowledge, we examine attacks on the EMV contactless protocol. We provide a comprehensive framework encompassing its desired security properties and adversary models. We also identify and categorize a comprehensive collection of protocol flaws and show how different subsets thereof can be combined into attacks. In addition to this systematization, we examine the underlying reasons for the many attacks against EMV and point to a better way forward.

摘要: EMV是智能卡支付的全球标准，其基于NFC的版本EMV非接触式版本被广泛使用，也用于移动支付。在知识的系统化中，我们研究了对EMV非接触式协议的攻击。我们提供了一个全面的框架，涵盖其所需的安全属性和对手模型。我们还识别和分类了一系列全面的协议缺陷，并展示了如何将其不同子集组合为攻击。除了这种系统化之外，我们还研究了针对EMV的许多攻击的根本原因，并指出更好的前进道路。



## **2. SLACK: Attacking LiDAR-based SLAM with Adversarial Point Injections**

SLACK：使用对抗点注入攻击基于LiDAR的SLAM cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03089v1) [paper-pdf](http://arxiv.org/pdf/2504.03089v1)

**Authors**: Prashant Kumar, Dheeraj Vattikonda, Kshitij Madhav Bhat, Kunal Dargan, Prem Kalra

**Abstract**: The widespread adoption of learning-based methods for the LiDAR makes autonomous vehicles vulnerable to adversarial attacks through adversarial \textit{point injections (PiJ)}. It poses serious security challenges for navigation and map generation. Despite its critical nature, no major work exists that studies learning-based attacks on LiDAR-based SLAM. Our work proposes SLACK, an end-to-end deep generative adversarial model to attack LiDAR scans with several point injections without deteriorating LiDAR quality. To facilitate SLACK, we design a novel yet simple autoencoder that augments contrastive learning with segmentation-based attention for precise reconstructions. SLACK demonstrates superior performance on the task of \textit{point injections (PiJ)} compared to the best baselines on KITTI and CARLA-64 dataset while maintaining accurate scan quality. We qualitatively and quantitatively demonstrate PiJ attacks using a fraction of LiDAR points. It severely degrades navigation and map quality without deteriorating the LiDAR scan quality.

摘要: LiDART广泛采用基于学习的方法，使自动驾驶汽车容易受到通过对抗\textit{point injection（PiJ）}的对抗攻击。它给导航和地图生成带来了严重的安全挑战。尽管其性质至关重要，但目前还没有研究对基于LiDART的SLAM进行基于学习的攻击的主要工作。我们的工作提出了SLACK，这是一种端到端的深度生成对抗模型，可以通过多次点注射攻击LiDART扫描，而不会降低LiDART质量。为了促进SLACK，我们设计了一种新颖而简单的自动编码器，它通过基于分段的注意力增强对比学习，以实现精确重建。与KITTI和CARLA-64数据集的最佳基线相比，SLACK在\textit{点注射（PiJ）}任务中表现出卓越的性能，同时保持准确的扫描质量。我们使用一小部分LiDART点定性和定量地演示了PiJ攻击。它会严重降低导航和地图质量，而不会降低LiDART扫描质量。



## **3. Integrating Identity-Based Identification against Adaptive Adversaries in Federated Learning**

在联邦学习中集成针对自适应对手的基于身份的识别 cs.CR

10 pages, 5 figures, research article, IEEE possible publication (in  submission)

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03077v1) [paper-pdf](http://arxiv.org/pdf/2504.03077v1)

**Authors**: Jakub Kacper Szelag, Ji-Jian Chin, Lauren Ansell, Sook-Chin Yip

**Abstract**: Federated Learning (FL) has recently emerged as a promising paradigm for privacy-preserving, distributed machine learning. However, FL systems face significant security threats, particularly from adaptive adversaries capable of modifying their attack strategies to evade detection. One such threat is the presence of Reconnecting Malicious Clients (RMCs), which exploit FLs open connectivity by reconnecting to the system with modified attack strategies. To address this vulnerability, we propose integration of Identity-Based Identification (IBI) as a security measure within FL environments. By leveraging IBI, we enable FL systems to authenticate clients based on cryptographic identity schemes, effectively preventing previously disconnected malicious clients from re-entering the system. Our approach is implemented using the TNC-IBI (Tan-Ng-Chin) scheme over elliptic curves to ensure computational efficiency, particularly in resource-constrained environments like Internet of Things (IoT). Experimental results demonstrate that integrating IBI with secure aggregation algorithms, such as Krum and Trimmed Mean, significantly improves FL robustness by mitigating the impact of RMCs. We further discuss the broader implications of IBI in FL security, highlighting research directions for adaptive adversary detection, reputation-based mechanisms, and the applicability of identity-based cryptographic frameworks in decentralized FL architectures. Our findings advocate for a holistic approach to FL security, emphasizing the necessity of proactive defence strategies against evolving adaptive adversarial threats.

摘要: 联邦学习（FL）最近已经成为隐私保护，分布式机器学习的一个有前途的范例。然而，FL系统面临着重大的安全威胁，特别是来自能够修改其攻击策略以逃避检测的自适应对手。其中一个威胁是重新连接恶意客户端（RMC）的存在，它通过修改攻击策略重新连接到系统来利用FL开放连接。为了解决这个漏洞，我们建议整合基于身份的识别（IBI）作为FL环境中的安全措施。通过利用IBI，我们使FL系统能够基于加密身份方案对客户端进行身份验证，有效防止之前断开连接的恶意客户端重新进入系统。我们的方法是使用椭圆曲线上的TNC-IBI（Tan-Ng-Chin）方案实施的，以确保计算效率，特别是在物联网（IoT）等资源受限的环境中。实验结果表明，将IBI与Krum和Trimmed Mean等安全聚合算法集成，通过减轻RMC的影响来显着提高FL鲁棒性。我们进一步讨论了IBI在FL安全中的更广泛影响，重点介绍了自适应对手检测、基于声誉的机制以及基于身份的加密框架在去中心化FL架构中的适用性的研究方向。我们的研究结果主张对FL安全采取整体方法，强调针对不断变化的适应性对抗威胁采取积极主动的防御策略的必要性。



## **4. Moving Target Defense Against Adversarial False Data Injection Attacks In Power Grids**

移动目标防御电网中对抗性虚假数据注入攻击 eess.SY

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.03065v1) [paper-pdf](http://arxiv.org/pdf/2504.03065v1)

**Authors**: Yexiang Chen, Subhash Lakshminarayana, H. Vincent Poor

**Abstract**: Machine learning (ML)-based detectors have been shown to be effective in detecting stealthy false data injection attacks (FDIAs) that can bypass conventional bad data detectors (BDDs) in power systems. However, ML models are also vulnerable to adversarial attacks. A sophisticated perturbation signal added to the original BDD-bypassing FDIA can conceal the attack from ML-based detectors. In this paper, we develop a moving target defense (MTD) strategy to defend against adversarial FDIAs in power grids. We first develop an MTD-strengthened deep neural network (DNN) model, which deploys a pool of DNN models rather than a single static model that cooperate to detect the adversarial attack jointly. The MTD model pool introduces randomness to the ML model's decision boundary, thereby making the adversarial attacks detectable. Furthermore, to increase the effectiveness of the MTD strategy and reduce the computational costs associated with developing the MTD model pool, we combine this approach with the physics-based MTD, which involves dynamically perturbing the transmission line reactance and retraining the DNN-based detector to adapt to the new system topology. Simulations conducted on IEEE test bus systems demonstrate that the MTD-strengthened DNN achieves up to 94.2% accuracy in detecting adversarial FDIAs. When combined with a physics-based MTD, the detection accuracy surpasses 99%, while significantly reducing the computational costs of updating the DNN models. This approach requires only moderate perturbations to transmission line reactances, resulting in minimal increases in OPF cost.

摘要: 基于机器学习(ML)的检测器已被证明在检测电力系统中可以绕过传统坏数据检测器(BDDS)的隐蔽虚假数据注入攻击(FDIA)方面是有效的。然而，ML模型也容易受到对手攻击。在原始的绕过BDD的FDIA中添加一个复杂的扰动信号，可以对基于ML的检测器隐藏攻击。在本文中，我们提出了一种移动目标防御(MTD)策略来防御电网中的敌意FDIA。首先提出了一种MTD增强的深度神经网络(DNN)模型，该模型部署了一组DNN模型，而不是单一的静态模型，它们共同协作检测敌方攻击。MTD模型池将随机性引入ML模型的决策边界，从而使对抗性攻击变得可检测。此外，为了提高MTD策略的有效性并降低与开发MTD模型池相关的计算成本，我们将该方法与基于物理的MTD相结合，该方法涉及动态扰动传输线电抗并重新训练基于DNN的检测器以适应新的系统拓扑。在IEEE测试节点系统上进行的仿真实验表明，MTD增强的DNN在检测敌意FDIA时的准确率高达94.2%。当与基于物理的MTD相结合时，检测准确率超过99%，同时显著降低了更新DNN模型的计算成本。这种方法只需要对传输线电抗进行适度的扰动，从而最大限度地降低了OPF成本。



## **5. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

对抗环境中的联邦学习：网络安全中的测试床设计和毒害韧性 cs.CR

6 pages, 4 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.09794v2) [paper-pdf](http://arxiv.org/pdf/2409.09794v2)

**Authors**: Hao Jian Huang, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using Raspberry Pi and Nvidia Jetson hardware by running the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, the testbed's capabilities are shown in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. The results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.

摘要: 本文介绍了联邦学习（FL）测试平台的设计和实现，重点关注其在网络安全中的应用以及评估其对中毒攻击的弹性。联合学习允许多个客户协作训练全球模型，同时保持数据去中心化，满足数据隐私和安全的关键需求，特别是在网络安全等敏感领域。我们的测试平台使用Raspberry Pi和Nvidia Jetson硬件通过运行Flower框架构建，促进了各种FL框架的实验，评估其性能、可扩展性和集成易用性。通过对联邦入侵检测系统的案例研究，展示了测试平台在检测异常和保护关键基础设施而不暴露敏感网络数据的能力。针对模型和数据完整性的全面中毒测试评估系统在对抗条件下的稳健性。结果表明，虽然联邦学习增强了数据隐私和分布式学习，但它仍然容易受到中毒攻击，必须减轻中毒攻击以确保其在现实世界应用程序中的可靠性。



## **6. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

ERPO：通过前推理偏好优化推进安全一致 cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严重的安全挑战。现有的对齐方法通常难以覆盖各种安全场景，并且仍然容易受到对抗性攻击。在这项工作中，我们提出了前-Ante推理偏好优化（ERPO），一种新的安全对齐框架，通过思想链为LLM提供明确的抢先推理，并通过嵌入预定义的安全规则为安全判断提供明确的证据。具体来说，我们的方法包括三个阶段：第一，通过使用构造的推理模块进行监督微调（SFT），为模型配备Ex-Ante推理;第二，通过直接偏好优化（DPO）提高安全性，有用性和效率;第三，通过长度控制的迭代偏好优化策略减轻推理延迟。在多个开源LLM上的实验表明，ERPO显着增强了安全性能，同时保持了响应效率。



## **7. No Free Lunch with Guardrails**

没有带护栏的免费午餐 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.

摘要: 随着大型语言模型（LLM）和生成式人工智能的广泛采用，护栏已成为确保其安全使用的关键工具。然而，添加护栏并非没有权衡;更强的安全措施可能会降低可用性，而更灵活的系统可能会为对抗性攻击留下缺口。在这项工作中，我们探索当前的护栏是否有效防止滥用，同时保持实用性。我们引入了一个框架来评估这些权衡，衡量不同的护栏如何平衡风险、安全性和可用性，并构建高效的护栏。   我们的调查结果证实，有护栏就没有免费的午餐;加强安全性往往是以牺牲可用性为代价的。为了解决这个问题，我们提出了一个设计更好护栏的蓝图，在保持可用性的同时最大限度地减少风险。我们评估各种行业护栏，包括Azure内容安全、Bedrock Guardrails、OpenAI的Moderation API、Guardrails AI、Nemo Guardrails和Enkrypt AI护栏。此外，我们还评估GPT-4 o、Gemini 2.0-Flash、Claude 3.5-十四行诗和Mistral Large-Latest等LLM如何在不同的系统提示下做出响应，包括简单提示、详细提示和具有思想链（CoT）推理的详细提示。我们的研究对不同护栏的性能进行了清晰的比较，强调了平衡安全性和可用性的挑战。



## **8. A Survey and Evaluation of Adversarial Attacks for Object Detection**

目标检测中的对抗性攻击综述与评价 cs.CV

17 pages

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2408.01934v3) [paper-pdf](http://arxiv.org/pdf/2408.01934v3)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.

摘要: 深度学习模型在计算机视觉任务中实现了惊人的准确性，但仍然容易受到对手例子的影响--精心设计的扰动输入图像，可能会欺骗这些模型做出自信但错误的预测。该漏洞在自动驾驶车辆、安全监控和安全关键检查系统等高风险应用中构成了重大风险。虽然现有的文献广泛地涵盖了图像分类中的对抗性攻击，但对此类攻击对目标检测系统的全面分析仍然有限。提出了一种新的分类框架，用于对特定于目标检测体系结构的敌意攻击进行分类，综合现有的健壮性度量标准，并对流行的目标检测模型(包括传统检测器和带有视觉语言预训练的现代检测器)的最新攻击方法进行了全面的经验评估。通过对开源攻击实施及其在不同检测体系结构中的有效性的严格分析，我们得出了对攻击特征的关键见解。此外，我们描绘了关键的研究差距和新出现的挑战，以指导未来在确保目标检测系统免受对手威胁方面的调查。我们的发现为开发更稳健的检测模型奠定了基础，同时强调了在这个快速发展的领域对标准化评估方案的迫切需求。



## **9. Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems**

协作推理系统模型倒置稳健性和条件熵最大化的理论见解 cs.LG

accepted by CVPR2025

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.00383v2) [paper-pdf](http://arxiv.org/pdf/2503.00383v2)

**Authors**: Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Ling-Yu Duan, Alex C. Kot, Xudong Jiang

**Abstract**: By locally encoding raw data into intermediate features, collaborative inference enables end users to leverage powerful deep learning models without exposure of sensitive raw data to cloud servers. However, recent studies have revealed that these intermediate features may not sufficiently preserve privacy, as information can be leaked and raw data can be reconstructed via model inversion attacks (MIAs). Obfuscation-based methods, such as noise corruption, adversarial representation learning, and information filters, enhance the inversion robustness by obfuscating the task-irrelevant redundancy empirically. However, methods for quantifying such redundancy remain elusive, and the explicit mathematical relation between this redundancy minimization and inversion robustness enhancement has not yet been established. To address that, this work first theoretically proves that the conditional entropy of inputs given intermediate features provides a guaranteed lower bound on the reconstruction mean square error (MSE) under any MIA. Then, we derive a differentiable and solvable measure for bounding this conditional entropy based on the Gaussian mixture estimation and propose a conditional entropy maximization (CEM) algorithm to enhance the inversion robustness. Experimental results on four datasets demonstrate the effectiveness and adaptability of our proposed CEM; without compromising feature utility and computing efficiency, plugging the proposed CEM into obfuscation-based defense mechanisms consistently boosts their inversion robustness, achieving average gains ranging from 12.9\% to 48.2\%. Code is available at \href{https://github.com/xiasong0501/CEM}{https://github.com/xiasong0501/CEM}.

摘要: 通过将原始数据本地编码为中间特征，协作推理使最终用户能够利用强大的深度学习模型，而无需将敏感的原始数据暴露给云服务器。然而，最近的研究表明，这些中间特征可能不足以保护隐私，因为信息可能会泄露，并且可以通过模型倒置攻击（MIA）重建原始数据。基于模糊的方法，例如噪音破坏、对抗性表示学习和信息过滤器，通过经验上模糊与任务无关的冗余来增强倒置的鲁棒性。然而，量化此类冗余的方法仍然难以捉摸，并且这种冗余最小化和逆鲁棒性增强之间的明确数学关系尚未建立。为了解决这一问题，这项工作首先从理论上证明，给定中间特征的输入的条件熵为任何MIA下的重建均方误差（SSE）提供了有保证的下限。然后，我们基于高斯混合估计推导出一个可微且可解的方法来限制该条件信息，并提出一种条件信息最大化（MBE）算法来增强逆的鲁棒性。四个数据集的实验结果证明了我们提出的MBE的有效性和适应性;在不损害特征效用和计算效率的情况下，将提出的MBE插入基于模糊的防御机制可持续增强其倒置鲁棒性，实现平均收益范围从12.9%到48.2%。代码可访问\href{https：//github.com/xiasong0501/MBE}{https：//github.com/xiasong0501/MBE}。



## **10. Robust Unsupervised Domain Adaptation for 3D Point Cloud Segmentation Under Source Adversarial Attacks**

源对抗攻击下3D点云分割的鲁棒无监督域自适应 cs.CV

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.01659v2) [paper-pdf](http://arxiv.org/pdf/2504.01659v2)

**Authors**: Haosheng Li, Junjie Chen, Yuecong Xu, Kemi Ding

**Abstract**: Unsupervised domain adaptation (UDA) frameworks have shown good generalization capabilities for 3D point cloud semantic segmentation models on clean data. However, existing works overlook adversarial robustness when the source domain itself is compromised. To comprehensively explore the robustness of the UDA frameworks, we first design a stealthy adversarial point cloud generation attack that can significantly contaminate datasets with only minor perturbations to the point cloud surface. Based on that, we propose a novel dataset, AdvSynLiDAR, comprising synthesized contaminated LiDAR point clouds. With the generated corrupted data, we further develop the Adversarial Adaptation Framework (AAF) as the countermeasure. Specifically, by extending the key point sensitive (KPS) loss towards the Robust Long-Tail loss (RLT loss) and utilizing a decoder branch, our approach enables the model to focus on long-tail classes during the pre-training phase and leverages high-confidence decoded point cloud information to restore point cloud structures during the adaptation phase. We evaluated our AAF method on the AdvSynLiDAR dataset, where the results demonstrate that our AAF method can mitigate performance degradation under source adversarial perturbations for UDA in the 3D point cloud segmentation application.

摘要: 无监督域自适应(UDA)框架对干净数据上的三维点云语义分割模型具有良好的泛化能力。然而，现有的工作忽略了当源域本身受到危害时的对抗健壮性。为了全面探索UDA框架的健壮性，我们首先设计了一种隐形的对抗性点云生成攻击，该攻击只需对点云表面进行微小的扰动就可以显著污染数据集。在此基础上，我们提出了一种新的数据集AdvSynLiDAR，它包含了合成的受污染的LiDAR点云。利用产生的破坏数据，我们进一步开发了对抗适应框架(AAF)作为对策。具体地说，通过将关键点敏感损失(KPS)扩展到稳健的长尾损失(RLT Loss)，并利用译码分支，我们的方法使模型在预训练阶段专注于长尾类，并在适应阶段利用高置信度解码的点云信息来恢复点云结构。我们在AdvSynLiDAR数据集上对我们的AAF方法进行了测试，结果表明，我们的AAF方法在3D点云分割应用中可以缓解UDA源对抗扰动下的性能下降。



## **11. Secure Generalization through Stochastic Bidirectional Parameter Updates Using Dual-Gradient Mechanism**

使用双梯度机制通过随机双向参数更新进行安全推广 cs.LG

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02213v1) [paper-pdf](http://arxiv.org/pdf/2504.02213v1)

**Authors**: Shourya Goel, Himanshi Tibrewal, Anant Jain, Anshul Pundhir, Pravendra Singh

**Abstract**: Federated learning (FL) has gained increasing attention due to privacy-preserving collaborative training on decentralized clients, mitigating the need to upload sensitive data to a central server directly. Nonetheless, recent research has underscored the risk of exposing private data to adversaries, even within FL frameworks. In general, existing methods sacrifice performance while ensuring resistance to privacy leakage in FL. We overcome these issues and generate diverse models at a global server through the proposed stochastic bidirectional parameter update mechanism. Using diverse models, we improved the generalization and feature representation in the FL setup, which also helped to improve the robustness of the model against privacy leakage without hurting the model's utility. We use global models from past FL rounds to follow systematic perturbation in parameter space at the server to ensure model generalization and resistance against privacy attacks. We generate diverse models (in close neighborhoods) for each client by using systematic perturbations in model parameters at a fine-grained level (i.e., altering each convolutional filter across the layers of the model) to improve the generalization and security perspective. We evaluated our proposed approach on four benchmark datasets to validate its superiority. We surpassed the state-of-the-art methods in terms of model utility and robustness towards privacy leakage. We have proven the effectiveness of our method by evaluating performance using several quantitative and qualitative results.

摘要: 由于对去中心化客户端进行保护隐私的协作培训，联合学习（FL）受到了越来越多的关注，减少了将敏感数据直接上传到中央服务器的需要。尽管如此，最近的研究强调了将私人数据暴露给对手的风险，即使在FL框架内也是如此。一般来说，现有方法会牺牲性能，同时确保对FL中隐私泄露的抵抗。我们克服了这些问题，并通过提出的随机双向参数更新机制在全球服务器上生成不同的模型。使用不同的模型，我们改进了FL设置中的概括和特征表示，这也有助于提高模型针对隐私泄露的鲁棒性，而不损害模型的实用性。我们使用过去FL回合的全局模型来跟踪服务器参数空间的系统性扰动，以确保模型的概括性和抵御隐私攻击。我们通过在细粒度级别上使用模型参数的系统性扰动（即，跨模型层改变每个卷积过滤器）以提高概括性和安全性。我们在四个基准数据集上评估了我们提出的方法，以验证其优势。在模型实用性和针对隐私泄露的鲁棒性方面，我们超越了最先进的方法。我们通过使用几个定量和定性结果评估性能来证明了我们方法的有效性。



## **12. FairDAG: Consensus Fairness over Concurrent Causal Design**

FairDAQ：并行因果设计之上的共识公平性 cs.DB

17 pages, 15 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02194v1) [paper-pdf](http://arxiv.org/pdf/2504.02194v1)

**Authors**: Dakai Kang, Junchao Chen, Tien Tuan Anh Dinh, Mohammad Sadoghi

**Abstract**: The rise of cryptocurrencies like Bitcoin and Ethereum has driven interest in blockchain technology, with Ethereum's smart contracts enabling the growth of decentralized finance (DeFi). However, research has shown that adversaries exploit transaction ordering to extract profits through attacks like front-running, sandwich attacks, and liquidation manipulation. This issue affects both permissionless and permissioned blockchains, as block proposers have full control over transaction ordering. To address this, a more fair approach to transaction ordering is essential.   Existing fairness protocols, such as Pompe and Themis, operate on leader-based consensus protocols, which not only suffer from low throughput but also allow adversaries to manipulate transaction ordering. To address these limitations, we propose FairDAG-AB and FairDAG-RL, which leverage DAG-based consensus protocols.   We theoretically demonstrate that FairDAG protocols not only uphold fairness guarantees, as previous fairness protocols do, but also achieve higher throughput and greater resilience to adversarial ordering manipulation. Our deployment and evaluation on CloudLab further validate these claims.

摘要: 比特币和以太坊等加密货币的兴起激发了人们对区块链技术的兴趣，以太坊的智能合约推动了去中心化金融（DeFi）的发展。然而，研究表明，对手利用交易排序通过抢先运行、三明治攻击和清算操纵等攻击来获取利润。这个问题会影响无许可区块链和有许可区块链，因为区块提议者可以完全控制交易排序。为了解决这个问题，更公平的交易排序方法至关重要。   现有的公平协议，例如Pompe和Themis，基于领导者的共识协议运行，该协议不仅吞吐量低，而且允许对手操纵交易排序。为了解决这些限制，我们提出了FairDAG-AB和FairDAG-RL，它们利用基于DAB的共识协议。   我们从理论上证明，FairDAQ协议不仅像以前的公平协议那样坚持公平保证，而且还实现了更高的吞吐量和更大的对抗性排序操纵的弹性。我们在CloudLab上的部署和评估进一步验证了这些说法。



## **13. Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs**

学会撒谎：强化学习攻击损害人类人工智能团队和LLM团队 cs.HC

17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI  Coevolution

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2503.21983v2) [paper-pdf](http://arxiv.org/pdf/2503.21983v2)

**Authors**: Abed Kareem Musaffar, Anand Gokhale, Sirui Zeng, Rasta Tadayon, Xifeng Yan, Ambuj Singh, Francesco Bullo

**Abstract**: As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.

摘要: 随着人工智能（AI）助手在安全关键领域越来越广泛地采用，开发针对潜在故障或对抗攻击的防护措施变得重要。开发这些保护措施的一个关键先决条件是了解这些人工智能助手误导人类队友的能力。我们在一个推理策略游戏的背景下调查了这个攻击问题，其中三名人类和一名人工智能助理组成的团队合作回答一系列琐碎问题。人类不知道的是，人工智能助手是敌对的。利用基于模型的强化学习（MBRL）的技术，人工智能助手学习人类信任演变的模型，并使用该模型来操纵群体决策过程以伤害团队。我们评估了两个模型--一个受文献启发，另一个受数据驱动--并发现两者都可以有效地伤害人类团队。此外，我们发现，在这种情况下，我们的数据驱动模型能够准确预测人类代理如何在先前互动的有限信息的情况下评估其队友。最后，我们将最先进的LLM模型与人类代理在影响力分配任务中的性能进行了比较，以评估LLM是否以类似于人类的方式分配影响力，或者它们是否对我们的攻击更稳健。这些结果增强了我们对小型人工智能团队决策动态的理解，并为防御策略奠定了基础。



## **14. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 以OpenAI的ChatGPT为例，大型语言模型（LLM）的广泛采用使防御这些模型上的对抗威胁的必要性变得更加突出。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性以及用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，利用LLM Transformer层之间的剩余激活分析。我们应用一种新颖的方法来分析剩余流中的独特激活模式，以进行攻击提示分类。我们整理了多个数据集，以展示这种分类方法如何在多种类型的攻击场景（包括我们新创建的攻击数据集）中具有高准确性。此外，我们通过集成LLM的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击能力的影响。结果强调了我们的方法在增强对抗性输入的检测和缓解、推进LLC运作的安全框架方面的有效性。



## **15. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片就是一切：用一张图片毒害视觉文档检索增强生成 cs.CL

8 pages, 6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02132v1) [paper-pdf](http://arxiv.org/pdf/2504.02132v1)

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings.

摘要: 多模式检索增强生成（M-RAG）最近出现了作为一种通过事实知识库（KB）抑制大型多模式模型（LSYS）幻觉的方法。然而，M-RAG还为对手引入了新的攻击载体，旨在通过将恶意条目注入知识库来破坏系统。在这项工作中，我们提出了针对M-RAG的中毒攻击，目标是视觉文档检索应用程序，其中KB包含文档页面的图像。我们的目标是制作一个针对各种不同用户查询检索的单个图像，并一致影响生成模型产生的输出，从而对M-RAG系统创建通用拒绝服务（Dock）攻击。我们证明，虽然我们的攻击对各种广泛使用的、最先进的检索器（嵌入模型）和生成器（LSYS）有效，但对稳健的嵌入模型也可能无效。我们的攻击不仅凸显了M-RAG管道对中毒攻击的脆弱性，而且还揭示了一个根本性弱点，即使在良性环境下，该弱点也可能阻碍其性能。



## **16. Graph Analytics for Cyber-Physical System Resilience Quantification**

用于网络物理系统弹性量化的图形分析 cs.CR

32 pages, 11 figures, 3 tables

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02120v1) [paper-pdf](http://arxiv.org/pdf/2504.02120v1)

**Authors**: Romain Dagnas, Michel Barbeau, Joaquin Garcia-Alfaro, Reda Yaich

**Abstract**: Critical infrastructures integrate a wide range of smart technologies and become highly connected to the cyber world. This is especially true for Cyber-Physical Systems (CPSs), which integrate hardware and software components. Despite the advantages of smart infrastructures, they remain vulnerable to cyberattacks. This work focuses on the cyber resilience of CPSs. We propose a methodology based on knowledge graph modeling and graph analytics to quantify the resilience potential of complex systems by using a multilayered model based on knowledge graphs. Our methodology also allows us to identify critical points. These critical points are components or functions of an architecture that can generate critical failures if attacked. Thus, identifying them can help enhance resilience and avoid cascading effects. We use the SWaT (Secure Water Treatment) testbed as a use case to achieve this objective. This system mimics the actual behavior of a water treatment station in Singapore. We model three resilient designs of SWaT according to our multilayered model. We conduct a resilience assessment based on three relevant metrics used in graph analytics. We compare the results obtained with each metric and discuss their accuracy in identifying critical points. We perform an experimentation analysis based on the knowledge gained by a cyber adversary about the system architecture. We show that the most resilient SWaT design has the necessary potential to bounce back and absorb the attacks. We discuss our results and conclude this work by providing further research axes.

摘要: 关键基础设施集成了广泛的智能技术，并与网络世界高度相连。对于集成硬件和软件组件的网络物理系统（CPS）来说尤其如此。尽管智能基础设施具有优势，但它们仍然容易受到网络攻击。这项工作的重点是CPS的网络弹性。我们提出了一种基于知识图建模和图分析的方法论，通过使用基于知识图的多层模型来量化复杂系统的弹性潜力。我们的方法还使我们能够识别关键点。这些关键点是架构的组件或功能，如果受到攻击，它们可能会产生严重故障。因此，识别它们可以帮助增强韧性并避免连锁效应。我们使用SWaT（安全水处理）测试台作为实现这一目标的用例。该系统模仿了新加坡水处理站的实际行为。我们根据我们的多层模型对SWaT的三种弹性设计进行建模。我们根据图形分析中使用的三个相关指标进行弹性评估。我们比较每个指标获得的结果，并讨论其识别临界点的准确性。我们根据网络对手获得的有关系统架构的知识进行实验分析。我们表明，最有弹性的SWaT设计具有反弹和吸收攻击的必要潜力。我们讨论了我们的结果，并通过提供进一步的研究轴来结束这项工作。



## **17. AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization**

AdPO：通过偏好优化增强大型视觉语言模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01735v1) [paper-pdf](http://arxiv.org/pdf/2504.01735v1)

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research.

摘要: GPT-4 o和LLaVA等大型视觉语言模型（LVLM）最近取得了显着的进步，并越来越多地部署在现实世界的应用程序中。然而，由于继承了视觉神经网络的敏感性，LVLM仍然容易受到对抗攻击，这可能会导致错误或恶意输出。虽然现有的工作利用对抗性微调来增强稳健性，但它们经常会在干净的输入上出现性能下降。本文提出了一种基于偏好优化的LVLM新型对抗防御策略AdPO。我们首次将对抗性训练重新定义为偏好优化问题，旨在增强模型在干净输入上生成正常输出的偏好，同时拒绝对抗性示例的潜在误导性输出。值得注意的是，AdPO仅通过修改图像编码器来实现这一点，例如，CLIP ViT，在各种降级任务中带来卓越的干净和对抗性能。考虑到训练涉及大型语言模型（LLM），计算成本显着增加。我们验证了在较小的LVLM上进行训练并随后转移到较大的模型可以实现有竞争力的性能，同时保持与基线方法相当的效率。我们全面的实验证实了拟议AdPO的有效性，为未来的对抗性防御研究提供了新的视角。



## **18. Sky of Unlearning (SoUL): Rewiring Federated Machine Unlearning via Selective Pruning**

取消学习天空（SoUL）：通过选择性修剪重新连接联邦机器取消学习 cs.LG

6 pages, 6 figures, IEEE International Conference on Communications  (ICC 2025)

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01705v1) [paper-pdf](http://arxiv.org/pdf/2504.01705v1)

**Authors**: Md Mahabub Uz Zaman, Xiang Sun, Jingjing Yao

**Abstract**: The Internet of Drones (IoD), where drones collaborate in data collection and analysis, has become essential for applications such as surveillance and environmental monitoring. Federated learning (FL) enables drones to train machine learning models in a decentralized manner while preserving data privacy. However, FL in IoD networks is susceptible to attacks like data poisoning and model inversion. Federated unlearning (FU) mitigates these risks by eliminating adversarial data contributions, preventing their influence on the model. This paper proposes sky of unlearning (SoUL), a federated unlearning framework that efficiently removes the influence of unlearned data while maintaining model performance. A selective pruning algorithm is designed to identify and remove neurons influential in unlearning but minimally impact the overall performance of the model. Simulations demonstrate that SoUL outperforms existing unlearning methods, achieves accuracy comparable to full retraining, and reduces computation and communication overhead, making it a scalable and efficient solution for resource-constrained IoD networks.

摘要: 无人机互联网（IoD）是无人机协作进行数据收集和分析的平台，对于监控和环境监测等应用来说已变得至关重要。联合学习（FL）使无人机能够以去中心化的方式训练机器学习模型，同时保护数据隐私。然而，IoD网络中的FL很容易受到数据中毒和模型倒置等攻击。联合取消学习（FU）通过消除对抗性数据贡献来减轻这些风险，防止它们对模型的影响。本文提出了一种联合学习框架，即天空学习（SoUL），它可以有效地消除未学习数据的影响，同时保持模型性能。设计了一种选择性剪枝算法，用于识别和删除对学习有影响的神经元，但对模型的整体性能影响最小。仿真结果表明，SoUL优于现有的非学习方法，实现了与完全再训练相当的准确性，并减少了计算和通信开销，使其成为资源受限的IoD网络的可扩展和有效的解决方案。



## **19. Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation**

重叠感知特征学习用于3D语义分割的鲁棒无监督领域自适应 cs.CV

8 pages,6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01668v1) [paper-pdf](http://arxiv.org/pdf/2504.01668v1)

**Authors**: Junjie Chen, Yuecong Xu, Haosheng Li, Kemi Ding

**Abstract**: 3D point cloud semantic segmentation (PCSS) is a cornerstone for environmental perception in robotic systems and autonomous driving, enabling precise scene understanding through point-wise classification. While unsupervised domain adaptation (UDA) mitigates label scarcity in PCSS, existing methods critically overlook the inherent vulnerability to real-world perturbations (e.g., snow, fog, rain) and adversarial distortions. This work first identifies two intrinsic limitations that undermine current PCSS-UDA robustness: (a) unsupervised features overlap from unaligned boundaries in shared-class regions and (b) feature structure erosion caused by domain-invariant learning that suppresses target-specific patterns. To address the proposed problems, we propose a tripartite framework consisting of: 1) a robustness evaluation model quantifying resilience against adversarial attack/corruption types through robustness metrics; 2) an invertible attention alignment module (IAAM) enabling bidirectional domain mapping while preserving discriminative structure via attention-guided overlap suppression; and 3) a contrastive memory bank with quality-aware contrastive learning that progressively refines pseudo-labels with feature quality for more discriminative representations. Extensive experiments on SynLiDAR-to-SemanticPOSS adaptation demonstrate a maximum mIoU improvement of 14.3\% under adversarial attack.

摘要: 3D点云语义分割（PCSS）是机器人系统和自动驾驶环境感知的基石，通过逐点分类实现精确的场景理解。虽然无监督域适应（UDA）缓解了PCSS中的标签稀缺性，但现有方法严重忽视了对现实世界扰动的固有脆弱性（例如，雪、雾、雨）和对抗性扭曲。这项工作首先确定了破坏当前PCSS-UDA鲁棒性的两个内在限制：（a）无监督特征与共享类区域中的未对齐边界重叠;（b）由抑制特定目标模式的域不变学习引起的特征结构侵蚀。为了解决提出的问题，我们提出了一个三方框架，包括：1）稳健性评估模型，通过稳健性指标量化对抗性攻击/腐败类型的弹性; 2）可逆注意力对齐模块（IAAM），实现双向域映射，同时通过注意力引导的重叠抑制保留区分结构;以及3）具有质量感知对比学习的对比存储器库，该质量感知对比学习逐渐地改进具有特征质量的伪标签以用于更具区别性的表示。对SynLiDAR到SemanticPOSS适配的广泛实验表明，在对抗攻击下，最大mIoU提高了14.3%。



## **20. Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions**

通过自然和对抗性局部腐蚀对DNN的空间鲁棒性进行基准测试 cs.CV

Under review

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01632v1) [paper-pdf](http://arxiv.org/pdf/2504.01632v1)

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: The robustness of DNNs is a crucial factor in safety-critical applications, particularly in complex and dynamic environments where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remained underexplored. This paper fills this gap by introducing specialized metrics for benchmarking the spatial robustness of segmentation models, alongside with an evaluation framework to assess the impact of localized corruptions. Furthermore, we uncover the inherent complexity of characterizing worst-case robustness using a single localized adversarial perturbation. To address this, we propose region-aware multi-attack adversarial analysis, a method that enables a deeper understanding of model robustness against adversarial perturbations applied to specific regions. The proposed metrics and analysis were evaluated on 15 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones and vice-versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.

摘要: DNN的鲁棒性是安全关键型应用程序中的一个关键因素，特别是在可能出现局部损坏的复杂动态环境中。虽然以前的研究已经评估了语义分割（SS）模型在整个图像自然或对抗性腐败下的鲁棒性，但对密集视觉模型在局部腐败下的空间鲁棒性的全面调查仍然没有得到充分的探索。本文填补了这一空白，引入了专门的度量基准分割模型的空间鲁棒性，以及评估框架，以评估本地化的腐败的影响。此外，我们揭示了固有的复杂性，使用一个单一的本地化对抗扰动的特征最坏情况下的鲁棒性。为了解决这个问题，我们提出了区域感知的多攻击对抗分析，这种方法可以更深入地理解模型针对应用于特定区域的对抗扰动的稳健性。在驾驶场景中对15个细分模型进行了评估，揭示了对自然和对抗形式的局部腐败影响的关键见解。结果表明，模型对这两种类型的威胁的反应不同;例如，基于变换器的分割模型对局部自然破坏表现出显着的鲁棒性，但极易受到对抗性破坏的影响，而基于CNN的模型则反之亦然。因此，我们还通过集成模型解决了平衡对自然和对抗局部破坏的鲁棒性的挑战，从而实现更广泛的威胁覆盖范围并提高密集视觉任务的可靠性。



## **21. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01550v1) [paper-pdf](http://arxiv.org/pdf/2504.01550v1)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已成为强大的工具，但其固有的安全风险--从有害内容生成到更广泛的社会危害--带来了重大挑战。最近的对抗攻击、微调漏洞以及在高风险环境中增加部署LLM可能会放大这些风险。现有的安全增强技术，例如利用人类反馈进行微调或对抗性训练，仍然很脆弱，因为它们可以解决特定的威胁，并且通常无法对不可见的攻击进行概括，或者需要手动系统级防御。本文介绍了RepBend，这是一种新颖的方法，它从根本上破坏了LLM中有害行为的潜在表现，提供了可扩展的解决方案来增强（潜在固有的）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **22. A Volumetric Approach to Privacy of Dynamical Systems**

动态系统隐私的体积方法 eess.SY

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2501.02893v3) [paper-pdf](http://arxiv.org/pdf/2501.02893v3)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: Information-theoretic metrics, such as mutual information, have been widely used to evaluate privacy leakage in dynamic systems. However, these approaches are typically limited to stochastic systems and face computational challenges. In this paper, we introduce a novel volumetric framework for analyzing privacy in systems affected by unknown but bounded noise. Our model considers a dynamic system comprising public and private states, where an observation set of the public state is released. An adversary utilizes the observed public state to infer an uncertainty set of the private state, referred to as the inference attack. We define the evolution dynamics of these inference attacks and quantify the privacy level of the private state using the volume of its uncertainty sets. We then develop an approximate computation method leveraging interval analysis to compute the private state set. We investigate the properties of the proposed volumetric privacy measure and demonstrate that it is bounded by the information gain derived from the observation set. Furthermore, we propose an optimization approach to designing privacy filter using randomization and linear programming based on the proposed privacy measure. The effectiveness of the optimal privacy filter design is evaluated through a production-inventory case study, illustrating its robustness against inference attacks and its superiority compared to a truncated Gaussian mechanism.

摘要: 信息论指标（例如互信息）已被广泛用于评估动态系统中的隐私泄露。然而，这些方法通常仅限于随机系统并面临计算挑战。本文中，我们引入了一种新型的体积框架，用于分析受未知但有界噪音影响的系统中的隐私。我们的模型考虑了一个由公共和私人国家组成的动态系统，其中释放了公共国家的观察集。对手利用观察到的公共状态来推断私有状态的不确定性集，称为推断攻击。我们定义了这些推理攻击的进化动态，并使用其不确定性集的量量化私有状态的隐私级别。然后，我们开发一种利用区间分析来计算私有状态集的近似计算方法。我们研究了所提出的体积隐私测量的属性，并证明它受到来自观察集的信息收益的限制。此外，我们提出了一种优化方法来设计隐私过滤器使用随机化和线性规划的基础上提出的隐私措施。最优隐私过滤器设计的有效性进行评估，通过生产库存的情况下研究，说明其对推理攻击的鲁棒性和其优越性相比，截断高斯机制。



## **23. An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer**

一个可优化的后缀胜过一千个模板：通过LLM作为优化器，在没有肯定短语的情况下高效黑匣子越狱 cs.AI

Be accepeted as NAACL2025 Findings

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2408.11313v2) [paper-pdf](http://arxiv.org/pdf/2408.11313v2)

**Authors**: Weipeng Jiang, Zhenting Wang, Juan Zhai, Shiqing Ma, Zhengyu Zhao, Chao Shen

**Abstract**: Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.

摘要: 尽管之前做出了安全调整，但主流LLM在遭受越狱攻击时仍然可能生成有害和不道德的内容。现有的越狱方法分为两大类：基于模板的方法和基于优化的方法。前者需要大量的手工工作和领域知识，而后者（以贪婪坐标梯度（GCG）为例）试图通过代币级优化最大化有害LLM输出的可能性，但也遇到了几个限制：需要白盒访问、需要预先构建的肯定短语以及效率低下。在本文中，我们提出了ECLIPSE，这是一种利用可优化后缀的新颖且高效的黑匣子越狱方法。我们从LLM强大的生成和优化能力中汲取灵感，利用任务提示将越狱目标翻译成自然语言指令。这指导LLM为恶意查询生成对抗性后缀。特别是，危害性评分器提供持续的反馈，使LLM自我反思和迭代优化能够自主有效地产生有效的后缀。实验结果表明，ECLIPSE在三种开源LLM和GPT-3.5-Turbo上的平均攻击成功率（ASB）为0.92，大幅超越GCG 2.4倍。此外，ECLIPSE与ASR中基于模板的方法相当，同时提供卓越的攻击效率，将平均攻击开销降低了83%。



## **24. Leveraging Generalizability of Image-to-Image Translation for Enhanced Adversarial Defense**

利用图像到图像翻译的泛化能力增强对抗性防御 cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01399v1) [paper-pdf](http://arxiv.org/pdf/2504.01399v1)

**Authors**: Haibo Zhang, Zhihua Yao, Kouichi Sakurai, Takeshi Saitoh

**Abstract**: In the rapidly evolving field of artificial intelligence, machine learning emerges as a key technology characterized by its vast potential and inherent risks. The stability and reliability of these models are important, as they are frequent targets of security threats. Adversarial attacks, first rigorously defined by Ian Goodfellow et al. in 2013, highlight a critical vulnerability: they can trick machine learning models into making incorrect predictions by applying nearly invisible perturbations to images. Although many studies have focused on constructing sophisticated defensive mechanisms to mitigate such attacks, they often overlook the substantial time and computational costs of training and maintaining these models. Ideally, a defense method should be able to generalize across various, even unseen, adversarial attacks with minimal overhead. Building on our previous work on image-to-image translation-based defenses, this study introduces an improved model that incorporates residual blocks to enhance generalizability. The proposed method requires training only a single model, effectively defends against diverse attack types, and is well-transferable between different target models. Experiments show that our model can restore the classification accuracy from near zero to an average of 72\% while maintaining competitive performance compared to state-of-the-art methods.

摘要: 在快速发展的人工智能领域，机器学习成为一项关键技术，其特点是其巨大的潜力和固有的风险。这些模型的稳定性和可靠性非常重要，因为它们经常成为安全威胁的目标。对抗性攻击由Ian Goodfellow等人于2013年首次严格定义，它凸显了一个关键漏洞：它们可以通过对图像应用几乎不可见的扰动来欺骗机器学习模型做出错误的预测。尽管许多研究的重点是构建复杂的防御机制来减轻此类攻击，但他们经常忽视训练和维护这些模型的大量时间和计算成本。理想情况下，防御方法应该能够以最小的费用概括各种甚至是不可见的对抗性攻击。本研究在我们之前关于基于图像到图像描述的防御的工作的基础上，引入了一种改进的模型，该模型结合了残余块以增强可概括性。所提出的方法仅需要训练单个模型，有效防御不同的攻击类型，并且可以在不同的目标模型之间很好地转移。实验表明，我们的模型可以将分类准确率从接近零恢复到平均72%，同时与最先进的方法相比保持竞争性能。



## **25. Adversarial Example Soups: Improving Transferability and Stealthiness for Free**

对抗性示例汤：免费提高可转移性和隐蔽性 cs.CV

Accepted by TIFS 2025

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2402.18370v3) [paper-pdf](http://arxiv.org/pdf/2402.18370v3)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yulong Yang, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: Transferable adversarial examples cause practical security risks since they can mislead a target model without knowing its internal knowledge. A conventional recipe for maximizing transferability is to keep only the optimal adversarial example from all those obtained in the optimization pipeline. In this paper, for the first time, we revisit this convention and demonstrate that those discarded, sub-optimal adversarial examples can be reused to boost transferability. Specifically, we propose ``Adversarial Example Soups'' (AES), with AES-tune for averaging discarded adversarial examples in hyperparameter tuning and AES-rand for stability testing. In addition, our AES is inspired by ``model soups'', which averages weights of multiple fine-tuned models for improved accuracy without increasing inference time. Extensive experiments validate the global effectiveness of our AES, boosting 10 state-of-the-art transfer attacks and their combinations by up to 13\% against 10 diverse (defensive) target models. We also show the possibility of generalizing AES to other types, \textit{e.g.}, directly averaging multiple in-the-wild adversarial examples that yield comparable success. A promising byproduct of AES is the improved stealthiness of adversarial examples since the perturbation variances are naturally reduced.

摘要: 可传输的对抗性示例会带来实际的安全风险，因为它们可能会在不了解目标模型内部知识的情况下误导目标模型。最大化可移植性的传统方法是仅保留优化管道中获得的所有示例中的最佳对抗示例。在本文中，我们首次重新审视这一惯例，并证明那些被丢弃的次优对抗性示例可以重新使用以提高可移植性。具体来说，我们提出了“对抗示例汤”（AES），其中AES-tune用于对超参数调整中丢弃的对抗示例进行平均，AES-rand用于稳定性测试。此外，我们的AES受到“模型汤”的启发，它对多个微调模型的权重进行平均，以在不增加推理时间的情况下提高准确性。大量实验验证了我们的AES的全球有效性，针对10种不同（防御性）目标模型，将10种最先进的传输攻击及其组合提高了高达13%。我们还展示了将AES推广到其他类型的可能性，\textit{e.g.}，直接对多个野外对抗示例进行平均，从而产生相当的成功。AES的一个有希望的副产品是增强了对抗示例的隐蔽性，因为扰动方差自然地减少了。



## **26. Breaking BERT: Gradient Attack on Twitter Sentiment Analysis for Targeted Misclassification**

突破BERT：针对有针对性的错误分类对Twitter情绪分析的梯度攻击 cs.CL

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01345v1) [paper-pdf](http://arxiv.org/pdf/2504.01345v1)

**Authors**: Akil Raj Subedi, Taniya Shah, Aswani Kumar Cherukuri, Thanos Vasilakos

**Abstract**: Social media platforms like Twitter have increasingly relied on Natural Language Processing NLP techniques to analyze and understand the sentiments expressed in the user generated content. One such state of the art NLP model is Bidirectional Encoder Representations from Transformers BERT which has been widely adapted in sentiment analysis. BERT is susceptible to adversarial attacks. This paper aims to scrutinize the inherent vulnerabilities of such models in Twitter sentiment analysis. It aims to formulate a framework for constructing targeted adversarial texts capable of deceiving these models, while maintaining stealth. In contrast to conventional methodologies, such as Importance Reweighting, this framework core idea resides in its reliance on gradients to prioritize the importance of individual words within the text. It uses a whitebox approach to attain fine grained sensitivity, pinpointing words that exert maximal influence on the classification outcome. This paper is organized into three interdependent phases. It starts with fine-tuning a pre-trained BERT model on Twitter data. It then analyzes gradients of the model to rank words on their importance, and iteratively replaces those with feasible candidates until an acceptable solution is found. Finally, it evaluates the effectiveness of the adversarial text against the custom trained sentiment classification model. This assessment would help in gauging the capacity of the adversarial text to successfully subvert classification without raising any alarm.

摘要: Twitter等社交媒体平台越来越依赖自然语言处理NLP技术来分析和理解用户生成的内容中表达的情感。其中一种最先进的NLP模型是来自Transformers BERT的双向编码器表示，该模型已广泛应用于情感分析。BERT容易受到对抗攻击。本文旨在仔细审查此类模型在Twitter情绪分析中的固有漏洞。它旨在制定一个框架，用于构建能够欺骗这些模型的有针对性的对抗性文本，同时保持隐形。与传统的方法（如重要性重新加权）相比，该框架的核心思想在于它依赖于梯度来优先考虑文本中单个单词的重要性。它使用白盒方法来获得细粒度的敏感性，精确定位对分类结果产生最大影响的单词。本文件分为三个相互依存的阶段。它首先在Twitter数据上微调预训练的BERT模型。然后，它分析模型的梯度，根据单词的重要性对单词进行排名，并迭代地用可行的候选项替换这些单词，直到找到可接受的解决方案。最后，它根据自定义训练的情感分类模型评估对抗文本的有效性。这项评估将有助于衡量对抗性文本成功颠覆分类而不引起任何警报的能力。



## **27. STEREO: A Two-Stage Framework for Adversarially Robust Concept Erasing from Text-to-Image Diffusion Models**

STEREO：从文本到图像扩散模型中消除对抗鲁棒概念的两阶段框架 cs.CV

Accepted to CVPR-2025. Code:  https://github.com/koushiksrivats/robust-concept-erasing

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2408.16807v2) [paper-pdf](http://arxiv.org/pdf/2408.16807v2)

**Authors**: Koushik Srivatsan, Fahad Shamshad, Muzammal Naseer, Vishal M. Patel, Karthik Nandakumar

**Abstract**: The rapid proliferation of large-scale text-to-image diffusion (T2ID) models has raised serious concerns about their potential misuse in generating harmful content. Although numerous methods have been proposed for erasing undesired concepts from T2ID models, they often provide a false sense of security; concept-erased models (CEMs) can still be manipulated via adversarial attacks to regenerate the erased concept. While a few robust concept erasure methods based on adversarial training have emerged recently, they compromise on utility (generation quality for benign concepts) to achieve robustness and/or remain vulnerable to advanced embedding space attacks. These limitations stem from the failure of robust CEMs to thoroughly search for "blind spots" in the embedding space. To bridge this gap, we propose STEREO, a novel two-stage framework that employs adversarial training as a first step rather than the only step for robust concept erasure. In the first stage, STEREO employs adversarial training as a vulnerability identification mechanism to search thoroughly enough. In the second robustly erase once stage, STEREO introduces an anchor-concept-based compositional objective to robustly erase the target concept in a single fine-tuning stage, while minimizing the degradation of model utility. We benchmark STEREO against seven state-of-the-art concept erasure methods, demonstrating its superior robustness to both white-box and black-box attacks, while largely preserving utility.

摘要: 大规模文本到图像扩散（T2 ID）模型的迅速普及引发了人们对它们在生成有害内容方面可能被滥用的严重担忧。尽管已经提出了许多方法来从T2 ID模型中擦除不需要的概念，但它们通常会提供错误的安全感;概念擦除模型（CEMS）仍然可以通过对抗攻击来操纵以重新生成被擦除的概念。虽然最近出现了一些基于对抗训练的稳健概念擦除方法，但它们在效用（良性概念的生成质量）上妥协，以实现稳健性和/或仍然容易受到高级嵌入空间攻击。这些局限性源于稳健的OEM未能彻底搜索嵌入空间中的“盲点”。为了弥合这一差距，我们提出了STEREO，这是一种新颖的两阶段框架，它将对抗训练作为鲁棒概念擦除的第一步，而不是唯一步骤。在第一阶段，STEREO采用对抗训练作为漏洞识别机制，以进行足够彻底的搜索。在第二个一次稳健擦除阶段中，STEREO引入了基于锚概念的合成目标，以便在单个微调阶段中稳健地擦除目标概念，同时最大限度地减少模型效用的退化。我们将STEREO与七种最先进的概念擦除方法进行了基准测试，证明了其对白盒和黑匣子攻击的卓越鲁棒性，同时在很大程度上保留了实用性。



## **28. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

保护视觉语言模型：缓解基于扰动的攻击中高斯噪音的脆弱性 cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01308v1) [paper-pdf](http://arxiv.org/pdf/2504.01308v1)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yichen Fu, Yichun Feng, Kin-man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

摘要: 视觉语言模型（VLMS）通过合并视觉信息扩展了大型语言模型（LLM）的功能，但它们仍然容易受到越狱攻击，尤其是在处理嘈杂或损坏的图像时。尽管现有的VLM在培训期间采取安全措施来减轻此类攻击，但与噪音增强视觉输入相关的漏洞被忽视了。在这项工作中，我们发现错过噪音增强训练会导致严重的安全漏洞：许多VLM甚至容易受到高斯噪音等简单扰动的影响。为了应对这一挑战，我们提出了Robust-VLGuard，这是一个具有对齐/未对齐图像-文本对的多模式安全数据集，结合了噪音增强微调，可以降低攻击成功率，同时保留VLM的功能。对于更强的基于优化的视觉扰动攻击，我们提出了DiffPure-VLM，利用扩散模型将对抗性扰动转换为类高斯噪声，可以通过具有噪声增强安全微调的VLM进行防御。实验结果表明，扩散模型的分布偏移特性与我们微调的VLM很好地吻合，显著减轻了不同强度的对抗性扰动。数据集和代码可在https://github.com/JarvisUSTC/DiffPure-RobustVLM上获取。



## **29. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

展望未来：通过揭露对抗性合同来防止DeFi攻击 cs.CR

23 pages, 7 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2401.07261v5) [paper-pdf](http://arxiv.org/pdf/2401.07261v5)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: Decentralized Finance (DeFi) incidents stemming from the exploitation of smart contract vulnerabilities have culminated in financial damages exceeding 3 billion US dollars. Existing defense mechanisms typically focus on detecting and reacting to malicious transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively. Based on the fact that most attack logic rely on deploying one or more intermediate smart contracts as supporting components to the exploitation of victim contracts, detection methods have been proposed that focus on identifying these adversarial contracts instead of adversarial transactions. However, previous state-of-the-art approaches in this direction have failed to produce results satisfactory enough for real-world deployment. In this paper, we propose a new framework for effectively detecting DeFi attacks via unveiling adversarial contracts. Our approach allows us to leverage common attack patterns, code semantics and intrinsic characteristics found in malicious smart contracts to build the LookAhead system based on Machine Learning (ML) classifiers and a transformer model that is able to effectively distinguish adversarial contracts from benign ones, and make timely predictions of different types of potential attacks. Experiments show that LookAhead achieves an F1-score as high as 0.8966, which represents an improvement of over 44.4% compared to the previous state-of-the-art solution Forta, with a False Positive Rate (FPR) at only 0.16%.

摘要: 因利用智能合约漏洞而引发的去中心化金融（DeFi）事件已导致超过30亿美元的财务损失。现有的防御机制通常专注于检测和反应攻击者执行的针对受害者合同的恶意交易。然而，随着私有交易池的出现，交易直接发送给矿工，而无需首先出现在公共内存池中，当前的检测工具在有效识别攻击活动方面面临着重大挑战。基于大多数攻击逻辑依赖于部署一个或多个中间智能合同作为利用受害者合同的支持组件这一事实，人们提出了专注于识别这些对抗性合同而不是对抗性交易的检测方法。然而，之前在此方向上的最新方法未能产生足够令人满意的结果，以满足现实世界的部署。在本文中，我们提出了一个新的框架，用于通过公布对抗性合同来有效检测DeFi攻击。我们的方法使我们能够利用恶意智能合同中发现的常见攻击模式、代码语义和内在特征来构建基于机器学习（ML）分类器和Transformer模型的LookAhead系统，该模型能够有效地区分对抗性合同与良性合同，并对不同类型的潜在攻击做出及时预测。实验表明，LookAhead的F1评分高达0.8966，与之前最先进的解决方案Forta相比提高了44.4%以上，假阳性率（FPR）仅为0.16%。



## **30. Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning**

全球战略，本地适应：具有双重学习的多轮红色团队代理 cs.AI

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01278v1) [paper-pdf](http://arxiv.org/pdf/2504.01278v1)

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios.

摘要: 随着大型语言模型（LLM）变得更加强大和广泛，出于恶意目的利用这些模型会带来巨大的安全风险。虽然大多数现有的红色团队框架专注于单回合攻击，但现实世界的对手通常在多回合场景中操作，迭代地探测漏洞并根据威胁模型响应调整其提示。在本文中，我们提出了\AlgName，这是一种新型的多回合红色团队代理，它通过补充的学习维度来模拟复杂的人类攻击者：随着时间的推移积累知识并推广到新的攻击目标的全球战术学习，以及在初始尝试失败时细化特定目标的实现的局部预算学习。与之前依赖固定策略集的多回合方法不同，\AlgName使代理能够识别新的越狱策略、开发基于目标的策略选择框架，并完善所选策略的提示公式。JailbreakBench上的经验评估证明了我们框架的卓越性能，在5次对话中针对GPT-3.5-Turbo和Llama-3.1- 70 B实现了超过90%的攻击成功率，超过了最先进的基线。这些结果凸显了动态学习在现实多转弯场景中识别和利用模型漏洞方面的有效性。



## **31. Towards Resilient Federated Learning in CyberEdge Networks: Recent Advances and Future Trends**

在CyberEdge网络中实现弹性联邦学习：最近的进展和未来的趋势 cs.CR

15 pages, 8 figures, 4 tables, 122 references, journal paper

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01240v1) [paper-pdf](http://arxiv.org/pdf/2504.01240v1)

**Authors**: Kai Li, Zhengyang Zhang, Azadeh Pourkabirian, Wei Ni, Falko Dressler, Ozgur B. Akan

**Abstract**: In this survey, we investigate the most recent techniques of resilient federated learning (ResFL) in CyberEdge networks, focusing on joint training with agglomerative deduction and feature-oriented security mechanisms. We explore adaptive hierarchical learning strategies to tackle non-IID data challenges, improving scalability and reducing communication overhead. Fault tolerance techniques and agglomerative deduction mechanisms are studied to detect unreliable devices, refine model updates, and enhance convergence stability. Unlike existing FL security research, we comprehensively analyze feature-oriented threats, such as poisoning, inference, and reconstruction attacks that exploit model features. Moreover, we examine resilient aggregation techniques, anomaly detection, and cryptographic defenses, including differential privacy and secure multi-party computation, to strengthen FL security. In addition, we discuss the integration of 6G, large language models (LLMs), and interoperable learning frameworks to enhance privacy-preserving and decentralized cross-domain training. These advancements offer ultra-low latency, artificial intelligence (AI)-driven network management, and improved resilience against adversarial attacks, fostering the deployment of secure ResFL in CyberEdge networks.

摘要: 在这项调查中，我们研究了CyberEdge网络中弹性联邦学习（ResFL）的最新技术，重点关注与凝聚演绎和面向特征的安全机制的联合训练。我们探索自适应分层学习策略来应对非IID数据挑战，提高可扩展性并减少通信负担。研究了故障容忍技术和凝聚推理机制，以检测不可靠设备、细化模型更新并增强收敛稳定性。与现有的FL安全研究不同，我们全面分析面向特征的威胁，例如利用模型特征的中毒、推理和重建攻击。此外，我们还研究了弹性聚合技术、异常检测和加密防御，包括差异隐私和安全多方计算，以加强FL安全性。此外，我们还讨论了6G、大型语言模型（LLM）和互操作学习框架的集成，以增强隐私保护和去中心化的跨领域培训。这些进步提供了超低延迟、人工智能（AI）驱动的网络管理，并提高了针对对抗攻击的弹性，促进了在CyberEdge网络中部署安全ResFL。



## **32. TenAd: A Tensor-based Low-rank Black Box Adversarial Attack for Video Classification**

TenAd：一种基于张量的视频分类低等级黑匣子对抗攻击 cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01228v1) [paper-pdf](http://arxiv.org/pdf/2504.01228v1)

**Authors**: Kimia haghjooei, Mansoor Rezghi

**Abstract**: Deep learning models have achieved remarkable success in computer vision but remain vulnerable to adversarial attacks, particularly in black-box settings where model details are unknown. Existing adversarial attack methods(even those works with key frames) often treat video data as simple vectors, ignoring their inherent multi-dimensional structure, and require a large number of queries, making them inefficient and detectable. In this paper, we propose \textbf{TenAd}, a novel tensor-based low-rank adversarial attack that leverages the multi-dimensional properties of video data by representing videos as fourth-order tensors. By exploiting low-rank attack, our method significantly reduces the search space and the number of queries needed to generate adversarial examples in black-box settings. Experimental results on standard video classification datasets demonstrate that \textbf{TenAd} effectively generates imperceptible adversarial perturbations while achieving higher attack success rates and query efficiency compared to state-of-the-art methods. Our approach outperforms existing black-box adversarial attacks in terms of success rate, query efficiency, and perturbation imperceptibility, highlighting the potential of tensor-based methods for adversarial attacks on video models.

摘要: 深度学习模型在计算机视觉方面取得了显着的成功，但仍然容易受到对抗攻击，特别是在模型细节未知的黑匣子环境中。现有的对抗性攻击方法（即使是那些适用于关键帧的方法）经常将视频数据视为简单的载体，忽略了其固有的多维结构，并且需要大量查询，从而使其效率低下且可检测。在本文中，我们提出了\textBF{TenAd}，这是一种新型的基于张量的低等级对抗攻击，通过将视频表示为四阶张量来利用视频数据的多维属性。通过利用低排名攻击，我们的方法显着减少了在黑匣子环境中生成对抗性示例所需的搜索空间和查询数量。标准视频分类数据集的实验结果表明，与最先进的方法相比，\textBF{TenAd}有效地生成难以感知的对抗性扰动，同时实现更高的攻击成功率和查询效率。我们的方法在成功率、查询效率和扰动不可感知性方面优于现有的黑匣子对抗攻击，凸显了基于张量的方法对视频模型进行对抗攻击的潜力。



## **33. A Survey on Adversarial Contention Resolution**

对抗性竞争解决方法研究 cs.DC

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2403.03876v3) [paper-pdf](http://arxiv.org/pdf/2403.03876v3)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We also highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 争用解决方案解决了协调多个进程对共享资源(如内存、磁盘存储或通信通道)的访问的挑战。争用解决最初是由数据库系统和总线网络中的挑战推动的，尽管经历了几十年的技术变革，但它作为资源共享的一个重要抽象概念一直存在。在这里，我们回顾了关于解决最坏情况争用的文献，在这种情况下，进程的数量和每个进程可能开始寻求访问资源的时间由对手决定。我们还重点介绍了争用解决方案的演变，其中新的关注点--如安全性、服务质量和能源效率--是由现代系统驱动的。这些努力使人们深入了解了随机化和确定性方法的局限性，以及不同模型假设的影响，如全球时钟同步、处理器数量的知识、访问尝试的反馈以及对共享资源可用性的攻击。



## **34. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

不，我当然可以！使用无害微调数据可以利用拒绝机制 cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2502.19537v3) [paper-pdf](http://arxiv.org/pdf/2502.19537v3)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Krishnamurthy Dvijotham

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.

摘要: OpenAI和Google等领先的语言模型（LM）提供商提供微调API，允许客户根据特定用例调整LM。为了防止滥用，这些LM提供商实施过滤机制来阻止有害的微调数据。因此，寻求通过这些API产生不安全LM的对手必须制作不会造成可识别有害的对抗训练数据。我们在此背景下做出了三点贡献：1。我们表明，许多使用无害数据创建不安全LM的现有攻击依赖于消除响应的前几个令牌中的模型拒绝。2.我们表明，此类先前的攻击可以通过一个简单的防御来阻止，该防御预先填充对齐模型中的前几个令牌，然后让微调模型填充其余的令牌。3.我们描述了一种新的数据中毒攻击“不，当然我可以执行”（NOICE），它利用LM的公式化拒绝机制来引发有害响应。通过训练LM在满足这些请求之前基于安全性拒绝良性请求，我们能够越狱几个开源模型和一个开源模型（GPT-4 o）。我们显示针对GPT-4 o的攻击成功率（ASB）为57%;我们的攻击从OpenAI获得了Bug赏金。针对受简单防御保护的开源模型，与之前仅使用无害数据的性能最佳的攻击相比，我们将ASC平均提高了3.25倍。NOICE展示了重复拒绝机制的可利用性，并扩大了对封闭源模型面临的无害数据威胁的理解。



## **35. Multilingual and Multi-Accent Jailbreaking of Audio LLMs**

多语言和多口音音频LL越狱 cs.SD

21 pages, 6 figures, 15 tables

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01094v1) [paper-pdf](http://arxiv.org/pdf/2504.01094v1)

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.

摘要: 大型音频语言模型（LALM）具有显着提高的音频理解能力，但会带来严重的安全风险，特别是通过音频越狱。虽然之前的工作重点是以英语为中心的攻击，但我们暴露了一个更严重的漏洞：对抗性的多语言和多口音音频越狱，其中语言和声学差异极大地放大了攻击的成功。在本文中，我们引入了Multi-AudioJail，这是第一个利用这些漏洞的系统框架，通过（1）对抗干扰的多语言/多口音音频越狱提示的新颖数据集，以及（2）分层评估管道揭示了声学干扰（例如，回响、回声和耳语效果）与跨语言语音相互作用，导致越狱成功率（JSR）激增高达+57.25个百分点（例如，对MEaLiON产生了肯尼亚口音的攻击）。至关重要的是，我们的工作进一步揭示了多模式LLM本质上比单模式系统更容易受到攻击：攻击者只需要利用最弱的环节（例如，非英语音频输入）来损害整个模型，我们通过多语言纯音频攻击的成功率比纯文本攻击高出3.1倍。我们计划发布我们的数据集，以刺激对跨模式防御的研究，敦促社区随着LALM的发展，以多模式解决这一不断扩大的攻击面。



## **36. A Survey on Unlearnable Data**

关于不可学习数据的调查 cs.LG

31 pages, 3 figures, Code in  https://github.com/LiJiahao-Alex/Awesome-UnLearnable-Data

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2503.23536v2) [paper-pdf](http://arxiv.org/pdf/2503.23536v2)

**Authors**: Jiahao Li, Yiqiang Chen, Yunbing Xing, Yang Gu, Xiangyuan Lan

**Abstract**: Unlearnable data (ULD) has emerged as an innovative defense technique to prevent machine learning models from learning meaningful patterns from specific data, thus protecting data privacy and security. By introducing perturbations to the training data, ULD degrades model performance, making it difficult for unauthorized models to extract useful representations. Despite the growing significance of ULD, existing surveys predominantly focus on related fields, such as adversarial attacks and machine unlearning, with little attention given to ULD as an independent area of study. This survey fills that gap by offering a comprehensive review of ULD, examining unlearnable data generation methods, public benchmarks, evaluation metrics, theoretical foundations and practical applications. We compare and contrast different ULD approaches, analyzing their strengths, limitations, and trade-offs related to unlearnability, imperceptibility, efficiency and robustness. Moreover, we discuss key challenges, such as balancing perturbation imperceptibility with model degradation and the computational complexity of ULD generation. Finally, we highlight promising future research directions to advance the effectiveness and applicability of ULD, underscoring its potential to become a crucial tool in the evolving landscape of data protection in machine learning.

摘要: 不可学习数据（ULD）已成为一种创新的防御技术，可防止机器学习模型从特定数据中学习有意义的模式，从而保护数据隐私和安全。通过对训练数据引入扰动，ULD会降低模型性能，使未经授权的模型难以提取有用的表示。尽管ULD的重要性越来越大，但现有的调查主要关注相关领域，例如对抗性攻击和机器取消学习，很少关注ULD作为一个独立的研究领域。这项调查通过对ULD进行全面审查、检查难以学习的数据生成方法、公共基准、评估指标、理论基础和实际应用来填补这一空白。我们比较和对比不同的ULD方法，分析它们的优点、局限性以及与不可学习性、不可感知性、效率和稳健性相关的权衡。此外，我们还讨论了关键挑战，例如平衡扰动不可感知性与模型退化以及ULD生成的计算复杂性。最后，我们强调了未来有前途的研究方向，以提高ULD的有效性和适用性，强调其成为机器学习数据保护不断变化的环境中重要工具的潜力。



## **37. S3C2 Summit 2024-08: Government Secure Supply Chain Summit**

S3 C2峰会2024-08：政府安全供应链峰会 cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00924v1) [paper-pdf](http://arxiv.org/pdf/2504.00924v1)

**Authors**: Courtney Miller, William Enck, Yasemin Acar, Michel Cukier, Alexandros Kapravelos, Christian Kastner, Dominik Wermke, Laurie Williams

**Abstract**: Supply chain security has become a very important vector to consider when defending against adversary attacks. Due to this, more and more developers are keen on improving their supply chains to make them more robust against future threats. On August 29, 2024 researchers from the Secure Software Supply Chain Center (S3C2) gathered 14 practitioners from 10 government agencies to discuss the state of supply chain security. The goal of the summit is to share insights between companies and developers alike to foster new collaborations and ideas moving forward. Through this meeting, participants were questions on best practices and thoughts how to improve things for the future. In this paper we summarize the responses and discussions of the summit.

摘要: 供应链安全已成为防御对手攻击时需要考虑的一个非常重要的载体。因此，越来越多的开发商热衷于改善其供应链，使其更强大地应对未来的威胁。2024年8月29日，安全软件供应链中心（S3 C2）的研究人员召集了来自10个政府机构的14名从业者，讨论供应链安全状况。峰会的目标是在公司和开发人员之间分享见解，以促进新的合作和前进的想法。通过这次会议，与会者就最佳实践和如何为未来改进的想法提出了问题。本文总结了峰会的回应和讨论。



## **38. Alleviating Performance Disparity in Adversarial Spatiotemporal Graph Learning Under Zero-Inflated Distribution**

零膨胀分布下对抗性时空图学习的性能差异 cs.LG

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00721v1) [paper-pdf](http://arxiv.org/pdf/2504.00721v1)

**Authors**: Songran Bai, Yuheng Ji, Yue Liu, Xingwei Zhang, Xiaolong Zheng, Daniel Dajun Zeng

**Abstract**: Spatiotemporal Graph Learning (SGL) under Zero-Inflated Distribution (ZID) is crucial for urban risk management tasks, including crime prediction and traffic accident profiling. However, SGL models are vulnerable to adversarial attacks, compromising their practical utility. While adversarial training (AT) has been widely used to bolster model robustness, our study finds that traditional AT exacerbates performance disparities between majority and minority classes under ZID, potentially leading to irreparable losses due to underreporting critical risk events. In this paper, we first demonstrate the smaller top-k gradients and lower separability of minority class are key factors contributing to this disparity. To address these issues, we propose MinGRE, a framework for Minority Class Gradients and Representations Enhancement. MinGRE employs a multi-dimensional attention mechanism to reweight spatiotemporal gradients, minimizing the gradient distribution discrepancies across classes. Additionally, we introduce an uncertainty-guided contrastive loss to improve the inter-class separability and intra-class compactness of minority representations with higher uncertainty. Extensive experiments demonstrate that the MinGRE framework not only significantly reduces the performance disparity across classes but also achieves enhanced robustness compared to existing baselines. These findings underscore the potential of our method in fostering the development of more equitable and robust models.

摘要: 零膨胀分布（ZID）下的时空图学习（SGL）对于城市风险管理任务（包括犯罪预测和交通事故分析）至关重要。然而，SGL模型很容易受到对抗攻击，从而损害了其实际实用性。虽然对抗训练（AT）已被广泛用于增强模型稳健性，但我们的研究发现，传统的AT加剧了ZID下多数族裔和少数族裔班级之间的表现差异，可能会因低估关键风险事件而导致不可挽回的损失。在本文中，我们首先证明了较小的top-k梯度和较低的少数群体可分离性是导致这种差异的关键因素。为了解决这些问题，我们提出了MinGPT，这是一个针对少数族裔群体和代表增强的框架。MinGER采用多维注意力机制来重新加权时空梯度，最大限度地减少类别之间的梯度分布差异。此外，我们引入了一种不确定性引导的对比损失，以提高具有较高不确定性的少数族裔表示的类间可分离性和类内紧凑性。大量实验表明，与现有基线相比，MinGPT框架不仅显着降低了类别之间的性能差异，而且还实现了增强的稳健性。这些发现强调了我们的方法在促进开发更公平和更稳健的模型方面的潜力。



## **39. Impact of Data Duplication on Deep Neural Network-Based Image Classifiers: Robust vs. Standard Models**

数据重复对基于深度神经网络的图像分类器的影响：稳健模型与标准模型 cs.LG

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00638v1) [paper-pdf](http://arxiv.org/pdf/2504.00638v1)

**Authors**: Alireza Aghabagherloo, Aydin Abadi, Sumanta Sarkar, Vishnu Asutosh Dasu, Bart Preneel

**Abstract**: The accuracy and robustness of machine learning models against adversarial attacks are significantly influenced by factors such as training data quality, model architecture, the training process, and the deployment environment. In recent years, duplicated data in training sets, especially in language models, has attracted considerable attention. It has been shown that deduplication enhances both training performance and model accuracy in language models. While the importance of data quality in training image classifier Deep Neural Networks (DNNs) is widely recognized, the impact of duplicated images in the training set on model generalization and performance has received little attention.   In this paper, we address this gap and provide a comprehensive study on the effect of duplicates in image classification. Our analysis indicates that the presence of duplicated images in the training set not only negatively affects the efficiency of model training but also may result in lower accuracy of the image classifier. This negative impact of duplication on accuracy is particularly evident when duplicated data is non-uniform across classes or when duplication, whether uniform or non-uniform, occurs in the training set of an adversarially trained model. Even when duplicated samples are selected in a uniform way, increasing the amount of duplication does not lead to a significant improvement in accuracy.

摘要: 机器学习模型针对对抗性攻击的准确性和稳健性受到训练数据质量、模型架构、训练过程和部署环境等因素的显着影响。近年来，训练集中的重复数据，尤其是语言模型中的重复数据，引起了相当大的关注。事实证明，去重可以增强语言模型中的训练性能和模型准确性。虽然训练图像分类器深度神经网络（DNN）中数据质量的重要性已被广泛认识到，但训练集中重复图像对模型概括性和性能的影响却很少受到关注。   本文中，我们解决了这一差距，并对图像分类中重复的影响进行了全面的研究。我们的分析表明，训练集中重复图像的存在不仅会对模型训练的效率产生负面影响，还会导致图像分类器的准确性较低。当重复的数据在类别中不一致时，或者当重复（无论是一致还是不一致）发生在对抗训练模型的训练集中时，重复对准确性的负面影响尤其明显。即使以统一的方式选择重复样本，增加重复数量也不会导致准确性的显着提高。



## **40. Robust Recommender System: A Survey and Future Directions**

稳健的推荐系统：调查和未来方向 cs.IR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2309.02057v2) [paper-pdf](http://arxiv.org/pdf/2309.02057v2)

**Authors**: Kaike Zhang, Qi Cao, Fei Sun, Yunfan Wu, Shuchang Tao, Huawei Shen, Xueqi Cheng

**Abstract**: With the rapid growth of information, recommender systems have become integral for providing personalized suggestions and overcoming information overload. However, their practical deployment often encounters ``dirty'' data, where noise or malicious information can lead to abnormal recommendations. Research on improving recommender systems' robustness against such dirty data has thus gained significant attention. This survey provides a comprehensive review of recent work on recommender systems' robustness. We first present a taxonomy to organize current techniques for withstanding malicious attacks and natural noise. We then explore state-of-the-art methods in each category, including fraudster detection, adversarial training, certifiable robust training for defending against malicious attacks, and regularization, purification, self-supervised learning for defending against malicious attacks. Additionally, we summarize evaluation metrics and commonly used datasets for assessing robustness. We discuss robustness across varying recommendation scenarios and its interplay with other properties like accuracy, interpretability, privacy, and fairness. Finally, we delve into open issues and future research directions in this emerging field. Our goal is to provide readers with a comprehensive understanding of robust recommender systems and to identify key pathways for future research and development. To facilitate ongoing exploration, we maintain a continuously updated GitHub repository with related research: https://github.com/Kaike-Zhang/Robust-Recommender-System.

摘要: 随着信息的快速增长，推荐系统已成为提供个性化建议和克服信息过载的不可或缺的组成部分。然而，他们的实际部署经常会遇到“脏”数据，其中噪音或恶意信息可能会导致异常推荐。因此，关于提高推荐系统针对此类肮脏数据的鲁棒性的研究受到了广泛关注。这项调查对最近关于推荐系统稳健性的工作进行了全面回顾。我们首先提出了一个分类法来组织当前抵御恶意攻击和自然噪音的技术。然后，我们探索每个类别的最先进方法，包括欺诈者检测、对抗训练、用于防御恶意攻击的可认证稳健训练，以及用于防御恶意攻击的正规化、净化、自我监督学习。此外，我们还总结了用于评估稳健性的评估指标和常用数据集。我们讨论了不同推荐场景的稳健性及其与准确性、可解释性、隐私性和公平性等其他属性的相互作用。最后，我们深入探讨了这个新兴领域的开放问题和未来的研究方向。我们的目标是让读者全面了解强大的推荐系统，并确定未来研究和开发的关键途径。为了促进持续的探索，我们维护了一个持续更新的GitHub存储库以及相关研究：https://github.com/Kaike-Zhang/Robust-Recommender-System。



## **41. The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances**

魔术师的提示：用语言细微差别揭露大型语言模型的事实弱点 cs.CL

work in progress

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.02865v1) [paper-pdf](http://arxiv.org/pdf/2504.02865v1)

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms.

摘要: 随着大型语言模型（LLM）的不断发展，非专家用户越来越依赖它们作为实时信息来源。为了确保它们提供的信息的真实性，许多研究都集中在减轻LLM响应中的幻觉上，但仅限于正式用户查询的背景下，而不是恶意制作的查询。在这项研究中，我们引入了幻觉者的提示，这是一种新颖的幻觉攻击，将语言细微差别融入到对抗性询问中，针对五种事实增强策略，挑战LLM的事实准确性。我们的攻击会自动生成高度可转移的幻觉提示，以引发内部事实错误，同时保留用户意图和语义。大量实验证实了我们的攻击在攻击黑匣子LLM（包括GPT-4 o和Gemini-2.0等商业API）方面的有效性，即使有各种防御机制。



## **42. Unleashing the Power of Pre-trained Encoders for Universal Adversarial Attack Detection**

释放预培训编码器的力量进行通用对抗攻击检测 cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00429v1) [paper-pdf](http://arxiv.org/pdf/2504.00429v1)

**Authors**: Yinghe Zhang, Chi Liu, Shuai Zhou, Sheng Shen, Peng Gui

**Abstract**: Adversarial attacks pose a critical security threat to real-world AI systems by injecting human-imperceptible perturbations into benign samples to induce misclassification in deep learning models. While existing detection methods, such as Bayesian uncertainty estimation and activation pattern analysis, have achieved progress through feature engineering, their reliance on handcrafted feature design and prior knowledge of attack patterns limits generalization capabilities and incurs high engineering costs. To address these limitations, this paper proposes a lightweight adversarial detection framework based on the large-scale pre-trained vision-language model CLIP. Departing from conventional adversarial feature characterization paradigms, we innovatively adopt an anomaly detection perspective. By jointly fine-tuning CLIP's dual visual-text encoders with trainable adapter networks and learnable prompts, we construct a compact representation space tailored for natural images. Notably, our detection architecture achieves substantial improvements in generalization capability across both known and unknown attack patterns compared to traditional methods, while significantly reducing training overhead. This study provides a novel technical pathway for establishing a parameter-efficient and attack-agnostic defense paradigm, markedly enhancing the robustness of vision systems against evolving adversarial threats.

摘要: 对抗性攻击通过将人类难以感知的扰动注入良性样本中，从而在深度学习模型中引发错误分类，对现实世界的人工智能系统构成了严重的安全威胁。虽然现有的检测方法，例如Bayesian不确定性估计和激活模式分析，已经通过特征工程取得了进展，但它们对手工特征设计和攻击模式的先验知识的依赖限制了概括能力并产生了高昂的工程成本。为了解决这些局限性，本文提出了一种基于大规模预训练视觉语言模型CLIP的轻量级对抗检测框架。与传统的对抗性特征描述范式不同，我们创新性地采用异常检测视角。通过将CLIP的双视觉文本编码器与可训练的适配器网络和可学习的提示联合微调，我们构建了一个专为自然图像量身定制的紧凑表示空间。值得注意的是，与传统方法相比，我们的检测架构在已知和未知攻击模式的概括能力方面实现了大幅提高，同时显着减少了训练负担。这项研究为建立参数高效且攻击不可知的防御范式提供了一种新颖的技术途径，显着增强视觉系统针对不断变化的对抗威胁的稳健性。



## **43. CopyQNN: Quantum Neural Network Extraction Attack under Varying Quantum Noise**

CopyQNN：变化量子噪音下的量子神经网络提取攻击 quant-ph

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00366v1) [paper-pdf](http://arxiv.org/pdf/2504.00366v1)

**Authors**: Zhenxiao Fu, Leyi Zhao, Xuhong Zhang, Yilun Xu, Gang Huang, Fan Chen

**Abstract**: Quantum Neural Networks (QNNs) have shown significant value across domains, with well-trained QNNs representing critical intellectual property often deployed via cloud-based QNN-as-a-Service (QNNaaS) platforms. Recent work has examined QNN model extraction attacks using classical and emerging quantum strategies. These attacks involve adversaries querying QNNaaS platforms to obtain labeled data for training local substitute QNNs that replicate the functionality of cloud-based models. However, existing approaches have largely overlooked the impact of varying quantum noise inherent in noisy intermediate-scale quantum (NISQ) computers, limiting their effectiveness in real-world settings. To address this limitation, we propose the CopyQNN framework, which employs a three-step data cleaning method to eliminate noisy data based on its noise sensitivity. This is followed by the integration of contrastive and transfer learning within the quantum domain, enabling efficient training of substitute QNNs using a limited but cleaned set of queried data. Experimental results on NISQ computers demonstrate that a practical implementation of CopyQNN significantly outperforms state-of-the-art QNN extraction attacks, achieving an average performance improvement of 8.73% across all tasks while reducing the number of required queries by 90x, with only a modest increase in hardware overhead.

摘要: 量子神经网络（QNN）在各个领域都表现出了巨大的价值，训练有素的QNN代表关键知识产权，通常通过基于云的QNN即服务（QNNaaz）平台部署。最近的工作检查了使用经典和新兴量子策略的QNN模型提取攻击。这些攻击涉及对手查询QNNaas平台以获取标记数据，用于训练复制基于云的模型功能的本地替代QNN。然而，现有的方法在很大程度上忽视了有噪音的中等规模量子（NISQ）计算机中固有的不同量子噪音的影响，限制了它们在现实世界环境中的有效性。为了解决这一局限性，我们提出了CopyQNN框架，该框架采用三步数据清理方法根据其噪音敏感性来消除有噪数据。随后，在量子域中集成了对比学习和迁移学习，从而能够使用有限但干净的查询数据集高效训练替代QNN。NISQ计算机上的实验结果表明，CopyQNN的实际实现显着优于最先进的QNN提取攻击，在所有任务中实现了8.73%的平均性能改进，同时将所需的查询数量减少了90倍，硬件费用仅略有增加。



## **44. System Identification from Partial Observations under Adversarial Attacks**

对抗性攻击下的部分观测结果识别系统 math.OC

9 pages, 2 figures

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00244v1) [paper-pdf](http://arxiv.org/pdf/2504.00244v1)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper is concerned with the partially observed linear system identification, where the goal is to obtain reasonably accurate estimation of the balanced truncation of the true system up to the order $k$ from output measurements. We consider the challenging case of system identification under adversarial attacks, where the probability of having an attack at each time is $\Theta(1/k)$ while the value of the attack is arbitrary. We first show that the $l_1$-norm estimator exactly identifies the true Markov parameter matrix for nilpotent systems under any type of attack. We then build on this result to extend it to general systems and show that the estimation error exponentially decays as $k$ grows. The estimated balanced truncation model accordingly shows an exponentially decaying error for the identification of the true system up to the similarity transformation. This work is the first to provide the input-output analysis of the system with partial observations under arbitrary attacks.

摘要: 本文研究部分观测线性系统的辨识问题，其目标是从输出测量获得对真实系统的平衡截断的合理准确的估计，最高可达$k。我们考虑了对抗性攻击下系统识别的挑战性情形，其中每次攻击的概率为$\theta(1/k)$，而攻击的值是任意的。我们首先证明了在任何类型的攻击下，$L_1$-范数估计精确地识别了幂零系统的真马尔可夫参数矩阵。然后，我们在此结果的基础上将其推广到一般系统，并证明了估计误差随着$k$的增长呈指数衰减。相应地，估计的平衡截断模型显示出直到相似变换的真实系统辨识的指数衰减误差。这项工作首次提供了任意攻击下具有部分观测的系统的输入输出分析。



## **45. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

通过弯曲正规化实现对抗稳健的数据集蒸馏 cs.LG

14 pages, 3 figures

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2403.10045v3) [paper-pdf](http://arxiv.org/pdf/2403.10045v3)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks.

摘要: 数据集蒸馏(DD)允许将数据集提取到原始大小的一小部分，同时保留丰富的分布信息，以便在提取的数据集上训练的模型可以在节省大量计算量的同时获得类似的精度。最近在这一领域的研究一直集中在提高在提取的数据集上训练的模型的准确性。在本文中，我们旨在探索一种新的研究视角。我们研究了如何在提取的数据集中嵌入对抗健壮性，使在这些数据集上训练的模型在保持较高准确率的同时获得更好的对抗健壮性。我们提出了一种新的方法，通过将曲率正则化引入到蒸馏过程中来实现这一目标，与标准的对抗性训练相比，计算开销要小得多。大量的实验表明，我们的方法不仅在准确率和稳健性方面都优于标准的对抗性训练，而且还能够生成能够抵抗各种对抗性攻击的健壮的提取数据集。



## **46. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

$\textit{Agents Under Siege}$：Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks cs.MA

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00218v1) [paper-pdf](http://arxiv.org/pdf/2504.00218v1)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.

摘要: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **47. Backdoor Detection through Replicated Execution of Outsourced Training**

通过复制执行外包培训进行后门检测 cs.CR

Published in the 3rd IEEE Conference on Secure and Trustworthy  Machine Learning (IEEE SaTML 2025)

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00170v1) [paper-pdf](http://arxiv.org/pdf/2504.00170v1)

**Authors**: Hengrui Jia, Sierra Wyllie, Akram Bin Sediq, Ahmed Ibrahim, Nicolas Papernot

**Abstract**: It is common practice to outsource the training of machine learning models to cloud providers. Clients who do so gain from the cloud's economies of scale, but implicitly assume trust: the server should not deviate from the client's training procedure. A malicious server may, for instance, seek to insert backdoors in the model. Detecting a backdoored model without prior knowledge of both the backdoor attack and its accompanying trigger remains a challenging problem. In this paper, we show that a client with access to multiple cloud providers can replicate a subset of training steps across multiple servers to detect deviation from the training procedure in a similar manner to differential testing. Assuming some cloud-provided servers are benign, we identify malicious servers by the substantial difference between model updates required for backdooring and those resulting from clean training. Perhaps the strongest advantage of our approach is its suitability to clients that have limited-to-no local compute capability to perform training; we leverage the existence of multiple cloud providers to identify malicious updates without expensive human labeling or heavy computation. We demonstrate the capabilities of our approach on an outsourced supervised learning task where $50\%$ of the cloud providers insert their own backdoor; our approach is able to correctly identify $99.6\%$ of them. In essence, our approach is successful because it replaces the signature-based paradigm taken by existing approaches with an anomaly-based detection paradigm. Furthermore, our approach is robust to several attacks from adaptive adversaries utilizing knowledge of our detection scheme.

摘要: 将机器学习模型的培训外包给云提供商是常见的做法。这样做的客户从云的规模经济中受益，但隐含地承担信任：服务器不应偏离客户的培训过程。例如，恶意服务器可能会试图在模型中插入后门。在没有后门攻击及其伴随触发的先验知识的情况下检测后门模型仍然是一个具有挑战性的问题。在本文中，我们展示了可以访问多个云提供商的客户端可以在多个服务器上复制训练步骤的子集，以类似于差分测试的方式检测训练过程的偏差。假设一些云提供的服务器是良性的，我们通过后门所需的模型更新和干净训练所产生的模型更新之间的实质性差异来识别恶意服务器。也许我们方法的最大优势是它适用于本地计算能力有限或没有本地计算能力来执行训练的客户端;我们利用多个云提供商的存在来识别恶意更新，而无需昂贵的人工标记或繁重的计算。我们展示了我们的方法在外包监督学习任务上的能力，其中50%的云提供商插入自己的后门，我们的方法能够正确识别其中的99.6%。本质上，我们的方法是成功的，因为它用基于异常的检测范式取代了现有方法采用的基于签名的范式。此外，我们的方法对利用我们的检测方案知识的自适应对手的多种攻击具有鲁棒性。



## **48. Privacy-Preserving Secure Neighbor Discovery for Wireless Networks**

无线网络的隐私保护安全邻居发现 cs.CR

10 pages, 6 figures. Author's version; accepted and presented at the  IEEE 23rd International Conference on Trust, Security and Privacy in  Computing and Communications (TrustCom) 2024

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.22232v2) [paper-pdf](http://arxiv.org/pdf/2503.22232v2)

**Authors**: Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Traditional Neighbor Discovery (ND) and Secure Neighbor Discovery (SND) are key elements for network functionality. SND is a hard problem, satisfying not only typical security properties (authentication, integrity) but also verification of direct communication, which involves distance estimation based on time measurements and device coordinates. Defeating relay attacks, also known as "wormholes", leading to stealthy Byzantine links and significant degradation of communication and adversarial control, is key in many wireless networked systems. However, SND is not concerned with privacy; it necessitates revealing the identity and location of the device(s) participating in the protocol execution. This can be a deterrent for deployment, especially involving user-held devices in the emerging Internet of Things (IoT) enabled smart environments. To address this challenge, we present a novel Privacy-Preserving Secure Neighbor Discovery (PP-SND) protocol, enabling devices to perform SND without revealing their actual identities and locations, effectively decoupling discovery from the exposure of sensitive information. We use Homomorphic Encryption (HE) for computing device distances without revealing their actual coordinates, as well as employing a pseudonymous device authentication to hide identities while preserving communication integrity. PP-SND provides SND [1] along with pseudonymity, confidentiality, and unlinkability. Our presentation here is not specific to one wireless technology, and we assess the performance of the protocols (cryptographic overhead) on a Raspberry Pi 4 and provide a security and privacy analysis.

摘要: 传统邻居发现(ND)和安全邻居发现(SND)是网络功能的关键要素。SND是一个困难的问题，它不仅满足典型的安全特性(认证、完整性)，而且还满足直接通信的验证，这涉及到基于时间测量和设备坐标的距离估计。在许多无线网络系统中，击败中继攻击是关键，这种攻击也被称为“虫洞”，导致秘密的拜占庭链路，通信和敌方控制的显著降级。然而，SND并不关心隐私；它需要透露参与协议执行的设备(S)的身份和位置。这可能会阻碍部署，尤其是涉及支持物联网(IoT)的新兴智能环境中的用户手持设备。为了应对这一挑战，我们提出了一种新的隐私保护安全邻居发现协议(PP-SND)，使设备能够在不透露其实际身份和位置的情况下执行SND，有效地将发现与敏感信息的暴露分离。我们使用同态加密(HE)来计算设备距离而不暴露它们的实际坐标，以及使用假名设备认证来隐藏身份同时保持通信完整性。PP-SND提供SND[1]以及假名、机密性和不可链接性。我们在这里的演示并不是特定于一种无线技术，我们评估了Raspberry PI 4上协议的性能(密码开销)，并提供了安全和隐私分析。



## **49. Adversarially Robust Learning with Optimal Transport Regularized Divergences**

具有最佳传输正规化分歧的对抗鲁棒学习 cs.LG

33 pages, 2 figures

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2309.03791v2) [paper-pdf](http://arxiv.org/pdf/2309.03791v2)

**Authors**: Jeremiah Birrell, Reza Ebrahimi

**Abstract**: We introduce a new class of optimal-transport-regularized divergences, $D^c$, constructed via an infimal convolution between an information divergence, $D$, and an optimal-transport (OT) cost, $C$, and study their use in distributionally robust optimization (DRO). In particular, we propose the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These DRO-based methods are defined by minimizing the maximum expected loss over a $D^c$-neighborhood of the empirical distribution of the training data. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence; the addition of a principled and dynamical adversarial re-weighting on top of adversarial sample transport is a key innovation of $ARMOR_D$. $ARMOR_D$ can be viewed as a generalization of the best-performing loss functions and OT costs in the adversarial training literature; we demonstrate this flexibility by using $ARMOR_D$ to augment the UDR, TRADES, and MART methods and obtain improved performance on CIFAR-10 and CIFAR-100 image recognition. Specifically, augmenting with $ARMOR_D$ leads to 1.9\% and 2.1\% improvement against AutoAttack, a powerful ensemble of adversarial attacks, on CIFAR-10 and CIFAR-100 respectively. To foster reproducibility, we made the code accessible at https://github.com/star-ailab/ARMOR.

摘要: 我们引入了一类新的最优传输正规化分歧$D ' c '，通过信息分歧$D$和最优传输（OT）成本$C$之间的小卷积构建，并研究它们在分布式鲁棒优化（DRO）中的用途。特别是，我们提出了$ARMOR_D$方法作为增强深度学习模型对抗鲁棒性的新颖方法。这些基于DRO的方法是通过最小化训练数据经验分布的$D ' c '附近的最大预期损失来定义的。作为构建对抗性样本的工具，我们的方法允许根据OT成本传输样本，并根据信息差异重新加权;在对抗性样本传输之上添加有原则且动态的对抗性重新加权是$ARMOR_D$的关键创新。$ARMOR_D$可以被视为对抗训练文献中表现最佳的损失函数和OT成本的概括;我们通过使用$ARMOR_D$来增强UDR、TRADES和MART方法并在CIFAR-10和CIFAR-100图像识别上获得更好的性能来证明这种灵活性。具体来说，在CIFAR-10和CIFAR-100上，使用$ARMOR_D$进行增强后，AutoAttack（一种强大的对抗性攻击集合）分别提高了1.9%和2.1%。为了提高可重复性，我们在https://github.com/star-ailab/ARMOR上提供了该代码。



## **50. Pay More Attention to the Robustness of Prompt for Instruction Data Mining**

重视教学提示数据挖掘的鲁棒性 cs.AI

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.24028v1) [paper-pdf](http://arxiv.org/pdf/2503.24028v1)

**Authors**: Qiang Wang, Dawei Feng, Xu Zhang, Ao Shen, Yang Xu, Bo Ding, Huaimin Wang

**Abstract**: Instruction tuning has emerged as a paramount method for tailoring the behaviors of LLMs. Recent work has unveiled the potential for LLMs to achieve high performance through fine-tuning with a limited quantity of high-quality instruction data. Building upon this approach, we further explore the impact of prompt's robustness on the selection of high-quality instruction data. This paper proposes a pioneering framework of high-quality online instruction data mining for instruction tuning, focusing on the impact of prompt's robustness on the data mining process. Our notable innovation, is to generate the adversarial instruction data by conducting the attack for the prompt of online instruction data. Then, we introduce an Adversarial Instruction-Following Difficulty metric to measure how much help the adversarial instruction data can provide to the generation of the corresponding response. Apart from it, we propose a novel Adversarial Instruction Output Embedding Consistency approach to select high-quality online instruction data. We conduct extensive experiments on two benchmark datasets to assess the performance. The experimental results serve to underscore the effectiveness of our proposed two methods. Moreover, the results underscore the critical practical significance of considering prompt's robustness.

摘要: 指令调优已成为定制LLM行为的最重要方法。最近的工作揭示了LLM通过有限数量的高质量指令数据进行微调来实现高性能的潜力。在这种方法的基础上，我们进一步探讨了提示的稳健性对高质量指令数据选择的影响。本文提出了一种用于指令调优的高质量在线指令数据挖掘的开创性框架，重点关注提示的鲁棒性对数据挖掘过程的影响。我们值得注意的创新是通过对在线指令数据的提示进行攻击来生成对抗性指令数据。然后，我们引入对抗性指令跟踪难度指标来衡量对抗性指令数据可以为相应响应的生成提供多少帮助。除此之外，我们还提出了一种新型的对抗性指令输出嵌入一致性方法来选择高质量的在线指令数据。我们对两个基准数据集进行了广泛的实验以评估性能。实验结果强调了我们提出的两种方法的有效性。此外，结果强调了考虑提示稳健性的关键实际意义。



