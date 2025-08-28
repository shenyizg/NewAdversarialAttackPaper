# Latest Adversarial Attack Papers
**update at 2025-08-28 10:40:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Scaling Decentralized Learning with FLock**

利用Flock扩展去中心化学习 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **2. Cell-Free Massive MIMO-Based Physical-Layer Authentication**

无细胞大规模基于MMO的物理层认证 eess.SP

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19931v1) [paper-pdf](http://arxiv.org/pdf/2508.19931v1)

**Authors**: Isabella W. G. da Silva, Zahra Mobini, Hien Quoc Ngo, Michail Matthaiou

**Abstract**: In this paper, we exploit the cell-free massive multiple-input multiple-output (CF-mMIMO) architecture to design a physical-layer authentication (PLA) framework that can simultaneously authenticate multiple distributed users across the coverage area. Our proposed scheme remains effective even in the presence of active adversaries attempting impersonation attacks to disrupt the authentication process. Specifically, we introduce a tag-based PLA CFmMIMO system, wherein the access points (APs) first estimate their channels with the legitimate users during an uplink training phase. Subsequently, a unique secret key is generated and securely shared between each user and the APs. We then formulate a hypothesis testing problem and derive a closed-form expression for the probability of detection for each user in the network. Numerical results validate the effectiveness of the proposed approach, demonstrating that it maintains a high detection probability even as the number of users in the system increases.

摘要: 本文利用无单元大规模多输入多输出（CF-mMMO）架构设计了一个物理层认证（PLA）框架，该框架可以同时对覆盖区域内的多个分布式用户进行认证。即使存在试图模仿攻击以破坏身份验证过程的活跃对手，我们提出的方案仍然有效。具体来说，我们引入了一种基于标签的PLA CFmMMO系统，其中接入点（AP）首先在上行链路训练阶段估计其与合法用户的信道。随后，生成唯一的秘密密钥并在每个用户和AP之间安全共享。然后，我们制定一个假设测试问题，并推导出网络中每个用户的检测概率的封闭形式表达。数值结果验证了所提出方法的有效性，表明即使系统中用户数量增加，它也能保持高检测概率。



## **3. When AIOps Become "AI Oops": Subverting LLM-driven IT Operations via Telemetry Manipulation**

当AIops成为“AI Oops”：通过远程操纵颠覆LLM驱动的IT运营 cs.CR

v0.2

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.06394v2) [paper-pdf](http://arxiv.org/pdf/2508.06394v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese, Omer Akgul, Athanasios Theocharis, Petros Efstathopoulos

**Abstract**: AI for IT Operations (AIOps) is transforming how organizations manage complex software systems by automating anomaly detection, incident diagnosis, and remediation. Modern AIOps solutions increasingly rely on autonomous LLM-based agents to interpret telemetry data and take corrective actions with minimal human intervention, promising faster response times and operational cost savings.   In this work, we perform the first security analysis of AIOps solutions, showing that, once again, AI-driven automation comes with a profound security cost. We demonstrate that adversaries can manipulate system telemetry to mislead AIOps agents into taking actions that compromise the integrity of the infrastructure they manage. We introduce techniques to reliably inject telemetry data using error-inducing requests that influence agent behavior through a form of adversarial reward-hacking; plausible but incorrect system error interpretations that steer the agent's decision-making. Our attack methodology, AIOpsDoom, is fully automated--combining reconnaissance, fuzzing, and LLM-driven adversarial input generation--and operates without any prior knowledge of the target system.   To counter this threat, we propose AIOpsShield, a defense mechanism that sanitizes telemetry data by exploiting its structured nature and the minimal role of user-generated content. Our experiments show that AIOpsShield reliably blocks telemetry-based attacks without affecting normal agent performance.   Ultimately, this work exposes AIOps as an emerging attack vector for system compromise and underscores the urgent need for security-aware AIOps design.

摘要: IT运营人工智能（AIops）正在通过自动化异常检测、事件诊断和修复来改变组织管理复杂软件系统的方式。现代AIops解决方案越来越依赖基于LLM的自主代理来解释遥感数据并以最少的人为干预采取纠正措施，从而承诺更快的响应时间并节省运营成本。   在这项工作中，我们对AIops解决方案进行了首次安全分析，再次表明人工智能驱动的自动化带来了巨大的安全成本。我们证明，对手可以操纵系统遥感来误导AIops代理采取损害其管理基础设施完整性的行动。我们引入了使用导致错误的请求可靠地注入遥感数据的技术，这些请求通过一种对抗性奖励黑客的形式影响代理的行为;看似合理但不正确的系统错误解释来指导代理的决策。我们的攻击方法AIOpsDoom是完全自动化的--结合了侦察、模糊处理和LLM驱动的对抗输入生成--并且在不了解目标系统的情况下运行。   为了应对这一威胁，我们提出了AIOpsShield，这是一种防御机制，通过利用遥感数据的结构化性质和用户生成内容的最小作用来净化遥感数据。我们的实验表明，AIOpsShield可以可靠地阻止基于远程测量的攻击，而不会影响正常的代理性能。   最终，这项工作暴露了AIops作为系统危害的新兴攻击载体，并强调了对安全意识的AIops设计的迫切需要。



## **4. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

Databench：从对抗角度评估深度学习中的数据集审计 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.05622v2) [paper-pdf](http://arxiv.org/pdf/2507.05622v2)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available at https://github.com/shaoshuo-ss/DATABench.

摘要: 深度学习在不同领域的广泛应用关键取决于训练数据集的质量和组成。然而，普遍缺乏对其使用情况的披露，引发了严重的隐私和版权问题。数据集审计技术旨在确定特定数据集是否用于训练给定的可疑模型，为解决这些透明度差距提供了有希望的解决方案。虽然之前的工作已经开发了各种审计方法，但它们对专门对抗攻击的弹性在很大程度上仍未被探索。为了弥合这一差距，本文发起了一项全面的研究，从对抗的角度评估数据集审计。我们首先引入一种新颖的分类法，根据现有方法对内部特征（IF）（数据固有的）和外部特征（EF）（人为引入以进行审计）的依赖来对现有方法进行分类。随后，我们制定了两种主要的攻击类型：逃避攻击（旨在隐藏数据集的使用）和伪造攻击（旨在错误地暗示未使用的数据集）。在对现有方法和攻击目标的理解的基础上，我们进一步提出了系统性攻击策略：脱钩、删除和检测逃避;基于对抗性示例的伪造方法。这些公式和策略导致了我们的新基准Databench，其中包括17种规避攻击、5种伪造攻击和9种代表性审计方法。使用Databench进行的广泛评估表明，在对抗环境下，所评估的审计方法都不够稳健或独特。这些发现凸显了开发一种能够承受复杂对抗操纵的更安全、更可靠的数据集审计方法的迫切需要。代码可在https://github.com/shaoshuo-ss/DATABench上获取。



## **5. Secure Set-based State Estimation for Safety-Critical Applications under Adversarial Attacks on Sensors**

传感器对抗攻击下安全关键应用的安全基于集的状态估计 eess.SY

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2309.05075v3) [paper-pdf](http://arxiv.org/pdf/2309.05075v3)

**Authors**: M. Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: Set-based state estimation provides guaranteed state inclusion certificates that are crucial for the safety verification of dynamical systems. However, when system sensors are subject to cyberattacks, maintaining both safety and security guarantees becomes a fundamental challenge that existing point-based secure state estimation methods cannot adequately address due to their inherent inability to provide state inclusion certificates. This paper introduces a novel approach that simultaneously ensures safety guarantees through guaranteed state inclusion and security guarantees against sensor attacks, without imposing conservative restrictions on system operation. We propose a Secure Set-based State Estimation (S3E) algorithm that maintains the true system state within the estimated set under sensor attacks, provided the initialization set contains the initial state and the system remains observable from the uncompromised sensor subset. The algorithm gives the estimated set as a collection of constrained zonotopes (agreement sets), which can be employed as robust certificates for verifying whether the system adheres to safety constraints. Furthermore, we demonstrate that the estimated set remains unaffected by attack signals of sufficiently large magnitude and also establish sufficient conditions for attack detection, identification, and filtering. This compels the attacker to only inject signals of small magnitudes to evade detection, thus preserving the accuracy of the estimated set. To address the computational complexity of the algorithm, we offer several strategies for complexity-performance trade-offs. The efficacy of the proposed algorithm is illustrated through several examples, including its application to a three-story building model.

摘要: 基于集的状态估计提供了有保证的状态包含证书，这对于动态系统的安全验证至关重要。然而，当系统传感器遭受网络攻击时，维持安全性和安全保障成为现有的基于点的安全状态估计方法无法充分解决的根本挑战，因为它们固有地无法提供状态包含证书。本文介绍了一种新颖的方法，通过保证的状态包容性和针对传感器攻击的安全保证同时确保安全保证，而不会对系统操作施加保守限制。我们提出了一种基于安全集的状态估计（S3 E）算法，只要初始化集包含初始状态并且系统保持可从未受损害的传感器子集观察，该算法可以在传感器攻击下在估计集中维持真实的系统状态。该算法将估计集提供为受约束的分区（协议集）的集合，其可以用作鲁棒证书，用于验证系统是否遵守安全约束。此外，我们证明估计集不受足够大幅度的攻击信号的影响，并且还为攻击检测、识别和过滤建立了充分的条件。这迫使攻击者只注入小幅度的信号来逃避检测，从而保持估计集的准确性。为了解决算法的计算复杂性，我们提供了多种复杂性与性能权衡的策略。通过几个例子，包括它的应用程序的三层建筑模型所提出的算法的有效性进行说明。



## **6. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

安全调整不应仅仅是一些注意力 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.

摘要: 目前的大型语言模型（LLM）的安全对齐仍然存在漏洞，因为对抗性提示可以有效地绕过它们的安全措施。我们的调查表明，这些安全机制主要依赖于有限的注意头子集：删除或消融这些头会严重危及模型安全。为了识别和评估这些安全关键组件，我们引入了RDSHA，这是一种有针对性的消融方法，它利用模型的拒绝方向来确定主要负责安全行为的注意力。进一步的分析表明，现有的越狱攻击通过选择性地绕过或操纵这些关键注意力头部来利用这种集中。为了解决这个问题，我们提出了AHD，一种新的训练策略，旨在促进分布式编码的安全相关的行为在众多的注意头。实验结果表明，AHD成功地将安全相关功能分配给更多的注意力头。此外，在几种主流越狱攻击下的评估表明，用AHD训练的模型表现出更强的安全鲁棒性，同时保持整体功能效用。



## **7. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

ProARD：渐进式对抗稳健性蒸馏：提供广泛的稳健学生 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2506.07666v3) [paper-pdf](http://arxiv.org/pdf/2506.07666v3)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.

摘要: 对抗鲁棒性蒸馏（ARD）已成为增强轻量级深度神经网络抵御对抗攻击鲁棒性的有效方法。当前的ARD方法利用了一个强大的教师网络来培训一个强大的轻量级学生。然而，由于边缘设备的多样性和资源限制，当前的方法需要从头开始训练新的学生网络以满足特定的限制，从而导致巨大的计算成本和二氧化碳排放量增加。本文提出了渐进对抗鲁棒蒸馏（ProARD），可以对动态网络进行高效的一次性训练，该网络支持各种准确且稳健的学生网络，而无需再培训。我们首先基于动态层构建动态深度神经网络，通过涵盖每个设计阶段的宽度、深度和扩展的变化，以支持广泛的架构。然后，我们将规模最大的学生网络视为动态教师网络。ProARD使用权重共享机制训练这个动态网络，以联合优化动态教师网络及其内部学生网络。然而，由于计算动态网络中所有学生的精确梯度的计算成本很高，因此需要采样机制来选择学生的子集。我们表明，每次迭代中的随机学生抽样无法产生准确和稳健的学生。



## **8. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

R-TPT：通过测试时提示调优提高视觉语言模型的对抗鲁棒性 cs.LG

CVPR 2025 (Corrected the results on the Aircraft dataset)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2504.11195v2) [paper-pdf](http://arxiv.org/pdf/2504.11195v2)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.

摘要: CLIP等视觉语言模型（VLM）作为基础模型已受到广泛欢迎，并开发了多种微调方法来增强下游任务的性能。然而，由于其固有的脆弱性以及从有限的开源模型集中进行选择的常见做法，VLM比传统视觉模型面临更高的对抗攻击风险。现有的防御技术通常依赖于训练期间的对抗微调，这需要标记数据并且缺乏下游任务的灵活性。为了解决这些限制，我们提出了鲁棒的测试时即时调优（R-TPT），它可以减轻推理阶段对抗性攻击的影响。我们首先通过消除在对抗条件下引入冲突的术语来重新制定经典的边际熵目标，只保留逐点的熵最小化。此外，我们引入了一种即插即用的、基于可靠性的加权集成策略，该策略从可靠的增强视图中聚合有用信息以加强防御。R-TPT增强了对对抗攻击的防御，而不需要标记的训练数据，同时为推理任务提供高度灵活性。对广泛使用的具有各种攻击的基准进行了大量实验，证明了R-TPT的有效性。该代码可在https://github.com/TomSheng21/R-TPT上找到。



## **9. PromptKeeper: Safeguarding System Prompts for LLMs**

PretKeeper：保护LLM的系统预算 cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 系统提示被广泛用于指导大型语言模型（LLM）的输出。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种防御机制，旨在通过解决两个核心挑战来保护系统提示：可靠地检测泄漏和减轻发生泄漏时的侧通道漏洞。通过将检测视为假设测试问题，SpectKeeper有效地识别显式和微妙的泄漏。检测到泄漏后，它会使用虚拟提示重新生成响应，确保在不存在泄漏时输出与典型交互没有区别。EntKeeper确保针对通过对抗性或常规查询的即时提取攻击提供强大的保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **10. A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives**

模型提取攻击和防御的系统性调查：最新技术水平和观点 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.15031v2) [paper-pdf](http://arxiv.org/pdf/2508.15031v2)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at https://github.com/kzhao5/ModelExtractionPapers.

摘要: 机器学习（ML）模型的复杂性和实用性显着增长，推动了多个领域的进步。然而，大量的计算资源和专业知识历来限制了它们的广泛采用。机器学习即服务（MLaaz）平台通过用户友好的API提供对复杂ML模型的可扩展、方便且经济实惠的访问，从而解决了这些障碍。虽然这种可访问性促进了高级ML功能的广泛使用，但它也引入了通过模型提取攻击（MEAs）利用的漏洞。最近的研究表明，对手可以通过与公开的界面交互来系统性地复制目标模型的功能，从而对知识产权、隐私和系统安全构成威胁。在本文中，我们对多边环境协定和相应的防御策略进行了全面的调查。我们提出了一种新颖的分类法，根据攻击机制、防御方法和计算环境对多边环境进行分类。我们的分析涵盖了各种攻击技术，评估了它们的有效性，并强调了现有防御所面临的挑战，特别是保留模型实用性和确保安全性之间的关键权衡。我们进一步评估不同计算范式中的多边环境协定，并讨论其技术、道德、法律和社会影响，以及未来研究的有希望的方向。这项系统性调查旨在为从事人工智能安全和隐私的研究人员、从业者和政策制定者提供宝贵的参考。此外，我们还在https://github.com/kzhao5/ModelExtractionPapers上维护了一个在线知识库，不断更新相关文献。



## **11. Servant, Stalker, Predator: How An Honest, Helpful, And Harmless (3H) Agent Unlocks Adversarial Skills**

仆人、跟踪者、掠夺者：诚实、乐于助人、无害（3 H）特工如何释放对抗技能 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19500v1) [paper-pdf](http://arxiv.org/pdf/2508.19500v1)

**Authors**: David Noever

**Abstract**: This paper identifies and analyzes a novel vulnerability class in Model Context Protocol (MCP) based agent systems. The attack chain describes and demonstrates how benign, individually authorized tasks can be orchestrated to produce harmful emergent behaviors. Through systematic analysis using the MITRE ATLAS framework, we demonstrate how 95 agents tested with access to multiple services-including browser automation, financial analysis, location tracking, and code deployment-can chain legitimate operations into sophisticated attack sequences that extend beyond the security boundaries of any individual service. These red team exercises survey whether current MCP architectures lack cross-domain security measures necessary to detect or prevent a large category of compositional attacks. We present empirical evidence of specific attack chains that achieve targeted harm through service orchestration, including data exfiltration, financial manipulation, and infrastructure compromise. These findings reveal that the fundamental security assumption of service isolation fails when agents can coordinate actions across multiple domains, creating an exponential attack surface that grows with each additional capability. This research provides a barebones experimental framework that evaluate not whether agents can complete MCP benchmark tasks, but what happens when they complete them too well and optimize across multiple services in ways that violate human expectations and safety constraints. We propose three concrete experimental directions using the existing MCP benchmark suite.

摘要: 本文识别并分析了基于模型上下文协议（MAO）的代理系统中的一种新型漏洞类别。攻击链描述并演示了如何精心策划良性的、单独授权的任务来产生有害的紧急行为。通过使用MITRE ATLAS框架的系统分析，我们展示了95个经过测试的代理如何访问多种服务（包括浏览器自动化、财务分析、位置跟踪和代码部署）可以将合法操作链接到复杂的攻击序列中，这些攻击序列超出了任何单个服务的安全边界。这些红队练习调查当前的LCP架构是否缺乏检测或防止大型组合攻击所需的跨域安全措施。我们提供了特定攻击链的经验证据，这些攻击链通过服务编排（包括数据泄露、金融操纵和基础设施损害）来实现有针对性的伤害。这些发现表明，当代理可以协调多个域之间的操作时，服务隔离的基本安全假设就会失败，从而创建指数级攻击面，并且随着每一个额外的能力而增长。这项研究提供了一个基本的实验框架，该框架不是评估代理是否能够完成LCP基准任务，而是评估当他们完成得太好并以违反人类期望和安全约束的方式在多个服务中进行优化时会发生什么。我们使用现有的LCP基准套件提出了三个具体的实验方向。



## **12. PoolFlip: A Multi-Agent Reinforcement Learning Security Environment for Cyber Defense**

PoolFlip：用于网络防御的多智能体强化学习安全环境 cs.LG

Accepted at GameSec 2025

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19488v1) [paper-pdf](http://arxiv.org/pdf/2508.19488v1)

**Authors**: Xavier Cadet, Simona Boboila, Sie Hendrata Dharmawan, Alina Oprea, Peter Chin

**Abstract**: Cyber defense requires automating defensive decision-making under stealthy, deceptive, and continuously evolving adversarial strategies. The FlipIt game provides a foundational framework for modeling interactions between a defender and an advanced adversary that compromises a system without being immediately detected. In FlipIt, the attacker and defender compete to control a shared resource by performing a Flip action and paying a cost. However, the existing FlipIt frameworks rely on a small number of heuristics or specialized learning techniques, which can lead to brittleness and the inability to adapt to new attacks. To address these limitations, we introduce PoolFlip, a multi-agent gym environment that extends the FlipIt game to allow efficient learning for attackers and defenders. Furthermore, we propose Flip-PSRO, a multi-agent reinforcement learning (MARL) approach that leverages population-based training to train defender agents equipped to generalize against a range of unknown, potentially adaptive opponents. Our empirical results suggest that Flip-PSRO defenders are $2\times$ more effective than baselines to generalize to a heuristic attack not exposed in training. In addition, our newly designed ownership-based utility functions ensure that Flip-PSRO defenders maintain a high level of control while optimizing performance.

摘要: 网络防御需要在隐秘、欺骗性和不断演变的对抗策略下实现防御决策的自动化。FlipIt游戏提供了一个基础框架，用于建模防御者和高级对手之间的交互，这种交互可以在不被立即检测到的情况下危害系统。在FlipIt中，攻击者和防御者通过执行翻转动作并支付费用来竞争控制共享资源。然而，现有的FlipIt框架依赖于少数启发式或专业学习技术，这可能会导致脆弱性和无法适应新的攻击。为了解决这些限制，我们引入了PoolFlip，这是一个多代理健身房环境，它扩展了FlipIt游戏，使攻击者和防御者能够进行高效学习。此外，我们提出了Flip-PSRO，这是一种多智能体强化学习（MARL）方法，它利用基于人群的训练来训练防御者智能体，能够针对一系列未知的、潜在的适应性对手进行概括。我们的经验结果表明，Flip-PSRO防御者在推广到训练中未暴露的启发式攻击方面比基线有效2倍。此外，我们新设计的基于所有权的实用程序功能可确保Flip-PSRO防御者在优化性能的同时保持高水平的控制。



## **13. ReLATE+: Unified Framework for Adversarial Attack Detection, Classification, and Resilient Model Selection in Time-Series Classification**

ReLATE+：时间序列分类中对抗性攻击检测、分类和弹性模型选择的统一框架 cs.CR

Under review at IEEE TSMC Journal. arXiv admin note: text overlap  with arXiv:2503.07882

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19456v1) [paper-pdf](http://arxiv.org/pdf/2508.19456v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge due to the high complexity of model architectures and the large volume of sequential data that must be processed in real time. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. To address this challenge, we propose ReLATE+, a comprehensive framework that detects and classifies adversarial attacks, adaptively selects deep learning models based on dataset-level similarity, and thus substantially reduces retraining costs relative to conventional methods that do not leverage prior knowledge, while maintaining strong performance. ReLATE+ first checks whether the incoming data is adversarial and, if so, classifies the attack type, using this insight to identify a similar dataset from a repository and enable the reuse of the best-performing associated model. This approach ensures strong performance while reducing the need for retraining, and it generalizes well across different domains with varying data distributions and feature spaces. Experiments show that ReLATE+ reduces computational overhead by an average of 77.68%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 2.02% of Oracle.

摘要: 由于模型架构的高复杂性和必须实时处理的大量顺序数据，最大限度地减少时间序列分类中的计算负担，特别是深度学习模型中的计算负担带来了重大挑战。对抗性攻击进一步加剧了这一挑战，强调了对确保稳健性能和高效模型选择的弹性方法的需要。为了应对这一挑战，我们提出了ReLATE+，这是一个全面的框架，可以检测和分类对抗性攻击，根据互联网级别的相似性自适应地选择深度学习模型，从而相对于不利用先验知识的传统方法大幅降低了再培训成本，同时保持强劲的性能。ReLATE+首先检查输入的数据是否是对抗性的，如果是，则对攻击类型进行分类，使用此见解从存储库中识别类似的数据集，并启用性能最佳的关联模型的重用。这种方法确保了强大的性能，同时减少了再培训的需要，并且它可以很好地在具有不同数据分布和特征空间的不同领域中推广。实验表明，ReLATE+平均降低了77.68%的计算负担，增强了对抗弹性并简化了稳健的模型选择，而所有这些都不会牺牲性能，仅在Oracle的2.02%之内。



## **14. On Surjectivity of Neural Networks: Can you elicit any behavior from your model?**

关于神经网络的满摄性：你能从你的模型中引出任何行为吗？ cs.LG

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19445v1) [paper-pdf](http://arxiv.org/pdf/2508.19445v1)

**Authors**: Haozhe Jiang, Nika Haghtalab

**Abstract**: Given a trained neural network, can any specified output be generated by some input? Equivalently, does the network correspond to a function that is surjective? In generative models, surjectivity implies that any output, including harmful or undesirable content, can in principle be generated by the networks, raising concerns about model safety and jailbreak vulnerabilities. In this paper, we prove that many fundamental building blocks of modern neural architectures, such as networks with pre-layer normalization and linear-attention modules, are almost always surjective. As corollaries, widely used generative frameworks, including GPT-style transformers and diffusion models with deterministic ODE solvers, admit inverse mappings for arbitrary outputs. By studying surjectivity of these modern and commonly used neural architectures, we contribute a formalism that sheds light on their unavoidable vulnerability to a broad class of adversarial attacks.

摘要: 给定一个经过训练的神经网络，任何指定的输出都可以由某些输入生成吗？同样，网络是否对应于满射函数？在生成模型中，主观性意味着任何输出，包括有害或不受欢迎的内容，原则上都可以由网络生成，这引发了对模型安全性和越狱漏洞的担忧。在本文中，我们证明了现代神经架构的许多基本构建模块，例如具有预层规范化和线性注意力模块的网络，几乎总是满射的。作为推论，广泛使用的生成式框架（包括GPT式转换器和具有确定性ODE解算器的扩散模型）允许任意输出的逆映射。通过研究这些现代和常用的神经架构的主观性，我们提出了一种形式主义，揭示了它们不可避免地容易受到一类对抗攻击的脆弱性。



## **15. Formal Verification of Physical Layer Security Protocols for Next-Generation Communication Networks**

下一代通信网络物理层安全协议的形式化验证 cs.CR

Submitted to ICFEM2025; 23 pages, 2 tables, and 6 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19430v1) [paper-pdf](http://arxiv.org/pdf/2508.19430v1)

**Authors**: Kangfeng Ye, Roberto Metere, Jim Woodcock, Poonam Yadav

**Abstract**: Formal verification is crucial for ensuring the robustness of security protocols against adversarial attacks. The Needham-Schroeder protocol, a foundational authentication mechanism, has been extensively studied, including its integration with Physical Layer Security (PLS) techniques such as watermarking and jamming. Recent research has used ProVerif to verify these mechanisms in terms of secrecy. However, the ProVerif-based approach limits the ability to improve understanding of security beyond verification results. To overcome these limitations, we re-model the same protocol using an Isabelle formalism that generates sound animation, enabling interactive and automated formal verification of security protocols. Our modelling and verification framework is generic and highly configurable, supporting both cryptography and PLS. For the same protocol, we have conducted a comprehensive analysis (secrecy and authenticity in four different eavesdropper locations under both passive and active attacks) using our new web interface. Our findings not only successfully reproduce and reinforce previous results on secrecy but also reveal an uncommon but expected outcome: authenticity is preserved across all examined scenarios, even in cases where secrecy is compromised. We have proposed a PLS-based Diffie-Hellman protocol that integrates watermarking and jamming, and our analysis shows that it is secure for deriving a session key with required authentication. These highlight the advantages of our novel approach, demonstrating its robustness in formally verifying security properties beyond conventional methods.

摘要: 形式验证对于确保安全协议抵御对抗攻击的稳健性至关重要。Needham-Schroeder协议是一种基础认证机制，已被广泛研究，包括其与水印和干扰等物理层安全（SCS）技术的集成。最近的研究使用ProVerif来验证这些机制的保密性。然而，基于ProVerif的方法限制了提高对验证结果之外的安全性理解的能力。为了克服这些限制，我们使用Isabelle形式主义重新建模相同的协议，该形式主义生成声音动画，从而实现安全协议的交互式和自动化形式验证。我们的建模和验证框架是通用的且高度可配置的，支持加密技术和最大限度地支持。对于同一协议，我们使用新的网络界面进行了全面的分析（被动和主动攻击下四个不同窃听者位置的保密性和真实性）。我们的发现不仅成功地复制和强化了之前的保密结果，而且揭示了一个不寻常但预期的结果：在所有检查的场景中，真实性都得到了保留，即使在保密性受到损害的情况下。我们提出了一种基于PL的迪夫-赫尔曼协议，集成了水印和干扰，我们的分析表明，它对于推导具有所需认证的会话密钥是安全的。这些凸显了我们的新型方法的优势，证明了其在正式验证安全属性方面的鲁棒性，超出了传统方法。



## **16. Attackers Strike Back? Not Anymore -- An Ensemble of RL Defenders Awakens for APT Detection**

攻击者反击？不再是--RL捍卫者群体为APT检测觉醒 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19072v1) [paper-pdf](http://arxiv.org/pdf/2508.19072v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) represent a growing menace to modern digital infrastructure. Unlike traditional cyberattacks, APTs are stealthy, adaptive, and long-lasting, often bypassing signature-based detection systems. This paper introduces a novel framework for APT detection that unites deep learning, reinforcement learning (RL), and active learning into a cohesive, adaptive defense system. Our system combines auto-encoders for latent behavioral encoding with a multi-agent ensemble of RL-based defenders, each trained to distinguish between benign and malicious process behaviors. We identify a critical challenge in existing detection systems: their static nature and inability to adapt to evolving attack strategies. To this end, our architecture includes multiple RL agents (Q-Learning, PPO, DQN, adversarial defenders), each analyzing latent vectors generated by an auto-encoder. When any agent is uncertain about its decision, the system triggers an active learning loop to simulate expert feedback, thus refining decision boundaries. An ensemble voting mechanism, weighted by each agent's performance, ensures robust final predictions.

摘要: 高级持续威胁（APT）对现代数字基础设施构成了日益严重的威胁。与传统的网络攻击不同，APT具有隐蔽性、适应性和持久性，通常绕过基于签名的检测系统。本文介绍了一种新颖的APT检测框架，该框架将深度学习、强化学习（RL）和主动学习结合到一个有凝聚力的自适应防御系统中。我们的系统将用于潜在行为编码的自动编码器与基于RL的防御者的多代理集成相结合，每个防御者都经过训练以区分良性和恶意进程行为。我们发现了现有检测系统中的一个关键挑战：它们的静态性质以及无法适应不断变化的攻击策略。为此，我们的架构包括多个RL代理（Q-Learning、PPO、DQN、对抗防御者），每个代理都分析自动编码器生成的潜在载体。当任何代理人对其决策不确定时，系统会触发主动学习循环来模拟专家反馈，从而细化决策边界。由每个代理的表现加权的集成投票机制确保了稳健的最终预测。



## **17. Beyond-Diagonal RIS: Adversarial Channels and Optimality of Low-Complexity Architectures**

Beyond-Diagonal RIS：对抗渠道和低复杂度架构的最佳性 eess.SP

\copyright 2025 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19000v1) [paper-pdf](http://arxiv.org/pdf/2508.19000v1)

**Authors**: Atso Iivanainen, Robin Rajamäki, Visa Koivunen

**Abstract**: Beyond-diagonal reconfigurable intelligent surfaces (BD-RISs) have recently gained attention as an enhancement to conventional RISs. BD-RISs allow optimizing not only the phase, but also the amplitude responses of their discrete surface elements by introducing adjustable inter-element couplings. Various BD-RIS architectures have been proposed to optimally trade off between average performance and complexity of the architecture. However, little attention has been paid to worst-case performance. This paper characterizes novel sets of adversarial channels for which certain low-complexity BD-RIS architectures have suboptimal performance in terms of received signal power at an intended communications user. Specifically, we consider two recent BD-RIS models: the so-called group-connected and tree-connected architecture. The derived adversarial channel sets reveal new surprising connections between the two architectures. We validate our analytical results numerically, demonstrating that adversarial channels can cause a significant performance loss. Our results pave the way towards efficient BD-RIS designs that are robust to adversarial propagation conditions and malicious attacks.

摘要: 超对角线可重构智能表面（BD-RISs）最近作为传统RISs的增强而受到关注。通过引入可调节的元件间耦合，BD-RIS不仅可以优化其离散表面元件的相响应，还可以优化其离散表面元件的幅度响应。人们提出了各种BD-RIS架构，以在平均性能和架构复杂性之间进行最佳权衡。然而，人们很少关注最坏情况的性能。本文描述了一组新型对抗性通道的特征，对于这些通道，某些低复杂性的BD-RIS架构在预期通信用户的接收信号功率方面具有次优的性能。具体来说，我们考虑了两种最近的BD-RIS模型：所谓的组连接和树连接架构。衍生的对抗通道集揭示了两种架构之间新的令人惊讶的联系。我们通过数字方式验证了我们的分析结果，证明对抗性通道可能会导致显着的性能损失。我们的结果为高效的BD-RIS设计铺平了道路，该设计对对抗传播条件和恶意攻击具有鲁棒性。



## **18. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

基于LLM的数据重建的双刃剑：理解和缓解词级差异隐私文本清理中的上下文漏洞 cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.

摘要: 差异隐私文本清理是指在差异隐私（DP）框架下将文本私有化的过程，提供可证明的隐私保证，同时还根据经验防御试图损害隐私的对手。尽管它们很简单，但在词级操作的DP文本清理方法表现出许多缺点，其中包括由于清理期间的随机性，倾向于从原始文本中留下上下文线索$\unicode{x2013}$我们将其称为$\textit{contextual vulnerability}$。鉴于大型语言模型（LLM）强大的上下文理解和推理能力，我们探索可以在多大程度上利用LLM来利用DP清理文本的上下文脆弱性。我们不仅扩展了之前的工作，还扩展了高级LLM的使用，还扩展了各种隐私级别的更广泛的清理机制。我们的实验揭示了基于LLM的数据重建攻击对隐私和实用性的双刃剑效应：虽然LLM确实可以推断原始语义，有时会降低经验隐私保护，但它们也可以被永久使用，以提高DP净化文本的质量和隐私。根据我们的研究结果，我们提出了使用LLM数据重建作为后处理步骤的建议，通过敌对思维来增加隐私保护。



## **19. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper will be submitted to the Computer Science Review

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.16843v2) [paper-pdf](http://arxiv.org/pdf/2508.16843v2)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **20. EnerSwap: Large-Scale, Privacy-First Automated Market Maker for V2G Energy Trading**

EnerSwap：大规模、隐私优先的V2G能源交易自动化做市商 cs.CR

11 pages, 7 figures, 1 table, 1 algorithm, Paper accepted in 27th  MSWiM Conference

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18942v1) [paper-pdf](http://arxiv.org/pdf/2508.18942v1)

**Authors**: Ahmed Mounsf Rafik Bendada, Yacine Ghamri-Doudane

**Abstract**: With the rapid growth of Electric Vehicle (EV) technology, EVs are destined to shape the future of transportation. The large number of EVs facilitates the development of the emerging vehicle-to-grid (V2G) technology, which realizes bidirectional energy exchanges between EVs and the power grid. This has led to the setting up of electricity markets that are usually confined to a small geographical location, often with a small number of participants. Usually, these markets are manipulated by intermediaries responsible for collecting bids from prosumers, determining the market-clearing price, incorporating grid constraints, and accounting for network losses. While centralized models can be highly efficient, they grant excessive power to the intermediary by allowing them to gain exclusive access to prosumers \textquotesingle price preferences. This opens the door to potential market manipulation and raises significant privacy concerns for users, such as the location of energy providers. This lack of protection exposes users to potential risks, as untrustworthy servers and malicious adversaries can exploit this information to infer trading activities and real identities. This work proposes a secure, decentralized exchange market built on blockchain technology, utilizing a privacy-preserving Automated Market Maker (AMM) model to offer open and fair, and equal access to traders, and mitigates the most common trading-manipulation attacks. Additionally, it incorporates a scalable architecture based on geographical dynamic sharding, allowing for efficient resource allocation and improved performance as the market grows.

摘要: 随着电动汽车（EV）技术的快速发展，电动汽车注定会塑造交通的未来。电动汽车的大量使用促进了新兴的汽车转网（V2 G）技术的发展，该技术实现了电动汽车与电网之间的双向能量交换。这导致了电力市场的建立，这些市场通常仅限于一个较小的地理位置，参与者往往很少。通常，这些市场由负责收集生产者出价、确定市场出清价格、纳入电网限制并核算网络损失的中介机构操纵。虽然集中式模型可能非常高效，但它们通过允许中间商独家访问产品消费者\文本引用单一价格偏好而赋予了他们过多的权力。这为潜在的市场操纵打开了大门，并给用户带来了重大的隐私问题，例如能源提供商的位置。这种缺乏保护使用户面临潜在风险，因为不值得信赖的服务器和恶意对手可以利用这些信息来推断交易活动和真实身份。这项工作提出了一个基于区块链技术的安全、去中心化的交易所市场，利用保护隐私的自动化做市商（AMM）模型为交易员提供开放、公平和平等的准入机会，并减轻最常见的交易操纵攻击。此外，它还结合了基于地理动态分片的可扩展架构，随着市场的增长，可以高效地分配资源并提高性能。



## **21. Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models**

隐藏的尾巴：视觉语言模型中导致隐性资源消耗的敌对图像 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18805v1) [paper-pdf](http://arxiv.org/pdf/2508.18805v1)

**Authors**: Rui Zhang, Zihan Wang, Tianli Yang, Hongwei Li, Wenbo Jiang, Qingchuan Zhao, Yang Liu, Guowen Xu

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in real-world applications, but their high inference cost makes them vulnerable to resource consumption attacks. Prior attacks attempt to extend VLM output sequences by optimizing adversarial images, thereby increasing inference costs. However, these extended outputs often introduce irrelevant abnormal content, compromising attack stealthiness. This trade-off between effectiveness and stealthiness poses a major limitation for existing attacks. To address this challenge, we propose \textit{Hidden Tail}, a stealthy resource consumption attack that crafts prompt-agnostic adversarial images, inducing VLMs to generate maximum-length outputs by appending special tokens invisible to users. Our method employs a composite loss function that balances semantic preservation, repetitive special token induction, and suppression of the end-of-sequence (EOS) token, optimized via a dynamic weighting strategy. Extensive experiments show that \textit{Hidden Tail} outperforms existing attacks, increasing output length by up to 19.2$\times$ and reaching the maximum token limit, while preserving attack stealthiness. These results highlight the urgent need to improve the robustness of VLMs against efficiency-oriented adversarial threats. Our code is available at https://github.com/zhangrui4041/Hidden_Tail.

摘要: 视觉语言模型（VLM）越来越多地部署在现实世界的应用程序中，但其高推理成本使其容易受到资源消耗攻击。先前的攻击试图通过优化对抗图像来扩展VLM输出序列，从而增加推理成本。然而，这些扩展输出通常会引入不相关的异常内容，从而损害攻击的隐蔽性。有效性和隐蔽性之间的这种权衡对现有攻击构成了重大限制。为了应对这一挑战，我们提出了\textit{Hidden Tail}，这是一种隐形的资源消耗攻击，它可以制作不可知的对抗图像，通过附加用户不可见的特殊令牌来诱导VLM生成最大长度的输出。我们的方法采用了一个复合损失函数，平衡语义保存，重复的特殊令牌感应，并通过动态加权策略优化的序列结束（EOS）令牌的抑制。大量的实验表明，\textit{Hidden Tail}优于现有的攻击，将输出长度增加了19.2$\times$，达到了最大令牌限制，同时保持了攻击的隐蔽性。这些结果强调了迫切需要提高VLM对效率导向的对抗性威胁的鲁棒性。我们的代码可在https://github.com/zhangrui4041/Hidden_Tail上获取。



## **22. FLAegis: A Two-Layer Defense Framework for Federated Learning Against Poisoning Attacks**

FLAegis：针对中毒攻击的联邦学习的两层防御框架 cs.LG

15 pages, 5 tables, and 5 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18737v1) [paper-pdf](http://arxiv.org/pdf/2508.18737v1)

**Authors**: Enrique Mármol Campos, Aurora González Vidal, José Luis Hernández Ramos, Antonio Skarmeta

**Abstract**: Federated Learning (FL) has become a powerful technique for training Machine Learning (ML) models in a decentralized manner, preserving the privacy of the training datasets involved. However, the decentralized nature of FL limits the visibility of the training process, relying heavily on the honesty of participating clients. This assumption opens the door to malicious third parties, known as Byzantine clients, which can poison the training process by submitting false model updates. Such malicious clients may engage in poisoning attacks, manipulating either the dataset or the model parameters to induce misclassification. In response, this study introduces FLAegis, a two-stage defensive framework designed to identify Byzantine clients and improve the robustness of FL systems. Our approach leverages symbolic time series transformation (SAX) to amplify the differences between benign and malicious models, and spectral clustering, which enables accurate detection of adversarial behavior. Furthermore, we incorporate a robust FFT-based aggregation function as a final layer to mitigate the impact of those Byzantine clients that manage to evade prior defenses. We rigorously evaluate our method against five poisoning attacks, ranging from simple label flipping to adaptive optimization-based strategies. Notably, our approach outperforms state-of-the-art defenses in both detection precision and final model accuracy, maintaining consistently high performance even under strong adversarial conditions.

摘要: 联邦学习（FL）已经成为一种以分散方式训练机器学习（ML）模型的强大技术，保护了所涉及的训练数据集的隐私。然而，FL的分散性质限制了培训过程的可见性，严重依赖参与客户的诚实。这种假设为恶意的第三方打开了大门，这些第三方被称为拜占庭客户端，它们可以通过提交错误的模型更新来毒害训练过程。这种恶意客户端可能参与中毒攻击，操纵数据集或模型参数以引起错误分类。作为回应，本研究引入了FLAegis，这是一个两阶段防御框架，旨在识别拜占庭客户并提高FL系统的稳健性。我们的方法利用符号时间序列变换（NSX）来放大良性和恶意模型之间的差异，并利用谱集群来实现对对抗行为的准确检测。此外，我们还将强大的基于快速傅立叶变换的聚合功能作为最后一层，以减轻那些设法逃避先前防御的拜占庭客户的影响。我们针对五种中毒攻击严格评估了我们的方法，范围从简单的标签翻转到基于自适应优化的策略。值得注意的是，我们的方法在检测精度和最终模型准确性方面都优于最先进的防御，即使在强烈的对抗条件下也能保持一致的高性能。



## **23. UniC-RAG: Universal Knowledge Corruption Attacks to Retrieval-Augmented Generation**

UniC-RAG：对检索增强一代的普遍知识腐败攻击 cs.CR

21 pages, 4 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18652v1) [paper-pdf](http://arxiv.org/pdf/2508.18652v1)

**Authors**: Runpeng Geng, Yanting Wang, Ying Chen, Jinyuan Jia

**Abstract**: Retrieval-augmented generation (RAG) systems are widely deployed in real-world applications in diverse domains such as finance, healthcare, and cybersecurity. However, many studies showed that they are vulnerable to knowledge corruption attacks, where an attacker can inject adversarial texts into the knowledge database of a RAG system to induce the LLM to generate attacker-desired outputs. Existing studies mainly focus on attacking specific queries or queries with similar topics (or keywords). In this work, we propose UniC-RAG, a universal knowledge corruption attack against RAG systems. Unlike prior work, UniC-RAG jointly optimizes a small number of adversarial texts that can simultaneously attack a large number of user queries with diverse topics and domains, enabling an attacker to achieve various malicious objectives, such as directing users to malicious websites, triggering harmful command execution, or launching denial-of-service attacks. We formulate UniC-RAG as an optimization problem and further design an effective solution to solve it, including a balanced similarity-based clustering method to enhance the attack's effectiveness. Our extensive evaluations demonstrate that UniC-RAG is highly effective and significantly outperforms baselines. For instance, UniC-RAG could achieve over 90% attack success rate by injecting 100 adversarial texts into a knowledge database with millions of texts to simultaneously attack a large set of user queries (e.g., 2,000). Additionally, we evaluate existing defenses and show that they are insufficient to defend against UniC-RAG, highlighting the need for new defense mechanisms in RAG systems.

摘要: 检索增强一代（RAG）系统广泛部署在金融、医疗保健和网络安全等不同领域的现实世界应用中。然而，许多研究表明，它们很容易受到知识腐败攻击，攻击者可以将对抗文本注入RAG系统的知识数据库，以诱导LLM生成攻击者所需的输出。现有的研究主要集中在攻击特定查询或具有相似主题（或关键词）的查询上。在这项工作中，我们提出了UniC-RAG，这是一种针对RAG系统的通用知识腐败攻击。与之前的工作不同，UniC-RAG联合优化了少量对抗性文本，这些文本可以同时攻击大量具有不同主题和域的用户查询，使攻击者能够实现各种恶意目标，例如将用户引导到恶意网站、触发有害命令执行或发起拒绝服务攻击。我们将UniC-RAG表述为一个优化问题，并进一步设计一个有效的解决方案来解决它，包括基于平衡相似性的集群方法来增强攻击的有效性。我们的广泛评估表明UniC-RAG非常有效，并且显着优于基线。例如，UniC-RAG可以通过将100个对抗文本注入到具有数百万个文本的知识数据库中来同时攻击大量用户查询（例如，2，000）。此外，我们评估了现有的防御措施，并表明它们不足以防御UniC-RAG，强调了RAG系统中新防御机制的必要性。



## **24. PRISM: Robust VLM Alignment with Principled Reasoning for Integrated Safety in Multimodality**

PRism：与原则推理的鲁棒VLM对齐，以实现多模式中的综合安全 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18649v1) [paper-pdf](http://arxiv.org/pdf/2508.18649v1)

**Authors**: Nanxi Li, Zhengyue Zhao, Chaowei Xiao

**Abstract**: Safeguarding vision-language models (VLMs) is a critical challenge, as existing methods often suffer from over-defense, which harms utility, or rely on shallow alignment, failing to detect complex threats that require deep reasoning. To this end, we introduce PRISM (Principled Reasoning for Integrated Safety in Multimodality), a system2-like framework that aligns VLMs by embedding a structured, safety-aware reasoning process. Our framework consists of two key components: PRISM-CoT, a dataset that teaches safety-aware chain-of-thought reasoning, and PRISM-DPO, generated via Monte Carlo Tree Search (MCTS) to further refine this reasoning through Direct Preference Optimization to help obtain a delicate safety boundary. Comprehensive evaluations demonstrate PRISM's effectiveness, achieving remarkably low attack success rates including 0.15% on JailbreakV-28K for Qwen2-VL and 90% improvement over the previous best method on VLBreak for LLaVA-1.5. PRISM also exhibits strong robustness against adaptive attacks, significantly increasing computational costs for adversaries, and generalizes effectively to out-of-distribution challenges, reducing attack success rates to just 8.70% on the challenging multi-image MIS benchmark. Remarkably, this robust defense is achieved while preserving, and in some cases enhancing, model utility. To promote reproducibility, we have made our code, data, and model weights available at https://github.com/SaFoLab-WISC/PRISM.

摘要: 保护视觉语言模型（VLM）是一项严峻的挑战，因为现有的方法经常遭受过度防御，从而损害实用性，或者依赖于浅层对齐，无法检测到需要深度推理的复杂威胁。为此，我们引入了PRism（多模式综合安全原则推理），这是一个类似系统2的框架，通过嵌入结构化的安全意识推理过程来对齐VLM。我们的框架由两个关键组件组成：PRISM-CoT，一个教授安全意识思维链推理的数据集，以及PRISM-DPO，通过蒙特卡洛树搜索（MCTS）生成，通过直接偏好优化进一步完善这一推理，以帮助获得微妙的安全边界。全面评估证明了PRISM的有效性，实现了极低的攻击成功率，包括针对Qwen 2-BL的Jailbreak V-28 K的攻击成功率为0.15%，针对LLaVA-1.5的VLBreak的攻击成功率比之前的最佳方法提高了90%。PRISM还对自适应攻击表现出强大的鲁棒性，显著增加了对手的计算成本，并有效地推广到分布外的挑战，在具有挑战性的多图像MIS基准测试中，攻击成功率仅为8.70%。值得注意的是，这种强大的防御是在保持，甚至在某些情况下增强模型效用的同时实现的。为了提高可重复性，我们在https://github.com/SaFoLab-WISC/PRISM上提供了我们的代码、数据和模型权重。



## **25. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .

摘要: 事实证明，大型语言模型（LLM）很容易受到越狱攻击，其中对抗性提示旨在引发有害反应。虽然现有的防御措施通过检测和过滤不安全的输入有效地减轻了单回合攻击，但它们无法对抗利用多次交互中的上下文漂移的多回合越狱，从而逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一个基于安全控制理论的安全引导框架，确保多回合对话中不变的安全性。我们的方法使用状态空间表示对与LLM的对话进行建模，并引入一种新型的神经屏障函数（NBF）来主动检测和过滤不断变化的上下文中出现的有害查询。我们的方法通过学习一个考虑对抗性查询的安全预测器，在每一轮对话中实现不变的安全性，防止潜在的上下文漂移到越狱。在多个LLM下进行的大量实验表明，我们基于NBF的安全转向优于安全对准，基于转向的转向和轻型LLM护栏基线，为多转向越狱提供更强的防御，同时在安全性，有用性和过度拒绝之间保持更好的权衡。查看网站https://sites.google.com/view/llm-nbf/home。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **26. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

传输样式以减少纹理偏差并提高语义分割网络中的鲁棒性 cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.10239v2) [paper-pdf](http://arxiv.org/pdf/2507.10239v2)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.

摘要: 最近的研究调查了深度神经网络（DNN）在图像分类中的形状和纹理偏差，这些偏差会影响其泛化能力和鲁棒性。研究表明，与常规DNN训练相比，使用风格化图像进行训练可以减少图像分类中的纹理偏差，并提高图像损坏的鲁棒性。为了推进这一领域的研究，我们研究了风格转移是否同样可以在语义分割中产生这两种效果。为此，我们执行风格转移，风格在人工图像区域之间有所不同。这些随机区域由选定数量的Voronoi细胞形成。然后使用生成的风格传输数据来训练语义分割DNN，目标是减少它们对纹理线索的依赖，同时增强它们对基于形状的特征的依赖。在我们的实验中，事实证明，在语义分割中，风格转移增强减少了纹理偏差，并大大提高了针对常见图像损坏和对抗攻击的鲁棒性。这些观察结果适用于Cityscapes数据集和Pascal Content上的卷积神经网络和Transformer架构，显示了所提出方法的通用性。



## **27. Quantum-Classical Hybrid Framework for Zero-Day Time-Push GNSS Spoofing Detection**

用于零日时间推送式全球导航卫星欺骗检测的量子经典混合框架 cs.LG

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18085v1) [paper-pdf](http://arxiv.org/pdf/2508.18085v1)

**Authors**: Abyad Enan, Mashrur Chowdhury, Sagar Dasgupta, Mizanur Rahman

**Abstract**: Global Navigation Satellite Systems (GNSS) are critical for Positioning, Navigation, and Timing (PNT) applications. However, GNSS are highly vulnerable to spoofing attacks, where adversaries transmit counterfeit signals to mislead receivers. Such attacks can lead to severe consequences, including misdirected navigation, compromised data integrity, and operational disruptions. Most existing spoofing detection methods depend on supervised learning techniques and struggle to detect novel, evolved, and unseen attacks. To overcome this limitation, we develop a zero-day spoofing detection method using a Hybrid Quantum-Classical Autoencoder (HQC-AE), trained solely on authentic GNSS signals without exposure to spoofed data. By leveraging features extracted during the tracking stage, our method enables proactive detection before PNT solutions are computed. We focus on spoofing detection in static GNSS receivers, which are particularly susceptible to time-push spoofing attacks, where attackers manipulate timing information to induce incorrect time computations at the receiver. We evaluate our model against different unseen time-push spoofing attack scenarios: simplistic, intermediate, and sophisticated. Our analysis demonstrates that the HQC-AE consistently outperforms its classical counterpart, traditional supervised learning-based models, and existing unsupervised learning-based methods in detecting zero-day, unseen GNSS time-push spoofing attacks, achieving an average detection accuracy of 97.71% with an average false negative rate of 0.62% (when an attack occurs but is not detected). For sophisticated spoofing attacks, the HQC-AE attains an accuracy of 98.23% with a false negative rate of 1.85%. These findings highlight the effectiveness of our method in proactively detecting zero-day GNSS time-push spoofing attacks across various stationary GNSS receiver platforms.

摘要: 全球导航卫星系统（GNSS）对于定位、导航和授时（PNT）应用至关重要。然而，全球导航卫星系统非常容易受到欺骗攻击，对手会发送伪造信号来误导接收器。此类攻击可能会导致严重的后果，包括导航错误、数据完整性受损和运营中断。大多数现有的欺骗检测方法依赖于监督学习技术，并且很难检测新颖的、进化的和不可见的攻击。为了克服这一限制，我们使用混合量子经典自动编码器（HQC-AE）开发了一种零日欺骗检测方法，该方法仅根据真实的GPS信号进行训练，而不会暴露于欺骗数据。通过利用在跟踪阶段提取的特征，我们的方法能够在计算PNT解决方案之前进行主动检测。我们重点关注静态GNSS接收器中的欺骗检测，这些接收器特别容易受到时间推送欺骗攻击，攻击者操纵计时信息以在接收器上引发不正确的时间计算。我们针对不同不可见的时间推送欺骗攻击场景来评估我们的模型：简单化、中间化和复杂化。我们的分析表明，HQC-AE在检测零日、不可见的GNSS时间推送欺骗攻击方面始终优于其经典对应物、传统的基于监督学习的模型和现有的基于无监督学习的方法，实现了97.71%的平均检测准确率，平均误报率为0.62%（当攻击发生但未被检测到时）。对于复杂的欺骗攻击，HQC-AE的准确率为98.23%，误报率为1.85%。这些发现凸显了我们的方法在跨各种固定的GNSS接收器平台主动检测零日GNSS时间推送欺骗攻击方面的有效性。



## **28. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

通过基于损失的客户端集群实现对抗性攻击下的鲁棒联邦学习 cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.12672v3) [paper-pdf](http://arxiv.org/pdf/2508.12672v3)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.

摘要: 联合学习（FL）支持跨多个客户端的协作模型训练，而无需共享私有数据。我们考虑FL场景，其中FL客户端受到对抗性（拜占庭）攻击，而FL服务器是可信的（诚实的），并有一个值得信赖的侧数据集。这可以对应于，例如，服务器在联合之前拥有受信任数据，或者存在暂时承担服务器角色的受信任客户端的情况。我们的方法只需要两个诚实的参与者，即服务器和一个客户端，在不了解恶意客户端数量的情况下有效运行。理论分析表明，即使在强大的拜占庭攻击下，也存在有限的最优性差距。实验结果表明，我们的算法显着优于标准和强大的FL基线，如平均值，修剪平均值，中位数，克鲁姆，和多克鲁姆下的各种攻击策略，包括标签翻转，符号翻转，高斯噪声添加在MNIST，FMNIST，和CIFAR-10基准使用花框架。



## **29. FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning**

FedGreed：一种用于联邦学习的拜占庭鲁棒的基于损失的聚合方法 cs.LG

8 pages, 4 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18060v1) [paper-pdf](http://arxiv.org/pdf/2508.18060v1)

**Authors**: Emmanouil Kritharakis, Antonios Makris, Dusan Jakovetic, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients while preserving data privacy by keeping local datasets on-device. In this work, we address FL settings where clients may behave adversarially, exhibiting Byzantine attacks, while the central server is trusted and equipped with a reference dataset. We propose FedGreed, a resilient aggregation strategy for federated learning that does not require any assumptions about the fraction of adversarial participants. FedGreed orders clients' local model updates based on their loss metrics evaluated against a trusted dataset on the server and greedily selects a subset of clients whose models exhibit the minimal evaluation loss. Unlike many existing approaches, our method is designed to operate reliably under heterogeneous (non-IID) data distributions, which are prevalent in real-world deployments. FedGreed exhibits convergence guarantees and bounded optimality gaps under strong adversarial behavior. Experimental evaluations on MNIST, FMNIST, and CIFAR-10 demonstrate that our method significantly outperforms standard and robust federated learning baselines, such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum, in the majority of adversarial scenarios considered, including label flipping and Gaussian noise injection attacks. All experiments were conducted using the Flower federated learning framework.

摘要: 联合学习（FL）支持跨多个客户端进行协作模型训练，同时通过将本地数据集保留在设备上来保护数据隐私。在这项工作中，我们解决了FL设置，其中客户端可能表现出敌对行为，表现出拜占庭式攻击，而中央服务器是受信任的并配备了参考数据集。我们提出FedGreed，这是一种针对联邦学习的弹性聚合策略，不需要对对抗参与者的比例进行任何假设。FedGreed根据针对服务器上受信任数据集评估的损失指标来订购客户的本地模型更新，并贪婪地选择其模型表现出最小评估损失的客户子集。与许多现有方法不同，我们的方法旨在在现实世界部署中普遍存在的异类（非IID）数据分布下可靠运行。FedGreed在强对抗行为下表现出收敛保证和有界最优性差距。对MNIST、FMNIST和CIFAR-10的实验评估表明，在所考虑的大多数对抗场景中，我们的方法显着优于标准和稳健的联邦学习基线，例如Mean、Trimmed Mean、Median、Krum和Multi-Krum。所有实验都使用Flower联邦学习框架进行。



## **30. Does simple trump complex? Comparing strategies for adversarial robustness in DNNs**

简单胜过复杂吗？比较DNN中对抗鲁棒性的策略 cs.LG

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18019v1) [paper-pdf](http://arxiv.org/pdf/2508.18019v1)

**Authors**: William Brooks, Marelie H. Davel, Coenraad Mouton

**Abstract**: Deep Neural Networks (DNNs) have shown substantial success in various applications but remain vulnerable to adversarial attacks. This study aims to identify and isolate the components of two different adversarial training techniques that contribute most to increased adversarial robustness, particularly through the lens of margins in the input space -- the minimal distance between data points and decision boundaries. Specifically, we compare two methods that maximize margins: a simple approach which modifies the loss function to increase an approximation of the margin, and a more complex state-of-the-art method (Dynamics-Aware Robust Training) which builds upon this approach. Using a VGG-16 model as our base, we systematically isolate and evaluate individual components from these methods to determine their relative impact on adversarial robustness. We assess the effect of each component on the model's performance under various adversarial attacks, including AutoAttack and Projected Gradient Descent (PGD). Our analysis on the CIFAR-10 dataset reveals which elements most effectively enhance adversarial robustness, providing insights for designing more robust DNNs.

摘要: 深度神经网络（DNN）在各种应用中取得了巨大成功，但仍然容易受到对抗攻击。这项研究旨在识别和隔离两种不同对抗训练技术中对提高对抗鲁棒性贡献最大的组成部分，特别是通过输入空间中的裕度（数据点和决策边界之间的最小距离）的视角。具体来说，我们比较了两种最大限度地提高利润率的方法：一种修改损失函数以增加利润率逼近的简单方法，以及一种基于这种方法的更复杂的最先进方法（动态感知稳健训练）。使用VGG-16模型作为基础，我们系统地分离和评估这些方法中的各个成分，以确定它们对对抗稳健性的相对影响。我们评估了每个组件在各种对抗攻击（包括AutoAttack和投影梯度下降（PVD））下对模型性能的影响。我们对CIFAR-10数据集的分析揭示了哪些元素最有效地增强了对抗稳健性，为设计更稳健的DNN提供了见解。



## **31. Efficient Model-Based Purification Against Adversarial Attacks for LiDAR Segmentation**

LiDAR分割中抗对抗性攻击的有效模型净化 cs.CV

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19290v1) [paper-pdf](http://arxiv.org/pdf/2508.19290v1)

**Authors**: Alexandros Gkillas, Ioulia Kapsali, Nikos Piperigkos, Aris S. Lalos

**Abstract**: LiDAR-based segmentation is essential for reliable perception in autonomous vehicles, yet modern segmentation networks are highly susceptible to adversarial attacks that can compromise safety. Most existing defenses are designed for networks operating directly on raw 3D point clouds and rely on large, computationally intensive generative models. However, many state-of-the-art LiDAR segmentation pipelines operate on more efficient 2D range view representations. Despite their widespread adoption, dedicated lightweight adversarial defenses for this domain remain largely unexplored. We introduce an efficient model-based purification framework tailored for adversarial defense in 2D range-view LiDAR segmentation. We propose a direct attack formulation in the range-view domain and develop an explainable purification network based on a mathematical justified optimization problem, achieving strong adversarial resilience with minimal computational overhead. Our method achieves competitive performance on open benchmarks, consistently outperforming generative and adversarial training baselines. More importantly, real-world deployment on a demo vehicle demonstrates the framework's ability to deliver accurate operation in practical autonomous driving scenarios.

摘要: 基于LiDART的分割对于自动驾驶汽车的可靠感知至关重要，但现代分割网络极易受到可能危及安全性的对抗攻击。大多数现有的防御都是为直接在原始3D点云上运行的网络设计的，并依赖于大型计算密集型生成模型。然而，许多最先进的LiDART分割管道都在更有效的2D范围视图表示上运行。尽管它们被广泛采用，但该领域的专用轻量级对抗性防御在很大程度上仍然未被探索。我们引入了一个高效的基于模型的净化框架，专为2D距离视图LiDART分割中的对抗防御而定制。我们在距离视图领域提出了一种直接攻击公式，并基于数学合理的优化问题开发了一个可解释的净化网络，以最小的计算费用实现强大的对抗弹性。我们的方法在开放基准上实现了有竞争力的性能，始终优于生成性和对抗性训练基线。更重要的是，在演示车辆上的现实世界部署证明了该框架在实际自动驾驶场景中提供准确操作的能力。



## **32. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

两阶段学习推迟中的对抗稳健性：算法和保证 stat.ML

Accepted at the 42nd International Conference on Machine Learning  (ICML 2025)

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.01027v4) [paper-pdf](http://arxiv.org/pdf/2502.01027v4)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.

摘要: 两阶段学习延迟（L2 D）通过将每个输入分配给固定的主模型或多个离线专家之一来实现最佳任务委托，支持复杂的多代理环境中的可靠决策。然而，现有的L2 D框架假设干净的输入，并且容易受到可以操纵查询分配的对抗性扰动的影响，从而导致代价高昂的错误路由或专家超载。我们首次对两阶段L2 D系统中的对抗鲁棒性进行了全面研究。我们引入了两种新颖的攻击策略--无针对性和有针对性--它们分别扰乱最佳分配或强制向特定代理进行查询。为了抵御此类威胁，我们提出了SAARD，这是一种凸学习算法，它建立在一系列可证明Bayes-一致且$（\mathCal{R}，\mathCal{G}）$-一致的替代损失之上。这些保证适用于分类、回归和多任务设置。经验结果表明，SAARD显着提高了对抗攻击下的鲁棒性，同时保持了强大的清洁性能，标志着迈向安全且值得信赖的L2 D部署的关键一步。



## **33. A Predictive Framework for Adversarial Energy Depletion in Inbound Threat Scenarios**

远程威胁场景中对抗性能源消耗的预测框架 eess.SY

7 pages, 1 figure, 1 table, preprint submitted to the American  Control Conference (ACC) 2026

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17805v1) [paper-pdf](http://arxiv.org/pdf/2508.17805v1)

**Authors**: Tam W. Nguyen

**Abstract**: This paper presents a predictive framework for adversarial energy-depletion defense against a maneuverable inbound threat (IT). The IT solves a receding-horizon problem to minimize its own energy while reaching a high-value asset (HVA) and avoiding interceptors and static lethal zones modeled by Gaussian barriers. Expendable interceptors (EIs), coordinated by a central node (CN), maintain proximity to the HVA and patrol centers via radius-based tether costs, deny attack corridors by harassing and containing the IT, and commit to intercept only when a geometric feasibility test is confirmed. No explicit opponent-energy term is used, and the formulation is optimization-implementable. No simulations are included.

摘要: 本文提出了一个针对可攻击的入境威胁（IT）的对抗性能量耗尽防御的预测框架。IT解决了后退问题，以最大限度地减少自己的能量，同时获得高价值资产（HVA）并避免拦截器和由高斯屏障建模的静态致命区。由中心节点（CN）协调的消耗性拦截器（EI）通过基于半径的系绳成本保持与HVA和巡逻中心的接近性，通过骚扰和遏制IT来拒绝攻击走廊，并仅在几何可行性测试得到确认时承诺拦截。没有使用明确的余能项，并且该公式是可优化实现的。不包括模拟。



## **34. Robustness Feature Adapter for Efficient Adversarial Training**

用于高效对抗训练的鲁棒性特征适配器 cs.LG

The paper has been accepted for presentation at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17680v1) [paper-pdf](http://arxiv.org/pdf/2508.17680v1)

**Authors**: Quanwei Wu, Jun Guo, Wei Wang, Yi Wang

**Abstract**: Adversarial training (AT) with projected gradient descent is the most popular method to improve model robustness under adversarial attacks. However, computational overheads become prohibitively large when AT is applied to large backbone models. AT is also known to have the issue of robust overfitting. This paper contributes to solving both problems simultaneously towards building more trustworthy foundation models. In particular, we propose a new adapter-based approach for efficient AT directly in the feature space. We show that the proposed adapter-based approach can improve the inner-loop convergence quality by eliminating robust overfitting. As a result, it significantly increases computational efficiency and improves model accuracy by generalizing adversarial robustness to unseen attacks. We demonstrate the effectiveness of the new adapter-based approach in different backbone architectures and in AT at scale.

摘要: 采用投影梯度下降的对抗训练（AT）是提高模型在对抗攻击下鲁棒性的最常用方法。然而，当AT应用于大型骨干模型时，计算开销变得过大。AT还已知具有鲁棒过拟合的问题。本文有助于同时解决这两个问题，建立更值得信赖的基础模型。特别是，我们提出了一个新的适配器为基础的方法，直接在特征空间中的高效AT。我们表明，所提出的基于自适应的方法可以通过消除鲁棒过拟合来提高内环收敛质量。因此，它通过将对抗性鲁棒性推广到不可见的攻击来显着提高计算效率并提高模型准确性。我们证明了新的适配器为基础的方法在不同的骨干架构和AT的规模的有效性。



## **35. Prompt-in-Content Attacks: Exploiting Uploaded Inputs to Hijack LLM Behavior**

内容预算攻击：利用非法输入劫持LLM行为 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19287v1) [paper-pdf](http://arxiv.org/pdf/2508.19287v1)

**Authors**: Zhuotao Lian, Weiyu Wang, Qingkui Zeng, Toru Nakanishi, Teruaki Kitasuka, Chunhua Su

**Abstract**: Large Language Models (LLMs) are widely deployed in applications that accept user-submitted content, such as uploaded documents or pasted text, for tasks like summarization and question answering. In this paper, we identify a new class of attacks, prompt in content injection, where adversarial instructions are embedded in seemingly benign inputs. When processed by the LLM, these hidden prompts can manipulate outputs without user awareness or system compromise, leading to biased summaries, fabricated claims, or misleading suggestions. We demonstrate the feasibility of such attacks across popular platforms, analyze their root causes including prompt concatenation and insufficient input isolation, and discuss mitigation strategies. Our findings reveal a subtle yet practical threat in real-world LLM workflows.

摘要: 大型语言模型（LLM）广泛部署在接受用户提交的内容（例如上传的文档或粘贴的文本）的应用程序中，以执行总结和问答等任务。在本文中，我们识别了一类新的攻击，在内容注入中提示，其中对抗性指令被嵌入到看似良性的输入中。当LLM处理时，这些隐藏的提示可能会在用户意识不到或系统损害的情况下操纵输出，从而导致有偏见的摘要、捏造的主张或误导性的建议。我们展示了此类攻击在流行平台上的可行性，分析了其根本原因，包括迅速级联和输入隔离不足，并讨论了缓解策略。我们的研究结果揭示了现实世界LLM工作流程中一个微妙但实际的威胁。



## **36. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **37. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

TombRaider：进入历史宝库越狱大型语言模型 cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.

摘要: 警告：本文包含可能涉及潜在有害行为的内容，严格出于研究目的进行讨论。   越狱攻击可能会阻碍大型语言模型（LLM）应用程序的安全性，尤其是聊天机器人。研究越狱技术是提高这些应用安全性的一项重要人工智能红色团队任务。本文中，我们介绍了TombRaider，这是一种新型越狱技术，它利用了存储、检索和使用LLM历史知识的能力。TombRaider使用两个代理，检查员代理提取相关历史信息，攻击者代理生成对抗提示，从而有效绕过安全过滤器。我们对TombRaider的六款热门型号进行了深入评估。实验结果表明，TombRaider的性能优于最先进的越狱技术，在裸模型上实现了近100%的攻击成功率（ASB），并在防御机制下保持超过55.4%的ASB。我们的调查结果强调了现有LLM保障措施中的关键漏洞，强调了更强大的安全防御的必要性。



## **38. Defending against Jailbreak through Early Exit Generation of Large Language Models**

通过早期退出生成大型语言模型抵御越狱 cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，随着一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了降低此类风险，“对齐”技术的概念被开发出来。然而，最近的研究表明，使用复杂的即时工程或对抗性后缀（一种被称为“越狱”的技术）可能会破坏这种对齐。“我们的研究从LLM的类人类生成过程中汲取线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示的嵌入。利用这一发现，我们建议利用LLM的早期Transformer输出作为检测恶意输入并立即终止生成的手段。我们为LLM引入了一种简单但重要的防御方法，称为EEG-Defender。我们对三种模型的十种越狱方法进行了全面实验。我们的结果表明，EEG-Defender能够大幅降低攻击成功率（ASB），大约为85%，而当前SOTA的攻击成功率为50%，对LLM的实用性影响最小。



## **39. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

SoK：类人生物生态系统的网络安全评估 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17481v1) [paper-pdf](http://arxiv.org/pdf/2508.17481v1)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.

摘要: 类人猿正在向医疗保健、工业、国防和服务业的实际部署迈进。虽然通常被认为是网络物理系统（CPS），但它们对传统网络软件栈的依赖（例如，Linux操作系统）、机器人操作系统（LOS）中间件和空中更新通道创建了一个独特的安全配置文件，使它们暴露在传统CPS模型无法完全解决的漏洞中。之前的研究主要检查了特定的威胁，例如LiDART欺骗或对抗性机器学习（ML）。这种狭隘的焦点忽视了针对一个组件的攻击如何对整个机器人的互连系统造成连锁伤害。我们通过知识系统化（SoK）来解决这一差距，该知识采用全面的方法，整合机器人、CPS和网络安全领域的碎片化研究。我们为人形机器人引入了一个七层安全模型，组织了整个人形生态系统中的39种已知攻击和35种防御--从硬件到人机交互。在此安全模型的基础上，我们开发了一个具有风险加权评分的量化39 x35攻击-防御矩阵，并通过蒙特卡洛分析进行验证。我们通过评估三个现实世界的机器人：Pepper、G1 EDU和Digit来演示我们的方法。评分分析显示，各个平台的安全成熟度水平各不相同，评分范围从39.9%到79.5%不等。这项工作引入了一种结构化的、基于证据的评估方法，可以实现系统性的安全评估，支持跨平台基准测试，并指导人形机器人安全投资的优先顺序。



## **40. FRAME : Comprehensive Risk Assessment Framework for Adversarial Machine Learning Threats**

FRAME：对抗性机器学习威胁的全面风险评估框架 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17405v1) [paper-pdf](http://arxiv.org/pdf/2508.17405v1)

**Authors**: Avishag Shapira, Simon Shigol, Asaf Shabtai

**Abstract**: The widespread adoption of machine learning (ML) systems increased attention to their security and emergence of adversarial machine learning (AML) techniques that exploit fundamental vulnerabilities in ML systems, creating an urgent need for comprehensive risk assessment for ML-based systems. While traditional risk assessment frameworks evaluate conventional cybersecurity risks, they lack ability to address unique challenges posed by AML threats. Existing AML threat evaluation approaches focus primarily on technical attack robustness, overlooking crucial real-world factors like deployment environments, system dependencies, and attack feasibility. Attempts at comprehensive AML risk assessment have been limited to domain-specific solutions, preventing application across diverse systems. Addressing these limitations, we present FRAME, the first comprehensive and automated framework for assessing AML risks across diverse ML-based systems. FRAME includes a novel risk assessment method that quantifies AML risks by systematically evaluating three key dimensions: target system's deployment environment, characteristics of diverse AML techniques, and empirical insights from prior research. FRAME incorporates a feasibility scoring mechanism and LLM-based customization for system-specific assessments. Additionally, we developed a comprehensive structured dataset of AML attacks enabling context-aware risk assessment. From an engineering application perspective, FRAME delivers actionable results designed for direct use by system owners with only technical knowledge of their systems, without expertise in AML. We validated it across six diverse real-world applications. Our evaluation demonstrated exceptional accuracy and strong alignment with analysis by AML experts. FRAME enables organizations to prioritize AML risks, supporting secure AI deployment in real-world environments.

摘要: 机器学习（ML）系统的广泛采用增加了人们对其安全性的关注，以及利用ML系统基本漏洞的对抗性机器学习（ML）技术的出现，迫切需要对基于ML的系统进行全面风险评估。虽然传统的风险评估框架评估传统的网络安全风险，但它们缺乏应对反洗钱威胁带来的独特挑战的能力。现有的反洗钱威胁评估方法主要关注技术攻击的稳健性，而忽略了部署环境、系统依赖性和攻击可行性等关键现实世界因素。全面的反洗钱风险评估的尝试仅限于特定领域的解决方案，从而阻止了跨不同系统的应用。为了解决这些限制，我们提出了FRAME，这是第一个用于评估各种基于ML的系统中的反洗钱风险的全面自动化框架。FRAME包括一种新颖的风险评估方法，通过系统性评估三个关键维度来量化反洗钱风险：目标系统的部署环境、各种反洗钱技术的特征以及来自先前研究的经验见解。FRAME结合了可行性评分机制和基于LLM的定制，用于特定于系统的评估。此外，我们还开发了一个全面的结构化的反洗钱攻击数据集，从而实现了上下文感知的风险评估。从工程应用程序的角度来看，FRAME提供可操作的结果，供仅具有系统技术知识而不具备反洗钱专业知识的系统所有者直接使用。我们在六个不同的现实世界应用程序中验证了它。我们的评估证明了异常准确性，并且与反洗钱专家的分析高度一致。FRAME使组织能够优先考虑反洗钱风险，支持在现实世界环境中安全的人工智能部署。



## **41. Bridging Models to Defend: A Population-Based Strategy for Robust Adversarial Defense**

桥梁模型以防御：基于人口的稳健对抗防御策略 cs.AI

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2303.10225v2) [paper-pdf](http://arxiv.org/pdf/2303.10225v2)

**Authors**: Ren Wang, Yuxuan Li, Can Chen, Dakuo Wang, Jinjun Xiong, Pin-Yu Chen, Sijia Liu, Mohammad Shahidehpour, Alfred Hero

**Abstract**: Adversarial robustness is a critical measure of a neural network's ability to withstand adversarial attacks at inference time. While robust training techniques have improved defenses against individual $\ell_p$-norm attacks (e.g., $\ell_2$ or $\ell_\infty$), models remain vulnerable to diversified $\ell_p$ perturbations. To address this challenge, we propose a novel Robust Mode Connectivity (RMC)-oriented adversarial defense framework comprising two population-based learning phases. In Phase I, RMC searches the parameter space between two pre-trained models to construct a continuous path containing models with high robustness against multiple $\ell_p$ attacks. To improve efficiency, we introduce a Self-Robust Mode Connectivity (SRMC) module that accelerates endpoint generation in RMC. Building on RMC, Phase II presents RMC-based optimization, where RMC modules are composed to further enhance diversified robustness. To increase Phase II efficiency, we propose Efficient Robust Mode Connectivity (ERMC), which leverages $\ell_1$- and $\ell_\infty$-adversarially trained models to achieve robustness across a broad range of $p$-norms. An ensemble strategy is employed to further boost ERMC's performance. Extensive experiments across diverse datasets and architectures demonstrate that our methods significantly improve robustness against $\ell_\infty$, $\ell_2$, $\ell_1$, and hybrid attacks. Code is available at https://github.com/wangren09/MCGR.

摘要: 对抗鲁棒性是神经网络在推理时抵御对抗攻击能力的关键指标。虽然强大的训练技术改进了对个体$\ell_p$-norm攻击的防御（例如，$\ell_2 $或$\ell_\infty$），模型仍然容易受到多元化$\ell_p$扰动的影响。为了应对这一挑战，我们提出了一种新型的面向鲁棒模式连接性（RMC）的对抗防御框架，该框架包括两个基于群体的学习阶段。在第一阶段，RMC搜索两个预先训练好的模型之间的参数空间，以构建一个包含对多种$\ell_p$攻击具有高鲁棒性的模型的连续路径。为了提高效率，我们引入了一个自鲁棒模式连接（SRMC）模块，加快端点生成RMC。在RMC的基础上，第二阶段提出了基于RMC的优化，其中RMC模块的组成，以进一步提高多样化的鲁棒性。为了提高第二阶段的效率，我们提出了高效鲁棒模式连接（ERMC），它利用$\ell_1 $-和$\ell_\infty$-逆向训练模型来实现广泛的$p$-范数的鲁棒性。采用整体策略进一步提高ERMC的性能。跨不同数据集和架构的广泛实验表明，我们的方法显着提高了针对$\ell_\infty$、$\ell_2 $、$\ell_1 $和混合攻击的稳健性。代码可在https://github.com/wangren09/MCGR上获取。



## **42. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅有效，而且可以跨模型（GPT-4 o、Claude 3.5、Gemini 2.0）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **43. Risk Assessment and Security Analysis of Large Language Models**

大型语言模型的风险评估与安全性分析 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.

摘要: 由于大型语言模型（LLM）在高风险应用中暴露出系统性的安全挑战，包括隐私泄露，偏见放大和恶意滥用，因此迫切需要一个涵盖其整个生命周期的动态风险评估和协作防御框架。本文重点研究了大型语言模型在关键应用场景中的安全问题，如用户数据泄露的可能性、有害指令的故意输入、模型偏差等。为了解决这些问题，我们描述了一个系统的设计，动态风险评估和分级防御系统，允许不同级别的保护合作。本文提出了一种能够同时评估静态和动态指标的风险评估系统。它使用熵加权来计算基本数据，例如敏感词的频率、API调用是否典型、实时风险熵值是否重要以及上下文偏离程度。实验结果表明，该系统能够识别角色逃避等隐藏攻击，并能够进行快速风险评估。该论文在输入层使用了一种名为BERT-RF（来自Transformers的双向编码器表示）的混合模型来识别和过滤恶意命令。模型层结合使用动态对抗训练和差异隐私噪音注入技术。输出层还具有一个可以跟踪内容来源的神经水印系统。在实践中，这种方法的质量对于金融行业的客户服务尤其重要。



## **44. AdaGAT: Adaptive Guidance Adversarial Training for the Robustness of Deep Neural Networks**

AdaGAT：深度神经网络鲁棒性的自适应引导对抗训练 cs.CV

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17265v1) [paper-pdf](http://arxiv.org/pdf/2508.17265v1)

**Authors**: Zhenyu Liu, Huizhi Liang, Xinrun Li, Vaclav Snasel, Varun Ojha

**Abstract**: Adversarial distillation (AD) is a knowledge distillation technique that facilitates the transfer of robustness from teacher deep neural network (DNN) models to lightweight target (student) DNN models, enabling the target models to perform better than only training the student model independently. Some previous works focus on using a small, learnable teacher (guide) model to improve the robustness of a student model. Since a learnable guide model starts learning from scratch, maintaining its optimal state for effective knowledge transfer during co-training is challenging. Therefore, we propose a novel Adaptive Guidance Adversarial Training (AdaGAT) method. Our method, AdaGAT, dynamically adjusts the training state of the guide model to install robustness to the target model. Specifically, we develop two separate loss functions as part of the AdaGAT method, allowing the guide model to participate more actively in backpropagation to achieve its optimal state. We evaluated our approach via extensive experiments on three datasets: CIFAR-10, CIFAR-100, and TinyImageNet, using the WideResNet-34-10 model as the target model. Our observations reveal that appropriately adjusting the guide model within a certain accuracy range enhances the target model's robustness across various adversarial attacks compared to a variety of baseline models.

摘要: 对抗蒸馏（AD）是一种知识蒸馏技术，可促进鲁棒性从教师深度神经网络（DNN）模型转移到轻量级目标（学生）DNN模型，使目标模型比仅独立训练学生模型表现得更好。之前的一些作品专注于使用小型、可学习的教师（指导）模型来提高学生模型的稳健性。由于可学习的指导模型从零开始学习，因此在联合培训期间保持其有效知识转移的最佳状态具有挑战性。因此，我们提出了一种新型的自适应引导对抗训练（AdaGAT）方法。我们的方法AdaGAT动态调整引导模型的训练状态，以为目标模型提供鲁棒性。具体来说，我们开发了两个独立的损失函数，作为AdaGAT方法的一部分，允许引导模型更积极地参与反向传播以实现其最佳状态。我们使用WideResNet-34-10模型作为目标模型，通过对三个数据集（CIFAR-10、CIFAR-100和TinyImageNet）进行广泛实验来评估我们的方法。我们的观察表明，与各种基线模型相比，在一定的准确性范围内适当调整引导模型可以增强目标模型在各种对抗攻击中的鲁棒性。



## **45. Uncovering and Mitigating Destructive Multi-Embedding Attacks in Deepfake Proactive Forensics**

发现和缓解Deepfake主动取证中的破坏性多重嵌入攻击 cs.CV

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17247v1) [paper-pdf](http://arxiv.org/pdf/2508.17247v1)

**Authors**: Lixin Jia, Haiyang Sun, Zhiqing Guo, Yunfeng Diao, Dan Ma, Gaobo Yang

**Abstract**: With the rapid evolution of deepfake technologies and the wide dissemination of digital media, personal privacy is facing increasingly serious security threats. Deepfake proactive forensics, which involves embedding imperceptible watermarks to enable reliable source tracking, serves as a crucial defense against these threats. Although existing methods show strong forensic ability, they rely on an idealized assumption of single watermark embedding, which proves impractical in real-world scenarios. In this paper, we formally define and demonstrate the existence of Multi-Embedding Attacks (MEA) for the first time. When a previously protected image undergoes additional rounds of watermark embedding, the original forensic watermark can be destroyed or removed, rendering the entire proactive forensic mechanism ineffective. To address this vulnerability, we propose a general training paradigm named Adversarial Interference Simulation (AIS). Rather than modifying the network architecture, AIS explicitly simulates MEA scenarios during fine-tuning and introduces a resilience-driven loss function to enforce the learning of sparse and stable watermark representations. Our method enables the model to maintain the ability to extract the original watermark correctly even after a second embedding. Extensive experiments demonstrate that our plug-and-play AIS training paradigm significantly enhances the robustness of various existing methods against MEA.

摘要: 随着Deepfake技术的快速发展和数字媒体的广泛传播，个人隐私面临着日益严重的安全威胁。Deepfake主动取证涉及嵌入不可感知的水印以实现可靠的源跟踪，是抵御这些威胁的重要防御措施。尽管现有的方法显示出很强的取证能力，但它们依赖于单个水印嵌入的理想化假设，这在现实世界的场景中被证明是不切实际的。本文首次正式定义并证明了多重嵌入攻击（MEA）的存在性。当之前保护的图像经历额外几轮水印嵌入时，原始的取证水印可能会被破坏或删除，从而导致整个主动取证机制无效。为了解决这个漏洞，我们提出了一种名为对抗干扰模拟（AIS）的通用训练范式。AIS没有修改网络架构，而是在微调期间显式地模拟了多边环境协议（MTA）场景，并引入了顺从驱动的损失函数来强制学习稀疏和稳定的水印表示。我们的方法使模型即使在第二次嵌入之后也能够保持正确提取原始水印的能力。大量实验表明，我们的即插即用的AIS训练范式显着增强了各种现有方法针对MTA的鲁棒性。



## **46. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

如何让医疗人工智能系统更安全？模拟多模式医疗RAG系统中的漏洞和威胁 cs.LG

Sumbitted to 2025 AAAI main track

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17215v1) [paper-pdf](http://arxiv.org/pdf/2508.17215v1)

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.

摘要: 通过检索增强生成（RAG）增强的大型视觉语言模型（LVLM）越来越多地用于医疗人工智能，以通过外部临床图像文本检索增强事实基础。然而，这种依赖造成了重大的攻击面。我们提出了MedThreatRAG，这是一种新型的多模式中毒框架，通过注入对抗性图像-文本对来系统性地探索医疗RAG系统中的漏洞。我们方法的一个关键创新是构建模拟的半开放攻击环境，模仿现实世界的医疗系统，允许通过用户或管道贡献定期更新知识库。在此背景下，我们引入并强调跨模式冲突注入（CCGI），它嵌入了医学图像及其配对报告之间的微妙语义矛盾。这些不匹配通过扰乱跨模式对齐而降低检索和生成，同时保持足够合理以逃避传统过滤器。虽然为了完整性而包含了基本的文本和视觉攻击，但CMCI表现出了最严重的降级。对IU-X射线和MIIC-CXR QA任务的评估表明，MedThreatRAG将答案F1评分降低高达27.66%，并将LLaBA-Med-1.5 F1评分降低至低至51.36%。我们的研究结果揭示了临床RAG系统中的根本安全漏洞，并强调了对威胁感知设计和强大的多模式一致性检查的迫切需求。最后，我们提出了一套简洁的指南，为未来多模式医疗RAG系统的安全开发提供信息。



## **47. Adversarial Illusions in Multi-Modal Embeddings**

多模式嵌入中的对抗幻象 cs.CR

In USENIX Security'24

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2308.11804v5) [paper-pdf](http://arxiv.org/pdf/2308.11804v5)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, thermal images, sounds, and videos into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary can align any image or sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future tasks, as well as modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.

摘要: 多模式嵌入将文本、图像、热图像、声音和视频编码到单个嵌入空间中，将不同模式的表示对齐（例如，将狗的图像与吠叫的声音联系起来）。在本文中，我们表明多模式嵌入可能容易受到我们称之为“对抗错觉”的攻击。“给定图像或声音，对手可以扰乱它，使其嵌入接近另一种模式中任意的、对手选择的输入。   这些攻击是跨模式和有针对性的：对手可以将任何图像或声音与他选择的任何目标对齐。对抗幻象利用嵌入空间中的邻近性，因此对下游任务和模式不可知，从而能够对当前和未来任务以及对手无法使用的模式进行大规模妥协。使用Image Bind和AudioCLIP嵌入，我们演示了在不了解特定下游任务的情况下生成的反向对齐输入如何误导图像生成、文本生成、零镜头分类和音频检索。   我们研究错觉在不同嵌入中的可移植性，并开发我们方法的黑匣子版本，我们用它来演示对亚马逊商业专有泰坦嵌入的第一次对抗性对齐攻击。最后，我们分析了应对措施和规避攻击。



## **48. Sharpness-Aware Geometric Defense for Robust Out-Of-Distribution Detection**

用于鲁棒性分布外检测的敏锐度感知几何防御 cs.LG

under review

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17174v1) [paper-pdf](http://arxiv.org/pdf/2508.17174v1)

**Authors**: Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen

**Abstract**: Out-of-distribution (OOD) detection ensures safe and reliable model deployment. Contemporary OOD algorithms using geometry projection can detect OOD or adversarial samples from clean in-distribution (ID) samples. However, this setting regards adversarial ID samples as OOD, leading to incorrect OOD predictions. Existing efforts on OOD detection with ID and OOD data under attacks are minimal. In this paper, we develop a robust OOD detection method that distinguishes adversarial ID samples from OOD ones. The sharp loss landscape created by adversarial training hinders model convergence, impacting the latent embedding quality for OOD score calculation. Therefore, we introduce a {\bf Sharpness-aware Geometric Defense (SaGD)} framework to smooth out the rugged adversarial loss landscape in the projected latent geometry. Enhanced geometric embedding convergence enables accurate ID data characterization, benefiting OOD detection against adversarial attacks. We use Jitter-based perturbation in adversarial training to extend the defense ability against unseen attacks. Our SaGD framework significantly improves FPR and AUC over the state-of-the-art defense approaches in differentiating CIFAR-100 from six other OOD datasets under various attacks. We further examine the effects of perturbations at various adversarial training levels, revealing the relationship between the sharp loss landscape and adversarial OOD detection.

摘要: 分发外（OOD）检测确保安全可靠的模型部署。使用几何投影的当代OOD算法可以从干净的内分布（ID）样本中检测OOD或对抗样本。然而，这种设置将对抗ID样本视为OOD，从而导致OOD预测错误。目前在攻击下使用ID和OOD数据进行OOD检测的工作很少。本文中，我们开发了一种强大的OOD检测方法，可以区分对抗性ID样本和OOD样本。对抗训练造成的急剧损失格局阻碍了模型收敛，影响了OOD分数计算的潜在嵌入质量。因此，我们引入了一个{\BF敏锐性几何防御（SaVD）}框架，以平滑投影潜在几何中崎岖的对抗损失景观。增强的几何嵌入融合可以实现准确的ID数据特征，有利于OOD检测对抗攻击。我们在对抗训练中使用基于抖动的扰动来扩展针对不可见攻击的防御能力。与最先进的防御方法相比，我们的SaVD框架显着提高了FPR和AUC，将CIFAR-100与各种攻击下的其他六个OOD数据集区分开来。我们进一步研究了不同对抗训练水平下扰动的影响，揭示了急剧损失景观和对抗性OOD检测之间的关系。



## **49. Towards Safeguarding LLM Fine-tuning APIs against Cipher Attacks**

保护LLM微调API免受密码攻击 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17158v1) [paper-pdf](http://arxiv.org/pdf/2508.17158v1)

**Authors**: Jack Youstra, Mohammed Mahfoud, Yang Yan, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: Large language model fine-tuning APIs enable widespread model customization, yet pose significant safety risks. Recent work shows that adversaries can exploit access to these APIs to bypass model safety mechanisms by encoding harmful content in seemingly harmless fine-tuning data, evading both human monitoring and standard content filters. We formalize the fine-tuning API defense problem, and introduce the Cipher Fine-tuning Robustness benchmark (CIFR), a benchmark for evaluating defense strategies' ability to retain model safety in the face of cipher-enabled attackers while achieving the desired level of fine-tuning functionality. We include diverse cipher encodings and families, with some kept exclusively in the test set to evaluate for generalization across unseen ciphers and cipher families. We then evaluate different defenses on the benchmark and train probe monitors on model internal activations from multiple fine-tunes. We show that probe monitors achieve over 99% detection accuracy, generalize to unseen cipher variants and families, and compare favorably to state-of-the-art monitoring approaches. We open-source CIFR and the code to reproduce our experiments to facilitate further research in this critical area. Code and data are available online https://github.com/JackYoustra/safe-finetuning-api

摘要: 大型语言模型微调API可以实现广泛的模型定制，但也会带来重大的安全风险。最近的工作表明，对手可以利用对这些API的访问来绕过模型安全机制，将有害内容编码在看似无害的微调数据中，从而逃避人类监控和标准内容过滤器。我们形式化了微调API防御问题，并引入了Cipher微调稳健性基准（CIFR），这是一个评估防御策略在面对启用密码的攻击者时保持模型安全性的能力的基准，同时实现了所需的微调功能水平。我们包括不同的密码编码和系列，其中一些仅保留在测试集中，以评估未见密码和密码系列的通用性。然后，我们评估基准上的不同防御，并根据多个微调的模型内部激活训练探测器监视器。我们表明，探针监测器实现了超过99%的检测准确率，可推广到未见的密码变体和家族，并且与最先进的监测方法相比具有优势。我们开源CIFR和复制我们实验的代码，以促进这一关键领域的进一步研究。代码和数据可在线获取https://github.com/JackYoustra/safe-finetuning-api



## **50. POT: Inducing Overthinking in LLMs via Black-Box Iterative Optimization**

POT：通过黑匣子迭代优化在LLM中引发过度思考 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.19277v1) [paper-pdf](http://arxiv.org/pdf/2508.19277v1)

**Authors**: Xinyu Li, Tianjin Huang, Ronghui Mu, Xiaowei Huang, Gaojie Jin

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially enhanced the reasoning capabilities of large language models (LLMs), enabling sophisticated problem-solving through explicit multi-step reasoning traces. However, these enhanced reasoning processes introduce novel attack surfaces, particularly vulnerabilities to computational inefficiency through unnecessarily verbose reasoning chains that consume excessive resources without corresponding performance gains. Prior overthinking attacks typically require restrictive conditions including access to external knowledge sources for data poisoning, reliance on retrievable poisoned content, and structurally obvious templates that limit practical applicability in real-world scenarios. To address these limitations, we propose POT (Prompt-Only OverThinking), a novel black-box attack framework that employs LLM-based iterative optimization to generate covert and semantically natural adversarial prompts, eliminating dependence on external data access and model retrieval. Extensive experiments across diverse model architectures and datasets demonstrate that POT achieves superior performance compared to other methods.

摘要: 思想链（CoT）提示的最新进展大大增强了大型语言模型（LLM）的推理能力，通过显式的多步骤推理痕迹实现复杂的问题解决。然而，这些增强的推理过程引入了新颖的攻击表面，特别是由于不必要的冗长推理链而导致计算效率低下的脆弱性，这些推理链消耗了过多的资源而没有相应的性能提升。之前的过度思考攻击通常需要限制性条件，包括访问外部知识源进行数据中毒、依赖可检索的中毒内容以及限制现实世界场景中实际适用性的结构明显模板。为了解决这些限制，我们提出了POT（仅预算过度思考），这是一种新型黑匣子攻击框架，它采用基于LLM的迭代优化来生成隐蔽且语义自然的对抗提示，消除了对外部数据访问和模型检索的依赖。跨不同模型架构和数据集的广泛实验表明，与其他方法相比，POT实现了更卓越的性能。



