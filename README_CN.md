# Latest Adversarial Attack Papers
**update at 2025-04-14 11:11:16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

HCP安全审计：具有模型上下文协议的LLM允许重大安全漏洞 cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner

摘要: 为了减少开发费用并实现构成任何给定生成式人工智能应用程序的潜在组件之间的无缝集成，模型上下文协议（HCP）（Anthropic，2024）最近发布并随后广泛采用。HCP是一种开放协议，可同步化对大型语言模型（LLM）、数据源和代理工具的API调用。通过连接多个HCP服务器（每个服务器都定义了一组工具、资源和提示），用户能够定义完全由LLM驱动的自动化工作流程。然而，我们表明当前的LCP设计对最终用户来说存在广泛的安全风险。特别是，我们证明了行业领先的LLM可能会被迫使用LCP工具通过各种攻击（例如恶意代码执行、远程访问控制和凭证盗窃）来危害人工智能开发人员的系统。为了主动缓解这些攻击和相关攻击，我们引入了安全审计工具MCPSafetyScanner，这是第一个评估任意LCP服务器安全性的代理工具。MCPScanner使用多个代理来（a）在给定HCP服务器的工具和资源的情况下自动确定对抗样本;（b）根据这些样本搜索相关漏洞和补救措施;以及（c）生成详细说明所有发现结果的安全报告。我们的工作强调了通用代理工作流程的严重安全问题，同时还提供了一种主动工具来审计LCP服务器的安全性并在部署之前解决检测到的漏洞。   所描述的LCP服务器审计工具MCPSafetyScanner可在以下网址免费获取：https://github.com/johnhalloran321/mcpSafetyScanner



## **2. Enabling Safety for Aerial Robots: Planning and Control Architectures**

实现空中机器人的安全：规划和控制架构 cs.RO

2025 ICRA Workshop on 25 years of Aerial Robotics: Challenges and  Opportunities

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08601v1) [paper-pdf](http://arxiv.org/pdf/2504.08601v1)

**Authors**: Kaleb Ben Naveed, Devansh R. Agrawal, Daniel M. Cherenson, Haejoon Lee, Alia Gilbert, Hardik Parwana, Vishnu S. Chipade, William Bentz, Dimitra Panagou

**Abstract**: Ensuring safe autonomy is crucial for deploying aerial robots in real-world applications. However, safety is a multifaceted challenge that must be addressed from multiple perspectives, including navigation in dynamic environments, operation under resource constraints, and robustness against adversarial attacks and uncertainties. In this paper, we present the authors' recent work that tackles some of these challenges and highlights key aspects that must be considered to enhance the safety and performance of autonomous aerial systems. All presented approaches are validated through hardware experiments.

摘要: 确保安全自主性对于在现实世界应用中部署空中机器人至关重要。然而，安全是一个多方面的挑战，必须从多个角度解决，包括动态环境中的导航、资源限制下的操作以及对抗对抗攻击和不确定性的鲁棒性。在本文中，我们介绍了作者最近针对其中一些挑战的工作，并强调了增强自主航空系统安全性和性能必须考虑的关键方面。所有提出的方法都通过硬件实验进行了验证。



## **3. An Early Experience with Confidential Computing Architecture for On-Device Model Protection**

用于设备上模型保护的机密计算架构的早期经验 cs.CR

Accepted to the 8th Workshop on System Software for Trusted Execution  (SysTEX 2025)

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08508v1) [paper-pdf](http://arxiv.org/pdf/2504.08508v1)

**Authors**: Sina Abdollahi, Mohammad Maheri, Sandra Siby, Marios Kogias, Hamed Haddadi

**Abstract**: Deploying machine learning (ML) models on user devices can improve privacy (by keeping data local) and reduce inference latency. Trusted Execution Environments (TEEs) are a practical solution for protecting proprietary models, yet existing TEE solutions have architectural constraints that hinder on-device model deployment. Arm Confidential Computing Architecture (CCA), a new Arm extension, addresses several of these limitations and shows promise as a secure platform for on-device ML. In this paper, we evaluate the performance-privacy trade-offs of deploying models within CCA, highlighting its potential to enable confidential and efficient ML applications. Our evaluations show that CCA can achieve an overhead of, at most, 22% in running models of different sizes and applications, including image classification, voice recognition, and chat assistants. This performance overhead comes with privacy benefits; for example, our framework can successfully protect the model against membership inference attack by an 8.3% reduction in the adversary's success rate. To support further research and early adoption, we make our code and methodology publicly available.

摘要: 在用户设备上部署机器学习（ML）模型可以改善隐私（通过将数据保留在本地）并减少推理延迟。可信执行环境（TEK）是保护专有模型的实用解决方案，但现有的TEK解决方案具有阻碍设备上模型部署的架构限制。Arm机密计算架构（CAA）是一个新的Arm扩展，解决了其中的几个限制，并显示出作为设备上ML安全平台的前景。在本文中，我们评估了在PCA中部署模型的性能与隐私权衡，强调了其实现保密和高效ML应用程序的潜力。我们的评估表明，在运行不同大小和应用程序（包括图像分类、语音识别和聊天助手）的模型时，CAA最多可以实现22%的额外费用。这种性能负担带来了隐私好处;例如，我们的框架可以通过将对手的成功率降低8.3%来成功保护模型免受成员资格推断攻击。为了支持进一步的研究和早期采用，我们公开了我们的代码和方法。



## **4. Nanopass Back-Translation of Call-Return Trees for Mechanized Secure Compilation Proofs**

用于机械化安全编译证明的调用返回树的Nanopass反向翻译 cs.PL

ITP'25 submission, updated with link to Rocq development

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2503.19609v2) [paper-pdf](http://arxiv.org/pdf/2503.19609v2)

**Authors**: Jérémy Thibault, Joseph Lenormand, Catalin Hritcu

**Abstract**: Researchers aim to build secure compilation chains enforcing that if there is no attack a source context can mount against a source program then there is also no attack an adversarial target context can mount against the compiled program. Proving that these compilation chains are secure is, however, challenging, and involves a non-trivial back-translation step: for any attack a target context mounts against the compiled program one has to exhibit a source context mounting the same attack against the source program. We describe a novel back-translation technique, which results in simpler proofs that can be more easily mechanized in a proof assistant. Given a finite set of finite trace prefixes, capturing the interaction recorded during an attack between a target context and the compiled program, we build a call-return tree that we back-translate into a source context producing the same trace prefixes. We use state in the generated source context to record the current location in the call-return tree. The back-translation is done in several small steps, each adding to the tree new information describing how the location should change depending on how the context regains control. To prove this back-translation correct we give semantics to every intermediate call-return tree language, using ghost state to store information and explicitly enforce execution invariants. We prove several small forward simulations, basically seeing the back-translation as a verified nanopass compiler. Thanks to this modular structure, we are able to mechanize this complex back-translation and its correctness proof in the Rocq prover without too much effort.

摘要: 研究人员的目标是构建安全的编译链，确保如果源上下文不存在针对源程序的攻击，那么对抗性目标上下文也不存在针对已编译的程序的攻击。然而，证明这些编译链是安全的是具有挑战性的，并且涉及一个不平凡的反向翻译步骤：对于目标上下文针对已编译的程序发起的任何攻击，必须展示针对源程序发起的相同攻击的源上下文。我们描述了一种新颖的反向翻译技术，它可以产生更简单的证明，并且可以更容易地在证明助手中机械化。给定一组有限的有限跟踪前置，捕获攻击期间目标上下文和已编译的程序之间记录的交互，我们构建一个调用返回树，将其反向翻译为源上下文，从而产生相同的跟踪前置。我们使用生成的源上下文中的状态来记录调用返回树中的当前位置。反向翻译是分几个小步骤完成的，每个步骤都会向树中添加新信息，描述位置应该如何根据上下文如何重新获得控制而更改。为了证明这种反向翻译是正确的，我们为每个中间调用返回树语言赋予语义，使用幽灵状态来存储信息并显式地强制执行执行不变量。我们证明了几个小型的正向模拟，基本上将反向翻译视为经过验证的纳米ass编译器。得益于这种模块化结构，我们能够在Rocq证明器中机械化这种复杂的反向翻译及其正确性证明，而无需付出太多的努力。



## **5. Toward Realistic Adversarial Attacks in IDS: A Novel Feasibility Metric for Transferability**

走向IDS中的现实对抗攻击：一种新型的可移植性可行性指标 cs.CR

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08480v1) [paper-pdf](http://arxiv.org/pdf/2504.08480v1)

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi Vincenzo Mancini

**Abstract**: Transferability-based adversarial attacks exploit the ability of adversarial examples, crafted to deceive a specific source Intrusion Detection System (IDS) model, to also mislead a target IDS model without requiring access to the training data or any internal model parameters. These attacks exploit common vulnerabilities in machine learning models to bypass security measures and compromise systems. Although the transferability concept has been widely studied, its practical feasibility remains limited due to assumptions of high similarity between source and target models. This paper analyzes the core factors that contribute to transferability, including feature alignment, model architectural similarity, and overlap in the data distributions that each IDS examines. We propose a novel metric, the Transferability Feasibility Score (TFS), to assess the feasibility and reliability of such attacks based on these factors. Through experimental evidence, we demonstrate that TFS and actual attack success rates are highly correlated, addressing the gap between theoretical understanding and real-world impact. Our findings provide needed guidance for designing more realistic transferable adversarial attacks, developing robust defenses, and ultimately improving the security of machine learning-based IDS in critical systems.

摘要: 基于可移植性的对抗性攻击利用了对抗性示例的能力，这些示例旨在欺骗特定源入侵检测系统（IDS）模型，在不需要访问训练数据或任何内部模型参数的情况下也误导目标IDS模型。这些攻击利用机器学习模型中的常见漏洞来绕过安全措施并危及系统。尽管可移植性概念已得到广泛研究，但由于源模型和目标模型之间高度相似性的假设，其实际可行性仍然有限。本文分析了影响可移植性的核心因素，包括特征对齐、模型架构相似性以及每个IDS检查的数据分布中的重叠。我们提出了一种新的度量，可转移性可行性分数（TFS），评估这些因素的基础上，这种攻击的可行性和可靠性。通过实验证据，我们证明了TFS和实际攻击成功率高度相关，解决了理论理解和现实影响之间的差距。我们的研究结果为设计更现实的可转移对抗性攻击，开发强大的防御，并最终提高关键系统中基于机器学习的IDS的安全性提供了必要的指导。



## **6. TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors**

TACO：卡车对抗性伪装优化以愚弄物体检测器 cs.CV

This version matches the final published version in Big Data and  Cognitive Computing (MDPI). Please cite the journal version when referencing  this work (doi: https://doi.org/10.3390/bdcc9030072)

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2410.21443v2) [paper-pdf](http://arxiv.org/pdf/2410.21443v2)

**Authors**: Adonisz Dimitriu, Tamás Michaletzky, Viktor Remeli

**Abstract**: Adversarial attacks threaten the reliability of machine learning models in critical applications like autonomous vehicles and defense systems. As object detectors become more robust with models like YOLOv8, developing effective adversarial methodologies is increasingly challenging. We present Truck Adversarial Camouflage Optimization (TACO), a novel framework that generates adversarial camouflage patterns on 3D vehicle models to deceive state-of-the-art object detectors. Adopting Unreal Engine 5, TACO integrates differentiable rendering with a Photorealistic Rendering Network to optimize adversarial textures targeted at YOLOv8. To ensure the generated textures are both effective in deceiving detectors and visually plausible, we introduce the Convolutional Smooth Loss function, a generalized smooth loss function. Experimental evaluations demonstrate that TACO significantly degrades YOLOv8's detection performance, achieving an AP@0.5 of 0.0099 on unseen test data. Furthermore, these adversarial patterns exhibit strong transferability to other object detection models such as Faster R-CNN and earlier YOLO versions.

摘要: 对抗性攻击威胁着自动驾驶汽车和防御系统等关键应用中机器学习模型的可靠性。随着物体检测器在YOLOv 8等模型中变得更加强大，开发有效的对抗方法变得越来越具有挑战性。我们提出了卡车对抗伪装优化（TACO），这是一种新颖的框架，可以在3D车辆模型上生成对抗伪装图案，以欺骗最先进的物体检测器。TACO采用虚幻引擎5，将差异渲染与真实感渲染网络集成，以优化针对YOLOv 8的对抗性纹理。为了确保生成的纹理在欺骗检测器和视觉上都是有效的，我们引入了卷积平滑损失函数，一个广义的平滑损失函数。实验评估表明，TACO显著降低了YOLOv 8的检测性能，在看不见的测试数据上实现了0.0099的AP@0.5。此外，这些对抗模式表现出很强的可移植性，可以移植到其他对象检测模型，如Faster R-CNN和早期的YOLO版本。



## **7. Adversarial Examples in Environment Perception for Automated Driving (Review)**

自动驾驶环境感知中的对抗示例（评论） cs.CV

One chapter of upcoming Springer book: Recent Advances in Autonomous  Vehicle Technology, 2025

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08414v1) [paper-pdf](http://arxiv.org/pdf/2504.08414v1)

**Authors**: Jun Yan, Huilin Yin

**Abstract**: The renaissance of deep learning has led to the massive development of automated driving. However, deep neural networks are vulnerable to adversarial examples. The perturbations of adversarial examples are imperceptible to human eyes but can lead to the false predictions of neural networks. It poses a huge risk to artificial intelligence (AI) applications for automated driving. This survey systematically reviews the development of adversarial robustness research over the past decade, including the attack and defense methods and their applications in automated driving. The growth of automated driving pushes forward the realization of trustworthy AI applications. This review lists significant references in the research history of adversarial examples.

摘要: 深度学习的复兴带动了自动驾驶的大规模发展。然而，深度神经网络很容易受到对抗性例子的影响。对抗示例的扰动人眼无法察觉，但可能导致神经网络的错误预测。它对自动驾驶的人工智能（AI）应用构成了巨大风险。本文系统回顾了过去十年对抗鲁棒性研究的发展，包括攻击和防御方法及其在自动驾驶中的应用。自动驾驶的发展推动了值得信赖的AI应用的实现。这篇综述列出了对抗性例子研究历史上的重要文献。



## **8. Statistical Linear Regression Approach to Kalman Filtering and Smoothing under Cyber-Attacks**

网络攻击下Kalman滤波与平滑的统计线性回归方法 eess.SP

5 pages, 4 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08404v1) [paper-pdf](http://arxiv.org/pdf/2504.08404v1)

**Authors**: Kundan Kumar, Muhammad Iqbal, Simo Särkkä

**Abstract**: Remote state estimation in cyber-physical systems is often vulnerable to cyber-attacks due to wireless connections between sensors and computing units. In such scenarios, adversaries compromise the system by injecting false data or blocking measurement transmissions via denial-of-service attacks, distorting sensor readings. This paper develops a Kalman filter and Rauch--Tung--Striebel (RTS) smoother for linear stochastic state-space models subject to cyber-attacked measurements. We approximate the faulty measurement model via generalized statistical linear regression (GSLR). The GSLR-based approximated measurement model is then used to develop a Kalman filter and RTS smoother for the problem. The effectiveness of the proposed algorithms under cyber-attacks is demonstrated through a simulated aircraft tracking experiment.

摘要: 由于传感器和计算单元之间的无线连接，网络物理系统中的远程状态估计通常容易受到网络攻击。在这种情况下，对手通过拒绝服务攻击注入虚假数据或阻止测量传输、扭曲传感器读数来损害系统。本文为遭受网络攻击测量的线性随机状态空间模型开发了一种卡尔曼过滤器和Rauch--Tung-Striebel（RTS）平滑器。我们通过广义统计线性回归（GSLR）来逼近错误的测量模型。然后使用基于GSLR的近似测量模型来开发针对该问题的卡尔曼过滤器和RTS平滑器。通过模拟飞机跟踪实验验证了所提出算法在网络攻击下的有效性。



## **9. To See or Not to See -- Fingerprinting Devices in Adversarial Environments Amid Advanced Machine Learning**

看还是不看--先进机器学习中对抗环境中的指纹设备 cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08264v1) [paper-pdf](http://arxiv.org/pdf/2504.08264v1)

**Authors**: Justin Feng, Nader Sehatbakhsh

**Abstract**: The increasing use of the Internet of Things raises security concerns. To address this, device fingerprinting is often employed to authenticate devices, detect adversaries, and identify eavesdroppers in an environment. This requires the ability to discern between legitimate and malicious devices which is achieved by analyzing the unique physical and/or operational characteristics of IoT devices. In the era of the latest progress in machine learning, particularly generative models, it is crucial to methodically examine the current studies in device fingerprinting. This involves explaining their approaches and underscoring their limitations when faced with adversaries armed with these ML tools. To systematically analyze existing methods, we propose a generic, yet simplified, model for device fingerprinting. Additionally, we thoroughly investigate existing methods to authenticate devices and detect eavesdropping, using our proposed model. We further study trends and similarities between works in authentication and eavesdropping detection and present the existing threats and attacks in these domains. Finally, we discuss future directions in fingerprinting based on these trends to develop more secure IoT fingerprinting schemes.

摘要: 物联网的使用越来越多引发了安全担忧。为了解决这个问题，通常使用设备指纹识别来验证设备、检测对手和识别环境中的窃听者。这需要能够区分合法设备和恶意设备，这是通过分析物联网设备的独特物理和/或操作特征来实现的。在机器学习（特别是生成模型）取得最新进展的时代，系统地检查当前设备指纹识别的研究至关重要。这涉及解释他们的方法并强调他们在面对配备这些ML工具的对手时的局限性。为了系统地分析现有方法，我们提出了一种通用但简化的设备指纹识别模型。此外，我们还使用我们提出的模型彻底研究了现有的验证设备和检测窃听的方法。我们进一步研究身份验证和窃听检测领域工作之间的趋势和相似之处，并呈现这些领域中现有的威胁和攻击。最后，我们根据这些趋势讨论了指纹识别的未来方向，以开发更安全的物联网指纹识别方案。



## **10. EO-VLM: VLM-Guided Energy Overload Attacks on Vision Models**

EO-VLM：对视觉模型的VLM引导能量过载攻击 cs.CV

Presented as a poster at ACSAC 2024

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08205v1) [paper-pdf](http://arxiv.org/pdf/2504.08205v1)

**Authors**: Minjae Seo, Myoungsung You, Junhee Lee, Jaehan Kim, Hwanjo Heo, Jintae Oh, Jinwoo Kim

**Abstract**: Vision models are increasingly deployed in critical applications such as autonomous driving and CCTV monitoring, yet they remain susceptible to resource-consuming attacks. In this paper, we introduce a novel energy-overloading attack that leverages vision language model (VLM) prompts to generate adversarial images targeting vision models. These images, though imperceptible to the human eye, significantly increase GPU energy consumption across various vision models, threatening the availability of these systems. Our framework, EO-VLM (Energy Overload via VLM), is model-agnostic, meaning it is not limited by the architecture or type of the target vision model. By exploiting the lack of safety filters in VLMs like DALL-E 3, we create adversarial noise images without requiring prior knowledge or internal structure of the target vision models. Our experiments demonstrate up to a 50% increase in energy consumption, revealing a critical vulnerability in current vision models.

摘要: 视觉模型越来越多地部署在自动驾驶和闭路电视监控等关键应用中，但它们仍然容易受到消耗资源的攻击。在本文中，我们引入了一种新型的能量超载攻击，该攻击利用视觉语言模型（VLM）提示来生成针对视觉模型的对抗图像。这些图像虽然人眼无法感知，但却显着增加了各种视觉模型的图形处理器能耗，威胁到这些系统的可用性。我们的框架EO-VLM（通过VLM实现能源过载）是模型不可知的，这意味着它不受目标视觉模型的架构或类型的限制。通过利用DALL-E 3等VLM中缺乏安全过滤器的情况，我们创建对抗性噪音图像，而不需要目标视觉模型的先验知识或内部结构。我们的实验表明，能源消耗增加了高达50%，揭示了当前视觉模型中的一个关键漏洞。



## **11. Adversarial Attacks on Data Attribution**

对数据归因的对抗性攻击 cs.LG

Accepted at the 13th International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2409.05657v3) [paper-pdf](http://arxiv.org/pdf/2409.05657v3)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities and proposing principled adversarial attack methods on data attribution. We present two methods, Shadow Attack and Outlier Attack, which generate manipulated datasets to inflate the compensation adversarially. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%. Our implementation is ready at https://github.com/TRAIS-Lab/adversarial-attack-data-attribution.

摘要: 数据归因旨在量化单个训练数据点对人工智能模型输出的贡献，人工智能模型已用于衡量训练数据的价值并补偿数据提供商。鉴于对财务决策和薪酬机制的影响，出现了一个关于数据归因方法的对抗稳健性的关键问题。然而，几乎没有针对这个问题的系统性研究。在这项工作中，我们的目标是通过详细描述威胁模型来弥合这一差距，其中对对手的目标和能力做出明确假设，并提出关于数据属性的有原则的对抗攻击方法。我们提出了两种方法，影子攻击和离群点攻击，它们生成操纵数据集以不利地夸大补偿。影子攻击利用有关人工智能应用程序中数据分布的知识，并通过“影子训练”（一种常用于成员资格推理攻击的技术）来推导对抗性扰动。相比之下，离群点攻击不假设有关数据分布的任何知识，而是仅依赖于对目标模型预测的黑匣子查询。它利用了许多数据归因方法中存在的归纳偏差--离群数据点更有可能具有影响力--并利用对抗性示例来生成操纵的数据集。从经验上看，在图像分类和文本生成任务中，影子攻击可以将基于数据属性的补偿膨胀至少200%，而异常值攻击则实现了从185%到高达643%的补偿膨胀。我们的实施已在https://github.com/TRAIS-Lab/adversarial-attack-data-attribution上准备好。



## **12. Adversarial Attacks on AI-Generated Text Detection Models: A Token Probability-Based Approach Using Embeddings**

对人工智能生成文本检测模型的对抗攻击：使用嵌入的基于令牌概率的方法 cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2501.18998v2) [paper-pdf](http://arxiv.org/pdf/2501.18998v2)

**Authors**: Ahmed K. Kadhim, Lei Jiao, Rishad Shafik, Ole-Christoffer Granmo

**Abstract**: In recent years, text generation tools utilizing Artificial Intelligence (AI) have occasionally been misused across various domains, such as generating student reports or creative writings. This issue prompts plagiarism detection services to enhance their capabilities in identifying AI-generated content. Adversarial attacks are often used to test the robustness of AI-text generated detectors. This work proposes a novel textual adversarial attack on the detection models such as Fast-DetectGPT. The method employs embedding models for data perturbation, aiming at reconstructing the AI generated texts to reduce the likelihood of detection of the true origin of the texts. Specifically, we employ different embedding techniques, including the Tsetlin Machine (TM), an interpretable approach in machine learning for this purpose. By combining synonyms and embedding similarity vectors, we demonstrates the state-of-the-art reduction in detection scores against Fast-DetectGPT. Particularly, in the XSum dataset, the detection score decreased from 0.4431 to 0.2744 AUROC, and in the SQuAD dataset, it dropped from 0.5068 to 0.3532 AUROC.

摘要: 近年来，利用人工智能（AI）的文本生成工具偶尔会在各个领域被滥用，例如生成学生报告或创意作品。这个问题促使剽窃检测服务增强其识别AI生成内容的能力。对抗性攻击通常用于测试AI文本生成检测器的鲁棒性。这项工作提出了一种新的文本对抗攻击的检测模型，如快速检测GPT。该方法采用嵌入模型进行数据扰动，旨在重建人工智能生成的文本，以降低检测文本真实起源的可能性。具体来说，我们采用了不同的嵌入技术，包括Tsetlin Machine（TM），这是一种用于此目的的机器学习中的可解释方法。通过结合同义词和嵌入相似性载体，我们展示了针对Fast-DetectGPT的最新检测分数降低。特别是，在XSum数据集中，检测评分从0.4431下降到0.2744 AUROC，在SQuAD数据集中，检测评分从0.5068下降到0.3532 AUROC。



## **13. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

大型语言模型中对偏见激发的对抗鲁棒性进行基准测试：利用LLM作为评委的可扩展自动化评估 cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07887v1) [paper-pdf](http://arxiv.org/pdf/2504.07887v1)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.

摘要: 大型语言模型（LLM）彻底改变了人工智能，推动了机器翻译、摘要和对话代理的进步。然而，他们越来越融入关键社会领域，引发了人们对根深蒂固的偏见的担忧，这可能会延续刻板印象并损害公平性。这些偏见源于各种来源，包括训练数据的历史不平等、语言不平衡和对抗操纵。尽管做出了缓解措施，但最近的研究表明，LLM仍然容易受到旨在引发偏见反应的对抗攻击。这项工作提出了一个可扩展的基准测试框架，以评估LLM针对对抗性偏见引发的稳健性。我们的方法包括：（i）采用针对各个社会文化维度的偏见的多任务方法系统地探索模型，（ii）使用LLM作为法官的方法通过安全评分量化稳健性，以自动评估模型响应，以及（iii）采用越狱技术来调查安全机制中的漏洞。我们的分析检查了小型和大型最先进模型中普遍存在的偏差及其对模型安全性的影响。此外，我们还评估针对医学等关键领域进行微调的特定领域模型的安全性。最后，我们发布了一个精心策划的偏差相关提示数据集ClearAR-Bias，以促进系统性漏洞基准测试。我们的研究结果揭示了模型大小和安全性之间的关键权衡，有助于开发更公平、更强大的未来语言模型。



## **14. QubitHammer Attacks: Qubit Flipping Attacks in Multi-tenant Superconducting Quantum Computers**

QubitHammer攻击：多租户超导量子计算机中的Qubit翻转攻击 quant-ph

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07875v1) [paper-pdf](http://arxiv.org/pdf/2504.07875v1)

**Authors**: Yizhuo Tan, Navnil Choudhury, Kanad Basu, Jakub Szefer

**Abstract**: Quantum computing is rapidly evolving its capabilities, with a corresponding surge in its deployment within cloud-based environments. Various quantum computers are accessible today via pay-as-you-go cloud computing models, offering unprecedented convenience. Due to its rapidly growing demand, quantum computers are shifting from a single-tenant to a multi-tenant model to enhance resource utilization. However, this widespread accessibility to shared multi-tenant systems also introduces potential security vulnerabilities. In this work, we present for the first time a set of novel attacks, named together as the QubitHammer attacks, which target state-of-the-art superconducting quantum computers. We show that in a multi-tenant cloud-based quantum system, an adversary with the basic capability to deploy custom pulses, similar to any standard user today, can utilize the QubitHammer attacks to significantly degrade the fidelity of victim circuits located on the same quantum computer. Upon extensive evaluation, the QubitHammer attacks achieve a very high variational distance of up to 0.938 from the expected outcome, thus demonstrating their potential to degrade victim computation. Our findings exhibit the effectiveness of these attacks across various superconducting quantum computers from a leading vendor, suggesting that QubitHammer represents a new class of security attacks. Further, the attacks are demonstrated to bypass all existing defenses proposed so far for ensuring the reliability in multi-tenant superconducting quantum computers.

摘要: 量子计算的能力正在迅速发展，其在云环境中的部署也相应激增。如今，各种量子计算机都可以通过现收现付云计算模型访问，提供了前所未有的便利。由于需求迅速增长，量子计算机正在从单租户模式转向多租户模式，以提高资源利用率。然而，这种对共享多租户系统的广泛访问性也带来了潜在的安全漏洞。在这项工作中，我们首次提出了一组新颖的攻击，统称为QubitHammer攻击，其目标是最先进的超导量子计算机。我们表明，在基于云的多租户量子系统中，具有部署自定义脉冲基本能力的对手（类似于当今的任何标准用户）可以利用QubitHammer攻击来显着降低位于同一量子计算机上的受害者电路的保真度。经过广泛评估，QubitHammer攻击与预期结果之间的变化距离非常高，高达0.938，从而证明了它们有可能降低受害者计算能力。我们的研究结果展示了这些攻击对领先供应商的各种超导量子计算机的有效性，表明QubitHammer代表了一类新型安全攻击。此外，这些攻击被证明可以绕过迄今为止为确保多租户超导量子计算机的可靠性而提出的所有现有防御措施。



## **15. MUFFLER: Secure Tor Traffic Obfuscation with Dynamic Connection Shuffling and Splitting**

MUFFLER：通过动态连接洗牌和拆分实现安全Tor流量混淆 cs.CR

To appear in IEEE INFOCOM 2025

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07543v1) [paper-pdf](http://arxiv.org/pdf/2504.07543v1)

**Authors**: Minjae Seo, Myoungsung You, Jaehan Kim, Taejune Park, Seungwon Shin, Jinwoo Kim

**Abstract**: Tor, a widely utilized privacy network, enables anonymous communication but is vulnerable to flow correlation attacks that deanonymize users by correlating traffic patterns from Tor's ingress and egress segments. Various defenses have been developed to mitigate these attacks; however, they have two critical limitations: (i) significant network overhead during obfuscation and (ii) a lack of dynamic obfuscation for egress segments, exposing traffic patterns to adversaries. In response, we introduce MUFFLER, a novel connection-level traffic obfuscation system designed to secure Tor egress traffic. It dynamically maps real connections to a distinct set of virtual connections between the final Tor nodes and targeted services, either public or hidden. This approach creates egress traffic patterns fundamentally different from those at ingress segments without adding intentional padding bytes or timing delays. The mapping of real and virtual connections is adjusted in real-time based on ongoing network conditions, thwarting adversaries' efforts to detect egress traffic patterns. Extensive evaluations show that MUFFLER mitigates powerful correlation attacks with a TPR of 1% at an FPR of 10^-2 while imposing only a 2.17% bandwidth overhead. Moreover, it achieves up to 27x lower latency overhead than existing solutions and seamlessly integrates with the current Tor architecture.

摘要: Tor是一种广泛使用的隐私网络，可以实现匿名通信，但很容易受到流量相关攻击，这些攻击通过关联Tor入口和出口段的流量模式来使用户去匿名化。人们开发了各种防御措施来缓解这些攻击;然而，它们有两个关键局限性：（i）混淆期间的网络负担很大;（ii）出口段缺乏动态混淆，从而将流量模式暴露给对手。作为回应，我们引入MUFFLER，这是一种新型的连接级流量混淆系统，旨在保护Tor出口流量。它将真实连接动态映射到最终Tor节点和目标服务（公共或隐藏）之间的一组不同的虚拟连接。这种方法创建的出口流量模式与入口段的流量模式根本不同，而无需故意添加填充字节或时间延迟。真实和虚拟连接的映射会根据当前的网络状况实时调整，从而阻碍对手检测出口流量模式的努力。广泛的评估表明，MUFFLER在FPR为10 ' 2时，以1%的TPA缓解了强大的相关性攻击，同时仅施加2.17%的带宽负担。此外，它的延迟负担比现有解决方案低27倍，并与当前的Tor架构无缝集成。



## **16. The Gradient Puppeteer: Adversarial Domination in Gradient Leakage Attacks through Model Poisoning**

梯度木偶师：通过模型中毒在梯度泄漏攻击中的对抗统治 cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2502.04106v2) [paper-pdf](http://arxiv.org/pdf/2502.04106v2)

**Authors**: Kunlan Xiang, Haomiao Yang, Meng Hao, Shaofeng Li, Haoxin Wang, Zikang Ding, Wenbo Jiang, Tianwei Zhang

**Abstract**: In Federated Learning (FL), clients share gradients with a central server while keeping their data local. However, malicious servers could deliberately manipulate the models to reconstruct clients' data from shared gradients, posing significant privacy risks. Although such active gradient leakage attacks (AGLAs) have been widely studied, they suffer from two severe limitations: (i) coverage: no existing AGLAs can reconstruct all samples in a batch from the shared gradients; (ii) stealthiness: no existing AGLAs can evade principled checks of clients. In this paper, we address these limitations with two core contributions. First, we introduce a new theoretical analysis approach, which uniformly models AGLAs as backdoor poisoning. This analysis approach reveals that the core principle of AGLAs is to bias the gradient space to prioritize the reconstruction of a small subset of samples while sacrificing the majority, which theoretically explains the above limitations of existing AGLAs. Second, we propose Enhanced Gradient Global Vulnerability (EGGV), the first AGLA that achieves complete attack coverage while evading client-side detection. In particular, EGGV employs a gradient projector and a jointly optimized discriminator to assess gradient vulnerability, steering the gradient space toward the point most prone to data leakage. Extensive experiments show that EGGV achieves complete attack coverage and surpasses state-of-the-art (SOTA) with at least a 43% increase in reconstruction quality (PSNR) and a 45% improvement in stealthiness (D-SNR).

摘要: 在联合学习（FL）中，客户端与中央服务器共享梯度，同时将其数据保留在本地。然而，恶意服务器可能会故意操纵模型，从共享梯度重建客户端数据，从而构成重大的隐私风险。尽管此类主动梯度泄漏攻击（AGLA）已被广泛研究，但它们存在两个严重的局限性：（i）覆盖范围：没有现有的AGLA可以从共享梯度重建批次中的所有样本;（ii）隐蔽性：没有现有的AGLA可以逃避对客户的原则性检查。在本文中，我们通过两个核心贡献来解决这些限制。首先，我们引入一种新的理论分析方法，该方法将AGLA统一建模为后门中毒。这种分析方法揭示了AGLA的核心原则是偏向梯度空间，以优先重建小样本子集，同时牺牲大部分样本，这从理论上解释了现有AGLA的上述局限性。其次，我们提出了增强型梯度全局漏洞（EGGV），这是第一个在逃避客户端检测的同时实现完全攻击覆盖的AGLA。特别是，EGGV采用梯度投影仪和联合优化的NPS来评估梯度脆弱性，将梯度空间引导到最容易发生数据泄露的点。大量实验表明，EGGV实现了完全的攻击覆盖，并超越了最新技术水平（SOTA），重建质量（PSNR）至少提高了43%，隐蔽性（D-SNR）至少提高了45%。



## **17. Code Generation with Small Language Models: A Deep Evaluation on Codeforces**

使用小语言模型的代码生成：对代码力量的深入评估 cs.SE

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.07343v1) [paper-pdf](http://arxiv.org/pdf/2504.07343v1)

**Authors**: Débora Souza, Rohit Gheyi, Lucas Albuquerque, Gustavo Soares, Márcio Ribeiro

**Abstract**: Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws.

摘要: 大型语言模型（LLM）已经展示了代码生成的能力，这可能会提高开发人员的生产力。然而，它们的广泛采用仍然受到高计算成本、高能源需求以及数据泄露和对抗性攻击等安全风险的限制。作为一种轻量级的替代方案，小型语言模型（SLC）提供更快的推理、更低的部署负担以及对特定领域任务的更好的适应性，使其成为现实世界应用程序的有吸引力的选择。虽然之前的研究已经根据竞争性编程任务对LLM进行了基准测试，但此类评估通常狭隘地关注Elo分数或通过率等指标，而忽视了对模型行为、失败模式和问题多样性的更深入见解。此外，Slms解决竞争性编程等复杂任务的潜力仍然没有得到充分的开发。在这项研究中，我们对五个开放的LM进行了基准测试- LLAMA 3.2 3B、GEGMA 2 9 B、GEGMA 3 12 B、DEEPSEEK-R1 14 B和PHI-4 14 B-跨越280个Codeforce问题，涵盖Elo评级从800到2100，涵盖36个不同的主题。所有模型的任务都是生成Python解决方案。PHI-4 14 B实现了SLS中最好的性能，通过率@3为63.6%，接近专有的O3-MINI-HIGH（86.8%）。此外，我们在C++上评估了PHI-4 14 B，发现结合Python和C++的输出可以将其总通过率@3增加到73.6%。对PHI-4 14 B错误输出的定性分析显示，一些失败是由于小的实现问题（例如处理边缘情况或纠正变量初始化）而不是更深层次的推理缺陷。



## **18. EveGuard: Defeating Vibration-based Side-Channel Eavesdropping with Audio Adversarial Perturbations**

EveGuard：通过音频对抗性扰动击败基于振动的侧通道发射器丢弃 cs.CR

In the 46th IEEE Symposium on Security and Privacy (IEEE S&P), May  2025

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2411.10034v2) [paper-pdf](http://arxiv.org/pdf/2411.10034v2)

**Authors**: Jung-Woo Chang, Ke Sun, David Xia, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Vibrometry-based side channels pose a significant privacy risk, exploiting sensors like mmWave radars, light sensors, and accelerometers to detect vibrations from sound sources or proximate objects, enabling speech eavesdropping. Despite various proposed defenses, these involve costly hardware solutions with inherent physical limitations. This paper presents EveGuard, a software-driven defense framework that creates adversarial audio, protecting voice privacy from side channels without compromising human perception. We leverage the distinct sensing capabilities of side channels and traditional microphones, where side channels capture vibrations and microphones record changes in air pressure, resulting in different frequency responses. EveGuard first proposes a perturbation generator model (PGM) that effectively suppresses sensor-based eavesdropping while maintaining high audio quality. Second, to enable end-to-end training of PGM, we introduce a new domain translation task called Eve-GAN for inferring an eavesdropped signal from a given audio. We further apply few-shot learning to mitigate the data collection overhead for Eve-GAN training. Our extensive experiments show that EveGuard achieves a protection rate of more than 97 percent from audio classifiers and significantly hinders eavesdropped audio reconstruction. We further validate the performance of EveGuard across three adaptive attack mechanisms. We have conducted a user study to verify the perceptual quality of our perturbed audio.

摘要: 基于振动测量的侧通道构成了巨大的隐私风险，利用毫米波雷达、光传感器和加速度计等传感器来检测来自声音源或附近物体的振动，从而实现语音窃听。尽管提出了各种防御措施，但这些措施涉及昂贵的硬件解决方案，具有固有的物理限制。本文介绍了EveGuard，这是一个软件驱动的防御框架，可以创建对抗性音频，在不损害人类感知的情况下保护侧通道的语音隐私。我们利用侧通道和传统麦克风的独特传感能力，侧通道捕获振动，麦克风记录气压的变化，从而产生不同的频率响应。EveGuard首先提出了一种扰动生成器模型（CGM），可以有效抑制基于传感器的窃听，同时保持高音频质量。其次，为了实现CGM的端到端训练，我们引入了一个名为Eve-GAN的新域翻译任务，用于从给定音频中推断被窃听的信号。我们进一步应用少量学习来减轻Eve-GAN训练的数据收集负担。我们广泛的实验表明，EveGuard对音频分类器的保护率超过97%，并显着阻碍了被窃听的音频重建。我们进一步验证了EveGuard在三种自适应攻击机制中的性能。我们进行了一项用户研究来验证受干扰音频的感知质量。



## **19. LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks**

LLM保障是一把双刃剑：利用假阳性进行拒绝服务攻击 cs.CR

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2410.02916v3) [paper-pdf](http://arxiv.org/pdf/2410.02916v3)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.

摘要: 安全性是开放部署中大型语言模型（LLM）的首要问题，这促使开发了通过安全对齐或护栏机制来执行道德和负责任使用的保障方法。利用安全措施的\“假阴性\”的越狱攻击已经成为LLM安全领域的一个突出的研究热点。然而，我们发现恶意攻击者也可以利用安全措施的误报，即，欺骗保护模型错误地阻止安全内容，导致影响LLM用户的拒绝服务（DoS）。为了弥合这个被忽视的威胁的知识差距，我们探索了多种攻击方法，包括在用户提示模板中插入简短的对抗提示，以及通过有毒微调损坏服务器上的LLM。通过这两种方式，攻击都会触发对客户端用户请求的安全拒绝。我们的评估表明了这种威胁在多种场景下的严重性。例如，在白盒对抗性提示注入的场景中，攻击者可以使用我们的优化过程自动生成看似安全的对抗性提示，大约只有30个字符长，普遍阻止Llama Guard 3上超过97%的用户请求。这些发现揭示了LLM保障评估的一个新维度--对假阳性的对抗稳健性。



## **20. Towards Communication-Efficient Adversarial Federated Learning for Robust Edge Intelligence**

迈向通信高效的对抗联邦学习以实现稳健的边缘智能 cs.CV

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2501.15257v2) [paper-pdf](http://arxiv.org/pdf/2501.15257v2)

**Authors**: Yu Qiao, Apurba Adhikary, Huy Q. Le, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) has gained significant attention for enabling decentralized training on edge networks without exposing raw data. However, FL models remain susceptible to adversarial attacks and performance degradation in non-IID data settings, thus posing challenges to both robustness and accuracy. This paper aims to achieve communication-efficient adversarial federated learning (AFL) by leveraging a pre-trained model to enhance both robustness and accuracy under adversarial attacks and non-IID challenges in AFL. By leveraging the knowledge from a pre-trained model for both clean and adversarial images, we propose a pre-trained model-guided adversarial federated learning (PM-AFL) framework. This framework integrates vanilla and adversarial mixture knowledge distillation to effectively balance accuracy and robustness while promoting local models to learn from diverse data. Specifically, for clean accuracy, we adopt a dual distillation strategy where the class probabilities of randomly paired images, and their blended versions are aligned between the teacher model and the local models. For adversarial robustness, we employ a similar distillation approach but replace clean samples on the local side with adversarial examples. Moreover, by considering the bias between local and global models, we also incorporate a consistency regularization term to ensure that local adversarial predictions stay aligned with their corresponding global clean ones. These strategies collectively enable local models to absorb diverse knowledge from the teacher model while maintaining close alignment with the global model, thereby mitigating overfitting to local optima and enhancing the generalization of the global model. Experiments demonstrate that the PM-AFL-based framework not only significantly outperforms other methods but also maintains communication efficiency.

摘要: 联合学习（FL）因在不暴露原始数据的情况下在边缘网络上实现去中心化训练而受到了广泛关注。然而，FL模型在非IID数据环境中仍然容易受到对抗攻击和性能下降的影响，从而对稳健性和准确性构成挑战。本文旨在通过利用预先训练的模型来实现通信高效的对抗性联邦学习（AFL），以增强AFL中对抗性攻击和非IID挑战下的稳健性和准确性。通过利用来自干净图像和对抗图像的预训练模型的知识，我们提出了一个预训练模型引导的对抗联邦学习（PM-AFL）框架。该框架集成了香草和对抗混合知识蒸馏，以有效地平衡准确性和鲁棒性，同时促进本地模型从不同的数据中学习。具体来说，为了达到干净的准确性，我们采用了双重蒸馏策略，其中随机配对图像的类概率及其混合版本在教师模型和本地模型之间对齐。对于对抗性鲁棒性，我们采用了类似的蒸馏方法，但用对抗性示例替换了局部的干净样本。此外，通过考虑局部和全局模型之间的偏差，我们还引入了一致性正则化项，以确保局部对抗预测与相应的全局干净预测保持一致。这些策略共同使本地模型能够从教师模型中吸收多样化的知识，同时保持与全球模型的密切一致，从而减轻对本地最优值的过度适应并增强全球模型的概括性。实验表明，基于PM-AFL的框架不仅显着优于其他方法，而且还保持了通信效率。



## **21. Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction**

利用基于元学习的中毒攻击进行图链接预测 cs.LG

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06492v1) [paper-pdf](http://arxiv.org/pdf/2504.06492v1)

**Authors**: Mingchen Li, Di Zhuang, Keyu Chen, Dumindu Samaraweera, Morris Chang

**Abstract**: Link prediction in graph data utilizes various algorithms and machine learning/deep learning models to predict potential relationships between graph nodes. This technique has found widespread use in numerous real-world applications, including recommendation systems, community networks, and biological structures. However, recent research has highlighted the vulnerability of link prediction models to adversarial attacks, such as poisoning and evasion attacks. Addressing the vulnerability of these models is crucial to ensure stable and robust performance in link prediction applications. While many works have focused on enhancing the robustness of the Graph Convolution Network (GCN) model, the Variational Graph Auto-Encoder (VGAE), a sophisticated model for link prediction, has not been thoroughly investigated in the context of graph adversarial attacks. To bridge this gap, this article proposes an unweighted graph poisoning attack approach using meta-learning techniques to undermine VGAE's link prediction performance. We conducted comprehensive experiments on diverse datasets to evaluate the proposed method and its parameters, comparing it with existing approaches in similar settings. Our results demonstrate that our approach significantly diminishes link prediction performance and outperforms other state-of-the-art methods.

摘要: 图数据中的链接预测利用各种算法和机器学习/深度学习模型来预测图节点之间的潜在关系。该技术已广泛应用于许多现实世界的应用程序中，包括推荐系统、社区网络和生物结构。然而，最近的研究强调了链接预测模型对对抗攻击（例如中毒和逃避攻击）的脆弱性。解决这些模型的脆弱性对于确保链接预测应用程序中稳定和稳健的性能至关重要。虽然许多作品都集中在增强图卷积网络（GCN）模型的鲁棒性上，但变分图自动编码器（VGAE）是一种复杂的链接预测模型，尚未在图对抗攻击的背景下进行彻底研究。为了弥合这一差距，本文提出了一种使用元学习技术的无加权图中毒攻击方法来破坏VGAE的链接预测性能。我们对不同的数据集进行了全面的实验，以评估所提出的方法及其参数，并将其与类似环境下的现有方法进行比较。我们的结果表明，我们的方法显着降低了链接预测性能，并且优于其他最先进的方法。



## **22. Mitigating Adversarial Effects of False Data Injection Attacks in Power Grid**

减轻电网中虚假数据注入攻击的对抗性影响 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2301.12487v3) [paper-pdf](http://arxiv.org/pdf/2301.12487v3)

**Authors**: Farhin Farhad Riya, Shahinul Hoque, Yingyuan Yang, Jiangnan Li, Jinyuan Stella Sun, Hairong Qi

**Abstract**: Deep Neural Networks have proven to be highly accurate at a variety of tasks in recent years. The benefits of Deep Neural Networks have also been embraced in power grids to detect False Data Injection Attacks (FDIA) while conducting critical tasks like state estimation. However, the vulnerabilities of DNNs along with the distinct infrastructure of the cyber-physical-system (CPS) can favor the attackers to bypass the detection mechanism. Moreover, the divergent nature of CPS engenders limitations to the conventional defense mechanisms for False Data Injection Attacks. In this paper, we propose a DNN framework with an additional layer that utilizes randomization to mitigate the adversarial effect by padding the inputs. The primary advantage of our method is when deployed to a DNN model it has a trivial impact on the model's performance even with larger padding sizes. We demonstrate the favorable outcome of the framework through simulation using the IEEE 14-bus, 30-bus, 118-bus, and 300-bus systems. Furthermore to justify the framework we select attack techniques that generate subtle adversarial examples that can bypass the detection mechanism effortlessly.

摘要: 近年来，深度神经网络已被证明在各种任务中具有高度准确性。深度神经网络的好处也已被应用于电网中，用于检测错误数据注入攻击（FDIA），同时执行状态估计等关键任务。然而，DNN的漏洞以及网络物理系统（CPS）的独特基础设施可能有利于攻击者绕过检测机制。此外，CPS的不同性质给虚假数据注入攻击的传统防御机制带来了限制。在本文中，我们提出了一个带有额外层的DNN框架，该层利用随机化通过填充输入来减轻对抗效应。我们方法的主要优点是，当部署到DNN模型时，即使填充大小更大，它也对模型的性能产生微小的影响。我们通过使用IEEE 14-节点、30-节点、118-节点和300-节点系统进行模拟，证明了该框架的良好结果。此外，为了证明该框架的合理性，我们选择了可以生成微妙的对抗示例的攻击技术，这些示例可以毫不费力地绕过检测机制。



## **23. Quantum Covert Communication under Extreme Adversarial Control**

极端对抗控制下的量子秘密通信 quant-ph

Submitted to Quantum

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06359v1) [paper-pdf](http://arxiv.org/pdf/2504.06359v1)

**Authors**: Trey Li

**Abstract**: Secure quantum communication traditionally assumes that the adversary controls only the public channel. We consider a more powerful adversary who can demand private information of users. This type of adversary has been studied in public key cryptography in recent years, initiated by Persiano, Phan, and Yung at Eurocrypt 2022. We introduce a similar attacker to quantum communication, referring to it as the controller. The controller is a quantum computer that controls the entire communication infrastructure, including both classical and quantum channels. It can even ban classical public key cryptography and post-quantum public key cryptography, leaving only quantum cryptography and post-quantum symmetric key cryptography as the remaining options. We demonstrate how such a controller can control quantum communication and how users can achieve covert communication under its control.

摘要: 安全量子通信传统上假设对手仅控制公共通道。我们考虑的是一个更强大的对手，可以要求用户的私人信息。近年来，这种类型的对手在公钥加密技术中得到了研究，由Persiano、Phan和Yung在EuroCrypt 2022上发起。我们将类似的攻击者引入量子通信，将其称为控制器。控制器是一台量子计算机，控制整个通信基础设施，包括经典和量子通道。它甚至可以禁止经典公钥加密术和后量子公钥加密术，只留下量子加密术和后量子对称密钥加密术作为剩余的选择。我们演示了这样的控制器如何控制量子通信，以及用户如何在其控制下实现秘密通信。



## **24. Towards Calibration Enhanced Network by Inverse Adversarial Attack**

通过反向对抗攻击实现校准增强网络 cs.CV

11 pages

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06358v1) [paper-pdf](http://arxiv.org/pdf/2504.06358v1)

**Authors**: Yupeng Cheng, Zi Pong Lim, Sarthak Ketanbhai Modi, Yon Shin Teo, Yushi Cao, Shang-Wei Lin

**Abstract**: Test automation has become increasingly important as the complexity of both design and content in Human Machine Interface (HMI) software continues to grow. Current standard practice uses Optical Character Recognition (OCR) techniques to automatically extract textual information from HMI screens for validation. At present, one of the key challenges faced during the automation of HMI screen validation is the noise handling for the OCR models. In this paper, we propose to utilize adversarial training techniques to enhance OCR models in HMI testing scenarios. More specifically, we design a new adversarial attack objective for OCR models to discover the decision boundaries in the context of HMI testing. We then adopt adversarial training to optimize the decision boundaries towards a more robust and accurate OCR model. In addition, we also built an HMI screen dataset based on real-world requirements and applied multiple types of perturbation onto the clean HMI dataset to provide a more complete coverage for the potential scenarios. We conduct experiments to demonstrate how using adversarial training techniques yields more robust OCR models against various kinds of noises, while still maintaining high OCR model accuracy. Further experiments even demonstrate that the adversarial training models exhibit a certain degree of robustness against perturbations from other patterns.

摘要: 随着人机界面（HM）软件设计和内容的复杂性不断增长，测试自动化变得越来越重要。当前的标准实践使用光学字符识别（OCR）技术自动从人机界面屏幕中提取文本信息以进行验证。目前，人机界面屏幕验证自动化过程中面临的关键挑战之一是OCR模型的噪音处理。在本文中，我们建议利用对抗训练技术来增强人机界面测试场景中的OCR模型。更具体地说，我们为OCR模型设计了一个新的对抗攻击目标，以发现人机界面测试背景下的决策边界。然后，我们采用对抗训练来优化决策边界，以建立更稳健、更准确的OCR模型。此外，我们还根据现实世界的要求构建了一个人机界面屏幕数据集，并对干净的人机界面数据集应用了多种类型的扰动，以为潜在场景提供更完整的覆盖。我们进行实验来演示如何使用对抗训练技术来产生针对各种噪音的更稳健的OCR模型，同时仍然保持高的OCR模型准确性。进一步的实验甚至表明，对抗性训练模型对来自其他模式的扰动表现出一定程度的鲁棒性。



## **25. Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots**

探索自主移动机器人基于搜索的路径规划中的对抗障碍攻击 cs.RO

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.06154v1) [paper-pdf](http://arxiv.org/pdf/2504.06154v1)

**Authors**: Adrian Szvoren, Jianwei Liu, Dimitrios Kanoulas, Nilufer Tuptuk

**Abstract**: Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path.   We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.

摘要: 路径规划算法（例如基于搜索的A*）是自主移动机器人技术的关键组成部分，使机器人能够高效、安全地从起点导航到目的地。我们研究了A* 算法在面临潜在的对抗性干预（即障碍攻击）时的弹性。对手的目标是通过在机器人的原始路径上设置障碍物来推迟机器人及时到达目的地。   我们开发了恶意软件来执行攻击，并进行了实验来评估其影响，无论是在Gazebo中使用TurtleBot进行模拟还是在现实世界中使用Unitree Go 1机器人进行部署。在模拟中，攻击导致平均延迟为36%，其中最显着的延迟发生在机器人被迫采取更长的替代路径的情况下。在现实世界的实验中，延迟甚至更加明显，所有攻击都成功地改变了机器人的路线并造成了可测量的中断。这些结果凸显了该算法的鲁棒性不仅仅是其设计的属性，而且还受到操作环境的显着影响。例如，在隧道等受限环境中，由于替代路线的可用性有限，延误被最大化。



## **26. Frequency maps reveal the correlation between Adversarial Attacks and Implicit Bias**

频率图揭示了对抗性攻击和隐性偏见之间的相关性 cs.LG

Accepted at IJCNN 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2305.15203v3) [paper-pdf](http://arxiv.org/pdf/2305.15203v3)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto d'Onofrio, Luca Manzoni, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the correlation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse a representation of the network's implicit bias through the lens of the Fourier transform. Specifically, we identify unique fingerprints of implicit bias and adversarial attacks by calculating the minimal, essential frequencies needed for accurate classification of each image, as well as the frequencies that drive misclassification in its adversarially perturbed counterpart. This approach enables us to uncover and analyse the correlation between these essential frequencies, providing a precise map of how the network's biases align or contrast with the frequency components exploited by adversarial attacks. To this end, among other methods, we use a newly introduced technique capable of detecting nonlinear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.

摘要: 尽管神经网络在分类任务中的表现令人印象深刻，但众所周知，神经网络很容易受到对抗攻击，即旨在欺骗模型的输入数据的微妙扰动。在这项工作中，我们研究了这些扰动与用基于梯度的算法训练的神经网络的隐式偏差之间的相关性。为此，我们通过傅里叶变换的镜头分析网络隐含偏差的表示。具体来说，我们通过计算对每张图像进行准确分类所需的最小、基本频率，以及导致其受对抗干扰的对应图像中误分类的频率，来识别隐性偏见和对抗攻击的独特指纹。这种方法使我们能够发现和分析这些基本频率之间的相关性，提供网络偏差如何与对抗性攻击利用的频率分量对齐或对比的精确地图。为此，除其他方法外，我们使用了一种新引入的能够检测多维数据集之间非线性相关性的技术。我们的结果提供了经验证据，证明傅里叶空间中的网络偏差和对抗性攻击的目标频率高度相关，并提出了对抗性防御的新潜在策略。



## **27. Mind the Trojan Horse: Image Prompt Adapter Enabling Scalable and Deceptive Jailbreaking**

小心特洛伊木马：图像提示适配器支持可扩展和欺骗性越狱 cs.CV

Accepted by CVPR2025 as Highlight

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05838v1) [paper-pdf](http://arxiv.org/pdf/2504.05838v1)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie

**Abstract**: Recently, the Image Prompt Adapter (IP-Adapter) has been increasingly integrated into text-to-image diffusion models (T2I-DMs) to improve controllability. However, in this paper, we reveal that T2I-DMs equipped with the IP-Adapter (T2I-IP-DMs) enable a new jailbreak attack named the hijacking attack. We demonstrate that, by uploading imperceptible image-space adversarial examples (AEs), the adversary can hijack massive benign users to jailbreak an Image Generation Service (IGS) driven by T2I-IP-DMs and mislead the public to discredit the service provider. Worse still, the IP-Adapter's dependency on open-source image encoders reduces the knowledge required to craft AEs. Extensive experiments verify the technical feasibility of the hijacking attack. In light of the revealed threat, we investigate several existing defenses and explore combining the IP-Adapter with adversarially trained models to overcome existing defenses' limitations. Our code is available at https://github.com/fhdnskfbeuv/attackIPA.

摘要: 最近，图像提示适配器（IP适配器）越来越多地集成到文本到图像扩散模型（T2 I-DM）中，以提高可控性。然而，在本文中，我们揭示了配备IP适配器（T2 I-IP-DMs）的T2 I-DM会启用一种名为劫持攻击的新越狱攻击。我们证明，通过上传难以察觉的图像空间对抗示例（AE），对手可以劫持大量良性用户来越狱由T2 I-IP-DM驱动的图像生成服务（IRS），并误导公众抹黑服务提供商。更糟糕的是，IP适配器对开源图像编码器的依赖减少了制作AE所需的知识。大量实验验证了劫持攻击的技术可行性。鉴于所揭示的威胁，我们调查了几种现有的防御措施，并探索将IP适配器与对抗训练模型相结合，以克服现有防御措施的局限性。我们的代码可在https://github.com/fhdnskfbeuv/attackIPA上获取。



## **28. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

StealthRank：通过StealthPropriation优化进行LLM排名操纵 cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.

摘要: 将大型语言模型（LLM）集成到信息检索系统中引入了新的攻击表面，特别是对于对抗性排名操纵。我们介绍了StealthRank，这是一种新型的对抗性排名攻击，它可以操纵LLM驱动的产品推荐系统，同时保持文本流畅性和隐蔽性。与经常引入可检测异常的现有方法不同，StealthRank采用基于能量的优化框架与Langevin动态相结合来生成StealthRank脚本（SPP）-嵌入产品描述中的对抗性文本序列，微妙而有效地影响LLM排名机制。我们在多个LLM中评估StealthRank，证明其能够秘密提高目标产品的排名，同时避免容易检测到的显式操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面始终优于最先进的对抗排名基线，凸显了LLM驱动的推荐系统中的关键漏洞。



## **29. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

用于机器学习文本分类器的自动可信度Oracle生成 cs.SE

Accepted to FSE 2025

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2410.22663v2) [paper-pdf](http://arxiv.org/pdf/2410.22663v2)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.

摘要: 用于文本分类的机器学习（ML）已广泛应用于各个领域。这些应用程序可能会显着影响道德、经济和人类行为，从而引发人们对信任ML决策的严重担忧。研究表明，传统指标不足以建立人类对ML模型的信任。这些模型经常学习虚假相关性并基于它们进行预测。在现实世界中，他们的表现可能会显着恶化。为了避免这种情况，常见的做法是根据数据中的有效模式测试预测是否合理。与此同时，还引入了一个称为可信度Oracle问题的挑战。由于缺乏自动可信度预言，评估需要对解释方法披露的决策过程进行手动验证。然而，这耗时、容易出错且不可扩展。   我们提出了TOKI，这是第一种文本分类器的自动可信Oracle生成方法。TOKI自动检查对预测贡献最大的单词是否与预测的类别在语义上相关。具体来说，我们利用ML解释来提取影响决策的单词，并基于单词嵌入来测量它们与类的语义相关性。我们还引入了一种新颖的对抗攻击方法，该方法针对TOKI识别的可信度漏洞。为了评估它们与人类判断的一致性，我们进行了实验。我们将TOKI与仅基于模型置信度和TOKI引导的对抗攻击方法与A2 T（一种SOTA对抗攻击方法）进行比较。结果表明，依赖预测不确定性无法有效区分可信和不可信的预测，TOKI的准确性比原始基线高出142%，TOKI引导的攻击方法比A2 T更有效，干扰更少。



## **30. Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**

Nes 2Net：基础模型驱动语音反欺骗的轻量级嵌套架构 eess.AS

This manuscript has been submitted for peer review

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05657v1) [paper-pdf](http://arxiv.org/pdf/2504.05657v1)

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.

摘要: 语音基础模型通过提供出色的表示能力，显着推进了各种语音相关任务。然而，它们的多维输出特征通常会与下游任务模型产生不匹配，下游任务模型通常需要较低维度的输入。常见的解决方案是应用降维（DR）层，但这种方法增加了参数负担、计算成本，并存在丢失有价值信息的风险。为了解决这些问题，我们提出了Nested Res 2Net（Nes 2Net），这是一种轻量级的后台架构，旨在直接处理多维特征，而无需DR层。嵌套结构增强了多尺度特征提取，改善了特征交互，并保留了多维信息。我们首先在CtrSVD（歌唱声深度伪造检测数据集）上验证了Nes 2Net，并报告与最先进的基线相比，性能提高了22%，后台计算成本降低了87%。此外，对四个不同数据集进行了广泛的测试：ASVspoof 2021、ASVspoof 5、PartialSpoof和In-the-Wild，涵盖了完全欺骗的语音、对抗性攻击、部分欺骗和现实世界场景，一致强调了Nes 2Net卓越的鲁棒性和概括能力。代码包和预训练模型可在https://github.com/Liu-Tianchi/Nes2Net上获取。



## **31. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.

摘要: 大型语言模型（LLM）已经成为越来越广泛的应用程序的组成部分。然而，它们仍然是越狱攻击的威胁，攻击者操纵设计的提示，使模型引发恶意输出。分析越狱方法可以帮助我们深入研究LLM的弱点并对其进行改进。本文通过分析模型的输出对输入和后续输出对先前输出的注意力权重，揭示了大型语言模型（LLM）中的一个漏洞，我们称之为防御阈值衰减（DTD）：当模型生成大量良性内容时，其注意力权重从输入转移到先前输出，使其更容易受到越狱攻击。为了证明DTD的可利用性，我们提出了一种新的越狱攻击方法，糖衣毒药（SCP），它诱导模型通过良性输入和对抗性推理生成大量良性内容，随后产生恶意内容。为了减轻这种攻击，我们引入了一种简单而有效的防御策略POSD，它可以显着降低越狱成功率，同时保留模型的泛化能力。



## **32. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

SceneRAP：针对现实世界环境中视觉语言模型的场景一致印刷对抗规划器 cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.

摘要: 大型视觉语言模型（LVLM）在解释视觉内容方面表现出了非凡的能力。虽然现有的作品证明了这些模型对故意放置的对抗文本的脆弱性，但此类文本通常很容易被识别为异常文本。在本文中，我们提出了第一种生成场景一致印刷对抗攻击的方法，这种攻击可以误导高级LVLM，同时通过基于LLM的代理的能力保持视觉自然性。我们的方法解决了三个关键问题：生成什么对抗文本、将其放置在场景中的位置以及如何无缝集成它。我们提出了一种免培训、多模式LLM驱动的场景一致印刷对抗性规划（SceneRAP），该规划采用三阶段流程：场景理解、对抗性规划和无缝集成。SceneRAP利用思想链推理来理解场景、制定有效的对抗文本、战略性地规划其放置，并为图像中的自然整合提供详细的说明。随后是场景一致的文本扩散用户，它使用本地扩散机制执行攻击。我们通过打印并将生成的补丁放置在物理环境中，将我们的方法扩展到现实世界场景，展示其实际含义。大量实验表明，即使在捕获物理设置的新图像之后，我们的场景连贯对抗文本也能成功误导最先进的LVLM，包括ChatGPT-4 o。我们的评估表明，攻击成功率显着提高，同时保持视觉自然性和上下文适当性。这项工作强调了当前视觉语言模型对复杂、场景一致的对抗攻击的脆弱性，并提供了对潜在防御机制的见解。



## **33. ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs**

ShadowCoT：LLM中秘密推理后门的认知劫持 cs.CR

Zhao et al., 16 pages, 2025, uploaded by Hanzhou Wu, Shanghai  University

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05605v1) [paper-pdf](http://arxiv.org/pdf/2504.05605v1)

**Authors**: Gejian Zhao, Hanzhou Wu, Xinpeng Zhang, Athanasios V. Vasilakos

**Abstract**: Chain-of-Thought (CoT) enhances an LLM's ability to perform complex reasoning tasks, but it also introduces new security issues. In this work, we present ShadowCoT, a novel backdoor attack framework that targets the internal reasoning mechanism of LLMs. Unlike prior token-level or prompt-based attacks, ShadowCoT directly manipulates the model's cognitive reasoning path, enabling it to hijack multi-step reasoning chains and produce logically coherent but adversarial outcomes. By conditioning on internal reasoning states, ShadowCoT learns to recognize and selectively disrupt key reasoning steps, effectively mounting a self-reflective cognitive attack within the target model. Our approach introduces a lightweight yet effective multi-stage injection pipeline, which selectively rewires attention pathways and perturbs intermediate representations with minimal parameter overhead (only 0.15% updated). ShadowCoT further leverages reinforcement learning and reasoning chain pollution (RCP) to autonomously synthesize stealthy adversarial CoTs that remain undetectable to advanced defenses. Extensive experiments across diverse reasoning benchmarks and LLMs show that ShadowCoT consistently achieves high Attack Success Rate (94.4%) and Hijacking Success Rate (88.4%) while preserving benign performance. These results reveal an emergent class of cognition-level threats and highlight the urgent need for defenses beyond shallow surface-level consistency.

摘要: 思想链（CoT）增强了LLM执行复杂推理任务的能力，但也引入了新的安全问题。在这项工作中，我们提出了ShadowCoT，这是一种针对LLM内部推理机制的新型后门攻击框架。与之前的代币级或基于预算的攻击不同，ShadowCoT直接操纵模型的认知推理路径，使其能够劫持多步推理链并产生逻辑上连贯但具有对抗性的结果。通过以内部推理状态为条件，ShadowCoT学会识别并选择性地破坏关键推理步骤，有效地在目标模型内发起自我反思认知攻击。我们的方法引入了一种轻量级但有效的多阶段注入管道，它选择性地重新连接注意力路径并以最小的参数负担（仅更新0.15%）扰乱中间表示。ShadowCoT进一步利用强化学习和推理链污染（PGP）来自主合成先进防御系统无法检测到的隐形对抗CoT。跨各种推理基准和LLM的广泛实验表明，ShadowCoT始终实现高攻击成功率（94.4%）和劫持成功率（88.4%），同时保持良性性能。这些结果揭示了一类新出现的认知层面威胁，并凸显了对超越浅层表面一致性的防御的迫切需求。



## **34. Impact Assessment of Cyberattacks in Inverter-Based Microgrids**

基于逆变器的微电网网络攻击的影响评估 eess.SY

IEEE Workshop on the Electronic Grid (eGrid 2025)

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05592v1) [paper-pdf](http://arxiv.org/pdf/2504.05592v1)

**Authors**: Kerd Topallaj, Colin McKerrell, Suraj Ramanathan, Ioannis Zografopoulos

**Abstract**: In recent years, the evolution of modern power grids has been driven by the growing integration of remotely controlled grid assets. Although Distributed Energy Resources (DERs) and Inverter-Based Resources (IBR) enhance operational efficiency, they also introduce cybersecurity risks. The remote accessibility of such critical grid components creates entry points for attacks that adversaries could exploit, posing threats to the stability of the system. To evaluate the resilience of energy systems under such threats, this study employs real-time simulation and a modified version of the IEEE 39-bus system that incorporates a Microgrid (MG) with solar-based IBR. The study assesses the impact of remote attacks impacting the MG stability under different levels of IBR penetrations through Hardware-in-the-Loop (HIL) simulations. Namely, we analyze voltage, current, and frequency profiles before, during, and after cyberattack-induced disruptions. The results demonstrate that real-time HIL testing is a practical approach to uncover potential risks and develop robust mitigation strategies for resilient MG operations.

摘要: 近年来，远程控制电网资产的日益整合推动了现代电网的发展。尽管分布式能源资源（BER）和基于逆变器的资源（IBR）提高了运营效率，但它们也带来了网络安全风险。此类关键网格组件的远程访问性为对手可能利用的攻击创建了切入点，从而对系统的稳定性构成威胁。为了评估能源系统在此类威胁下的弹性，本研究采用了实时模拟和IEEE 39节点系统的修改版本，该系统结合了微电网（MG）和基于太阳能的IBR。该研究通过硬件在环（HIL）模拟评估了远程攻击在不同IBR渗透水平下对MG稳定性的影响。也就是说，我们分析网络攻击引发的中断之前、期间和之后的电压、电流和频率分布。结果表明，实时HIL测试是发现潜在风险并为有弹性的MG运营制定稳健的缓解策略的实用方法。



## **35. Secure Diagnostics: Adversarial Robustness Meets Clinical Interpretability**

安全诊断：对抗稳健性满足临床可解释性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05483v1) [paper-pdf](http://arxiv.org/pdf/2504.05483v1)

**Authors**: Mohammad Hossein Najafi, Mohammad Morsali, Mohammadreza Pashanejad, Saman Soleimani Roudi, Mohammad Norouzi, Saeed Bagheri Shouraki

**Abstract**: Deep neural networks for medical image classification often fail to generalize consistently in clinical practice due to violations of the i.i.d. assumption and opaque decision-making. This paper examines interpretability in deep neural networks fine-tuned for fracture detection by evaluating model performance against adversarial attack and comparing interpretability methods to fracture regions annotated by an orthopedic surgeon. Our findings prove that robust models yield explanations more aligned with clinically meaningful areas, indicating that robustness encourages anatomically relevant feature prioritization. We emphasize the value of interpretability for facilitating human-AI collaboration, in which models serve as assistants under a human-in-the-loop paradigm: clinically plausible explanations foster trust, enable error correction, and discourage reliance on AI for high-stakes decisions. This paper investigates robustness and interpretability as complementary benchmarks for bridging the gap between benchmark performance and safe, actionable clinical deployment.

摘要: 由于违反i.i.d，用于医学图像分类的深度神经网络在临床实践中往往无法一致地概括。假设和不透明的决策。本文通过评估模型针对对抗性攻击的性能并将可解释性方法与骨科医生注释的骨折区域进行比较，研究了深度神经网络的可解释性，该网络针对骨折检测进行了微调。我们的研究结果证明，稳健的模型可以产生与临床有意义的区域更加一致的解释，这表明稳健性鼓励解剖学相关的特征优先级。我们强调可解释性对于促进人类与人工智能合作的价值，在这种合作中，模型在人在回路范式下充当助手：临床上合理的解释可以促进信任，实现纠错，并阻止对人工智能的依赖。本文研究了作为补充基准的鲁棒性和可解释性，以弥合基准性能与安全、可操作的临床部署之间的差距。



## **36. Adversarial KA**

对手KA cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05255v1) [paper-pdf](http://arxiv.org/pdf/2504.05255v1)

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or {\guillemotleft}expressing{\guillemotright} functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs.

摘要: 将Kolmogorov和Arnold（KA）的表示定理视为表示或{\guillemotleft}表达{\guillemotleft}函数的算法，我们通过分析其抵御对抗攻击的能力来测试其稳健性。我们发现KA对于连续对手的可计数集合来说是稳健的，但我们发现了一个关于外部函数的等连续性的问题，到目前为止，该问题阻碍了采取限制和击败连续对手组。这个关于外部函数规律性的问题与KA对NN一般理论的适用性的争论有关。



## **37. Security Risks in Vision-Based Beam Prediction: From Spatial Proxy Attacks to Feature Refinement**

基于视觉的射束预测中的安全风险：从空间代理攻击到特征细化 cs.NI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05222v1) [paper-pdf](http://arxiv.org/pdf/2504.05222v1)

**Authors**: Avi Deb Raha, Kitae Kim, Mrityunjoy Gain, Apurba Adhikary, Zhu Han, Eui-Nam Huh, Choong Seon Hong

**Abstract**: The rapid evolution towards the sixth-generation (6G) networks demands advanced beamforming techniques to address challenges in dynamic, high-mobility scenarios, such as vehicular communications. Vision-based beam prediction utilizing RGB camera images emerges as a promising solution for accurate and responsive beam selection. However, reliance on visual data introduces unique vulnerabilities, particularly susceptibility to adversarial attacks, thus potentially compromising beam accuracy and overall network reliability. In this paper, we conduct the first systematic exploration of adversarial threats specifically targeting vision-based mmWave beam selection systems. Traditional white-box attacks are impractical in this context because ground-truth beam indices are inaccessible and spatial dynamics are complex. To address this, we propose a novel black-box adversarial attack strategy, termed Spatial Proxy Attack (SPA), which leverages spatial correlations between user positions and beam indices to craft effective perturbations without requiring access to model parameters or labels. To counteract these adversarial vulnerabilities, we formulate an optimization framework aimed at simultaneously enhancing beam selection accuracy under clean conditions and robustness against adversarial perturbations. We introduce a hybrid deep learning architecture integrated with a dedicated Feature Refinement Module (FRM), designed to systematically filter irrelevant, noisy and adversarially perturbed visual features. Evaluations using standard backbone models such as ResNet-50 and MobileNetV2 demonstrate that our proposed method significantly improves performance, achieving up to an +21.07\% gain in Top-K accuracy under clean conditions and a 41.31\% increase in Top-1 adversarial robustness compared to different baseline models.

摘要: 向第六代（6 G）网络的快速发展需要先进的束成形技术来应对动态、高移动性场景（例如车辆通信）中的挑战。利用Ruby相机图像的基于视觉的射束预测成为准确且响应灵敏的射束选择的有前途的解决方案。然而，对视觉数据的依赖会带来独特的漏洞，特别是对对抗攻击的敏感性，从而可能会损害射束准确性和整体网络可靠性。在本文中，我们对专门针对基于视觉的毫米波射束选择系统的对抗威胁进行了首次系统性探索。传统的白盒攻击在这种情况下是不切实际的，因为地面实况波束索引是不可访问的，空间动态是复杂的。为了解决这个问题，我们提出了一种新的黑盒对抗攻击策略，称为空间代理攻击（SPA），它利用用户位置和波束索引之间的空间相关性来制作有效的扰动，而无需访问模型参数或标签。为了抵消这些对抗性漏洞，我们制定了一个优化框架，旨在同时提高在清洁条件下的波束选择精度和对抗性扰动的鲁棒性。我们引入了一种混合深度学习架构，该架构集成了专用的特征细化模块（FRM），旨在系统地过滤不相关的、嘈杂的和不利干扰的视觉特征。使用ResNet-50和ALENetV 2等标准主干模型进行的评估表明，与不同的基线模型相比，我们提出的方法显着提高了性能，在清洁条件下Top-K准确性提高了+21.07%%，Top-1对抗鲁棒性提高了41. 31%%'。



## **38. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Models**

DiffPatch：使用扩散模型生成可定制的对抗补丁 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.01440v3) [paper-pdf](http://arxiv.org/pdf/2412.01440v3)

**Authors**: Zhixiang Wang, Xiaosen Wang, Bo Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can enable individuals to evade person detectors, but most existing methods prioritize attack effectiveness over stealthiness, resulting in aesthetically unpleasing patches. While generative adversarial networks and diffusion models can produce more natural-looking patches, they often fail to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these limitations, we propose DiffPatch, a novel diffusion-based framework for generating customizable and naturalistic adversarial patches. Our approach allows users to start from a reference image (rather than random noise) and incorporates masks to create patches of various shapes, not limited to squares. To preserve the original semantics during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Our method achieves attack performance comparable to state-of-the-art non-naturalistic patches while maintaining a natural appearance. Using DiffPatch, we construct AdvT-shirt-1K, the first physical adversarial T-shirt dataset comprising over a thousand images captured in diverse scenarios. AdvT-shirt-1K can serve as a useful dataset for training or testing future defense methods.

摘要: 印在衣服上的物理对抗性补丁可以使个人逃避人员检测器，但大多数现有的方法优先考虑攻击有效性而不是隐蔽性，导致美观的补丁。虽然生成式对抗网络和扩散模型可以生成更自然的补丁，但它们往往无法平衡隐蔽性和攻击有效性，并且缺乏用户定制的灵活性。为了解决这些限制，我们提出了DiffPatch，一种新的基于扩散的框架，用于生成可定制的和自然的对抗补丁。我们的方法允许用户从参考图像（而不是随机噪声）开始，并结合掩模来创建各种形状的补丁，而不限于正方形。为了在扩散过程中保留原始语义，我们采用零文本倒置将随机噪音样本映射到单个输入图像，并通过不完全扩散优化（IDO）生成补丁。我们的方法在保持自然外观的同时实现了与最先进的非自然主义补丁相当的攻击性能。使用迪夫补丁，我们构建了AdvT-shirt-1 K，这是第一个物理对抗性T恤数据集，包含在不同场景中捕获的一千多张图像。AdvT-shirt-1 K可以作为训练或测试未来防御方法的有用数据集。



## **39. Adversarial Robustness for Deep Learning-based Wildfire Prediction Models**

基于深度学习的野火预测模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2412.20006v3) [paper-pdf](http://arxiv.org/pdf/2412.20006v3)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Rapidly growing wildfires have recently devastated societal assets, exposing a critical need for early warning systems to expedite relief efforts. Smoke detection using camera-based Deep Neural Networks (DNNs) offers a promising solution for wildfire prediction. However, the rarity of smoke across time and space limits training data, raising model overfitting and bias concerns. Current DNNs, primarily Convolutional Neural Networks (CNNs) and transformers, complicate robustness evaluation due to architectural differences. To address these challenges, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating wildfire detection models' adversarial robustness. WARP addresses inherent limitations in data diversity by generating adversarial examples through image-global and -local perturbations. Global and local attacks superimpose Gaussian noise and PNG patches onto image inputs, respectively; this suits both CNNs and transformers while generating realistic adversarial scenarios. Using WARP, we assessed real-time CNNs and Transformers, uncovering key vulnerabilities. At times, transformers exhibited over 70% precision degradation under global attacks, while both models generally struggled to differentiate cloud-like PNG patches from real smoke during local attacks. To enhance model robustness, we proposed four wildfire-oriented data augmentation techniques based on WARP's methodology and results, which diversify smoke image data and improve model precision and robustness. These advancements represent a substantial step toward developing a reliable early wildfire warning system, which may be our first safeguard against wildfire destruction.

摘要: 最近，迅速蔓延的野火摧毁了社会资产，暴露出迫切需要预警系统来加快救援工作。使用基于相机的深度神经网络（DNN）进行烟雾检测为野火预测提供了一个有前途的解决方案。然而，烟雾在时间和空间中的稀有性限制了训练数据，从而引发了模型过度匹配和偏见的担忧。当前的DNN（主要是卷积神经网络（CNN）和变换器）由于架构差异而使稳健性评估变得复杂。为了应对这些挑战，我们引入了WARP（野火对抗鲁棒性程序），这是第一个用于评估野火检测模型对抗鲁棒性的模型不可知框架。WARP通过图像全局和局部扰动生成对抗性示例来解决数据多样性的固有限制。全局和局部攻击分别将高斯噪音和PNG补丁叠加到图像输入上;这适合CNN和变形器，同时生成现实的对抗场景。使用WARP，我们评估了实时CNN和变形金刚，发现了关键漏洞。有时，在全球攻击下，变压器的精确度会下降超过70%，而这两种模型在局部攻击期间通常很难区分云状的PNG补丁和真正的烟雾。为了增强模型的鲁棒性，我们基于WARP的方法和结果提出了四种面向野火的数据增强技术，这些技术使烟雾图像数据多样化并提高模型精度和鲁棒性。这些进步是朝着开发可靠的早期野火预警系统迈出的重要一步，这可能是我们防止野火破坏的第一个保障。



## **40. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **41. A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models**

大型语言模型中基于领域的越狱漏洞分类 cs.CL

21 pages, 5 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04976v1) [paper-pdf](http://arxiv.org/pdf/2504.04976v1)

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study.

摘要: 大型语言模型（LLM）的研究是开放世界机器学习的一个关键领域。尽管LLM表现出出色的自然语言处理能力，但它们也面临着一些挑战，包括一致性问题、幻觉和越狱漏洞。越狱是指绕过对齐保障措施的提示，导致不安全的输出，从而损害LLM的完整性。这项工作特别关注越狱漏洞的挑战，并引入了一种基于LLM训练领域的新颖越狱攻击分类法。它通过概括性、目标和稳健性差距来描述对齐失败。我们的主要贡献是对越狱的看法，通过LLM培训和调整期间出现的不同语言领域来框架。这一观点强调了现有方法的局限性，并使我们能够根据越狱攻击所利用的基础模型缺陷对越狱攻击进行分类。与基于即时构建方法对攻击进行分类的传统分类不同（例如，提示模板），我们的方法提供了一个更深入的了解LLM行为。我们引入了一个分类法，分为四个类别-不匹配的泛化，竞争目标，对抗性鲁棒性和混合攻击-提供了对越狱漏洞的基本性质的见解。最后，我们提出了从这一分类学研究中得出的关键教训。



## **42. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

努力图表：量化人工智能使用风险以进行漏洞评估 cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.16392v2) [paper-pdf](http://arxiv.org/pdf/2503.16392v2)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.

摘要: 随着基于人工智能的软件的广泛使用，利用其功能（例如高度自动化和复杂模式识别）的风险可能会显着增加。用于攻击非人工智能资产的人工智能称为进攻性人工智能。   当前的研究探讨了如何利用攻击性人工智能以及如何对其使用进行分类。此外，正在为组织内基于人工智能的资产开发威胁建模方法。然而，也有一些差距需要解决。首先，需要量化导致人工智能威胁的因素。其次，需要创建威胁模型，分析被人工智能攻击的风险，以便对组织所有资产进行漏洞评估。这在复杂的基础设施和访问控制环境普遍存在的云环境中尤其重要和具有挑战性。量化和进一步分析攻击性人工智能构成的威胁的能力使分析师能够对漏洞进行排名并优先考虑主动应对措施的实施。   为了解决这些差距，本文引入了“努力图”，这是一种直观、灵活且有效的威胁建模方法，用于分析对手使用攻击性人工智能进行漏洞利用所需的努力。虽然威胁模型是实用的并提供了有价值的支持，但其设计选择需要在未来的工作中进一步的经验验证。



## **43. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04858v1) [paper-pdf](http://arxiv.org/pdf/2504.04858v1)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动，对视觉系统构成重大威胁。传统的防御方法通常需要重新培训或微调，这使得它们对于现实世界的部署来说不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了用于对抗性补丁检测的视觉语言模型（VLM）。通过检索视觉上相似的补丁和图像，这些补丁和图像类似于不断扩展的数据库中存储的攻击，VRAG执行生成式推理以识别不同的攻击类型，而所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **44. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

针对3D资产保护的多视图扩散模型的潜在特征和注意力双重擦除攻击 cs.CV

This paper has been accepted by ICME 2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.11408v2) [paper-pdf](http://arxiv.org/pdf/2408.11408v2)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.

摘要: 多视图扩散模型（MVDM）使3D几何重建领域取得了显着进步，但由于未经授权的模仿，知识产权问题越来越受到关注。最近，一些作品利用对抗攻击来保护版权。然而，所有这些工作都集中在单图像生成任务上，只需要考虑图像的内部特征。以前的方法在攻击MVDM时效率低下，因为它们缺乏考虑破坏生成的多视图图像之间的几何和视觉一致性。本文是第一篇探讨MVDM引起的知识产权侵权问题的论文。因此，我们提出了一种新型的潜在特征和注意力双重擦除攻击，以同时破坏多视图和多域生成图像中潜在特征的分布和一致性。在SOTA MVDM上进行的实验表明，我们的方法在攻击有效性、可转移性和针对防御方法的鲁棒性方面实现了卓越的性能。因此，本文提供了一种有效的解决方案来保护3D资产免受基于MVDM的3D几何重建的影响。



## **45. Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios**

在安全关键场景下对自动驾驶的安全性和鲁棒性进行基准测试和评估 cs.RO

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2503.23708v2) [paper-pdf](http://arxiv.org/pdf/2503.23708v2)

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment.

摘要: 自动驾驶在学术界和工业界都取得了重大进展，包括感知任务的性能改进和端到端自动驾驶系统的开发。然而，自动驾驶的安全性和稳健性评估尚未得到足够的关注。目前对自动驾驶的评估通常是在自然驾驶场景中进行的。然而，许多事故通常发生在边缘情况下，也称为安全关键情况。这些安全关键场景很难收集，目前还没有明确的定义什么是安全关键场景。在这项工作中，我们探索了自动驾驶在安全关键场景中的安全性和稳健性。首先，我们提供了安全关键场景的定义，包括静态交通场景（例如对抗性攻击场景和自然分布变化）以及动态交通场景（例如事故场景）。然后，我们开发自动驾驶安全测试平台，对自动驾驶系统进行全面评估，不仅包括感知模块的评估，还包括系统级评估。我们的工作系统地构建了自动驾驶的安全验证流程，为行业建立标准化测试框架并降低现实道路部署风险提供技术支持。



## **46. SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection**

SINCon：缓解LLM生成的恶意消息注入攻击以进行谣言检测 cs.CR

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.07135v1) [paper-pdf](http://arxiv.org/pdf/2504.07135v1)

**Authors**: Mingqing Zhang, Qiang Liu, Xiang Tao, Shu Wu, Liang Wang

**Abstract**: In the era of rapidly evolving large language models (LLMs), state-of-the-art rumor detection systems, particularly those based on Message Propagation Trees (MPTs), which represent a conversation tree with the post as its root and the replies as its descendants, are facing increasing threats from adversarial attacks that leverage LLMs to generate and inject malicious messages. Existing methods are based on the assumption that different nodes exhibit varying degrees of influence on predictions. They define nodes with high predictive influence as important nodes and target them for attacks. If the model treats nodes' predictive influence more uniformly, attackers will find it harder to target high predictive influence nodes. In this paper, we propose Similarizing the predictive Influence of Nodes with Contrastive Learning (SINCon), a defense mechanism that encourages the model to learn graph representations where nodes with varying importance have a more uniform influence on predictions. Extensive experiments on the Twitter and Weibo datasets demonstrate that SINCon not only preserves high classification accuracy on clean data but also significantly enhances resistance against LLM-driven message injection attacks.

摘要: 在大型语言模型（LLM）快速发展的时代，最先进的谣言检测系统，特别是基于消息传播树（MPTS）的谣言检测系统，其代表以帖子为根、以回复为后代的对话树，正面临着越来越多的威胁。利用LLM来生成和注入恶意消息的对抗性攻击。现有的方法基于这样的假设：不同的节点对预测表现出不同程度的影响。他们将具有高预测影响力的节点定义为重要节点，并针对它们进行攻击。如果模型更均匀地对待节点的预测影响力，攻击者将发现更难瞄准高预测影响力的节点。在本文中，我们提出用对比学习来模拟节点的预测影响（SINCon），这是一种防御机制，鼓励模型学习图表示，其中重要性不同的节点对预测产生更均匀的影响。Twitter和微博数据集上的大量实验表明，SINCon不仅在干净数据上保持了高分类准确性，而且还显着增强了对LLM驱动的消息注入攻击的抵抗力。



## **47. Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models**

两胜一：强大而紧凑的模型的有效集成防御 cs.CV

Accepted to CVPR2025

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04747v1) [paper-pdf](http://arxiv.org/pdf/2504.04747v1)

**Authors**: Yoojin Jung, Byung Cheol Song

**Abstract**: Deep learning-based computer vision systems adopt complex and large architectures to improve performance, yet they face challenges in deployment on resource-constrained mobile and edge devices. To address this issue, model compression techniques such as pruning, quantization, and matrix factorization have been proposed; however, these compressed models are often highly vulnerable to adversarial attacks. We introduce the \textbf{Efficient Ensemble Defense (EED)} technique, which diversifies the compression of a single base model based on different pruning importance scores and enhances ensemble diversity to achieve high adversarial robustness and resource efficiency. EED dynamically determines the number of necessary sub-models during the inference stage, minimizing unnecessary computations while maintaining high robustness. On the CIFAR-10 and SVHN datasets, EED demonstrated state-of-the-art robustness performance compared to existing adversarial pruning techniques, along with an inference speed improvement of up to 1.86 times. This proves that EED is a powerful defense solution in resource-constrained environments.

摘要: 基于深度学习的计算机视觉系统采用复杂且大型的架构来提高性能，但它们在资源有限的移动和边缘设备上部署时面临挑战。为了解决这个问题，人们提出了修剪、量化和矩阵分解等模型压缩技术;然而，这些压缩模型通常极易受到对抗攻击。我们引入了\textBF{高效集合防御（EED）}技术，该技术根据不同的修剪重要性分数使单个基本模型的压缩多样化，并增强集合多样性，以实现高对抗鲁棒性和资源效率。EED在推理阶段动态确定必要子模型的数量，最大限度地减少不必要的计算，同时保持高稳健性。在CIFAR-10和SVHN数据集上，与现有的对抗性修剪技术相比，EED表现出了最先进的鲁棒性性能，并且推理速度提高了高达1.86倍。这证明EED是资源有限环境中强大的防御解决方案。



## **48. A Survey and Evaluation of Adversarial Attacks for Object Detection**

目标检测中的对抗性攻击综述与评价 cs.CV

Accepted for publication in the IEEE Transactions on Neural Networks  and Learning Systems (TNNLS)

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2408.01934v4) [paper-pdf](http://arxiv.org/pdf/2408.01934v4)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.

摘要: 深度学习模型在计算机视觉任务中实现了非凡的准确性，但仍然容易受到对抗性示例的影响--对输入图像精心设计的扰动，可能会欺骗这些模型做出自信但不正确的预测。该漏洞在自动驾驶汽车、安全监控和安全关键检查系统等高风险应用中构成了重大风险。虽然现有文献广泛涵盖了图像分类中的对抗攻击，但对对象检测系统上的此类攻击的全面分析仍然有限。本文提出了一种新颖的分类框架，用于对特定于对象检测架构的对抗性攻击进行分类，综合了现有的鲁棒性指标，并对流行对象检测模型（包括传统检测器和具有视觉语言预训练的现代检测器）的最新攻击方法进行了全面的实证评估。通过严格分析开源攻击实现及其在不同检测架构中的有效性，我们获得了对攻击特征的关键见解。此外，我们描述了关键的研究差距和新出现的挑战，以指导未来的调查，以确保对象检测系统免受对抗性威胁。我们的研究结果为开发更强大的检测模型奠定了基础，同时强调了在这个快速发展的领域迫切需要标准化的评估协议。



## **49. On the Robustness of GUI Grounding Models Against Image Attacks**

图形用户界面基础模型对抗图像攻击的鲁棒性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04716v1) [paper-pdf](http://arxiv.org/pdf/2504.04716v1)

**Authors**: Haoren Zhao, Tianyi Chen, Zhen Wang

**Abstract**: Graphical User Interface (GUI) grounding models are crucial for enabling intelligent agents to understand and interact with complex visual interfaces. However, these models face significant robustness challenges in real-world scenarios due to natural noise and adversarial perturbations, and their robustness remains underexplored. In this study, we systematically evaluate the robustness of state-of-the-art GUI grounding models, such as UGround, under three conditions: natural noise, untargeted adversarial attacks, and targeted adversarial attacks. Our experiments, which were conducted across a wide range of GUI environments, including mobile, desktop, and web interfaces, have clearly demonstrated that GUI grounding models exhibit a high degree of sensitivity to adversarial perturbations and low-resolution conditions. These findings provide valuable insights into the vulnerabilities of GUI grounding models and establish a strong benchmark for future research aimed at enhancing their robustness in practical applications. Our code is available at https://github.com/ZZZhr-1/Robust_GUI_Grounding.

摘要: 图形用户界面（GUI）基础模型对于使智能代理能够理解复杂的可视化界面并与之交互至关重要。然而，由于自然噪声和对抗性扰动，这些模型在现实世界的场景中面临着显著的鲁棒性挑战，并且它们的鲁棒性仍然未得到充分研究。在这项研究中，我们系统地评估了最先进的GUI接地模型（如UGround）在三种条件下的鲁棒性：自然噪声、非针对性对抗攻击和针对性对抗攻击。我们的实验在广泛的GUI环境中进行，包括移动，桌面和Web界面，已经清楚地表明GUI接地模型对对抗性扰动和低分辨率条件表现出高度的敏感性。这些发现为有关图形用户界面基础模型的漏洞提供了宝贵的见解，并为未来旨在增强其在实际应用中稳健性的研究建立了强有力的基准。我们的代码可在https://github.com/ZZZhr-1/Robust_GUI_Grounding上获取。



## **50. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

保护视觉语言模型：缓解基于扰动的攻击中高斯噪音的脆弱性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01308v2) [paper-pdf](http://arxiv.org/pdf/2504.01308v2)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

摘要: 视觉语言模型（VLMS）通过合并视觉信息扩展了大型语言模型（LLM）的功能，但它们仍然容易受到越狱攻击，尤其是在处理嘈杂或损坏的图像时。尽管现有的VLM在培训期间采取安全措施来减轻此类攻击，但与噪音增强视觉输入相关的漏洞被忽视了。在这项工作中，我们发现错过噪音增强训练会导致严重的安全漏洞：许多VLM甚至容易受到高斯噪音等简单扰动的影响。为了应对这一挑战，我们提出了Robust-VLGuard，这是一个具有对齐/未对齐图像-文本对的多模式安全数据集，结合了噪音增强微调，可以降低攻击成功率，同时保留VLM的功能。对于更强的基于优化的视觉扰动攻击，我们提出了DiffPure-VLM，利用扩散模型将对抗性扰动转换为类高斯噪声，可以通过具有噪声增强安全微调的VLM进行防御。实验结果表明，扩散模型的分布偏移特性与我们微调的VLM很好地吻合，显著减轻了不同强度的对抗性扰动。数据集和代码可在https://github.com/JarvisUSTC/DiffPure-RobustVLM上获取。



