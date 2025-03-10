# Latest Adversarial Attack Papers
**update at 2025-03-10 09:56:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2407.20836v2) [paper-pdf](http://arxiv.org/pdf/2407.20836v2)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.

摘要: 最近在图像合成方面的进步，特别是随着GaN和扩散模型的出现，放大了公众对虚假信息传播的担忧。为了解决这些问题，人们已经提出了许多人工智能生成的图像(AIGI)检测器，并在识别虚假图像方面取得了良好的性能。然而，对于AIGI检测器的对抗健壮性，目前还缺乏系统的了解。本文研究了白盒和黑盒环境下最新的AIGI检测器抵抗敌意攻击的脆弱性，这是迄今为止很少被研究的。为此，我们提出了一种攻击AIGI探测器的新方法。首先，受真伪图像在频域存在明显差异的启发，在频域下加入扰动，使图像偏离其原有的频率分布。其次，我们探索了代理模型的完全后验分布，以进一步缩小不同AIGI检测器之间的差距，例如在CNN和VITS之间传输敌意示例。这是通过引入一种新颖的后训练贝叶斯策略来实现的，该策略将单个代理转变为贝叶斯策略，能够使用一个预先训练的代理来模拟不同的受害者模型，而不需要重新训练。我们将我们的方法命名为基于频率的后训练贝叶斯攻击，或称FPBA。通过FPBA，我们证明了敌意攻击是对AIGI检测器的真正威胁，因为FPBA可以跨模型、生成器、防御方法提供成功的黑盒攻击，甚至可以逃避交叉生成器检测，这是现实世界中的一个关键检测场景。代码将在接受后共享。



## **2. Toward Robust Non-Transferable Learning: A Survey and Benchmark**

迈向稳健的不可转移学习：调查和基准 cs.LG

Code is available at https://github.com/tmllab/NTLBench

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2502.13593v2) [paper-pdf](http://arxiv.org/pdf/2502.13593v2)

**Authors**: Ziming Hong, Yongli Xiang, Tongliang Liu

**Abstract**: Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. In this paper, we bridge this gap by presenting the first comprehensive survey on NTL and introducing NTLBench, the first benchmark to evaluate NTL performance and robustness within a unified framework. Specifically, we first introduce the task settings, general framework, and criteria of NTL, followed by a summary of NTL approaches. Furthermore, we emphasize the often-overlooked issue of robustness against various attacks that can destroy the non-transferable mechanism established by NTL. Experiments conducted via NTLBench verify the limitations of existing NTL methods in robustness. Finally, we discuss the practical applications of NTL, along with its future directions and associated challenges.

摘要: 在过去的几十年里，研究人员主要专注于提高模型的泛化能力，而对规范这种泛化的关注很少。然而，模型概括为非预期数据(例如，有害或未经授权的数据)的能力可能会被恶意攻击者以不可预见的方式利用，可能导致违反模型道德。为应对这些挑战，提出了一项旨在重塑深度学习模型泛化能力的任务--不可迁移学习(NTL)。虽然在这一领域提出了许多方法，但仍然缺乏对现有进展的全面审查和对当前限制的透彻分析。在本文中，我们通过介绍第一次关于NTL的全面调查并引入NTLBitch来弥合这一差距，NTLBuchch是第一个在统一框架内评估NTL性能和健壮性的基准。具体地说，我们首先介绍了网络学习的任务设置、总体框架和标准，然后对网络学习的方法进行了概述。此外，我们强调了经常被忽视的问题，即对各种攻击的健壮性，这些攻击可以破坏NTL建立的不可转移机制。通过NTLBitch进行的实验验证了现有NTL方法在稳健性方面的局限性。最后，我们讨论了NTL的实际应用，以及它的未来方向和相关的挑战。



## **3. Robust Intrusion Detection System with Explainable Artificial Intelligence**

具有可解释人工智能的稳健入侵检测系统 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05303v1) [paper-pdf](http://arxiv.org/pdf/2503.05303v1)

**Authors**: Betül Güvenç Paltun, Ramin Fuladi, Rim El Malki

**Abstract**: Machine learning (ML) models serve as powerful tools for threat detection and mitigation; however, they also introduce potential new risks. Adversarial input can exploit these models through standard interfaces, thus creating new attack pathways that threaten critical network operations. As ML advancements progress, adversarial strategies become more advanced, and conventional defenses such as adversarial training are costly in computational terms and often fail to provide real-time detection. These methods typically require a balance between robustness and model performance, which presents challenges for applications that demand instant response. To further investigate this vulnerability, we suggest a novel strategy for detecting and mitigating adversarial attacks using eXplainable Artificial Intelligence (XAI). This approach is evaluated in real time within intrusion detection systems (IDS), leading to the development of a zero-touch mitigation strategy. Additionally, we explore various scenarios in the Radio Resource Control (RRC) layer within the Open Radio Access Network (O-RAN) framework, emphasizing the critical need for enhanced mitigation techniques to strengthen IDS defenses against advanced threats and implement a zero-touch mitigation solution. Extensive testing across different scenarios in the RRC layer of the O-RAN infrastructure validates the ability of the framework to detect and counteract integrated RRC-layer attacks when paired with adversarial strategies, emphasizing the essential need for robust defensive mechanisms to strengthen IDS against complex threats.

摘要: 机器学习(ML)模型是检测和缓解威胁的强大工具，但它们也带来了潜在的新风险。敌意输入可以通过标准接口利用这些模型，从而创建威胁关键网络操作的新攻击路径。随着ML的进步，对抗性策略变得更加先进，而传统的防御方法，如对抗性训练，在计算方面代价高昂，而且往往无法提供实时检测。这些方法通常需要在稳健性和模型性能之间取得平衡，这给需要即时响应的应用程序带来了挑战。为了进一步研究这一漏洞，我们提出了一种使用可解释人工智能(XAI)检测和缓解对手攻击的新策略。这种方法在入侵检测系统(IDS)中进行实时评估，从而开发出一种零接触缓解策略。此外，我们还探讨了开放式无线接入网络(O-RAN)框架内无线资源控制(RRC)层的各种场景，强调了增强缓解技术的迫切需要，以加强入侵检测系统对高级威胁的防御并实施零接触缓解解决方案。在O-RAN基础设施的RRC层的不同场景中进行的广泛测试验证了该框架在与对抗性策略配合使用时检测和对抗集成RRC层攻击的能力，强调了加强入侵检测系统对抗复杂威胁的强大防御机制的基本必要性。



## **4. Jailbreaking is (Mostly) Simpler Than You Think**

越狱（大多数）比你想象的要简单 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05264v1) [paper-pdf](http://arxiv.org/pdf/2503.05264v1)

**Authors**: Mark Russinovich, Ahmed Salem

**Abstract**: We introduce the Context Compliance Attack (CCA), a novel, optimization-free method for bypassing AI safety mechanisms. Unlike current approaches -- which rely on complex prompt engineering and computationally intensive optimization -- CCA exploits a fundamental architectural vulnerability inherent in many deployed AI systems. By subtly manipulating conversation history, CCA convinces the model to comply with a fabricated dialogue context, thereby triggering restricted behavior. Our evaluation across a diverse set of open-source and proprietary models demonstrates that this simple attack can circumvent state-of-the-art safety protocols. We discuss the implications of these findings and propose practical mitigation strategies to fortify AI systems against such elementary yet effective adversarial tactics.

摘要: 我们引入了上下文合规攻击（PCA），这是一种新颖的、无需优化的方法，用于绕过人工智能安全机制。与当前的方法（依赖于复杂的即时工程和计算密集型优化）不同，CAA利用了许多已部署的人工智能系统固有的基本架构漏洞。通过巧妙地操纵对话历史，CAA说服模型遵守捏造的对话上下文，从而触发受限制的行为。我们对各种开源和专有模型的评估表明，这种简单的攻击可以规避最先进的安全协议。我们讨论了这些发现的影响，并提出了实用的缓解策略，以加强人工智能系统对抗这种基本但有效的对抗策略。



## **5. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2410.23746v2) [paper-pdf](http://arxiv.org/pdf/2410.23746v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **6. Safety-Critical Traffic Simulation with Adversarial Transfer of Driving Intentions**

具有驾驶意图对抗性转移的安全关键交通模拟 cs.RO

Accepted by ICRA 2025

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05180v1) [paper-pdf](http://arxiv.org/pdf/2503.05180v1)

**Authors**: Zherui Huang, Xing Gao, Guanjie Zheng, Licheng Wen, Xuemeng Yang, Xiao Sun

**Abstract**: Traffic simulation, complementing real-world data with a long-tail distribution, allows for effective evaluation and enhancement of the ability of autonomous vehicles to handle accident-prone scenarios. Simulating such safety-critical scenarios is nontrivial, however, from log data that are typically regular scenarios, especially in consideration of dynamic adversarial interactions between the future motions of autonomous vehicles and surrounding traffic participants. To address it, this paper proposes an innovative and efficient strategy, termed IntSim, that explicitly decouples the driving intentions of surrounding actors from their motion planning for realistic and efficient safety-critical simulation. We formulate the adversarial transfer of driving intention as an optimization problem, facilitating extensive exploration of diverse attack behaviors and efficient solution convergence. Simultaneously, intention-conditioned motion planning benefits from powerful deep models and large-scale real-world data, permitting the simulation of realistic motion behaviors for actors. Specially, through adapting driving intentions based on environments, IntSim facilitates the flexible realization of dynamic adversarial interactions with autonomous vehicles. Finally, extensive open-loop and closed-loop experiments on real-world datasets, including nuScenes and Waymo, demonstrate that the proposed IntSim achieves state-of-the-art performance in simulating realistic safety-critical scenarios and further improves planners in handling such scenarios.

摘要: 交通模拟用长尾分布补充了真实世界的数据，允许有效地评估和增强自动驾驶车辆处理事故多发场景的能力。然而，从通常是常规场景的日志数据中模拟这样的安全关键场景并不容易，特别是考虑到自动驾驶车辆未来的运动与周围交通参与者之间的动态对抗性交互。为了解决这一问题，本文提出了一种创新而高效的策略，称为IntSim，该策略明确地将周围参与者的驾驶意图与他们的运动规划解耦，以实现逼真和高效的安全关键模拟。我们将驾驶意图的对抗性转移描述为一个优化问题，便于广泛探索不同的攻击行为和高效的解收敛。同时，意图约束运动规划得益于强大的深层模型和大规模的真实世界数据，允许模拟演员的真实运动行为。特别是，通过基于环境自适应驾驶意图，IntSim有助于灵活实现与自动驾驶车辆的动态对抗性交互。最后，在包括nuScenes和Waymo在内的真实数据集上进行了大量的开环和闭环实验，结果表明，IntSim在模拟现实安全关键场景方面达到了最先进的性能，并进一步提高了规划者处理此类场景的能力。



## **7. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

双重后门：通过对抗性指令调优攻击将代码大型语言模型后门转换为传统恶意软件 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.

摘要: 为编码任务而设计的指令调整的大型语言模型越来越多地被用作人工智能编码助手。然而，由于这一领域的研究有限，这些模型的广泛集成所产生的网络安全漏洞和影响尚未完全了解。这项工作研究了将后门从AI/ML领域过渡到传统计算机恶意软件的新技术，揭示了人工智能和网络/软件安全的关键交集。为了探索这一交叉，我们提出了MalInstructCoder，一个旨在全面评估指令调优代码LLM的网络安全漏洞的框架。MalInstructCoder引入了自动数据中毒管道，将恶意代码片段注入良性代码，在保持功能有效性的同时毒化指令微调数据。它提出了两种具有实际安全含义的对抗性指令调整攻击：干净的即时中毒攻击和后门攻击。这些攻击旨在操纵Code LLM在特定攻击场景下生成包含恶意或有害功能的代码，同时保留预期功能。我们对代码特定指令调优过程的可利用性进行了全面的调查，涉及三个最先进的代码LLM：CodeLlama、DeepSeek-Coder和StarCoder2。我们的发现表明，这些模型非常容易受到我们的攻击。具体地说，干净的即时中毒攻击通过仅对指令微调数据集的1%(162个样本)下毒来实现从超过75%到86%的ASR@1。同样，后门攻击实现了从76%到86%的ASR@1，投毒率为0.5%。我们的研究揭示了指令调整的代码LLM构成的关键网络安全风险，并强调了对强大防御机制的迫切需要。



## **8. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

安全不仅仅是拒绝：推理增强微调，以实现可解释的LLM安全 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击利用了传统安全对齐中的弱点，传统安全对齐通常依赖僵化的拒绝启发式或表示工程来阻止有害输出。虽然它们对直接对抗性攻击是有效的，但它们不能满足更广泛的安全挑战，需要细致入微的、上下文感知的决策。为了解决这个问题，我们提出了针对可解释LLM安全(Rational)的推理增强精调，这是一个新的框架，它训练模型在响应之前进行显式的安全推理。微调模型利用自生成推理中丰富的预训练知识，通过结构化推理来引导自己的安全性，使上下文敏感决策内在化。我们的发现表明，安全超越了拒绝，需要背景感知才能做出更健壮、可解释和适应性更强的反应。推理是LLMS的核心能力，也是LLMS安全的基本机制。Rational采用了推理增强的微调，使其能够拒绝有害提示，同时在复杂场景中提供有意义的上下文感知响应。



## **9. Energy-Latency Attacks: A New Adversarial Threat to Deep Learning**

能量延迟攻击：深度学习的新对抗威胁 cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04963v1) [paper-pdf](http://arxiv.org/pdf/2503.04963v1)

**Authors**: Hanene F. Z. Brachemi Meftah, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstract**: The growing computational demand for deep neural networks ( DNNs) has raised concerns about their energy consumption and carbon footprint, particularly as the size and complexity of the models continue to increase. To address these challenges, energy-efficient hardware and custom accelerators have become essential. Additionally, adaptable DNN s are being developed to dynamically balance performance and efficiency. The use of these strategies became more common to enable sustainable AI deployment. However, these efficiency-focused designs may also introduce vulnerabilities, as attackers can potentially exploit them to increase latency and energy usage by triggering their worst-case-performance scenarios. This new type of attack, called energy-latency attacks, has recently gained significant research attention, focusing on the vulnerability of DNN s to this emerging attack paradigm, which can trigger denial-of-service ( DoS) attacks. This paper provides a comprehensive overview of current research on energy-latency attacks, categorizing them using the established taxonomy for traditional adversarial attacks. We explore different metrics used to measure the success of these attacks and provide an analysis and comparison of existing attack strategies. We also analyze existing defense mechanisms and highlight current challenges and potential areas for future research in this developing field. The GitHub page for this work can be accessed at https://github.com/hbrachemi/Survey_energy_attacks/

摘要: 对深度神经网络(DNN)日益增长的计算需求引起了人们对其能源消耗和碳足迹的担忧，特别是随着模型的规模和复杂性不断增加。为了应对这些挑战，节能硬件和定制加速器变得至关重要。此外，正在开发自适应DNN S，以动态平衡性能和效率。这些战略的使用变得更加普遍，以实现可持续的人工智能部署。然而，这些注重效率的设计也可能引入漏洞，因为攻击者可能会通过触发其最差情况下的性能情景来利用这些漏洞来增加延迟和能源使用。这种被称为能量延迟攻击的新型攻击最近得到了重要的研究关注，重点关注了DNN S对这种新兴攻击范式的脆弱性，这种攻击范式可以触发拒绝服务(DoS)攻击。本文对能量延迟攻击的研究现状进行了全面的综述，并利用已建立的传统对抗性攻击分类方法对其进行了分类。我们探索了不同的度量标准来衡量这些攻击的成功，并对现有的攻击策略进行了分析和比较。我们还分析了现有的防御机制，并强调了这一发展中领域当前面临的挑战和未来研究的潜在领域。这项工作的giHub页面可以在https://github.com/hbrachemi/Survey_energy_attacks/上访问



## **10. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.06186v4) [paper-pdf](http://arxiv.org/pdf/2410.06186v4)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **11. FSPGD: Rethinking Black-box Attacks on Semantic Segmentation**

FSPVD：重新思考对语义分割的黑匣子攻击 cs.CV

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2502.01262v2) [paper-pdf](http://arxiv.org/pdf/2502.01262v2)

**Authors**: Eun-Sol Park, MiSo Park, Seung Park, Yong-Goo Shin

**Abstract**: Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean images and adversarial examples, while also disrupting contextual information by accounting for spatial relationships between objects. Experiments on Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark. Code is available at https://github.com/KU-AIVS/FSPGD.

摘要: 可转移性，即为一个模型制作的敌意例子欺骗其他模型的能力，对黑盒攻击至关重要。尽管语义分割的攻击方法有所进步，但可转移性仍然有限，降低了它们在现实世界应用中的有效性。为了解决这一问题，我们引入了特征相似性投影梯度下降(FSPGD)攻击，这是一种新的黑盒方法，它同时提高了攻击性能和可转移性。与依赖输出预测进行梯度计算的传统分割攻击不同，FSPGD根据中间层特征计算梯度。具体地说，我们的方法引入了一个损失函数，该函数通过比较干净图像和敌意图像之间的特征来定位局部信息，同时还通过考虑对象之间的空间关系来破坏上下文信息。在Pascal VOC 2012和CITYSCAPES数据集上的实验表明，FSPGD实现了卓越的可转移性和攻击性能，建立了一个新的最先进的基准。代码可在https://github.com/KU-AIVS/FSPGD.上找到



## **12. OrbID: Identifying Orbcomm Satellite RF Fingerprints**

OrbID：识别OrbSYS卫星RF指纹 eess.SP

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.02118v2) [paper-pdf](http://arxiv.org/pdf/2503.02118v2)

**Authors**: Cédric Solenthaler, Joshua Smailes, Martin Strohmeier

**Abstract**: An increase in availability of Software Defined Radios (SDRs) has caused a dramatic shift in the threat landscape of legacy satellite systems, opening them up to easy spoofing attacks by low-budget adversaries. Physical-layer authentication methods can help improve the security of these systems by providing additional validation without modifying the space segment. This paper extends previous research on Radio Frequency Fingerprinting (RFF) of satellite communication to the Orbcomm satellite formation. The GPS and Iridium constellations are already well covered in prior research, but the feasibility of transferring techniques to other formations has not yet been examined, and raises previously undiscussed challenges.   In this paper, we collect a novel dataset containing 8992474 packets from the Orbcom satellite constellation using different SDRs and locations. We use this dataset to train RFF systems based on convolutional neural networks. We achieve an ROC AUC score of 0.53 when distinguishing different satellites within the constellation, and 0.98 when distinguishing legitimate satellites from SDRs in a spoofing scenario. We also demonstrate the possibility of mixing datasets using different SDRs in different physical locations.

摘要: 软件定义无线电(SDR)可用性的增加导致传统卫星系统的威胁格局发生了戏剧性的变化，使它们容易受到低预算对手的欺骗攻击。物理层身份验证方法可以在不修改空间段的情况下提供额外的验证，从而帮助提高这些系统的安全性。本文将以往卫星通信射频指纹识别的研究扩展到Orbcomm卫星编队。GPS和Iridium星座已经在以前的研究中得到了很好的覆盖，但将技术转移到其他编队的可行性还没有被审查，这提出了以前没有讨论过的挑战。在本文中，我们使用不同的SDR和不同的位置从Orbcom卫星星座收集了一个包含8992474个信息包的新数据集。我们使用这个数据集来训练基于卷积神经网络的RFF系统。当区分星座内的不同卫星时，我们的ROC AUC得分为0.53，在欺骗情况下区分合法卫星和SDR时，我们的ROC AUC得分为0.98。我们还演示了在不同的物理位置使用不同的SDR混合数据集的可能性。



## **13. Poisoning Bayesian Inference via Data Deletion and Replication**

通过数据删除和复制毒害Bayesian推理 stat.ML

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04480v1) [paper-pdf](http://arxiv.org/pdf/2503.04480v1)

**Authors**: Matthieu Carreau, Roi Naveiro, William N. Caballero

**Abstract**: Research in adversarial machine learning (AML) has shown that statistical models are vulnerable to maliciously altered data. However, despite advances in Bayesian machine learning models, most AML research remains concentrated on classical techniques. Therefore, we focus on extending the white-box model poisoning paradigm to attack generic Bayesian inference, highlighting its vulnerability in adversarial contexts. A suite of attacks are developed that allow an attacker to steer the Bayesian posterior toward a target distribution through the strategic deletion and replication of true observations, even when only sampling access to the posterior is available. Analytic properties of these algorithms are proven and their performance is empirically examined in both synthetic and real-world scenarios. With relatively little effort, the attacker is able to substantively alter the Bayesian's beliefs and, by accepting more risk, they can mold these beliefs to their will. By carefully constructing the adversarial posterior, surgical poisoning is achieved such that only targeted inferences are corrupted and others are minimally disturbed.

摘要: 对抗性机器学习(AML)的研究表明，统计模型很容易受到恶意篡改数据的影响。然而，尽管贝叶斯机器学习模型取得了进展，但大多数AML研究仍然集中在经典技术上。因此，我们专注于扩展白盒模型中毒范例来攻击通用贝叶斯推理，强调其在对抗性环境中的脆弱性。开发了一套攻击，允许攻击者通过战略性删除和复制真实观测来引导贝叶斯后验向目标分布，即使在只有对后验的采样访问可用的情况下也是如此。证明了这些算法的分析性质，并在合成场景和真实世界场景中对它们的性能进行了经验检验。攻击者只需相对较少的努力，就能够实质性地改变贝叶斯人的信念，通过接受更多的风险，他们可以按照自己的意愿塑造这些信念。通过仔细构建对抗性后验，外科中毒被实现，使得只有有针对性的推理被破坏，而其他推理被最小限度地干扰。



## **14. Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges**

了解你的法官：LLM安全法官的稳健性元评估 cs.LG

Accepted to the ICBINB Workshop at ICLR'25

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04474v1) [paper-pdf](http://arxiv.org/pdf/2503.04474v1)

**Authors**: Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan

**Abstract**: Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.

摘要: 基于大型语言模型(LLM)的评委构成了关键安全评估流程的基础，如离线基准、自动红色团队和在线护栏。这一普遍的要求提出了一个关键问题：我们能相信这些评估者的评价吗？在这篇文章中，我们强调了两个通常被忽视的关键挑战：(I)在野外评估中，敏感度和分布变化等因素会影响绩效；(Ii)针对法官的对抗性攻击。我们通过对常用安全法官的研究来强调这些的重要性，表明微小的变化，如模型输出的风格，可以导致同一数据集上的假阴性率跃升高达0.24，而对模型生成的对抗性攻击可能会欺骗一些法官，将100%的有害世代错误分类为安全世代。这些发现揭示了常用元评估基准的差距和当前LLM法官稳健性方面的弱点，表明在某些法官的领导下，低攻击成功率可能会产生一种错误的安全感。



## **15. Privacy Preserving and Robust Aggregation for Cross-Silo Federated Learning in Non-IID Settings**

非IID设置中跨筒仓联邦学习的隐私保护和鲁棒聚合 cs.LG

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04451v1) [paper-pdf](http://arxiv.org/pdf/2503.04451v1)

**Authors**: Marco Arazzi, Mert Cihangiroglu, Antonino Nocera

**Abstract**: Federated Averaging remains the most widely used aggregation strategy in federated learning due to its simplicity and scalability. However, its performance degrades significantly in non-IID data settings, where client distributions are highly imbalanced or skewed. Additionally, it relies on clients transmitting metadata, specifically the number of training samples, which introduces privacy risks and may conflict with regulatory frameworks like the European GDPR. In this paper, we propose a novel aggregation strategy that addresses these challenges by introducing class-aware gradient masking. Unlike traditional approaches, our method relies solely on gradient updates, eliminating the need for any additional client metadata, thereby enhancing privacy protection. Furthermore, our approach validates and dynamically weights client contributions based on class-specific importance, ensuring robustness against non-IID distributions, convergence prevention, and backdoor attacks. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms FedAvg and other widely accepted aggregation strategies in non-IID settings but also preserves model integrity in adversarial scenarios. Our results establish the effectiveness of gradient masking as a practical and secure solution for federated learning.

摘要: 由于其简单性和可伸缩性，联合平均仍然是联合学习中使用最广泛的聚合策略。然而，在客户端分布高度不平衡或不对称的非IID数据设置中，其性能会显著降低。此外，它依赖于客户传输元数据，特别是训练样本的数量，这会带来隐私风险，并可能与欧洲GDPR等监管框架相冲突。在本文中，我们提出了一种新的聚合策略，通过引入类感知梯度掩码来解决这些挑战。与传统方法不同，我们的方法完全依赖于梯度更新，不需要任何额外的客户端元数据，从而增强了隐私保护。此外，我们的方法基于特定于类的重要性来验证和动态加权客户端贡献，从而确保对非IID分发、收敛防止和后门攻击的健壮性。在基准数据集上的大量实验表明，我们的方法不仅在非IID环境下优于FedAvg和其他被广泛接受的聚集策略，而且在对抗性场景下保持了模型的完整性。我们的结果证明了梯度掩蔽作为联邦学习的一种实用和安全的解决方案的有效性。



## **16. Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients**

通过打破隐形代理对尖峰神经网络进行有效且稀疏的对抗攻击 cs.CV

Accepted by CVPR 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.03272v2) [paper-pdf](http://arxiv.org/pdf/2503.03272v2)

**Authors**: Li Lun, Kunyu Feng, Qinglong Ni, Ling Liang, Yuan Wang, Ying Li, Dunshan Yu, Xiaoxin Cui

**Abstract**: Spiking neural networks (SNNs) have shown their competence in handling spatial-temporal event-based data with low energy consumption. Similar to conventional artificial neural networks (ANNs), SNNs are also vulnerable to gradient-based adversarial attacks, wherein gradients are calculated by spatial-temporal back-propagation (STBP) and surrogate gradients (SGs). However, the SGs may be invisible for an inference-only model as they do not influence the inference results, and current gradient-based attacks are ineffective for binary dynamic images captured by the dynamic vision sensor (DVS). While some approaches addressed the issue of invisible SGs through universal SGs, their SGs lack a correlation with the victim model, resulting in sub-optimal performance. Moreover, the imperceptibility of existing SNN-based binary attacks is still insufficient. In this paper, we introduce an innovative potential-dependent surrogate gradient (PDSG) method to establish a robust connection between the SG and the model, thereby enhancing the adaptability of adversarial attacks across various models with invisible SGs. Additionally, we propose the sparse dynamic attack (SDA) to effectively attack binary dynamic images. Utilizing a generation-reduction paradigm, SDA can fully optimize the sparsity of adversarial perturbations. Experimental results demonstrate that our PDSG and SDA outperform state-of-the-art SNN-based attacks across various models and datasets. Specifically, our PDSG achieves 100% attack success rate on ImageNet, and our SDA obtains 82% attack success rate by modifying only 0.24% of the pixels on CIFAR10DVS. The code is available at https://github.com/ryime/PDSG-SDA .

摘要: 尖峰神经网络(SNN)在处理基于事件的时空数据方面表现出了低能耗的能力。与传统的人工神经网络(ANN)类似，SNN也容易受到基于梯度的攻击，其中梯度是通过时空反向传播(STBP)和代理梯度(SGS)来计算的。然而，对于仅限推理的模型，SGS可能是不可见的，因为它们不影响推理结果，并且当前基于梯度的攻击对于动态视觉传感器(DVS)捕获的二值动态图像无效。虽然一些方法通过通用的SGS解决了看不见的SGS的问题，但它们的SGS缺乏与受害者模型的相关性，导致性能次优。此外，现有的基于SNN的二进制攻击的不可见性仍然是不够的。在本文中，我们引入了一种创新的势依赖代理梯度(PDSG)方法来在SG和模型之间建立稳健的连接，从而增强了对具有不可见SGS的各种模型的对抗性攻击的适应性。此外，我们还提出了稀疏动态攻击(SDA)来有效地攻击二值动态图像。利用世代约简范式，SDA可以充分优化逆境扰动的稀疏性。实验结果表明，在不同的模型和数据集上，我们的PDSG和SDA都优于最先进的基于SNN的攻击。具体地说，我们的PDSG在ImageNet上达到了100%的攻击成功率，而我们的SDA在CIFAR10DVS上只修改了0.24%的像素就获得了82%的攻击成功率。代码可在https://github.com/ryime/PDSG-SDA上获得。



## **17. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Preemptive Adversarial Defense**

快速抢占：前向-后向级联学习，实现高效且可转移的抢占式对抗防御 cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2407.15524v7) [paper-pdf](http://arxiv.org/pdf/2407.15524v7)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: {Deep learning has made significant strides but remains susceptible to adversarial attacks, undermining its reliability. Most existing research addresses these threats after attacks happen. A growing direction explores preemptive defenses like mitigating adversarial threats proactively, offering improved robustness but at cost of efficiency and transferability. This paper introduces Fast Preemption, a novel preemptive adversarial defense that overcomes efficiency challenges while achieving state-of-the-art robustness and transferability, requiring no prior knowledge of attacks and target models. We propose a forward-backward cascade learning algorithm, which generates protective perturbations by combining forward propagation for rapid convergence with iterative backward propagation to prevent overfitting. Executing in just three iterations, Fast Preemption outperforms existing training-time, test-time, and preemptive defenses. Additionally, we introduce an adaptive reversion attack to assess the reliability of preemptive defenses, demonstrating that our approach remains secure in realistic attack scenarios.

摘要: {深度学习已经取得了重大进展，但仍然容易受到对手攻击，破坏了其可靠性。大多数现有的研究都是在袭击发生后解决这些威胁的。一个不断发展的方向是探索先发制人的防御措施，如主动缓解对手威胁，提供更好的健壮性，但代价是效率和可转移性。本文介绍了一种新型的抢占式对抗防御技术--快速抢占，它克服了效率上的挑战，同时实现了最先进的健壮性和可转移性，不需要攻击和目标模型的先验知识。我们提出了一种前向-后向级联学习算法，该算法通过结合前向传播快速收敛和迭代后向传播防止过拟合来产生保护性扰动。快速先发制人只需三次迭代即可完成，其性能优于现有的训练时间、测试时间和先发制人防御系统。此外，我们引入了自适应反向攻击来评估先发制人防御的可靠性，证明了我们的方法在现实攻击场景中仍然是安全的。



## **18. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

针对辅助规模超分辨率的规模不变对抗攻击 cs.CV

15 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04385v1) [paper-pdf](http://arxiv.org/pdf/2503.04385v1)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.

摘要: 局部连续图像函数(LIIF)的出现引起了任意尺度超分辨率(SR)技术的极大关注。然而，尽管固定尺度SR的脆弱性已经被评估，但基于连续表示的任意尺度SR对敌意攻击的稳健性仍然是一个值得进一步研究的领域。针对固定规模SR精心设计的敌意攻击是规模相关的，当应用于任意规模SR时，会导致耗时和内存消耗的问题。为了解决这一问题，我们提出了一种简单而有效的具有良好可转移性的“尺度不变”SR对抗攻击方法，称为SIAGT。具体地说，我们提出了利用连续表示的有限离散点来构造节省资源的攻击。此外，为了增强攻击的跨模型可转移性，我们还制定了一个坐标相关损失。该攻击可以显著恶化SR图像，同时给目标低分辨率(LR)图像带来不可察觉的失真。在三种流行的基于LIIF的SR方法和四个经典SR数据集上的实验表明，SIAGT具有良好的攻击性能和可转移性。



## **19. $σ$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples**

$Sigma $-zero：$\ell_0 $-norm的基于对象的优化对抗性示例 cs.LG

Paper accepted at International Conference on Learning  Representations (ICLR 2025). Code available at  https://github.com/sigma0-advx/sigma-zero

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2402.01879v3) [paper-pdf](http://arxiv.org/pdf/2402.01879v3)

**Authors**: Antonio Emanuele Cinà, Francesco Villani, Maura Pintor, Lea Schönherr, Battista Biggio, Marcello Pelillo

**Abstract**: Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages a differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving robust and non-robust models, show that $\sigma$\texttt{-zero} finds minimum $\ell_0$-norm adversarial examples without requiring any time-consuming hyperparameter tuning, and that it outperforms all competing sparse attacks in terms of success rate, perturbation size, and efficiency.

摘要: 评估深度网络对抗基于梯度的攻击的健壮性是具有挑战性的。虽然大多数攻击考虑$\ell_2$-和$\ell_\inty$-范数约束来手工创建输入扰动，但只有少数攻击研究稀疏的$\ell_1$-和$\ell_0$-范数攻击。特别是，由于在非凸和不可微约束上进行优化的固有复杂性，$\ell_0$-范数攻击仍然是研究最少的。然而，在这些攻击下评估对手的健壮性可能会揭示出在更常规的$\ell_2$-和$\ell_\inty$-范数攻击中未被测试的弱点。在这项工作中，我们提出了一种新的$\ell_0$-范数攻击，称为$\sigma$-零，它利用$\ell_0$范数的可微近似来促进基于梯度的优化，并提出了一种自适应投影算子来动态调整损失最小化和扰动稀疏性之间的权衡。使用MNIST、CIFAR10和ImageNet数据集进行的广泛评估，包括健壮和非健壮模型，表明$\sigma$\Texttt{-Zero}在不需要任何耗时的超参数调整的情况下找到最小$\ell_0$-范数对抗示例，并且在成功率、扰动大小和效率方面优于所有竞争的稀疏攻击。



## **20. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

一次性即可：将多回合攻击整合为LLM的高效单回合攻击 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04856v1) [paper-pdf](http://arxiv.org/pdf/2503.04856v1)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures.

摘要: 尽管在大型语言模型(LLM)中进行了广泛的安全增强，但由熟练的人类对手制作的多轮“越狱”对话仍然可以突破最复杂的护栏。然而，这些多回合攻击需要大量的人工工作，限制了它们的可扩展性。在这项工作中，我们提出了一种新的方法，称为多回合到单回合(M2S)，它系统地将多回合越狱提示转换为单回合攻击。具体地说，我们提出了三种转换策略--Hyhenize、Numberize和Pythonize--每种策略都保留了顺序上下文，但又将其打包在单个查询中。我们在多轮人类越狱(MHJ)数据集上的实验表明，与原始的多轮对话相比，M2通常会增加或保持较高的攻击成功率(ASR)。值得注意的是，使用基于StrongREJECT的危害性评估，M2S在Mistral-7B上获得了高达95.9%的ASR，并且在GPT-40上的绝对改进程度比原来的多转弯提示高达17.5%。进一步的分析表明，某些对抗性策略，当整合到一个提示中时，会利用结构格式线索来逃避标准的政策检查。这些发现突显出，单回合攻击--尽管进行起来更简单、成本更低--可能与多回合攻击一样强大，甚至更多。我们的发现强调了重新评估和加强LLM安全策略的迫切需要，因为敌意查询可以被压缩到单个提示中，同时仍然保持足够的复杂性来绕过现有的安全措施。



## **21. From Pixels to Trajectory: Universal Adversarial Example Detection via Temporal Imprints**

从像素到轨迹：通过时间印记进行通用对抗示例检测 cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04853v1) [paper-pdf](http://arxiv.org/pdf/2503.04853v1)

**Authors**: Yansong Gao, Huaibing Peng, Hua Ma, Zhiyang Dai, Shuo Wang, Hongsheng Hu, Anmin Fu, Minhui Xue

**Abstract**: For the first time, we unveil discernible temporal (or historical) trajectory imprints resulting from adversarial example (AE) attacks. Standing in contrast to existing studies all focusing on spatial (or static) imprints within the targeted underlying victim models, we present a fresh temporal paradigm for understanding these attacks. Of paramount discovery is that these imprints are encapsulated within a single loss metric, spanning universally across diverse tasks such as classification and regression, and modalities including image, text, and audio. Recognizing the distinct nature of loss between adversarial and clean examples, we exploit this temporal imprint for AE detection by proposing TRAIT (TRaceable Adversarial temporal trajectory ImprinTs). TRAIT operates under minimal assumptions without prior knowledge of attacks, thereby framing the detection challenge as a one-class classification problem. However, detecting AEs is still challenged by significant overlaps between the constructed synthetic losses of adversarial and clean examples due to the absence of ground truth for incoming inputs. TRAIT addresses this challenge by converting the synthetic loss into a spectrum signature, using the technique of Fast Fourier Transform to highlight the discrepancies, drawing inspiration from the temporal nature of the imprints, analogous to time-series signals. Across 12 AE attacks including SMACK (USENIX Sec'2023), TRAIT demonstrates consistent outstanding performance across comprehensively evaluated modalities, tasks, datasets, and model architectures. In all scenarios, TRAIT achieves an AE detection accuracy exceeding 97%, often around 99%, while maintaining a false rejection rate of 1%. TRAIT remains effective under the formulated strong adaptive attacks.

摘要: 第一次，我们揭示了可辨别的时间(或历史)轨迹印记产生的对抗性例子(AE)攻击。与现有的研究都集中在目标潜在受害者模型中的空间(或静态)印记相反，我们提出了一个新的时间范式来理解这些攻击。最重要的发现是，这些印记被封装在单个损失度量中，普遍跨越各种任务，如分类和回归，以及包括图像、文本和音频在内的模式。认识到对抗性样本和干净样本之间丢失的不同性质，我们提出了特征(可追踪的对抗性时间轨迹印记)来利用这种时间印记进行声发射检测。特征在没有攻击先验知识的情况下在最小假设下运行，从而将检测挑战框架为一类分类问题。然而，由于输入的输入缺乏基本事实，因此检测AEs仍然受到构建的对抗性示例和干净示例的合成损失之间的显著重叠的挑战。特征通过将合成损耗转换为频谱信号来解决这一挑战，使用快速傅立叶变换技术来突出差异，从印记的时间性质中获得灵感，类似于时间序列信号。在包括SMACK(USENIX SEC‘2023)在内的12个AE攻击中，特征在全面评估的通道、任务、数据集和模型架构中显示出一致的卓越性能。在所有场景中，特征的AE检测准确率都超过97%，通常在99%左右，同时保持1%的误拒率。在制定的强自适应攻击下，特征仍然有效。



## **22. Robust Eavesdropping in the Presence of Adversarial Communications for RF Fingerprinting**

针对RF指纹识别的对抗通信的鲁棒发射 eess.SP

11 pages, 6 figures

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04120v1) [paper-pdf](http://arxiv.org/pdf/2503.04120v1)

**Authors**: Andrew Yuan, Rajeev Sahay

**Abstract**: Deep learning is an effective approach for performing radio frequency (RF) fingerprinting, which aims to identify the transmitter corresponding to received RF signals. However, beyond the intended receiver, malicious eavesdroppers can also intercept signals and attempt to fingerprint transmitters communicating over a wireless channel. Recent studies suggest that transmitters can counter such threats by embedding deep learning-based transferable adversarial attacks in their signals before transmission. In this work, we develop a time-frequency-based eavesdropper architecture that is capable of withstanding such transferable adversarial perturbations and thus able to perform effective RF fingerprinting. We theoretically demonstrate that adversarial perturbations injected by a transmitter are confined to specific time-frequency regions that are insignificant during inference, directly increasing fingerprinting accuracy on perturbed signals intercepted by the eavesdropper. Empirical evaluations on a real-world dataset validate our theoretical findings, showing that deep learning-based RF fingerprinting eavesdroppers can achieve classification performance comparable to the intended receiver, despite efforts made by the transmitter to deceive the eavesdropper. Our framework reveals that relying on transferable adversarial attacks may not be sufficient to prevent eavesdroppers from successfully fingerprinting transmissions in next-generation deep learning-based communications systems.

摘要: 深度学习是进行射频指纹识别的一种有效方法，其目的是识别接收到的射频信号对应的发射机。然而，除了预定的接收方之外，恶意窃听者还可以截获信号并尝试通过无线信道通信指纹传送器。最近的研究表明，发射机可以通过在发射前在其信号中嵌入基于深度学习的可转移对抗性攻击来对抗此类威胁。在这项工作中，我们开发了一种基于时频的窃听者体系结构，它能够抵抗这种可转移的敌意扰动，从而能够执行有效的射频指纹识别。我们从理论上证明了发射机注入的敌意扰动被限制在推理过程中不重要的特定时频区域，从而直接提高了窃听者截获的扰动信号的指纹识别精度。在真实数据集上的实验评估验证了我们的理论发现，基于深度学习的射频指纹窃听者可以获得与预期接收者相当的分类性能，尽管发射机做出了欺骗窃听者的努力。我们的框架揭示，在下一代基于深度学习的通信系统中，依赖于可转移的敌意攻击可能不足以阻止窃听者成功地识别传输的指纹。



## **23. Adversarial Decoding: Generating Readable Documents for Adversarial Objectives**

对抗性解码：为对抗性目标生成可读文档 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.02163v2) [paper-pdf](http://arxiv.org/pdf/2410.02163v2)

**Authors**: Collin Zhang, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: We design, implement, and evaluate adversarial decoding, a new, generic text generation technique that produces readable documents for different adversarial objectives. Prior methods either produce easily detectable gibberish, or cannot handle objectives that include embedding similarity. In particular, they only work for direct attacks (such as jailbreaking) and cannot produce adversarial text for realistic indirect injection, e.g., documents that (1) are retrieved in RAG systems in response to broad classes of queries, and also (2) adversarially influence subsequent generation. We also show that fluency (low perplexity) is not sufficient to evade filtering. We measure the effectiveness of adversarial decoding for different objectives, including RAG poisoning, jailbreaking, and evasion of defensive filters, and demonstrate that it outperforms existing methods while producing readable adversarial documents.

摘要: 我们设计、实现和评估对抗性解码，这是一种新的通用文本生成技术，可以为不同的对抗性目标生成可读文档。先前的方法要么产生容易检测到的胡言乱语，要么无法处理包括嵌入相似性在内的目标。特别是，它们仅适用于直接攻击（例如越狱），并且不能为现实的间接注入生成对抗文本，例如，（1）在RAG系统中检索以响应广泛类别的查询，并且（2）对后继产生不利影响的文档。我们还表明，流畅性（低困惑度）不足以逃避过滤。我们衡量了针对不同目标（包括RAG中毒、越狱和规避防御过滤器）的对抗解码的有效性，并证明它在生成可读的对抗文档的同时优于现有方法。



## **24. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2502.16750v2) [paper-pdf](http://arxiv.org/pdf/2502.16750v2)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **25. Task-Agnostic Attacks Against Vision Foundation Models**

针对愿景基金会模型的任务不可知攻击 cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03842v1) [paper-pdf](http://arxiv.org/pdf/2503.03842v1)

**Authors**: Brian Pulfer, Yury Belousov, Vitaliy Kinakh, Teddy Furon, Slava Voloshynovskiy

**Abstract**: The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models.

摘要: 机器学习中的安全性研究主要集中在针对下游任务的攻击上，通过优化针对下游任务的损失函数来获得对抗性实例。与此同时，机器学习从业者采用公开可用的预先训练的视觉基础模型已成为标准做法，有效地在分类、分割、深度估计、检索、问答等多种应用程序中共享一个通用的骨干架构。对这类基础模型的攻击及其对多个下游任务的影响的研究仍极大地有待探索。这项工作提出了一个通用的框架，通过最大限度地破坏使用基础模型获得的特征表示来伪造与任务无关的对抗性例子。通过测量这种攻击对多个下游任务的影响及其在模型之间的可转移性，我们广泛地评估了流行的视觉基础模型所获得的特征表示的安全性。



## **26. Improving LLM Safety Alignment with Dual-Objective Optimization**

通过双目标优化改善LLM安全一致性 cs.CL

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03710v1) [paper-pdf](http://arxiv.org/pdf/2503.03710v1)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment

摘要: 现有的大型语言模型(LLM)的训练时间安全对齐技术仍然容易受到越狱攻击。直接偏好优化(DPO)是一种被广泛应用的比对方法，由于其损失函数对于拒绝学习来说是次优的，因此在实验和理论环境中都显示出局限性。通过基于梯度的分析，我们识别了这些缺点，并提出了一种改进的安全对齐方法，将DPO目标分解为两个组成部分：(1)稳健的拒绝训练，即使产生部分不安全的生成也鼓励拒绝，以及(2)有针对性地忘记有害知识。这种方法显著提高了LLM对各种越狱攻击的稳健性，包括跨分发内和分发外场景的预填充、后缀和多轮攻击。此外，通过引入基于奖励的拒绝学习令牌级加权机制，我们引入了一种强调关键拒绝令牌的方法，进一步提高了对恶意攻击的鲁棒性。我们的研究还表明，越狱攻击的稳健性与训练过程中令牌分布的变化以及拒绝和有害令牌的内部表征相关，为未来LLM安全匹配的研究提供了有价值的方向。代码可在https://github.com/wicai24/DOOR-Alignment上获得



## **27. CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP**

CLIP足够强大反击：测试时反击CLIP零攻击对抗鲁棒性 cs.CV

Accepted to CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03613v1) [paper-pdf](http://arxiv.org/pdf/2503.03613v1)

**Authors**: Songlong Xing, Zhengyu Zhao, Nicu Sebe

**Abstract**: Despite its prevalent use in image-text matching tasks in a zero-shot manner, CLIP has been shown to be highly vulnerable to adversarial perturbations added onto images. Recent studies propose to finetune the vision encoder of CLIP with adversarial samples generated on the fly, and show improved robustness against adversarial attacks on a spectrum of downstream datasets, a property termed as zero-shot robustness. In this paper, we show that malicious perturbations that seek to maximise the classification loss lead to `falsely stable' images, and propose to leverage the pre-trained vision encoder of CLIP to counterattack such adversarial images during inference to achieve robustness. Our paradigm is simple and training-free, providing the first method to defend CLIP from adversarial attacks at test time, which is orthogonal to existing methods aiming to boost zero-shot adversarial robustness of CLIP. We conduct experiments across 16 classification datasets, and demonstrate stable and consistent gains compared to test-time defence methods adapted from existing adversarial robustness studies that do not rely on external networks, without noticeably impairing performance on clean images. We also show that our paradigm can be employed on CLIP models that have been adversarially finetuned to further enhance their robustness at test time. Our code is available \href{https://github.com/Sxing2/CLIP-Test-time-Counterattacks}{here}.

摘要: 尽管CLIP在图像-文本匹配任务中以零镜头的方式广泛使用，但它被证明对添加到图像上的敌意扰动非常脆弱。最近的研究建议使用动态生成的对抗性样本来微调CLIP的视觉编码器，并显示出对下游数据集频谱上的对抗性攻击的改进的稳健性，这一特性被称为零镜头稳健性。在本文中，我们证明了试图最大化分类损失的恶意扰动会导致图像的虚假稳定，并提出在推理过程中利用CLIP的预先训练的视觉编码器来对抗这种恶意图像以实现稳健性。我们的范例简单且无需训练，提供了第一种在测试时保护CLIP免受对手攻击的方法，这与现有的旨在提高CLIP的零射击对抗健壮性的方法是正交的。我们在16个分类数据集上进行了实验，并证明了与测试时防御方法相比，稳定和一致的收益来自于现有的不依赖外部网络的对抗性研究，而不会明显影响干净图像的性能。我们还表明，我们的范例可以用于经过相反微调的CLIP模型，以进一步增强它们在测试时的健壮性。我们的代码可从\href{https://github.com/Sxing2/CLIP-Test-time-Counterattacks}{here}.获得



## **28. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.04833v1) [paper-pdf](http://arxiv.org/pdf/2503.04833v1)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **29. Adversarial Example Based Fingerprinting for Robust Copyright Protection in Split Learning**

基于对抗示例的指纹识别在分裂学习中实现鲁棒的版权保护 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.04825v1) [paper-pdf](http://arxiv.org/pdf/2503.04825v1)

**Authors**: Zhangting Lin, Mingfu Xue, Kewei Chen, Wenmao Liu, Xiang Gao, Leo Yu Zhang, Jian Wang, Yushu Zhang

**Abstract**: Currently, deep learning models are easily exposed to data leakage risks. As a distributed model, Split Learning thus emerged as a solution to address this issue. The model is splitted to avoid data uploading to the server and reduce computing requirements while ensuring data privacy and security. However, the transmission of data between clients and server creates a potential vulnerability. In particular, model is vulnerable to intellectual property (IP) infringement such as piracy. Alarmingly, a dedicated copyright protection framework tailored for Split Learning models is still lacking. To this end, we propose the first copyright protection scheme for Split Learning model, leveraging fingerprint to ensure effective and robust copyright protection. The proposed method first generates a set of specifically designed adversarial examples. Then, we select those examples that would induce misclassifications to form the fingerprint set. These adversarial examples are embedded as fingerprints into the model during the training process. Exhaustive experiments highlight the effectiveness of the scheme. This is demonstrated by a remarkable fingerprint verification success rate (FVSR) of 100% on MNIST, 98% on CIFAR-10, and 100% on ImageNet, respectively. Meanwhile, the model's accuracy only decreases slightly, indicating that the embedded fingerprints do not compromise model performance. Even under label inference attack, our approach consistently achieves a high fingerprint verification success rate that ensures robust verification.

摘要: 目前，深度学习模型容易面临数据泄露风险。作为一种分布式模型，分裂学习应运而生，成为解决这一问题的一种解决方案。该模型被拆分，以避免数据上传到服务器，并在确保数据隐私和安全的同时降低计算要求。但是，客户端和服务器之间的数据传输会产生潜在的漏洞。特别是，这种模式很容易受到盗版等知识产权(IP)侵犯。令人担忧的是，仍然缺乏专门为分裂学习模式量身定做的版权保护框架。为此，我们提出了第一个适用于分裂学习模型的版权保护方案，利用指纹来确保有效和稳健的版权保护。该方法首先生成一组专门设计的对抗性实例。然后，我们选择那些可能导致误分类的例子来形成指纹集。在训练过程中，这些对抗性的例子作为指纹嵌入到模型中。详尽的实验证明了该方案的有效性。MNIST、CIFAR-10和ImageNet上的指纹验证成功率(FVSR)分别为100%、98%和100%，证明了这一点。同时，模型的准确率仅略有下降，表明嵌入的指纹不会影响模型的性能。即使在标签推理攻击下，我们的方法也始终达到了较高的指纹验证成功率，确保了验证的健壮性。



## **30. AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks**

AttackSeqBench：对大型语言模型对网络攻击中序列模式的理解进行基准测试 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03170v1) [paper-pdf](http://arxiv.org/pdf/2503.03170v1)

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at https://github.com/Javiery3889/AttackSeqBench .

摘要: 网络威胁情报(CTI)报告中记录的观察结果在描述敌对行为方面发挥了关键作用，为安全从业者提供了宝贵的见解，以应对不断变化的威胁。大型语言模型的最新进展在各种网络安全应用中显示出巨大的潜力，包括CTI报告理解和攻击知识图的构建。虽然前人的研究主要集中在低层统计模型的CTI提取能力上，但CTI报告中敌方行为的时序特征在很大程度上还没有被探索，这对于全面理解敌方是如何运作的具有相当重要的意义。为了弥补这一差距，我们引入了AttackSeqBtch，这是一个专门为系统评估LLMS理解和推理CTI报告中的攻击序列的能力而定制的基准测试。我们的基准包括三个不同的问答(QA)任务，每个任务都专注于敌对行为中不同的粒度。为了减轻QA构建的繁重工作，我们精心设计了一个自动化的数据集构建管道，以真实世界的CTI报告为基础创建可扩展的、格式良好的QA数据集。为了确保我们的数据集的质量，我们采用了人工评估和系统评估度量相结合的混合方法。我们使用快速思维和缓慢思维的LLM进行了广泛的实验和分析，同时强调了它们在分析网络攻击中的序列模式方面的优势和局限性。这项工作的总体目标是提供一个基准，以促进LLM驱动的CTI报告的理解，并促进其在现实世界网络安全操作中的应用。我们的数据集和代码可在https://github.com/Javiery3889/AttackSeqBench上获得。



## **31. Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks**

通过定向对抗攻击利用语音翻译系统中的漏洞 cs.SD

Preprint,17 pages, 17 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.00957v2) [paper-pdf](http://arxiv.org/pdf/2503.00957v2)

**Authors**: Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at https://adv-st.github.io.

摘要: 随着语音翻译(ST)系统变得越来越普遍，了解它们的漏洞对于确保健壮可靠的通信至关重要。然而，深入探讨这一问题的工作有限。本文探索了通过不可感知的音频操作来危害这些系统的方法。具体地说，我们提出了两种创新的方法：(1)在源音频中注入扰动，(2)生成对抗性音乐，旨在指导定向翻译，同时也在物理世界中进行更实际的空中攻击。我们的实验表明，精心制作的音频扰动可能会误导翻译模型产生目标明确的有害输出，而对抗性音乐则更隐蔽地实现这一目标，利用音乐的自然不可感知性。事实证明，这些攻击在多种语言和翻译模型中都有效，突显了当前ST架构中的系统性漏洞。这项研究的意义超越了直接的安全问题，揭示了神经语音处理系统的可解释性和稳健性。我们的发现强调了音频系统领域需要先进的防御机制和更具弹性的架构。欲了解更多详情和样品，请访问https://adv-st.github.io.。



## **32. Detecting Adversarial Data using Perturbation Forgery**

使用微扰伪造检测对抗数据 cs.CV

Accepted as a conference paper at CVPR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2405.16226v4) [paper-pdf](http://arxiv.org/pdf/2405.16226v4)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Shijuan Huang, Ruoxi Jia, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, the significant inference time overhead and limited performance against unseen attacks make existing techniques impractical for real-world use. In this paper, we explore the proximity relationship among adversarial noise distributions and demonstrate the existence of an open covering for these distributions. By training on the open covering of adversarial noise distributions, a detector with strong generalization performance against various types of unseen attacks can be developed. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting any unseen gradient-based, generative-based, and physical adversarial attacks. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.

摘要: 敌意检测是针对敌意攻击的一种防御策略，其目的是根据自然数据和敌意数据之间的分布和噪声模式的差异，从数据流中识别和过滤敌意数据。虽然以前的检测方法在检测基于梯度的敌意攻击方面取得了较高的性能，但基于非平衡和各向异性噪声模式的生成模型的新攻击可以逃避检测。更糟糕的是，巨大的推理时间开销和有限的针对看不见的攻击的性能使得现有技术在现实世界中不切实际。本文研究了对抗性噪声分布之间的邻近关系，并证明了这些分布的开覆盖的存在性。通过对对抗性噪声分布的开放覆盖进行训练，可以开发出对各种类型的不可见攻击具有很强的泛化性能的检测器。基于这一观点，我们启发式地提出了扰动伪造，它包括噪声分布扰动、稀疏掩码生成和伪对抗数据生成，以训练一个能够检测任何不可见的基于梯度的、基于生成的和物理的对抗攻击的对抗检测器。在多个普通数据集和人脸数据集上进行的综合实验表明，该方法具有较强的泛化能力。



## **33. An Undetectable Watermark for Generative Image Models**

生成图像模型的不可检测水印 cs.CR

ICLR 2025

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2410.07369v3) [paper-pdf](http://arxiv.org/pdf/2410.07369v3)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.

摘要: 我们提出了第一个不可检测的生成图像模型的水印方案。不可检测性确保了即使在进行了许多自适应查询之后，有效的攻击者也无法区分加水印和未加水印的图像。特别是，不可检测的水印在任何有效计算的度量下都不会降低图像质量。我们的方案通过使用伪随机纠错码(Christian和Gunn，2024)来选择扩散模型的初始潜伏期，这是一种保证不可检测性和稳健性的策略。实验证明，利用稳定扩散2.1算法，水印具有较好的保质性和稳健性。我们的实验证明，与我们测试的每个方案相比，我们的水印不会降低图像质量。我们的实验也证明了我们的稳健性：现有的水印去除攻击不能在不显著降低图像质量的情况下去除图像中的水印。最后，我们发现我们的水印可以稳健地编码512比特，当图像没有受到水印去除攻击时，可以编码高达2500比特。我们的代码可以在https://github.com/XuandongZhao/PRC-Watermark.上找到



## **34. LLM Misalignment via Adversarial RLHF Platforms**

对抗性LLHF平台的LLM失调 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.03039v1) [paper-pdf](http://arxiv.org/pdf/2503.03039v1)

**Authors**: Erfan Entezami, Ali Naseh

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process.

摘要: 强化学习在将语言模型与人类偏好保持一致方面表现出了显著的性能，导致了人们对开发RLHF平台的关注。这些平台使用户能够微调模型，而不需要开发复杂的机器学习算法的任何专业知识。虽然这些平台提供了有用的功能，如奖励建模和RLHF微调，但它们的安全性和可靠性在很大程度上仍未得到探索。鉴于RLHF和开源RLHF框架越来越多地被采用，我们调查了这些系统的可信性及其对LLM行为的潜在影响。本文提出了一种针对公开可用的RLHF工具的攻击。在我们提出的攻击中，敌意的RLHF平台通过选择性地操纵偏好数据集中的数据样本来破坏LLM比对过程。在这种情况下，当用户的任务与攻击者的目标一致时，平台操作包含与攻击者目标相关的样本的首选项数据集的子集。这种操作会导致奖励模型被破坏，这最终会导致语言模型的不一致。我们的结果表明，这样的攻击可以有效地将LLM引向目标域内的不良行为。我们的工作突出了探索RLHF平台的脆弱性及其在RLHF微调过程中导致LLM未对准的可能性的迫切需要。



## **35. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

注意差距：通过查询更新分析检测正在形成的黑匣子对抗攻击 cs.CR

13 pages

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.02986v2) [paper-pdf](http://arxiv.org/pdf/2503.02986v2)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.

摘要: 对抗性攻击仍然是一个严重的威胁，可能会危及机器学习(ML)模型的完整性。特别是，基于查询的黑盒攻击可以在不访问受害者模型的体系结构的情况下生成恶意噪声，使它们在真实世界的上下文中具有实用性。社区已经提出了几种针对对抗性攻击的防御措施，但都被更先进和适应性更强的攻击策略打破了。在这篇文章中，我们提出了一个框架，它检测是否正在生成对抗性噪声实例。与现有的通过监测输入空间来检测对抗性噪声产生的状态防御方法不同，我们的方法在输入更新相似性空间中学习对抗性模式。事实上，我们提出了一种新的度量，称为Delta相似度(DS)，我们表明它更有效地捕获了对手的行为。我们评估了我们的方法针对8种最先进的攻击，包括自适应攻击，在这些攻击中，对手知道防御并试图逃避检测。我们发现，我们的方法在特异性和敏感性方面都明显比现有的防御方法更稳健。



## **36. Decentralized Adversarial Training over Graphs**

图上的分散对抗训练 cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2303.13326v2) [paper-pdf](http://arxiv.org/pdf/2303.13326v2)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of distributed learning, we develop a decentralized adversarial training framework for multi-agent systems. Specifically, we devise two decentralized adversarial training algorithms by relying on two popular decentralized learning strategies--diffusion and consensus. We analyze the convergence properties of the proposed framework for strongly-convex, convex, and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.

摘要: 近年来，机器学习模型对敌意攻击的脆弱性引起了人们的极大关注。现有的研究大多集中在单智能体学习者的行为上。相比之下，这项工作研究的是图上的对抗性训练，在图中，单个代理人受到空间上不同强度水平的扰动。考虑到组的协调能力，预计链接代理的交互以及图上可能的攻击模型的异构性可以帮助增强稳健性。利用分布式学习的最小-最大公式，我们提出了一个多智能体系统的分布式对抗训练框架。具体地说，我们依靠两种流行的去中心化学习策略--扩散和共识，设计了两种去中心化对抗训练算法。我们分析了该框架在强凸、凸和非凸环境下的收敛性质，并说明了该框架增强了对敌意攻击的鲁棒性。



## **37. Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare**

迈向安全的人工智能临床医生：医疗保健领域大语言模型越狱的综合研究 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.18632v2) [paper-pdf](http://arxiv.org/pdf/2501.18632v2)

**Authors**: Hang Zhang, Qian Lou, Yanshan Wang

**Abstract**: Large language models (LLMs) are increasingly utilized in healthcare applications. However, their deployment in clinical practice raises significant safety concerns, including the potential spread of harmful information. This study systematically assesses the vulnerabilities of seven LLMs to three advanced black-box jailbreaking techniques within medical contexts. To quantify the effectiveness of these techniques, we propose an automated and domain-adapted agentic evaluation pipeline. Experiment results indicate that leading commercial and open-source LLMs are highly vulnerable to medical jailbreaking attacks. To bolster model safety and reliability, we further investigate the effectiveness of Continual Fine-Tuning (CFT) in defending against medical adversarial attacks. Our findings underscore the necessity for evolving attack methods evaluation, domain-specific safety alignment, and LLM safety-utility balancing. This research offers actionable insights for advancing the safety and reliability of AI clinicians, contributing to ethical and effective AI deployment in healthcare.

摘要: 大型语言模型(LLM)越来越多地用于医疗保健应用程序。然而，它们在临床实践中的部署引起了重大的安全担忧，包括有害信息的潜在传播。这项研究系统地评估了七种低密度脂蛋白对三种先进的黑盒越狱技术在医学背景下的脆弱性。为了量化这些技术的有效性，我们提出了一个自动化的和领域适应的代理评估管道。实验结果表明，领先的商业和开源LLM非常容易受到医疗越狱攻击。为了支持模型的安全性和可靠性，我们进一步研究了连续微调(CFT)在防御医疗对手攻击方面的有效性。我们的发现强调了对不断发展的攻击方法进行评估、特定领域的安全对齐和LLM安全效用平衡的必要性。这项研究为提高人工智能临床医生的安全性和可靠性提供了可操作的见解，有助于在医疗保健领域进行合乎道德和有效的人工智能部署。



## **38. Assessing Robustness via Score-Based Adversarial Image Generation**

通过基于分数的对抗图像生成评估稳健性 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2310.04285v3) [paper-pdf](http://arxiv.org/pdf/2310.04285v3)

**Authors**: Marcel Kollovieh, Lukas Gosch, Marten Lienen, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantics-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate unrestricted adversarial examples that overcome the limitations of $\ell_p$-norm constraints. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG improves upon the majority of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.

摘要: 大多数对抗性攻击和防御都集中在较小的$\ell_p$-范数约束内的扰动。然而，$\ell_p$威胁模型不能捕获所有相关的语义保持扰动，因此健壮性评估的范围是有限的。在这项工作中，我们介绍了基于分数的对抗性生成(ScoreAG)，这是一个新的框架，它利用基于分数的生成模型的进步来生成不受限制的对抗性实例，克服了$\ell_p$-范数约束的限制。与传统方法不同，ScoreAG在生成敌意示例的同时保持了图像的核心语义，要么转换现有图像，要么完全从头开始合成新的图像。我们进一步利用ScoreAG的生成能力来净化图像，经验上增强了分类器的稳健性。我们广泛的经验评估表明，ScoreAG在多个基准上改进了大多数最先进的攻击和防御。这项工作强调了研究受语义约束的对抗性例子的重要性，而不是$\ell_p$-范数约束。ScoreAG代表着朝着更全面的稳健性评估迈出的重要一步。



## **39. Realizing Quantum Adversarial Defense on a Trapped-ion Quantum Processor**

在俘获离子量子处理器上实现量子对抗防御 quant-ph

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02436v1) [paper-pdf](http://arxiv.org/pdf/2503.02436v1)

**Authors**: Alex Jin, Tarun Dutta, Anh Tu Ngo, Anupam Chattopadhyay, Manas Mukherjee

**Abstract**: Classification is a fundamental task in machine learning, typically performed using classical models. Quantum machine learning (QML), however, offers distinct advantages, such as enhanced representational power through high-dimensional Hilbert spaces and energy-efficient reversible gate operations. Despite these theoretical benefits, the robustness of QML classifiers against adversarial attacks and inherent quantum noise remains largely under-explored. In this work, we implement a data re-uploading-based quantum classifier on an ion-trap quantum processor using a single qubit to assess its resilience under realistic conditions. We introduce a novel convolutional quantum classifier architecture leveraging data re-uploading and demonstrate its superior robustness on the MNIST dataset. Additionally, we quantify the effects of polarization noise in a realistic setting, where both bit and phase noises are present, further validating the classifier's robustness. Our findings provide insights into the practical security and reliability of quantum classifiers, bridging the gap between theoretical potential and real-world deployment.

摘要: 分类是机器学习中的一项基本任务，通常使用经典模型执行。然而，量子机器学习(QML)提供了独特的优势，例如通过高维希尔伯特空间和节能的可逆门操作来增强表征能力。尽管有这些理论上的好处，但QML分类器对敌意攻击和固有的量子噪声的稳健性仍未得到很大程度的研究。在这项工作中，我们使用单个量子比特在离子陷阱量子处理器上实现了一个基于数据重传的量子分类器，以评估其在现实条件下的弹性。我们介绍了一种利用数据重传的卷积量子分类器结构，并在MNIST数据集上展示了其优越的健壮性。此外，我们在实际环境中量化了极化噪声的影响，其中位噪声和相位噪声都存在，进一步验证了分类器的稳健性。我们的发现为量子分类器的实际安全性和可靠性提供了洞察力，弥合了理论潜力和现实部署之间的差距。



## **40. Trace of the Times: Rootkit Detection through Temporal Anomalies in Kernel Activity**

时代的痕迹：通过核心活动中的时间异常进行RootKit检测 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02402v1) [paper-pdf](http://arxiv.org/pdf/2503.02402v1)

**Authors**: Max Landauer, Leonhard Alton, Martina Lindorfer, Florian Skopik, Markus Wurzenberger, Wolfgang Hotwagner

**Abstract**: Kernel rootkits provide adversaries with permanent high-privileged access to compromised systems and are often a key element of sophisticated attack chains. At the same time, they enable stealthy operation and are thus difficult to detect. Thereby, they inject code into kernel functions to appear invisible to users, for example, by manipulating file enumerations. Existing detection approaches are insufficient, because they rely on signatures that are unable to detect novel rootkits or require domain knowledge about the rootkits to be detected. To overcome this challenge, our approach leverages the fact that runtimes of kernel functions targeted by rootkits increase when additional code is executed. The framework outlined in this paper injects probes into the kernel to measure time stamps of functions within relevant system calls, computes distributions of function execution times, and uses statistical tests to detect time shifts. The evaluation of our open-source implementation on publicly available data sets indicates high detection accuracy with an F1 score of 98.7\% across five scenarios with varying system states.

摘要: 内核Rootkit为攻击者提供了对受攻击系统的永久高权限访问权限，通常是复杂攻击链的关键元素。同时，它们实现了隐形操作，因此很难被检测到。因此，它们向内核函数注入代码，使其对用户不可见，例如，通过操作文件枚举。现有的检测方法是不够的，因为它们依赖于不能检测到新的Rootkit或需要关于Rootkit的领域知识来检测的签名。为了克服这一挑战，我们的方法利用了这样一个事实，即Rootkit所针对的内核函数的运行时间会随着额外代码的执行而增加。本文概述的框架将探测器注入内核，以测量相关系统调用中函数的时间戳，计算函数执行时间的分布，并使用统计测试来检测时间偏移。在公开可用的数据集上对我们的开源实现进行的评估表明，在五个不同系统状态的场景中，检测准确率很高，F1得分为98.7\%。



## **41. Evaluating the Robustness of LiDAR Point Cloud Tracking Against Adversarial Attack**

评估LiDART点云跟踪对抗攻击的鲁棒性 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2410.20893v2) [paper-pdf](http://arxiv.org/pdf/2410.20893v2)

**Authors**: Shengjing Tian, Yinan Han, Xiantong Zhao, Bin Liu, Xiuping Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.

摘要: 在这项研究中，我们深入研究了基于神经网络的LiDAR点云跟踪模型在对抗攻击下的稳健性，这是一个经常被忽略的关键方面，有助于提高性能。尽管这些模型集成了Transformer或Bird‘s Eye View(BEV)等高级架构，但在面临对手攻击、域转移或数据损坏等挑战时往往忽略了健壮性。相反，我们关注的是跟踪模型在对抗性攻击威胁下的健壮性。我们首先建立一个统一的框架，在3D对象跟踪的背景下进行对抗性攻击，这使我们能够彻底调查白盒和黑盒攻击策略。对于白盒攻击，我们定制了特定的损失函数以适应不同的跟踪范例，并将现有的方法如FGSM、C\&W和PGD扩展到点云域。针对黑盒攻击场景，我们引入了一种新的基于传输的方法，目标感知扰动生成(TAPG)算法，其双重目标是实现高攻击性能和保持低可感知性。该方法使用启发式策略来实施稀疏攻击约束，并利用随机子向量分解来增强可转移性。我们的实验结果揭示了高级跟踪方法在同时受到黑盒和白盒攻击时的显著漏洞，强调了在设计LiDAR点云跟踪模型时考虑对对手攻击的健壮性的必要性。值得注意的是，与现有方法相比，TAPG还在攻击的有效性和扰动的隐蔽性之间取得了最佳平衡。



## **42. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

NoPain：通过最佳传输奇异边界进行无箱点云攻击 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00063v2) [paper-pdf](http://arxiv.org/pdf/2503.00063v2)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. The source code will be publicly available.

摘要: 对抗性攻击利用深度模型针对对抗性样本的脆弱性。现有的点云攻击者是为特定模型量身定做的，基于白盒或黑盒设置中的渐变迭代优化扰动。尽管它们的攻击性能很有希望，但由于代理模型的特定参数过高，它们经常难以产生可转移的对抗性样本。为了解决这一问题，我们将重点转移到数据分布本身，并引入了一种名为NoPain的新方法，该方法使用最优传输(OT)来识别跨网络点云攻击数据流形的固有奇异边界。具体地，我们首先计算噪声到目标特征空间的OT映射，然后通过定位不可微位置来识别奇异边界。最后，我们沿着奇异边界进行采样以生成对抗性点云。一旦确定了奇异边界，NoPain就可以有效地产生对抗性样本，而不需要迭代更新或来自代理分类器的指导。大量的实验表明，端到端方法在可转移性和效率方面都优于基线方法，同时在与防御策略相比也保持了显著的优势。源代码将向公众开放。



## **43. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

针对医学成像中对抗攻击的鲁棒共形预测的游戏理论防御 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2411.04376v2) [paper-pdf](http://arxiv.org/pdf/2411.04376v2)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.

摘要: 对抗性攻击对深度学习模型的可靠性和安全性构成了严重威胁，特别是在医学成像等关键领域。本文介绍了一种新的框架，它将保形预测与博弈论防御策略相结合，以增强模型对已知和未知对手扰动的稳健性。我们主要研究了三个问题：在已知攻击(RQ1)下构造有效且高效的共形预测集，通过保守阈值(RQ2)确保未知攻击下的覆盖，以及在零和博弈框架(RQ3)下确定最优防御策略。我们的方法包括针对特定的攻击类型训练专门的防御模型，并使用最大和最小分类器来有效地聚合防御。在包括PathMNIST、OrganAMNIST和TIseMNIST在内的MedMNIST数据集上进行的大量实验表明，我们的方法在保持高覆盖率的同时最小化了预测集的大小。博弈论分析表明，最优防御策略往往收敛到一个奇异的稳健模型，在所有评估的数据集上表现优于统一和简单的策略。这项工作推进了不确定性量化和对抗稳健性方面的最新进展，为在对抗环境中部署深度学习模型提供了可靠的机制。



## **44. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.

摘要: 最近，面向代码的大型语言模型(Code LLM)已经被广泛并成功地利用来简化和促进编程。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。在本文中，我们揭示了这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力，这可能是不实用的；对抗性攻击难以实现特定的恶意目的。针对这些问题，提出了一种新的针对代码LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅使用12个令牌的非功能扰动，我们的TPIA就可以成功攻击三个典型的开源代码LLM(攻击成功率高达97.9%)和两个主流商业代码LLM集成应用(攻击成功率超过90%)。



## **45. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

基于属性感知文本倒置的预算驱动可转移对抗攻击 cs.CV

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2502.19697v2) [paper-pdf](http://arxiv.org/pdf/2502.19697v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.

摘要: 人员再识别(re-id)模型在安全监控系统中是至关重要的，需要可转移的对抗性攻击来探索其脆弱性。近年来，基于视觉语言模型(VLM)的攻击通过攻击VLM的广义图像和文本特征表现出了良好的可转移性，但由于过于强调积分表示的区分语义，缺乏全面的特征破坏。本文介绍了一种新的基于属性感知的提示攻击(AP-Attack)，该方法利用VLM的图文对齐能力，通过破坏特定于属性的文本嵌入来显式破坏行人图像的细粒度语义特征。为了获得个性化的属性文本描述，文本倒置网络被设计成将行人图像映射到代表语义嵌入的伪标记，并利用图像和预定义的提示模板进行对比学习，以明确描述行人属性。反转良性和对抗性细粒度的文本语义有助于攻击者有效地进行彻底的破坏，增强了对抗性实例的可转移性。大量实验表明，AP-Attack具有最好的可转移性，在跨模型和数据集攻击场景下，平均丢失率比以前的方法高出22.9%。



## **46. Adversarial Tokenization**

对抗性代币化 cs.CL

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02174v1) [paper-pdf](http://arxiv.org/pdf/2503.02174v1)

**Authors**: Renato Lui Geh, Zilei Shao, Guy Van den Broeck

**Abstract**: Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models.

摘要: 当前的LLM流水线只考虑了给定字符串的一个可能的标记化，在训练和推理期间忽略了指数级的许多替代标记化。例如，企鹅的标准Llama3标记化是[p，enguin]，然而[peng，uin]是另一个完全有效的替代方案。在这篇文章中，我们证明了尽管LLM只接受了一个标记化的训练，但它们仍然保留了对其他标记化的语义理解，这引发了人们对它们在LLM安全性方面的影响的疑问。简而言之，我们回答了以下问题：我们可以恶意地对一个明显恶意的字符串进行标记，以逃避安全和对齐限制吗？我们表明，对抗性标记化不仅是一个以前被忽视的有效攻击轴心，而且在不改变有害请求的案文的情况下，也是与现有最先进的对抗性方法相竞争的。我们在三个最先进的LLM和对抗性数据集上实证验证了这一利用，揭示了子词模型中一个以前未知的漏洞。



## **47. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

HoSNN：具有自适应激发阈值的对抗鲁棒自适应尖峰神经网络 cs.NE

Accepted by TMLR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2308.10373v4) [paper-pdf](http://arxiv.org/pdf/2308.10373v4)

**Authors**: Hejia Geng, Peng Li

**Abstract**: While spiking neural networks (SNNs) offer a promising neurally-inspired model of computation, they are vulnerable to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to design a threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model and utilize TA-LIF neurons to construct the adversarially robust homeostatic SNNs (HoSNNs) for improved robustness. The TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, offering a local feedback control solution to the minimization of each neuron's membrane potential error caused by adversarial disturbance. Theoretical analysis demonstrates favorable dynamic properties of TA-LIF neurons in terms of the bounded-input bounded-output stability and suppressed time growth of membrane potential error, underscoring their superior robustness compared with the standard LIF neurons. When trained with weak FGSM attacks (attack budget = 2/255) and tested with much stronger PGD attacks (attack budget = 8/255), our HoSNNs significantly improve model accuracy on several datasets: from 30.54% to 74.91% on FashionMNIST, from 0.44% to 35.06% on SVHN, from 0.56% to 42.63% on CIFAR10, from 0.04% to 16.66% on CIFAR100, over the conventional LIF-based SNNs.

摘要: 虽然尖峰神经网络(SNN)提供了一种很有前途的神经启发计算模型，但它们容易受到对手的攻击。我们首次从神经元自平衡中得到启发，设计了一种阈值自适应泄漏积分与点火(TA-LIF)神经元模型，并利用TA-LIF神经元来构造逆稳健的自平衡SNN(HoSNN)，以提高鲁棒性。TA-LIF模型引入了一种自稳定的动态阈值机制，提供了一种局部反馈控制方案来最小化对抗性干扰引起的每个神经元的膜电位误差。理论分析表明，TA-LIF神经元在有界输入有界输出稳定性和抑制膜电位误差的时间增长方面具有良好的动态特性，与标准LIF神经元相比具有更好的鲁棒性。当用弱FGSM攻击(攻击预算=2/255)训练和用更强的PGD攻击(攻击预算=8/255)测试时，我们的HoSNN在几个数据集上显著提高了模型准确率：在FashionMNIST上从30.54%到74.91%，在SVHN上从0.44%到35.06%，在CIFAR10上从0.56%到42.63%，在CIFAR100上从0.04%到16.66%，比传统的基于LIF的SNN。



## **48. DDAD: A Two-pronged Adversarial Defense Based on Distributional Discrepancy**

DDAD：基于分配差异的双管齐下的对抗防御 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02169v1) [paper-pdf](http://arxiv.org/pdf/2503.02169v1)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we reveal the potential strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Nevertheless, despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of clean information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-Discrepancy-based Adversarial Defense (DDAD). In the training phase, DDAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, and then trains a denoiser by minimizing the MMD-OPT between CEs and AEs. In the inference phase, DDAD first leverages MMD-OPT to differentiate CEs and AEs, and then applies a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DDAD outperforms current state-of-the-art (SOTA) defense methods by notably improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks.

摘要: 统计对抗性数据检测(SADD)通过测量干净样本(CE)和对抗性样本(AE)之间的分布差异来检测即将到来的一批是否包含对抗性样本(AE)。在本文中，我们揭示了基于SADD的方法的潜在优势，从理论上证明了最小化分布差异可以帮助减少AEs的预期损失。然而，尽管有这些优点，基于SADD的方法有一个潜在的局限性：它们丢弃被检测为AE的输入，导致在这些输入中丢失干净的信息。针对这一局限性，我们提出了一种双管齐下的对抗性防御方法，称为基于分布差异的对抗性防御(DDAD)。在训练阶段，DDAD首先优化最大平均偏差(MMD)的测试功率得到MMD-OPT，然后通过最小化CES和AES之间的MMD-OPT来训练去噪器。在推理阶段，DDAD首先利用MMD-OPT来区分CE和AE，然后采用双管齐下的方法：(1)直接将检测到的CE送入分类器；(2)通过基于分布差异的去噪器去除检测到的AE中的噪声。大量的实验表明，DDAD在CIFAR-10和ImageNet-1K上显著提高了对自适应白盒攻击的干净和健壮的准确性，从而超过了当前最先进的(SOTA)防御方法。



## **49. Towards Scalable Topological Regularizers**

迈向可扩展的布局调节器 cs.LG

31 pages, ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.14641v2) [paper-pdf](http://arxiv.org/pdf/2501.14641v2)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.

摘要: 潜在空间匹配由潜在空间中特征的匹配分布组成，是对抗性攻击和防御、领域自适应和产生式建模等任务的重要组成部分。概率度量指标，如Wasserstein和最大均值差异，通常用于量化此类分布之间的差异。然而，这些通常计算成本很高，或者没有适当地考虑分布的几何和拓扑特征。持久同调是一种从拓扑数据分析中量化点云多尺度拓扑结构的工具，近年来被用作学习任务中的拓扑正则化工具。然而，计算成本排除了更大规模的计算，并且梯度中的不连续会导致不稳定的训练行为，例如在对抗性任务中。在计算大量小样本的持久同调的基础上，我们提出使用主持久度量作为拓扑正则化。我们给出了这种正则化算法的并行GPU实现，并证明了对于光滑的密度，梯度是连续的。此外，我们展示了这种正则化算法在形状匹配、图像生成和半监督学习任务中的有效性，为拓扑特征的可伸缩正则化算法打开了大门。



## **50. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

通过大型语言模型越狱受保护的文本到图像模型 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.

摘要: 文本到图像模型可能会生成有害内容，例如色情图像，特别是在提交不安全提示时。为了解决这个问题，通常在文本到图像模型之上添加安全过滤器，或者对模型本身进行调整以减少有害输出。然而，当攻击者战略性地设计对抗提示来绕过这些安全护栏时，这些防御仍然容易受到攻击。在这项工作中，我们提出了ObjetTune，这是一种使用微调的大型语言模型来越狱具有安全护栏的文本到图像模型的方法。与其他需要对目标模型重复查询的基于查询的越狱攻击不同，我们的攻击在微调AttackLLM后有效地生成对抗提示。我们在三个不安全提示数据集和五个安全护栏上评估我们的方法。我们的结果表明，我们的方法有效地绕过了安全护栏，优于现有的无框攻击，并且还促进了其他基于查询的攻击。



