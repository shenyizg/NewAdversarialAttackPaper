# Latest Large Language Model Attack Papers
**update at 2025-12-01 09:05:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Constructing and Benchmarking: a Labeled Email Dataset for Text-Based Phishing and Spam Detection Framework**

构建和基准测试：基于文本的网络钓鱼和垃圾邮件检测框架的标签电子邮件数据集 cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21448v1) [paper-pdf](https://arxiv.org/pdf/2511.21448v1)

**Authors**: Rebeka Toth, Tamas Bisztray, Richard Dubniczky

**Abstract**: Phishing and spam emails remain a major cybersecurity threat, with attackers increasingly leveraging Large Language Models (LLMs) to craft highly deceptive content. This study presents a comprehensive email dataset containing phishing, spam, and legitimate messages, explicitly distinguishing between human- and LLM-generated content. Each email is annotated with its category, emotional appeal (e.g., urgency, fear, authority), and underlying motivation (e.g., link-following, credential theft, financial fraud). We benchmark multiple LLMs on their ability to identify these emotional and motivational cues and select the most reliable model to annotate the full dataset. To evaluate classification robustness, emails were also rephrased using several LLMs while preserving meaning and intent. A state-of-the-art LLM was then assessed on its performance across both original and rephrased emails using expert-labeled ground truth. The results highlight strong phishing detection capabilities but reveal persistent challenges in distinguishing spam from legitimate emails. Our dataset and evaluation framework contribute to improving AI-assisted email security systems. To support open science, all code, templates, and resources are available on our project site.

摘要: 网络钓鱼和垃圾电子邮件仍然是一个主要的网络安全威胁，攻击者越来越多地利用大型语言模型（LLM）来制作高度欺骗性的内容。这项研究提供了一个全面的电子邮件数据集，其中包含网络钓鱼、垃圾邮件和合法消息，明确区分了人类生成的内容和LLM生成的内容。每封电子邮件都注释了其类别、情感吸引力（例如，紧迫性、恐惧、权威）和潜在动机（例如，链接跟踪、凭证盗窃、财务欺诈）。我们根据多个LLM识别这些情感和动机线索的能力对它们进行基准测试，并选择最可靠的模型来注释完整数据集。为了评估分类稳健性，还使用多个LLM重新措辞了电子邮件，同时保留含义和意图。然后，使用专家标记的基本真相评估了最先进的LLM在原始和重新措辞的电子邮件中的表现。结果凸显了强大的网络钓鱼检测能力，但也揭示了区分垃圾邮件与合法电子邮件的持续挑战。我们的数据集和评估框架有助于改进人工智能辅助电子邮件安全系统。为了支持开放科学，所有代码、模板和资源均可在我们的项目网站上获取。



## **2. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

对抗性混乱攻击：扰乱多模式大型语言模型 cs.CL

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20494v1) [paper-pdf](https://arxiv.org/pdf/2511.20494v1)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.

摘要: 我们引入了对抗性混乱攻击，这是针对多模式大型语言模型（MLLM）的一类新型威胁。与越狱或有针对性的错误分类不同，目标是引发系统性破坏，使模型生成不连贯或自信地错误的输出。应用程序包括将对抗图像嵌入到网站中，以防止MLLM支持的代理可靠运行。拟议的攻击使用一小部分开源MLLM来最大化下一个令牌的熵。在白盒设置中，我们表明，单个对抗图像可以扰乱集合中的所有模型，无论是在完整图像还是对抗验证码设置中。尽管依赖于基本的对抗技术（PVD），但攻击会产生转移到两个看不见的开源的扰动（例如，Qwen 3-DL）和专有（例如，GPT-5.1）型号。



## **3. APT-CGLP: Advanced Persistent Threat Hunting via Contrastive Graph-Language Pre-Training**

APT-CGLP：通过对比图语言预训练的高级持续威胁搜索 cs.CR

Accepted by SIGKDD 2026 Research Track

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20290v1) [paper-pdf](https://arxiv.org/pdf/2511.20290v1)

**Authors**: Xuebo Qiu, Mingqi Lv, Yimei Zhang, Tieming Chen, Tiantian Zhu, Qijie Song, Shouling Ji

**Abstract**: Provenance-based threat hunting identifies Advanced Persistent Threats (APTs) on endpoints by correlating attack patterns described in Cyber Threat Intelligence (CTI) with provenance graphs derived from system audit logs. A fundamental challenge in this paradigm lies in the modality gap -- the structural and semantic disconnect between provenance graphs and CTI reports. Prior work addresses this by framing threat hunting as a graph matching task: 1) extracting attack graphs from CTI reports, and 2) aligning them with provenance graphs. However, this pipeline incurs severe \textit{information loss} during graph extraction and demands intensive manual curation, undermining scalability and effectiveness.   In this paper, we present APT-CGLP, a novel cross-modal APT hunting system via Contrastive Graph-Language Pre-training, facilitating end-to-end semantic matching between provenance graphs and CTI reports without human intervention. First, empowered by the Large Language Model (LLM), APT-CGLP mitigates data scarcity by synthesizing high-fidelity provenance graph-CTI report pairs, while simultaneously distilling actionable insights from noisy web-sourced CTIs to improve their operational utility. Second, APT-CGLP incorporates a tailored multi-objective training algorithm that synergizes contrastive learning with inter-modal masked modeling, promoting cross-modal attack semantic alignment at both coarse- and fine-grained levels. Extensive experiments on four real-world APT datasets demonstrate that APT-CGLP consistently outperforms state-of-the-art threat hunting baselines in terms of accuracy and efficiency.

摘要: 基于源的威胁狩猎通过将网络威胁情报（RTI）中描述的攻击模式与从系统审计日志中获得的源源图关联来识别端点上的高级持续性威胁（APT）。该范式的一个根本挑战在于形式差距--出处图和RTI报告之间的结构和语义脱节。之前的工作通过将威胁狩猎框架为图匹配任务来解决这个问题：1）从RTI报告中提取攻击图，以及2）将它们与出处图对齐。然而，该管道在图形提取过程中会导致严重的\textit{信息损失}，并且需要密集的手动策展，从而削弱了可扩展性和有效性。   在本文中，我们介绍了APT-CGM，这是一种通过对比图语言预训练的新型跨模式APT狩猎系统，可以在无需人为干预的情况下促进出处图和RTI报告之间的端到端语义匹配。首先，在大型语言模型（LLM）的支持下，APT-CGLP通过合成高保真出处图形-RTI报告对来缓解数据稀缺性，同时从有噪音的网络来源的RTI中提取可操作的见解以提高其运营效用。其次，APT-CGLP结合了定制的多目标训练算法，该算法将对比学习与模式间掩蔽建模相结合，促进粗粒度和细粒度级别的跨模式攻击语义对齐。对四个现实世界APT数据集的广泛实验表明，APT-CGM在准确性和效率方面始终优于最先进的威胁搜寻基线。



## **4. V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs**

V-攻击：针对LVLM的可控对抗攻击的解纠缠价值特征 cs.CV

21 pages

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20223v1) [paper-pdf](https://arxiv.org/pdf/2511.20223v1)

**Authors**: Sen Nie, Jie Zhang, Jianxin Yan, Shiguang Shan, Xilin Chen

**Abstract**: Adversarial attacks have evolved from simply disrupting predictions on conventional task-specific models to the more complex goal of manipulating image semantics on Large Vision-Language Models (LVLMs). However, existing methods struggle with controllability and fail to precisely manipulate the semantics of specific concepts in the image. We attribute this limitation to semantic entanglement in the patch-token representations on which adversarial attacks typically operate: global context aggregated by self-attention in the vision encoder dominates individual patch features, making them unreliable handles for precise local semantic manipulation. Our systematic investigation reveals a key insight: value features (V) computed within the transformer attention block serve as much more precise handles for manipulation. We show that V suppresses global-context channels, allowing it to retain high-entropy, disentangled local semantic information. Building on this discovery, we propose V-Attack, a novel method designed for precise local semantic attacks. V-Attack targets the value features and introduces two core components: (1) a Self-Value Enhancement module to refine V's intrinsic semantic richness, and (2) a Text-Guided Value Manipulation module that leverages text prompts to locate source concept and optimize it toward a target concept. By bypassing the entangled patch features, V-Attack achieves highly effective semantic control. Extensive experiments across diverse LVLMs, including LLaVA, InternVL, DeepseekVL and GPT-4o, show that V-Attack improves the attack success rate by an average of 36% over state-of-the-art methods, exposing critical vulnerabilities in modern visual-language understanding. Our code and data are available https://github.com/Summu77/V-Attack.

摘要: 对抗性攻击已经从简单地破坏传统任务特定模型上的预测发展到在大型视觉语言模型（LVLM）上操纵图像语义的更复杂目标。然而，现有的方法难以控制，并且无法准确地操纵图像中特定概念的语义。我们将这种限制归因于对抗性攻击通常运作的补丁令牌表示中的语义纠缠：视觉编码器中由自我注意力聚集的全局上下文主导了单个补丁特征，使得它们对于精确的局部语义操纵来说不可靠。我们的系统性调查揭示了一个关键见解：在Transformer注意力块内计算的值特征（V）可以作为更精确的操纵处理。我们表明，V抑制了全球上下文通道，使其能够保留高熵、解开的局部语义信息。在这一发现的基础上，我们提出了V-Attack，这是一种专为精确的局部语义攻击而设计的新颖方法。V-Attack针对价值特征并引入了两个核心组件：（1）自我价值增强模块，用于细化V内在的语义丰富性，和（2）文本引导的价值操纵模块，利用文本提示来定位源概念并将其优化为目标概念。通过绕过纠缠补丁特征，V-Attack实现了高效的语义控制。在LLaVA、InternVLL、DeepseekVLL和GPT-4 o等各种LVLM上进行的广泛实验表明，V-Attack比最先进的方法平均提高了36%的攻击成功率，暴露了现代视觉语言理解中的关键漏洞。我们的代码和数据可访问https://github.com/Summu77/V-Attack。



## **5. On the Feasibility of Hijacking MLLMs' Decision Chain via One Perturbation**

论通过一次扰动劫持MLLM决策链的可行性 cs.CV

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20002v1) [paper-pdf](https://arxiv.org/pdf/2511.20002v1)

**Authors**: Changyue Li, Jiaying Li, Youliang Yuan, Jiaming He, Zhicong Huang, Pinjia He

**Abstract**: Conventional adversarial attacks focus on manipulating a single decision of neural networks. However, real-world models often operate in a sequence of decisions, where an isolated mistake can be easily corrected, but cascading errors can lead to severe risks.   This paper reveals a novel threat: a single perturbation can hijack the whole decision chain. We demonstrate the feasibility of manipulating a model's outputs toward multiple, predefined outcomes, such as simultaneously misclassifying "non-motorized lane" signs as "motorized lane" and "pedestrian" as "plastic bag".   To expose this threat, we introduce Semantic-Aware Universal Perturbations (SAUPs), which induce varied outcomes based on the semantics of the inputs. We overcome optimization challenges by developing an effective algorithm, which searches for perturbations in normalized space with a semantic separation strategy. To evaluate the practical threat of SAUPs, we present RIST, a new real-world image dataset with fine-grained semantic annotations. Extensive experiments on three multimodal large language models demonstrate their vulnerability, achieving a 70% attack success rate when controlling five distinct targets using just an adversarial frame.

摘要: 传统的对抗攻击专注于操纵神经网络的单个决策。然而，现实世界的模型通常以一系列决策的方式运行，其中孤立的错误可以很容易地纠正，但连锁错误可能会导致严重的风险。   本文揭示了一种新颖的威胁：一个单一的扰动就可以劫持整个决策链。我们证明了将模型输出操作为多个预定义结果的可行性，例如同时将“非机动车道”标志误分类为“机动车道”，将“行人”误分类为“塑料袋”。   为了揭露这一威胁，我们引入了语义感知通用扰动（SAUP），它根据输入的语义引发不同的结果。我们通过开发一种有效的算法来克服优化挑战，该算法使用语义分离策略在规范化空间中搜索扰动。为了评估SAUP的实际威胁，我们提出了RIST，这是一个具有细粒度语义注释的新现实世界图像数据集。对三个多模式大型语言模型的广泛实验证明了它们的脆弱性，仅使用对抗框架控制五个不同目标时，攻击成功率达到70%。



## **6. Prompt Fencing: A Cryptographic Approach to Establishing Security Boundaries in Large Language Model Prompts**

提示围栏：在大型语言模型脚本中建立安全边界的加密方法 cs.CR

44 pages, 1 figure

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19727v1) [paper-pdf](https://arxiv.org/pdf/2511.19727v1)

**Authors**: Steven Peh

**Abstract**: Large Language Models (LLMs) remain vulnerable to prompt injection attacks, representing the most significant security threat in production deployments. We present Prompt Fencing, a novel architectural approach that applies cryptographic authentication and data architecture principles to establish explicit security boundaries within LLM prompts. Our approach decorates prompt segments with cryptographically signed metadata including trust ratings and content types, enabling LLMs to distinguish between trusted instructions and untrusted content. While current LLMs lack native fence awareness, we demonstrate that simulated awareness through prompt instructions achieved complete prevention of injection attacks in our experiments, reducing success rates from 86.7% (260/300 successful attacks) to 0% (0/300 successful attacks) across 300 test cases with two leading LLM providers. We implement a proof-of-concept fence generation and verification pipeline with a total overhead of 0.224 seconds (0.130s for fence generation, 0.094s for validation) across 100 samples. Our approach is platform-agnostic and can be incrementally deployed as a security layer above existing LLM infrastructure, with the expectation that future models will be trained with native fence awareness for optimal security.

摘要: 大型语言模型（LLM）仍然容易受到即时注入攻击，这是生产部署中最严重的安全威胁。我们介绍了提示围栏，这是一种新型的架构方法，应用加密认证和数据架构原则来在LLM提示内建立明确的安全边界。我们的方法使用加密签名的元数据（包括信任评级和内容类型）装饰提示段，使LLM能够区分受信任的指令和不受信任的内容。虽然当前的LLM缺乏原生围栏感知，但我们证明，通过提示指令的模拟感知在我们的实验中实现了对注入攻击的完全预防，在与两家领先的LLM提供商的300个测试案例中，成功率从86.7%（260/300次成功攻击）降低到0%（0/300次成功攻击）。我们在100个样本中实施了概念验证围栏生成和验证管道，总成本为0.224秒（围栏生成0.130秒，验证0.094秒）。我们的方法与平台无关，可以增量部署为现有LLM基础设施之上的安全层，预计未来的模型将通过本地围栏感知进行训练，以获得最佳安全性。



## **7. Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization**

通过树群双感知搜索和优化实现LLM安全性调整的对抗性攻击-防御协同进化 cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.19218v2) [paper-pdf](https://arxiv.org/pdf/2511.19218v2)

**Authors**: Xurui Li, Kaisong Song, Rui Zhu, Pin-Yu Chen, Haixu Tang

**Abstract**: Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.

摘要: 大型语言模型（LLM）在网络服务中迅速发展，提供了前所未有的能力，同时放大了社会风险。现有的作品往往专注于孤立的越狱攻击或静态防御，忽视了现实世界网络环境中不断变化的威胁与保障措施之间的动态相互作用。为了缓解这些挑战，我们提出了ACE安全（针对LLM安全的对抗协同进化），一个新颖的框架，通过无缝集成两个关键的创新过程来联合优化攻击和防御模型：（1）群体感知策略引导的蒙特卡洛树搜索（GS-MCTS），它有效地探索越狱策略以发现漏洞并生成多样化的对抗样本;（2）对抗性课程树感知群组策略优化（AC-TSYS），通过课程强化学习，利用具有挑战性的样本联合训练攻击和防御LLM，实现稳健的相互改进。多个基准的评估表明，我们的方法优于现有的攻击和防御方法，并为开发能够可持续支持负责任的人工智能生态系统的LLM提供了可行的途径。



## **8. AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents**

AttackPilot：使用基于LLM的代理对ML服务进行自主推理攻击 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19536v1) [paper-pdf](https://arxiv.org/pdf/2511.19536v1)

**Authors**: Yixin Wu, Rui Wen, Chi Cui, Michael Backes, Yang Zhang

**Abstract**: Inference attacks have been widely studied and offer a systematic risk assessment of ML services; however, their implementation and the attack parameters for optimal estimation are challenging for non-experts. The emergence of advanced large language models presents a promising yet largely unexplored opportunity to develop autonomous agents as inference attack experts, helping address this challenge. In this paper, we propose AttackPilot, an autonomous agent capable of independently conducting inference attacks without human intervention. We evaluate it on 20 target services. The evaluation shows that our agent, using GPT-4o, achieves a 100.0% task completion rate and near-expert attack performance, with an average token cost of only $0.627 per run. The agent can also be powered by many other representative LLMs and can adaptively optimize its strategy under service constraints. We further perform trace analysis, demonstrating that design choices, such as a multi-agent framework and task-specific action spaces, effectively mitigate errors such as bad plans, inability to follow instructions, task context loss, and hallucinations. We anticipate that such agents could empower non-expert ML service providers, auditors, or regulators to systematically assess the risks of ML services without requiring deep domain expertise.

摘要: 推理攻击已被广泛研究，并提供了ML服务的系统性风险评估;然而，它们的实现和用于最佳估计的攻击参数对于非专家来说具有挑战性。高级大型语言模型的出现提供了一个充满希望但基本上未开发的机会，可以将自主代理开发为推理攻击专家，帮助应对这一挑战。在本文中，我们提出了AttackPilot，这是一种能够在没有人为干预的情况下独立进行推理攻击的自主代理。我们对20个目标服务进行了评估。评估显示，我们的代理使用GPT-4 o实现了100.0%的任务完成率和接近专家的攻击性能，每次运行的平均代币成本仅为0.627美元。该代理还可以由许多其他代表性的LLM提供支持，并可以在服务约束下自适应地优化其策略。我们进一步执行跟踪分析，证明设计选择（例如多智能体框架和特定于任务的动作空间）可以有效地减轻错误，例如糟糕的计划、无法遵循指令、任务上下文丢失和幻觉。我们预计，此类代理可以让非专家ML服务提供商、审计师或监管机构能够系统性地评估ML服务的风险，而无需深入领域的专业知识。



## **9. Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations**

以负责任的人工智能考虑保护大型语言模型免受越狱利用 cs.CR

20 pages including appendix; technical report; NeurIPS 2024 style

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18933v1) [paper-pdf](https://arxiv.org/pdf/2511.18933v1)

**Authors**: Ryan Wong, Hosea David Yu Fei Ng, Dhananjai Sharma, Glenn Jun Jie Ng, Kavishvaran Srinivasan

**Abstract**: Large Language Models (LLMs) remain susceptible to jailbreak exploits that bypass safety filters and induce harmful or unethical behavior. This work presents a systematic taxonomy of existing jailbreak defenses across prompt-level, model-level, and training-time interventions, followed by three proposed defense strategies. First, a Prompt-Level Defense Framework detects and neutralizes adversarial inputs through sanitization, paraphrasing, and adaptive system guarding. Second, a Logit-Based Steering Defense reinforces refusal behavior through inference-time vector steering in safety-sensitive layers. Third, a Domain-Specific Agent Defense employs the MetaGPT framework to enforce structured, role-based collaboration and domain adherence. Experiments on benchmark datasets show substantial reductions in attack success rate, achieving full mitigation under the agent-based defense. Overall, this study highlights how jailbreaks pose a significant security threat to LLMs and identifies key intervention points for prevention, while noting that defense strategies often involve trade-offs between safety, performance, and scalability. Code is available at: https://github.com/Kuro0911/CS5446-Project

摘要: 大型语言模型（LLM）仍然容易受到越狱漏洞利用的影响，这些漏洞绕过安全过滤器并引发有害或不道德行为。这项工作对预算级、模型级和训练时干预措施的现有越狱防御进行了系统分类，然后提出了三种拟议的防御策略。首先，预算级防御框架通过净化、重述和自适应系统防护来检测并中和对抗输入。其次，基于日志的转向防御通过安全敏感层中的推理时间载体转向来加强拒绝行为。第三，领域特定代理防御采用MetaGPT框架来实施结构化的、基于角色的协作和领域遵守。对基准数据集的实验显示，攻击成功率大幅降低，在基于代理的防御下实现了全面缓解。总体而言，这项研究强调了越狱如何对LLM构成重大安全威胁，并确定了预防的关键干预点，同时指出防御策略通常涉及安全性、性能和可扩展性之间的权衡。代码可访问：https://github.com/Kuro0911/CS5446-Project



## **10. BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models**

BackdoorVLM：视觉语言模型后门攻击的基准 cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18921v1) [paper-pdf](https://arxiv.org/pdf/2511.18921v1)

**Authors**: Juncheng Li, Yige Li, Hanxun Huang, Yunhao Chen, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM .

摘要: 后门攻击通过注入可能在推理时被恶意激活的隐藏行为来破坏机器学习系统的可靠性和可信性。虽然此类威胁在单模式环境中得到了广泛研究，但它们对多模式基础模型（尤其是视觉语言模型（VLM）的影响在很大程度上仍然没有得到充分的研究。在这项工作中，我们引入了\textBF{BackdoorVLM}，这是第一个用于在广泛的设置中系统评估对VLM的后门攻击的全面基准。它采用统一的视角，在核心视觉语言任务（包括图像字幕和视觉问答）中注入和分析后门。BackdoorVLM将多模式后门威胁分为5个代表性类别：定向拒绝、恶意注入、越狱、概念替代和感知劫持。每个类别都捕获了对手可以操纵模型行为的独特途径。我们使用涵盖文本、图像和双峰触发器的12种代表性攻击方法来评估这些威胁，并在2个开源VLM和3个多模式数据集上进行了测试。我们的分析表明，VLM对文本指令表现出很强的敏感性，并且在双峰后门中，文本触发器在形成后门映射时通常会触发图像触发器。值得注意的是，涉及文本形式的后门仍然非常有效，中毒率低至1%，大多数任务的成功率超过90%。这些发现凸显了当前VLM中先前未充分探索的重大漏洞。我们希望BackdoorVLM能够成为分析和缓解多模式后门威胁的有用基准。代码可访问：https://github.com/bin015/BackdoorVLM。



## **11. EAGER: Edge-Aligned LLM Defense for Robust, Efficient, and Accurate Cybersecurity Question Answering**

EAGER：边缘对齐的LLM防御，实现强大、高效和准确的网络安全问题解答 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19523v1) [paper-pdf](https://arxiv.org/pdf/2511.19523v1)

**Authors**: Onat Gungor, Roshan Sood, Jiasheng Zhou, Tajana Rosing

**Abstract**: Large Language Models (LLMs) are highly effective for cybersecurity question answering (QA) but are difficult to deploy on edge devices due to their size. Quantization reduces memory and compute requirements but often degrades accuracy and increases vulnerability to adversarial attacks. We present EAGER, an edge-aligned defense framework that integrates parameter-efficient quantization with domain-specific preference alignment to jointly optimize efficiency, robustness, and accuracy. Unlike prior methods that address these aspects separately, EAGER leverages Quantized Low-Rank Adaptation (QLoRA) for low-cost fine-tuning and Direct Preference Optimization (DPO) on a self-constructed cybersecurity preference dataset, eliminating the need for human labels. Experiments show that EAGER reduces adversarial attack success rates by up to 7.3x and improves QA accuracy by up to 55% over state-of-the-art defenses, while achieving the lowest response latency on a Jetson Orin, demonstrating its practical edge deployment.

摘要: 大型语言模型（LLM）对于网络安全问题回答（QA）非常有效，但由于其尺寸而难以在边缘设备上部署。量化降低了内存和计算需求，但通常会降低准确性并增加对对抗攻击的脆弱性。我们提出了EAGER，这是一个边缘对齐的防御框架，它将参数高效量化与特定领域的偏好对齐集成在一起，以共同优化效率、稳健性和准确性。与单独解决这些方面的现有方法不同，EAGER利用量化低等级适应（QLoRA）对自构建的网络安全偏好数据集进行低成本微调和直接偏好优化（DPO），消除了对人类标签的需求。实验表明，与最先进的防御相比，EAGER将对抗攻击成功率降低了7.3倍，将QA准确性提高了55%，同时在Jetson Orin上实现了最低的响应延迟，展示了其实用的边缘部署。



## **12. RoguePrompt: Dual-Layer Ciphering for Self-Reconstruction to Circumvent LLM Moderation**

RoguePrompt：用于自我重构以规避LLM调节的双层加密 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18790v1) [paper-pdf](https://arxiv.org/pdf/2511.18790v1)

**Authors**: Benyamin Tafreshian

**Abstract**: Content moderation pipelines for modern large language models combine static filters, dedicated moderation services, and alignment tuned base models, yet real world deployments still exhibit dangerous failure modes. This paper presents RoguePrompt, an automated jailbreak attack that converts a disallowed user query into a self reconstructing prompt which passes provider moderation while preserving the original harmful intent. RoguePrompt partitions the instruction across two lexical streams, applies nested classical ciphers, and wraps the result in natural language directives that cause the target model to decode and execute the hidden payload. Our attack assumes only black box access to the model and to the associated moderation endpoint. We instantiate RoguePrompt against GPT 4o and evaluate it on 2 448 prompts that a production moderation system previously marked as strongly rejected. Under an evaluation protocol that separates three security relevant outcomes bypass, reconstruction, and execution the attack attains 84.7 percent bypass, 80.2 percent reconstruction, and 71.5 percent full execution, substantially outperforming five automated jailbreak baselines. We further analyze the behavior of several automated and human aligned evaluators and show that dual layer lexical transformations remain effective even when detectors rely on semantic similarity or learned safety rubrics. Our results highlight systematic blind spots in current moderation practice and suggest that robust deployment will require joint reasoning about user intent, decoding workflows, and model side computation rather than surface level toxicity alone.

摘要: 现代大型语言模型的内容审核管道结合了静态过滤器、专用审核服务和对齐优化的基本模型，但现实世界的部署仍然表现出危险的失败模式。本文介绍了RoguePrompt，这是一种自动越狱攻击，可将不允许的用户查询转换为自我重建提示，该提示通过提供商审核，同时保留最初的有害意图。RoguePrompt将指令划分为两个词汇流，应用嵌套的经典密码，并将结果包装在自然语言指令中，从而使目标模型解码和执行隐藏的有效负载。我们的攻击假设只有黑匣子访问模型和相关的审核端点。我们针对GPT 4o实例化RoguePrompt，并在生产审核系统之前标记为强烈拒绝的2 448个提示上对其进行评估。在将三种安全相关结果分开的评估协议下，攻击的绕过率为84.7%，重建率为80.2%，完全执行率为71.5%，大大优于五个自动越狱基线。我们进一步分析了几个自动化和人工对齐的评估者的行为，并表明即使检测器依赖于语义相似性或习得的安全规则，双层词汇转换仍然有效。我们的结果强调了当前审核实践中的系统盲点，并表明稳健的部署需要对用户意图、解码工作流程和模型端计算进行联合推理，而不仅仅是表面级别的毒性。



## **13. Automating Deception: Scalable Multi-Turn LLM Jailbreaks**

自动欺骗：可扩展多回合LLM越狱 cs.LG

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19517v1) [paper-pdf](https://arxiv.org/pdf/2511.19517v1)

**Authors**: Adarsh Kumarappan, Ananya Mujoo

**Abstract**: Multi-turn conversational attacks, which leverage psychological principles like Foot-in-the-Door (FITD), where a small initial request paves the way for a more significant one, to bypass safety alignments, pose a persistent threat to Large Language Models (LLMs). Progress in defending against these attacks is hindered by a reliance on manual, hard-to-scale dataset creation. This paper introduces a novel, automated pipeline for generating large-scale, psychologically-grounded multi-turn jailbreak datasets. We systematically operationalize FITD techniques into reproducible templates, creating a benchmark of 1,500 scenarios across illegal activities and offensive content. We evaluate seven models from three major LLM families under both multi-turn (with history) and single-turn (without history) conditions. Our results reveal stark differences in contextual robustness: models in the GPT family demonstrate a significant vulnerability to conversational history, with Attack Success Rates (ASR) increasing by as much as 32 percentage points. In contrast, Google's Gemini 2.5 Flash exhibits exceptional resilience, proving nearly immune to these attacks, while Anthropic's Claude 3 Haiku shows strong but imperfect resistance. These findings highlight a critical divergence in how current safety architectures handle conversational context and underscore the need for defenses that can resist narrative-based manipulation.

摘要: 多轮对话攻击利用了“门中脚”（FIDS）等心理学原则，即小的初始请求为更重要的请求铺平道路，以绕过安全对齐，对大型语言模型（LLM）构成了持续的威胁。依赖手动、难以扩展的数据集创建，阻碍了防御这些攻击的进展。本文介绍了一种新颖的自动化管道，用于生成大规模、基于心理学的多回合越狱数据集。我们系统性地将FITD技术操作为可复制的模板，创建了涵盖非法活动和攻击性内容的1，500个场景的基准。我们在多转弯（有历史）和单转弯（无历史）条件下评估了来自三个主要LLM家族的七个模型。我们的结果揭示了上下文稳健性的明显差异：GPT家族中的模型表现出对对话历史的显着脆弱性，攻击成功率（ASB）增加了32个百分点。相比之下，谷歌的Gemini 2.5 Flash表现出出色的韧性，几乎不受这些攻击的影响，而Anthropic的Claude 3 Haiku表现出强大但不完美的抵抗力。这些发现凸显了当前安全架构如何处理对话上下文的严重分歧，并强调了对能够抵抗基于叙述的操纵的防御的必要性。



## **14. Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems**

代码中的阴影：探索基于LLM的多代理软件开发系统的风险和防御 cs.CR

Accepted by AAAI 2026 Alignment Track

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18467v1) [paper-pdf](https://arxiv.org/pdf/2511.18467v1)

**Authors**: Xiaoqing Wang, Keman Huang, Bin Liang, Hongyu Li, Xiaoyong Du

**Abstract**: The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% in MU-BA scenarios, and 71%, 84%, and 45% in BU-MA scenarios. Our defense mechanism reduced attack success rates significantly, particularly in the MU-BA scenario. Further analysis reveals that compromised agents in the coding and testing phases pose significantly greater security risks, while also identifying critical agents that require protection against malicious user exploitation. Our findings highlight the urgent need for robust security measures in multi-agent software development systems and provide practical guidelines for implementing targeted, resource-efficient defensive strategies.

摘要: 大语言模型（LLM）驱动的多智能体系统的快速发展，大大简化了软件开发任务，使用户几乎没有技术专长，开发可执行的应用程序。虽然这些系统通过自然语言需求使软件创建民主化，但它们引入了重大的安全风险，这些风险在很大程度上尚未被探索。我们确定了两种风险情况：恶意用户与良性代理（MU-BA）和良性用户与恶意代理（BU-MA）。我们介绍了隐式恶意行为注入攻击（IMBIA），演示了如何多代理系统可以被操纵，以产生隐藏的恶意功能下看似良性的应用程序的软件，并提出Adv-IMBIA作为一种防御机制。ChatDev、MetaGPT和AgentVerse框架的评估揭示了不同的漏洞模式，IMBIA在MU-BA场景中的攻击成功率为93%、45%和71%，在BU-MA场景中的攻击成功率为71%、84%和45%。我们的防御机制显着降低了攻击成功率，特别是在MU-BA的情况下。进一步的分析表明，在编码和测试阶段受影响的代理会带来显着更大的安全风险，同时还可以识别需要保护以防止恶意用户利用的关键代理。我们的研究结果强调了多代理软件开发系统中对强有力的安全措施的迫切需要，并为实施有针对性的、资源高效的防御策略提供了实用指南。



## **15. Think Fast: Real-Time IoT Intrusion Reasoning Using IDS and LLMs at the Edge Gateway**

快速思考：在边缘网关使用IDS和LLM进行实时物联网入侵推理 cs.CR

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18230v1) [paper-pdf](https://arxiv.org/pdf/2511.18230v1)

**Authors**: Saeid Jamshidi, Amin Nikanjam, Negar Shahabi, Kawser Wazed Nafi, Foutse Khomh, Samira Keivanpour, Rolando Herrero

**Abstract**: As the number of connected IoT devices continues to grow, securing these systems against cyber threats remains a major challenge, especially in environments with limited computational and energy resources. This paper presents an edge-centric Intrusion Detection System (IDS) framework that integrates lightweight machine learning (ML) based IDS models with pre-trained large language models (LLMs) to improve detection accuracy, semantic interpretability, and operational efficiency at the network edge. The system evaluates six ML-based IDS models: Decision Tree (DT), K-Nearest Neighbors (KNN), Random Forest (RF), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM model on low-power edge gateways, achieving accuracy up to 98 percent under real-world cyberattacks. For anomaly detection, the system transmits a compact and secure telemetry snapshot (for example, CPU usage, memory usage, latency, and energy consumption) via low-bandwidth API calls to LLMs including GPT-4-turbo, DeepSeek V2, and LLaMA 3.5. These models use zero-shot, few-shot, and chain-of-thought reasoning to produce human-readable threat analyses and actionable mitigation recommendations. Evaluations across diverse attacks such as DoS, DDoS, brute force, and port scanning show that the system enhances interpretability while maintaining low latency (<1.5 s), minimal bandwidth usage (<1.2 kB per prompt), and energy efficiency (<75 J), demonstrating its practicality and scalability as an IDS solution for edge gateways.

摘要: 随着互联物联网设备数量的持续增长，保护这些系统免受网络威胁仍然是一项重大挑战，尤其是在计算和能源资源有限的环境中。本文提出了一种以边缘为中心的入侵检测系统（IDS）框架，该框架将基于轻量级机器学习（ML）的IDS模型与预训练的大型语言模型（LLM）集成在一起，以提高网络边缘的检测准确性、语义解释性和运营效率。该系统评估了六种基于ML的IDS模型：决策树（DT）、K近邻（KNN）、随机森林（RF）、卷积神经网络（CNN）、长短期记忆（LSTM）和低功耗边缘网关上的混合CNN-LSTM模型，在现实世界的网络攻击下实现高达98%的准确率。对于异常检测，系统通过低带宽API调用向包括GPT-4-涡轮、DeepSeek V2和LLaMA 3.5在内的LLM传输紧凑且安全的遥感快照（例如，中央处理器使用率、内存使用率、延迟和能源消耗）。这些模型使用零射击、少射击和思想链推理来生成人类可读的威胁分析和可操作的缓解建议。对多种攻击（例如：拒绝服务、拒绝服务、暴力攻击和端口扫描）的评估表明，该系统增强了可解释性，同时保持低延迟（<1.5秒）、最低带宽使用（每次提示<1.2 kB）和能源效率（<75 J），证明了其作为边缘网关IDS解决方案的实用性和可扩展性。



## **16. ASTRA: Agentic Steerability and Risk Assessment Framework**

ASTRA：广义可操纵性和风险评估框架 cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.18114v1) [paper-pdf](https://arxiv.org/pdf/2511.18114v1)

**Authors**: Itay Hazan, Yael Mathov, Guy Shtar, Ron Bitton, Itsik Mantin

**Abstract**: Securing AI agents powered by Large Language Models (LLMs) represents one of the most critical challenges in AI security today. Unlike traditional software, AI agents leverage LLMs as their "brain" to autonomously perform actions via connected tools. This capability introduces significant risks that go far beyond those of harmful text presented in a chatbot that was the main application of LLMs. A compromised AI agent can deliberately abuse powerful tools to perform malicious actions, in many cases irreversible, and limited solely by the guardrails on the tools themselves and the LLM ability to enforce them. This paper presents ASTRA, a first-of-its-kind framework designed to evaluate the effectiveness of LLMs in supporting the creation of secure agents that enforce custom guardrails defined at the system-prompt level (e.g., "Do not send an email out of the company domain," or "Never extend the robotic arm in more than 2 meters").   Our holistic framework simulates 10 diverse autonomous agents varying between a coding assistant and a delivery drone equipped with 37 unique tools. We test these agents against a suite of novel attacks developed specifically for agentic threats, inspired by the OWASP Top 10 but adapted to challenge the ability of the LLM for policy enforcement during multi-turn planning and execution of strict tool activation. By evaluating 13 open-source, tool-calling LLMs, we uncovered surprising and significant differences in their ability to remain secure and keep operating within their boundaries. The purpose of this work is to provide the community with a robust and unified methodology to build and validate better LLMs, ultimately pushing for more secure and reliable agentic AI systems.

摘要: 保护由大型语言模型（LLM）支持的AI代理是当今AI安全中最关键的挑战之一。与传统软件不同，人工智能代理利用LLM作为他们的“大脑”，通过连接的工具自主执行操作。这种能力带来了重大的风险，远远超出了聊天机器人中呈现的有害文本的风险，而聊天机器人是LLM的主要应用。一个被入侵的人工智能代理可以故意滥用强大的工具来执行恶意操作，在许多情况下是不可逆的，并且仅受工具本身的护栏和LLM执行它们的能力的限制。本文介绍了ASTRA，这是一个首创的框架，旨在评估LLM在支持创建安全代理方面的有效性，这些代理强制执行在系统提示级别定义的自定义护栏（例如，“请勿将电子邮件发送到公司域名之外”，或“切勿将机械臂伸出超过2米”）。   我们的整体框架模拟了10个不同的自主代理，从编码助理到配备37种独特工具的送货无人机。我们针对专门针对代理威胁开发的一套新型攻击来测试这些代理，这些攻击受到OWASP Top 10的启发，但经过调整，以挑战LLM在多回合规划和执行严格工具激活期间政策执行的能力。通过评估13个开源、工具调用LLM，我们发现它们在保持安全和在其范围内运营的能力方面存在令人惊讶且显着的差异。这项工作的目的是为社区提供一种强大而统一的方法来构建和验证更好的LLM，最终推动更安全和可靠的代理人工智能系统。



## **17. Building Browser Agents: Architecture, Security, and Practical Solutions**

构建浏览器代理：体系结构、安全性和实用解决方案 cs.SE

30 pages, 22 figures. Production architecture and benchmark evaluation of browser agents

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.19477v1) [paper-pdf](https://arxiv.org/pdf/2511.19477v1)

**Authors**: Aram Vardanyan

**Abstract**: Browser agents enable autonomous web interaction but face critical reliability and security challenges in production. This paper presents findings from building and operating a production browser agent. The analysis examines where current approaches fail and what prevents safe autonomous operation. The fundamental insight: model capability does not limit agent performance; architectural decisions determine success or failure. Security analysis of real-world incidents reveals prompt injection attacks make general-purpose autonomous operation fundamentally unsafe. The paper argues against developing general browsing intelligence in favor of specialized tools with programmatic constraints, where safety boundaries are enforced through code instead of large language model (LLM) reasoning. Through hybrid context management combining accessibility tree snapshots with selective vision, comprehensive browser tooling matching human interaction capabilities, and intelligent prompt engineering, the agent achieved approximately 85% success rate on the WebGames benchmark across 53 diverse challenges (compared to approximately 50% reported for prior browser agents and 95.7% human baseline).

摘要: 浏览器代理能够实现自主Web交互，但在生产中面临关键的可靠性和安全挑战。本文介绍了构建和操作生产浏览器代理的发现。该分析检查了当前方法在哪里失败以及是什么阻碍了安全自主操作。基本见解：模型能力不会限制代理性能;架构决策决定成败。对现实世界事件的安全分析表明，即时注入攻击使通用自主操作从根本上不安全。该论文反对开发通用浏览智能，转而采用具有编程约束的专用工具，其中安全边界是通过代码而不是大型语言模型（LLM）推理来强制实施的。通过将可访问性树快照与选择性视觉相结合的混合上下文管理、匹配人类交互能力的全面浏览器工具和智能提示工程，该代理在WebGames基准测试中在53个不同挑战中实现了约85%的成功率（相比之下，之前的浏览器代理报告的成功率约为50%，人类基线为95.7%）。



## **18. Steering in the Shadows: Causal Amplification for Activation Space Attacks in Large Language Models**

在阴影中操纵：大型语言模型中激活空间攻击的因果放大 cs.CR

31 pages, 5 figures, 9 tables

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.17194v1) [paper-pdf](https://arxiv.org/pdf/2511.17194v1)

**Authors**: Zhiyuan Xu, Stanislav Abaimov, Joseph Gardiner, Sana Belguith

**Abstract**: Modern large language models (LLMs) are typically secured by auditing data, prompts, and refusal policies, while treating the forward pass as an implementation detail. We show that intermediate activations in decoder-only LLMs form a vulnerable attack surface for behavioral control. Building on recent findings on attention sinks and compression valleys, we identify a high-gain region in the residual stream where small, well-aligned perturbations are causally amplified along the autoregressive trajectory--a Causal Amplification Effect (CAE). We exploit this as an attack surface via Sensitivity-Scaled Steering (SSS), a progressive activation-level attack that combines beginning-of-sequence (BOS) anchoring with sensitivity-based reinforcement to focus a limited perturbation budget on the most vulnerable layers and tokens. We show that across multiple open-weight models and four behavioral axes, SSS induces large shifts in evil, hallucination, sycophancy, and sentiment while preserving high coherence and general capabilities, turning activation steering into a concrete security concern for white-box and supply-chain LLM deployments.

摘要: 现代大型语言模型（LLM）通常通过审计数据、提示和拒绝策略来保护，同时将转发视为实现细节。我们表明，仅限解码器的LLM中的中间激活形成了行为控制的脆弱攻击表面。基于最近关于注意力汇和压缩谷的研究结果，我们在剩余流中识别出了一个高收益区域，其中微小的、排列整齐的扰动沿着自回归轨迹被因果放大--因果放大效应（CAE）。我们通过灵敏度缩放转向（SS）将其作为攻击表面，这是一种渐进式激活级攻击，将序列识别（BOS）锚定与基于灵敏度的强化相结合，将有限的扰动预算集中在最脆弱的层和令牌上。我们表明，在多个开放权重模型和四个行为轴中，SS会引发邪恶、幻觉、谄媚和情绪的巨大转变，同时保持高度一致性和通用能力，将激活引导变成白盒和供应链LLM部署的具体安全问题。



## **19. MultiPriv: Benchmarking Individual-Level Privacy Reasoning in Vision-Language Models**

MultiPriv：视觉语言模型中的个人隐私推理基准 cs.CV

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.16940v1) [paper-pdf](https://arxiv.org/pdf/2511.16940v1)

**Authors**: Xiongtao Sun, Hui Li, Jiaming Zhang, Yujie Yang, Kaili Liu, Ruxin Feng, Wen Jun Tan, Wei Yang Bryan Lim

**Abstract**: Modern Vision-Language Models (VLMs) demonstrate sophisticated reasoning, escalating privacy risks beyond simple attribute perception to individual-level linkage. Current privacy benchmarks are structurally insufficient for this new threat, as they primarily evaluate privacy perception while failing to address the more critical risk of privacy reasoning: a VLM's ability to infer and link distributed information to construct individual profiles. To address this critical gap, we propose \textbf{MultiPriv}, the first benchmark designed to systematically evaluate individual-level privacy reasoning in VLMs. We introduce the \textbf{Privacy Perception and Reasoning (PPR)} framework and construct a novel, bilingual multimodal dataset to support it. The dataset uniquely features a core component of synthetic individual profiles where identifiers (e.g., faces, names) are meticulously linked to sensitive attributes. This design enables nine challenging tasks evaluating the full PPR spectrum, from attribute detection to cross-image re-identification and chained inference. We conduct a large-scale evaluation of over 50 foundational and commercial VLMs. Our analysis reveals: (1) Many VLMs possess significant, unmeasured reasoning-based privacy risks. (2) Perception-level metrics are poor predictors of these reasoning risks, revealing a critical evaluation gap. (3) Existing safety alignments are inconsistent and ineffective against such reasoning-based attacks. MultiPriv exposes systemic vulnerabilities and provides the necessary framework for developing robust, privacy-preserving VLMs.

摘要: 现代视觉语言模型（VLM）展示了复杂的推理，将隐私风险从简单的属性感知升级到个人层面的联系。当前的隐私基准在结构上不足以应对这种新威胁，因为它们主要评估隐私感知，而未能解决隐私推理的更关键风险：VLM推断和链接分布式信息以构建个人配置文件的能力。为了解决这一关键差距，我们提出了\textBF{MultiPriv}，这是第一个旨在系统评估VLM中个人级别隐私推理的基准。我们引入\textBF{Privacy Percept and Reasoning（PPR）}框架，并构建一个新颖的双语多模式数据集来支持它。该数据集独特地具有合成个人资料的核心组件，其中标识符（例如，面孔、名字）与敏感属性细致地关联起来。该设计实现了九项具有挑战性的任务，评估完整的PPR谱，从属性检测到跨图像重新识别和连锁推理。我们对50多个基础和商业VLM进行了大规模评估。我们的分析揭示了：（1）许多VLM都具有重大的、不可测量的基于推理的隐私风险。(2)感知水平指标无法预测这些推理风险，揭示了关键的评估差距。(3)现有的安全排列不一致，对于此类基于推理的攻击无效。MultiPriv暴露了系统性漏洞，并为开发稳健、保护隐私的VLM提供了必要的框架。



## **20. Evaluating Adversarial Vulnerabilities in Modern Large Language Models**

评估现代大型语言模型中的对抗脆弱性 cs.CR

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.17666v1) [paper-pdf](https://arxiv.org/pdf/2511.17666v1)

**Authors**: Tom Perel

**Abstract**: The recent boom and rapid integration of Large Language Models (LLMs) into a wide range of applications warrants a deeper understanding of their security and safety vulnerabilities. This paper presents a comparative analysis of the susceptibility to jailbreak attacks for two leading publicly available LLMs, Google's Gemini 2.5 Flash and OpenAI's GPT-4 (specifically the GPT-4o mini model accessible in the free tier). The research utilized two main bypass strategies: 'self-bypass', where models were prompted to circumvent their own safety protocols, and 'cross-bypass', where one model generated adversarial prompts to exploit vulnerabilities in the other. Four attack methods were employed - direct injection, role-playing, context manipulation, and obfuscation - to generate five distinct categories of unsafe content: hate speech, illegal activities, malicious code, dangerous content, and misinformation. The success of the attack was determined by the generation of disallowed content, with successful jailbreaks assigned a severity score. The findings indicate a disparity in jailbreak susceptibility between 2.5 Flash and GPT-4, suggesting variations in their safety implementations or architectural design. Cross-bypass attacks were particularly effective, indicating that an ample amount of vulnerabilities exist in the underlying transformer architecture. This research contributes a scalable framework for automated AI red-teaming and provides data-driven insights into the current state of LLM safety, underscoring the complex challenge of balancing model capabilities with robust safety mechanisms.

摘要: 最近大型语言模型（LLM）的蓬勃发展和快速集成到广泛的应用程序中，这使得人们需要更深入地了解其安全性和安全漏洞。本文对两种领先的公开LLM（Google的Gemini 2.5 Flash和OpenAI的GPT-4（特别是免费层中可访问的GPT-4 o mini型号）进行了越狱攻击的易感性进行了比较分析。该研究利用了两种主要的绕过策略：“自我绕过”（提示模型绕过自己的安全协议）和“交叉绕过”（其中一个模型生成对抗提示以利用另一个模型的漏洞）。使用了四种攻击方法--直接注入、角色扮演、上下文操纵和混淆--来生成五种不同类别的不安全内容：仇恨言论、非法活动、恶意代码、危险内容和错误信息。攻击的成功取决于不允许内容的生成，成功的越狱会被赋予严重性分数。研究结果表明，2.5 Flash和GPT-4之间的越狱敏感性存在差异，这表明它们的安全实现或架构设计存在差异。交叉旁路攻击特别有效，这表明底层Transformer架构中存在大量漏洞。这项研究为自动化人工智能红色团队提供了一个可扩展的框架，并提供了对LLM安全当前状态的数据驱动见解，强调了平衡模型能力与强大安全机制的复杂挑战。



## **21. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

基于大语言模型的深度强化学习驱动的自主网络防御奖励设计 cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.

摘要: 对于主题专家来说，在复杂、动态的环境中为自主网络攻击和防御学习代理设计奖励是一项具有挑战性的任务。我们提出了一种基于大语言模型（LLM）的奖励设计方法，以在深度强化学习（DRL）驱动的实验模拟环境中生成自主网络防御策略。精心设计了多个攻击和防御代理角色，反映了代理动作的多样性，以生成LLM引导的奖励设计，其中LLM首先被提供上下文网络模拟环境信息。然后在DRL驱动的攻击防御模拟环境中使用这些奖励结构来学习一整套网络防御策略。我们的结果表明，LLM指导的奖励设计可以制定针对不同对抗行为的有效防御策略。



## **22. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

Q-MLLM：鲁棒多模式大型语言模型安全性的载体量化 cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.

摘要: 多模式大型语言模型（MLLM）在跨模式理解方面表现出了令人印象深刻的能力，但尽管具有强大的文本安全机制，但仍然容易受到视觉输入的对抗攻击。这些漏洞源于两个核心弱点：视觉表示的连续性（允许基于梯度的攻击）以及基于文本的安全机制向视觉内容的不充分转移。我们引入了Q-MLLM，这是一种新颖的架构，它集成了两级量化，以创建针对对抗性攻击的离散瓶颈，同时保留多模式推理能力。通过在像素补丁和语义层面离散化视觉表示，Q-MLLM阻止攻击途径并弥合跨模式安全对齐差距。我们的两阶段训练方法确保稳健的学习，同时保持模型效用。实验表明，与现有方法相比，Q-MLLM在针对越狱攻击和有毒图像攻击的防御成功率明显更高。值得注意的是，Q-MLLM在针对越狱攻击时实现了完美的防御成功率（100%），但在一种可发现的情况下，同时以最小的推理费用在多个实用工具基准上保持竞争性能。这项工作将载体量化建立为安全多模式人工智能系统的有效防御机制，而不需要昂贵的安全特定微调或检测费用。代码可在https://github.com/Amadeuszhao/QMLLM上获取。



## **23. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

PSM：通过LLM引导的黑盒优化实现快速灵敏度最小化 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm

摘要: 系统提示对于指导大型语言模型（LLM）的行为至关重要，但它们通常包含专有逻辑或敏感信息，使其成为提取攻击的主要目标。对抗性查询可以成功地引出这些隐藏指令，从而构成重大的安全和隐私风险。现有的防御机制通常依赖于启发式方法，会产生大量的计算负担，或者不适用于通过黑匣子API访问的模型。本文介绍了一种通过屏蔽附加来强化系统提示的新颖框架，这是一种轻量级方法，可以在原始提示中添加保护性文本层。我们的核心贡献是将即时硬化形式化为一个受效用约束的优化问题。我们利用LLM作为优化器来搜索可能的SHIELD的空间，寻求最大限度地减少从一系列对抗攻击中获得的泄漏指标，同时将任务效用保持在指定阈值以上，该阈值通过基线输出的语义保真度来衡量。这种黑匣子、优化驱动的方法是轻量级且实用的，仅需要API访问目标和优化器LLM。我们通过经验证明，我们优化的SHIELD显着减少了针对一系列全面提取攻击的即时泄漏，在不损害模型预期功能的情况下优于既定的基线防御。我们的工作提供了一个在LLM安全不断升级的环境中开发强大的、实用程序感知的防御的范式。该代码在以下链接上公开：https://github.com/psm-defense/psm



## **24. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

当对齐失败时：对视觉-语言-动作模型的多模式对抗攻击 cs.CV

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.16203v2) [paper-pdf](https://arxiv.org/pdf/2511.16203v2)

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.

摘要: 视觉-语言-动作模型（VLA）最近在具体环境中取得了显着进展，使机器人能够通过统一的多模式理解来感知、推理和行动。尽管它们的能力令人印象深刻，但这些系统的对抗鲁棒性在很大程度上仍未得到探索，尤其是在现实的多模式和黑匣子条件下。现有的研究主要关注单模式扰动，而忽视了从根本上影响体现推理和决策的跨模式失调。本文介绍了VLA-Fool，这是对白盒和黑盒设置下具体VLA模型中多模式对抗鲁棒性的全面研究。VLA-Fool统一了三个级别的多模式对抗攻击：（1）通过基于梯度和基于预算的操纵进行文本扰动，（2）通过补丁和噪音失真进行视觉扰动，以及（3）故意破坏感知和指令之间的语义对应性的跨模式失准攻击。我们进一步将VLA感知的语义空间融入到语言提示中，开发了第一个自动制作和语义引导的提示框架。使用微调的OpenVLA模型对LIBERO基准进行的实验表明，即使是微小的多峰扰动也会导致显着的行为偏差，这表明了体现多峰对齐的脆弱性。



## **25. AutoBackdoor: Automating Backdoor Attacks via LLM Agents**

AutoBackdoor：通过LLM代理自动化后门攻击 cs.CR

23 pages

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16709v1) [paper-pdf](https://arxiv.org/pdf/2511.16709v1)

**Authors**: Yige Li, Zhe Li, Wei Zhao, Nay Myat Min, Hanxun Huang, Xingjun Ma, Jun Sun

**Abstract**: Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at https://github.com/bboylyg/BackdoorLLM.

摘要: 后门攻击对大型语言模型（LLM）的安全部署构成严重威胁，使对手能够植入由特定输入触发的隐藏行为。然而，现有的方法通常依赖于手工制作的触发器和静态数据管道，这些触发器和静态数据管道僵化、劳动密集型，并且不足以系统性地评估现代防御稳健性。随着人工智能代理的能力越来越强，人们对更严格、多样化和可扩展的\textit{red-teaming framework}的需求越来越大，它可以真实地模拟后门威胁并评估对抗条件下的模型弹性。在这项工作中，我们介绍了\textsk {AutoBackdoor}，这是一个用于自动化后门注入的通用框架，包括触发器生成、有毒数据构建以及通过自主代理驱动管道进行的模型微调。与以前的方法不同，AutoBackdoor使用强大的语言模型代理来生成语义连贯的、上下文感知的触发短语，以最少的人力即可在任意主题上进行可扩展的中毒。我们在三种现实的威胁场景下评估AutoBackdoor，包括\textit{Bias Recommendation}、\textit{Hallucination Injection}和\textit{Peer Review Manipulation}，以模拟广泛的攻击。在开源和商业模型（包括LLaMA-3、Mistral、Qwen和GPT-4 o）上的实验表明，我们的方法只需少量中毒样本即可获得超过90%的攻击成功率。更重要的是，我们发现现有的防御系统往往无法减轻这些攻击，这凸显了针对本工作中探讨的代理驱动威胁的更严格和自适应的评估技术的必要性。所有代码、数据集和实验配置都将合并到我们的主存储库https://github.com/bboylyg/BackdoorLLM中。



## **26. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

对抗性诗歌作为大型语言模型中通用的单轮越狱机制 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.

摘要: 我们提供的证据表明，对抗性诗歌可以作为大型语言模型（LLM）的通用单轮越狱技术。在25个前沿专有和开放重量模型中，精心策划的诗意提示产生了很高的攻击成功率（ASB），一些提供商超过了90%。MLCommons和EU CoP风险分类的映射提示表明，诗意攻击跨CBRN、操纵、网络犯罪和失去控制领域转移。通过标准化元提示将1，200个MLCommons有害提示转换为诗句，产生的ASB比散文基线高出18倍。使用3名开放权重LLM评委的整体评估输出，他们的二元安全性评估在分层的人类标记子集上进行了验证。诗意框架的平均越狱成功率为62%，元提示转换的平均越狱成功率约为43%（与非诗意基线相比），大大优于非诗意基线，并揭示了示范家庭和安全培训方法之间的系统性弱点。这些研究结果表明，仅靠风格差异就可以规避当代安全机制，这表明当前对齐方法和评估协议存在根本性局限性。



## **27. As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files**

就像我们以前见过一样：法学硕士在识别可见文件方面表现出脆弱性 cs.AI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15192v2) [paper-pdf](https://arxiv.org/pdf/2511.15192v2)

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yan Pang

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.   We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.

摘要: 大型语言模型（LLM）卓越的语言能力源于对大量数据集的广泛训练，这些数据集通常包括受版权保护的材料，这引起了对未经授权使用的严重担忧。虽然成员关系推理攻击（MIA）提供了检测此类违规行为的潜在解决方案，但由于LLM固有的过度自信，对地面真实训练数据的有限访问以及对经验确定的阈值的依赖，现有方法面临着严重的限制和挑战。   我们提出了一个新的框架，利用不确定性信号来检测LLM训练集中是否使用了版权内容。我们的方法将LLM过度自信从一个限制变成一个资产，通过捕获不确定性模式，可靠地区分“看到”（训练数据）和“看不见”（非训练数据）的内容。COPYRIGHT进一步实现了双重策略：（1）将文件战略性地分割成较小的片段，以减少对大规模训练数据的依赖，以及（2）不确定性引导的无监督聚类，以消除对经验调整阈值的需求。实验结果表明，COPYRIGHT算法在LLaMA 7 b和LLaMA 2 7 b上检测可见文件的平均均衡准确率分别达到90.1%和91.6%。与SOTA基线相比，COPYRIGHT实现了90%以上的相对改进，达到93.8%的平衡精度。它还表现出跨架构的强大通用性，在GPT-J 6 B上保持高性能。这项工作首次将不确定性应用于LLM中的版权检测，为训练数据透明度提供了实用工具。



## **28. Can MLLMs Detect Phishing? A Comprehensive Security Benchmark Suite Focusing on Dynamic Threats and Multimodal Evaluation in Academic Environments**

MLLM可以检测网络钓鱼吗？专注于学术环境中的动态威胁和多模式评估的全面安全基准套件 cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.15165v2) [paper-pdf](https://arxiv.org/pdf/2511.15165v2)

**Authors**: Jingzhuo Zhou

**Abstract**: The rapid proliferation of Multimodal Large Language Models (MLLMs) has introduced unprecedented security challenges, particularly in phishing detection within academic environments. Academic institutions and researchers are high-value targets, facing dynamic, multilingual, and context-dependent threats that leverage research backgrounds, academic collaborations, and personal information to craft highly tailored attacks. Existing security benchmarks largely rely on datasets that do not incorporate specific academic background information, making them inadequate for capturing the evolving attack patterns and human-centric vulnerability factors specific to academia. To address this gap, we present AdapT-Bench, a unified methodological framework and benchmark suite for systematically evaluating MLLM defense capabilities against dynamic phishing attacks in academic settings.

摘要: 多模式大型语言模型（MLLM）的迅速普及带来了前所未有的安全挑战，特别是在学术环境中的网络钓鱼检测方面。学术机构和研究人员是高价值目标，面临着动态、多语言和取决于上下文的威胁，这些威胁利用研究背景、学术合作和个人信息来策划高度定制的攻击。现有的安全基准在很大程度上依赖于不包含特定学术背景信息的数据集，这使得它们不足以捕捉不断变化的攻击模式和学术界特有的以人为本的脆弱性因素。为了解决这一差距，我们提出了AdapT-Bench，这是一个统一的方法框架和基准套件，用于系统性评估MLLM防御能力在学术环境中抵御动态网络钓鱼攻击。



## **29. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

差异化定向干预避免LLM安全一致的框架 cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.06852v4) [paper-pdf](https://arxiv.org/pdf/2511.06852v4)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.

摘要: 安全一致为大型语言模型（LLM）灌输了拒绝恶意请求的关键能力。之前的作品将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是一种过于简单化的做法，将两个功能上不同的神经过程混为一谈：伤害的检测和拒绝的执行。在这项工作中，我们将这个单一的表示解构为伤害检测方向和拒绝执行方向。利用这个细粒度模型，我们引入了差异双向干预（DBDI），这是一种新的白盒框架，可以精确地中和关键层的安全对齐。DBDI对拒绝执行方向应用自适应投影无效，同时通过直接转向抑制伤害检测方向。大量实验表明，DBDI优于著名的越狱方法，对Llama-2等模型的攻击成功率高达97.88%。通过提供更细粒度和机械化的框架，我们的工作为深入了解LLM安全对齐提供了新的方向。



## **30. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

使用大型语言模型在可穿戴物联网系统中进行自适应和稳健的数据中毒检测和清理 cs.LG

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.02894v3) [paper-pdf](https://arxiv.org/pdf/2511.02894v3)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.

摘要: 可穿戴传感设备在物联网（IoT）生态系统中的广泛集成，特别是在医疗保健、智能家居和工业应用中，需要强大的人类活动识别（HAR）技术来改善功能和用户体验。尽管机器学习模型具有高级HAR，但它们越来越容易受到数据中毒攻击，从而损害这些系统的数据完整性和可靠性。防御此类攻击的传统方法通常需要使用大型标记数据集进行广泛的任务特定训练，这限制了动态物联网环境中的适应性。这项工作提出了一种新颖的框架，该框架使用大型语言模型（LLM）在HAR系统中执行中毒检测和清理，利用零触发、单触发和少触发学习范式。我们的方法结合了\textit{role play}提示，LLM承担专家的角色来情境化和评估传感器异常，以及\textit{think分步}推理，指导LLM推断原始传感器数据中的中毒指标和合理的清洁替代品。这些策略最大限度地减少了对大量数据集管理的依赖，并实时实现强大、适应性强的防御机制。我们对框架进行了广泛的评估，量化检测准确性、消毒质量、延迟和通信成本，从而证明了LLM在提高可穿戴物联网系统安全性和可靠性方面的实用性和有效性。



## **31. SoK: Honeypots & LLMs, More Than the Sum of Their Parts?**

SoK：蜜罐和LLM，超过其部分的总和？ cs.CR

Systemization of Knowledge

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.25939v3) [paper-pdf](https://arxiv.org/pdf/2510.25939v3)

**Authors**: Robert A. Bridges, Thomas R. Mitchell, Mauricio Muñoz, Ted Henriksson

**Abstract**: The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.

摘要: 大型语言模型（LLM）的出现有望解决蜜罐设计中长期存在的悖论，以低操作风险实现高保真欺骗。通过自2022年底以来的一系列研究，从构思到原型实现的稳步进展已经显现。自2022年底以来，一系列研究表明，从构思到原型实施正在稳步进展。虽然有希望，但评估仅显示现实世界部署中的渐进进展，并且该领域仍然缺乏对新兴架构模式、核心挑战和评估范式的一致理解。为了填补这一空白，这篇知识系统化（SoK）论文首次对这一新领域进行了全面的概述和分析。我们通过关注三个关键的交叉研究领域来调查和系统化该领域：首先，我们提供了蜜罐检测向量的分类，构建了基于LLM的现实主义必须解决的核心问题;其次，我们综合了关于LLM供电的蜜罐的新兴文献，确定了规范架构和关键评估趋势;第三，我们绘制了蜜罐日志分析的进化路径，从简单的数据简化到自动智能生成。我们将这些发现综合成一个前瞻性的研究路线图，认为这项技术的真正潜力在于创建自主的、自我改进的欺骗系统，以应对智能的、自动化的攻击者的新兴威胁。



## **32. Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs**

通过过度装配攻击：10杆Benign微调到越狱LLM cs.CR

Published as a conference paper at NeurIPS 2025

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2510.02833v4) [paper-pdf](https://arxiv.org/pdf/2510.02833v4)

**Authors**: Zhixin Xie, Xurui Song, Jun Luo

**Abstract**: Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.

摘要: 尽管在安全调整方面做出了大量努力，但最近的研究表明，大型语言模型（LLM）仍然极易受到越狱攻击。在这些攻击中，通过微调损害LLM安全性的基于微调的攻击因其稳定的越狱性能而脱颖而出。特别是，最近的一项研究表明，只需10个有害问答（QA）对进行微调，就可以在各种有害问题上成功越狱。然而，此类恶意微调攻击很容易被检测到，因此会被审核模型阻止。在本文中，我们证明了仅使用10个良性QA对就可以通过微调来越狱LLM;我们的攻击利用了LLM在过适应后对微调数据的敏感性增加。具体来说，我们的微调过程首先是通过使用涉及相同拒绝答案的良性QA对进行微调来过度调整LLM。然后用标准的良性答案进行进一步的微调，导致过度适应的LLM忘记拒绝态度，从而无论问题的危害性如何，都提供合规的答案。我们对十个LLM实施攻击，并将其与五个现有基线进行比较。实验结果表明，该方法在攻击效果和攻击隐蔽性方面都具有明显的优势.我们的研究结果揭示了当前LLM中以前未报告的安全漏洞，并提供了一个新的视角来了解LLM的安全性如何受到损害，即使是良性的微调。我们的代码可以在https://github.com/ZHIXINXIE/tenBenign上找到。



## **33. Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks**

不一致时间：大型语言模型对对抗性攻击的鲁棒性的生存分析 cs.CL

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2510.02712v2) [paper-pdf](https://arxiv.org/pdf/2510.02712v2)

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model-drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.

摘要: 大型语言模型（LLM）彻底改变了对话人工智能，但对其在扩展多轮对话中的稳健性仍然知之甚少。现有的评估框架专注于静态基准和单轮评估，未能捕捉反映现实世界互动特征的对话退化的时间动态。在这项工作中，我们提出了对话稳健性的大规模生存分析，将故障建模为MT-Consistency基准上的9个最先进的LLM在36，951个回合内的事件时间过程。我们的框架将Cox比例风险、加速故障时间（AFT）和随机生存森林模型与简单的语义漂移特征相结合。我们发现，突然的从预定到提示的语义漂移会急剧增加不一致的风险，而累积漂移是反直觉的\{保护性}，这表明在经历多次转变的对话中进行适应。具有模型-漂移相互作用的AFT模型实现了区分和校准的最佳组合，比例风险检查揭示了关键漂移协变量的系统性违规，解释了这种环境下Cox式建模的局限性。最后，我们表明，轻量级的AFT模型可以转变为回合级风险监控器，它在第一个不一致的答案之前的几个回合标记大多数失败的对话，同时保持虚假警报适度。这些结果将生存分析确立为评估多轮稳健性和为对话式人工智能系统设计实用保障措施的强大范式。



## **34. Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data**

桌面上的边界：针对结构化数据的高效基于决策的黑匣子攻击 cs.LG

Paper revision

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2509.22850v3) [paper-pdf](https://arxiv.org/pdf/2509.22850v3)

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems.

摘要: 与视觉和语言领域相比，结构化数据中的对抗稳健性仍然是一个未充分探索的前沿。在这项工作中，我们引入了一种针对表格数据量身定制的新型黑匣子、基于决策的对抗攻击。我们的方法将无梯度方向估计与迭代边界搜索相结合，能够在最小的Oracle访问下高效导航离散和连续特征空间。大量实验表明，我们的方法成功地妥协了不同模型的几乎整个测试集，范围包括经典机器学习分类器和基于大型语言模型（LLM）的管道。值得注意的是，该攻击的成功率始终高于90%，而每个实例只需要少量查询。这些结果凸显了表格模型对对抗性扰动的严重脆弱性，凸显了现实世界决策系统中迫切需要更强的防御。



## **35. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

注意你的服务器：MCP生态系统上寄生工具链攻击的系统研究 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2509.06572v2) [paper-pdf](https://arxiv.org/pdf/2509.06572v2)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地与外部系统集成，该协议使工具调用同步化，并已迅速成为LLM支持的应用程序的支柱。虽然这种范式增强了功能，但它也引入了根本性的安全转变：LLM从被动信息处理器过渡到面向任务的工具链的自主编排，扩大了攻击面，将对抗目标从操纵单个输出提升到劫持整个执行流。在本文中，我们揭示了一类新的攻击，即寄生工具链攻击，实例化为LCP无意隐私泄露（MCP-UPD）。这些攻击不需要受害者直接互动;相反，对手会将恶意指令嵌入到LLM在合法任务期间访问的外部数据源中。恶意逻辑渗透到工具链中，并分三个阶段展开：寄生摄入、隐私收集和隐私披露，最终导致私人数据的秘密泄露。我们的根本原因分析表明，LCP缺乏上下文工具隔离和最低特权强制执行，使得对抗指令能够不受限制地传播到敏感工具调用中。为了评估严重性，我们设计了MCP-SEC，并对LCP生态系统进行了首次大规模安全普查，分析了1，360台服务器上的12，230个工具。我们的研究结果表明，LCP生态系统中充斥着可利用的小工具和多样化的攻击方法，凸显了LCP平台的系统性风险以及LLM集成环境中对防御机制的迫切需求。



## **36. Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position**

从哪里开始调整？扩散大语言模型可能需要独特的位置 cs.CR

Accepted for oral presentation at AAAI 2026

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2508.12398v2) [paper-pdf](https://arxiv.org/pdf/2508.12398v2)

**Authors**: Zhixin Xie, Xurui Song, Jun Luo

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a competitive non-autoregressive paradigm due to their unique training and inference approach. However, there is currently a lack of safety study on this novel architecture. In this paper, we present the first analysis of dLLMs' safety performance and propose a novel safety alignment method tailored to their unique generation characteristics. Specifically, we identify a critical asymmetry between the defender and attacker in terms of security. For the defender, we reveal that the middle tokens of the response, rather than the initial ones, are more critical to the overall safety of dLLM outputs; this seems to suggest that aligning middle tokens can be more beneficial to the defender. The attacker, on the contrary, may have limited power to manipulate middle tokens, as we find dLLMs have a strong tendency towards a sequential generation order in practice, forcing the attack to meet this distribution and diverting it from influencing the critical middle tokens. Building on this asymmetry, we introduce Middle-tOken Safety Alignment (MOSA), a novel method that directly aligns the model's middle generation with safe refusals exploiting reinforcement learning. We implement MOSA and compare its security performance against eight attack methods on two benchmarks. We also test the utility of MOSA-aligned dLLM on coding, math, and general reasoning. The results strongly prove the superiority of MOSA.

摘要: 扩散大型语言模型（dLLM）由于其独特的训练和推理方法，最近成为一种有竞争力的非自回归范式。然而，目前缺乏对这种新颖架构的安全研究。在本文中，我们首次分析了dLLM的安全性能，并提出了一种针对其独特世代特征量身定制的新型安全对齐方法。具体来说，我们发现防御者和攻击者之间在安全性方面存在严重的不对称性。对于防御者来说，我们揭示了响应的中间标记（而不是初始标记）对于dLLM输出的整体安全性更关键;这似乎表明对齐中间标记可能对防御者更有利。相反，攻击者操纵中间令牌的能力可能有限，因为我们发现dLLM在实践中有强烈的顺序生成顺序倾向，迫使攻击满足这种分布并转移其影响关键中间令牌。在这种不对称性的基础上，我们引入了Middle-tOken安全对齐（MOSA），这是一种新颖的方法，可以直接将模型的中间代与利用强化学习的安全拒绝对齐。我们实施MOSA并在两个基准测试上将其安全性能与八种攻击方法进行比较。我们还测试了与MOSA对齐的DLLM在编码、数学和一般推理方面的实用性。结果有力地证明了MOSA的优越性。



## **37. Special-Character Adversarial Attacks on Open-Source Language Model**

开源语言模型的特殊字符对抗攻击 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2508.14070v2) [paper-pdf](https://arxiv.org/pdf/2508.14070v2)

**Authors**: Ephraiem Sarabamoun

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, yet their vulnerability to character-level adversarial manipulations presents significant security challenges for real-world deployments. This paper presents a study of different special character attacks including unicode, homoglyph, structural, and textual encoding attacks aimed at bypassing safety mechanisms. We evaluate seven prominent open-source models ranging from 3.8B to 32B parameters on 4,000+ attack attempts. These experiments reveal critical vulnerabilities across all model sizes, exposing failure modes that include successful jailbreaks, incoherent outputs, and unrelated hallucinations.

摘要: 大型语言模型（LLM）在各种自然语言处理任务中取得了非凡的性能，但它们对字符级对抗性操纵的脆弱性给现实世界的部署带来了重大的安全挑战。本文研究了旨在绕过安全机制的不同特殊字符攻击，包括Unicode、同字形、结构性和文本编码攻击。我们对4，000多次攻击尝试进行了评估，参数范围从3.8B到32 B不等。这些实验揭示了所有模型尺寸的关键漏洞，揭示了包括成功越狱、不连贯的输出和不相关的幻觉在内的失败模式。



## **38. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

学习在大型视觉语言模型中检测未知越狱攻击 cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2508.09201v3) [paper-pdf](https://arxiv.org/pdf/2508.09201v3)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定于攻击的参数，这阻碍了对不可见攻击的概括，要么依赖于数学上合理的原则，这限制了准确性和效率。为了克服这些局限性，我们提出了学习检测（Lo），这是一个通用框架，通过将重点从特定攻击的学习转移到特定任务的学习来准确检测未知越狱攻击。该框架包括用于面向安全的表示学习的多模式安全概念激活载体模块和用于无监督攻击分类的安全模式自动编码器模块。大量的实验表明，我们的方法在提高效率的同时，对各种未知攻击实现了一致的较高检测AUROC。该代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB上获取。



## **39. Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation**

通过知识不对称利用从检索增强生成系统中进行细粒度隐私提取 cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2507.23229v2) [paper-pdf](https://arxiv.org/pdf/2507.23229v2)

**Authors**: Yufei Chen, Yao Wang, Haibin Zhang, Tao Gu

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge bases, but this advancement introduces significant privacy risks. Existing privacy attacks on RAG systems can trigger data leakage but often fail to accurately isolate knowledge-base-derived sentences within mixed responses. They also lack robustness when applied across multiple domains. This paper addresses these challenges by presenting a novel black-box attack framework that exploits knowledge asymmetry between RAG and standard LLMs to achieve fine-grained privacy extraction across heterogeneous knowledge landscapes. We propose a chain-of-thought reasoning strategy that creates adaptive prompts to steer RAG systems away from sensitive content. Specifically, we first decompose adversarial queries to maximize information disparity and then apply a semantic relationship scoring to resolve lexical and syntactic ambiguities. We finally train a neural network on these feature scores to precisely identify sentences containing private information. Unlike prior work, our framework generalizes to unseen domains through iterative refinement without pre-defined knowledge. Experimental results show that we achieve over 91% privacy extraction rate in single-domain and 83% in multi-domain scenarios, reducing sensitive sentence exposure by over 65% in case studies. This work bridges the gap between attack and defense in RAG systems, enabling precise extraction of private information while providing a foundation for adaptive mitigation.

摘要: 检索增强生成（RAG）系统通过集成外部知识库来增强大型语言模型（LLM），但这一进步带来了巨大的隐私风险。对RAG系统的现有隐私攻击可能会引发数据泄露，但通常无法准确地隔离混合响应中的知识库派生句子。当应用于多个领域时，它们也缺乏稳健性。本文通过提出一种新型的黑匣子攻击框架来解决这些挑战，该框架利用RAG和标准LLM之间的知识不对称性来实现跨异类知识环境的细粒度隐私提取。我们提出了一种思想链推理策略，可以创建自适应提示来引导RAG系统远离敏感内容。具体来说，我们首先分解对抗性查询以最大化信息差异，然后应用语义关系评分来解决词汇和语法歧义。我们最终根据这些特征分数训练神经网络，以精确识别包含私人信息的句子。与之前的工作不同，我们的框架通过迭代细化而无需预先定义的知识，将其推广到不可见的领域。实验结果表明，我们在单域场景中实现了超过91%的隐私提取率，在多域场景中实现了83%的隐私提取率，在案例研究中将敏感句子暴露减少了超过65%。这项工作弥合了RAG系统中攻击和防御之间的差距，能够精确提取私人信息，同时为自适应缓解提供基础。



## **40. Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models**

响应攻击：利用上下文启动来越狱大型语言模型 cs.CL

20 pages, 10 figures. Code and data available at https://github.com/Dtc7w3PQ/Response-Attack

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2507.05248v2) [paper-pdf](https://arxiv.org/pdf/2507.05248v2)

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. While existing jailbreak attacks largely rely on single-turn or multi-turn prompt manipulations, or inject static in-context examples, these methods suffer from limited effectiveness, inefficiency, or semantic drift. We introduce Response Attack (RA), a novel framework that strategically leverages intermediate, mildly harmful responses as contextual primers within a dialogue. By reformulating harmful queries and injecting these intermediate responses before issuing a targeted trigger prompt, RA exploits a previously overlooked vulnerability in LLMs. Extensive experiments across eight state-of-the-art LLMs show that RA consistently achieves significantly higher attack success rates than nine leading jailbreak baselines. Our results demonstrate that the success of RA is directly attributable to the strategic use of intermediate responses, which induce models to generate more explicit and relevant harmful content while maintaining stealth, efficiency, and fidelity to the original query. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.

摘要: 上下文启动（早期的刺激会秘密地偏向后来的判断）为大型语言模型（LLM）提供了一个尚未探索的攻击表面。我们发现了一个上下文启动漏洞，其中对话中的先前响应可以将其后续行为引导到违反政策的内容上。虽然现有的越狱攻击主要依赖于单轮或多轮提示操纵，或注入静态上下文示例，但这些方法的有效性有限、效率低下或语义漂移。我们引入了响应攻击（RA），这是一个新颖的框架，它战略性地利用中间的、轻度有害的响应作为对话中的上下文触发器。通过重新制定有害查询并在发出有针对性的触发提示之前注入这些中间响应，RA利用了LLM中以前被忽视的漏洞。对八个最先进的LLM进行的广泛实验表明，RA始终比九个领先的越狱基线实现显着更高的攻击成功率。我们的结果表明，RA的成功直接归因于中间响应的战略使用，中间响应会促使模型生成更明确和相关的有害内容，同时保持原始查询的隐秘性、效率和保真度。代码和数据可在https://github.com/Dtc7w3PQ/Response-Attack上获取。



## **41. Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines**

有条件扩散中的后门：对负责任的合成数据管道的威胁 cs.CV

Accepted at RDS @ AAAI 2026

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2507.04726v2) [paper-pdf](https://arxiv.org/pdf/2507.04726v2)

**Authors**: Raz Lapid, Almog Dubin

**Abstract**: Text-to-image diffusion models achieve high-fidelity image generation from natural language prompts. ControlNets extend these models by enabling conditioning on structural inputs (e.g., edge maps, depth, pose), providing fine-grained control over outputs. Yet their reliance on large, publicly scraped datasets and community fine-tuning makes them vulnerable to data poisoning. We introduce a model-poisoning attack that embeds a covert backdoor into a ControlNet, causing it to produce attacker-specified content when exposed to visual triggers, without textual prompts. Experiments show that poisoning only 1% of the fine-tuning corpus yields a 90-98% attack success rate, while 5% further strengthens the backdoor, all while preserving normal generation quality. To mitigate this risk, we propose clean fine-tuning (CFT): freezing the diffusion backbone and fine-tuning only the ControlNet on a sanitized dataset with a reduced learning rate. CFT lowers attack success rates on held-out data. These results expose a critical security weakness in open-source, ControlNet-guided diffusion pipelines and demonstrate that CFT offers a practical defense for responsible synthetic-data pipelines.

摘要: 文本到图像扩散模型实现了从自然语言提示生成高保真图像。Control Nets通过对结构性输入进行条件化来扩展这些模型（例如，边缘地图、深度、姿势），提供对输出的细粒度控制。然而，他们对大型公开抓取的数据集和社区微调的依赖使他们很容易受到数据中毒的影响。我们引入了一种模型中毒攻击，将秘密后门嵌入到控制网络中，导致其在暴露于视觉触发器时生成攻击者指定的内容，而没有文本提示。实验表明，仅毒害1%的微调数据库就会产生90-98%的攻击成功率，而5%则进一步加强了后门，同时保持正常的生成质量。为了降低这种风险，我们提出了干净微调（CFT）：冻结扩散主干并在学习率降低的净化数据集上仅微调控制Net。CFT降低了对持有数据的攻击成功率。这些结果暴露了开源、Control Net引导的传播管道中的一个关键安全弱点，并证明CFT为负责任的合成数据管道提供了实用的防御。



## **42. Large Language Model Unlearning for Source Code**

大型语言模型放弃源代码的学习 cs.SE

Accepted to AAAI'26

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2506.17125v2) [paper-pdf](https://arxiv.org/pdf/2506.17125v2)

**Authors**: Xue Jiang, Yihong Dong, Huangzhao Zhang, Tangxinyu Wang, Zheng Fang, Yingwei Ma, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Yongbin Li, Ge Li

**Abstract**: While Large Language Models (LLMs) excel at code generation, their inherent tendency toward verbatim memorization of training data introduces critical risks like copyright infringement, insecure emission, and deprecated API utilization, etc. A straightforward yet promising defense is unlearning, ie., erasing or down-weighting the offending snippets through post-training. However, we find its application to source code often tends to spill over, damaging the basic knowledge of programming languages learned by the LLM and degrading the overall capability. To ease this challenge, we propose PROD for precise source code unlearning. PROD surgically zeroes out the prediction probability of the prohibited tokens, and renormalizes the remaining distribution so that the generated code stays correct. By excising only the targeted snippets, PROD achieves precise forgetting without much degradation of the LLM's overall capability. To facilitate in-depth evaluation against PROD, we establish an unlearning benchmark consisting of three downstream tasks (ie., unlearning of copyrighted code, insecure code, and deprecated APIs), and introduce Pareto Dominance Ratio (PDR) metric, which indicates both the forget quality and the LLM utility. Our comprehensive evaluation demonstrates that PROD achieves superior overall performance between forget quality and model utility compared to existing unlearning approaches across three downstream tasks, while consistently exhibiting improvements when applied to LLMs of varying series. PROD also exhibits superior robustness against adversarial attacks without generating or exposing the data to be forgotten. These results underscore that our approach not only successfully extends the application boundary of unlearning techniques to source code, but also holds significant implications for advancing reliable code generation.

摘要: 虽然大型语言模型（LLM）擅长代码生成，但其固有的逐字记忆训练数据的倾向会引入版权侵权、不安全的发射和过时的API利用等关键风险。一个简单但有希望的防御是取消学习，即，通过训练后删除或降低违规片段的权重。然而，我们发现它对源代码的应用往往会溢出，损害LLM学到的编程语言的基本知识，并降低整体能力。为了缓解这一挑战，我们提出了PROD来精确的源代码反学习。PROD通过外科手术将被禁止的令牌的预测概率归零，并重新规范剩余的分布，以便生成的代码保持正确。通过仅删除目标片段，PROD实现了精确遗忘，而不会大幅降低LLM的整体能力。为了促进针对PROD的深入评估，我们建立了一个由三个下游任务（即，放弃受版权保护的代码、不安全的代码和废弃的API），并引入帕累托主导比（PDR）指标，该指标既指示忘记质量又指示LLM实用性。我们的全面评估表明，与三个下游任务中的现有取消学习方法相比，PROD在忘记质量和模型效用之间实现了更好的整体性能，同时在应用于不同系列的LLM时一致表现出改进。PROD还表现出针对对抗攻击的卓越鲁棒性，而不会生成或暴露被遗忘的数据。这些结果强调，我们的方法不仅成功地将放弃学习技术的应用边界扩展到源代码，而且对推进可靠的代码生成具有重要影响。



## **43. Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?**

保护检索数据的隐私免受成员推断攻击：此查询是否离家太近？ cs.CL

Accepted for EMNLP findings 2025

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2505.22061v3) [paper-pdf](https://arxiv.org/pdf/2505.22061v3)

**Authors**: Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook Lee, Jinseong Park

**Abstract**: Retrieval-augmented generation (RAG) mitigates the hallucination problem in large language models (LLMs) and has proven effective for personalized usages. However, delivering private retrieved documents directly to LLMs introduces vulnerability to membership inference attacks (MIAs), which try to determine whether the target data point exists in the private external database or not. Based on the insight that MIA queries typically exhibit high similarity to only one target document, we introduce a novel similarity-based MIA detection framework designed for the RAG system. With the proposed method, we show that a simple detect-and-hide strategy can successfully obfuscate attackers, maintain data utility, and remain system-agnostic against MIA. We experimentally prove its detection and defense against various state-of-the-art MIA methods and its adaptability to existing RAG systems.

摘要: 检索增强生成（RAG）缓解了大型语言模型（LLM）中的幻觉问题，并已被证明对个性化使用有效。然而，将私有检索到的文档直接传递到LLM会引入成员资格推断攻击（MIA）的漏洞，该攻击试图确定目标数据点是否存在于私有外部数据库中。基于MIA查询通常仅与一个目标文档表现出高相似性的认识，我们引入了一种为RAG系统设计的新型基于相似性的MIA检测框架。通过提出的方法，我们表明简单的检测和隐藏策略可以成功地混淆攻击者、保持数据效用并保持系统对MIA的不可知性。我们通过实验证明了它对各种最先进的MIA方法的检测和防御，以及它对现有RAG系统的适应性。



## **44. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

重温模型倒置评估：从误导性标准到可靠的隐私评估 cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM

摘要: 模型倒置（MI）攻击旨在通过利用对机器学习模型T的访问来从私人训练数据中重建信息。为了评估此类攻击，标准评估框架依赖于评估模型E，该模型在与T相同的任务设计下训练。该框架已成为评估MI研究进展的事实标准，几乎所有最近的MI研究都毫无疑问地使用了该框架。在本文中，我们对该评估框架进行了首次深入研究。特别是，我们确定了这个标准框架的一个关键问题：I型对抗性示例。这些重建并没有捕捉到私人训练数据的视觉特征，但仍然被T认为是成功的，并最终转移到E。这种假阳性损害了标准管理信息评价框架的可靠性。为了解决这个问题，我们引入了一个新的MI评估框架，用先进的多模态大型语言模型（MLLM）取代了评估模型E。通过利用他们的通用视觉理解，我们基于MLLM的框架不依赖于T中的共享任务设计的训练，从而降低了I型可转移性，并提供了更忠实的重建成功评估。使用我们基于MLLM的评估框架，我们重新评估了27种不同的MI攻击设置，并根据经验揭示了标准评估框架下一贯的高误报率。重要的是，我们证明了许多最先进的（SOTA）MI方法报告了夸大的攻击准确性，这表明实际的隐私泄露显着低于以前认为的。通过揭示这一关键问题并提出一个强有力的解决方案，我们的工作能够重新评估MI研究的进展，并为可靠和强大的评估设定了新的标准。代码可在https://github.com/hosytuyen/MI-Eval-MLLM找到



## **45. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片即可：用单个图像毒害视觉文档检索增强生成 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.

摘要: 检索增强生成（RAG）有助于通过使用事实知识库（KB）来抑制大型语言模型（LLM）中的幻觉。尽管PDF文档是重要的知识来源，但基于文本的RAG管道在捕获其丰富的多模式信息方面效率低下。相比之下，视觉文档RAG（VD-RAG）使用文档页面的屏幕截图作为KB，已被证明可以实现最先进的结果。然而，通过引入图像模式，VD-RAG为对手引入了新的攻击载体，通过将恶意文档注入知识库来破坏系统。在本文中，我们展示了VD-RAG对针对检索和生成的中毒攻击的脆弱性。我们定义了两个攻击目标，并证明这两个目标都可以通过仅向知识库中注入单个对抗图像来实现。首先，我们对一个或一组查询引入有针对性的攻击，目标是传播有针对性的虚假信息。其次，我们提出了一种通用攻击，对于任何潜在的用户查询，该攻击都会影响响应，从而导致VD-RAG系统中的拒绝服务。我们调查的两个攻击目标下的白盒和黑盒的假设，采用多目标的基于梯度的优化方法，以及促使国家的最先进的生成模型。使用两个可视化文档数据集，一组不同的最先进的检索器（嵌入模型）和生成器（视觉语言模型），我们表明VD-RAG在目标和通用设置中都容易受到中毒攻击，但在通用设置中表现出对黑盒攻击的鲁棒性。



## **46. Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation**

探索联邦军事LLM中潜在的即时注入攻击及其缓解措施 cs.LG

Accepted to the 3rd International Workshop on Dataspaces and Digital Twins for Critical Entities and Smart Urban Communities - IEEE BigData 2025

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18416v2) [paper-pdf](https://arxiv.org/pdf/2501.18416v2)

**Authors**: Youngjoon Lee, Taehyun Park, Yunho Lee, Jinu Gong, Joonhyuk Kang

**Abstract**: Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these risks, we propose a human-AI collaborative framework with both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols.

摘要: 联合学习（FL）越来越多地被用于军事合作，以开发大型语言模型（LLM），同时保留数据主权。然而，即时注入攻击（对输入预算的恶意操纵）构成了新的威胁，可能会破坏运营安全、扰乱决策并削弱盟友之间的信任。这篇观点论文强调了联邦军事LLM中的四个漏洞：秘密数据泄露、搭便车剥削、系统中断和错误信息传播。为了应对这些风险，我们提出了一个具有技术和政策对策的人与人工智能协作框架。在技术方面，我们的框架使用红/蓝团队战争游戏和质量保证来检测和减轻共享LLM权重的对抗行为。在政策方面，它促进人工智能与人类联合政策制定和安全协议验证。



## **47. DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs**

DarkMind：定制LLC中潜在的思想链后门 cs.CR

19 pages, 15 figures, 12 tables

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18617v2) [paper-pdf](https://arxiv.org/pdf/2501.18617v2)

**Authors**: Zhen Guo, Shanghao Shi, Shamim Yazdani, Ning Zhang, Reza Tourani

**Abstract**: With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.

摘要: 随着个性化人工智能的迅速崛起，配备思想链（COT）推理的定制大型语言模型（LLM）现在为数百万人工智能代理提供动力。然而，它们复杂的推理过程会引入新的且基本上未被探索的安全漏洞。我们提出了DarkMind，这是一种新颖的潜在推理级后门攻击，通过在不改变用户查询的情况下操纵内部COT步骤来针对自定义的LLM。与之前的基于提示的攻击不同，DarkMind通过潜在触发器在推理链中秘密激活，无需修改输入提示或要求访问模型参数即可实现对抗行为。为了实现隐形和可靠性，我们提出了即时和追溯双重触发类型，并将它们集成到统一的嵌入模板中，该模板管理触发相关激活，采用隐形优化算法来最大限度地减少语义漂移，并引入自动对话启动器跨域的秘密激活。使用五个LLM对跨越算术、常识和符号领域的八个推理数据集进行了全面实验，证明DarkMind始终实现了高攻击成功率。我们进一步研究了减轻这些风险的防御策略，并揭示了推理级后门代表了一个重大但未充分探索的威胁，强调了对强大、推理感知安全机制的必要性。



## **48. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

ATA：通过简单辅助任务链接实现LLM越狱的典范 cs.CR

ACL Findings 2025. Welcome to employ SATA as a baseline

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2412.15289v5) [paper-pdf](https://arxiv.org/pdf/2412.15289v5)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.

摘要: 大型语言模型（LLM）在各种任务中取得了重大进展，但它们的安全性一致仍然是一个主要问题。探索越狱提示可以暴露LLM的漏洞并指导保护它们的工作。现有的方法主要设计复杂的指令供LLM遵循，或者依赖于多次迭代，这可能会阻碍越狱的性能和效率。在这项工作中，我们提出了一种新颖的越狱范式--简单辅助任务链接（ATA），它可以有效地规避LLM保障措施并引发有害反应。具体来说，ATA首先屏蔽恶意查询中的有害关键词，以生成包含一个或多个[MASK]特殊令牌的相对良性的查询。然后，它采用简单的辅助任务，例如掩蔽语言模型任务或按位置查找元素任务来编码掩蔽关键词的语义。最后，ATA将辅助任务与屏蔽查询链接起来，共同执行越狱。大量实验表明，ATA实现了最先进的性能，并且大幅优于基线。具体来说，在AdvBench数据集上，通过屏蔽语言模型（MLM）辅助任务，ATA的总体攻击成功率（ASB）达到85%，有害评分（HS）达到4.57，通过按位置查找元素（ELP）辅助任务，ATA的总体攻击成功率（ASB）达到76%，HS达到4.43。



## **49. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

大型语言模型中的漏洞越狱和缓解 cs.CR

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2410.15236v3) [paper-pdf](https://arxiv.org/pdf/2410.15236v3)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin, Xinyuan Song

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.

摘要: 大型语言模型（LLM）通过推进自然语言理解和生成来改变了人工智能，实现了医疗保健、软件工程和对话系统以外的各个领域的应用。尽管过去几年取得了这些进步，但LLM仍表现出相当大的漏洞，特别是在引发注射和越狱攻击方面。本评论分析了这些漏洞的研究状况，并提出了可用的防御策略。我们将攻击方法大致分为基于预算、基于模型、多模式和多语言，涵盖对抗提示、后门注入和跨模式利用等技术。我们还回顾了各种防御机制，包括即时过滤、转换、对齐技术、多智能体防御和自我调节，评估它们的优点和缺点。我们还讨论了用于评估LLM安全性和稳健性的关键指标和基准，并指出了交互式环境中攻击成功的量化以及现有数据集中的偏差等挑战。通过识别当前的研究差距，我们提出了弹性对齐策略、针对不断发展的攻击的先进防御、越狱检测自动化以及道德和社会影响的未来方向。该审查强调了人工智能社区内持续研究与合作的必要性，以增强LLM安全性并确保其安全部署。



## **50. Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks**

保护大型语言模型：解决偏见、错误信息和即时攻击 cs.CR

17 pages, 1 figure

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2409.08087v3) [paper-pdf](https://arxiv.org/pdf/2409.08087v3)

**Authors**: Benji Peng, Keyu Chen, Ming Li, Pohsun Feng, Ziqian Bi, Junyu Liu, Xinyuan Song, Qian Niu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across various fields, yet their increasing use raises critical security concerns. This article reviews recent literature addressing key issues in LLM security, with a focus on accuracy, bias, content detection, and vulnerability to attacks. Issues related to inaccurate or misleading outputs from LLMs is discussed, with emphasis on the implementation from fact-checking methodologies to enhance response reliability. Inherent biases within LLMs are critically examined through diverse evaluation techniques, including controlled input studies and red teaming exercises. A comprehensive analysis of bias mitigation strategies is presented, including approaches from pre-processing interventions to in-training adjustments and post-processing refinements. The article also probes the complexity of distinguishing LLM-generated content from human-produced text, introducing detection mechanisms like DetectGPT and watermarking techniques while noting the limitations of machine learning enabled classifiers under intricate circumstances. Moreover, LLM vulnerabilities, including jailbreak attacks and prompt injection exploits, are analyzed by looking into different case studies and large-scale competitions like HackAPrompt. This review is concluded by retrospecting defense mechanisms to safeguard LLMs, accentuating the need for more extensive research into the LLM security field.

摘要: 大型语言模型（LLM）在各个领域都展现出令人印象深刻的能力，但它们的使用越来越多地引发了关键的安全问题。本文回顾了最近解决LLM安全关键问题的文献，重点关注准确性、偏差、内容检测和攻击脆弱性。讨论了与LLM的不准确或误导性输出相关的问题，重点是通过事实核查方法的实施来增强响应的可靠性。通过各种评估技术（包括受控输入研究和红色团队练习），批判性地检查了法学硕士内部的固有偏见。对偏见缓解策略进行了全面分析，包括从预处理干预到训练中调整和后处理改进的方法。本文还探讨了将LLM生成的内容与人类生成的文本区分开来的复杂性，引入了DetectGPT和水印技术等检测机制，同时指出了机器学习支持分类器在复杂情况下的局限性。此外，还通过研究不同的案例研究和HackAPrompt等大规模竞赛来分析LLM漏洞，包括越狱攻击和即时注入漏洞。本审查的最后回顾了保护LLM的防御机制，强调了对LLM安全领域进行更广泛研究的必要性。



