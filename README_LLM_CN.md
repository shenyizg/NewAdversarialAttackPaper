# Latest Large Language Model Attack Papers
**update at 2025-11-25 15:22:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization**

通过树群双感知搜索和优化实现LLM安全性调整的对抗性攻击-防御协同进化 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.19218v1) [paper-pdf](https://arxiv.org/pdf/2511.19218v1)

**Authors**: Xurui Li, Kaisong Song, Rui Zhu, Pin-Yu Chen, Haixu Tang

**Abstract**: Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.

摘要: 大型语言模型（LLM）在网络服务中迅速发展，提供了前所未有的能力，同时放大了社会风险。现有的作品往往专注于孤立的越狱攻击或静态防御，忽视了现实世界网络环境中不断变化的威胁与保障措施之间的动态相互作用。为了缓解这些挑战，我们提出了ACE安全（针对LLM安全的对抗协同进化），一个新颖的框架，通过无缝集成两个关键的创新过程来联合优化攻击和防御模型：（1）群体感知策略引导的蒙特卡洛树搜索（GS-MCTS），它有效地探索越狱策略以发现漏洞并生成多样化的对抗样本;（2）对抗性课程树感知群组策略优化（AC-TSYS），通过课程强化学习，利用具有挑战性的样本联合训练攻击和防御LLM，实现稳健的相互改进。多个基准的评估表明，我们的方法优于现有的攻击和防御方法，并为开发能够可持续支持负责任的人工智能生态系统的LLM提供了可行的途径。



## **2. Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations**

以负责任的人工智能考虑保护大型语言模型免受越狱利用 cs.CR

20 pages including appendix; technical report; NeurIPS 2024 style

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18933v1) [paper-pdf](https://arxiv.org/pdf/2511.18933v1)

**Authors**: Ryan Wong, Hosea David Yu Fei Ng, Dhananjai Sharma, Glenn Jun Jie Ng, Kavishvaran Srinivasan

**Abstract**: Large Language Models (LLMs) remain susceptible to jailbreak exploits that bypass safety filters and induce harmful or unethical behavior. This work presents a systematic taxonomy of existing jailbreak defenses across prompt-level, model-level, and training-time interventions, followed by three proposed defense strategies. First, a Prompt-Level Defense Framework detects and neutralizes adversarial inputs through sanitization, paraphrasing, and adaptive system guarding. Second, a Logit-Based Steering Defense reinforces refusal behavior through inference-time vector steering in safety-sensitive layers. Third, a Domain-Specific Agent Defense employs the MetaGPT framework to enforce structured, role-based collaboration and domain adherence. Experiments on benchmark datasets show substantial reductions in attack success rate, achieving full mitigation under the agent-based defense. Overall, this study highlights how jailbreaks pose a significant security threat to LLMs and identifies key intervention points for prevention, while noting that defense strategies often involve trade-offs between safety, performance, and scalability. Code is available at: https://github.com/Kuro0911/CS5446-Project

摘要: 大型语言模型（LLM）仍然容易受到越狱漏洞利用的影响，这些漏洞绕过安全过滤器并引发有害或不道德行为。这项工作对预算级、模型级和训练时干预措施的现有越狱防御进行了系统分类，然后提出了三种拟议的防御策略。首先，预算级防御框架通过净化、重述和自适应系统防护来检测并中和对抗输入。其次，基于日志的转向防御通过安全敏感层中的推理时间载体转向来加强拒绝行为。第三，领域特定代理防御采用MetaGPT框架来实施结构化的、基于角色的协作和领域遵守。对基准数据集的实验显示，攻击成功率大幅降低，在基于代理的防御下实现了全面缓解。总体而言，这项研究强调了越狱如何对LLM构成重大安全威胁，并确定了预防的关键干预点，同时指出防御策略通常涉及安全性、性能和可扩展性之间的权衡。代码可访问：https://github.com/Kuro0911/CS5446-Project



## **3. BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models**

BackdoorVLM：视觉语言模型后门攻击的基准 cs.CV

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18921v1) [paper-pdf](https://arxiv.org/pdf/2511.18921v1)

**Authors**: Juncheng Li, Yige Li, Hanxun Huang, Yunhao Chen, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM .

摘要: 后门攻击通过注入可能在推理时被恶意激活的隐藏行为来破坏机器学习系统的可靠性和可信性。虽然此类威胁在单模式环境中得到了广泛研究，但它们对多模式基础模型（尤其是视觉语言模型（VLM）的影响在很大程度上仍然没有得到充分的研究。在这项工作中，我们引入了\textBF{BackdoorVLM}，这是第一个用于在广泛的设置中系统评估对VLM的后门攻击的全面基准。它采用统一的视角，在核心视觉语言任务（包括图像字幕和视觉问答）中注入和分析后门。BackdoorVLM将多模式后门威胁分为5个代表性类别：定向拒绝、恶意注入、越狱、概念替代和感知劫持。每个类别都捕获了对手可以操纵模型行为的独特途径。我们使用涵盖文本、图像和双峰触发器的12种代表性攻击方法来评估这些威胁，并在2个开源VLM和3个多模式数据集上进行了测试。我们的分析表明，VLM对文本指令表现出很强的敏感性，并且在双峰后门中，文本触发器在形成后门映射时通常会触发图像触发器。值得注意的是，涉及文本形式的后门仍然非常有效，中毒率低至1%，大多数任务的成功率超过90%。这些发现凸显了当前VLM中先前未充分探索的重大漏洞。我们希望BackdoorVLM能够成为分析和缓解多模式后门威胁的有用基准。代码可访问：https://github.com/bin015/BackdoorVLM。



## **4. RoguePrompt: Dual-Layer Ciphering for Self-Reconstruction to Circumvent LLM Moderation**

RoguePrompt：用于自我重构以规避LLM调节的双层加密 cs.CR

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.18790v1) [paper-pdf](https://arxiv.org/pdf/2511.18790v1)

**Authors**: Benyamin Tafreshian

**Abstract**: Content moderation pipelines for modern large language models combine static filters, dedicated moderation services, and alignment tuned base models, yet real world deployments still exhibit dangerous failure modes. This paper presents RoguePrompt, an automated jailbreak attack that converts a disallowed user query into a self reconstructing prompt which passes provider moderation while preserving the original harmful intent. RoguePrompt partitions the instruction across two lexical streams, applies nested classical ciphers, and wraps the result in natural language directives that cause the target model to decode and execute the hidden payload. Our attack assumes only black box access to the model and to the associated moderation endpoint. We instantiate RoguePrompt against GPT 4o and evaluate it on 2 448 prompts that a production moderation system previously marked as strongly rejected. Under an evaluation protocol that separates three security relevant outcomes bypass, reconstruction, and execution the attack attains 84.7 percent bypass, 80.2 percent reconstruction, and 71.5 percent full execution, substantially outperforming five automated jailbreak baselines. We further analyze the behavior of several automated and human aligned evaluators and show that dual layer lexical transformations remain effective even when detectors rely on semantic similarity or learned safety rubrics. Our results highlight systematic blind spots in current moderation practice and suggest that robust deployment will require joint reasoning about user intent, decoding workflows, and model side computation rather than surface level toxicity alone.

摘要: 现代大型语言模型的内容审核管道结合了静态过滤器、专用审核服务和对齐优化的基本模型，但现实世界的部署仍然表现出危险的失败模式。本文介绍了RoguePrompt，这是一种自动越狱攻击，可将不允许的用户查询转换为自我重建提示，该提示通过提供商审核，同时保留最初的有害意图。RoguePrompt将指令划分为两个词汇流，应用嵌套的经典密码，并将结果包装在自然语言指令中，从而使目标模型解码和执行隐藏的有效负载。我们的攻击假设只有黑匣子访问模型和相关的审核端点。我们针对GPT 4o实例化RoguePrompt，并在生产审核系统之前标记为强烈拒绝的2 448个提示上对其进行评估。在将三种安全相关结果分开的评估协议下，攻击的绕过率为84.7%，重建率为80.2%，完全执行率为71.5%，大大优于五个自动越狱基线。我们进一步分析了几个自动化和人工对齐的评估者的行为，并表明即使检测器依赖于语义相似性或习得的安全规则，双层词汇转换仍然有效。我们的结果强调了当前审核实践中的系统盲点，并表明稳健的部署需要对用户意图、解码工作流程和模型端计算进行联合推理，而不仅仅是表面级别的毒性。



## **5. Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems**

代码中的阴影：探索基于LLM的多代理软件开发系统的风险和防御 cs.CR

Accepted by AAAI 2026 Alignment Track

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18467v1) [paper-pdf](https://arxiv.org/pdf/2511.18467v1)

**Authors**: Xiaoqing Wang, Keman Huang, Bin Liang, Hongyu Li, Xiaoyong Du

**Abstract**: The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% in MU-BA scenarios, and 71%, 84%, and 45% in BU-MA scenarios. Our defense mechanism reduced attack success rates significantly, particularly in the MU-BA scenario. Further analysis reveals that compromised agents in the coding and testing phases pose significantly greater security risks, while also identifying critical agents that require protection against malicious user exploitation. Our findings highlight the urgent need for robust security measures in multi-agent software development systems and provide practical guidelines for implementing targeted, resource-efficient defensive strategies.

摘要: 大语言模型（LLM）驱动的多智能体系统的快速发展，大大简化了软件开发任务，使用户几乎没有技术专长，开发可执行的应用程序。虽然这些系统通过自然语言需求使软件创建民主化，但它们引入了重大的安全风险，这些风险在很大程度上尚未被探索。我们确定了两种风险情况：恶意用户与良性代理（MU-BA）和良性用户与恶意代理（BU-MA）。我们介绍了隐式恶意行为注入攻击（IMBIA），演示了如何多代理系统可以被操纵，以产生隐藏的恶意功能下看似良性的应用程序的软件，并提出Adv-IMBIA作为一种防御机制。ChatDev、MetaGPT和AgentVerse框架的评估揭示了不同的漏洞模式，IMBIA在MU-BA场景中的攻击成功率为93%、45%和71%，在BU-MA场景中的攻击成功率为71%、84%和45%。我们的防御机制显着降低了攻击成功率，特别是在MU-BA的情况下。进一步的分析表明，在编码和测试阶段受影响的代理会带来显着更大的安全风险，同时还可以识别需要保护以防止恶意用户利用的关键代理。我们的研究结果强调了多代理软件开发系统中对强有力的安全措施的迫切需要，并为实施有针对性的、资源高效的防御策略提供了实用指南。



## **6. Think Fast: Real-Time IoT Intrusion Reasoning Using IDS and LLMs at the Edge Gateway**

快速思考：在边缘网关使用IDS和LLM进行实时物联网入侵推理 cs.CR

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18230v1) [paper-pdf](https://arxiv.org/pdf/2511.18230v1)

**Authors**: Saeid Jamshidi, Amin Nikanjam, Negar Shahabi, Kawser Wazed Nafi, Foutse Khomh, Samira Keivanpour, Rolando Herrero

**Abstract**: As the number of connected IoT devices continues to grow, securing these systems against cyber threats remains a major challenge, especially in environments with limited computational and energy resources. This paper presents an edge-centric Intrusion Detection System (IDS) framework that integrates lightweight machine learning (ML) based IDS models with pre-trained large language models (LLMs) to improve detection accuracy, semantic interpretability, and operational efficiency at the network edge. The system evaluates six ML-based IDS models: Decision Tree (DT), K-Nearest Neighbors (KNN), Random Forest (RF), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM model on low-power edge gateways, achieving accuracy up to 98 percent under real-world cyberattacks. For anomaly detection, the system transmits a compact and secure telemetry snapshot (for example, CPU usage, memory usage, latency, and energy consumption) via low-bandwidth API calls to LLMs including GPT-4-turbo, DeepSeek V2, and LLaMA 3.5. These models use zero-shot, few-shot, and chain-of-thought reasoning to produce human-readable threat analyses and actionable mitigation recommendations. Evaluations across diverse attacks such as DoS, DDoS, brute force, and port scanning show that the system enhances interpretability while maintaining low latency (<1.5 s), minimal bandwidth usage (<1.2 kB per prompt), and energy efficiency (<75 J), demonstrating its practicality and scalability as an IDS solution for edge gateways.

摘要: 随着互联物联网设备数量的持续增长，保护这些系统免受网络威胁仍然是一项重大挑战，尤其是在计算和能源资源有限的环境中。本文提出了一种以边缘为中心的入侵检测系统（IDS）框架，该框架将基于轻量级机器学习（ML）的IDS模型与预训练的大型语言模型（LLM）集成在一起，以提高网络边缘的检测准确性、语义解释性和运营效率。该系统评估了六种基于ML的IDS模型：决策树（DT）、K近邻（KNN）、随机森林（RF）、卷积神经网络（CNN）、长短期记忆（LSTM）和低功耗边缘网关上的混合CNN-LSTM模型，在现实世界的网络攻击下实现高达98%的准确率。对于异常检测，系统通过低带宽API调用向包括GPT-4-涡轮、DeepSeek V2和LLaMA 3.5在内的LLM传输紧凑且安全的遥感快照（例如，中央处理器使用率、内存使用率、延迟和能源消耗）。这些模型使用零射击、少射击和思想链推理来生成人类可读的威胁分析和可操作的缓解建议。对多种攻击（例如：拒绝服务、拒绝服务、暴力攻击和端口扫描）的评估表明，该系统增强了可解释性，同时保持低延迟（<1.5秒）、最低带宽使用（每次提示<1.2 kB）和能源效率（<75 J），证明了其作为边缘网关IDS解决方案的实用性和可扩展性。



## **7. ASTRA: Agentic Steerability and Risk Assessment Framework**

ASTRA：广义可操纵性和风险评估框架 cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.18114v1) [paper-pdf](https://arxiv.org/pdf/2511.18114v1)

**Authors**: Itay Hazan, Yael Mathov, Guy Shtar, Ron Bitton, Itsik Mantin

**Abstract**: Securing AI agents powered by Large Language Models (LLMs) represents one of the most critical challenges in AI security today. Unlike traditional software, AI agents leverage LLMs as their "brain" to autonomously perform actions via connected tools. This capability introduces significant risks that go far beyond those of harmful text presented in a chatbot that was the main application of LLMs. A compromised AI agent can deliberately abuse powerful tools to perform malicious actions, in many cases irreversible, and limited solely by the guardrails on the tools themselves and the LLM ability to enforce them. This paper presents ASTRA, a first-of-its-kind framework designed to evaluate the effectiveness of LLMs in supporting the creation of secure agents that enforce custom guardrails defined at the system-prompt level (e.g., "Do not send an email out of the company domain," or "Never extend the robotic arm in more than 2 meters").   Our holistic framework simulates 10 diverse autonomous agents varying between a coding assistant and a delivery drone equipped with 37 unique tools. We test these agents against a suite of novel attacks developed specifically for agentic threats, inspired by the OWASP Top 10 but adapted to challenge the ability of the LLM for policy enforcement during multi-turn planning and execution of strict tool activation. By evaluating 13 open-source, tool-calling LLMs, we uncovered surprising and significant differences in their ability to remain secure and keep operating within their boundaries. The purpose of this work is to provide the community with a robust and unified methodology to build and validate better LLMs, ultimately pushing for more secure and reliable agentic AI systems.

摘要: 保护由大型语言模型（LLM）支持的AI代理是当今AI安全中最关键的挑战之一。与传统软件不同，人工智能代理利用LLM作为他们的“大脑”，通过连接的工具自主执行操作。这种能力带来了重大的风险，远远超出了聊天机器人中呈现的有害文本的风险，而聊天机器人是LLM的主要应用。一个被入侵的人工智能代理可以故意滥用强大的工具来执行恶意操作，在许多情况下是不可逆的，并且仅受工具本身的护栏和LLM执行它们的能力的限制。本文介绍了ASTRA，这是一个首创的框架，旨在评估LLM在支持创建安全代理方面的有效性，这些代理强制执行在系统提示级别定义的自定义护栏（例如，“请勿将电子邮件发送到公司域名之外”，或“切勿将机械臂伸出超过2米”）。   我们的整体框架模拟了10个不同的自主代理，从编码助理到配备37种独特工具的送货无人机。我们针对专门针对代理威胁开发的一套新型攻击来测试这些代理，这些攻击受到OWASP Top 10的启发，但经过调整，以挑战LLM在多回合规划和执行严格工具激活期间政策执行的能力。通过评估13个开源、工具调用LLM，我们发现它们在保持安全和在其范围内运营的能力方面存在令人惊讶且显着的差异。这项工作的目的是为社区提供一种强大而统一的方法来构建和验证更好的LLM，最终推动更安全和可靠的代理人工智能系统。



## **8. Steering in the Shadows: Causal Amplification for Activation Space Attacks in Large Language Models**

在阴影中操纵：大型语言模型中激活空间攻击的因果放大 cs.CR

31 pages, 5 figures, 9 tables

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.17194v1) [paper-pdf](https://arxiv.org/pdf/2511.17194v1)

**Authors**: Zhiyuan Xu, Stanislav Abaimov, Joseph Gardiner, Sana Belguith

**Abstract**: Modern large language models (LLMs) are typically secured by auditing data, prompts, and refusal policies, while treating the forward pass as an implementation detail. We show that intermediate activations in decoder-only LLMs form a vulnerable attack surface for behavioral control. Building on recent findings on attention sinks and compression valleys, we identify a high-gain region in the residual stream where small, well-aligned perturbations are causally amplified along the autoregressive trajectory--a Causal Amplification Effect (CAE). We exploit this as an attack surface via Sensitivity-Scaled Steering (SSS), a progressive activation-level attack that combines beginning-of-sequence (BOS) anchoring with sensitivity-based reinforcement to focus a limited perturbation budget on the most vulnerable layers and tokens. We show that across multiple open-weight models and four behavioral axes, SSS induces large shifts in evil, hallucination, sycophancy, and sentiment while preserving high coherence and general capabilities, turning activation steering into a concrete security concern for white-box and supply-chain LLM deployments.

摘要: 现代大型语言模型（LLM）通常通过审计数据、提示和拒绝策略来保护，同时将转发视为实现细节。我们表明，仅限解码器的LLM中的中间激活形成了行为控制的脆弱攻击表面。基于最近关于注意力汇和压缩谷的研究结果，我们在剩余流中识别出了一个高收益区域，其中微小的、排列整齐的扰动沿着自回归轨迹被因果放大--因果放大效应（CAE）。我们通过灵敏度缩放转向（SS）将其作为攻击表面，这是一种渐进式激活级攻击，将序列识别（BOS）锚定与基于灵敏度的强化相结合，将有限的扰动预算集中在最脆弱的层和令牌上。我们表明，在多个开放权重模型和四个行为轴中，SS会引发邪恶、幻觉、谄媚和情绪的巨大转变，同时保持高度一致性和通用能力，将激活引导变成白盒和供应链LLM部署的具体安全问题。



## **9. MultiPriv: Benchmarking Individual-Level Privacy Reasoning in Vision-Language Models**

MultiPriv：视觉语言模型中的个人隐私推理基准 cs.CV

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.16940v1) [paper-pdf](https://arxiv.org/pdf/2511.16940v1)

**Authors**: Xiongtao Sun, Hui Li, Jiaming Zhang, Yujie Yang, Kaili Liu, Ruxin Feng, Wen Jun Tan, Wei Yang Bryan Lim

**Abstract**: Modern Vision-Language Models (VLMs) demonstrate sophisticated reasoning, escalating privacy risks beyond simple attribute perception to individual-level linkage. Current privacy benchmarks are structurally insufficient for this new threat, as they primarily evaluate privacy perception while failing to address the more critical risk of privacy reasoning: a VLM's ability to infer and link distributed information to construct individual profiles. To address this critical gap, we propose \textbf{MultiPriv}, the first benchmark designed to systematically evaluate individual-level privacy reasoning in VLMs. We introduce the \textbf{Privacy Perception and Reasoning (PPR)} framework and construct a novel, bilingual multimodal dataset to support it. The dataset uniquely features a core component of synthetic individual profiles where identifiers (e.g., faces, names) are meticulously linked to sensitive attributes. This design enables nine challenging tasks evaluating the full PPR spectrum, from attribute detection to cross-image re-identification and chained inference. We conduct a large-scale evaluation of over 50 foundational and commercial VLMs. Our analysis reveals: (1) Many VLMs possess significant, unmeasured reasoning-based privacy risks. (2) Perception-level metrics are poor predictors of these reasoning risks, revealing a critical evaluation gap. (3) Existing safety alignments are inconsistent and ineffective against such reasoning-based attacks. MultiPriv exposes systemic vulnerabilities and provides the necessary framework for developing robust, privacy-preserving VLMs.

摘要: 现代视觉语言模型（VLM）展示了复杂的推理，将隐私风险从简单的属性感知升级到个人层面的联系。当前的隐私基准在结构上不足以应对这种新威胁，因为它们主要评估隐私感知，而未能解决隐私推理的更关键风险：VLM推断和链接分布式信息以构建个人配置文件的能力。为了解决这一关键差距，我们提出了\textBF{MultiPriv}，这是第一个旨在系统评估VLM中个人级别隐私推理的基准。我们引入\textBF{Privacy Percept and Reasoning（PPR）}框架，并构建一个新颖的双语多模式数据集来支持它。该数据集独特地具有合成个人资料的核心组件，其中标识符（例如，面孔、名字）与敏感属性细致地关联起来。该设计实现了九项具有挑战性的任务，评估完整的PPR谱，从属性检测到跨图像重新识别和连锁推理。我们对50多个基础和商业VLM进行了大规模评估。我们的分析揭示了：（1）许多VLM都具有重大的、不可测量的基于推理的隐私风险。(2)感知水平指标无法预测这些推理风险，揭示了关键的评估差距。(3)现有的安全排列不一致，对于此类基于推理的攻击无效。MultiPriv暴露了系统性漏洞，并为开发稳健、保护隐私的VLM提供了必要的框架。



## **10. Evaluating Adversarial Vulnerabilities in Modern Large Language Models**

评估现代大型语言模型中的对抗脆弱性 cs.CR

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.17666v1) [paper-pdf](https://arxiv.org/pdf/2511.17666v1)

**Authors**: Tom Perel

**Abstract**: The recent boom and rapid integration of Large Language Models (LLMs) into a wide range of applications warrants a deeper understanding of their security and safety vulnerabilities. This paper presents a comparative analysis of the susceptibility to jailbreak attacks for two leading publicly available LLMs, Google's Gemini 2.5 Flash and OpenAI's GPT-4 (specifically the GPT-4o mini model accessible in the free tier). The research utilized two main bypass strategies: 'self-bypass', where models were prompted to circumvent their own safety protocols, and 'cross-bypass', where one model generated adversarial prompts to exploit vulnerabilities in the other. Four attack methods were employed - direct injection, role-playing, context manipulation, and obfuscation - to generate five distinct categories of unsafe content: hate speech, illegal activities, malicious code, dangerous content, and misinformation. The success of the attack was determined by the generation of disallowed content, with successful jailbreaks assigned a severity score. The findings indicate a disparity in jailbreak susceptibility between 2.5 Flash and GPT-4, suggesting variations in their safety implementations or architectural design. Cross-bypass attacks were particularly effective, indicating that an ample amount of vulnerabilities exist in the underlying transformer architecture. This research contributes a scalable framework for automated AI red-teaming and provides data-driven insights into the current state of LLM safety, underscoring the complex challenge of balancing model capabilities with robust safety mechanisms.

摘要: 最近大型语言模型（LLM）的蓬勃发展和快速集成到广泛的应用程序中，这使得人们需要更深入地了解其安全性和安全漏洞。本文对两种领先的公开LLM（Google的Gemini 2.5 Flash和OpenAI的GPT-4（特别是免费层中可访问的GPT-4o mini型号）进行了越狱攻击的易感性进行了比较分析。该研究利用了两种主要的绕过策略：“自我绕过”（提示模型绕过自己的安全协议）和“交叉绕过”（其中一个模型生成对抗提示以利用另一个模型的漏洞）。使用了四种攻击方法--直接注入、角色扮演、上下文操纵和混淆--来生成五种不同类别的不安全内容：仇恨言论、非法活动、恶意代码、危险内容和错误信息。攻击的成功取决于不允许内容的生成，成功的越狱会被赋予严重性分数。研究结果表明，2.5 Flash和GPT-4之间的越狱敏感性存在差异，这表明它们的安全实现或架构设计存在差异。交叉旁路攻击特别有效，这表明底层变压器架构中存在大量漏洞。这项研究为自动化人工智能红色团队提供了一个可扩展的框架，并提供了对LLM安全当前状态的数据驱动见解，强调了平衡模型能力与强大安全机制的复杂挑战。



## **11. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

基于大语言模型的深度强化学习驱动的自主网络防御奖励设计 cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.

摘要: 对于主题专家来说，在复杂、动态的环境中为自主网络攻击和防御学习代理设计奖励是一项具有挑战性的任务。我们提出了一种基于大语言模型（LLM）的奖励设计方法，以在深度强化学习（DRL）驱动的实验模拟环境中生成自主网络防御策略。精心设计了多个攻击和防御代理角色，反映了代理动作的多样性，以生成LLM引导的奖励设计，其中LLM首先被提供上下文网络模拟环境信息。然后在DRL驱动的攻击防御模拟环境中使用这些奖励结构来学习一整套网络防御策略。我们的结果表明，LLM指导的奖励设计可以制定针对不同对抗行为的有效防御策略。



## **12. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

Q-MLLM：鲁棒多模式大型语言模型安全性的载体量化 cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.

摘要: 多模式大型语言模型（MLLM）在跨模式理解方面表现出了令人印象深刻的能力，但尽管具有强大的文本安全机制，但仍然容易受到视觉输入的对抗攻击。这些漏洞源于两个核心弱点：视觉表示的连续性（允许基于梯度的攻击）以及基于文本的安全机制向视觉内容的不充分转移。我们引入了Q-MLLM，这是一种新颖的架构，它集成了两级量化，以创建针对对抗性攻击的离散瓶颈，同时保留多模式推理能力。通过在像素补丁和语义层面离散化视觉表示，Q-MLLM阻止攻击途径并弥合跨模式安全对齐差距。我们的两阶段训练方法确保稳健的学习，同时保持模型效用。实验表明，与现有方法相比，Q-MLLM在针对越狱攻击和有毒图像攻击的防御成功率明显更高。值得注意的是，Q-MLLM在针对越狱攻击时实现了完美的防御成功率（100%），但在一种可发现的情况下，同时以最小的推理费用在多个实用工具基准上保持竞争性能。这项工作将载体量化建立为安全多模式人工智能系统的有效防御机制，而不需要昂贵的安全特定微调或检测费用。代码可在https://github.com/Amadeuszhao/QMLLM上获取。



## **13. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

PSM：通过LLM引导的黑盒优化实现快速灵敏度最小化 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm

摘要: 系统提示对于指导大型语言模型（LLM）的行为至关重要，但它们通常包含专有逻辑或敏感信息，使其成为提取攻击的主要目标。对抗性查询可以成功地引出这些隐藏指令，从而构成重大的安全和隐私风险。现有的防御机制通常依赖于启发式方法，会产生大量的计算负担，或者不适用于通过黑匣子API访问的模型。本文介绍了一种通过屏蔽附加来强化系统提示的新颖框架，这是一种轻量级方法，可以在原始提示中添加保护性文本层。我们的核心贡献是将即时硬化形式化为一个受效用约束的优化问题。我们利用LLM作为优化器来搜索可能的SHIELD的空间，寻求最大限度地减少从一系列对抗攻击中获得的泄漏指标，同时将任务效用保持在指定阈值以上，该阈值通过基线输出的语义保真度来衡量。这种黑匣子、优化驱动的方法是轻量级且实用的，仅需要API访问目标和优化器LLM。我们通过经验证明，我们优化的SHIELD显着减少了针对一系列全面提取攻击的即时泄漏，在不损害模型预期功能的情况下优于既定的基线防御。我们的工作提供了一个在LLM安全不断升级的环境中开发强大的、实用程序感知的防御的范式。该代码在以下链接上公开：https://github.com/psm-defense/psm



## **14. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

当对齐失败时：对视觉-语言-动作模型的多模式对抗攻击 cs.CV

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.16203v2) [paper-pdf](https://arxiv.org/pdf/2511.16203v2)

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.

摘要: 视觉-语言-动作模型（VLA）最近在具体环境中取得了显着进展，使机器人能够通过统一的多模式理解来感知、推理和行动。尽管它们的能力令人印象深刻，但这些系统的对抗鲁棒性在很大程度上仍未得到探索，尤其是在现实的多模式和黑匣子条件下。现有的研究主要关注单模式扰动，而忽视了从根本上影响体现推理和决策的跨模式失调。本文介绍了VLA-Fool，这是对白盒和黑盒设置下具体VLA模型中多模式对抗鲁棒性的全面研究。VLA-Fool统一了三个级别的多模式对抗攻击：（1）通过基于梯度和基于预算的操纵进行文本扰动，（2）通过补丁和噪音失真进行视觉扰动，以及（3）故意破坏感知和指令之间的语义对应性的跨模式失准攻击。我们进一步将VLA感知的语义空间融入到语言提示中，开发了第一个自动制作和语义引导的提示框架。使用微调的OpenVLA模型对LIBERO基准进行的实验表明，即使是微小的多峰扰动也会导致显着的行为偏差，这表明了体现多峰对齐的脆弱性。



## **15. AutoBackdoor: Automating Backdoor Attacks via LLM Agents**

AutoBackdoor：通过LLM代理自动化后门攻击 cs.CR

23 pages

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16709v1) [paper-pdf](https://arxiv.org/pdf/2511.16709v1)

**Authors**: Yige Li, Zhe Li, Wei Zhao, Nay Myat Min, Hanxun Huang, Xingjun Ma, Jun Sun

**Abstract**: Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at https://github.com/bboylyg/BackdoorLLM.

摘要: 后门攻击对大型语言模型（LLM）的安全部署构成严重威胁，使对手能够植入由特定输入触发的隐藏行为。然而，现有的方法通常依赖于手工制作的触发器和静态数据管道，这些触发器和静态数据管道僵化、劳动密集型，并且不足以系统性地评估现代防御稳健性。随着人工智能代理的能力越来越强，人们对更严格、多样化和可扩展的\textit{red-teaming framework}的需求越来越大，它可以真实地模拟后门威胁并评估对抗条件下的模型弹性。在这项工作中，我们介绍了\textsk {AutoBackdoor}，这是一个用于自动化后门注入的通用框架，包括触发器生成、有毒数据构建以及通过自主代理驱动管道进行的模型微调。与以前的方法不同，AutoBackdoor使用强大的语言模型代理来生成语义连贯的、上下文感知的触发短语，以最少的人力即可在任意主题上进行可扩展的中毒。我们在三种现实的威胁场景下评估AutoBackdoor，包括\textit{Bias Recommendation}、\textit{Hallucination Injection}和\textit{Peer Review Manipulation}，以模拟广泛的攻击。在开源和商业模型（包括LLaMA-3、Mistral、Qwen和GPT-4 o）上的实验表明，我们的方法只需少量中毒样本即可获得超过90%的攻击成功率。更重要的是，我们发现现有的防御系统往往无法减轻这些攻击，这凸显了针对本工作中探讨的代理驱动威胁的更严格和自适应的评估技术的必要性。所有代码、数据集和实验配置都将合并到我们的主存储库https://github.com/bboylyg/BackdoorLLM中。



## **16. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs**

您的功能揭示了什么：针对拆分DNN的数据高效黑匣子功能倒置攻击 cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15316v1) [paper-pdf](https://arxiv.org/pdf/2511.15316v1)

**Authors**: Zhihan Ren, Lijun He, Jiaxi Liang, Xinzhu Fu, Haixia Bi, Fan Li

**Abstract**: Split DNNs enable edge devices by offloading intensive computation to a cloud server, but this paradigm exposes privacy vulnerabilities, as the intermediate features can be exploited to reconstruct the private inputs via Feature Inversion Attack (FIA). Existing FIA methods often produce limited reconstruction quality, making it difficult to assess the true extent of privacy leakage. To reveal the privacy risk of the leaked features, we introduce FIA-Flow, a black-box FIA framework that achieves high-fidelity image reconstruction from intermediate features. To exploit the semantic information within intermediate features, we design a Latent Feature Space Alignment Module (LFSAM) to bridge the semantic gap between the intermediate feature space and the latent space. Furthermore, to rectify distributional mismatch, we develop Deterministic Inversion Flow Matching (DIFM), which projects off-manifold features onto the target manifold with one-step inference. This decoupled design simplifies learning and enables effective training with few image-feature pairs. To quantify privacy leakage from a human perspective, we also propose two metrics based on a large vision-language model. Experiments show that FIA-Flow achieves more faithful and semantically aligned feature inversion across various models (AlexNet, ResNet, Swin Transformer, DINO, and YOLO11) and layers, revealing a more severe privacy threat in Split DNNs than previously recognized.

摘要: 拆分DNN通过将密集计算卸载到云服务器来支持边缘设备，但这种范式暴露了隐私漏洞，因为可以利用中间功能通过特征倒置攻击（FIA）重建私人输入。现有的FIA方法通常产生有限的重建质量，因此很难评估隐私泄露的真实程度。为了揭示泄露特征的隐私风险，我们引入了FIA-Flow，这是一种黑匣子FIA框架，可以从中间特征实现高保真图像重建。为了利用中间特征中的语义信息，我们设计了一个潜在特征空间对齐模块（LFSam）来弥合中间特征空间和潜在空间之间的语义差距。此外，为了纠正分布不匹配，我们开发了确定性反演流匹配（DIFM），该方法通过一步推理将流形外特征投影到目标流形上。这种解耦的设计简化了学习，并且能够使用很少的图像特征对进行有效的训练。为了从人类的角度量化隐私泄露，我们还提出了两个基于大型视觉语言模型的指标。实验表明，FIA-Flow在各种模型（AlexNet、ResNet、Swin Transformer、DINO和YOLO 11）和层中实现了更忠实、语义一致的特征倒置，揭示了Split DNN中比之前认识到的更严重的隐私威胁。



## **17. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

对抗性诗歌作为大型语言模型中通用的单轮越狱机制 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.

摘要: 我们提供的证据表明，对抗性诗歌可以作为大型语言模型（LLM）的通用单轮越狱技术。在25个前沿专有和开放重量模型中，精心策划的诗意提示产生了很高的攻击成功率（ASB），一些提供商超过了90%。MLCommons和EU CoP风险分类的映射提示表明，诗意攻击跨CBRN、操纵、网络犯罪和失去控制领域转移。通过标准化元提示将1，200个MLCommons有害提示转换为诗句，产生的ASB比散文基线高出18倍。使用3名开放权重LLM评委的整体评估输出，他们的二元安全性评估在分层的人类标记子集上进行了验证。诗意框架的平均越狱成功率为62%，元提示转换的平均越狱成功率约为43%（与非诗意基线相比），大大优于非诗意基线，并揭示了示范家庭和安全培训方法之间的系统性弱点。这些研究结果表明，仅靠风格差异就可以规避当代安全机制，这表明当前对齐方法和评估协议存在根本性局限性。



## **18. Securing AI Agents Against Prompt Injection Attacks**

保护人工智能代理免受即时注入攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15759v1) [paper-pdf](https://arxiv.org/pdf/2511.15759v1)

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.

摘要: 检索增强生成（RAG）系统已被广泛用于增强大型语言模型能力，但它们通过提示注入攻击引入了严重的安全漏洞。我们提出了一个全面的基准来评估支持RAG的人工智能代理中的即时注入风险，并提出了一个多层防御框架。我们的基准测试包括跨越五种攻击类别的847个对抗测试案例：直接注入、上下文操纵、指令覆盖、数据溢出和跨上下文污染。我们评估了三种防御机制：基于嵌入的异常检测的内容过滤、分层系统提示护栏和跨七种最先进语言模型的多阶段响应验证。我们的组合框架将成功攻击率从73.2%降低到8.7%，同时保持94.3%的基线任务性能。我们发布了我们的基准数据集和防御实施，以支持未来的人工智能代理安全研究。



## **19. Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks**

以IPI为中心的LLM代理防御框架的分类、评估和开发 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15203v1) [paper-pdf](https://arxiv.org/pdf/2511.15203v1)

**Authors**: Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, Shuai Wang

**Abstract**: Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.

摘要: 具有函数调用功能的基于大型语言模型（LLM）的代理被越来越多地部署，但仍然容易受到劫持其工具调用的间接提示注入（IPI）攻击。作为回应，出现了许多以IPI为中心的防御框架。然而，这些防御措施支离破碎，缺乏统一的分类和全面的评估。在本知识系统化（SoK）中，我们首次对以IPI为中心的防御框架进行了全面分析。我们对这些防御系统进行了全面的分类，并将它们按五个维度进行了分类。然后，我们彻底评估代表性防御框架的安全性和可用性。通过对评估中防御失败的分析，我们确定了规避防御的六个根本原因。基于这些发现，我们设计了三种新颖的自适应攻击，它们显着提高了针对特定框架的攻击成功率，并证明了这些防御中缺陷的严重性。我们的论文为未来开发更安全和可用的以IP为中心的代理防御框架提供了基础和重要见解。



## **20. As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files**

就像我们以前见过一样：法学硕士在识别可见文件方面表现出脆弱性 cs.AI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15192v2) [paper-pdf](https://arxiv.org/pdf/2511.15192v2)

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yan Pang

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.   We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.

摘要: 大型语言模型（LLM）卓越的语言能力源于对大量数据集的广泛训练，这些数据集通常包括受版权保护的材料，这引起了对未经授权使用的严重担忧。虽然成员关系推理攻击（MIA）提供了检测此类违规行为的潜在解决方案，但由于LLM固有的过度自信，对地面真实训练数据的有限访问以及对经验确定的阈值的依赖，现有方法面临着严重的限制和挑战。   我们提出了一个新的框架，利用不确定性信号来检测LLM训练集中是否使用了版权内容。我们的方法将LLM过度自信从一个限制变成一个资产，通过捕获不确定性模式，可靠地区分“看到”（训练数据）和“看不见”（非训练数据）的内容。COPYRIGHT进一步实现了双重策略：（1）将文件战略性地分割成较小的片段，以减少对大规模训练数据的依赖，以及（2）不确定性引导的无监督聚类，以消除对经验调整阈值的需求。实验结果表明，COPYRIGHT算法在LLaMA 7 b和LLaMA 2 7 b上检测可见文件的平均均衡准确率分别达到90.1%和91.6%。与SOTA基线相比，COPYRIGHT实现了90%以上的相对改进，达到93.8%的平衡精度。它还表现出跨架构的强大通用性，在GPT-J 6 B上保持高性能。这项工作首次将不确定性应用于LLM中的版权检测，为训练数据透明度提供了实用工具。



## **21. Can MLLMs Detect Phishing? A Comprehensive Security Benchmark Suite Focusing on Dynamic Threats and Multimodal Evaluation in Academic Environments**

MLLM可以检测网络钓鱼吗？专注于学术环境中的动态威胁和多模式评估的全面安全基准套件 cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.15165v2) [paper-pdf](https://arxiv.org/pdf/2511.15165v2)

**Authors**: Jingzhuo Zhou

**Abstract**: The rapid proliferation of Multimodal Large Language Models (MLLMs) has introduced unprecedented security challenges, particularly in phishing detection within academic environments. Academic institutions and researchers are high-value targets, facing dynamic, multilingual, and context-dependent threats that leverage research backgrounds, academic collaborations, and personal information to craft highly tailored attacks. Existing security benchmarks largely rely on datasets that do not incorporate specific academic background information, making them inadequate for capturing the evolving attack patterns and human-centric vulnerability factors specific to academia. To address this gap, we present AdapT-Bench, a unified methodological framework and benchmark suite for systematically evaluating MLLM defense capabilities against dynamic phishing attacks in academic settings.

摘要: 多模式大型语言模型（MLLM）的迅速普及带来了前所未有的安全挑战，特别是在学术环境中的网络钓鱼检测方面。学术机构和研究人员是高价值目标，面临着动态、多语言和取决于上下文的威胁，这些威胁利用研究背景、学术合作和个人信息来策划高度定制的攻击。现有的安全基准在很大程度上依赖于不包含特定学术背景信息的数据集，这使得它们不足以捕捉不断变化的攻击模式和学术界特有的以人为本的脆弱性因素。为了解决这一差距，我们提出了AdapT-Bench，这是一个统一的方法框架和基准套件，用于系统性评估MLLM防御能力在学术环境中抵御动态网络钓鱼攻击。



## **22. Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education**

统一防御大型语言模型，防止教育领域的越狱和微调攻击 cs.CL

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14423v1) [paper-pdf](https://arxiv.org/pdf/2511.14423v1)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Large Language Models (LLMs) are increasingly integrated into educational applications. However, they remain vulnerable to jailbreak and fine-tuning attacks, which can compromise safety alignment and lead to harmful outputs. Existing studies mainly focus on general safety evaluations, with limited attention to the unique safety requirements of educational scenarios. To address this gap, we construct EduHarm, a benchmark containing safe-unsafe instruction pairs across five representative educational scenarios, enabling systematic safety evaluation of educational LLMs. Furthermore, we propose a three-stage shield framework (TSSF) for educational LLMs that simultaneously mitigates both jailbreak and fine-tuning attacks. First, safety-aware attention realignment redirects attention toward critical unsafe tokens, thereby restoring the harmfulness feature that discriminates between unsafe and safe inputs. Second, layer-wise safety judgment identifies harmfulness features by aggregating safety cues across multiple layers to detect unsafe instructions. Finally, defense-driven dual routing separates safe and unsafe queries, ensuring normal processing for benign inputs and guarded responses for harmful ones. Extensive experiments across eight jailbreak attack strategies demonstrate that TSSF effectively strengthens safety while preventing over-refusal of benign queries. Evaluations on three fine-tuning attack datasets further show that it consistently achieves robust defense against harmful queries while maintaining preserving utility gains from benign fine-tuning.

摘要: 大型语言模型（LLM）越来越多地集成到教育应用程序中。然而，它们仍然容易受到越狱和微调攻击，这可能会损害安全一致并导致有害输出。现有的研究主要关注一般安全评估，对教育场景独特的安全要求的关注有限。为了解决这一差距，我们构建了EduHarm，这是一个包含五种代表性教育场景中安全与不安全指令对的基准，能够对教育学LLM进行系统性安全评估。此外，我们为教育LLM提出了一个三阶段盾牌框架（TSSF），该框架同时减轻越狱和微调攻击。首先，安全意识的注意力重新调整将注意力重新引导到关键的不安全代币上，从而恢复区分不安全和安全输入的有害性特征。其次，分层安全判断通过聚集多层安全线索来检测不安全指令来识别有害特征。最后，防御驱动的双重路由将安全和不安全的查询分开，确保良性输入的正常处理和有害输入的受保护响应。针对八种越狱攻击策略的广泛实验表明，TSSF有效地增强了安全性，同时防止了对良性查询的过度拒绝。对三个微调攻击数据集的评估进一步表明，它始终实现了针对有害查询的强大防御，同时保持良性微调的效用收益。



## **23. Beyond Fixed and Dynamic Prompts: Embedded Jailbreak Templates for Advancing LLM Security**

超越固定和动态预算：用于提高LLM安全性的嵌入式越狱模板 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14140v1) [paper-pdf](https://arxiv.org/pdf/2511.14140v1)

**Authors**: Hajun Kim, Hyunsik Na, Daeseon Choi

**Abstract**: As the use of large language models (LLMs) continues to expand, ensuring their safety and robustness has become a critical challenge. In particular, jailbreak attacks that bypass built-in safety mechanisms are increasingly recognized as a tangible threat across industries, driving the need for diverse templates to support red-teaming efforts and strengthen defensive techniques. However, current approaches predominantly rely on two limited strategies: (i) substituting harmful queries into fixed templates, and (ii) having the LLM generate entire templates, which often compromises intent clarity and reproductibility. To address this gap, this paper introduces the Embedded Jailbreak Template, which preserves the structure of existing templates while naturally embedding harmful queries within their context. We further propose a progressive prompt-engineering methodology to ensure template quality and consistency, alongside standardized protocols for generation and evaluation. Together, these contributions provide a benchmark that more accurately reflects real-world usage scenarios and harmful intent, facilitating its application in red-teaming and policy regression testing.

摘要: 随着大型语言模型（LLM）的使用不断扩大，确保其安全性和稳健性已成为一项关键挑战。特别是，绕过内置安全机制的越狱攻击越来越被视为跨行业的有形威胁，这促使人们需要多样化的模板来支持红色团队工作并加强防御技术。然而，当前的方法主要依赖于两种有限的策略：（i）将有害查询替换为固定模板，以及（ii）让LLM生成整个模板，这通常会损害意图的清晰性和可重复性。为了解决这一差距，本文引入了嵌入式越狱模板，它保留了现有模板的结构，同时自然地将有害查询嵌入到其上下文中。我们进一步提出了一种渐进的预算工程方法，以确保模板质量和一致性，以及用于生成和评估的标准化协议。这些贡献共同提供了一个更准确地反映现实世界使用场景和有害意图的基准，促进其在红色团队和政策回归测试中的应用。



## **24. GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards**

GRPO隐私面临风险：针对强化学习的会员推断攻击，具有可验证奖励 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14045v1) [paper-pdf](https://arxiv.org/pdf/2511.14045v1)

**Authors**: Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma

**Abstract**: Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.   To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.   To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces.

摘要: 对大型语言模型（LLM）的成员推断攻击（MIA）在模型训练的各个阶段都会带来重大的隐私风险。带可验证奖励的强化学习（WLVR）的最新进展给LLM培训带来了深刻的范式转变，特别是对于复杂的推理任务。然而，WLVR的政策性引入了一种独特的隐私泄露模式：由于训练依赖于自我生成的响应，而没有固定的地面真相输出，因此成员资格推断现在必须确定在微调期间是否使用给定的提示（独立于任何特定的响应）。这造成了一种威胁，其中泄漏不是由答案记忆引起的。   为了审计这种新颖的隐私风险，我们提出了行为分歧攻击（DIBA），这是第一个专门为WLVR设计的成员资格推断框架。DIBA将重点从记忆转移到行为改变，利用模型行为在两个轴上的可测量变化：员工端改进（例如，正确性收益）和逻辑端分歧（例如，政策漂移）。通过全面评估，我们证明DIBA的表现显着优于现有基线，实现了约0.8 AUT和更高数量级的TPR@0.1%FPR。我们验证了DIBA在多种环境中的优势--包括内分布、跨数据集、跨算法、黑匣子场景以及视觉语言模型的扩展。此外，在适度的防御措施下，我们的攻击仍然强劲。   据我们所知，这是第一个系统性分析WLVR中隐私漏洞的工作，揭示了即使在缺乏明确监督的情况下，也可以通过行为痕迹可靠地推断训练数据暴露。



## **25. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

差异化定向干预避免LLM安全一致的框架 cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2511.06852v4) [paper-pdf](https://arxiv.org/pdf/2511.06852v4)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.

摘要: 安全一致为大型语言模型（LLM）灌输了拒绝恶意请求的关键能力。之前的作品将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是一种过于简单化的做法，将两个功能上不同的神经过程混为一谈：伤害的检测和拒绝的执行。在这项工作中，我们将这个单一的表示解构为伤害检测方向和拒绝执行方向。利用这个细粒度模型，我们引入了差异双向干预（DBDI），这是一种新的白盒框架，可以精确地中和关键层的安全对齐。DBDI对拒绝执行方向应用自适应投影无效，同时通过直接转向抑制伤害检测方向。大量实验表明，DBDI优于著名的越狱方法，对Llama-2等模型的攻击成功率高达97.88%。通过提供更细粒度和机械化的框架，我们的工作为深入了解LLM安全对齐提供了新的方向。



## **26. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

使用大型语言模型在可穿戴物联网系统中进行自适应和稳健的数据中毒检测和清理 cs.LG

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2511.02894v3) [paper-pdf](https://arxiv.org/pdf/2511.02894v3)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.

摘要: 可穿戴传感设备在物联网（IoT）生态系统中的广泛集成，特别是在医疗保健、智能家居和工业应用中，需要强大的人类活动识别（HAR）技术来改善功能和用户体验。尽管机器学习模型具有高级HAR，但它们越来越容易受到数据中毒攻击，从而损害这些系统的数据完整性和可靠性。防御此类攻击的传统方法通常需要使用大型标记数据集进行广泛的任务特定训练，这限制了动态物联网环境中的适应性。这项工作提出了一种新颖的框架，该框架使用大型语言模型（LLM）在HAR系统中执行中毒检测和清理，利用零触发、单触发和少触发学习范式。我们的方法结合了\textit{role play}提示，LLM承担专家的角色来情境化和评估传感器异常，以及\textit{think分步}推理，指导LLM推断原始传感器数据中的中毒指标和合理的清洁替代品。这些策略最大限度地减少了对大量数据集管理的依赖，并实时实现强大、适应性强的防御机制。我们对框架进行了广泛的评估，量化检测准确性、消毒质量、延迟和通信成本，从而证明了LLM在提高可穿戴物联网系统安全性和可靠性方面的实用性和有效性。



## **27. DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion**

DRIP：通过逐令牌表示编辑和剩余指令融合来防御提示注入 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.00447v2) [paper-pdf](https://arxiv.org/pdf/2511.00447v2)

**Authors**: Ruofan Liu, Yun Lin, Zhiyong Huang, Jin Song Dong

**Abstract**: Large language models (LLMs) are increasingly integrated into IT infrastructures, where they process user data according to predefined instructions. However, conventional LLMs remain vulnerable to prompt injection, where malicious users inject directive tokens into the data to subvert model behavior. Existing defenses train LLMs to semantically separate data and instruction tokens, but still struggle to (1) balance utility and security and (2) prevent instruction-like semantics in the data from overriding the intended instructions.   We propose DRIP, which (1) precisely removes instruction semantics from tokens in the data section while preserving their data semantics, and (2) robustly preserves the effect of the intended instruction even under strong adversarial content. To "de-instructionalize" data tokens, DRIP introduces a data curation and training paradigm with a lightweight representation-editing module that edits embeddings of instruction-like tokens in the data section, enhancing security without harming utility. To ensure non-overwritability of instructions, DRIP adds a minimal residual module that reduces the ability of adversarial data to overwrite the original instruction. We evaluate DRIP on LLaMA 8B and Mistral 7B against StruQ, SecAlign, ISE, and PFT on three prompt-injection benchmarks (SEP, AlpacaFarm, and InjecAgent). DRIP improves role-separation score by 12-49\%, reduces attack success rate by over 66\% under adaptive attacks, and matches the utility of the undefended model, establishing a new state of the art for prompt-injection robustness.

摘要: 大型语言模型（LLM）越来越多地集成到IT基础设施中，它们根据预定义的指令处理用户数据。然而，传统的LLM仍然容易受到提示注入的影响，即恶意用户将指令令牌注入到数据中以颠覆模型行为。现有的防御措施训练LLM在语义上分离数据和指令令牌，但仍然难以（1）平衡实用性和安全性，以及（2）防止数据中类似描述的语义覆盖预期指令。   我们提出了DRIP，它（1）从数据部分中的令牌中精确地删除指令语义，同时保留其数据语义，（2）即使在强对抗性内容下也能稳健地保留预期指令的效果。为了“去伪化”数据令牌，DRIP引入了一种数据策展和训练范式，该范式具有轻量级的表示编辑模块，该模块可以编辑数据部分中类似描述的令牌的嵌入，从而在不损害实用性的情况下增强了安全性。为了确保指令的不可重写性，DRIP添加了一个最小剩余模块，该模块降低了对抗数据重写原始指令的能力。我们在三个预算注入基准（SDP、AlpacaFarm和InjecAgent）上针对StruQ、SecAlign、ISE和PFT评估了LLaMA 8B和Mistral 7 B上的DRIP。DRIP将角色分离分数提高了12- 49%，在适应性攻击下将攻击成功率降低了66%以上，并匹配了无防御模型的实用性，为预算注入鲁棒性建立了新的最新水平。



## **28. SoK: Honeypots & LLMs, More Than the Sum of Their Parts?**

SoK：蜜罐和LLM，超过其部分的总和？ cs.CR

Systemization of Knowledge

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.25939v3) [paper-pdf](https://arxiv.org/pdf/2510.25939v3)

**Authors**: Robert A. Bridges, Thomas R. Mitchell, Mauricio Muñoz, Ted Henriksson

**Abstract**: The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.

摘要: 大型语言模型（LLM）的出现有望解决蜜罐设计中长期存在的悖论，以低操作风险实现高保真欺骗。通过自2022年底以来的一系列研究，从构思到原型实现的稳步进展已经显现。自2022年底以来，一系列研究表明，从构思到原型实施正在稳步进展。虽然有希望，但评估仅显示现实世界部署中的渐进进展，并且该领域仍然缺乏对新兴架构模式、核心挑战和评估范式的一致理解。为了填补这一空白，这篇知识系统化（SoK）论文首次对这一新领域进行了全面的概述和分析。我们通过关注三个关键的交叉研究领域来调查和系统化该领域：首先，我们提供了蜜罐检测向量的分类，构建了基于LLM的现实主义必须解决的核心问题;其次，我们综合了关于LLM供电的蜜罐的新兴文献，确定了规范架构和关键评估趋势;第三，我们绘制了蜜罐日志分析的进化路径，从简单的数据简化到自动智能生成。我们将这些发现综合成一个前瞻性的研究路线图，认为这项技术的真正潜力在于创建自主的、自我改进的欺骗系统，以应对智能的、自动化的攻击者的新兴威胁。



## **29. Practical and Stealthy Touch-Guided Jailbreak Attacks on Deployed Mobile Vision-Language Agents**

对已部署的移动视觉语言代理进行实用且隐秘的触摸引导越狱攻击 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.07809v2) [paper-pdf](https://arxiv.org/pdf/2510.07809v2)

**Authors**: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

**Abstract**: Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities in their perception and interaction remain critically understudied. Existing research often relies on conspicuous overlays, elevated permissions, or unrealistic threat assumptions, limiting stealth and real-world feasibility. In this paper, we introduce a practical and stealthy jailbreak attack framework, which comprises three key components: (i) non-privileged perception compromise, which injects visual payloads into the application interface without requiring elevated system permissions; (ii) agent-attributable activation, which leverages input attribution signals to distinguish agent from human interactions and limits prompt exposure to transient intervals to preserve stealth from end users; and (iii) efficient one-shot jailbreak, a heuristic iterative deepening search algorithm (HG-IDA*) that performs keyword-level detoxification to bypass built-in safety alignment of LVLMs. Moreover, we developed three representative Android applications and curated a prompt-injection dataset for mobile agents. We evaluated our attack across multiple LVLM backends, including closed-source services and representative open-source models, and observed high planning and execution hijack rates (e.g., GPT-4o: 82.5% planning / 75.0% execution), exposing a fundamental security vulnerability in current mobile agents and underscoring critical implications for autonomous smartphone operation.

摘要: 大型视觉语言模型（LVLM）使自主移动代理能够操作智能手机用户界面，但它们的感知和交互中的漏洞仍然严重缺乏研究。现有的研究通常依赖于明显的叠加、较高的权限或不切实际的威胁假设，从而限制了隐形和现实世界的可行性。本文中，我们介绍了一个实用且隐蔽的越狱攻击框架，该框架由三个关键组件组成：（i）非特权感知妥协，它将视觉有效负载注入到应用程序界面中，而不需要提高系统权限;（ii）代理归因于激活，它利用输入归因信号将代理与人类互动区分开来，并限制即时暴露在瞬时间隔中以保持隐形最终用户;和（iii）高效的一次性越狱，这是一种启发式迭代深化搜索算法（HG-IDA*），可执行关键字级解毒以绕过LVLM的内置安全对齐。此外，我们开发了三个代表性的Android应用程序，并为移动代理策划了预算注入数据集。我们评估了多个LVLM后台（包括闭源服务和代表性开源模型）的攻击，并观察到很高的规划和执行劫持率（例如，GPT-4o：82.5%规划/ 75.0%执行），暴露了当前移动代理中的根本安全漏洞，并强调了对自主智能手机操作的关键影响。



## **30. Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks**

不一致时间：大型语言模型对对抗性攻击的鲁棒性的生存分析 cs.CL

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2510.02712v2) [paper-pdf](https://arxiv.org/pdf/2510.02712v2)

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model-drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.

摘要: 大型语言模型（LLM）彻底改变了对话人工智能，但对其在扩展多轮对话中的稳健性仍然知之甚少。现有的评估框架专注于静态基准和单轮评估，未能捕捉反映现实世界互动特征的对话退化的时间动态。在这项工作中，我们提出了对话稳健性的大规模生存分析，将故障建模为MT-Consistency基准上的9个最先进的LLM在36，951个回合内的事件时间过程。我们的框架将Cox比例风险、加速故障时间（AFT）和随机生存森林模型与简单的语义漂移特征相结合。我们发现，突然的从预定到提示的语义漂移会急剧增加不一致的风险，而累积漂移是反直觉的\{保护性}，这表明在经历多次转变的对话中进行适应。具有模型-漂移相互作用的AFT模型实现了区分和校准的最佳组合，比例风险检查揭示了关键漂移协变量的系统性违规，解释了这种环境下Cox式建模的局限性。最后，我们表明，轻量级的AFT模型可以转变为回合级风险监控器，它在第一个不一致的答案之前的几个回合标记大多数失败的对话，同时保持虚假警报适度。这些结果将生存分析确立为评估多轮稳健性和为对话式人工智能系统设计实用保障措施的强大范式。



## **31. Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data**

桌面上的边界：针对结构化数据的高效基于决策的黑匣子攻击 cs.LG

Paper revision

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2509.22850v3) [paper-pdf](https://arxiv.org/pdf/2509.22850v3)

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems.

摘要: 与视觉和语言领域相比，结构化数据中的对抗稳健性仍然是一个未充分探索的前沿。在这项工作中，我们引入了一种针对表格数据量身定制的新型黑匣子、基于决策的对抗攻击。我们的方法将无梯度方向估计与迭代边界搜索相结合，能够在最小的Oracle访问下高效导航离散和连续特征空间。大量实验表明，我们的方法成功地妥协了不同模型的几乎整个测试集，范围包括经典机器学习分类器和基于大型语言模型（LLM）的管道。值得注意的是，该攻击的成功率始终高于90%，而每个实例只需要少量查询。这些结果凸显了表格模型对对抗性扰动的严重脆弱性，凸显了现实世界决策系统中迫切需要更强的防御。



## **32. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

使用结构化攻击树的LLM驱动渗透测试中的引导推理 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2509.07939v2) [paper-pdf](https://arxiv.org/pdf/2509.07939v2)

**Authors**: Katsuaki Nakano, Reza Fayyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments

摘要: 大型语言模型（LLM）的最新进展激发了人们对自动化网络安全渗透测试工作流程的兴趣，为企业系统提供更快、更一致的漏洞评估。现有的用于渗透测试的LLM代理主要依赖于自我引导推理，这可能会产生不准确或幻觉的程序步骤。因此，LLM代理可能会采取非生产性的行动，例如利用未使用的软件库或生成重复先前策略的周期性响应。在这项工作中，我们提出了一个用于渗透测试LLM代理的引导推理管道，该管道结合了从MITRE ATT & CK矩阵（一个经过验证的渗透测试kll链）构建的确定性任务树，以将LLM的重组过程限制为明确定义的策略、技术和程序。这将推理锚定在经过验证的渗透测试方法中，并通过引导代理走向更有成效的攻击程序来过滤无效操作。为了评估我们的方法，我们使用三个LLM（Llama-3-8B、Gemini-1.5和GPT-4）构建了一个自动渗透测试LLM代理，并将其应用于导航10个HackTheBox网络安全演习，其中包含103个代表现实世界网络攻击场景的离散子任务。我们提出的推理管道使用Llama-3-8B、Gemini-1.5和GPT-4分别引导LLM代理完成71.8%、72.8%和78.6%的子任务。相比之下，使用自我引导推理的最先进的LLM渗透测试工具仅完成了13.5%、16.5%和75.7%的子任务，并且需要86.2%、118.7%和205.9%的模型查询。这表明将确定性任务树纳入LLM推理管道可以提高自动化网络安全评估的准确性和效率



## **33. PromptCOS: Towards Content-only System Prompt Copyright Auditing for LLMs**

Observtcos：面向LLM的纯内容系统提示版权审计 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2509.03117v2) [paper-pdf](https://arxiv.org/pdf/2509.03117v2)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Yuyi Wang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: System prompts are critical for shaping the behavior and output quality of large language model (LLM)-based applications, driving substantial investment in optimizing high-quality prompts beyond traditional handcrafted designs. However, as system prompts become valuable intellectual property, they are increasingly vulnerable to prompt theft and unauthorized use, highlighting the urgent need for effective copyright auditing, especially watermarking. Existing methods rely on verifying subtle logit distribution shifts triggered by a query. We observe that this logit-dependent verification framework is impractical in real-world content-only settings, primarily because (1) random sampling makes content-level generation unstable for verification, and (2) stronger instructions needed for content-level signals compromise prompt fidelity.   To overcome these challenges, we propose PromptCOS, the first content-only system prompt copyright auditing method based on content-level output similarity. PromptCOS achieves watermark stability by designing a cyclic output signal as the conditional instruction's target. It preserves prompt fidelity by injecting a small set of auxiliary tokens to encode the watermark, leaving the main prompt untouched. Furthermore, to ensure robustness against malicious removal, we optimize cover tokens, i.e., critical tokens in the original prompt, to ensure that removing auxiliary tokens causes severe performance degradation. Experimental results show that PromptCOS achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% higher than the best baseline), high fidelity (accuracy degradation no greater than 0.6%), robustness (resilience against four potential attack categories), and high computational efficiency (up to 98.1% cost saving). Our code is available at GitHub (https://github.com/LianPing-cyber/PromptCOS).

摘要: 系统提示对于塑造基于大型语言模型（LLM）的应用程序的行为和输出质量至关重要，推动了对传统手工设计之外的高质量提示进行大量投资。然而，随着系统提示成为宝贵的知识产权，它们越来越容易被及时盗窃和未经授权使用，这凸显了对有效版权审计的迫切需要，尤其是水印。现有的方法依赖于验证查询触发的微妙logit分布变化。我们观察到，这种依赖于逻辑的验证框架在现实世界的仅内容设置中是不切实际的，主要是因为（1）随机采样使得内容级生成对于验证来说不稳定，以及（2）内容级信号所需的更强指令会损害即时保真度。   为了克服这些挑战，我们提出了Inbox cos，这是第一个基于内容级输出相似性的纯内容系统提示版权审计方法。Intrutcos通过设计循环输出信号作为条件指令的目标来实现水印稳定性。它通过注入一小组辅助令牌来编码水印来保留提示的保真度，而不影响主提示。此外，为了确保针对恶意删除的鲁棒性，我们优化了掩护令牌，即原始提示中的关键令牌，以确保删除辅助令牌会导致严重的性能下降。实验结果表明，Intrutcos具有高有效性（平均水印相似度为99.3%）、强区别性（比最佳基线高60.8%）、高保真度（准确率下降不超过0.6%）、鲁棒性（对四种潜在攻击类别的弹性）和高计算效率（节省成本高达98.1%）。我们的代码可在GitHub上获取（https：//github.com/LianPing-cyber/Journaltcos）。



## **34. SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures**

SoK：通过检查生成方法、内容特征和对策来暴露LLM生成的网络钓鱼中的生成和检测差距 cs.CR

18 pages, 5 tables, 4 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2508.21457v2) [paper-pdf](https://arxiv.org/pdf/2508.21457v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.

摘要: 网络钓鱼活动涉及伪装成值得信赖的供应商的对手，试图触发用户行为，使他们能够泄露私人数据。虽然URL是网络钓鱼活动的重要组成部分，但文本和图像等通信元素是触发所需用户行为的核心。此外，由于网络钓鱼检测的进步，攻击者通过将活动规模扩大到更大的数量以及使内容多样化和个性化来做出反应。除了基于模板的生成等已建立的机制外，大型语言模型（LLM）还可用于网络钓鱼内容生成，使攻击在几分钟内扩展，通过个性化内容、隐形显式网络钓鱼关键词和动态适应各种攻击场景来挑战现有的网络钓鱼检测范式。应对这些动态变化的攻击活动需要全面了解复杂的LLM相关威胁格局。现有的研究是零散的，并且集中在特定领域。在这项工作中，我们对LLM生成的网络钓鱼内容进行了首次全面检查。首先，为了追踪LLM用于网络钓鱼内容生成的利用途径，我们采用模块化分类法，记录对手突破LLM安全护栏的九个阶段。然后，我们描述了LLM生成的网络钓鱼如何表现为威胁，揭示了它可以逃避检测器，同时强调人类认知操纵。第三，通过对与生成方法保持一致的防御技术进行分类，我们暴露了一个关键的不对称性，即攻击机制动态地适应攻击场景，而防御策略保持静态和反应性。最后，根据对现有文献的彻底分析，我们强调了见解和差距，并提出了大规模理解和打击LLM驱动的网络钓鱼的路线图。



## **35. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

学习在大型视觉语言模型中检测未知越狱攻击 cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2508.09201v3) [paper-pdf](https://arxiv.org/pdf/2508.09201v3)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定于攻击的参数，这阻碍了对不可见攻击的概括，要么依赖于数学上合理的原则，这限制了准确性和效率。为了克服这些局限性，我们提出了学习检测（Lo），这是一个通用框架，通过将重点从特定攻击的学习转移到特定任务的学习来准确检测未知越狱攻击。该框架包括用于面向安全的表示学习的多模式安全概念激活载体模块和用于无监督攻击分类的安全模式自动编码器模块。大量实验表明，我们的方法在提高效率的同时，对各种未知攻击实现了一致更高的AUROC检测。该代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB上获取。



## **36. Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers**

隐藏在噪音中：通过潜在声学模式触发器揭开音频LLM对齐中的后门 cs.SD

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.02175v3) [paper-pdf](https://arxiv.org/pdf/2508.02175v3)

**Authors**: Liang Lin, Miao Yu, Kaiwen Luo, Yibo Zhang, Lilan Peng, Dexian Wang, Xuehai Tang, Yuanhe Zhang, Xikang Yang, Zhenhong Zhou, Kun Wang, Yang Liu

**Abstract**: As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.

摘要: 随着音频大语言模型（ALLM）成为语音处理的强大工具，其安全性影响迫切需要关注。虽然大量研究探索了文本和视觉安全，但音频的独特特征带来了重大挑战。本文首先研究：ALLM是否容易受到利用声学触发器的后门攻击？为了应对这个问题，我们引入了Hidden in the Noise（HIN），这是一种新颖的后门攻击框架，旨在利用微妙的特定音频特征。HIN对原始音频波进行声学修改，例如改变时间动态和战略性地注入频谱定制的噪音。这些变化引入了ALLM的声学特征编码器捕获的一致模式，并在音频流中嵌入稳健的触发器。为了评估ALLM针对基于音频特征的触发器的稳健性，我们开发了AudioSafe基准，评估九种不同的风险类型。对AudioSafe和三个已建立的安全数据集的广泛实验揭示了现有ALLM中的关键漏洞：（I）环境噪音和语音速率变化等音频特征实现了超过90%的平均攻击成功率。(II)ALLMS在声学特征中表现出显着的灵敏度差异，特别是对作为触发器的体积的响应最小，并且（III）中毒样本包含物仅引起边际损失曲线波动，凸显了攻击的隐秘性。



## **37. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

AgentArmor：对Agent DeliverTrace执行程序分析以防止即时注入 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.01249v3) [paper-pdf](https://arxiv.org/pdf/2508.01249v3)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

摘要: 大型语言模型（LLM）代理通过将自然语言推理与外部工具的执行相结合，提供了一个强大的新范式来解决各种问题。然而，它们的动态和不透明行为会带来严重的安全风险，特别是在存在即时注入攻击的情况下。在这项工作中，我们提出了一种新颖的见解，将代理运行时跟踪视为具有可分析语义的结构化程序。因此，我们提出了AgentArmor，这是一个程序分析框架，它将代理跟踪转换为基于图形中间表示的结构化程序依赖性表示（例如，CGM、DFG和PDG）并通过类型系统强制执行安全策略。AgentArmor由三个关键组件组成：（1）一个图形构造器，将代理的运行时跟踪重建为基于图形的中间表示，其中描述了控制和数据流;（2）一个属性注册表，附加交互工具和数据的安全相关元数据，以及（3）一个类型系统，执行静态推断和检查中间表示。通过将代理行为表示为结构化程序，AgentArmor可以对敏感数据流、信任边界和策略违规进行程序分析。我们在AgentDojo基准测试上对AgentArmor进行了评估，结果表明AgentArmor可以将ASB降低至3%，而实用程序仅下降1%。



## **38. Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation**

通过知识不对称利用从检索增强生成系统中进行细粒度隐私提取 cs.CR

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2507.23229v2) [paper-pdf](https://arxiv.org/pdf/2507.23229v2)

**Authors**: Yufei Chen, Yao Wang, Haibin Zhang, Tao Gu

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge bases, but this advancement introduces significant privacy risks. Existing privacy attacks on RAG systems can trigger data leakage but often fail to accurately isolate knowledge-base-derived sentences within mixed responses. They also lack robustness when applied across multiple domains. This paper addresses these challenges by presenting a novel black-box attack framework that exploits knowledge asymmetry between RAG and standard LLMs to achieve fine-grained privacy extraction across heterogeneous knowledge landscapes. We propose a chain-of-thought reasoning strategy that creates adaptive prompts to steer RAG systems away from sensitive content. Specifically, we first decompose adversarial queries to maximize information disparity and then apply a semantic relationship scoring to resolve lexical and syntactic ambiguities. We finally train a neural network on these feature scores to precisely identify sentences containing private information. Unlike prior work, our framework generalizes to unseen domains through iterative refinement without pre-defined knowledge. Experimental results show that we achieve over 91% privacy extraction rate in single-domain and 83% in multi-domain scenarios, reducing sensitive sentence exposure by over 65% in case studies. This work bridges the gap between attack and defense in RAG systems, enabling precise extraction of private information while providing a foundation for adaptive mitigation.

摘要: 检索增强生成（RAG）系统通过集成外部知识库来增强大型语言模型（LLM），但这一进步带来了巨大的隐私风险。对RAG系统的现有隐私攻击可能会引发数据泄露，但通常无法准确地隔离混合响应中的知识库派生句子。当应用于多个领域时，它们也缺乏稳健性。本文通过提出一种新型的黑匣子攻击框架来解决这些挑战，该框架利用RAG和标准LLM之间的知识不对称性来实现跨异类知识环境的细粒度隐私提取。我们提出了一种思想链推理策略，可以创建自适应提示来引导RAG系统远离敏感内容。具体来说，我们首先分解对抗性查询以最大化信息差异，然后应用语义关系评分来解决词汇和语法歧义。我们最终根据这些特征分数训练神经网络，以精确识别包含私人信息的句子。与之前的工作不同，我们的框架通过迭代细化而无需预先定义的知识，将其推广到不可见的领域。实验结果表明，我们在单域场景中实现了超过91%的隐私提取率，在多域场景中实现了83%的隐私提取率，在案例研究中将敏感句子暴露减少了超过65%。这项工作弥合了RAG系统中攻击和防御之间的差距，能够精确提取私人信息，同时为自适应缓解提供基础。



## **39. Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models**

响应攻击：利用上下文启动来越狱大型语言模型 cs.CL

20 pages, 10 figures. Code and data available at https://github.com/Dtc7w3PQ/Response-Attack

**SubmitDate**: 2025-11-21    [abs](http://arxiv.org/abs/2507.05248v2) [paper-pdf](https://arxiv.org/pdf/2507.05248v2)

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. While existing jailbreak attacks largely rely on single-turn or multi-turn prompt manipulations, or inject static in-context examples, these methods suffer from limited effectiveness, inefficiency, or semantic drift. We introduce Response Attack (RA), a novel framework that strategically leverages intermediate, mildly harmful responses as contextual primers within a dialogue. By reformulating harmful queries and injecting these intermediate responses before issuing a targeted trigger prompt, RA exploits a previously overlooked vulnerability in LLMs. Extensive experiments across eight state-of-the-art LLMs show that RA consistently achieves significantly higher attack success rates than nine leading jailbreak baselines. Our results demonstrate that the success of RA is directly attributable to the strategic use of intermediate responses, which induce models to generate more explicit and relevant harmful content while maintaining stealth, efficiency, and fidelity to the original query. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.

摘要: 上下文启动（早期的刺激会秘密地偏向后来的判断）为大型语言模型（LLM）提供了一个尚未探索的攻击表面。我们发现了一个上下文启动漏洞，其中对话中的先前响应可以将其后续行为引导到违反政策的内容上。虽然现有的越狱攻击主要依赖于单轮或多轮提示操纵，或注入静态上下文示例，但这些方法的有效性有限、效率低下或语义漂移。我们引入了响应攻击（RA），这是一个新颖的框架，它战略性地利用中间的、轻度有害的响应作为对话中的上下文触发器。通过重新制定有害查询并在发出有针对性的触发提示之前注入这些中间响应，RA利用了LLM中以前被忽视的漏洞。对八个最先进的LLM进行的广泛实验表明，RA始终比九个领先的越狱基线实现显着更高的攻击成功率。我们的结果表明，RA的成功直接归因于中间响应的战略使用，中间响应会促使模型生成更明确和相关的有害内容，同时保持原始查询的隐秘性、效率和保真度。代码和数据可在https://github.com/Dtc7w3PQ/Response-Attack上获取。



## **40. Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines**

有条件扩散中的后门：对负责任的合成数据管道的威胁 cs.CV

Accepted at RDS @ AAAI 2026

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2507.04726v2) [paper-pdf](https://arxiv.org/pdf/2507.04726v2)

**Authors**: Raz Lapid, Almog Dubin

**Abstract**: Text-to-image diffusion models achieve high-fidelity image generation from natural language prompts. ControlNets extend these models by enabling conditioning on structural inputs (e.g., edge maps, depth, pose), providing fine-grained control over outputs. Yet their reliance on large, publicly scraped datasets and community fine-tuning makes them vulnerable to data poisoning. We introduce a model-poisoning attack that embeds a covert backdoor into a ControlNet, causing it to produce attacker-specified content when exposed to visual triggers, without textual prompts. Experiments show that poisoning only 1% of the fine-tuning corpus yields a 90-98% attack success rate, while 5% further strengthens the backdoor, all while preserving normal generation quality. To mitigate this risk, we propose clean fine-tuning (CFT): freezing the diffusion backbone and fine-tuning only the ControlNet on a sanitized dataset with a reduced learning rate. CFT lowers attack success rates on held-out data. These results expose a critical security weakness in open-source, ControlNet-guided diffusion pipelines and demonstrate that CFT offers a practical defense for responsible synthetic-data pipelines.

摘要: 文本到图像扩散模型实现了从自然语言提示生成高保真图像。Control Nets通过对结构性输入进行条件化来扩展这些模型（例如，边缘地图、深度、姿势），提供对输出的细粒度控制。然而，他们对大型公开抓取的数据集和社区微调的依赖使他们很容易受到数据中毒的影响。我们引入了一种模型中毒攻击，将秘密后门嵌入到控制网络中，导致其在暴露于视觉触发器时生成攻击者指定的内容，而没有文本提示。实验表明，仅毒害1%的微调数据库就会产生90-98%的攻击成功率，而5%则进一步加强了后门，同时保持正常的生成质量。为了降低这种风险，我们提出了干净微调（CFT）：冻结扩散主干并在学习率降低的净化数据集上仅微调控制Net。CFT降低了对持有数据的攻击成功率。这些结果暴露了开源、Control Net引导的传播管道中的一个关键安全弱点，并证明CFT为负责任的合成数据管道提供了实用的防御。



## **41. Large Language Model Unlearning for Source Code**

大型语言模型放弃源代码的学习 cs.SE

Accepted to AAAI'26

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2506.17125v2) [paper-pdf](https://arxiv.org/pdf/2506.17125v2)

**Authors**: Xue Jiang, Yihong Dong, Huangzhao Zhang, Tangxinyu Wang, Zheng Fang, Yingwei Ma, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Yongbin Li, Ge Li

**Abstract**: While Large Language Models (LLMs) excel at code generation, their inherent tendency toward verbatim memorization of training data introduces critical risks like copyright infringement, insecure emission, and deprecated API utilization, etc. A straightforward yet promising defense is unlearning, ie., erasing or down-weighting the offending snippets through post-training. However, we find its application to source code often tends to spill over, damaging the basic knowledge of programming languages learned by the LLM and degrading the overall capability. To ease this challenge, we propose PROD for precise source code unlearning. PROD surgically zeroes out the prediction probability of the prohibited tokens, and renormalizes the remaining distribution so that the generated code stays correct. By excising only the targeted snippets, PROD achieves precise forgetting without much degradation of the LLM's overall capability. To facilitate in-depth evaluation against PROD, we establish an unlearning benchmark consisting of three downstream tasks (ie., unlearning of copyrighted code, insecure code, and deprecated APIs), and introduce Pareto Dominance Ratio (PDR) metric, which indicates both the forget quality and the LLM utility. Our comprehensive evaluation demonstrates that PROD achieves superior overall performance between forget quality and model utility compared to existing unlearning approaches across three downstream tasks, while consistently exhibiting improvements when applied to LLMs of varying series. PROD also exhibits superior robustness against adversarial attacks without generating or exposing the data to be forgotten. These results underscore that our approach not only successfully extends the application boundary of unlearning techniques to source code, but also holds significant implications for advancing reliable code generation.

摘要: 虽然大型语言模型（LLM）擅长代码生成，但其固有的逐字记忆训练数据的倾向会引入版权侵权、不安全的发射和过时的API利用等关键风险。一个简单但有希望的防御是取消学习，即，通过训练后删除或降低违规片段的权重。然而，我们发现它对源代码的应用往往会溢出，损害LLM学到的编程语言的基本知识，并降低整体能力。为了缓解这一挑战，我们提出了PROD来精确的源代码反学习。PROD通过外科手术将被禁止的令牌的预测概率归零，并重新规范剩余的分布，以便生成的代码保持正确。通过仅删除目标片段，PROD实现了精确遗忘，而不会大幅降低LLM的整体能力。为了促进针对PROD的深入评估，我们建立了一个由三个下游任务（即，放弃受版权保护的代码、不安全的代码和废弃的API），并引入帕累托主导比（PDR）指标，该指标既指示忘记质量又指示LLM实用性。我们的全面评估表明，与三个下游任务中的现有取消学习方法相比，PROD在忘记质量和模型效用之间实现了更好的整体性能，同时在应用于不同系列的LLM时一致表现出改进。PROD还表现出针对对抗攻击的卓越鲁棒性，而不会生成或暴露被遗忘的数据。这些结果强调，我们的方法不仅成功地将放弃学习技术的应用边界扩展到源代码，而且对推进可靠的代码生成具有重要影响。



## **42. Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?**

保护检索数据的隐私免受成员推断攻击：此查询是否离家太近？ cs.CL

Accepted for EMNLP findings 2025

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2505.22061v3) [paper-pdf](https://arxiv.org/pdf/2505.22061v3)

**Authors**: Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook Lee, Jinseong Park

**Abstract**: Retrieval-augmented generation (RAG) mitigates the hallucination problem in large language models (LLMs) and has proven effective for personalized usages. However, delivering private retrieved documents directly to LLMs introduces vulnerability to membership inference attacks (MIAs), which try to determine whether the target data point exists in the private external database or not. Based on the insight that MIA queries typically exhibit high similarity to only one target document, we introduce a novel similarity-based MIA detection framework designed for the RAG system. With the proposed method, we show that a simple detect-and-hide strategy can successfully obfuscate attackers, maintain data utility, and remain system-agnostic against MIA. We experimentally prove its detection and defense against various state-of-the-art MIA methods and its adaptability to existing RAG systems.

摘要: 检索增强生成（RAG）缓解了大型语言模型（LLM）中的幻觉问题，并已被证明对个性化使用有效。然而，将私有检索到的文档直接传递到LLM会引入成员资格推断攻击（MIA）的漏洞，该攻击试图确定目标数据点是否存在于私有外部数据库中。基于MIA查询通常仅与一个目标文档表现出高相似性的认识，我们引入了一种为RAG系统设计的新型基于相似性的MIA检测框架。通过提出的方法，我们表明简单的检测和隐藏策略可以成功地混淆攻击者、保持数据效用并保持系统对MIA的不可知性。我们通过实验证明了它对各种最先进的MIA方法的检测和防御，以及它对现有RAG系统的适应性。



## **43. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

重温模型倒置评估：从误导性标准到可靠的隐私评估 cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM

摘要: 模型倒置（MI）攻击旨在通过利用对机器学习模型T的访问来从私人训练数据中重建信息。为了评估此类攻击，标准评估框架依赖于评估模型E，该模型在与T相同的任务设计下训练。该框架已成为评估MI研究进展的事实标准，几乎所有最近的MI研究都毫无疑问地使用了该框架。在本文中，我们对该评估框架进行了首次深入研究。特别是，我们确定了这个标准框架的一个关键问题：I型对抗性示例。这些重建并没有捕捉到私人训练数据的视觉特征，但仍然被T认为是成功的，并最终转移到E。这种假阳性损害了标准管理信息评价框架的可靠性。为了解决这个问题，我们引入了一个新的MI评估框架，用先进的多模态大型语言模型（MLLM）取代了评估模型E。通过利用他们的通用视觉理解，我们基于MLLM的框架不依赖于T中的共享任务设计的训练，从而降低了I型可转移性，并提供了更忠实的重建成功评估。使用我们基于MLLM的评估框架，我们重新评估了27种不同的MI攻击设置，并根据经验揭示了标准评估框架下一贯的高误报率。重要的是，我们证明了许多最先进的（SOTA）MI方法报告了夸大的攻击准确性，这表明实际的隐私泄露显着低于以前认为的。通过揭示这一关键问题并提出一个强有力的解决方案，我们的工作能够重新评估MI研究的进展，并为可靠和强大的评估设定了新的标准。代码可在https://github.com/hosytuyen/MI-Eval-MLLM找到



## **44. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片即可：用单个图像毒害视觉文档检索增强生成 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.

摘要: 检索增强生成（RAG）有助于通过使用事实知识库（KB）来抑制大型语言模型（LLM）中的幻觉。尽管PDF文档是重要的知识来源，但基于文本的RAG管道在捕获其丰富的多模式信息方面效率低下。相比之下，视觉文档RAG（VD-RAG）使用文档页面的屏幕截图作为KB，已被证明可以实现最先进的结果。然而，通过引入图像模式，VD-RAG为对手引入了新的攻击载体，通过将恶意文档注入知识库来破坏系统。在本文中，我们展示了VD-RAG对针对检索和生成的中毒攻击的脆弱性。我们定义了两个攻击目标，并证明这两个目标都可以通过仅向知识库中注入单个对抗图像来实现。首先，我们对一个或一组查询引入有针对性的攻击，目标是传播有针对性的虚假信息。其次，我们提出了一种通用攻击，对于任何潜在的用户查询，该攻击都会影响响应，从而导致VD-RAG系统中的拒绝服务。我们调查的两个攻击目标下的白盒和黑盒的假设，采用多目标的基于梯度的优化方法，以及促使国家的最先进的生成模型。使用两个可视化文档数据集，一组不同的最先进的检索器（嵌入模型）和生成器（视觉语言模型），我们表明VD-RAG在目标和通用设置中都容易受到中毒攻击，但在通用设置中表现出对黑盒攻击的鲁棒性。



## **45. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

LightDefense：通过转移代币分发针对越狱的轻量级不确定性驱动防御 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.01533v2) [paper-pdf](https://arxiv.org/pdf/2504.01533v2)

**Authors**: Zhuoran Yang, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.

摘要: 大型语言模型（LLM）面临来自越狱提示的威胁。现有的针对越狱攻击的防御方法主要基于辅助模型。然而，这些战略往往需要广泛的数据收集或培训。我们提出LightDefense，这是一种针对白盒模型的轻量级防御机制，利用以安全为导向的方向来调整词汇表中代币的概率，使安全免责声明在按概率降序排序后出现在前几名代币中。我们进一步创新性地利用LLM对提示的不确定性来衡量其危害性，并自适应地调整防御强度，有效地平衡了安全性和有益性。LightDefense在2个目标LLM上防御5种攻击方法的有效性，而不影响对良性用户查询的帮助，突出了其作为一种新型轻量级防御机制的潜力，增强了LLM的安全性。



## **46. IPAD: Inverse Prompt for AI Detection - A Robust and Interpretable LLM-Generated Text Detector**

iPad：人工智能检测的反向提示-一个强大且可解释的LLM生成文本检测器 cs.LG

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2502.15902v3) [paper-pdf](https://arxiv.org/pdf/2502.15902v3)

**Authors**: Zheng Chen, Yushi Feng, Jisheng Dang, Yue Deng, Changyang He, Hongxi Pu, Haoxuan Li, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.

摘要: 大型语言模型（LLM）在文本生成方面已经达到了人类水平的流畅性，这使得区分人类书面文本和LLM生成的文本变得复杂。这增加了误用的风险，并突出了对可靠探测器的需求。然而，现有的检测器表现出对分布外（OOD）数据和攻击数据的鲁棒性差，这对于现实世界的场景是至关重要的。此外，他们努力提供可解释的证据来支持他们的决定，从而破坏了可靠性。鉴于这些挑战，我们提出了iPad（人工智能检测反向提示），这是一个新颖的框架，由一个提示反向器和两个区分器组成，用于识别可能生成输入文本的预测提示，用于检查输入文本与预测提示对齐的可能性。经验评估表明，iPad在分销内数据上的表现比最强基线高出9.05%（平均召回），在分销外数据上的表现比最强基线高出12.93%（AUROC），在受攻击数据上的表现比最强基线高出5.48%（AUROC）。iPad还在结构化数据集上表现出色。此外，还进行了可解释性评估，以说明iPad通过允许用户直接检查决策证据来增强了人工智能检测的可信度，从而为其最先进的检测结果提供了可解释的支持。



## **47. Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation**

探索联邦军事LLM中潜在的即时注入攻击及其缓解措施 cs.LG

Accepted to the 3rd International Workshop on Dataspaces and Digital Twins for Critical Entities and Smart Urban Communities - IEEE BigData 2025

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18416v2) [paper-pdf](https://arxiv.org/pdf/2501.18416v2)

**Authors**: Youngjoon Lee, Taehyun Park, Yunho Lee, Jinu Gong, Joonhyuk Kang

**Abstract**: Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these risks, we propose a human-AI collaborative framework with both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols.

摘要: 联合学习（FL）越来越多地被用于军事合作，以开发大型语言模型（LLM），同时保留数据主权。然而，即时注入攻击（对输入预算的恶意操纵）构成了新的威胁，可能会破坏运营安全、扰乱决策并削弱盟友之间的信任。这篇观点论文强调了联邦军事LLM中的四个漏洞：秘密数据泄露、搭便车剥削、系统中断和错误信息传播。为了应对这些风险，我们提出了一个具有技术和政策对策的人与人工智能协作框架。在技术方面，我们的框架使用红/蓝团队战争游戏和质量保证来检测和减轻共享LLM权重的对抗行为。在政策方面，它促进人工智能与人类联合政策制定和安全协议验证。



## **48. DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs**

DarkMind：定制LLC中潜在的思想链后门 cs.CR

19 pages, 15 figures, 12 tables

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2501.18617v2) [paper-pdf](https://arxiv.org/pdf/2501.18617v2)

**Authors**: Zhen Guo, Shanghao Shi, Shamim Yazdani, Ning Zhang, Reza Tourani

**Abstract**: With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.

摘要: 随着个性化人工智能的迅速崛起，配备思想链（COT）推理的定制大型语言模型（LLM）现在为数百万人工智能代理提供动力。然而，它们复杂的推理过程会引入新的且基本上未被探索的安全漏洞。我们提出了DarkMind，这是一种新颖的潜在推理级后门攻击，通过在不改变用户查询的情况下操纵内部COT步骤来针对自定义的LLM。与之前的基于提示的攻击不同，DarkMind通过潜在触发器在推理链中秘密激活，无需修改输入提示或要求访问模型参数即可实现对抗行为。为了实现隐形和可靠性，我们提出了即时和追溯双重触发类型，并将它们集成到统一的嵌入模板中，该模板管理触发相关激活，采用隐形优化算法来最大限度地减少语义漂移，并引入自动对话启动器跨域的秘密激活。使用五个LLM对跨越算术、常识和符号领域的八个推理数据集进行了全面实验，证明DarkMind始终实现了高攻击成功率。我们进一步研究了减轻这些风险的防御策略，并揭示了推理级后门代表了一个重大但未充分探索的威胁，强调了对强大、推理感知安全机制的必要性。



## **49. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

ATA：通过简单辅助任务链接实现LLM越狱的典范 cs.CR

ACL Findings 2025. Welcome to employ SATA as a baseline

**SubmitDate**: 2025-11-24    [abs](http://arxiv.org/abs/2412.15289v5) [paper-pdf](https://arxiv.org/pdf/2412.15289v5)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.

摘要: 大型语言模型（LLM）在各种任务中取得了重大进展，但它们的安全性一致仍然是一个主要问题。探索越狱提示可以暴露LLM的漏洞并指导保护它们的工作。现有的方法主要设计复杂的指令供LLM遵循，或者依赖于多次迭代，这可能会阻碍越狱的性能和效率。在这项工作中，我们提出了一种新颖的越狱范式--简单辅助任务链接（ATA），它可以有效地规避LLM保障措施并引发有害反应。具体来说，ATA首先屏蔽恶意查询中的有害关键词，以生成包含一个或多个[MASK]特殊令牌的相对良性的查询。然后，它采用简单的辅助任务，例如掩蔽语言模型任务或按位置查找元素任务来编码掩蔽关键词的语义。最后，ATA将辅助任务与屏蔽查询链接起来，共同执行越狱。大量实验表明，ATA实现了最先进的性能，并且大幅优于基线。具体来说，在AdvBench数据集上，通过屏蔽语言模型（MLM）辅助任务，ATA的总体攻击成功率（ASB）达到85%，有害评分（HS）达到4.57，通过按位置查找元素（ELP）辅助任务，ATA的总体攻击成功率（ASB）达到76%，HS达到4.43。



## **50. Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization**

Eguard：通过文本互信息优化保护LLM嵌入免受倒置攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2411.05034v2) [paper-pdf](https://arxiv.org/pdf/2411.05034v2)

**Authors**: Tiantian Liu, Hongwei Yao, Feng Lin, Tong Wu, Zhan Qin, Kui Ren

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.

摘要: 嵌入已成为大型语言模型（LLM）功能的基石，因为它们能够将文本数据转换为捕获语义和语法属性的丰富、密集的数字表示。这些嵌入式载体数据库充当LLM的长期存储器，能够高效处理广泛的自然语言处理任务。然而，在LLM中嵌入载体数据库的普及率激增，同时也伴随着对隐私泄露的严重担忧。嵌入式载体数据库特别容易受到嵌入倒置攻击，对手可以利用嵌入进行反向工程并从原始文本数据中提取敏感信息。现有的防御机制已经显示出局限性，经常难以平衡安全性与下游任务的性能。为了解决这些挑战，我们引入了Eguard，这是一种新颖的防御机制，旨在减轻嵌入倒置攻击。Eguard采用基于转换器的投影网络和文本互信息优化来保护嵌入，同时保留LLM的实用性。我们的方法显着降低了隐私风险，保护超过95%的令牌免受倒置，同时在下游任务中保持与原始嵌入一致的高性能。



